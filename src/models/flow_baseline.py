"""flow_baseline.py — Vanilla flow-matching model (no SPRINT).

A clean control model: single flat TransformerNetwork, standard CFG
(full forward pass for both positive and negative guidance), no token
dropping, no concat-fusion.  Everything else (M-RoPE, prefix tokens,
timestep schedule, euler_cfg) is identical to flow.py so results are
directly comparable.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

# Re-use the shared low-level building blocks from flow.py so there is
# exactly one copy of the RoPE kernels, AttentionBlock, GLU, etc.
from src.models.flow import (
    AttentionBlock,
    GLU,
    _Block,
    MRoPEEmbedding,
    TransformerNetwork,
    build_mrope_position_ids,
    image_flatten,
    image_unflatten,
    time_shift,
    sample_from_distribution,
    create_distribution,
)


class FlowBaseline(nn.Module):
    """Vanilla flow-matching transformer — no SPRINT, no token dropping.

    Architecture:
      - Single flat ``TransformerNetwork`` of ``num_layers`` blocks.
      - Standard CFG: both positive and negative guidance passes run the
        full model.
      - No concat-fusion layer, no mask token, no encoder/middle/decoder
        split.

    Drop-in replacement for ``Flow`` in the trainer: same ``forward``
    signature (``use_mask`` and ``skip_middle`` are accepted but silently
    ignored so the same training loop works for both models).
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        dim,
        num_layers: int = 12,
        num_heads: int = 8,
        exp_fac: int = 4,
        rope_seq_length: int = 784,
        # --- token config ---
        n_time_tokens: int = 1,
        n_register_tokens: int = 0,
        use_text_embed: bool = False,
        text_dim: int = 768,
        pos_jitter_range: int = 0,
        compile_blocks: bool = False,
        patch_size: int = 16,
        # Accepted but unused — keeps the same config schema as Flow.
        sprint_split=None,
        sprint_group_size=None,
        sprint_keep_per_group=None,
    ):
        super().__init__()
        self.dim = dim
        self._patch_size = patch_size
        self.n_time_tokens = n_time_tokens
        self.n_register_tokens = n_register_tokens
        self.use_text_embed = use_text_embed
        self.pos_jitter_range = pos_jitter_range

        # --- Token-level projections / embeddings ---
        self.input_layer = nn.Linear(input_dim, dim)
        self.timestep_vector = nn.Linear(1, dim * n_time_tokens)

        if use_text_embed:
            self.token_embed = nn.Embedding(text_dim, dim)
            self.text_norm = nn.RMSNorm(dim, elementwise_affine=True)

        if n_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, n_register_tokens, dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        # --- Single flat transformer ---
        self.transformer = TransformerNetwork(
            input_dim=dim,
            output_dim=output_dim,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            exp_fac=exp_fac,
            rope_seq_length=rope_seq_length,
            input_proj=False,
            final_head=True,
            final_norm=True,
            compile_blocks=compile_blocks,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.timestep_vector.weight)
        nn.init.zeros_(self.timestep_vector.bias)

    def compile_blocks(self) -> None:
        """Compile transformer blocks with torch.compile.

        Call *after* any FX-based model wrapping (e.g. ramtorch
        MultiGPUWrapper.setup()) to avoid the
        "FX tracing a dynamo-optimized function" error.
        """
        self.transformer.compile_blocks()

    @property
    def device(self):
        return torch.cuda.current_device()

    def _build_prefix(self, x, t, text_tokens=None):
        """Assemble prefix tokens: [text, time, register].
        Returns (prefix, n_text, n_time, n_reg).
        """
        B = x.shape[0]
        parts = []

        n_text = 0
        if self.use_text_embed and text_tokens is not None:
            txt = self.text_norm(self.token_embed(text_tokens))
            parts.append(txt)
            n_text = txt.shape[1]

        time_vec = self.timestep_vector(t.view(-1, 1)).view(B, self.n_time_tokens, self.dim)
        parts.append(time_vec)

        n_reg = 0
        if self.n_register_tokens > 0:
            parts.append(self.register_tokens.expand(B, -1, -1))
            n_reg = self.n_register_tokens

        prefix = torch.cat(parts, dim=1)
        return prefix, n_text, self.n_time_tokens, n_reg

    def forward(
        self,
        x,
        t,
        attention_mask=None,
        text_tokens=None,
        use_mask: bool = False,    # ignored — no SPRINT
        skip_middle: bool = False, # ignored — no SPRINT
    ):
        """x: [B, C, H, W] image. Returns predicted x0 in the same shape."""
        B, C, H, W = x.shape
        patches, _ = image_flatten(x, shuffle_size=self._patch_size)
        patches_proj = self.input_layer(patches)

        prefix, n_text, n_time, n_reg = self._build_prefix(patches, t, text_tokens)
        n_prefix = n_text + n_time + n_reg

        tokens = torch.cat((prefix, patches_proj), dim=1)

        num_h = H // self._patch_size
        num_w = W // self._patch_size

        jitter = 0
        if self.training and self.pos_jitter_range > 0:
            jitter = int(torch.randint(0, self.pos_jitter_range + 1, (1,)).item())

        position_ids = build_mrope_position_ids(
            B, num_h, num_w, device=x.device,
            n_text=n_text, n_time=n_time, n_class=0, n_reg=n_reg,
            pos_jitter=jitter,
        )

        output_tokens = self.transformer(tokens, attention_mask, position_ids=position_ids)

        pred_patches = output_tokens[:, n_prefix:, ...]
        return image_unflatten(pred_patches, (B, C, H, W), shuffle_size=self._patch_size)

    def euler_cfg(
        self,
        x,
        cfg_scale: float = 4.0,
        num_steps: int = 100,
        skip_last_n: int = 0,
        return_intermediates: bool = False,
        text_tokens=None,
        uncond_text_tokens=None,
        schedule_mu: float | None = None,
        grid_points: int = 1024,
        autoguidance_mode: str = "classic",  # only "classic" supported
    ):
        """Euler CFG sampler stepping from t=1 (noise) to t=0 (clean).

        Only ``autoguidance_mode="classic"`` is supported (full forward pass
        for both positive and negative guidance).  Passing ``"pdg"`` raises
        a clear error rather than silently falling back.
        """
        if autoguidance_mode != "classic":
            raise ValueError(
                f"FlowBaseline only supports autoguidance_mode='classic', "
                f"got {autoguidance_mode!r}. Use Flow (SPRINT) for PDG."
            )

        if return_intermediates:
            trajectories = [x.cpu()]
        else:
            trajectories = None

        effective_steps = num_steps - skip_last_n

        # Build timestep schedule (identical logic to Flow.euler_cfg).
        if schedule_mu is None:
            t_seq = torch.linspace(1.0, 0.0, num_steps + 1, device=x.device, dtype=x.dtype)
        else:
            grid_t, grid_p = create_distribution(grid_points, device=x.device, mu=schedule_mu)
            grid_t = grid_t.to(x.dtype)
            grid_p = grid_p.to(x.dtype)

            cdf = torch.cumsum(grid_p, dim=0)
            cdf = cdf / cdf[-1].clamp(min=1e-8)

            q = torch.linspace(1.0, 0.0, num_steps + 1, device=x.device, dtype=x.dtype)
            idx = torch.searchsorted(cdf, q.clamp(0.0, 1.0)).clamp(1, grid_points - 1)
            cdf_lo, cdf_hi = cdf[idx - 1], cdf[idx]
            t_lo,   t_hi   = grid_t[idx - 1], grid_t[idx]
            frac = (q - cdf_lo) / (cdf_hi - cdf_lo).clamp(min=1e-8)
            t_seq = t_lo + frac * (t_hi - t_lo)
            t_seq[0] = 1.0
            t_seq[-1] = 0.0

        for i in tqdm(range(effective_steps), desc="Euler CFG Sampling"):
            with torch.no_grad():
                t_val  = t_seq[i]
                t_next = t_seq[i + 1]
                dt = t_val - t_next
                t = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype) * t_val

                x0_pos = self.forward(x, t, text_tokens=text_tokens)
                x0_neg = self.forward(x, t, text_tokens=uncond_text_tokens)

                v_pos = (x0_pos - x) / t.view(-1, 1, 1, 1)
                v_neg = (x0_neg - x) / t.view(-1, 1, 1, 1)

                velocity = v_neg + cfg_scale * (v_pos - v_neg)
                x = x + velocity * dt

            if return_intermediates:
                trajectories.append(x.cpu())

        return x, trajectories
