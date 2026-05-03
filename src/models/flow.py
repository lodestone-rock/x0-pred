"""flow.py — Flow-matching model definition.

Contains all model classes and helpers for the pixel-space flow-matching
transformer, including:
  - M-RoPE positional encoding
  - AttentionBlock, GLU, _Block, TransformerNetwork
  - Flow (top-level model with CFG sampling)
  - Image patch helpers (flatten / unflatten)
  - Timestep distribution helpers

"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm


# ---------------------------------------------------------------------------
# RoPE / M-RoPE kernels
# ---------------------------------------------------------------------------


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset=0):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_mrope(q, k, position_ids, inv_freq):
    """Apply M-RoPE with 3 independent axes (time, height, width).

    Args:
        q, k:          [B, heads, seq, head_dim]
        position_ids:  [3, B, seq]  — (time, height, width) per token
        inv_freq:      [dim_per_axis // 2]  — shared across all 3 axes

    head_dim is split into 3 equal slices of dim_per_axis each.
    Any remaining dims (head_dim % 3 != 0) are left unrotated.
    """
    head_dim = q.shape[-1]
    dim_per_axis = (head_dim // 3 // 2) * 2  # round down to even

    def axis_cos_sin(pos):  # pos: [B, seq]
        freqs = pos.unsqueeze(-1).float() * inv_freq.to(pos.device).unsqueeze(0).unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)  # [B, seq, dim_per_axis]
        return emb.cos(), emb.sin()

    cos_t, sin_t = axis_cos_sin(position_ids[0])  # time
    cos_h, sin_h = axis_cos_sin(position_ids[1])  # height
    cos_w, sin_w = axis_cos_sin(position_ids[2])  # width

    cos = torch.cat([cos_t, cos_h, cos_w], dim=-1).unsqueeze(1)  # [B, 1, seq, 3*dim_per_axis]
    sin = torch.cat([sin_t, sin_h, sin_w], dim=-1).unsqueeze(1)

    rotated_dims = 3 * dim_per_axis
    q_rot, q_pass = q[..., :rotated_dims], q[..., rotated_dims:]
    k_rot, k_pass = k[..., :rotated_dims], k[..., rotated_dims:]

    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_out = torch.cat([q_rot, q_pass], dim=-1)
    k_out = torch.cat([k_rot, k_pass], dim=-1)
    return q_out.to(q.dtype), k_out.to(k.dtype)


def build_mrope_position_ids(
    B: int,
    num_h_patches: int,
    num_w_patches: int,
    device,
    n_text: int = 0,
    n_time: int = 1,
    n_class: int = 1,
    n_reg: int = 0,
    pos_jitter: int = 0,
) -> torch.Tensor:
    """Build M-RoPE position ids for the full token sequence.

    Token layout: [text (n_text), time (n_time), class (n_class), register (n_reg), patches (H*W)]

    All prefix tokens sit on the diagonal [i, i, i] in sequence order.
    Image patches start at offset = n_text + n_time + n_class + n_reg + pos_jitter,
    with the time axis frozen at that offset and spatial axes expanding from it.

    pos_jitter: random integer added to the spatial start of patches (training only).

    Returns:
        position_ids: [3, B, seq_len]
    """
    num_patches = num_h_patches * num_w_patches
    n_prefix = n_text + n_time + n_class + n_reg
    seq_len = n_prefix + num_patches

    t_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
    h_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
    w_ids = torch.zeros(seq_len, dtype=torch.long, device=device)

    # All prefix tokens: diagonal [i, i, i]
    diag = torch.arange(n_prefix, dtype=torch.long, device=device)
    t_ids[:n_prefix] = diag
    h_ids[:n_prefix] = diag
    w_ids[:n_prefix] = diag

    # Image patches: time frozen at patch_start, spatial expands from patch_start
    patch_start = n_prefix + pos_jitter
    rows = torch.arange(num_h_patches, device=device).repeat_interleave(num_w_patches)
    cols = torch.arange(num_w_patches, device=device).repeat(num_h_patches)
    t_ids[n_prefix:] = patch_start
    h_ids[n_prefix:] = patch_start + rows
    w_ids[n_prefix:] = patch_start + cols

    position_ids = torch.stack([t_ids, h_ids, w_ids], dim=0)          # [3, seq_len]
    position_ids = position_ids.unsqueeze(1).expand(-1, B, -1)         # [3, B, seq_len]
    return position_ids


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class MRoPEEmbedding(nn.Module):
    """Stores inv_freq for a single axis; shared across all 3 M-RoPE axes."""
    def __init__(self, head_dim, max_seq_len=2048):
        super().__init__()
        dim_per_axis = (head_dim // 3 // 2) * 2
        inv_freq = 1.0 / (max_seq_len ** (torch.arange(0, dim_per_axis, 2).float() / dim_per_axis))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, q, k):
        """Apply M-RoPE. position_ids: [3, B, seq]. q,k: [B, heads, seq, head_dim]."""
        return apply_mrope(q, k, position_ids, self.inv_freq)





class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, max_seq_len=2048, use_rope=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=True)
        self.layer_norm = nn.RMSNorm(dim, elementwise_affine=True)
        self.rope = MRoPEEmbedding(self.head_dim, max_seq_len)
        self.q_norm = nn.RMSNorm(dim, elementwise_affine=True)
        self.k_norm = nn.RMSNorm(dim, elementwise_affine=True)
        self.add_module("layer_norm", self.layer_norm)

        nn.init.zeros_(self.wo.weight)
        self.use_compiled = False
        self.use_rope = use_rope

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, attention_mask=None, position_ids=None):
        residual = x
        x = self.layer_norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope and position_ids is not None:
            q, k = self.rope(position_ids, q, k)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        out = self.wo(out)
        return out + residual


class GLU(nn.Module):
    def __init__(self, dim, exp_fac=4):
        super().__init__()
        self.wi_0 = nn.Linear(dim, dim * exp_fac, bias=False)
        self.wi_1 = nn.Linear(dim, dim * exp_fac, bias=False)
        self.wo = nn.Linear(dim * exp_fac, dim, bias=True)
        self.layer_norm = nn.RMSNorm(dim, elementwise_affine=False)
        nn.init.zeros_(self.wo.weight)
        self.use_compiled = False

    @property
    def device(self):
        return next(self.parameters()).device

    def _fwd_glu(self, x, residual):
        return self.wo(F.silu(self.wi_0(x)) * self.wi_1(x)) + residual

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        if self.use_compiled:
            return torch.compile(self._fwd_glu)(x, residual)
        else:
            return self._fwd_glu(x, residual)


class _Block(nn.Module):
    """One (attn + glu) transformer block — a clean compile boundary."""
    def __init__(self, attn: AttentionBlock, glu: GLU):
        super().__init__()
        self.attn = attn
        self.glu = glu

    def forward(self, x, attention_mask=None, position_ids=None):
        x = self.attn(x, attention_mask, position_ids=position_ids)
        x = self.glu(x)
        return x


# ---------------------------------------------------------------------------
# SPRINT — Sparse–Dense Residual Fusion (arXiv:2510.21986)
# ---------------------------------------------------------------------------


def structured_group_subsample(
    B: int,
    num_h: int,
    num_w: int,
    group_size: int,
    k: int,
    device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Structured group-wise token subsample (paper §3.5).

    Partitions the (num_h × num_w) patch grid into non-overlapping
    (group_size × group_size) groups and randomly keeps ``k`` tokens per
    group, giving drop ratio ``r = 1 - k / group_size**2``.

    Args:
        B, num_h, num_w: patch grid dimensions.
        group_size: side of each non-overlapping group (paper default: 2).
        k: tokens kept per group (paper default: 1 → 75% drop).
        device: target device for the output mask.
        generator: optional torch.Generator for deterministic sampling.

    Returns:
        keep_mask: bool tensor of shape ``[B, num_h * num_w]`` with exactly
        ``k * num_groups`` True entries per sample (fixed across the batch).

    Requires ``num_h % group_size == 0`` and ``num_w % group_size == 0``.
    """
    g = group_size
    if num_h % g != 0 or num_w % g != 0:
        raise ValueError(
            f"num_h ({num_h}) and num_w ({num_w}) must be divisible by "
            f"group_size ({g}) for structured SPRINT subsampling."
        )
    gh, gw = num_h // g, num_w // g      # grid of groups
    n_per_group = g * g                   # tokens inside a single group
    n_groups = gh * gw

    # Random per-group ordering [B, n_groups, n_per_group]; pick the first k.
    rand = torch.rand(B, n_groups, n_per_group, device=device, generator=generator)
    # [B, n_groups, k] — positions inside each group that survive
    kept_local = rand.argsort(dim=-1)[..., :k]

    # Boolean mask over the n_per_group slots per group
    per_group_mask = torch.zeros(
        B, n_groups, n_per_group, dtype=torch.bool, device=device
    )
    per_group_mask.scatter_(-1, kept_local, True)

    # Reshape groups-of-tokens back to the native (num_h × num_w) layout.
    #   [B, gh, gw, g, g] → [B, gh, g, gw, g] → [B, num_h, num_w]
    per_group_mask = per_group_mask.view(B, gh, gw, g, g)
    keep_grid = per_group_mask.permute(0, 1, 3, 2, 4).contiguous().view(B, num_h, num_w)
    return keep_grid.view(B, num_h * num_w)


class SprintFusion(nn.Module):
    """Concat(f_t, g_t_pad) along channel dim and project back to ``dim``.

    The weight touching the sparse (deep) branch is zero-initialised so that
    a freshly-initialised SPRINT model behaves like the dense shallow path
    only — useful for warm-starting and for stable PDG inference before the
    fusion layer has trained.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim, bias=True)
        with torch.no_grad():
            w = self.proj.weight                   # [dim, 2*dim]
            w.zero_()
            # Identity on the dense (first) half; zero on the sparse half.
            eye = torch.eye(dim, dtype=w.dtype, device=w.device)
            w[:, :dim].copy_(eye)
            self.proj.bias.zero_()

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        # dense, sparse: [B, N, C]
        return self.proj(torch.cat([dense, sparse], dim=-1))


def _gather_kept_tokens(
    x: torch.Tensor,
    keep_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather tokens along seq dim using a per-sample bool mask.

    Args:
        x:         [B, S, C]
        keep_mask: [B, S] bool, with the *same* count ``S_kept`` True entries per row.

    Returns:
        [B, S_kept, C]
    """
    B, S, C = x.shape
    # torch.masked_select flattens; reshape back using the known per-row count.
    kept = x[keep_mask]                              # [B * S_kept, C]
    return kept.view(B, -1, C)


def _gather_kept_positions(
    position_ids: torch.Tensor,
    keep_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather M-RoPE position ids with a per-sample keep mask.

    Args:
        position_ids: [3, B, S] long
        keep_mask:    [B, S] bool (same True count per row).

    Returns:
        [3, B, S_kept] long.
    """
    # Expand keep_mask over the 3 axes and gather.
    keep3 = keep_mask.unsqueeze(0).expand_as(position_ids)   # [3, B, S]
    kept = position_ids[keep3]                                # [3*B*S_kept]
    return kept.view(3, position_ids.shape[1], -1)


def _scatter_with_mask_token(
    kept_tokens: torch.Tensor,
    keep_mask: torch.Tensor,
    mask_token: torch.Tensor,
) -> torch.Tensor:
    """Scatter ``kept_tokens`` back into a full-length tensor, filling the
    dropped positions with ``mask_token``.

    Args:
        kept_tokens: [B, S_kept, C]
        keep_mask:   [B, S] bool
        mask_token:  [1, 1, C] learned parameter (broadcast to [B, S, C]).

    Returns:
        [B, S, C]
    """
    B, S = keep_mask.shape
    C = kept_tokens.shape[-1]
    full = mask_token.expand(B, S, C).to(kept_tokens.dtype).clone()
    full[keep_mask] = kept_tokens.reshape(-1, C)
    return full


class TransformerNetwork(nn.Module):
    """A plain stack of transformer blocks.

    SPRINT composes three of these (encoder / middle / decoder) at the
    ``Flow`` level; this class stays dumb and generic.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        dim,
        num_layers,
        num_heads=8,
        exp_fac=4,
        rope_seq_length=2048,
        use_rope=True,
        final_head=True,
        input_proj=True,
        final_norm: bool = True,
        compile_blocks: bool = False,
    ):
        super().__init__()
        if input_proj:
            self.input_layer = nn.Linear(input_dim, dim)
        else:
            self.input_layer = nn.Identity()
            input_dim = dim
        blocks = [
            _Block(
                AttentionBlock(dim, num_heads, rope_seq_length, use_rope),
                GLU(dim, exp_fac),
            )
            for _ in range(num_layers)
        ]
        self._compile_blocks = compile_blocks
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = nn.RMSNorm(dim, elementwise_affine=True) if final_norm else nn.Identity()
        if final_head:
            self.output_layer = nn.Linear(dim, output_dim)
        else:
            self.output_layer = nn.Identity()

    def compile_blocks(self) -> None:
        """Compile each transformer block with torch.compile.

        Call this *after* any FX-based model wrapping (e.g. ramtorch
        MultiGPUWrapper.setup()) to avoid the
        "FX tracing a dynamo-optimized function" error.
        """
        self.blocks = nn.ModuleList([torch.compile(b) for b in self.blocks])

    def forward(self, x, attention_mask=None, position_ids=None):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x, attention_mask, position_ids=position_ids)
        x = self.out_norm(x)
        x = self.output_layer(x)
        return x


class Flow(nn.Module):
    """Flow-matching transformer with SPRINT (arXiv:2510.21986) architecture.

    The backbone is split into three ``TransformerNetwork`` stacks:
      - ``encoder`` (f_θ): processes all tokens → dense shallow features.
      - ``middle``  (g_θ): runs on a sparse subset of image-patch tokens
        during masked training / inference; bypassed entirely during PDG.
      - ``decoder`` (h_θ): runs on the fused full-length sequence.

    The fusion layer concatenates ``(dense, sparse_padded)`` along the
    channel dimension and projects back to ``dim``. Dropped positions and
    the PDG bypass are filled with a learned ``[MASK]`` token.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        dim,
        num_heads=8,
        exp_fac=4,
        rope_seq_length=784,
        # --- token config ---
        n_time_tokens: int = 1,
        n_register_tokens: int = 0,
        use_text_embed: bool = False,
        text_dim: int = 768,
        pos_jitter_range: int = 0,
        compile_blocks: bool = False,
        patch_size: int = 16,
        # --- SPRINT architecture ---
        sprint_split: tuple[int, int, int] | list[int] = (2, 8, 2),
        sprint_group_size: int = 2,
        sprint_keep_per_group: int = 1,
    ):
        super().__init__()
        n_f, n_g, n_h = sprint_split
        if n_f < 1 or n_g < 1 or n_h < 1:
            raise ValueError(
                f"sprint_split must have all three sections >= 1, got {tuple(sprint_split)}."
            )

        self.dim = dim
        self._patch_size = patch_size
        self.n_time_tokens = n_time_tokens
        self.n_register_tokens = n_register_tokens
        self.use_text_embed = use_text_embed
        self.pos_jitter_range = pos_jitter_range

        # --- SPRINT config ---
        self.sprint_split = tuple(sprint_split)
        self.sprint_group_size = sprint_group_size
        self.sprint_keep_per_group = sprint_keep_per_group

        # --- Token-level projections / embeddings ---
        self.input_layer = nn.Linear(input_dim, dim)
        self.timestep_vector = nn.Linear(1, dim * n_time_tokens)

        if use_text_embed:
            self.token_embed = nn.Embedding(text_dim, dim)
            self.text_norm = nn.RMSNorm(dim, elementwise_affine=True)

        if n_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, n_register_tokens, dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        # --- Three transformer stacks ---
        # Encoder and middle run as "feature extractors" (no norm, no head);
        # only the decoder produces the final patch prediction.
        tn_kw = dict(
            input_dim=dim, output_dim=dim, dim=dim, num_heads=num_heads,
            exp_fac=exp_fac, rope_seq_length=rope_seq_length,
            input_proj=False, compile_blocks=compile_blocks,
        )
        self.encoder = TransformerNetwork(
            num_layers=n_f, final_head=False, final_norm=False, **tn_kw,
        )
        self.middle = TransformerNetwork(
            num_layers=n_g, final_head=False, final_norm=False, **tn_kw,
        )
        self.decoder = TransformerNetwork(
            num_layers=n_h, final_head=True, final_norm=True,
            **{**tn_kw, "output_dim": output_dim},
        )

        # --- SPRINT fusion + mask token ---
        self.sprint_fusion = SprintFusion(dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.timestep_vector.weight)
        nn.init.zeros_(self.timestep_vector.bias)

    def compile_blocks(self) -> None:
        """Lazily compile all transformer blocks (encoder + middle + decoder).

        Call this *after* any FX-based model wrapping (e.g. ramtorch
        MultiGPUWrapper.setup()) to avoid the
        \"FX tracing a dynamo-optimized function\" error.
        """
        self.encoder.compile_blocks()
        self.middle.compile_blocks()
        self.decoder.compile_blocks()

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
            txt = self.text_norm(self.token_embed(text_tokens))  # [B, seq, dim]
            parts.append(txt)
            n_text = txt.shape[1]

        time_vec = self.timestep_vector(t.view(-1, 1)).view(B, self.n_time_tokens, self.dim)
        parts.append(time_vec)

        n_reg = 0
        if self.n_register_tokens > 0:
            parts.append(self.register_tokens.expand(B, -1, -1))
            n_reg = self.n_register_tokens

        prefix = torch.cat(parts, dim=1)  # [B, n_prefix, dim]
        return prefix, n_text, self.n_time_tokens, n_reg

    def forward(
        self,
        x,
        t,
        attention_mask=None,
        text_tokens=None,
        use_mask: bool = False,
        skip_middle: bool = False,
    ):
        """x: [B, C, H, W] image. Returns predicted x0 in the same [B, C, H, W] shape.

        Flow:
            1. encoder(tokens)              -> f_out  (dense, full-length)
            2. middle(..)                   -> g_out  (sparse / full / skipped)
            3. fusion(f_out, g_out_full)    -> fused
            4. decoder(fused)               -> pred patches

        Args:
            use_mask: if True, drop 75% of image-patch tokens (structured
                2×2 group-wise subsample) before the middle stack and
                scatter back with ``[MASK]`` tokens. Prefix tokens
                (text / time / register) are always kept.
            skip_middle: if True, bypass the middle stack entirely and
                replace ``g_out`` with broadcast ``[MASK]`` tokens. Used for
                Path-Drop Guidance (PDG, paper Eq. 4) at inference.
        """
        B, C, H, W = x.shape
        patches, _ = image_flatten(x, shuffle_size=self._patch_size)
        patches_proj = self.input_layer(patches)

        prefix, n_text, n_time, n_reg = self._build_prefix(patches, t, text_tokens)
        n_prefix = n_text + n_time + n_reg

        tokens = torch.cat((prefix, patches_proj), dim=1)          # [B, S, C]

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

        # 1. Encoder — always full-length, dense.
        f_out = self.encoder(tokens, attention_mask, position_ids=position_ids)

        # 2. Middle — sparse / full / skipped.
        if skip_middle:
            # PDG: bypass middle entirely; fill with the learned mask token.
            B_, S, C_ = f_out.shape
            g_full = self.mask_token.expand(B_, S, C_).to(f_out.dtype)
        elif use_mask:
            # Structured group-wise drop on image-patch tokens; prefix kept.
            patch_keep = structured_group_subsample(
                B, num_h, num_w,
                group_size=self.sprint_group_size,
                k=self.sprint_keep_per_group,
                device=x.device,
            )                                                      # [B, N]
            prefix_keep = torch.ones(B, n_prefix, dtype=torch.bool, device=x.device)
            keep_mask = torch.cat([prefix_keep, patch_keep], dim=1)  # [B, S]

            kept_tokens = _gather_kept_tokens(f_out, keep_mask)
            kept_pos    = _gather_kept_positions(position_ids, keep_mask)
            g_kept = self.middle(kept_tokens, attention_mask, position_ids=kept_pos)
            g_full = _scatter_with_mask_token(g_kept, keep_mask, self.mask_token)
        else:
            # Full-token middle pass (e.g. during mask-off fine-tuning).
            g_full = self.middle(f_out, attention_mask, position_ids=position_ids)

        # 3. Fuse dense encoder output with (possibly sparse) middle output.
        fused = self.sprint_fusion(f_out, g_full)

        # 4. Decoder — full-length, produces final patch prediction.
        output_tokens = self.decoder(fused, attention_mask, position_ids=position_ids)

        pred_patches = output_tokens[:, n_prefix:, ...]
        return image_unflatten(pred_patches, (B, C, H, W), shuffle_size=self._patch_size)

    def euler_cfg(
        self,
        x,
        cfg_scale=4.0,
        num_steps=100,
        skip_last_n=0,
        return_intermediates=False,
        text_tokens=None,
        uncond_text_tokens=None,
        schedule_mu: float | None = None,
        grid_points: int = 1024,
        autoguidance_mode: str = "classic",
    ):
        """Euler CFG sampler stepping from t=1 (noise) to t=0 (clean).

        Args:
            schedule_mu: if None, use uniform dt = 1/num_steps (legacy).
                If a float, build a non-uniform t schedule matched to the
                training timestep distribution — the inverted-parabola density
                from `create_distribution` shifted by `time_shift(mu)`.  Step
                boundaries are placed at equal-probability quantiles of that
                distribution's CDF, so more steps (smaller dt) land in the
                high-density region and fewer in the tails.
                Pass the same mu used during training (e.g. 1.0) to match it
                exactly; pass a larger mu to dwell longer in the high-noise
                region.
            grid_points: resolution of the CDF grid used to invert quantiles.
            autoguidance_mode: controls the negative guidance pass.
                ``"classic"`` — full forward pass with empty text
                    (standard CFG, default).
                ``"pdg"``     — Path-Drop Guidance (SPRINT paper §3.4,
                    arXiv:2510.21986 Eq. 4). The negative pass bypasses the
                    middle blocks entirely, replacing ``g_theta(f_theta(...))``
                    with the learned ``[MASK]`` token, while also using the
                    empty-text (∅) condition. Nearly halves inference FLOPs
                    per guided step since only one full forward is needed.
        """
        if return_intermediates:
            trajectories = [x.cpu()]
        else:
            trajectories = None

        effective_steps = num_steps - skip_last_n

        # ------------------------------------------------------------------
        # Build the timestep schedule: t_seq has length (num_steps + 1),
        # running from 1.0 down to 0.0.  dt_i = t_seq[i] - t_seq[i+1].
        # ------------------------------------------------------------------
        if schedule_mu is None:
            t_seq = torch.linspace(1.0, 0.0, num_steps + 1, device=x.device, dtype=x.dtype)
        else:
            # Dense grid of shifted t values + matching probabilities.
            grid_t, grid_p = create_distribution(
                grid_points, device=x.device, mu=schedule_mu
            )
            grid_t = grid_t.to(x.dtype)
            grid_p = grid_p.to(x.dtype)

            # CDF over the shifted grid (monotonically non-decreasing, in [0,1]).
            cdf = torch.cumsum(grid_p, dim=0)
            cdf = cdf / cdf[-1].clamp(min=1e-8)

            # Target quantiles for step boundaries, descending from t=1 to t=0.
            # Using (num_steps + 1) boundaries → num_steps intervals.
            # Quantile q maps to the t with CDF(t) = q; we walk q from 1 → 0.
            q = torch.linspace(1.0, 0.0, num_steps + 1, device=x.device, dtype=x.dtype)

            # Invert the CDF via linear interpolation.  torch.searchsorted on
            # the monotonic cdf gives the right-hand index for each q.
            idx = torch.searchsorted(cdf, q.clamp(0.0, 1.0)).clamp(1, grid_points - 1)
            cdf_lo = cdf[idx - 1]
            cdf_hi = cdf[idx]
            t_lo = grid_t[idx - 1]
            t_hi = grid_t[idx]
            frac = (q - cdf_lo) / (cdf_hi - cdf_lo).clamp(min=1e-8)
            t_seq = t_lo + frac * (t_hi - t_lo)

            # Pin endpoints exactly to avoid numerical drift off [0, 1].
            t_seq[0] = 1.0
            t_seq[-1] = 0.0

        # Resolve negative-pass settings from autoguidance_mode.
        if autoguidance_mode == "classic":
            neg_skip_middle = False
            neg_text_tokens = uncond_text_tokens
        elif autoguidance_mode == "pdg":
            neg_skip_middle = True
            neg_text_tokens = uncond_text_tokens   # empty text, no middle blocks
        else:
            raise ValueError(
                f"Unknown autoguidance_mode {autoguidance_mode!r}. "
                "Expected 'classic' or 'pdg'."
            )

        for i in tqdm(range(effective_steps), desc="Euler CFG Sampling"):
            with torch.no_grad():
                t_val = t_seq[i]
                t_next = t_seq[i + 1]
                dt = (t_val - t_next)  # positive scalar
                t = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype) * t_val

                x0_pos = self.forward(x, t, text_tokens=text_tokens)
                x0_neg = self.forward(
                    x, t,
                    text_tokens=neg_text_tokens,
                    skip_middle=neg_skip_middle,
                )

                v_pos = (x0_pos - x) / t.view(-1, 1, 1, 1)
                v_neg = (x0_neg - x) / t.view(-1, 1, 1, 1)

                velocity = v_neg + cfg_scale * (v_pos - v_neg)
                x = x + velocity * dt

            if return_intermediates:
                trajectories.append(x.cpu())

        return x, trajectories


# ---------------------------------------------------------------------------
# Image patch helpers
# ---------------------------------------------------------------------------


def image_flatten(latents, shuffle_size=16):
    return (
        rearrange(
            latents,
            "n c (h dh) (w dw) -> n (h w) (c dh dw)",
            dh=shuffle_size,
            dw=shuffle_size,
        ),
        latents.shape,
    )


def image_unflatten(latents, shape, shuffle_size=16):
    n, c, h, w = shape
    return rearrange(
        latents,
        "n (h w) (c dh dw) -> n c (h dh) (w dw)",
        dh=shuffle_size,
        dw=shuffle_size,
        c=c,
        h=h // shuffle_size,
        w=w // shuffle_size,
    )


# ---------------------------------------------------------------------------
# Timestep distribution helpers
# ---------------------------------------------------------------------------


def time_shift(t: torch.Tensor, mu: float) -> torch.Tensor:
    """Shift timestep distribution via logit-space translation.

    Equivalent to the logit-normal mu shift from JiT / SD3:
        t_shifted = sigmoid(logit(t) + mu)
                  = t / (t + (1 - t) * exp(-mu))

    mu < 0  -> more high-noise samples (smaller t)  [JiT default: -0.8]
    mu = 0  -> identity
    mu > 0  -> more low-noise samples (larger t)
    """
    return t / (t + (1.0 - t) * math.exp(-mu))


def sample_from_distribution(x, probabilities, n):
    indices = torch.multinomial(probabilities, n, replacement=True)
    return x[indices]


def create_distribution(num_points, device=None, mu: float = 0.0):
    x = torch.linspace(0, 1, num_points, device=device)
    probabilities = -7.7 * ((x - 0.5) ** 2) + 2
    probabilities /= probabilities.sum()
    if mu != 0.0:
        x = time_shift(x, mu)
    return x, probabilities
