"""flow.py — Flow-matching model definition.

Contains all model classes and helpers for the pixel-space flow-matching
transformer, including:
  - M-RoPE positional encoding
  - PinchedLinear (low-rank bottleneck projection)
  - SoftClamp activation
  - AttentionBlock, GLU, _Block, TransformerNetwork
  - Flow (top-level model with CFG sampling)
  - Image patch helpers (flatten / unflatten)
  - Timestep distribution helpers

TREAD token routing (arxiv 2501.04765):
  Training-only speedup: a random subset of image tokens bypasses a span of
  middle layers (identity transport), then rejoins the full sequence at the
  end layer.  Inference always uses the standard full forward pass.

  Configure via Flow(..., tread_route=[start, end, rate]) e.g. [2, 8, 0.5].
  Disable by passing tread_route=None (default) or switching to eval mode.
  Recommended absolute config for N layers: start=2, end=N-4, rate=0.5.
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
# Activation helpers
# ---------------------------------------------------------------------------


def soft_clamp(x, scale, alpha, shift):
    return scale * F.tanh(x * alpha) + shift


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


class PinchedLinear(nn.Module):
    """Low-rank bottleneck projection: in_dim -> pinch_dim -> out_dim.

    The down-projection (in_dim -> pinch_dim) has no bias so the two matrices
    can be merged into a single full-rank weight (W = up @ down) after training.
    The up-projection (pinch_dim -> out_dim) keeps its bias.

    If pinch_dim == -1 this degenerates to a plain nn.Linear.
    """
    def __init__(self, in_dim: int, out_dim: int, pinch_dim: int = -1, bias: bool = True):
        super().__init__()
        self.pinch_dim = pinch_dim
        if pinch_dim == -1:
            self.proj = nn.Linear(in_dim, out_dim, bias=bias)
        else:
            self.down = nn.Linear(in_dim, pinch_dim, bias=False)
            self.up   = nn.Linear(pinch_dim, out_dim, bias=bias)

    def forward(self, x):
        if self.pinch_dim == -1:
            return self.proj(x)
        return self.up(self.down(x))

    def merged_weight(self) -> torch.Tensor:
        """Return up.weight @ down.weight — the equivalent full-rank matrix."""
        if self.pinch_dim == -1:
            return self.proj.weight
        return self.up.weight @ self.down.weight


class SoftClamp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.use_compiled = False

    def forward(self, x):
        if self.use_compiled:
            return torch.compile(soft_clamp)(x, self.scale, self.alpha, self.shift)
        else:
            return soft_clamp(x, self.scale, self.alpha, self.shift)


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, max_seq_len=2048, use_rope=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=True)
        self.layer_norm = SoftClamp(dim)
        self.rope = MRoPEEmbedding(self.head_dim, max_seq_len)
        self.q_norm = SoftClamp(dim)
        self.k_norm = SoftClamp(dim)
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
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
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


class TransformerNetwork(nn.Module):
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
        pinch_dim: int = -1,
        compile_blocks: bool = False,
        tread_route: tuple | None = None,
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
        self.out_norm = SoftClamp(dim)
        if final_head:
            self.output_layer = PinchedLinear(dim, output_dim, pinch_dim=pinch_dim)
        else:
            self.output_layer = nn.Identity()

        # TREAD routing config — stored as plain tuple or None
        self.tread_route: tuple[int, int, float] | None = None
        if tread_route is not None:
            r_start, r_end, r_rate = int(tread_route[0]), int(tread_route[1]), float(tread_route[2])
            assert 0 < r_start < r_end <= num_layers, (
                f"tread_route start/end ({r_start}, {r_end}) must satisfy "
                f"0 < start < end <= num_layers ({num_layers})"
            )
            assert 0.0 < r_rate < 1.0, (
                f"tread_route selection_rate must be in (0, 1), got {r_rate}"
            )
            self.tread_route = (r_start, r_end, r_rate)

    def compile_blocks(self) -> None:
        """Compile each transformer block with torch.compile.

        Call this *after* any FX-based model wrapping (e.g. ramtorch
        MultiGPUWrapper.setup()) to avoid the
        "FX tracing a dynamo-optimized function" error.
        """
        self.blocks = nn.ModuleList([torch.compile(b) for b in self.blocks])

    def forward(
        self,
        x,
        attention_mask=None,
        act_ckpt=False,
        position_ids=None,
        n_prefix: int = 0,
    ):
        """Forward pass with optional TREAD token routing.

        Args:
            n_prefix: number of non-image prefix tokens at the start of the
                sequence (text + time + class + register).  Used by TREAD to
                exclude prefix tokens from routing.  Safe to leave as 0 when
                tread_route is None or model is in eval mode.
        """
        x = self.input_layer(x)

        # ----------------------------------------------------------------
        # TREAD routing setup (training only)
        # ----------------------------------------------------------------
        use_tread = self.training and self.tread_route is not None
        if use_tread:
            r_start, r_end, r_rate = self.tread_route  # type: ignore[misc]
            n_img    = x.shape[1] - n_prefix
            n_routed = int(n_img * r_rate)   # deterministic count → static shape per resolution

            # Random permutation of image-token indices.
            # First n_routed will bypass layers [r_start, r_end).
            perm       = torch.randperm(n_img, device=x.device)
            routed_idx = perm[:n_routed]          # relative to image-token block
            kept_idx   = perm[n_routed:]

            # Absolute positions in the full sequence
            routed_abs = routed_idx + n_prefix
            kept_abs   = kept_idx   + n_prefix
            prefix_abs = torch.arange(n_prefix, device=x.device)

            # Reduced sequence index: prefix + kept image tokens
            # Shape is constant for a given (n_img, r_rate) → compile-safe
            reduced_abs = torch.cat([prefix_abs, kept_abs], dim=0)

            # Slice position_ids to match reduced sequence
            orig_position_ids = position_ids
            if position_ids is not None:
                reduced_pos_ids = position_ids[:, :, reduced_abs]  # [3, B, n_reduced]
            else:
                reduced_pos_ids = None

            routed_tokens: torch.Tensor | None = None  # filled at r_start
        # ----------------------------------------------------------------

        for layer_idx, block in enumerate(self.blocks):

            if use_tread and layer_idx == r_start:
                # Stash routed tokens, then shrink sequence to prefix + kept
                routed_tokens = x[:, routed_abs, :]   # [B, n_routed, dim]
                x             = x[:, reduced_abs, :]
                position_ids  = reduced_pos_ids

            if use_tread and layer_idx == r_end:
                # Reinsert routed tokens at their original positions before
                # this layer so the full sequence is restored.
                full = torch.empty(
                    x.shape[0], n_prefix + n_img, x.shape[2],
                    dtype=x.dtype, device=x.device,
                )
                full[:, :n_prefix, :]  = x[:, :n_prefix, :]   # prefix unchanged
                full[:, kept_abs, :]   = x[:, n_prefix:, :]   # kept image tokens
                full[:, routed_abs, :] = routed_tokens         # type: ignore[index]
                x            = full
                position_ids = orig_position_ids               # restore full pos ids

            x = block(x, attention_mask, position_ids=position_ids)

        x = self.out_norm(x)
        x = self.output_layer(x)
        return x


class Flow(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        dim,
        num_layers,
        num_heads=8,
        exp_fac=4,
        rope_seq_length=784,
        class_count=10,
        cond_seq_len=40,
        pinch_dim: int = -1,
        # --- token config ---
        n_time_tokens: int = 1,
        n_register_tokens: int = 0,
        use_class_embed: bool = True,
        use_text_embed: bool = False,
        text_dim: int = 768,
        pos_jitter_range: int = 0,
        compile_blocks: bool = False,
        patch_size: int = 16,
        # --- TREAD token routing (arxiv 2501.04765) ---
        tread_route: list | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.class_count = class_count
        self._patch_size = patch_size
        self.n_time_tokens = n_time_tokens
        self.n_register_tokens = n_register_tokens
        self.use_class_embed = use_class_embed
        self.use_text_embed = use_text_embed
        self.pos_jitter_range = pos_jitter_range
        self.cond_seq_len = cond_seq_len

        self.input_layer = PinchedLinear(input_dim, dim, pinch_dim=pinch_dim)  # type: ignore[assignment]
        self.timestep_vector = nn.Linear(1, dim * n_time_tokens)

        if use_class_embed:
            self.class_embed = nn.Linear(cond_seq_len, dim)

        if use_text_embed:
            self.token_embed = nn.Embedding(text_dim, dim)
            self.text_norm = SoftClamp(dim=dim)

        if n_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, n_register_tokens, dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        self.transformer = TransformerNetwork(
            input_dim=dim,
            output_dim=output_dim,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            exp_fac=exp_fac,
            rope_seq_length=rope_seq_length,
            final_head=True,
            input_proj=False,
            pinch_dim=-1,
            compile_blocks=compile_blocks,
            tread_route=tread_route,
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.timestep_vector.weight)
        nn.init.zeros_(self.timestep_vector.bias)
        if self.use_class_embed:
            nn.init.zeros_(self.class_embed.weight)
            nn.init.zeros_(self.class_embed.bias)

    def compile_blocks(self) -> None:
        """Lazily compile transformer blocks with torch.compile.

        Call this *after* any FX-based model wrapping (e.g. ramtorch
        MultiGPUWrapper.setup()) to avoid the
        \"FX tracing a dynamo-optimized function\" error.
        """
        self.transformer.compile_blocks()

    @property
    def device(self):
        return torch.cuda.current_device()

    def _build_prefix(self, x, t, condition=None, text_tokens=None):
        """Assemble prefix tokens: [text, time, class, register].
        Returns (prefix, n_text, n_time, n_class, n_reg).
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

        n_class = 0
        if self.use_class_embed and condition is not None:
            class_vec = self.class_embed(condition.to(self.class_embed.weight.dtype))[:, None, :]
            parts.append(class_vec)
            n_class = 1

        n_reg = 0
        if self.n_register_tokens > 0:
            parts.append(self.register_tokens.expand(B, -1, -1))
            n_reg = self.n_register_tokens

        prefix = torch.cat(parts, dim=1)  # [B, n_prefix, dim]
        return prefix, n_text, self.n_time_tokens, n_class, n_reg

    def forward(self, x, t, condition=None, attention_mask=None, text_tokens=None):
        """x: [B, C, H, W] image. Returns predicted x0 in the same [B, C, H, W] shape."""
        B, C, H, W = x.shape
        patches, _ = image_flatten(x, shuffle_size=self._patch_size)
        patches_proj = self.input_layer(patches)

        prefix, n_text, n_time, n_class, n_reg = self._build_prefix(patches, t, condition, text_tokens)
        n_prefix = n_text + n_time + n_class + n_reg

        tokens = torch.cat((prefix, patches_proj), dim=1)

        num_h = H // self._patch_size
        num_w = W // self._patch_size

        jitter = 0
        if self.training and self.pos_jitter_range > 0:
            jitter = int(torch.randint(0, self.pos_jitter_range + 1, (1,)).item())

        position_ids = build_mrope_position_ids(
            B, num_h, num_w, device=x.device,
            n_text=n_text, n_time=n_time, n_class=n_class, n_reg=n_reg,
            pos_jitter=jitter,
        )

        output_tokens = self.transformer(
            tokens,
            attention_mask,
            position_ids=position_ids,
            n_prefix=n_prefix,
        )
        pred_patches = output_tokens[:, n_prefix:, ...]
        return image_unflatten(pred_patches, (B, C, H, W), shuffle_size=self._patch_size)

    def euler_cfg(
        self,
        x,
        pos_cond=None,
        cfg_scale=4.0,
        num_steps=100,
        skip_last_n=0,
        return_intermediates=False,
        text_tokens=None,
    ):
        if return_intermediates:
            trajectories = [x.cpu()]
        else:
            trajectories = None

        neg_cond = torch.zeros_like(pos_cond) if pos_cond is not None else None
        dt = 1.0 / num_steps
        effective_steps = num_steps - skip_last_n

        for i in tqdm(range(effective_steps), desc="Euler CFG Sampling"):
            with torch.no_grad():
                t_val = 1.0 - (i / num_steps)
                t = torch.ones(x.shape[0], 1).to(x.device, x.dtype) * t_val

                x0_pos = self.forward(x, t, pos_cond, text_tokens=text_tokens)
                x0_neg = self.forward(x, t, neg_cond, text_tokens=text_tokens)

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
