"""zeta.py — ZImageDCT pixel-space flow-matching model.

Self-contained: owns patchify/unpatchify, position-ID construction, and
Euler-CFG sampling.  Always predicts x0 then converts to v-prediction.

Architecture:
  - Noise refiner + context refiner (pre-conditioning transformer blocks)
  - Main transformer with AdaLN modulation
  - SimpleMLPAdaLN decoder head with NerfEmbedder (DCT positional encoding)

Usage (training):
    model = ZImageDCT(ZImageDCTParams(**cfg["model_config"]))
    v_pred = model(noisy_image, t, txt_embeds, txt_mask)  # [B, 3, H, W]

Usage (inference):
    images, _ = model.euler_cfg(noise, cfg_scale=4.0, num_steps=28,
                                txt=txt_embeds, txt_mask=txt_mask,
                                neg_txt=neg_embeds, neg_txt_mask=neg_mask)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from einops import rearrange
from torch import Tensor
from tqdm import tqdm

from src.models.flow import create_distribution


# ---------------------------------------------------------------------------
# Patch helpers (from utils.py)
# ---------------------------------------------------------------------------


def vae_flatten(latents: Tensor, patch_size: int = 2):
    """[N, C, H, W] → ([N, num_patches, patch_size²·C], original_shape)"""
    return (
        rearrange(
            latents,
            "n c (h dh) (w dw) -> n (h w) (dh dw c)",
            dh=patch_size,
            dw=patch_size,
        ),
        latents.shape,
    )


def vae_unflatten(latents: Tensor, shape: tuple, patch_size: int = 2) -> Tensor:
    """[N, num_patches, patch_size²·C] → [N, C, H, W]"""
    n, c, h, w = shape
    return rearrange(
        latents,
        "n (h w) (dh dw c) -> n c (h dh) (w dw)",
        dh=patch_size,
        dw=patch_size,
        c=c,
        h=h // patch_size,
        w=w // patch_size,
    )


def prepare_latent_image_ids(
    start_indices,
    height: int,
    width: int,
    patch_size: int = 2,
    max_offset: int = 0,
) -> Tensor:
    """Generate 3D positional IDs [B, num_patches, 3] for image patches.

    Dim 0: sequence index (set to start_indices per batch item).
    Dim 1: height index.
    Dim 2: width index.
    """
    if isinstance(start_indices, list):
        start_indices = torch.tensor(start_indices)

    batch_size = len(start_indices)
    h_patches = height // patch_size
    w_patches = width // patch_size

    latent_image_ids = torch.zeros(h_patches, w_patches, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(h_patches)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(w_patches)[None, :]

    if max_offset > 0:
        offset_y = torch.randint(0, max_offset + 1, (1,)).item()
        offset_x = torch.randint(0, max_offset + 1, (1,)).item()
        latent_image_ids[..., 1] += offset_y
        latent_image_ids[..., 2] += offset_x

    latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
    for i, start_idx in enumerate(start_indices):
        latent_image_ids[i, :, :, 0] = start_idx

    return latent_image_ids.reshape(batch_size, h_patches * w_patches, 3).int()


def make_text_position_ids(
    valid_len: Tensor,
    max_sequence_length: int,
    extra_padding: int = 0,
) -> Tensor:
    """Generate 3D positional IDs [B, max_sequence_length, 3] for text tokens."""
    device = valid_len.device
    valid_len = valid_len + extra_padding
    B = valid_len.shape[0]
    seq = torch.arange(1, max_sequence_length + 1, device=device).unsqueeze(0).expand(B, -1)
    increment_then_repeat = torch.minimum(seq, valid_len.unsqueeze(1))
    pos_ids = torch.zeros((B, max_sequence_length, 3), device=device)
    pos_ids[:, :, 0] = increment_then_repeat
    return pos_ids.int()


# ---------------------------------------------------------------------------
# Timestep schedule (from sampling.py)
# ---------------------------------------------------------------------------


def _time_shift(mu: float, sigma: float, t: Tensor) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    """Build a shifted timestep schedule from t=1 (noise) to t=0 (clean)."""
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        m = (max_shift - base_shift) / (4096 - 256)
        b = base_shift - m * 256
        mu = m * image_seq_len + b
        timesteps = _time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


# ---------------------------------------------------------------------------
# Attention helpers
# ---------------------------------------------------------------------------


def _process_mask(attn_mask: Optional[Tensor], dtype: torch.dtype) -> Optional[Tensor]:
    if attn_mask is None:
        return None
    if attn_mask.ndim == 2:
        attn_mask = attn_mask[:, None, None, :]
    if attn_mask.dtype == torch.bool:
        new_mask = torch.zeros_like(attn_mask, dtype=dtype)
        new_mask.masked_fill_(~attn_mask, float("-inf"))
        return new_mask
    return attn_mask


def _native_attention_wrapper(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    attn_mask = _process_mask(attn_mask, query.dtype)
    out = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
    )
    return out.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def apply_rotary_emb(x_in: Tensor, freqs_cis: Tensor) -> Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size: int, mid_size: Optional[int] = None, frequency_embedding_size: int = 256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period)
                * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
                / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        return self.mlp(t_freq)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ZImageAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads

        self.to_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(n_heads * self.head_dim, dim, bias=False)])

        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        freqs_cis: Optional[Tensor] = None,
    ) -> Tensor:
        query = self.to_q(hidden_states).unflatten(-1, (self.n_heads, -1))
        key   = self.to_k(hidden_states).unflatten(-1, (self.n_kv_heads, -1))
        value = self.to_v(hidden_states).unflatten(-1, (self.n_kv_heads, -1))

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key   = apply_rotary_emb(key,   freqs_cis)

        dtype = query.dtype
        hidden_states = _native_attention_wrapper(
            query.to(dtype), key.to(dtype), value,
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )
        return self.to_out[0](hidden_states.flatten(2, 3).to(dtype))


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
        adaln_embed_dim: int = 256,
    ):
        super().__init__()
        self.modulation = modulation

        self.attention    = ZImageAttention(dim, n_heads, n_kv_heads, qk_norm, norm_eps)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1       = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2       = RMSNorm(dim, eps=norm_eps)

        if modulation:
            self.adaLN_modulation = nn.ModuleList(
                [nn.Linear(min(dim, adaln_embed_dim), 4 * dim, bias=True)]
            )

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor,
        freqs_cis: Tensor,
        adaln_input: Optional[Tensor] = None,
    ) -> Tensor:
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = (
                self.adaLN_modulation[0](adaln_input).unsqueeze(1).chunk(4, dim=2)
            )
            gate_msa  = gate_msa.tanh()
            gate_mlp  = gate_mlp.tanh()
            scale_msa = 1.0 + scale_msa
            scale_mlp = 1.0 + scale_mlp

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            attn_out = self.attention(
                self.attention_norm1(x),
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256,
        axes_dims: List[int] = None,
        axes_lens: List[int] = None,
    ):
        self.theta     = theta
        self.axes_dims = axes_dims or [32, 48, 48]
        self.axes_lens = axes_lens or [1536, 512, 512]
        assert len(self.axes_dims) == len(self.axes_lens)
        self.freqs_cis: Optional[list] = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256):
        with torch.device("cpu"):
            freqs_cis = []
            for d, e in zip(dim, end):
                freqs = 1.0 / (
                    theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d)
                )
                timestep = torch.arange(e, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
            return freqs_cis

    def __call__(self, ids: Tensor) -> Tensor:
        assert ids.ndim >= 2 and ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, self.theta)
        if self.freqs_cis[0].device != device:
            self.freqs_cis = [f.to(device) for f in self.freqs_cis]

        return torch.cat([self.freqs_cis[i][ids[..., i]] for i in range(len(self.axes_dims))], dim=-1)


# ---------------------------------------------------------------------------
# Decoder components
# ---------------------------------------------------------------------------


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class NerfEmbedder(nn.Module):
    """Input projection with 2D DCT-like positional encoding."""

    def __init__(self, in_channels: int, hidden_size_input: int, max_freqs: int):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        self.embedder = nn.Sequential(
            nn.Linear(in_channels + max_freqs ** 2, hidden_size_input)
        )

    @lru_cache(maxsize=4)
    def _fetch_pos(self, patch_size: int, device, dtype):
        pos_x = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")

        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)

        freqs   = torch.linspace(0, self.max_freqs - 1, self.max_freqs, dtype=dtype, device=device)
        freqs_x = freqs[None, :, None]
        freqs_y = freqs[None, None, :]

        coeffs = (1 + freqs_x * freqs_y) ** -1
        dct_x  = torch.cos(pos_x * freqs_x * torch.pi)
        dct_y  = torch.cos(pos_y * freqs_y * torch.pi)
        return (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs ** 2)

    def forward(self, inputs: Tensor) -> Tensor:
        B, P2, C = inputs.shape
        original_dtype = inputs.dtype
        with torch.autocast("cuda", enabled=False):
            patch_size = int(P2 ** 0.5)
            inputs = inputs.float()
            dct    = self._fetch_pos(patch_size, inputs.device, torch.float32).repeat(B, 1, 1)
            inputs = torch.cat([inputs, dct], dim=-1)
            inputs = self.embedder.float()(inputs)
        return inputs.to(original_dtype)


class ResBlock(nn.Module):
    """Residual block with AdaLN modulation. Initialised to identity."""

    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        return x + gate * self.mlp(_modulate(self.in_ln(x), shift, scale))


class DCTFinalLayer(nn.Module):
    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear     = nn.Linear(model_channels, out_channels, bias=True)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias,   0)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.norm_final(x))


class SimpleMLPAdaLN(nn.Module):
    """MLP decoder: NerfEmbedder input projection + AdaLN ResBlocks."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
        patch_size: int,
        max_freqs: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.cond_embed    = nn.Linear(z_channels, patch_size ** 2 * model_channels)
        self.input_embedder = NerfEmbedder(in_channels, model_channels, max_freqs)
        self.res_blocks    = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer   = DCTFinalLayer(model_channels, out_channels)

        nn.init.xavier_uniform_(self.cond_embed.weight)
        nn.init.constant_(self.cond_embed.bias, 0)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x = self.input_embedder(x)
        y = self.cond_embed(c).reshape(c.shape[0], self.patch_size ** 2, -1)
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x)


# ---------------------------------------------------------------------------
# Model params
# ---------------------------------------------------------------------------


@dataclass
class ZImageDCTParams:
    # spatial_patch_size: the real pixel patch size used by vae_flatten/unflatten.
    # in_channels: flattened patch dim = spatial_patch_size² * 3 (RGB).
    # patch_size: internal model param (almost always 1 — the decoder operates on
    #   one flattened patch token at a time, not sub-patches).
    spatial_patch_size: int = 32
    patch_size: int = 1
    f_patch_size: int = 1
    in_channels: int = 3072   # = spatial_patch_size² * 3
    dim: int = 3840
    n_layers: int = 30
    n_refiner_layers: int = 2
    n_heads: int = 30
    n_kv_heads: int = 30
    norm_eps: float = 1e-5
    qk_norm: bool = True
    cap_feat_dim: int = 2560
    rope_theta: int = 256
    t_scale: float = 1000.0
    axes_dims: list[int] = field(default_factory=lambda: [32, 48, 48])
    axes_lens: list[int] = field(default_factory=lambda: [1536, 512, 512])
    adaln_embed_dim: int = 256
    # DCT decoder
    decoder_hidden_size: int = 3840
    decoder_num_res_blocks: int = 4
    decoder_max_freqs: int = 8
    # Training options
    pos_jitter_range: int = 0
    grad_checkpointing: bool = False
    compile_blocks: bool = False  # call model.compile_blocks() after wrapper.setup()


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class ZImageDCT(nn.Module):
    """ZImageDCT pixel-space flow-matching model.

    forward() accepts [B, 3, H, W] images and returns v-predictions in the
    same shape.  Patchify, position-ID construction, and x0→v conversion are
    all handled internally.
    """

    def __init__(self, params: ZImageDCTParams):
        super().__init__()
        self.in_channels        = params.in_channels
        self.spatial_patch_size = params.spatial_patch_size  # used by vae_flatten/unflatten
        self.patch_size         = params.patch_size           # internal decoder patch size (usually 1)
        self.f_patch_size       = params.f_patch_size
        self.dim                = params.dim
        self.n_heads            = params.n_heads
        self.rope_theta         = params.rope_theta
        self.t_scale            = params.t_scale
        self.adaln_embed_dim    = params.adaln_embed_dim
        self.pos_jitter_range   = params.pos_jitter_range
        self.grad_checkpointing = params.grad_checkpointing

        # Input embedder for the backbone
        self.x_embedder = nn.Linear(
            params.f_patch_size * params.patch_size * params.patch_size * params.in_channels,
            params.dim,
            bias=True,
        )

        # Noise refiner (modulated)
        self.noise_refiner = nn.ModuleList([
            ZImageTransformerBlock(
                1000 + i, params.dim, params.n_heads, params.n_kv_heads,
                params.norm_eps, params.qk_norm,
                modulation=True, adaln_embed_dim=params.adaln_embed_dim,
            )
            for i in range(params.n_refiner_layers)
        ])

        # Context refiner (unmodulated)
        self.context_refiner = nn.ModuleList([
            ZImageTransformerBlock(
                i, params.dim, params.n_heads, params.n_kv_heads,
                params.norm_eps, params.qk_norm,
                modulation=False,
            )
            for i in range(params.n_refiner_layers)
        ])

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(
            min(params.dim, params.adaln_embed_dim), mid_size=1024
        )

        # Caption embedder
        self.cap_embedder = nn.Sequential(
            RMSNorm(params.cap_feat_dim, eps=params.norm_eps),
            nn.Linear(params.cap_feat_dim, params.dim, bias=True),
        )

        # Padding tokens (kept for checkpoint compatibility)
        self.x_pad_token   = nn.Parameter(torch.empty((1, params.dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, params.dim)))

        # Main transformer layers
        self.layers = nn.ModuleList([
            ZImageTransformerBlock(
                i, params.dim, params.n_heads, params.n_kv_heads,
                params.norm_eps, params.qk_norm,
                modulation=True, adaln_embed_dim=params.adaln_embed_dim,
            )
            for i in range(params.n_layers)
        ])

        # RoPE embedder
        head_dim = params.dim // params.n_heads
        assert head_dim == sum(params.axes_dims), (
            f"head_dim {head_dim} != sum(axes_dims) {sum(params.axes_dims)}"
        )
        self.axes_dims = params.axes_dims
        self.axes_lens = params.axes_lens
        self.rope_embedder = RopeEmbedder(
            theta=params.rope_theta,
            axes_dims=params.axes_dims,
            axes_lens=params.axes_lens,
        )

        # SimpleMLPAdaLN decoder head
        self.dec_net = SimpleMLPAdaLN(
            in_channels=params.in_channels,
            model_channels=params.decoder_hidden_size,
            out_channels=params.in_channels,
            z_channels=params.dim,
            num_res_blocks=params.decoder_num_res_blocks,
            patch_size=params.patch_size,
            max_freqs=params.decoder_max_freqs,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def compile_blocks(self) -> None:
        """torch.compile each block list.

        Call *after* any FX-based model wrapping (e.g. MultiGPUWrapper.setup())
        to avoid the "FX tracing a dynamo-optimized function" error.
        torch.compile and torch.utils.checkpoint are compatible: compile sees
        the checkpoint boundary as an opaque call; rematerialization happens at
        the autograd level.
        """
        for block in self.noise_refiner:
            block.forward = torch.compile(block.forward)
        for block in self.context_refiner:
            block.forward = torch.compile(block.forward)
        for block in self.layers:
            block.forward = torch.compile(block.forward)
        self.dec_net.forward = torch.compile(self.dec_net.forward)

    # ------------------------------------------------------------------
    # Internal transformer forward (operates on patch sequences)
    # ------------------------------------------------------------------

    def _forward(
        self,
        img: Tensor,       # [B, N, C·P²]
        img_ids: Tensor,   # [B, N, 3]
        img_mask: Tensor,  # [B, N]  bool
        txt: Tensor,       # [B, L, cap_feat_dim]
        txt_ids: Tensor,   # [B, L, 3]
        txt_mask: Tensor,  # [B, L]  bool
        timesteps: Tensor, # [B]  in [0, 1]
    ) -> Tensor:
        B           = img.shape[0]
        num_patches = img.shape[1]

        # Store raw pixel values for the decoder: [B*N, P², C]
        pixel_values = img.reshape(B * num_patches, self.patch_size ** 2, self.in_channels)

        # ZImage uses 0-1000 scale where 0 = full noise, 1000 = full image
        t_scaled = (1 - timesteps) * self.t_scale
        t_emb    = self.t_embedder(t_scaled)

        img_hidden = self.x_embedder(img)
        txt_hidden = self.cap_embedder(txt)

        img_pe = self.rope_embedder(img_ids)
        txt_pe = self.rope_embedder(txt_ids)

        _ckpt = lambda fn, *args: ckpt.checkpoint(fn, *args, use_reentrant=False)

        # Noise refiner
        for layer in self.noise_refiner:
            if self.grad_checkpointing:
                img_hidden = _ckpt(layer, img_hidden, img_mask, img_pe, t_emb)
            else:
                img_hidden = layer(img_hidden, img_mask, img_pe, t_emb)

        # Context refiner
        for layer in self.context_refiner:
            if self.grad_checkpointing:
                txt_hidden = _ckpt(layer, txt_hidden, txt_mask, txt_pe)
            else:
                txt_hidden = layer(txt_hidden, txt_mask, txt_pe)

        # Fuse and run main layers
        mixed_hidden = torch.cat((txt_hidden, img_hidden), dim=1)
        mixed_mask   = torch.cat((txt_mask,   img_mask),   dim=1)
        mixed_pe     = torch.cat((txt_pe,     img_pe),     dim=1)

        for layer in self.layers:
            if self.grad_checkpointing:
                mixed_hidden = _ckpt(layer, mixed_hidden, mixed_mask, mixed_pe, t_emb)
            else:
                mixed_hidden = layer(mixed_hidden, mixed_mask, mixed_pe, t_emb)

        # Extract image hidden states (strip text prefix)
        img_hidden = mixed_hidden[:, txt.shape[1]:, :]  # [B, N, dim]

        # Decoder: [B*N, dim] → [B*N, P², C] → [B, N, C·P²]
        decoder_cond = img_hidden.reshape(B * num_patches, self.dim)
        if self.grad_checkpointing:
            output = _ckpt(self.dec_net, pixel_values, decoder_cond)
        else:
            output = self.dec_net(pixel_values, decoder_cond)
        output = output.reshape(B, num_patches, -1)

        return -output  # flip (matches original ZImage convention)

    # ------------------------------------------------------------------
    # x0 → v-prediction conversion
    # ------------------------------------------------------------------

    def _apply_x0_residual(self, predicted: Tensor, noisy: Tensor, timesteps: Tensor) -> Tensor:
        """Convert x0 prediction to v-prediction.

        eps avoids division by zero at t=0 during training.
        """
        eps = 5e-2 if self.training else 0.0
        return (noisy - predicted) / (timesteps.view(-1, 1, 1, 1) + eps)

    # ------------------------------------------------------------------
    # Public forward: [B, 3, H, W] → [B, 3, H, W] v-prediction
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,        # [B, 3, H, W] noisy image
        t: Tensor,        # [B] or [B, 1, 1, 1]  timestep in [0, 1]
        txt: Tensor,      # [B, L, cap_feat_dim]  text encoder hidden states
        txt_mask: Tensor, # [B, L]  bool attention mask
    ) -> Tensor:
        """Returns v-prediction [B, 3, H, W]."""
        t = t.view(-1)  # ensure [B]
        B, C, H, W = x.shape

        # Patchify using the spatial patch size
        img, orig_shape = vae_flatten(x, self.spatial_patch_size)  # [B, N, C·P²]
        N = img.shape[1]

        # Position IDs
        jitter = 0
        if self.training and self.pos_jitter_range > 0:
            jitter = int(torch.randint(0, self.pos_jitter_range + 1, (1,)).item())

        txt_lengths = txt_mask.sum(dim=1)                          # [B]
        max_txt_len = txt_mask.shape[1]
        # Image sequence starts after the longest text sequence in the batch
        img_start   = txt_lengths.max().item()

        img_ids = prepare_latent_image_ids(
            [img_start] * B, H, W,
            patch_size=self.spatial_patch_size,
            max_offset=jitter,
        ).to(x.device)                                             # [B, N, 3]

        txt_ids = make_text_position_ids(txt_lengths, max_txt_len).to(x.device)  # [B, L, 3]

        img_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)

        # Transformer forward
        out_patches = self._forward(img, img_ids, img_mask, txt, txt_ids, txt_mask, t)

        # Unpatchify
        out_image = vae_unflatten(out_patches, orig_shape, self.spatial_patch_size)  # [B, 3, H, W]

        # x0 → v-prediction (always)
        return self._apply_x0_residual(out_image, x, t)

    # ------------------------------------------------------------------
    # Euler CFG sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def euler_cfg(
        self,
        x: Tensor,
        cfg_scale: float,
        num_steps: int,
        txt: Tensor,
        txt_mask: Tensor,
        neg_txt: Tensor,
        neg_txt_mask: Tensor,
        schedule_mu: float | None = None,
        grid_points: int = 1024,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, list | None]:
        """Euler CFG sampler stepping from t=1 (noise) to t=0 (clean).

        Args:
            x:                  Initial noise [B, 3, H, W].
            cfg_scale:          Classifier-free guidance scale.
            num_steps:          Number of Euler steps.
            txt / txt_mask:     Positive text conditioning.
            neg_txt / neg_txt_mask: Negative text conditioning.
            schedule_mu:        Timestep shift strength (same as flow_baseline).
                                  None  → sequence-length auto-mu via get_schedule.
                                  0.0   → uniform linear schedule (no shift).
                                  float → shifted via create_distribution(mu=value).
            grid_points:        CDF grid resolution for schedule_mu path.
            return_intermediates: If True, return list of CPU tensors at each step.

        Returns:
            (denoised_image, trajectories_or_None)
        """
        B, C, H, W = x.shape
        # Use spatial_patch_size (e.g. 32) for the sequence-length-aware schedule,
        # not self.patch_size (which is 1, the internal decoder token size).
        num_patches = (H // self.spatial_patch_size) * (W // self.spatial_patch_size)

        # Build timestep schedule — identical logic to flow_baseline.euler_cfg.
        if schedule_mu is None:
            # Auto-mu from sequence length (original ZImage behaviour).
            t_seq = torch.tensor(
                get_schedule(num_steps, num_patches, shift=True),
                device=x.device, dtype=x.dtype,
            )
        elif schedule_mu == 0.0:
            # Uniform linear, no shift.
            t_seq = torch.linspace(1.0, 0.0, num_steps + 1, device=x.device, dtype=x.dtype)
        else:
            # Shifted via create_distribution CDF inversion.
            grid_t, grid_p = create_distribution(grid_points, device=x.device, mu=schedule_mu)
            grid_t = grid_t.to(x.dtype)
            grid_p = grid_p.to(x.dtype)

            cdf = torch.cumsum(grid_p, dim=0)
            cdf = cdf / cdf[-1].clamp(min=1e-8)

            q   = torch.linspace(1.0, 0.0, num_steps + 1, device=x.device, dtype=x.dtype)
            idx = torch.searchsorted(cdf, q.clamp(0.0, 1.0)).clamp(1, grid_points - 1)
            cdf_lo, cdf_hi = cdf[idx - 1], cdf[idx]
            t_lo,   t_hi   = grid_t[idx - 1], grid_t[idx]
            frac  = (q - cdf_lo) / (cdf_hi - cdf_lo).clamp(min=1e-8)
            t_seq = t_lo + frac * (t_hi - t_lo)
            t_seq[0]  = 1.0
            t_seq[-1] = 0.0

        trajectories = [x.cpu()] if return_intermediates else None

        for i in tqdm(range(num_steps), desc="Euler CFG"):
            t_curr = t_seq[i]
            t_prev = t_seq[i + 1]
            t_vec  = torch.full((B,), t_curr.item(), dtype=x.dtype, device=x.device)

            v_pos = self.forward(x, t_vec, txt,     txt_mask)
            v_neg = self.forward(x, t_vec, neg_txt, neg_txt_mask)

            velocity = v_neg + cfg_scale * (v_pos - v_neg)
            x = x + (t_prev - t_curr) * velocity

            if return_intermediates:
                trajectories.append(x.cpu())

        return x, trajectories
