"""radiance.py — Chroma/MM-DiT pixel-space flow-matching model.

Self-contained: owns patchify/unpatchify (via Conv2d/fold), position-ID
construction, and Euler-CFG sampling.  Always predicts x0 then converts to
v-prediction.

Architecture:
  - img_in_patch: Conv2d patchify (patch_size=16, hardcoded)
  - Approximator: generates all AdaLN modulation vectors in one shot (distilled,
    run in torch.no_grad() — intentional; see chroma_tobe_refactored notes)
  - depth × DoubleStreamBlock (MM-DiT, parallel img+txt attention)
  - depth_single_blocks × SingleStreamBlock (merged stream)
  - NerfGLUBlock hypernetwork decoder + NerfFinalLayerConv
  - T5 text encoder (4096-dim) in trainer; context_in_dim configurable

Usage (training):
    model = Radiance(RadianceParams(**cfg["model_config"]))
    v_pred = model(noisy_image, t, txt_embeds, txt_mask)  # [B, 3, H, W]

Usage (inference):
    images, _ = model.euler_cfg(noise, cfg_scale=4.0, num_steps=28,
                                txt=txt_embeds, txt_mask=txt_mask,
                                neg_txt=neg_embeds, neg_txt_mask=neg_mask)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from torch import Tensor
from tqdm import tqdm

from src.models.chroma_tobe_refactored.module.layers import (
    Approximator,
    DoubleStreamBlock,
    EmbedND,
    NerfEmbedder,
    NerfFinalLayerConv,
    NerfGLUBlock,
    SingleStreamBlock,
    distribute_modulations,
    timestep_embedding,
)
from src.models.chroma_tobe_refactored.model_dct import modify_mask_to_attend_padding
from src.models.flow import create_distribution


# ---------------------------------------------------------------------------
# Patch helpers (pixel-space, (dh dw c) ordering — same as zeta.py)
# ---------------------------------------------------------------------------

PATCH_SIZE = 16  # hardcoded; empirically better than 32


def vae_flatten(x: Tensor, patch_size: int = PATCH_SIZE):
    """[B, C, H, W] → ([B, N, P²·C], original_shape)  where N = H/P * W/P"""
    from einops import rearrange
    return (
        rearrange(x, "n c (h dh) (w dw) -> n (h w) (dh dw c)", dh=patch_size, dw=patch_size),
        x.shape,
    )


def vae_unflatten(x: Tensor, shape: tuple, patch_size: int = PATCH_SIZE) -> Tensor:
    """[B, N, P²·C] → [B, C, H, W]"""
    from einops import rearrange
    n, c, h, w = shape
    return rearrange(
        x,
        "n (h w) (dh dw c) -> n c (h dh) (w dw)",
        dh=patch_size,
        dw=patch_size,
        c=c,
        h=h // patch_size,
        w=w // patch_size,
    )


# ---------------------------------------------------------------------------
# Position ID helpers
# ---------------------------------------------------------------------------

def prepare_latent_image_ids(batch_size: int, height: int, width: int,
                              patch_size: int = PATCH_SIZE) -> Tensor:
    """Generate [B, N, 3] RoPE position IDs for image patches.

    Dim 0 = 0 (unused / zero for all img patches, Flux/Chroma convention).
    Dim 1 = patch row index.
    Dim 2 = patch col index.
    """
    h_p = height // patch_size
    w_p = width  // patch_size
    ids = torch.zeros(h_p, w_p, 3)
    ids[..., 1] = torch.arange(h_p)[:, None]
    ids[..., 2] = torch.arange(w_p)[None, :]
    ids = ids.reshape(1, h_p * w_p, 3).expand(batch_size, -1, -1)
    return ids.contiguous()


def make_text_position_ids(batch_size: int, seq_len: int) -> Tensor:
    """Generate [B, L, 3] RoPE position IDs for text tokens.

    Dim 0 = token index 0..L-1.
    Dims 1-2 = 0 (Flux/Chroma convention — text has no spatial position).
    """
    ids = torch.zeros(seq_len, 3)
    ids[:, 0] = torch.arange(seq_len)
    return ids.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


# ---------------------------------------------------------------------------
# Timestep schedule helpers (copied from zeta.py — used in euler_cfg)
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
    """Build a shifted timestep schedule from t=1→0."""
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        base_seq   = 256
        max_seq    = 4096
        mu         = base_shift + (max_shift - base_shift) * (image_seq_len - base_seq) / (max_seq - base_seq)
        mu         = float(torch.clamp(torch.tensor(mu), min=base_shift, max=max_shift))
        timesteps  = _time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


# ---------------------------------------------------------------------------
# Model params
# ---------------------------------------------------------------------------

@dataclass
class RadianceParams:
    # Image / text
    in_channels: int = 3              # RGB input (patchify done by Conv2d)
    context_in_dim: int = 4096        # T5-XXL hidden dim (or 2560 for Qwen3)

    # Backbone
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19                   # DoubleStreamBlock count
    depth_single_blocks: int = 38     # SingleStreamBlock count
    axes_dim: list[int] = field(default_factory=lambda: [16, 56, 56])
    theta: int = 10_000
    qkv_bias: bool = True

    # Approximator (modulation distillation)
    approximator_in_dim: int = 64
    approximator_depth: int = 5
    approximator_hidden_size: int = 5120

    # NeRF decoder head
    nerf_hidden_size: int = 64
    nerf_mlp_ratio: int = 4
    nerf_depth: int = 4
    nerf_max_freqs: int = 8

    # Training options
    grad_checkpointing: bool = False
    _use_compiled: bool = False       # per-block torch.compile (call compile_blocks() after setup)

    # patch_size is hardcoded to 16; kept here for reference / config serialisation
    patch_size: int = PATCH_SIZE


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class Radiance(nn.Module):
    """Radiance: Chroma/MM-DiT pixel-space flow-matching model.

    forward() accepts [B, 3, H, W] images and returns v-predictions in the
    same shape.  Patchify (Conv2d), position-ID construction, and x0→v
    conversion are all handled internally.
    """

    def __init__(self, params: RadianceParams):
        super().__init__()
        self.params = params
        self.patch_size = PATCH_SIZE   # hardcoded

        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"hidden_size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"sum(axes_dim)={sum(params.axes_dim)} must equal pe_dim={pe_dim}"
            )

        # RoPE positional embedding
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)

        # Image patchify projection: Conv2d zero-init (matches Chroma)
        self.img_in_patch = nn.Conv2d(
            params.in_channels,
            params.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        nn.init.zeros_(self.img_in_patch.weight)
        nn.init.zeros_(self.img_in_patch.bias)

        # Approximator: distills all block modulation vectors from (t, guidance)
        self.distilled_guidance_layer = Approximator(
            params.approximator_in_dim,
            params.hidden_size,
            params.approximator_hidden_size,
            params.approximator_depth,
        )

        # Text projection
        self.txt_in = nn.Linear(params.context_in_dim, params.hidden_size)

        # MM-DiT double-stream blocks (img + txt in parallel)
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(
                params.hidden_size,
                params.num_heads,
                mlp_ratio=params.mlp_ratio,
                qkv_bias=params.qkv_bias,
                use_compiled=params._use_compiled,
            )
            for _ in range(params.depth)
        ])

        # Single-stream blocks (fused img+txt)
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(
                params.hidden_size,
                params.num_heads,
                mlp_ratio=params.mlp_ratio,
                use_compiled=params._use_compiled,
            )
            for _ in range(params.depth_single_blocks)
        ])

        # NeRF decoder head
        self.nerf_image_embedder = NerfEmbedder(
            in_channels=params.in_channels,
            hidden_size_input=params.nerf_hidden_size,
            max_freqs=params.nerf_max_freqs,
        )
        self.nerf_blocks = nn.ModuleList([
            NerfGLUBlock(
                hidden_size_s=params.hidden_size,
                hidden_size_x=params.nerf_hidden_size,
                mlp_ratio=params.nerf_mlp_ratio,
                use_compiled=params._use_compiled,
            )
            for _ in range(params.nerf_depth)
        ])
        self.nerf_final_layer_conv = NerfFinalLayerConv(
            params.nerf_hidden_size,
            out_channels=params.in_channels,
            use_compiled=params._use_compiled,
        )

        # Modulation vector count (matches distribute_modulations layout):
        #   3 per single block + 2×3 per double block (img+txt) + 2 final-layer
        self.mod_index_length = (
            3 * params.depth_single_blocks
            + 2 * 6 * params.depth
            + 2
        )
        self.depth_single_blocks = params.depth_single_blocks
        self.depth_double_blocks  = params.depth
        self.approximator_in_dim  = params.approximator_in_dim

        self.register_buffer(
            "mod_index",
            torch.tensor(list(range(self.mod_index_length))),
            persistent=False,
        )

        self.grad_checkpointing = params.grad_checkpointing

    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def compile_blocks(self) -> None:
        """torch.compile each block.  Call *after* MultiGPUWrapper.setup()."""
        for block in self.double_blocks:
            block.forward = torch.compile(block.forward)
        for block in self.single_blocks:
            block.forward = torch.compile(block.forward)
        for block in self.nerf_blocks:
            block.forward = torch.compile(block.forward)

    # ------------------------------------------------------------------
    # Internal forward (operates on raw [B, C, H, W])
    # ------------------------------------------------------------------

    def _forward(
        self,
        img: Tensor,       # [B, C, H, W]  noisy image
        img_ids: Tensor,   # [B, N, 3]
        txt: Tensor,       # [B, L, context_in_dim]
        txt_ids: Tensor,   # [B, L, 3]
        txt_mask: Tensor,  # [B, L]  bool
        timesteps: Tensor, # [B]  in [0, 1]
    ) -> Tensor:
        """Returns predicted x0  [B, C, H, W]."""
        if img.ndim != 4:
            raise ValueError("img must be [B, C, H, W]")
        B, C, H, W = img.shape

        # --- Extract raw patch pixels for the NeRF decoder head ---
        # unfold → [B, C*P*P, NumPatches] → [B, NumPatches, C*P*P]
        nerf_pixels = F.unfold(img, kernel_size=self.patch_size, stride=self.patch_size)
        nerf_pixels = nerf_pixels.transpose(1, 2)   # [B, N, C*P*P]
        num_patches = nerf_pixels.shape[1]

        # --- Patchify image for transformer backbone ---
        img_hidden = self.img_in_patch(img)          # [B, hidden, H/P, W/P]
        img_hidden = img_hidden.flatten(2).transpose(1, 2)  # [B, N, hidden]

        # --- Text projection ---
        txt_hidden = self.txt_in(txt)                # [B, L, hidden]

        # --- Distill all modulation vectors (Approximator in no_grad) ---
        # Hardcode guidance=0 (pixel-space flow-matching has no guidance distillation)
        with torch.no_grad():
            self.mod_index = self.mod_index.to(img.device)
            distill_timestep = timestep_embedding(timesteps, self.approximator_in_dim // 4)
            distil_guidance  = timestep_embedding(
                torch.zeros(B, device=img.device, dtype=timesteps.dtype),
                self.approximator_in_dim // 4,
            )
            modulation_index = timestep_embedding(self.mod_index, self.approximator_in_dim // 2)
            modulation_index = modulation_index.unsqueeze(0).expand(B, -1, -1)
            timestep_guidance = (
                torch.cat([distill_timestep, distil_guidance], dim=1)
                .unsqueeze(1)
                .expand(-1, self.mod_index_length, -1)
            )
            input_vec  = torch.cat([timestep_guidance, modulation_index], dim=-1)
            mod_vectors = self.distilled_guidance_layer(input_vec.requires_grad_(True))

        mod_vectors_dict = distribute_modulations(
            mod_vectors, self.depth_single_blocks, self.depth_double_blocks
        )

        # --- RoPE position embeddings ---
        ids = torch.cat((txt_ids, img_ids), dim=1)  # [B, L+N, 3]
        pe  = self.pe_embedder(ids)                  # [B, 1, L+N, head_dim]

        # --- Attention mask ---
        max_len = txt_hidden.shape[1]
        with torch.no_grad():
            txt_mask_padded = modify_mask_to_attend_padding(txt_mask, max_len, num_extra_padding=1)
            txt_img_mask = torch.cat(
                [
                    txt_mask_padded,
                    torch.ones(B, num_patches, device=txt_mask.device),
                ],
                dim=1,
            )
            txt_img_mask = txt_img_mask.float().T @ txt_img_mask.float()
            txt_img_mask = (
                txt_img_mask[None, None, ...]
                .expand(B, self.params.num_heads, -1, -1)
                .int()
                .bool()
            )

        _ckpt = lambda fn, *args: ckpt.checkpoint(fn, *args, use_reentrant=False)

        # --- Double-stream blocks ---
        for i, block in enumerate(self.double_blocks):
            img_mod  = mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"]
            txt_mod  = mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"]
            distill_vec = [img_mod, txt_mod]
            if self.grad_checkpointing:
                img_hidden, txt_hidden = _ckpt(
                    block, img_hidden, txt_hidden, pe, distill_vec, txt_img_mask
                )
            else:
                img_hidden, txt_hidden = block(
                    img=img_hidden, txt=txt_hidden,
                    pe=pe, distill_vec=distill_vec, mask=txt_img_mask,
                )

        # --- Merge streams for single-stream blocks ---
        merged = torch.cat((txt_hidden, img_hidden), dim=1)  # [B, L+N, hidden]
        for i, block in enumerate(self.single_blocks):
            single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
            if self.grad_checkpointing:
                merged = _ckpt(block, merged, pe, single_mod, txt_img_mask)
            else:
                merged = block(merged, pe=pe, distill_vec=single_mod, mask=txt_img_mask)

        # Strip text prefix
        img_hidden = merged[:, txt_hidden.shape[1]:, :]  # [B, N, hidden]

        # --- NeRF decoder ---
        # reshape conditioning to [B*N, hidden]
        nerf_cond   = img_hidden.reshape(B * num_patches, self.params.hidden_size)
        # reshape pixels to [B*N, C, P*P] → [B*N, P*P, C]
        nerf_pixels = nerf_pixels.reshape(B * num_patches, C, self.patch_size ** 2)
        nerf_pixels = nerf_pixels.transpose(1, 2)          # [B*N, P*P, C]

        img_dct = self.nerf_image_embedder(nerf_pixels)     # [B*N, P*P, nerf_hidden]
        # reshape to [B*N, 1, hidden] expected by NerfGLUBlock's param_generator
        for block in self.nerf_blocks:
            if self.grad_checkpointing:
                img_dct = _ckpt(block, img_dct, nerf_cond)
            else:
                img_dct = block(img_dct, nerf_cond)

        # --- Reconstruct image via fold ---
        # norm is applied on [B*N, P*P, nerf_hidden]
        img_dct = self.nerf_final_layer_conv.norm(img_dct)
        # → [B*N, C, P*P]
        img_dct = img_dct.transpose(1, 2)
        # → [B, N, C*P*P]
        img_dct = img_dct.reshape(B, num_patches, -1)
        # → [B, C*P*P, N]
        img_dct = img_dct.transpose(1, 2)
        # fold → [B, nerf_hidden, H, W]
        img_dct = F.fold(
            img_dct,
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        # 3×3 conv projection → [B, C, H, W]
        output = self.nerf_final_layer_conv.conv(img_dct)

        return output  # predicted x0

    # ------------------------------------------------------------------
    # x0 → v-prediction conversion
    # ------------------------------------------------------------------

    def _apply_x0_residual(self, predicted: Tensor, noisy: Tensor, t: Tensor) -> Tensor:
        """Convert x0 prediction to v-prediction.

        v = (noisy - x0) / (t + eps)
        eps avoids divide-by-zero at t=0 during training.
        """
        eps = 5e-2 if self.training else 0.0
        return (noisy - predicted) / (t.view(-1, 1, 1, 1) + eps)

    # ------------------------------------------------------------------
    # Public forward: [B, 3, H, W] → [B, 3, H, W] v-prediction
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,         # [B, 3, H, W]  noisy image
        t: Tensor,         # [B] or [B, 1, 1, 1]  timestep in [0, 1]
        txt: Tensor,       # [B, L, context_in_dim]
        txt_mask: Tensor,  # [B, L]  bool
    ) -> Tensor:
        """Returns v-prediction [B, 3, H, W]."""
        t = t.view(-1)
        B, C, H, W = x.shape

        img_ids = prepare_latent_image_ids(B, H, W, self.patch_size).to(x.device)
        txt_ids = make_text_position_ids(B, txt.shape[1]).to(x.device)

        predicted_x0 = self._forward(x, img_ids, txt, txt_ids, txt_mask, t)
        return self._apply_x0_residual(predicted_x0, x, t)

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
            x:               Initial noise [B, 3, H, W].
            cfg_scale:       Classifier-free guidance scale.
            num_steps:       Number of Euler steps.
            txt / txt_mask:  Positive text conditioning.
            neg_txt / neg_txt_mask: Negative text conditioning.
            schedule_mu:     Timestep shift strength.
                               None  → sequence-length auto-mu (get_schedule).
                               0.0   → uniform linear (no shift).
                               float → shifted via create_distribution CDF inversion.
            grid_points:     CDF grid resolution for schedule_mu path.
            return_intermediates: If True, return CPU tensors at each step.

        Returns:
            (denoised_image, trajectories_or_None)
        """
        B, C, H, W = x.shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)

        # Build timestep schedule
        if schedule_mu is None:
            t_seq = torch.tensor(
                get_schedule(num_steps, num_patches, shift=True),
                device=x.device, dtype=x.dtype,
            )
        elif schedule_mu == 0.0:
            t_seq = torch.linspace(1.0, 0.0, num_steps + 1, device=x.device, dtype=x.dtype)
        else:
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
