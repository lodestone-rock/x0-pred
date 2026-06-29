"""plksr_diff.py — PLKSRDiff: flow-matching SR diffusion model.

Architecture overview
---------------------
- Noise lives at LR resolution with ``noise_ch = in_ch * upscale²`` channels
  (pixel-unshuffled space). A 2× model uses 12-channel noise; 4× uses 48.
- LR image is concatenated channel-wise with the noisy latent at the stem.
- LR conditioning uses an EA-style attention + GroupNorm path.
- Timestep is injected via AdaLN: a 1-D scalar ``t ∈ [0,1]`` is projected to
  ``(scale, shift)`` of shape ``(B, dim, 1, 1)`` and applied after each
  PLKBlock's GroupNorm.
- The model outputs a delta in the same shuffled space. At inference end,
  call ``F.pixel_shuffle(pred, upscale)`` and add the bicubic/nearest baseline.

All utility classes (DySample, DropPath, DCCM, PLKConv2d, EA, PLKBlock) are
inlined here verbatim — no neosr imports required.
"""
from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import trunc_normal_


# ---------------------------------------------------------------------------
# Inlined from neosr/archs/arch_util.py
# ---------------------------------------------------------------------------

def _drop_path(
    x: Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        return _drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class DySample(nn.Module):
    """Dynamic Sample upsampler.
    Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    """

    def _init_pos(self) -> Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def __init__(
        self,
        in_channels: int,
        out_ch: int,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
    ) -> None:
        super().__init__()
        assert in_channels >= groups and in_channels % groups == 0, \
            "Incorrect in_channels and groups values."

        out_channels = 2 * groups * scale ** 2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(in_channels, out_ch, kernel_size=1)

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        nn.init.trunc_normal_(self.offset.weight, std=0.02)
        nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer("init_pos", self._init_pos())

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H, device=x.device) + 0.5
        coords_w = torch.arange(W, device=x.device) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .to(dtype=x.dtype, device=x.device)
        )
        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)
        return output


# ---------------------------------------------------------------------------
# Inlined from neosr/archs/realplksr_arch.py
# ---------------------------------------------------------------------------

class DCCM(nn.Sequential):
    """Doubled Convolutional Channel Mixer."""

    def __init__(self, dim: int) -> None:
        super().__init__(
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.Mish(),
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
        )
        trunc_normal_(self[-1].weight, std=0.02)


class PLKConv2d(nn.Module):
    """Partial Large Kernel Convolutional Layer."""

    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)
        trunc_normal_(self.conv.weight, std=0.02)
        self.idx = dim

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.idx, x.size(1) - self.idx], dim=1)
            x1 = self.conv(x1)
            return torch.cat([x1, x2], dim=1)
        x[:, : self.idx] = self.conv(x[:, : self.idx])
        return x


class EA(nn.Module):
    """Element-wise Attention."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.Sigmoid())
        trunc_normal_(self.f[0].weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.f(x)


class PLKBlock(nn.Module):
    """PLK residual block (unchanged from realplksr_arch)."""

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        split_ratio: float,
        norm_groups: int,
        use_ea: bool = True,
    ) -> None:
        super().__init__()
        self.channel_mixer = DCCM(dim)
        pdim = int(dim * split_ratio)
        self.lk = PLKConv2d(pdim, kernel_size)
        self.attn = EA(dim) if use_ea else nn.Identity()
        self.refine = nn.Conv2d(dim, dim, 1, 1, 0)
        trunc_normal_(self.refine.weight, std=0.02)
        self.norm = nn.GroupNorm(norm_groups, dim)
        nn.init.constant_(self.norm.bias, 0)
        nn.init.constant_(self.norm.weight, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        x_skip = x
        x = self.channel_mixer(x)
        x = self.lk(x)
        x = self.attn(x)
        x = self.refine(x)
        x = self.norm(x)
        return x + x_skip


# ---------------------------------------------------------------------------
# New modules for diffusion conditioning
# ---------------------------------------------------------------------------

class TimestepEmbed(nn.Module):
    """Project scalar t ∈ [0,1] → (scale, shift) each (B, dim, 1, 1).

    Uses a small 2-layer MLP with SiLU activation so the embedding is
    nonlinear even for a scalar input.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )

    def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
        # t: (B,) or (B, 1)
        t = t.view(-1, 1).float()
        out = self.net(t)                        # (B, dim*2)
        scale, shift = out.chunk(2, dim=-1)      # each (B, dim)
        return scale.unsqueeze(-1).unsqueeze(-1), shift.unsqueeze(-1).unsqueeze(-1)


class PLKBlockCond(nn.Module):
    """PLKBlock wrapped with AdaLN timestep conditioning.

    After the PLKBlock residual is computed, applies:
        h = block(x)
        h = h * (1 + scale) + shift
    where scale and shift come from TimestepEmbed.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        split_ratio: float,
        norm_groups: int,
        use_ea: bool = True,
    ) -> None:
        super().__init__()
        self.block = PLKBlock(dim, kernel_size, split_ratio, norm_groups, use_ea)

    def forward(self, x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
        h = self.block(x)
        return h * (1.0 + scale) + shift


class LRConditioner(nn.Module):
    """Fuse noisy latent and LR image into a feature map.

    Takes ``concat([x_noisy, lr], dim=1)`` of shape ``(B, noise_ch + in_ch, H, W)``
    and produces a conditioned feature ``(B, dim, H, W)`` via:
        1×1 Conv → EA attention → GroupNorm
    """

    def __init__(self, noise_ch: int, in_ch: int, dim: int, norm_groups: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(noise_ch + in_ch, dim, 1, 1, 0)
        trunc_normal_(self.proj.weight, std=0.02)
        self.attn = EA(dim)
        self.norm = nn.GroupNorm(norm_groups, dim)
        nn.init.constant_(self.norm.bias, 0)
        nn.init.constant_(self.norm.weight, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        h = self.proj(x)
        h = self.attn(h)
        return self.norm(h)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class PLKSRDiff(nn.Module):
    """Flow-matching SR diffusion model backed by PLKBlocks.

    Input / output live in pixel-unshuffled (LR) space:
        noise_ch = in_ch * upscale²      (e.g. 12 for 2×, 48 for 4×)

    Forward signature::

        pred_delta = model(x_noisy, lr, t)

    where:
        x_noisy  : (B, noise_ch, H_lr, W_lr) — noisy latent
        lr       : (B, in_ch,   H_lr, W_lr)  — degraded LR image [0, 1]
        t        : (B,)                       — timestep scalar ∈ [0, 1]
        returns    (B, noise_ch, H_lr, W_lr)  — predicted clean delta x0

    At the last inference step call::

        sr = F.pixel_shuffle(pred_delta, upscale)
             + F.interpolate(lr_degraded, scale_factor=upscale, mode='nearest')
    """

    def __init__(
        self,
        in_ch: int = 3,
        dim: int = 64,
        n_blocks: int = 16,
        upscale: int = 2,
        kernel_size: int = 13,
        split_ratio: float = 0.25,
        norm_groups: int = 4,
        use_ea: bool = True,
        dysample: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.upscale = upscale
        self.in_ch = in_ch
        noise_ch = in_ch * upscale ** 2
        self.noise_ch = noise_ch

        # Stem: fuse noisy latent + LR channels
        self.stem = nn.Conv2d(noise_ch + in_ch, dim, 3, 1, 1)
        trunc_normal_(self.stem.weight, std=0.02)

        # LR conditioning branch (parallel path, additive)
        self.lr_cond = LRConditioner(noise_ch, in_ch, dim, norm_groups)

        # Timestep embedding
        self.t_embed = TimestepEmbed(dim)

        # Main block stack
        self.blocks = nn.ModuleList([
            PLKBlockCond(dim, kernel_size, split_ratio, norm_groups, use_ea)
            for _ in range(n_blocks)
        ])

        # Output head: back to shuffled noise space
        self.head = nn.Conv2d(dim, noise_ch, 3, 1, 1)
        trunc_normal_(self.head.weight, std=0.02)

        # Optional learned upsampler (used only at inference to go → HR)
        self.dysample = dysample
        if dysample and upscale != 1:
            groups_dy = in_ch if upscale % 2 != 0 else 4
            self.to_img = DySample(
                noise_ch, in_ch, upscale,
                groups=groups_dy, end_convolution=True,
            )
        else:
            self.to_img = nn.PixelShuffle(upscale)

    def forward(self, x_noisy: Tensor, lr: Tensor, t: Tensor) -> Tensor:
        """Predict clean delta x0 in shuffled (LR) space.

        Args:
            x_noisy : (B, noise_ch, H, W) — noisy latent
            lr      : (B, in_ch, H, W)    — LR image [0, 1]
            t       : (B,)                — timestep ∈ [0, 1]

        Returns:
            (B, noise_ch, H, W) — predicted clean delta x0
        """
        fused = torch.cat([x_noisy, lr], dim=1)   # (B, noise_ch+in_ch, H, W)

        # Conditioning feature (additive residual)
        lr_feat = self.lr_cond(fused)

        # Stem embedding
        h = self.stem(fused) + lr_feat

        # Timestep AdaLN scale/shift
        scale, shift = self.t_embed(t)

        for block in self.blocks:
            h = block(h, scale, shift)

        return self.head(h)

    @torch.no_grad()
    def upsample_delta(self, delta: Tensor) -> Tensor:
        """Convert shuffled delta → HR image patch (without adding baseline).

        delta : (B, noise_ch, H_lr, W_lr)
        returns (B, in_ch, H_hr, W_hr)
        """
        if self.dysample and self.upscale != 1:
            return self.to_img(delta)
        return F.pixel_shuffle(delta, self.upscale)

    @torch.no_grad()
    def euler_sample(
        self,
        lr: Tensor,
        num_steps: int = 30,
        t_start: float = 1.0,
        t_end: float = 0.0,
    ) -> Tensor:
        """Simple Euler sampler.  Returns the final SR image (HR space).

        lr : (B, in_ch, H_lr, W_lr) — LR conditioning image [0, 1]
        """
        device = lr.device
        B = lr.shape[0]
        x = torch.randn(B, self.noise_ch, lr.shape[2], lr.shape[3],
                        device=device, dtype=lr.dtype)

        ts = torch.linspace(t_start, t_end, num_steps + 1, device=device)
        for i in range(num_steps):
            t_cur = ts[i].expand(B)
            t_next = ts[i + 1]
            dt = t_next - ts[i]
            with torch.autocast("cuda", torch.bfloat16):
                v = self(x, lr, t_cur)
            x = x + v * dt

        # Decode: pixel-unshuffle + add nearest baseline
        delta_hr = self.upsample_delta(x.float())
        baseline = F.interpolate(lr.float(), scale_factor=self.upscale, mode="nearest")
        return (delta_hr + baseline).clamp(0.0, 1.0)
