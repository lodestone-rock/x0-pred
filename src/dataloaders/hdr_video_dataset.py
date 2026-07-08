"""hdr_video_dataset.py — DiffHDR-faithful HDR video dataset.

Extends the existing hdr_dataset_video.py projection pipeline with the paper's
LDR synthesis (§3.1.2) and luminance-based mask detection (§3.4.1).

Pipeline per clip:
  1. Load equirectangular HDRI (reuse HDRRawDataset).
  2. Project pseudo-video via perspective camera motion (reuse
     _sample_camera_path + _build_grid_with_roll + grid_sample).
  3. Exposure shift: Δ ∈ [-2, 2] stops, scale linear by 2^Δ.
  4. Camera noise: heteroscedastic Gaussian, AR(1) temporal correlation ρ=0.5.
  5. Quantization + clipping: sRGB → clip [0,1] → 8-bit quantize → LDR input.
  6. Luminance mask: Rec.709 luminance, τ_high=0.95, τ_low=0.05, EMA α=0.7.
  7. HDR target: scene-linear crops (before LDR synthesis).

Output (per clip, all on GPU float32):
  hdr_linear : (B, N, 3, H, W) — scene-linear HDR (target, before Log-Gamma)
  ldr_input  : (B, N, 3, H, W) — synthesized LDR in [-1, 1] (VAE input format)
  mask       : (B, 1, N, H, W) — binary exposure mask (1 = clipped)
  caption    : list[str]        — structured captions from precomputed JSON
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Reuse the existing projection + loading infrastructure.
_HDR_DIR = os.path.dirname(os.path.abspath(__file__))
# The hdr_dataset_video.py lives in /mnt/datapool_u2/lodestone/hdr/
import sys
sys.path.insert(0, "/mnt/datapool_u2/lodestone/hdr")
from hdr_dataset_gpu import HDRRawDataset, load_hdr  # noqa: E402
from hdr_dataset_video import (  # noqa: E402
    VideoAugConfig,
    _sample_camera_path,
    _build_grid_with_roll,
)

__all__ = ["DiffHDRVideoConfig", "DiffHDRVideoDataset", "DiffHDRVideoTransformGPU"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class DiffHDRVideoConfig:
    """Configuration for the DiffHDR video dataset.

    Camera motion parameters are inherited from VideoAugConfig; the key
    DiffHDR-specific knobs are the LDR synthesis and mask parameters.
    """

    # Clip shape
    num_frames: int = 81          # paper: 81 frames
    fps: float = 24.0
    frame_size: int = 720         # paper: 720p (height); width = frame_size * 16/9
    frame_aspect: float = 16 / 9  # width/height ratio
    proj_size: int = 768          # intermediate projection resolution

    # Camera motion (paper §3.1.1: zoom-in/out, rotation)
    yaw_speed_deg_per_sec: float = 30.0
    pitch_speed_deg_per_sec: float = 10.0
    fov_speed_deg_per_sec: float = 15.0
    roll_max_deg: float = 0.0
    fov_min_deg: float = 60.0
    fov_max_deg: float = 110.0
    pitch_start_std_deg: float = 10.0
    motion_smoothing: Literal["linear", "ease"] = "linear"
    jitter_yaw_deg: float = 0.5
    jitter_pitch_deg: float = 0.5
    jitter_fov_deg: float = 1.0
    hflip: bool = True

    # LDR synthesis (paper §3.1.2)
    ev_range: float = 2.0         # exposure shift Δ ∈ [-2, 2] stops
    ev_drift_per_sec: float = 0.5
    noise_sigma_s: float = 8.5e-4  # signal-dependent noise std
    noise_sigma_c: float = 1.5e-5  # stationary noise std
    noise_rho: float = 0.5         # AR(1) temporal correlation
    quantize_bits: int = 8         # 8-bit LDR quantization

    # Luminance mask (paper §3.4.1)
    mask_tau_high: float = 0.95    # overexposed threshold
    mask_tau_low: float = 0.05     # underexposed threshold
    mask_ema_alpha: float = 0.7    # temporal EMA smoothing

    # Color temp jitter (optional, mild)
    color_temp_jitter: float = 0.05

    # Captioning
    caption_json_path: str = "runs/polyheaven/captions.json"

    # HDRI loading
    hdr_dir: str = "runs/polyheaven"
    hdr_glob: str = "*_1k.hdr"
    hdr_max_dim: int = 1024

    # DataLoader
    num_workers: int = 4
    prefetch_factor: int = 2


# ---------------------------------------------------------------------------
# GPU transform
# ---------------------------------------------------------------------------
class DiffHDRVideoTransformGPU(nn.Module):
    """Generate DiffHDR training clips from equirectangular HDRIs on GPU.

    Called once per batch on the training device.
    """

    def __init__(self, cfg: DiffHDRVideoConfig | None = None):
        super().__init__()
        self.cfg = cfg or DiffHDRVideoConfig()

    @torch.no_grad()
    def forward(self, equirects: Tensor) -> dict[str, Tensor]:
        """Generate one DiffHDR clip per equirect in the batch.

        Args:
            equirects: (B, 3, H, W) scene-linear float32 on GPU.

        Returns:
            dict with keys:
              hdr_linear: (B, N, 3, H, W) — scene-linear HDR target
              ldr_input:  (B, N, 3, H, W) — synthesized LDR in [-1, 1]
              mask:       (B, 1, N, H, W) — binary exposure mask
        """
        cfg = self.cfg
        B, C, src_h, src_w = equirects.shape
        N = cfg.num_frames
        fs_h = cfg.frame_size
        fs_w = int(cfg.frame_size * cfg.frame_aspect)
        # proj_size must be >= max(fs_h, fs_w) so the crop fits.
        ps = max(cfg.proj_size, fs_h, fs_w)
        device, dtype = equirects.device, equirects.dtype

        # ── 1. Sample smooth camera trajectories ─────────────────────────
        yaw_path, pitch_path, fov_path, roll_path = _sample_camera_path(
            B, N, cfg, device, dtype
        )
        yaw_flat = yaw_path.reshape(B * N)
        pitch_flat = pitch_path.reshape(B * N)
        fov_flat = fov_path.reshape(B * N)
        roll_flat = roll_path.reshape(B * N)

        # ── 2. Tile equirects and project all frames ──────────────────────
        eq_tiled = equirects.repeat_interleave(N, dim=0)  # (B*N, 3, H, W)
        grid = _build_grid_with_roll(
            yaw_flat, pitch_flat, fov_flat, roll_flat,
            ps, ps, device, dtype,
        )
        proj = F.grid_sample(
            eq_tiled, grid, mode="bilinear",
            padding_mode="border", align_corners=False,
        ).clamp(min=0.0)  # (B*N, 3, ps, ps)

        # Centre-crop to target resolution
        y0 = (ps - fs_h) // 2
        x0 = (ps - fs_w) // 2
        crops = proj[:, :, y0:y0 + fs_h, x0:x0 + fs_w]  # (B*N, 3, fs_h, fs_w)

        # Optional hflip (per clip)
        if cfg.hflip:
            flip_mask = torch.rand(B, device=device) < 0.5
            flip_idx = flip_mask.repeat_interleave(N).nonzero(as_tuple=True)[0]
            if flip_idx.numel() > 0:
                crops[flip_idx] = crops[flip_idx].flip(-1)

        # Optional color temp jitter (pre-tonemap, on HDR)
        if cfg.color_temp_jitter > 0.0:
            scale = 1.0 + (torch.rand(B, 3, device=device, dtype=dtype) * 2 - 1) * cfg.color_temp_jitter
            scale = scale.unsqueeze(1).expand(B, N, 3).reshape(B * N, 3, 1, 1)
            crops = (crops * scale).clamp(min=0.0)

        # ── 3. Save HDR target (scene-linear, before LDR synthesis) ───────
        hdr_linear = crops.reshape(B, N, C, fs_h, fs_w).clone()

        # ── 4. Exposure shift: Δ ∈ [-ev_range, ev_range] stops ────────────
        ev_base = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * cfg.ev_range
        clip_dur = (N - 1) / max(cfg.fps, 1e-6)
        max_drift = cfg.ev_drift_per_sec * clip_dur
        ev_end = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * max_drift
        t = torch.linspace(0, 1, N, device=device, dtype=dtype)
        ev = ev_base.unsqueeze(1) + t.unsqueeze(0) * ev_end.unsqueeze(1)  # (B, N)
        gain = (2.0 ** ev).reshape(B * N, 1, 1, 1)
        exposed = crops * gain  # (B*N, 3, fs_h, fs_w) linear

        # ── 5. Linear → sRGB (standard LDR formation) ─────────────────────
        ldr_linear = _linear_to_srgb(exposed.clamp(min=0.0))

        # ── 6. Heteroscedastic AR(1) camera noise (paper §3.1.2, eq. 1-2) ─
        if cfg.noise_sigma_s > 0 or cfg.noise_sigma_c > 0:
            # Sample (σ_s, σ_c) per clip, shared across frames
            sigma_s = torch.rand(B, 1, device=device, dtype=dtype) * cfg.noise_sigma_s
            sigma_c = torch.rand(B, 1, device=device, dtype=dtype) * cfg.noise_sigma_c
            sigma_s = sigma_s.expand(B, N).reshape(B * N)
            sigma_c = sigma_c.expand(B, N).reshape(B * N)

            # AR(1) noise field: ε_t = ρ·ε_{t-1} + sqrt(1-ρ²)·u_t
            rho = cfg.noise_rho
            eps = torch.zeros(B * N, C, fs_h, fs_w, device=device, dtype=dtype)
            sqrt_one_minus_rho2 = math.sqrt(1.0 - rho * rho)
            for i in range(N):
                u = torch.randn(B, C, fs_h, fs_w, device=device, dtype=dtype)
                if i == 0:
                    eps_batch = u  # ε_0 = u_0 (stationary)
                else:
                    eps_prev = eps[(i - 1) * B: i * B]
                    eps_batch = rho * eps_prev + sqrt_one_minus_rho2 * u
                eps[i * B: (i + 1) * B] = eps_batch

            # n_t(L) = L * σ_s² + σ_c², then sqrt for std
            L = ldr_linear  # signal intensity in [0,1] linear
            var = L * sigma_s.view(B * N, 1, 1, 1) ** 2 + sigma_c.view(B * N, 1, 1, 1) ** 2
            noise = eps * torch.sqrt(var.clamp(min=0.0))
            ldr_linear = (L + noise).clamp(min=0.0)

        # ── 7. Clip to [0, 1] and quantize to 8-bit ────────────────────────
        ldr_01 = ldr_linear.clamp(0.0, 1.0)
        if cfg.quantize_bits > 0:
            levels = 2 ** cfg.quantize_bits
            ldr_01 = torch.round(ldr_01 * (levels - 1)) / (levels - 1)

        # ── 8. Luminance-based mask detection (paper §3.4.1) ──────────────
        # Linearize sRGB, compute Rec.709 luminance
        ldr_lin_for_mask = _srgb_to_linear(ldr_01)
        lum = 0.2126 * ldr_lin_for_mask[:, 0] + 0.7152 * ldr_lin_for_mask[:, 1] + 0.0722 * ldr_lin_for_mask[:, 2]
        # (B*N, fs_h, fs_w)
        mask_raw = ((lum > cfg.mask_tau_high) | (lum < cfg.mask_tau_low)).float()

        # Reshape to (B, N, fs_h, fs_w) for temporal EMA
        mask_seq = mask_raw.reshape(B, N, fs_h, fs_w)
        mask_ema = torch.zeros_like(mask_seq)
        mask_ema[:, 0] = mask_seq[:, 0]
        alpha = cfg.mask_ema_alpha
        for i in range(1, N):
            mask_ema[:, i] = alpha * mask_seq[:, i] + (1 - alpha) * mask_ema[:, i - 1]
        # Binarize after EMA
        mask = (mask_ema > 0.5).float()  # (B, N, fs_h, fs_w)

        # ── 9. Build output tensors ────────────────────────────────────────
        ldr_input = (ldr_01 * 2.0 - 1.0).reshape(B, N, C, fs_h, fs_w)  # [-1, 1]
        mask_out = mask.unsqueeze(1)  # (B, 1, N, fs_h, fs_w)

        return {
            "hdr_linear": hdr_linear,   # (B, N, 3, H, W) scene-linear
            "ldr_input": ldr_input,     # (B, N, 3, H, W) [-1, 1]
            "mask": mask_out,           # (B, 1, N, H, W) binary
        }


# ---------------------------------------------------------------------------
# Dataset (CPU side: loads raw equirect + caption)
# ---------------------------------------------------------------------------
class DiffHDRVideoDataset(torch.utils.data.Dataset):
    """CPU dataset that loads raw equirectangular HDRIs + captions.

    The GPU transform (DiffHDRVideoTransformGPU) handles projection and
    LDR synthesis.  This dataset just loads the raw HDRI and looks up its
    caption from a precomputed JSON.
    """

    def __init__(self, cfg: DiffHDRVideoConfig | None = None):
        self.cfg = cfg or DiffHDRVideoConfig()
        self.raw_dataset = HDRRawDataset(
            hdr_dir=self.cfg.hdr_dir,
            glob_pat=self.cfg.hdr_glob,
            max_dim=self.cfg.hdr_max_dim,
        )
        # Load captions
        self.captions: dict[str, dict] = {}
        if os.path.exists(self.cfg.caption_json_path):
            with open(self.cfg.caption_json_path) as f:
                self.captions = json.load(f)

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx: int):
        # Load raw equirect (3, H, W) float32
        raw = self.raw_dataset[idx % len(self.raw_dataset)]
        # Look up caption
        path = self.raw_dataset.files[idx % len(self.raw_dataset)]
        stem = os.path.splitext(os.path.basename(path))[0]
        cap = self.captions.get(stem, {})
        over = cap.get("overexposed", "")
        under = cap.get("underexposed", "")
        if over or under:
            caption = f"[Overexposed: {over}]; [Underexposed: {under}]"
        else:
            caption = ""
        return raw, caption


# ---------------------------------------------------------------------------
# Color space helpers (sRGB ↔ linear)
# ---------------------------------------------------------------------------
def _linear_to_srgb(x: Tensor) -> Tensor:
    """Linear → sRGB gamma. Input >= 0, output in [0, ~1]."""
    x = x.clamp(min=0.0)
    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * x.pow(1.0 / 2.4) - 0.055,
    )


def _srgb_to_linear(x: Tensor) -> Tensor:
    """sRGB → linear. Input in [0, 1], output >= 0."""
    x = x.clamp(0.0, 1.0)
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055).pow(2.4),
    )
