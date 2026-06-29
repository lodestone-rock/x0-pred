"""otf_sr_dataloader.py — On-the-fly degradation SR dataset.

Fully standalone: all degradation utilities (DiffJPEG, filter2D, blur kernels,
gaussian/poisson noise) are inlined here.  Zero neosr imports.

The dataset:
  - Scans a folder of HR images (jpg / png / webp / tiff).
  - Returns HR crop + two-pass blur/resize/noise/JPEG blur kernels per item.
  - Actual GPU-side degradation is applied in the trainer (like neosr's
    feed_data), so the dataloader only generates the CPU-side kernels.

__getitem__ returns a dict:
    hr           : (3, crop_size, crop_size) float32 [0, 1] RGB
    kernel1      : (21, 21) float32  — first-pass blur kernel
    kernel2      : (21, 21) float32  — second-pass blur kernel
    sinc_kernel  : (21, 21) float32  — final sinc (or pulse) kernel
    path         : str

Style follows parquet_dataloader.py: class with resample() and dummy_collate_fn.
"""
from __future__ import annotations

import itertools
import math
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Image extensions to scan
# ---------------------------------------------------------------------------
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}


def _scan_images(folder: str) -> list[str]:
    paths = []
    for root, _dirs, files in os.walk(folder):
        for f in sorted(files):
            if Path(f).suffix.lower() in _IMG_EXTS:
                paths.append(os.path.join(root, f))
    if not paths:
        raise FileNotFoundError(f"No images found under: {folder}")
    return paths


# ===========================================================================
# Inlined DiffJPEG
# Modified from https://github.com/mlomnitz/DiffJPEG
# Key change: removed hardcoded device = torch.device("cuda") globals.
# Tables are plain float32 numpy arrays; DiffJPEG moves them via .to(device).
# ===========================================================================

y_table_np = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
).T

c_table_np = np.empty((8, 8), dtype=np.float32)
c_table_np.fill(99)
c_table_np[:4, :4] = np.array(
    [[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]],
    dtype=np.float32,
).T


def _diff_round(x):
    return torch.round(x) + (x - torch.round(x)) ** 3


def _quality_to_factor(quality):
    quality = 5000.0 / quality if quality < 50 else 200.0 - quality * 2
    return quality / 100.0


class _RGB2YCbCr(nn.Module):
    def __init__(self):
        super().__init__()
        mat = np.array(
            [[0.299, 0.587, 0.114],
             [-0.168736, -0.331264, 0.5],
             [0.5, -0.418688, -0.081312]], dtype=np.float32).T
        self.register_buffer("shift", torch.tensor([0.0, 128.0, 128.0]))
        self.register_buffer("matrix", torch.from_numpy(mat))

    def forward(self, image):
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
        return result.view(image.shape)


class _ChromaSub(nn.Module):
    def forward(self, image):
        img2 = image.permute(0, 3, 1, 2).clone()
        cb = F.avg_pool2d(img2[:, 1:2], 2, 2, count_include_pad=False).permute(0, 2, 3, 1).squeeze(3)
        cr = F.avg_pool2d(img2[:, 2:3], 2, 2, count_include_pad=False).permute(0, 2, 3, 1).squeeze(3)
        return image[:, :, :, 0], cb, cr


class _BlockSplit(nn.Module):
    def forward(self, image):
        k = 8
        h, _ = image.shape[1:3]
        B = image.shape[0]
        r = image.view(B, h // k, k, -1, k)
        return r.permute(0, 1, 3, 2, 4).contiguous().view(B, -1, k, k)


class _DCT8x8(nn.Module):
    def __init__(self):
        super().__init__()
        t = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            t[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                (2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1.0 / np.sqrt(2)] + [1] * 7)
        self.register_buffer("tensor", torch.from_numpy(t).float())
        self.register_buffer("scale", torch.from_numpy(np.outer(alpha, alpha) * 0.25).float())

    def forward(self, image):
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


class _YQ(nn.Module):
    def __init__(self, rounding):
        super().__init__()
        self.rounding = rounding
        self.register_buffer("y_table", torch.from_numpy(y_table_np))

    def forward(self, image, factor=1):
        if isinstance(factor, (int, float)):
            image = image.float() / (self.y_table * factor)
        else:
            b = factor.size(0)
            table = self.y_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            image = image.float() / table
        return self.rounding(image)


class _CQ(nn.Module):
    def __init__(self, rounding):
        super().__init__()
        self.rounding = rounding
        self.register_buffer("c_table", torch.from_numpy(c_table_np))

    def forward(self, image, factor=1):
        if isinstance(factor, (int, float)):
            image = image.float() / (self.c_table * factor)
        else:
            b = factor.size(0)
            table = self.c_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            image = image.float() / table
        return self.rounding(image)


class _CompressJpeg(nn.Module):
    def __init__(self, rounding=torch.round):
        super().__init__()
        self.l1 = nn.Sequential(_RGB2YCbCr(), _ChromaSub())
        self.l2 = nn.Sequential(_BlockSplit(), _DCT8x8())
        self.c_q = _CQ(rounding)
        self.y_q = _YQ(rounding)

    def forward(self, image, factor=1):
        y, cb, cr = self.l1(image * 255)
        comp = {}
        for k, ch in zip(("y", "cb", "cr"), (y, cb, cr)):
            c = self.l2(ch)
            comp[k] = self.c_q(c, factor) if k in ("cb", "cr") else self.y_q(c, factor)
        return comp["y"], comp["cb"], comp["cr"]


class _YDQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("y_table", torch.from_numpy(y_table_np))

    def forward(self, image, factor=1):
        if isinstance(factor, (int, float)):
            return image * (self.y_table * factor)
        b = factor.size(0)
        return image * (self.y_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1))


class _CDQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("c_table", torch.from_numpy(c_table_np))

    def forward(self, image, factor=1):
        if isinstance(factor, (int, float)):
            return image * (self.c_table * factor)
        b = factor.size(0)
        return image * (self.c_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1))


class _iDCT8x8(nn.Module):
    def __init__(self):
        super().__init__()
        alpha = np.array([1.0 / np.sqrt(2)] + [1] * 7)
        self.register_buffer("alpha", torch.from_numpy(np.outer(alpha, alpha)).float())
        t = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            t[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
                (2 * v + 1) * y * np.pi / 16)
        self.register_buffer("tensor", torch.from_numpy(t).float())

    def forward(self, image):
        image = image * self.alpha
        result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128
        result.view(image.shape)
        return result


class _BlockMerge(nn.Module):
    def forward(self, patches, height, width):
        k = 8
        B = patches.shape[0]
        r = patches.view(B, height // k, width // k, k, k)
        return r.permute(0, 1, 3, 2, 4).contiguous().view(B, height, width)


class _ChromaUp(nn.Module):
    def forward(self, y, cb, cr):
        def repeat(x, k=2):
            h, w = x.shape[1:3]
            x = x.unsqueeze(-1).repeat(1, 1, k, k)
            return x.view(-1, h * k, w * k)
        cb, cr = repeat(cb), repeat(cr)
        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


class _YCbCr2RGB(nn.Module):
    def __init__(self):
        super().__init__()
        mat = np.array(
            [[1.0, 0.0, 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
            dtype=np.float32).T
        self.register_buffer("shift", torch.tensor([0.0, -128.0, -128.0]))
        self.register_buffer("matrix", torch.from_numpy(mat))

    def forward(self, image):
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        return result.view(image.shape).permute(0, 3, 1, 2)


class _DecompressJpeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_dq = _CDQ()
        self.y_dq = _YDQ()
        self.idct = _iDCT8x8()
        self.merge = _BlockMerge()
        self.chroma = _ChromaUp()
        self.colors = _YCbCr2RGB()

    def forward(self, y, cb, cr, imgh, imgw, factor=1):
        comp = {}
        for k, ch in zip(("y", "cb", "cr"), (y, cb, cr)):
            if k in ("cb", "cr"):
                c = self.c_dq(ch, factor)
                h_, w_ = imgh // 2, imgw // 2
            else:
                c = self.y_dq(ch, factor)
                h_, w_ = imgh, imgw
            c = self.idct(c)
            comp[k] = self.merge(c, h_, w_)
        image = self.chroma(comp["y"], comp["cb"], comp["cr"])
        image = self.colors(image)
        return image.clamp(0, 255) / 255.0


class DiffJPEG(nn.Module):
    """Differentiable JPEG (standalone, device-agnostic).
    
    Instantiate once; call .to(device) to move all buffers.
    """

    def __init__(self, differentiable: bool = False) -> None:
        super().__init__()
        rounding = _diff_round if differentiable else torch.round
        self.compress = _CompressJpeg(rounding)
        self.decompress = _DecompressJpeg()

    def forward(self, x: Tensor, quality) -> Tensor:
        """x: (B, 3, H, W) in [0, 1]; quality: float or (B,) tensor."""
        factor = quality
        if isinstance(factor, (int, float)):
            factor = _quality_to_factor(factor)
        else:
            factor = torch.stack([
                torch.tensor(_quality_to_factor(factor[i].item()),
                             dtype=x.dtype, device=x.device)
                for i in range(factor.size(0))
            ])
        h, w = x.shape[-2:]
        h_pad = (16 - h % 16) % 16
        w_pad = (16 - w % 16) % 16
        x = F.pad(x, (0, w_pad, 0, h_pad), mode="constant", value=0)
        y, cb, cr = self.compress(x, factor=factor)
        rec = self.decompress(y, cb, cr, h + h_pad, w + w_pad, factor=factor)
        return rec[:, :, :h, :w]


# ===========================================================================
# Inlined filter2D (from neosr/utils/diffjpeg.py)
# ===========================================================================

def filter2D(img: Tensor, kernel: Tensor) -> Tensor:
    """Apply a per-image 2-D kernel. img: (B,C,H,W), kernel: (B,k,k) or (1,k,k)."""
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 != 1:
        raise ValueError("Kernel size must be odd.")
    img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    ph, pw = img.shape[-2:]
    if kernel.size(0) == 1:
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    img = img.view(1, b * c, ph, pw)
    kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
    return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


# ===========================================================================
# Inlined noise functions (from neosr/data/degradations.py)
# ===========================================================================

def _generate_gaussian_noise_pt(img: Tensor, sigma=10, gray_noise=0) -> Tensor:
    b, _, h, w = img.size()
    if not isinstance(sigma, (float, int)):
        sigma = sigma.view(b, 1, 1, 1)
    if isinstance(gray_noise, (float, int)):
        cal_gray = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray = torch.sum(gray_noise) > 0
    if cal_gray:
        ng = torch.randn(b, 1, h, w, dtype=img.dtype, device=img.device) * sigma / 255.0
        ng = ng.expand(b, img.size(1), h, w)
    noise = torch.randn(*img.size(), dtype=img.dtype, device=img.device) * sigma / 255.0
    if cal_gray:
        noise = noise * (1 - gray_noise) + ng * gray_noise
    return noise


def random_add_gaussian_noise_pt(
    img: Tensor,
    sigma_range=(0, 1.0),
    gray_prob: float = 0,
    clip: bool = True,
    rounds: bool = False,
) -> Tensor:
    sigma = (
        torch.rand(img.size(0), dtype=img.dtype, device=img.device)
        * (sigma_range[1] - sigma_range[0])
        + sigma_range[0]
    )
    gray_noise = (torch.rand(img.size(0), dtype=img.dtype, device=img.device) < gray_prob).float()
    noise = _generate_gaussian_noise_pt(img, sigma, gray_noise)
    out = img + noise
    if clip and rounds:
        return torch.clamp((out * 255.0).round(), 0, 255) / 255.0
    elif clip:
        return torch.clamp(out, 0, 1)
    elif rounds:
        return (out * 255.0).round() / 255.0
    return out


def _generate_poisson_noise_pt(img: Tensor, scale=1.0, gray_noise=0) -> Tensor:
    b, _, h, w = img.size()
    if isinstance(gray_noise, (float, int)):
        cal_gray = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray = torch.sum(gray_noise) > 0
    if cal_gray:
        from torchvision.transforms.functional import rgb_to_grayscale
        img_gray = rgb_to_grayscale(img, num_output_channels=1)
        img_gray = torch.clamp((img_gray * 255.0).round(), 0, 255) / 255.0
        vals_list = [2 ** np.ceil(np.log2(max(len(torch.unique(img_gray[i])), 2))) for i in range(b)]
        vals = img_gray.new_tensor(vals_list).view(b, 1, 1, 1)
        out = torch.poisson(img_gray * vals) / vals
        noise_gray = (out - img_gray).expand(b, img.size(1), h, w)
    img_c = torch.clamp((img * 255.0).round(), 0, 255) / 255.0
    vals_list = [2 ** np.ceil(np.log2(max(len(torch.unique(img_c[i])), 2))) for i in range(b)]
    vals = img_c.new_tensor(vals_list).view(b, 1, 1, 1)
    out = torch.poisson(img_c * vals) / vals
    noise = out - img_c
    if cal_gray:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    if not isinstance(scale, (float, int)):
        scale = scale.view(b, 1, 1, 1)
    return noise * scale


def random_add_poisson_noise_pt(
    img: Tensor,
    scale_range=(0, 1.0),
    gray_prob: float = 0,
    clip: bool = True,
    rounds: bool = False,
) -> Tensor:
    scale = (
        torch.rand(img.size(0), dtype=img.dtype, device=img.device)
        * (scale_range[1] - scale_range[0])
        + scale_range[0]
    )
    gray_noise = (torch.rand(img.size(0), dtype=img.dtype, device=img.device) < gray_prob).float()
    noise = _generate_poisson_noise_pt(img, scale, gray_noise)
    out = img + noise
    if clip and rounds:
        return torch.clamp((out * 255.0).round(), 0, 255) / 255.0
    elif clip:
        return torch.clamp(out, 0, 1)
    elif rounds:
        return (out * 255.0).round() / 255.0
    return out


# ===========================================================================
# Inlined blur kernel generation (from neosr/data/degradations.py)
# ===========================================================================

def _sigma_matrix2(sig_x, sig_y, theta):
    d = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    u = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return np.dot(u, np.dot(d, u.T))


def _mesh_grid(kernel_size):
    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1))).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy


def _pdf2(sigma_matrix, grid):
    inv_sigma = np.linalg.inv(sigma_matrix)
    return np.exp(-0.5 * np.sum(np.dot(grid, inv_sigma) * grid, 2))


def _bivariate_gaussian(kernel_size, sig_x, sig_y, theta, isotropic=True):
    grid, _, _ = _mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]]) if isotropic \
        else _sigma_matrix2(sig_x, sig_y, theta)
    kernel = _pdf2(sigma_matrix, grid)
    return kernel / np.sum(kernel)


def _bivariate_generalized_gaussian(kernel_size, sig_x, sig_y, theta, beta, isotropic=True):
    grid, _, _ = _mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]]) if isotropic \
        else _sigma_matrix2(sig_x, sig_y, theta)
    inv_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inv_sigma) * grid, 2), beta))
    return kernel / np.sum(kernel)


def _bivariate_plateau(kernel_size, sig_x, sig_y, theta, beta, isotropic=True):
    grid, _, _ = _mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]]) if isotropic \
        else _sigma_matrix2(sig_x, sig_y, theta)
    inv_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.reciprocal(np.power(np.sum(np.dot(grid, inv_sigma) * grid, 2), beta) + 1)
    return kernel / np.sum(kernel)


def circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
    """2-D sinc filter."""
    from scipy import special as sp_special
    assert kernel_size % 2 == 1
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = np.fromfunction(
            lambda x, y: cutoff * sp_special.j1(
                cutoff * np.sqrt((x - (kernel_size - 1) / 2) ** 2 +
                                 (y - (kernel_size - 1) / 2) ** 2)
            ) / (2 * np.pi * np.sqrt((x - (kernel_size - 1) / 2) ** 2 +
                                      (y - (kernel_size - 1) / 2) ** 2)),
            [kernel_size, kernel_size],
        )
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff ** 2 / (4 * np.pi)
    kernel /= np.sum(kernel)
    if pad_to > kernel_size:
        pad = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad, pad), (pad, pad)))
    return kernel


def random_mixed_kernels(
    kernel_list,
    kernel_prob,
    kernel_size=21,
    sigma_x_range=(0.6, 5),
    sigma_y_range=(0.6, 5),
    rotation_range=(-math.pi, math.pi),
    betag_range=(0.5, 8),
    betap_range=(0.5, 8),
    noise_range=None,
    _rng: random.Random | None = None,
):
    """Generate a random mixed blur kernel."""
    rng_ = _rng or random
    kernel_type = rng_.choices(kernel_list, kernel_prob)[0]
    sigma_x = rng_.uniform(sigma_x_range[0], sigma_x_range[1])

    def _noisy(k):
        if noise_range is not None:
            noise = np.random.uniform(noise_range[0], noise_range[1], size=k.shape)
            k = k * noise
        return k / np.sum(k)

    if kernel_type == "iso":
        k = _bivariate_gaussian(kernel_size, sigma_x, sigma_x, 0, isotropic=True)
    elif kernel_type == "aniso":
        sigma_y = rng_.uniform(sigma_y_range[0], sigma_y_range[1])
        rot = rng_.uniform(rotation_range[0], rotation_range[1])
        k = _bivariate_gaussian(kernel_size, sigma_x, sigma_y, rot, isotropic=False)
    elif kernel_type == "generalized_iso":
        beta = rng_.uniform(betag_range[0], 1) if rng_.random() < 0.5 else rng_.uniform(1, betag_range[1])
        k = _bivariate_generalized_gaussian(kernel_size, sigma_x, sigma_x, 0, beta, isotropic=True)
    elif kernel_type == "generalized_aniso":
        sigma_y = rng_.uniform(sigma_y_range[0], sigma_y_range[1])
        rot = rng_.uniform(rotation_range[0], rotation_range[1])
        beta = rng_.uniform(betag_range[0], 1) if rng_.random() < 0.5 else rng_.uniform(1, betag_range[1])
        k = _bivariate_generalized_gaussian(kernel_size, sigma_x, sigma_y, rot, beta, isotropic=False)
    elif kernel_type == "plateau_iso":
        beta = rng_.uniform(betap_range[0], 1) if rng_.random() < 0.5 else rng_.uniform(1, betap_range[1])
        k = _bivariate_plateau(kernel_size, sigma_x, sigma_x, 0, beta, isotropic=True)
    elif kernel_type == "plateau_aniso":
        sigma_y = rng_.uniform(sigma_y_range[0], sigma_y_range[1])
        rot = rng_.uniform(rotation_range[0], rotation_range[1])
        beta = rng_.uniform(betap_range[0], 1) if rng_.random() < 0.5 else rng_.uniform(1, betap_range[1])
        k = _bivariate_plateau(kernel_size, sigma_x, sigma_y, rot, beta, isotropic=False)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    return _noisy(k)


# ===========================================================================
# OTF degradation (GPU-side) — apply_degradation()
# Call this in the trainer after moving kernel tensors to CUDA.
# ===========================================================================

def apply_degradation(
    gt: Tensor,
    kernel1: Tensor,
    kernel2: Tensor,
    sinc_kernel: Tensor,
    upscale: int,
    jpeger: DiffJPEG,
    cfg: dict,
    rng_: random.Random | None = None,
) -> Tensor:
    """Apply two-pass Real-ESRGAN-style degradation to produce an LR image.

    Args:
        gt          : (B, 3, H, W) HR image on GPU, [0, 1]
        kernel1     : (B, 21, 21) first blur kernel on GPU
        kernel2     : (B, 21, 21) second blur kernel on GPU
        sinc_kernel : (B, 21, 21) final sinc / pulse kernel on GPU
        upscale     : integer scale factor (2 or 4)
        jpeger      : DiffJPEG instance on the same device as gt
        cfg         : degradation sub-dict from config
        rng_        : optional seeded random.Random; uses module-level random if None

    Returns:
        lr : (B, 3, H//upscale, W//upscale) on GPU, [0, 1]
    """
    rng_ = rng_ or random
    ori_h, ori_w = gt.shape[2], gt.shape[3]
    device = gt.device

    # ---- Pass 1 ----
    out = filter2D(gt, kernel1)

    updown = rng_.choices(["up", "down", "keep"],
                          cfg.get("resize_prob", [0.2, 0.7, 0.1]))[0]
    rr = cfg.get("resize_range", [0.15, 1.5])
    scale = (rng_.uniform(1, rr[1]) if updown == "up"
             else rng_.uniform(rr[0], 1) if updown == "down" else 1)
    mode = rng_.choice(["area", "bilinear", "bicubic"])
    out = F.interpolate(out, scale_factor=scale, mode=mode)

    gnp = cfg.get("gaussian_noise_prob", 0.5)
    gp = cfg.get("gray_noise_prob", 0.4)
    if rng_.random() < gnp:
        out = random_add_gaussian_noise_pt(out, cfg.get("noise_range", [1, 30]),
                                           gray_prob=gp, clip=True, rounds=False)
    else:
        out = random_add_poisson_noise_pt(out, cfg.get("poisson_scale_range", [0.05, 3.0]),
                                          gray_prob=gp, clip=True, rounds=False)

    jpeg_p = out.new_zeros(out.size(0)).uniform_(*cfg.get("jpeg_range", [30, 95]))
    out = torch.clamp(out, 0, 1)
    out = jpeger(out, quality=jpeg_p)

    # ---- Pass 2 ----
    if rng_.random() < cfg.get("second_blur_prob", 0.8):
        k2_size = kernel2.shape[-1]
        if out.shape[-2] > k2_size and out.shape[-1] > k2_size:
            out = filter2D(out, kernel2)

    updown = rng_.choices(["up", "down", "keep"],
                          cfg.get("resize_prob2", [0.3, 0.4, 0.3]))[0]
    rr2 = cfg.get("resize_range2", [0.3, 1.2])
    scale2 = (rng_.uniform(1, rr2[1]) if updown == "up"
              else rng_.uniform(rr2[0], 1) if updown == "down" else 1)
    mode = rng_.choice(["area", "bilinear", "bicubic"])
    target_h = int(ori_h / upscale * scale2)
    target_w = int(ori_w / upscale * scale2)
    out = F.interpolate(out, size=(max(target_h, 1), max(target_w, 1)), mode=mode)

    gnp2 = cfg.get("gaussian_noise_prob2", 0.5)
    gp2 = cfg.get("gray_noise_prob2", 0.4)
    if rng_.random() < gnp2:
        out = random_add_gaussian_noise_pt(out, cfg.get("noise_range2", [1, 25]),
                                           gray_prob=gp2, clip=True, rounds=False)
    else:
        out = random_add_poisson_noise_pt(out, cfg.get("poisson_scale_range2", [0.05, 2.5]),
                                          gray_prob=gp2, clip=True, rounds=False)

    jpeg_p2 = out.new_zeros(out.size(0)).uniform_(*cfg.get("jpeg_range2", [30, 95]))
    out = torch.clamp(out, 0, 1)
    out = jpeger(out, quality=jpeg_p2)

    # ---- Final sinc ----
    sk_size = sinc_kernel.shape[-1]
    def _safe_sinc(x: Tensor) -> Tensor:
        if x.shape[-2] > sk_size and x.shape[-1] > sk_size:
            return filter2D(x, sinc_kernel)
        return x

    if rng_.random() < 0.5:
        out = _safe_sinc(out)
        out = F.interpolate(out, size=(ori_h // upscale, ori_w // upscale),
                            mode="bicubic", antialias=True)
    else:
        out = F.interpolate(out, size=(ori_h // upscale, ori_w // upscale),
                            mode="bicubic", antialias=True)
        out = _safe_sinc(out)

    return torch.clamp(out, 0, 1)


# ===========================================================================
# Dataset
# ===========================================================================

class OTFSRDataset(Dataset):
    """On-the-fly degradation SR dataset backed by a local image folder.

    Config keys (all under ``cfg``):
        hr_folder     : str — path to folder of HR images
        crop_size     : int — HR crop size (default 256)
        upscale       : int — 2 or 4 (default 2)
        use_hflip     : bool (default True)
        use_rot       : bool (default True)
        seed          : int (default 0)
        degradation   : dict — OTF degradation parameters (see apply_degradation)

    ``__getitem__`` returns a dict with keys:
        hr           : (3, crop_size, crop_size) float32 [0, 1] RGB
        kernel1      : (21, 21) float32
        kernel2      : (21, 21) float32
        sinc_kernel  : (21, 21) float32
        path         : str
    """

    # Kernel sizes from [7, 9, 11, ... 21]
    _KERNEL_RANGE = [2 * v + 1 for v in range(3, 11)]

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.hr_folder = cfg["hr_folder"]
        self.crop_size = int(cfg.get("crop_size", 256))
        self.upscale = int(cfg.get("upscale", 2))
        self.use_hflip = bool(cfg.get("use_hflip", True))
        self.use_rot = bool(cfg.get("use_rot", True))
        self.deg_cfg = cfg.get("degradation", {})
        self._rng = random.Random(int(cfg.get("seed", 0)))

        self.paths = _scan_images(self.hr_folder)
        self._rng.shuffle(self.paths)

        # Precomputed pulse tensor (identity kernel for sinc fallback)
        self._pulse = torch.zeros(21, 21, dtype=torch.float32)
        self._pulse[10, 10] = 1.0

    def resample(self) -> None:
        """Shuffle path order (call after each epoch)."""
        self._rng.shuffle(self.paths)

    @staticmethod
    def dummy_collate_fn(batch):
        return batch

    def __len__(self) -> int:
        return len(self.paths)

    def _load_image(self, path: str) -> np.ndarray:
        """Load image as float32 (H, W, 3) RGB in [0, 1]."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"Failed to load: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if self.use_hflip and self._rng.random() > 0.5:
            img = img[:, ::-1, :]
        if self.use_rot:
            k = self._rng.randint(0, 3)
            img = np.rot90(img, k)
        return np.ascontiguousarray(img)

    def _random_crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        cs = self.crop_size
        if h < cs or w < cs:
            # Pad then crop
            pad_h = max(0, cs - h)
            pad_w = max(0, cs - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                     cv2.BORDER_REFLECT_101)
            h, w = img.shape[:2]
        top = self._rng.randint(0, h - cs)
        left = self._rng.randint(0, w - cs)
        return img[top: top + cs, left: left + cs]

    def _make_kernel(self, cfg: dict) -> np.ndarray:
        kernel_size = self._rng.choice(self._KERNEL_RANGE)
        sinc_prob = cfg.get("sinc_prob", 0.1)
        blur_sigma = cfg.get("blur_sigma", [0.2, 3.0])
        betag = cfg.get("betag_range", [0.5, 4.0])
        betap = cfg.get("betap_range", [1.0, 2.0])
        klist = cfg.get("kernel_list",
                        ["iso", "aniso", "generalized_iso", "generalized_aniso",
                         "plateau_iso", "plateau_aniso"])
        kprob = cfg.get("kernel_prob",
                        [0.45, 0.25, 0.12, 0.03, 0.12, 0.03])
        if self._rng.random() < sinc_prob:
            omega = (self._rng.uniform(math.pi / 3, math.pi) if kernel_size < 13
                     else self._rng.uniform(math.pi / 5, math.pi))
            kernel = circular_lowpass_kernel(omega, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                klist, kprob, kernel_size,
                sigma_x_range=blur_sigma, sigma_y_range=blur_sigma,
                betag_range=betag, betap_range=betap,
                _rng=self._rng,
            )
        pad = (21 - kernel_size) // 2
        return np.pad(kernel, ((pad, pad), (pad, pad)))

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index % len(self.paths)]

        # Retry on load failure
        for attempt in range(5):
            try:
                img = self._load_image(path)
                break
            except OSError:
                index = self._rng.randint(0, len(self.paths) - 1)
                path = self.paths[index]
        else:
            raise RuntimeError(f"Failed to load image after retries: {path}")

        img = self._random_crop(img)
        img = self._augment(img)

        # HWC → CHW tensor
        hr = torch.from_numpy(img.transpose(2, 0, 1)).float()  # (3, H, W)

        # Blur kernels
        deg = self.deg_cfg
        k1 = self._make_kernel(deg)
        k2_cfg = {k.rstrip("2"): v for k, v in deg.items() if k.endswith("2") and k != "sinc_prob2"}
        k2_cfg.update({k: v for k, v in deg.items() if k.endswith("2")})
        # For second kernel, use the "2" suffixed keys if present, else same as first
        sinc_prob2 = deg.get("sinc_prob2", 0.1)
        blur_sigma2 = deg.get("blur_sigma2", deg.get("blur_sigma", [0.2, 3.0]))
        betag2 = deg.get("betag_range2", deg.get("betag_range", [0.5, 4.0]))
        betap2 = deg.get("betap_range2", deg.get("betap_range", [1.0, 2.0]))
        klist2 = deg.get("kernel_list2", deg.get("kernel_list",
                         ["iso", "aniso", "generalized_iso", "generalized_aniso",
                          "plateau_iso", "plateau_aniso"]))
        kprob2 = deg.get("kernel_prob2", deg.get("kernel_prob",
                         [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]))
        k2_kernel_size = self._rng.choice(self._KERNEL_RANGE)
        if self._rng.random() < sinc_prob2:
            omega = (self._rng.uniform(math.pi / 3, math.pi) if k2_kernel_size < 13
                     else self._rng.uniform(math.pi / 5, math.pi))
            k2 = circular_lowpass_kernel(omega, k2_kernel_size, pad_to=False)
        else:
            k2 = random_mixed_kernels(klist2, kprob2, k2_kernel_size,
                                      sigma_x_range=blur_sigma2,
                                      sigma_y_range=blur_sigma2,
                                      betag_range=betag2, betap_range=betap2,
                                      _rng=self._rng)
        pad2 = (21 - k2_kernel_size) // 2
        k2 = np.pad(k2, ((pad2, pad2), (pad2, pad2)))

        # Final sinc kernel
        final_sinc_prob = deg.get("final_sinc_prob", 0.8)
        if self._rng.random() < final_sinc_prob:
            ks = self._rng.choice(self._KERNEL_RANGE)
            omega = self._rng.uniform(math.pi / 3, math.pi)
            sinc = circular_lowpass_kernel(omega, ks, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc)
        else:
            sinc_kernel = self._pulse.clone()

        return {
            "hr": hr,
            "kernel1": torch.FloatTensor(k1),
            "kernel2": torch.FloatTensor(k2),
            "sinc_kernel": sinc_kernel,
            "path": path,
        }
