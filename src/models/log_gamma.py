"""log_gamma.py — Log-Gamma color mapping for DiffHDR (arXiv 2604.06161, §3.2).

Maps linear HDR radiance into a bounded range compatible with a pretrained
LDR video VAE, without finetuning the VAE itself.

    T(x) = ( log(1 + γ·x) / log(1 + γ·M) )^(1/γ)

where:
    x  – linear HDR radiance (>= 0)
    M  – maximum representable radiance (ceiling)
    γ  – gamma compression exponent

The logarithm compresses high dynamic range; the outer 1/γ power aligns the
distribution with natural LDR statistics so the pretrained VAE encodes/decodes
faithfully.  The inverse map recovers linear HDR radiance:

    T⁻¹(y) = ( exp( y^γ · log(1 + γ·M) ) - 1 ) / γ

All ops are differentiable and run on GPU.  Inputs may be any float dtype;
computation is done in float32 for numerical stability then cast back.
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["log_gamma", "log_gamma_inverse", "LogGamma"]


def log_gamma(x: Tensor, gamma: float = 2.2, M: float = 100.0, eps: float = 1e-8) -> Tensor:
    """Forward Log-Gamma mapping: linear HDR → VAE-compatible [0,1]-ish range.

    Args:
        x:     linear HDR radiance, any shape, values >= 0.
        gamma: gamma compression exponent (paper default ~2.2).
        M:     maximum representable radiance ceiling.
        eps:   small constant to avoid log(0) / div-by-zero.

    Returns:
        Mapped tensor in ~[0, 1], same dtype/device as input.
    """
    if x.dtype not in (torch.float32, torch.float64):
        x = x.float()
    x = x.clamp(min=0.0)
    num = torch.log1p(gamma * x)
    den = torch.log1p(torch.tensor(gamma * M, dtype=x.dtype, device=x.device))
    # inner ratio: 1.0 at x=M, >1 for x>M (VAE tolerates mild overshoot)
    inner = (num / den.clamp(min=eps)).clamp(min=0.0)
    y = inner.pow(1.0 / gamma)
    return y


def log_gamma_inverse(y: Tensor, gamma: float = 2.2, M: float = 100.0, eps: float = 1e-8) -> Tensor:
    """Inverse Log-Gamma mapping: VAE-decoded value → linear HDR radiance.

    Args:
        y:     mapped value in ~[0, 1].
        gamma: gamma compression exponent (must match forward).
        M:     maximum representable radiance ceiling (must match forward).
        eps:   small constant for numerical stability.

    Returns:
        Linear HDR radiance >= 0.
    """
    if y.dtype not in (torch.float32, torch.float64):
        y = y.float()
    y = y.clamp(min=0.0)
    inner = y.pow(gamma)
    den = torch.log1p(torch.tensor(gamma * M, dtype=y.dtype, device=y.device))
    x = (torch.exp(inner * den) - 1.0) / gamma
    return x.clamp(min=0.0)


class LogGamma(torch.nn.Module):
    """Stateless module wrapper so it can be moved to device / saved with the model."""

    def __init__(self, gamma: float = 2.2, M: float = 100.0):
        super().__init__()
        self.gamma = float(gamma)
        self.M = float(M)

    def forward(self, x: Tensor) -> Tensor:
        return log_gamma(x, self.gamma, self.M)

    def inverse(self, y: Tensor) -> Tensor:
        return log_gamma_inverse(y, self.gamma, self.M)
