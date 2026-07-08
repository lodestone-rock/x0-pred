"""wan_vae.py — Wan-2.1 Video VAE (AutoencoderKLWan) encode/decode helpers.

Spatiotemporal compression: 4 (temporal) × 8 (spatial), 16 latent channels.
Input video format: [B, C, T, H, W] float in [-1, 1].
Latent format:      [B, 16, T_lat, H/8, W/8] where T_lat = (T-1)//4 + 1.

Per the DiffHDR paper (§4.1, Appendix C): the VAE runs in FP32 to avoid
banding artifacts in smooth HDR gradients; the DiT runs in BF16.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["WanVAE"]


class WanVAE(nn.Module):
    """Thin wrapper around diffusers AutoencoderKLWan for training.

    The VAE is frozen (requires_grad=False) and kept in FP32 per the paper.
    """

    def __init__(self, model_id: str = "Wan-AI/Wan2.1-VACE-14B", subfolder: str = "vae"):
        super().__init__()
        from diffusers import AutoencoderKLWan

        self.ae = AutoencoderKLWan.from_pretrained(model_id, subfolder=subfolder)
        self.ae.eval().requires_grad_(False)

        # Spatiotemporal compression ratios.
        self.spatial_compression = 8
        self.temporal_compression = 4
        self.channels = self.ae.config.z_dim  # 16

        # Latent normalization buffers (from VAE config).
        latents_mean = self.ae.config.latents_mean  # list of 16 floats
        latents_std = self.ae.config.latents_std
        self.register_buffer(
            "latents_mean",
            torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1),
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1),
        )

    @property
    def device(self):
        return next(self.ae.parameters()).device

    @property
    def dtype(self):
        return next(self.ae.parameters()).dtype

    @torch.no_grad()
    def encode(self, video: Tensor) -> Tensor:
        """Encode video [B,C,T,H,W] in [-1,1] → normalized latent [B,16,T_lat,H/8,W/8].

        Runs in FP32 for tonal continuity (paper §4.1).
        """
        vae_dtype = self.dtype
        x = video.to(self.device, dtype=torch.float32)
        # VAE weights in FP32; input also FP32.
        with torch.autocast("cuda", torch.float32, enabled=False):
            posterior = self.ae.encode(x.to(vae_dtype))
            latent = posterior.latent_dist.mode()  # [B, 16, T_lat, H/8, W/8]
            # Normalize: (x - mean) * (1/std)  — matches diffusers pipeline convention
            latent = ((latent.float() - self.latents_mean) * self.latents_std).to(vae_dtype)
        return latent

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        """Decode latent [B,16,T_lat,H/8,W/8] → video [B,C,T,H,W] in [-1,1].

        Runs in FP32 to avoid banding (paper Appendix C).
        """
        vae_dtype = self.dtype
        z = latent.to(self.device, dtype=torch.float32)
        # Denormalize: x = latent/std + mean
        z = (z.float() / self.latents_std) + self.latents_mean
        with torch.autocast("cuda", torch.float32, enabled=False):
            decoded = self.ae.decode(z.to(vae_dtype)).sample  # [B, C, T, H, W]
        return decoded.float()

    @torch.no_grad()
    def encode_video_to_neg1(self, video: Tensor) -> Tensor:
        """Encode a video already in [-1,1] and return latent (for context/conditioning).

        Alias for :meth:`encode` — kept for API clarity in the trainer.
        """
        return self.encode(video)
