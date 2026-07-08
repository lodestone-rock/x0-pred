"""wan_vace.py — Wan-2.1-VACE-14B backbone wrapper for DiffHDR training.

Loads the diffusers WanVACETransformer3DModel + UMT5 text encoder, injects LoRA
into the DiT, and exposes a clean forward for flow-matching training.

VCU conditioning layout (96 channels, matching the pretrained vace_patch_embedding):
    control_hidden_states = cat([inactive_latent(16),
                                 reactive_latent(16),
                                 mask_expanded(64)], dim=1)
where:
    inactive_latent = VAE.encode(LDR * (1 - mask))   # unmasked LDR context
    reactive_latent = VAE.encode(LDR * mask)          # masked LDR regions
    mask_expanded   = binary mask replicated to 64ch  # 4 temporal groups × 16

For DiffHDR the "video" is the LDR input (Log-Gamma mapped), and the mask is the
luminance-based over/underexposure mask.  The model denoises to produce the HDR
latent (Log-Gamma mapped), which is decoded and inverse-mapped to linear HDR.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .wan_lora import (
    LoRALinear,
    inject_lora,
    lora_state_dict,
    load_lora_state_dict,
    trainable_param_count,
)

__all__ = ["WanVACEBackbone"]


class WanVACEBackbone(nn.Module):
    """Wraps the Wan VACE transformer + text encoder for training.

    Args:
        model_id: HF repo id with diffusers-format weights
                  (default ``linoyts/Wan-VACE-14B-diffusers``).
        lora_rank: LoRA rank (paper uses 32).
        lora_alpha: LoRA alpha (defaults to rank).
        lora_exclude_prefixes: parameter name prefixes to skip for LoRA.
    """

    VACE_IN_CHANNELS = 96  # 16 (inactive) + 16 (reactive) + 64 (mask)
    LATENT_CHANNELS = 16

    def __init__(
        self,
        model_id: str = "linoyts/Wan-VACE-14B-diffusers",
        lora_rank: int = 32,
        lora_alpha: float | None = None,
        lora_exclude_prefixes: tuple[str, ...] = ("patch_embedding", "vace_patch_embedding", "condition_embedder", "proj_out", "norm_out", "scale_shift_table"),
    ):
        super().__init__()
        from diffusers import WanVACETransformer3DModel

        self.model_id = model_id
        self.transformer = WanVACETransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", low_cpu_mem_usage=True,
        )

        # Inject LoRA into all nn.Linear in DiT blocks + VACE blocks.
        self.replaced = inject_lora(
            self.transformer,
            rank=lora_rank,
            alpha=lora_alpha,
            exclude_prefixes=lora_exclude_prefixes,
        )
        trainable, total = trainable_param_count(self.transformer)
        print(
            f"[WanVACE] LoRA injected: {trainable / 1e6:.1f}M trainable / "
            f"{total / 1e6:.1f}M total ({100 * trainable / max(total, 1):.2f}% trained)."
        )

    @property
    def patch_size(self):
        return self.transformer.config.patch_size  # (1, 2, 2)

    @property
    def vace_layers(self):
        return self.transformer.config.vace_layers

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    def forward(
        self,
        hidden_states: Tensor,            # [B, 16, T_lat, H/8, W/8] noisy latent
        timestep: Tensor,                 # [B] or scalar
        encoder_hidden_states: Tensor,    # [B, L, 4096] text embeddings
        control_hidden_states: Tensor,    # [B, 96, T_lat, H/8, W/8] VCU conditioning
        control_hidden_states_scale: Tensor | None = None,  # [num_vace_layers]
    ) -> Tensor:
        """Run the DiT forward, returning velocity prediction [B,16,T_lat,H/8,W/8]."""
        if control_hidden_states_scale is None:
            control_hidden_states_scale = torch.ones(
                len(self.vace_layers),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        out = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=None,
            control_hidden_states=control_hidden_states,
            control_hidden_states_scale=control_hidden_states_scale,
            return_dict=False,
        )
        return out[0]

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def save_lora(self, path: str):
        sd = lora_state_dict(self.transformer)
        if path.endswith((".safetensors", ".sft")):
            from safetensors.torch import save_file
            save_file(sd, path)
        else:
            torch.save(sd, path)
        print(f"[ckpt] Saved {len(sd)} LoRA tensors → {path}")

    def load_lora(self, path: str):
        if path.endswith((".safetensors", ".sft")):
            from safetensors.torch import load_file
            sd = load_file(path, device="cpu")
        else:
            sd = torch.load(path, map_location="cpu")
        load_lora_state_dict(self.transformer, sd)


def build_vace_conditioning(
    ldr_latent: Tensor,    # [B, 16, T_lat, H/8, W/8] — VAE-encoded LDR (Log-Gamma mapped)
    mask: Tensor,          # [B, 1, T_lat, H/8, W/8] — binary exposure mask (1=clipped)
) -> Tensor:
    """Build the 96-channel VCU conditioning tensor.

    Layout: cat([inactive(16), reactive(16), mask_expanded(64)], dim=1) = 96ch.

    Args:
        ldr_latent: VAE-encoded LDR video latent (the context).
        mask: Binary mask in latent spatial/temporal resolution.
              1 = over/underexposed (reactive), 0 = well-exposed (inactive).

    Returns:
        control_hidden_states [B, 96, T_lat, H/8, W/8].
    """
    B, C, T, H, W = ldr_latent.shape
    mask = mask.to(ldr_latent.dtype)
    # inactive = LDR * (1 - mask): well-exposed context
    inactive = ldr_latent * (1.0 - mask)
    # reactive = LDR * mask: clipped regions to hallucinate
    reactive = ldr_latent * mask
    # mask expanded to 64 channels = 4 groups × 16 (matches pretrained convention)
    mask_64 = mask.expand(B, 64, T, H, W)
    return torch.cat([inactive, reactive, mask_64], dim=1)
