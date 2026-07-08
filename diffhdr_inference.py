"""diffhdr_inference.py — DiffHDR inference with context-focused cross-attention.

Loads a Wan VACE + LoRA checkpoint, takes an LDR video + optional text/reference,
runs VACE sampling with the context-focused cross-attention (CFA) module
(paper §3.4.2, eq. 7-8), and outputs HDR video (EXR frames or tonemapped mp4).

Usage:
    python diffhdr_inference.py --input ldr_video.mp4 --output hdr_output/
    python diffhdr_inference.py --input ldr_video.mp4 --output hdr_output/ \
        --prompt "[Overexposed: bright sky]; [Underexposed: dark shadows]" \
        --alpha_over 1.0 --alpha_under 1.0
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from src.models.log_gamma import log_gamma, log_gamma_inverse
from src.models.wan_vae import WanVAE
from src.models.wan_vace import WanVACEBackbone, build_vace_conditioning
from src.models.wan_text_encoder import WanTextEncoder
from src.dataloaders.hdr_video_dataset import (
    DiffHDRVideoConfig,
    DiffHDRVideoTransformGPU,
    _linear_to_srgb,
    _srgb_to_linear,
)


# ---------------------------------------------------------------------------
# Luminance mask detection (paper §3.4.1)
# ---------------------------------------------------------------------------
def detect_exposure_mask(ldr_01: torch.Tensor, tau_high=0.95, tau_low=0.05, ema_alpha=0.7) -> torch.Tensor:
    """Detect over/underexposed regions from LDR video.

    Args:
        ldr_01: (B, N, 3, H, W) LDR in [0, 1] sRGB.
        tau_high, tau_low: luminance thresholds.
        ema_alpha: temporal EMA smoothing factor.

    Returns:
        (B, 1, N, H, W) binary mask (1 = clipped).
    """
    B, N, C, H, W = ldr_01.shape
    ldr_lin = _srgb_to_linear(ldr_01)
    lum = 0.2126 * ldr_lin[:, :, 0] + 0.7152 * ldr_lin[:, :, 1] + 0.0722 * ldr_lin[:, :, 2]
    mask_raw = ((lum > tau_high) | (lum < tau_low)).float()  # (B, N, H, W)

    # Temporal EMA
    mask_ema = torch.zeros_like(mask_raw)
    mask_ema[:, 0] = mask_raw[:, 0]
    for i in range(1, N):
        mask_ema[:, i] = ema_alpha * mask_raw[:, i] + (1 - ema_alpha) * mask_ema[:, i - 1]
    mask = (mask_ema > 0.5).float()
    return mask.unsqueeze(1)  # (B, 1, N, H, W)


# ---------------------------------------------------------------------------
# Context-Focused Cross-Attention (CFA) — paper §3.4.2, eq. 7-8
# ---------------------------------------------------------------------------
def apply_cfa(
    dit: WanVACEBackbone,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    base_embeds: torch.Tensor,
    over_embeds: torch.Tensor,
    under_embeds: torch.Tensor,
    mask_over: torch.Tensor,
    mask_under: torch.Tensor,
    control: torch.Tensor,
    alpha_over: float = 1.0,
    alpha_under: float = 1.0,
) -> torch.Tensor:
    """Apply context-focused cross-attention at inference.

    Instead of a single cross-attention pass, we compute three:
      r_base  = CA(x, c_base)
      r_over  = CA(x, c_over)
      r_under = CA(x, c_under)
    Then combine:
      r = r_base + alpha_over * mask_over * (r_over - r_base)
            + alpha_under * mask_under * (r_under - r_base)

    Since we can't easily hook into individual cross-attention layers of the
    diffusers transformer, we approximate this by running the full DiT three
    times with different text embeddings and blending the velocity predictions
    using the masks.  This is a practical approximation of the paper's
    per-layer CFA.

    Args:
        dit: WanVACEBackbone.
        hidden_states: (B, 16, T_lat, H/8, W/8) noisy latent.
        timestep: (B,) timestep.
        base_embeds, over_embeds, under_embeds: (B, L, 4096) text embeddings.
        mask_over, mask_under: (B, 1, T_lat, H/8, W/8) spatial masks.
        control: (B, 96, T_lat, H/8, W/8) VCU conditioning.
        alpha_over, alpha_under: control strengths.

    Returns:
        Blended velocity prediction (B, 16, T_lat, H/8, W/8).
    """
    r_base = dit(hidden_states, timestep, base_embeds, control)
    r_over = dit(hidden_states, timestep, over_embeds, control)
    r_under = dit(hidden_states, timestep, under_embeds, control)

    # Reshape masks for broadcasting over 16 latent channels
    mo = mask_over.expand_as(r_base)
    mu_ = mask_under.expand_as(r_base)

    r = r_base + alpha_over * mo * (r_over - r_base) + alpha_under * mu_ * (r_under - r_base)
    return r


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_hdr(
    dit: WanVACEBackbone,
    ae: WanVAE,
    encoder: WanTextEncoder,
    ldr_latent: torch.Tensor,
    mask_latent: torch.Tensor,
    prompt: str,
    over_prompt: str = "",
    under_prompt: str = "",
    steps: int = 30,
    cfg_scale: float = 5.0,
    alpha_over: float = 1.0,
    alpha_under: float = 1.0,
    lg_gamma: float = 2.2,
    lg_M: float = 100.0,
) -> torch.Tensor:
    """Run VACE sampling with CFA to produce HDR video.

    Args:
        ldr_latent: (1, 16, T_lat, H/8, W/8) encoded LDR context.
        mask_latent: (1, 1, T_lat, H/8, W/8) exposure mask.
        prompt: base text prompt.
        over_prompt: overexposed region description.
        under_prompt: underexposed region description.
        steps: number of sampling steps.
        cfg_scale: CFG guidance scale.
        alpha_over, alpha_under: CFA control strengths.

    Returns:
        hdr_linear (1, 3, T, H, W) scene-linear HDR radiance.
    """
    device = next(dit.parameters()).device
    B = ldr_latent.shape[0]

    # Build conditioning
    control = build_vace_conditioning(ldr_latent, mask_latent)

    # Encode prompts
    base_embeds = encoder.encode([prompt]).to(device)
    uncond_embeds = encoder.encode([""]).to(device)

    use_cfa = bool(over_prompt or under_prompt)
    if use_cfa:
        over_embeds = encoder.encode([f"[Overexposed: {over_prompt}]"]).to(device)
        under_embeds = encoder.encode([f"[Underexposed: {under_prompt}]"]).to(device)
        # Derive over/under masks from the exposure mask
        # (simplified: use the full mask for both, or split by luminance)
        mask_over = mask_latent
        mask_under = mask_latent

    # Start from noise
    noise = torch.randn_like(ldr_latent)
    img = noise

    # Timestep schedule: t from 1 → 0
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    with torch.autocast("cuda", torch.bfloat16):
        for i in range(steps):
            t_curr = ts[i]
            t_next = ts[i + 1]
            t_vec = torch.full((B,), t_curr, dtype=img.dtype, device=device)

            if use_cfa:
                cond = apply_cfa(
                    dit, img, t_vec,
                    base_embeds, over_embeds, under_embeds,
                    mask_over, mask_under, control,
                    alpha_over, alpha_under,
                )
                uncond = dit(img, t_vec, uncond_embeds, control)
            else:
                cond = dit(img, t_vec, base_embeds, control)
                uncond = dit(img, t_vec, uncond_embeds, control)

            v = uncond + cfg_scale * (cond - uncond)
            img = img + (t_next - t_curr) * v

    # Decode → inverse Log-Gamma → linear HDR
    hdr = _decode_latent_to_hdr(ae, img, lg_gamma, lg_M)
    return hdr


def _decode_latent_to_hdr(ae, latent, gamma, M):
    """Decode latent → [-1,1] → [0,1] → inverse Log-Gamma → linear HDR."""
    decoded = ae.decode(latent)  # (B, C, T, H, W) [-1, 1]
    mapped_01 = (decoded + 1.0) / 2.0
    flat = mapped_01.reshape(-1, *mapped_01.shape[2:])
    hdr_flat = log_gamma_inverse(flat, gamma=gamma, M=M)
    return hdr_flat.reshape(decoded.shape)


# ---------------------------------------------------------------------------
# LDR video loading
# ---------------------------------------------------------------------------
def load_ldr_video(path: str, max_frames: int = 81, target_h: int = 480) -> torch.Tensor:
    """Load an LDR video file and return as (1, N, 3, H, W) in [-1, 1].

    Args:
        path: path to video file (mp4, etc.) or image directory.
        max_frames: max number of frames to load.
        target_h: target height (width auto from aspect ratio).

    Returns:
        (1, N, 3, H, W) float32 in [-1, 1].
    """
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames loaded from {path}")

    # Stack and resize
    arr = np.stack(frames)  # (N, H, W, 3) uint8
    h, w = arr.shape[1], arr.shape[2]
    scale = target_h / h
    new_w = int(w * scale)
    resized = np.stack([
        cv2.resize(f, (new_w, target_h), interpolation=cv2.INTER_AREA)
        for f in arr
    ])

    # Normalize to [-1, 1]
    tensor = torch.from_numpy(resized).float().permute(0, 3, 1, 2) / 127.5 - 1.0  # (N, 3, H, W)
    return tensor.unsqueeze(0)  # (1, N, 3, H, W)


# ---------------------------------------------------------------------------
# Save HDR frames
# ---------------------------------------------------------------------------
def save_hdr_frames(hdr: torch.Tensor, output_dir: str, tonemap: bool = False):
    """Save HDR video frames as EXR or tonemapped PNG.

    Args:
        hdr: (1, 3, T, H, W) scene-linear HDR.
        output_dir: output directory.
        tonemap: if True, save Reinhard-tonemapped PNGs instead of EXR.
    """
    os.makedirs(output_dir, exist_ok=True)
    hdr = hdr[0]  # (3, T, H, W)
    T = hdr.shape[1]

    for t in range(T):
        frame = hdr[:, t]  # (3, H, W)
        if tonemap:
            # Reinhard tonemap
            tm = frame / (1.0 + frame)
            tm = (tm.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            from PIL import Image
            Image.fromarray(tm).save(os.path.join(output_dir, f"frame_{t:04d}.png"))
        else:
            # Save as EXR
            import cv2
            frame_np = frame.permute(1, 2, 0).cpu().numpy().astype(np.float32)
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"frame_{t:04d}.exr"), frame_np)

    print(f"Saved {T} frames to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DiffHDR inference")
    parser.add_argument("--input", required=True, help="LDR video file path")
    parser.add_argument("--output", default="hdr_output", help="Output directory")
    parser.add_argument("--config", default="config_diffhdr.json", help="Config JSON")
    parser.add_argument("--lora_ckpt", required=True, help="LoRA checkpoint path")
    parser.add_argument("--prompt", default="", help="Base text prompt")
    parser.add_argument("--over_prompt", default="", help="Overexposed region description")
    parser.add_argument("--under_prompt", default="", help="Underexposed region description")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--alpha_over", type=float, default=1.0)
    parser.add_argument("--alpha_under", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=81)
    parser.add_argument("--target_h", type=int, default=480)
    parser.add_argument("--tonemap", action="store_true", help="Save tonemapped PNGs instead of EXR")
    args = parser.parse_args()

    import json
    with open(args.config) as f:
        cfg = json.load(f)

    device = "cuda"
    model_id = cfg["model_id"]
    lg_gamma = cfg.get("log_gamma", 2.2)
    lg_M = cfg.get("log_gamma_M", 100.0)

    # Load models
    print("Loading VAE...")
    ae = WanVAE(model_id=model_id, subfolder="vae")
    ae.ae = ae.ae.to(torch.float32).to(device).eval().requires_grad_(False)

    print("Loading text encoder...")
    encoder = WanTextEncoder(model_id=model_id, max_length=512)
    encoder.encoder = encoder.encoder.to(torch.bfloat16).to(device).eval().requires_grad_(False)

    print("Loading DiT + LoRA...")
    dit = WanVACEBackbone(
        model_id=model_id,
        lora_rank=cfg.get("lora_rank", 32),
        lora_alpha=cfg.get("lora_alpha", float(cfg.get("lora_rank", 32))),
        lora_exclude_prefixes=tuple(cfg.get("lora_exclude_prefixes", [])),
    )
    dit.transformer = dit.transformer.to(torch.bfloat16).to(device)
    dit.load_lora(args.lora_ckpt)
    dit.eval()

    # Load LDR video
    print(f"Loading LDR video from {args.input}...")
    ldr = load_ldr_video(args.input, args.max_frames, args.target_h)  # (1, N, 3, H, W) [-1,1]
    ldr = ldr.to(device)
    print(f"  LDR video: {ldr.shape}")

    # Detect exposure mask
    ldr_01 = (ldr + 1.0) / 2.0
    mask = detect_exposure_mask(ldr_01)  # (1, 1, N, H, W)
    print(f"  Mask: {mask.shape}, clipped fraction: {mask.mean():.3f}")

    # Encode LDR → Log-Gamma → VAE latent
    print("Encoding LDR context...")
    from diffhdr_trainer import encode_ldr_to_latent, downsample_mask
    ldr_latent = encode_ldr_to_latent(ae, ldr)
    T_lat, H_lat, W_lat = ldr_latent.shape[2], ldr_latent.shape[3], ldr_latent.shape[4]
    mask_latent = downsample_mask(mask, (T_lat, H_lat, W_lat))

    # Sample HDR
    print("Sampling HDR video...")
    hdr = sample_hdr(
        dit, ae, encoder,
        ldr_latent, mask_latent,
        args.prompt, args.over_prompt, args.under_prompt,
        steps=args.steps, cfg_scale=args.cfg_scale,
        alpha_over=args.alpha_over, alpha_under=args.alpha_under,
        lg_gamma=lg_gamma, lg_M=lg_M,
    )
    print(f"  HDR output: {hdr.shape}")

    # Save
    save_hdr_frames(hdr, args.output, tonemap=args.tonemap)
    print("Done.")


if __name__ == "__main__":
    main()
