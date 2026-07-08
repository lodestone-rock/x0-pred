"""diffhdr_trainer.py — DiffHDR LoRA fine-tuner for Wan-2.1-VACE-14B.

Replicates the training procedure from DiffHDR (arXiv 2604.06161):
  - Backbone: Wan-2.1-VACE-14B (frozen) + rank-32 LoRA on DiT blocks.
  - VAE: Wan-2.1-VAE (frozen, FP32 to avoid banding).
  - Color: Log-Gamma mapping compresses HDR into VAE-compatible range.
  - Conditioning: VCU = LDR latent (inactive/active split by exposure mask)
    + 64ch mask expansion (96ch total).
  - Objective: Rectified flow-matching (Wan convention):
      x_t = (1 - t) * x_clean + t * x_noise
      v_target = x_noise - x_clean
      L = MSE(v_pred, v_target)
  - Data: Polyhaven HDRIs → pseudo-video via equirect projection + LDR synthesis.

Run:
    python diffhdr_trainer.py
    python diffhdr_trainer.py config_diffhdr.json
"""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import shutil
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from ramtorch.multi_gpu import MultiGPUWrapper

from src.models.log_gamma import log_gamma, log_gamma_inverse
from src.models.wan_vae import WanVAE
from src.models.wan_vace import WanVACEBackbone, build_vace_conditioning
from src.models.wan_text_encoder import WanTextEncoder
from src.dataloaders.hdr_video_dataset import (
    DiffHDRVideoConfig,
    DiffHDRVideoDataset,
    DiffHDRVideoTransformGPU,
)

torch.manual_seed(0)
CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config_diffhdr.json"


# ---------------------------------------------------------------------------
# Timestep sampling (shifted logit-uniform, same as K2/Wan)
# ---------------------------------------------------------------------------
def _mu_from_seq_len(seq_len, x1, x2, y1=0.5, y2=1.15):
    slope = (y2 - y1) / (x2 - x1)
    return slope * seq_len + (y1 - slope * x1)


def sample_timesteps(B, device, mu, sigma=1.0):
    u = torch.rand(B, device=device).clamp(1e-5, 1 - 1e-5)
    exp_mu = math.exp(mu)
    t = exp_mu / (exp_mu + (1.0 / u - 1.0) ** sigma)
    return t.float()


# ---------------------------------------------------------------------------
# VAE encode helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_video(vae: WanVAE, video: torch.Tensor) -> torch.Tensor:
    """Encode video [B,C,T,H,W] in [-1,1] → latent [B,16,T_lat,H/8,W/8].

    Applies Log-Gamma mapping *before* VAE encode if the input is HDR linear.
    For LDR context, the input is already in [-1,1] sRGB-like.
    """
    return vae.encode(video)


@torch.no_grad()
def encode_hdr_to_latent(vae: WanVAE, hdr_linear: torch.Tensor, gamma: float, M: float) -> torch.Tensor:
    """Encode scene-linear HDR → Log-Gamma → [-1,1] → VAE latent.

    Args:
        hdr_linear: (B, N, 3, H, W) scene-linear HDR radiance >= 0.
        gamma, M: Log-Gamma parameters.

    Returns:
        latent (B, 16, T_lat, H/8, W/8) bfloat16.
    """
    B, N, C, H, W = hdr_linear.shape
    # Flatten to (B*N, C, H, W) for Log-Gamma
    flat = hdr_linear.reshape(B * N, C, H, W)
    mapped = log_gamma(flat, gamma=gamma, M=M)  # [0, ~1]
    # Scale to [-1, 1] for VAE
    mapped_neg1 = mapped * 2.0 - 1.0
    # Reshape to (B, N, C, H, W) → permute to (B, C, N, H, W) for Wan VAE
    video = mapped_neg1.reshape(B, N, C, H, W).permute(0, 2, 1, 3, 4)
    return vae.encode(video)


@torch.no_grad()
def encode_ldr_to_latent(vae: WanVAE, ldr_input: torch.Tensor) -> torch.Tensor:
    """Encode LDR video [-1,1] → VAE latent.

    Args:
        ldr_input: (B, N, 3, H, W) in [-1, 1] (sRGB LDR).

    Returns:
        latent (B, 16, T_lat, H/8, W/8).
    """
    B, N, C, H, W = ldr_input.shape
    # LDR is already in [-1,1] sRGB — but we need to linearize + Log-Gamma map
    # to match the HDR target's color space. Per the paper, the LDR input is
    # "linearized and mapped to the Log-Gamma color space" before VAE encode.
    # Convert [-1,1] → [0,1] sRGB → linear → Log-Gamma → [-1,1]
    ldr_01 = (ldr_input + 1.0) / 2.0  # [0,1] sRGB
    # sRGB → linear
    ldr_lin = torch.where(
        ldr_01 <= 0.04045,
        ldr_01 / 12.92,
        ((ldr_01 + 0.055) / 1.055).pow(2.4),
    )
    flat = ldr_lin.reshape(B * N, C, H, W)
    mapped = log_gamma(flat.clamp(min=0), gamma=2.2, M=100.0)
    mapped_neg1 = mapped * 2.0 - 1.0
    video = mapped_neg1.reshape(B, N, C, H, W).permute(0, 2, 1, 3, 4)
    return vae.encode(video)


@torch.no_grad()
def decode_latent_to_hdr(vae: WanVAE, latent: torch.Tensor, gamma: float, M: float) -> torch.Tensor:
    """Decode latent → [-1,1] → [0,1] → inverse Log-Gamma → linear HDR.

    Args:
        latent: (B, 16, T_lat, H/8, W/8).
        gamma, M: Log-Gamma parameters (must match encode).

    Returns:
        hdr_linear (B, C, T, H, W) scene-linear radiance >= 0.
    """
    decoded = vae.decode(latent)  # (B, C, T, H, W) in [-1, 1]
    mapped_01 = (decoded + 1.0) / 2.0  # [0, ~1]
    # Inverse Log-Gamma → linear HDR
    flat = mapped_01.reshape(-1, *mapped_01.shape[2:])  # (B*T, C, H, W)
    hdr_flat = log_gamma_inverse(flat, gamma=gamma, M=M)
    return hdr_flat.reshape(decoded.shape)


# ---------------------------------------------------------------------------
# Mask downsampling to latent resolution
# ---------------------------------------------------------------------------
def downsample_mask(mask: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Downsample binary mask to latent spatial/temporal resolution.

    Args:
        mask: (B, 1, N, H, W) binary mask in pixel space.
        target_shape: (T_lat, H_lat, W_lat) latent dimensions.

    Returns:
        (B, 1, T_lat, H_lat, W_lat) downsampled mask (nearest-exact).
    """
    B, _, N, H, W = mask.shape
    T_lat, H_lat, W_lat = target_shape
    # Temporal: use F.interpolate on the time axis (handles non-divisible N).
    # mask: (B, 1, N, H, W) → (B*H*W, 1, N) → interpolate to T_lat → back
    mask = mask.float()
    mask = mask.permute(0, 3, 4, 1, 2).reshape(B * H * W, 1, N)  # (B*H*W, 1, N)
    mask = F.interpolate(mask, size=T_lat, mode="nearest-exact")  # (B*H*W, 1, T_lat)
    mask = mask.reshape(B, H, W, 1, T_lat).permute(0, 3, 4, 1, 2)  # (B, 1, T_lat, H, W)
    mask = (mask > 0.5).float()
    # Spatial: nearest-exact downsample
    mask = F.interpolate(
        mask.reshape(B * T_lat, 1, H, W),
        size=(H_lat, W_lat),
        mode="nearest-exact",
    ).reshape(B, 1, T_lat, H_lat, W_lat)
    return mask


# ---------------------------------------------------------------------------
# Per-GPU forward pass
# ---------------------------------------------------------------------------
def forward_fn(
    gpu_id: int,
    dit: WanVACEBackbone,
    hdr_linear: torch.Tensor,    # (B, N, 3, H, W) scene-linear HDR
    ldr_input: torch.Tensor,     # (B, N, 3, H, W) LDR in [-1, 1]
    mask: torch.Tensor,           # (B, 1, N, H, W) binary exposure mask
    captions: list[str],
    *,
    aes: list,
    encoders: list,
    uncond_ratio: float,
    log_gamma_gamma: float,
    log_gamma_M: float,
    mu_y1: float,
    mu_y2: float,
    mu_override: float | None,
    mu_sigma: float,
) -> tuple:
    """Encode → flow-matching → DiT forward → return (v_pred, v_target, t, ...)."""
    device = f"cuda:{gpu_id}"
    ae = aes[gpu_id]
    encoder = encoders[gpu_id]

    B, N, C, H, W = hdr_linear.shape

    # ---- Text conditioning -----------------------------------------------
    dropped = ["" if (torch.rand(1).item() < uncond_ratio) else c for c in captions]
    with torch.no_grad():
        txt_embeds = encoder.encode(dropped)  # (B, 512, 4096)
        txt_embeds = txt_embeds.to(device)

    # ---- VAE encode HDR target (Log-Gamma → VAE) -------------------------
    x1 = encode_hdr_to_latent(ae, hdr_linear.to(device), log_gamma_gamma, log_gamma_M)
    # x1: (B, 16, T_lat, H/8, W/8) bfloat16

    # ---- VAE encode LDR context (Log-Gamma → VAE) -------------------------
    ldr_latent = encode_ldr_to_latent(ae, ldr_input.to(device))

    # ---- Downsample mask to latent resolution ----------------------------
    T_lat, H_lat, W_lat = x1.shape[2], x1.shape[3], x1.shape[4]
    mask_latent = downsample_mask(mask.to(device), (T_lat, H_lat, W_lat))

    # ---- Build VCU conditioning (96ch) -----------------------------------
    control = build_vace_conditioning(ldr_latent, mask_latent)  # (B, 96, T_lat, H/8, W/8)

    # ---- Noise + flow-matching interpolation ------------------------------
    x0 = torch.randn_like(x1)
    # Sample timesteps (resolution-aware shifted schedule)
    patch = dit.patch_size  # (1, 2, 2)
    spatial_patch = patch[1] * patch[2]  # 4
    img_seq_len = T_lat * (H_lat // patch[1]) * (W_lat // patch[2])
    # Use a simple mu for now (Wan uses dynamic shifting based on seq len)
    if mu_override is not None:
        mu = mu_override
    else:
        # Rough interpolation endpoints
        x1_res = (256 // (8 * spatial_patch)) ** 2
        x2_res = (1280 // (8 * spatial_patch)) ** 2
        mu = _mu_from_seq_len(img_seq_len, x1_res, x2_res, mu_y1, mu_y2)
    t = sample_timesteps(B, device=device, mu=mu, sigma=mu_sigma)  # (B,)

    # x_t = (1 - t) * x_clean + t * x_noise
    t4 = t[:, None, None, None, None].to(x1.dtype)
    x_t = (1.0 - t4) * x1 + t4 * x0

    # v_target = x_noise - x_clean (Wan convention: v points clean→noise)
    v_target = x0 - x1

    # ---- DiT forward -----------------------------------------------------
    with torch.autocast("cuda", torch.bfloat16):
        v_pred = dit(
            hidden_states=x_t,
            timestep=t,
            encoder_hidden_states=txt_embeds,
            control_hidden_states=control,
        )

    return v_pred, v_target.to(v_pred.dtype), t, x1, ldr_latent, mask_latent, captions


# ---------------------------------------------------------------------------
# Per-GPU backward pass
# ---------------------------------------------------------------------------
def backward_fn(gpu_id, dit, fwd_output, accum_steps=1):
    v_pred, v_target, t, *_ = fwd_output
    loss = F.mse_loss(v_pred, v_target)
    (loss / accum_steps).backward()
    return loss.item()


# ---------------------------------------------------------------------------
# Preview / inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def preview_fn(
    gpu_id: int,
    dit: WanVACEBackbone,
    ae: WanVAE,
    encoder: WanTextEncoder,
    x1_clean: torch.Tensor,      # (B, 16, T_lat, H/8, W/8) — encoded HDR target
    ldr_latent: torch.Tensor,    # (B, 16, T_lat, H/8, W/8) — encoded LDR context
    mask_latent: torch.Tensor,   # (B, 1, T_lat, H/8, W/8)
    captions: list[str],
    log_gamma_gamma: float,
    log_gamma_M: float,
    steps: int = 30,
    cfg_scale: float = 5.0,
    n_samples: int = 1,
) -> torch.Tensor:
    """Run Euler+CFG sampling and return a tonemapped preview grid."""
    device = f"cuda:{gpu_id}"
    n_samples = min(n_samples, x1_clean.shape[0])
    x1_ref = x1_clean[:n_samples].to(device)
    ldr_ref = ldr_latent[:n_samples].to(device)
    mask_ref = mask_latent[:n_samples].to(device)

    # Build conditioning
    control = build_vace_conditioning(ldr_ref, mask_ref)

    # Encode prompts
    prompts = list(captions[:n_samples])
    txt = encoder.encode(prompts).to(device)
    untxt = encoder.encode([""] * n_samples).to(device)

    # Start from noise
    noise = torch.randn_like(x1_ref)
    B, _, T_lat, H_lat, W_lat = noise.shape

    # Timestep schedule: t from 1 → 0
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    img = noise
    with torch.autocast("cuda", torch.bfloat16):
        for i in range(steps):
            t_curr = ts[i]
            t_next = ts[i + 1]
            t_vec = torch.full((n_samples,), t_curr, dtype=img.dtype, device=device)
            cond = dit(img, t_vec, txt, control)
            uncond = dit(img, t_vec, untxt, control)
            v = uncond + cfg_scale * (cond - uncond)
            # Euler step: x_{t-dt} = x_t + (t_next - t_curr) * v
            img = img + (t_next - t_curr) * v

    # Decode → HDR → tonemap for preview
    hdr_out = decode_latent_to_hdr(ae, img, log_gamma_gamma, log_gamma_M)
    # hdr_out: (B, C, T, H, W). Take middle temporal frame.
    mid_t = hdr_out.shape[2] // 2
    hdr_mid = hdr_out[:, :, mid_t]  # (B, C, H, W)
    # Simple Reinhard tonemap for display
    tonemapped = hdr_mid / (1.0 + hdr_mid)
    tonemapped = (tonemapped * 2.0 - 1.0).clamp(-1, 1)  # (B, C, H, W) [-1,1]

    # Also decode ground truth
    gt_hdr = decode_latent_to_hdr(ae, x1_ref, log_gamma_gamma, log_gamma_M)
    gt_mid = gt_hdr[:, :, mid_t]  # (B, C, H, W)
    gt_tm = gt_mid / (1.0 + gt_mid)
    gt_tm = (gt_tm * 2.0 - 1.0).clamp(-1, 1)  # (B, C, H, W) [-1,1]

    # Also get the LDR input for reference (decode ldr_latent)
    ldr_decoded = ae.decode(ldr_ref)  # (B, C, T, H, W) [-1,1]
    ldr_mid = ldr_decoded[:, :, mid_t]  # (B, C, H, W)

    # Stack: [pred, gt, ldr_input] vertically
    return torch.cat([tonemapped, gt_tm, ldr_mid], dim=0)  # (3*B, C, H, W)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(cfg: dict):
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPU(s).")

    os.makedirs(cfg["ckpt_path"], exist_ok=True)
    os.makedirs(cfg["preview_path"], exist_ok=True)
    shutil.copy(CONFIG_PATH, os.path.join(cfg["ckpt_path"], os.path.basename(CONFIG_PATH)))

    dtype = torch.bfloat16
    model_id = cfg["model_id"]

    # ---- Load frozen VAE (one per GPU) -----------------------------------
    print("Loading Wan VAE (FP32)...")
    base_ae = WanVAE(model_id=model_id, subfolder="vae")
    base_ae.ae = base_ae.ae.to(torch.float32).eval().requires_grad_(False)
    # Enable tiling/slicing to save VAE memory.
    if hasattr(base_ae.ae, "enable_tiling"):
        base_ae.ae.enable_tiling()
    if hasattr(base_ae.ae, "enable_slicing"):
        base_ae.ae.enable_slicing()
    aes = []
    for gpu_id in range(n_gpus):
        a = copy.deepcopy(base_ae).to(f"cuda:{gpu_id}")
        a.eval().requires_grad_(False)
        aes.append(a)
    del base_ae
    print(f"  VAE ready on {n_gpus} GPU(s).")

    # ---- Load frozen text encoder (one per GPU) --------------------------
    print("Loading UMT5 text encoder...")
    base_encoder = WanTextEncoder(model_id=model_id, max_length=512)
    base_encoder.encoder = base_encoder.encoder.to(dtype).eval().requires_grad_(False)
    encoders = []
    for gpu_id in range(n_gpus):
        e = copy.deepcopy(base_encoder).to(f"cuda:{gpu_id}")
        e.eval().requires_grad_(False)
        encoders.append(e)
    del base_encoder
    print(f"  Text encoder ready on {n_gpus} GPU(s).")

    # ---- Load DiT + inject LoRA ------------------------------------------
    print("Loading Wan VACE transformer + LoRA...")
    base_dit = WanVACEBackbone(
        model_id=model_id,
        lora_rank=cfg.get("lora_rank", 32),
        lora_alpha=cfg.get("lora_alpha", float(cfg.get("lora_rank", 32))),
        lora_exclude_prefixes=tuple(cfg.get("lora_exclude_prefixes", [])),
    )
    base_dit.transformer = base_dit.transformer.to(dtype)
    # Enable gradient checkpointing to save activation memory.
    if cfg.get("gradient_checkpointing", True):
        base_dit.transformer.enable_gradient_checkpointing()
        print("  Gradient checkpointing enabled.")

    def dit_factory():
        return copy.deepcopy(base_dit)

    # ---- Optimizer -------------------------------------------------------
    lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 1e-4)
    warmup_steps = cfg.get("warmup", 200)
    accum_steps = cfg.get("accum", 4)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

    def optimizer_factory(params):
        trainable = [p for p in params if p.requires_grad]
        return AdamW(trainable, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    wrapper = MultiGPUWrapper(
        model_factory=dit_factory,
        optimizer_factory=optimizer_factory,
        gradient_accumulation_steps=accum_steps,
        max_grad_norm=max_grad_norm,
        scheduler_factory=lambda opt: LinearLR(
            opt, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps
        ),
    )
    wrapper.setup()
    for gpu_id in range(n_gpus):
        wrapper.models[gpu_id].to(dtype)

    # ---- Checkpoint load -------------------------------------------------
    lora_ckpt = cfg.get("lora_checkpoint")
    if lora_ckpt:
        for gpu_id in range(n_gpus):
            wrapper.models[gpu_id].load_lora(lora_ckpt)
        print(f"  LoRA checkpoint loaded on all {n_gpus} GPU(s).")

    def _save_checkpoint(path):
        wrapper.models[0].save_lora(path)

    if not lora_ckpt:
        _save_checkpoint(os.path.join(cfg["ckpt_path"], "untrained_lora.safetensors"))

    # ---- Dataset ---------------------------------------------------------
    ds_cfg = cfg["dataset"]
    video_cfg = DiffHDRVideoConfig(**ds_cfg)
    dataset = DiffHDRVideoDataset(video_cfg)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"] * n_gpus,
        shuffle=True,
        num_workers=ds_cfg.get("num_workers", 4),
        prefetch_factor=ds_cfg.get("prefetch_factor", 2),
        pin_memory=True,
        collate_fn=_collate_fn,
    )
    # GPU transform (shared, moved to each GPU in forward)
    gpu_transforms = []
    for gpu_id in range(n_gpus):
        t = DiffHDRVideoTransformGPU(video_cfg).to(f"cuda:{gpu_id}")
        t.eval()
        gpu_transforms.append(t)

    # ---- Training config -------------------------------------------------
    global_step = cfg.get("initial_global_step", 0)
    eval_interval = cfg.get("eval_interval", 200)
    save_every = cfg.get("save_every_n_steps", 1000)
    log_every = cfg.get("log_every_n_steps", 10)
    uncond_ratio = cfg.get("uncond_ratio", 0.1)
    lg_gamma = cfg.get("log_gamma", 2.2)
    lg_M = cfg.get("log_gamma_M", 100.0)
    mu_y1 = cfg.get("mu_y1", 0.5)
    mu_y2 = cfg.get("mu_y2", 1.15)
    mu_override = cfg.get("mu_override", None)
    mu_sigma = cfg.get("mu_sigma", 1.0)
    preview_spg = cfg.get("preview_samples_per_gpu", 1)
    preview_cfg = cfg.get("preview_cfg_scale", 5.0)
    preview_steps = cfg.get("preview_steps", 30)
    max_steps = cfg.get("max_steps", 0)
    master_seed = cfg.get("seed", 42)

    # ---- CSV loss log ----------------------------------------------------
    csv_path = os.path.join(cfg["ckpt_path"], "loss_log.csv")
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        csv_writer.writerow(["step", "loss", "lr", "time"])
    t0 = time.time()

    # ---- Training loop ---------------------------------------------------
    torch.manual_seed(master_seed)
    epoch = 0

    while True:
        epoch += 1
        torch.manual_seed(master_seed + epoch)
        for m in wrapper.models:
            m.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (raw_batch, caption_batch) in enumerate(pbar):
            spg = raw_batch.shape[0] // n_gpus
            if spg == 0:
                continue

            # ---- GPU transform: project + LDR synthesis per GPU ----
            all_outputs = []
            for gpu_id in range(n_gpus):
                raw_chunk = raw_batch[gpu_id * spg:(gpu_id + 1) * spg].to(f"cuda:{gpu_id}")
                cap_chunk = caption_batch[gpu_id * spg:(gpu_id + 1) * spg]
                out = gpu_transforms[gpu_id](raw_chunk)
                out["captions"] = list(cap_chunk)
                all_outputs.append(out)

            # ---- Forward -------------------------------------------
            outputs = {}
            for gpu_id in range(n_gpus):
                o = all_outputs[gpu_id]
                fwd = forward_fn(
                    gpu_id,
                    wrapper.models[gpu_id],
                    o["hdr_linear"],
                    o["ldr_input"],
                    o["mask"],
                    o["captions"],
                    aes=aes,
                    encoders=encoders,
                    uncond_ratio=uncond_ratio,
                    log_gamma_gamma=lg_gamma,
                    log_gamma_M=lg_M,
                    mu_y1=mu_y1,
                    mu_y2=mu_y2,
                    mu_override=mu_override,
                    mu_sigma=mu_sigma,
                )
                outputs[gpu_id] = fwd

            # ---- Backward ------------------------------------------
            raw_results = wrapper.run_concurrent(
                lambda gpu_id: backward_fn(
                    gpu_id,
                    wrapper.models[gpu_id],
                    outputs[gpu_id],
                    accum_steps=accum_steps,
                )
            )
            total_loss = sum(r for r in raw_results) / n_gpus

            # ---- Optimizer step ------------------------------------
            if (batch_idx + 1) % accum_steps == 0:
                wrapper.reduce_grads()
                wrapper.clip_grads()
                wrapper.optimizer_step()
                torch.cuda.synchronize()

            lr_now = wrapper.last_lr
            pbar.set_postfix(loss=f"{total_loss:.4f}", lr=f"{lr_now:.2e}", step=global_step)

            csv_writer.writerow([global_step, f"{total_loss:.6f}", f"{lr_now:.2e}", f"{time.time() - t0:.1f}"])
            if global_step % log_every == 0:
                csv_file.flush()

            # ---- Checkpoint -----------------------------------------
            if save_every > 0 and global_step > 0 and global_step % save_every == 0:
                _save_checkpoint(os.path.join(cfg["ckpt_path"], f"lora_step_{global_step}.safetensors"))

            # ---- Preview --------------------------------------------
            if global_step % eval_interval == 0 and global_step > 0:
                for m in wrapper.models:
                    m.eval()
                try:
                    all_rows = []
                    for gpu_id in range(n_gpus):
                        o = all_outputs[gpu_id]
                        fwd = outputs[gpu_id]
                        rows = preview_fn(
                            gpu_id,
                            wrapper.models[gpu_id],
                            aes[gpu_id],
                            encoders[gpu_id],
                            fwd[3],  # x1_clean
                            fwd[4],  # ldr_latent
                            fwd[5],  # mask_latent
                            fwd[6],  # captions
                            lg_gamma, lg_M,
                            steps=preview_steps,
                            cfg_scale=preview_cfg,
                            n_samples=min(preview_spg, spg),
                        )
                        all_rows.append(rows)
                    combined = torch.cat(all_rows, dim=0)
                    # Ensure 3 channels (preview may have extra dims)
                    if combined.shape[1] != 3:
                        combined = combined[:, :3]
                    grid = _make_grid((combined + 1) / 2, nrow=max(preview_spg * n_gpus, 1))
                    ext = "png" if cfg.get("preview_quality", 95) >= 100 else "jpg"
                    img_path = f"{cfg['preview_path']}/step_{global_step}.{ext}"
                    if _PIL_AVAILABLE:
                        grid_np = (grid.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                        Image.fromarray(grid_np).save(img_path)
                    print(f"[preview] Saved {img_path}")
                except Exception as e:
                    print(f"[preview] Failed at step {global_step}: {e}")
                for m in wrapper.models:
                    m.train()

            global_step += 1
            if max_steps > 0 and global_step >= max_steps:
                print(f"Reached max_steps={max_steps}. Saving final checkpoint.")
                _save_checkpoint(os.path.join(cfg["ckpt_path"], f"lora_step_{global_step}_final.safetensors"))
                csv_file.close()
                return

    csv_file.close()


# ---------------------------------------------------------------------------
# Collate + grid helpers
# ---------------------------------------------------------------------------
def _collate_fn(batch):
    """Collate raw equirects and captions."""
    raws = torch.stack([b[0] for b in batch])
    captions = [b[1] for b in batch]
    return raws, captions


def _make_grid(tensor, nrow):
    """Simple grid maker (avoids torchvision import issues)."""
    from torchvision.utils import make_grid
    return make_grid(tensor, nrow=nrow)


if __name__ == "__main__":
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    train(cfg)
