"""plksr_diff_trainer.py — Standalone flow-matching SR diffusion trainer.

Training target: predict the residual between HR and nearest-upsampled LR, in
pixel-unshuffled (LR) space.  Everything is self-contained — no neosr imports.

Run:
    python plksr_diff_trainer.py
    python plksr_diff_trainer.py config_plksr_diff.json

All settings live in config_plksr_diff.json.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import shutil
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from safetensors.torch import load_file as _st_load, save_file as _st_save
    _SAFETENSORS = True
except ImportError:
    _SAFETENSORS = False
    print("[warn] safetensors not available — checkpoints saved as .pt")

try:
    from torchvision.utils import make_grid, save_image
    _TV_SAVE = True
except ImportError:
    _TV_SAVE = False
    print("[warn] torchvision unavailable — no preview images saved")

from src.models.plksr_diff import PLKSRDiff
from src.models.metagan_standalone import MetaGAN
from src.dataloaders.otf_sr_dataloader import (
    OTFSRDataset,
    DiffJPEG,
    apply_degradation,
)

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config_plksr_diff.json"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_ckpt(path: str, state_dict: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if _SAFETENSORS and path.endswith(".safetensors"):
        _st_save(state_dict, path)
    else:
        torch.save(state_dict, path)


def load_ckpt(path: str, device: str = "cpu") -> dict:
    if _SAFETENSORS and path.endswith(".safetensors"):
        return _st_load(path, device=device)
    return torch.load(path, map_location=device)


# ---------------------------------------------------------------------------
# Timestep distribution helpers (ported from pixel_space_transformers.py)
# ---------------------------------------------------------------------------

def _logit_normal_pdf(t: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    """Unnormalised logit-normal density evaluated at t ∈ (0, 1)."""
    eps = 1e-6
    t = t.clamp(eps, 1 - eps)
    logit_t = torch.log(t / (1 - t))
    log_p = -0.5 * ((logit_t - mu) / sigma) ** 2 - torch.log(t * (1 - t))
    return log_p.exp()


def create_distribution(n: int, device, mu: float = 0.0, sigma: float = 1.0):
    """Build a discrete CDF over [0, 1] from a logit-normal with given mu."""
    t = torch.linspace(0, 1, n, device=device)
    p = _logit_normal_pdf(t, mu=mu, sigma=sigma)
    p = p / p.sum()
    return t, p


def sample_from_distribution(x_dist: torch.Tensor, probs: torch.Tensor, n: int) -> torch.Tensor:
    """Sample n timesteps from a discrete distribution."""
    idx = torch.multinomial(probs, n, replacement=True)
    return x_dist[idx]


# ---------------------------------------------------------------------------
# GAN losses (hinge)
# ---------------------------------------------------------------------------

def disc_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return (F.relu(1.0 - real_logits) + F.relu(1.0 + fake_logits)).mean()


def gen_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


# ---------------------------------------------------------------------------
# Preview helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_preview(
    model: PLKSRDiff,
    lr_batch: torch.Tensor,
    hr_batch: torch.Tensor,
    num_steps: int,
    preview_path: str,
    step: int,
    n_samples: int = 4,
) -> None:
    if not _TV_SAVE:
        return
    model.eval()
    device = lr_batch.device
    lr = lr_batch[:n_samples].to(device)
    hr = hr_batch[:n_samples].to(device)

    sr = model.euler_sample(lr, num_steps=num_steps)

    # Build grid: [LR upsampled | SR | HR]
    lr_up = F.interpolate(lr, scale_factor=model.upscale, mode="nearest").clamp(0, 1)
    grid = make_grid(torch.cat([lr_up, sr, hr], dim=0), nrow=n_samples)
    os.makedirs(preview_path, exist_ok=True)
    save_image(grid, os.path.join(preview_path, f"step_{step:07d}.png"))
    model.train()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device}  |  GPUs: {n_gpus}")

    os.makedirs(cfg["ckpt_path"], exist_ok=True)
    os.makedirs(cfg["preview_path"], exist_ok=True)
    shutil.copy(CONFIG_PATH, os.path.join(cfg["ckpt_path"], os.path.basename(CONFIG_PATH)))

    torch.manual_seed(cfg.get("seed", 42))
    random.seed(cfg.get("seed", 42))

    upscale = int(cfg.get("upscale", 2))
    batch_size = int(cfg["batch_size"])
    accum_steps = int(cfg.get("accum", 4))
    use_gan = bool(cfg.get("use_gan", False))
    gan_weight = float(cfg.get("gan_weight", 0.1))
    t_mu = float(cfg.get("t_mu", 0.0))

    # ------------------------------------------------------------------
    # Dataset + Dataloader
    # ------------------------------------------------------------------
    ds_cfg = dict(cfg.get("dataset", {}))
    ds_cfg["upscale"] = upscale
    ds_cfg["batch_size"] = batch_size
    dataset = OTFSRDataset(ds_cfg)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=ds_cfg.get("num_workers", 4),
        prefetch_factor=ds_cfg.get("prefetch_factor", 2),
        pin_memory=True,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_cfg = dict(cfg.get("model_config", {}))
    model_cfg["upscale"] = upscale
    model = PLKSRDiff(**model_cfg).to(device)

    # ------------------------------------------------------------------
    # DiffJPEG (shared, single instance on device)
    # ------------------------------------------------------------------
    jpeger = DiffJPEG(differentiable=False).to(device)

    # ------------------------------------------------------------------
    # GAN discriminator (optional)
    # ------------------------------------------------------------------
    disc: MetaGAN | None = None
    disc_opt: AdamW | None = None
    disc_sched = None
    in_ch = int(model_cfg.get("in_ch", 3))
    noise_ch = in_ch * upscale ** 2

    if use_gan:
        disc_cfg = cfg.get("disc_config", {})
        disc = MetaGAN(
            in_ch=noise_ch,
            **{k: v for k, v in disc_cfg.items() if k != "in_ch"},
        ).to(device)
        disc_opt = AdamW(disc.parameters(), lr=cfg.get("disc_lr", cfg["lr"]),
                         weight_decay=1e-4, betas=(0.9, 0.95))
        disc_sched = LinearLR(disc_opt, start_factor=1e-5, end_factor=1.0,
                              total_iters=cfg.get("warmup", 500))
        print(f"GAN enabled — disc in_ch={noise_ch}, gan_weight={gan_weight}")

    # ------------------------------------------------------------------
    # Optimizer + scheduler (generator)
    # ------------------------------------------------------------------
    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4, betas=(0.9, 0.95))
    sched = LinearLR(opt, start_factor=1e-5, end_factor=1.0,
                     total_iters=cfg.get("warmup", 500))

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------
    global_step = int(cfg.get("initial_global_step", 0))
    if cfg.get("model_checkpoint"):
        sd = load_ckpt(cfg["model_checkpoint"])
        model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint: {cfg['model_checkpoint']}")
    else:
        ckpt_path_init = os.path.join(cfg["ckpt_path"], "untrained.safetensors"
                                      if _SAFETENSORS else "untrained.pt")
        save_ckpt(ckpt_path_init, model.state_dict())
        print(f"Saved untrained checkpoint: {ckpt_path_init}")

    if cfg.get("disc_checkpoint") and disc is not None:
        sd = load_ckpt(cfg["disc_checkpoint"])
        disc.load_state_dict(sd, strict=False)
        print(f"Loaded disc checkpoint: {cfg['disc_checkpoint']}")

    # ------------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------------
    csv_path = os.path.join(cfg["ckpt_path"], "loss_log.csv")
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        header = ["step", "loss", "mse", "gan_g", "disc"]
        if use_gan:
            header += ["disc_real", "disc_fake"]
        csv_writer.writerow(header)

    # ------------------------------------------------------------------
    # Degradation config
    # ------------------------------------------------------------------
    deg_cfg = ds_cfg.get("degradation", {})
    _train_rng = random.Random(cfg.get("seed", 42))

    # Grad accumulation buffers
    _acc_loss = 0.0
    _acc_mse = 0.0
    _acc_gan_g = 0.0
    _acc_disc = 0.0
    _acc_steps_done = 0
    _t0 = time.time()

    log_every = int(cfg.get("log_every_n_steps", 10))
    save_every = int(cfg.get("save_every_n_steps", 1000))
    eval_every = int(cfg.get("eval_interval", 500))
    preview_steps = int(cfg.get("preview_steps", 30))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    model.train()
    if disc is not None:
        disc.train()

    # Keep a preview batch around for consistent eval images
    _preview_lr: torch.Tensor | None = None
    _preview_hr: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Training epochs
    # ------------------------------------------------------------------
    epoch = 0
    while True:
        epoch += 1
        dataset.resample()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            # ----------------------------------------------------------------
            # 1. Move HR + kernels to GPU
            # ----------------------------------------------------------------
            hr = batch["hr"].to(device, non_blocking=True)           # (B, 3, H, W)
            k1 = batch["kernel1"].to(device, non_blocking=True)      # (B, 21, 21)
            k2 = batch["kernel2"].to(device, non_blocking=True)
            sk = batch["sinc_kernel"].to(device, non_blocking=True)

            B = hr.shape[0]

            # ----------------------------------------------------------------
            # 2. GPU-side degradation → produce LR
            # ----------------------------------------------------------------
            with torch.no_grad():
                lr = apply_degradation(hr, k1, k2, sk, upscale, jpeger, deg_cfg, _train_rng)
                # lr : (B, 3, H//upscale, W//upscale) [0, 1]

            # ----------------------------------------------------------------
            # 3. Build flow-matching target (shuffled residual)
            # ----------------------------------------------------------------
            with torch.no_grad():
                lr_up = F.interpolate(lr, scale_factor=upscale, mode="nearest")
                # Residual in HR space, then pixel-unshuffle into LR+channel space
                residual_hr = hr - lr_up                               # (B, 3, H, W)
                target_delta = F.pixel_unshuffle(residual_hr, upscale) # (B, noise_ch, H_lr, W_lr)

            # ----------------------------------------------------------------
            # 4. Sample t and build noisy latent
            # ----------------------------------------------------------------
            x_dist, probs = create_distribution(1000, device=device, mu=t_mu)
            t_vec = sample_from_distribution(x_dist, probs, B)         # (B,)
            t = t_vec[:, None, None, None].to(hr.dtype)                 # (B,1,1,1)

            x0 = torch.randn_like(target_delta)
            x_noisy = x0 * t + target_delta * (1.0 - t)

            # ----------------------------------------------------------------
            # 5. Discriminator update (if GAN enabled, every step)
            # ----------------------------------------------------------------
            disc_loss_val = 0.0
            if use_gan and disc is not None:
                with torch.autocast("cuda", torch.bfloat16):
                    pred_delta = model(x_noisy.detach(), lr, t_vec)

                real_logits = disc(target_delta.detach())
                fake_logits = disc(pred_delta.detach())
                d_loss = disc_hinge_loss(real_logits, fake_logits)

                disc_opt.zero_grad(set_to_none=True)
                d_loss.backward()
                nn.utils.clip_grad_norm_(disc.parameters(), max_grad_norm)
                disc_opt.step()
                if disc_sched is not None:
                    disc_sched.step()

                disc_loss_val = d_loss.item()

            # ----------------------------------------------------------------
            # 6. Generator / model forward + loss
            # ----------------------------------------------------------------
            with torch.autocast("cuda", torch.bfloat16):
                pred_delta = model(x_noisy, lr, t_vec)

            target_v = (x_noisy - target_delta) / (t + 5e-2)
            pred_v   = (x_noisy - pred_delta)   / (t + 5e-2)
            mse = F.mse_loss(pred_v, target_v)

            gen_adv = torch.tensor(0.0, device=device)
            if use_gan and disc is not None:
                fake_logits_g = disc(pred_delta)
                gen_adv = gen_hinge_loss(fake_logits_g) * gan_weight

            total_loss = mse + gen_adv

            # Gradient accumulation
            (total_loss / accum_steps).backward()
            _acc_loss += total_loss.item()
            _acc_mse  += mse.item()
            _acc_gan_g += gen_adv.item()
            _acc_disc  += disc_loss_val
            _acc_steps_done += 1

            # ----------------------------------------------------------------
            # 7. Optimizer step (every accum_steps)
            # ----------------------------------------------------------------
            if _acc_steps_done % accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

                avg_loss  = _acc_loss  / accum_steps
                avg_mse   = _acc_mse   / accum_steps
                avg_gan_g = _acc_gan_g / accum_steps
                avg_disc  = _acc_disc  / accum_steps
                _acc_loss = _acc_mse = _acc_gan_g = _acc_disc = 0.0

                pbar.set_postfix(step=global_step, loss=f"{avg_loss:.4f}",
                                 mse=f"{avg_mse:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

                # Logging
                if global_step % log_every == 0:
                    row = [global_step, f"{avg_loss:.6f}", f"{avg_mse:.6f}",
                           f"{avg_gan_g:.6f}", f"{avg_disc:.6f}"]
                    csv_writer.writerow(row)
                    csv_file.flush()

                # Checkpoint
                if global_step % save_every == 0:
                    ext = ".safetensors" if _SAFETENSORS else ".pt"
                    ckpt = os.path.join(cfg["ckpt_path"], f"step_{global_step:07d}{ext}")
                    save_ckpt(ckpt, model.state_dict())
                    # rolling latest
                    save_ckpt(os.path.join(cfg["ckpt_path"], f"latest{ext}"),
                               model.state_dict())
                    if disc is not None:
                        save_ckpt(os.path.join(cfg["ckpt_path"], f"disc_latest{ext}"),
                                   disc.state_dict())
                    print(f"[step {global_step}] Saved checkpoint")

                # Preview
                if global_step % eval_every == 0:
                    # Cache the first preview batch so images are consistent
                    if _preview_lr is None:
                        _preview_lr = lr.detach().clone()
                        _preview_hr = hr.detach().clone()
                    run_preview(model, _preview_lr, _preview_hr,
                                preview_steps, cfg["preview_path"], global_step)

    csv_file.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        TRAINING_CONFIG = json.load(f)
    train(TRAINING_CONFIG)
