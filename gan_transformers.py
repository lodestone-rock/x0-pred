"""imagenet_gan_v2.py — Relativistic GAN trainer using MultiGPUWrapper.

Architecture
------------
* Generator  G(noise, text_tokens=None) → fake patches
* Discriminator D(image_patches, cls_label, image_shape, text_tokens, noise_level)
  → [B, disc_embed_dim] embedding; logit = dot product vs label vector.

Training gist
-------------
* Relativistic GAN (RaGAN) loss — no gradient penalty.
* Adaptive noise augmentation on D inputs: noise_lerp_val EMA tracks
  whether D is becoming too strong (logit >> ln2) and increases input
  noise accordingly to prevent vanishing gradients for G.
* D-optimality gate: G is only updated when D is not already too weak
  (avg_d_logits < target_d_loss).  This keeps D optimal.
* Grad accumulation pattern per batch:
    [D micro-step × d_accum]  →  D.optimizer_step()
    [G micro-step × g_accum]  →  G.optimizer_step()  (conditional)
* Two independent MultiGPUWrapper instances — one for G, one for D.
  ZeRO-1 sharding applies independently to each.

Run:
    python imagenet_gan_v2.py config_gan.json
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torchvision.io import write_jpeg
from torchvision.utils import make_grid
from tqdm import tqdm

from ramtorch.multi_gpu import MultiGPUWrapper

from src.models.flow import image_flatten, image_unflatten
from src.models.gan import (
    GANGenerator,
    GANDiscriminator,
    d_relativistic_loss,
    g_relativistic_loss,
)
from src.dataloaders.parquet_dataloader import ParquetTextImageDataset

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Replay buffer (unchanged from original)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Stores a pool of past fake samples; randomly replays some of them to D.

    This decorrelates the fake distribution seen by D from the current G,
    stabilising training (CycleGAN / pix2pix trick).
    """

    def __init__(self, max_size: int = 50, prob: float = 0.5):
        # max_size=0 disables the buffer entirely (pass-through, simple GAN mode).
        assert max_size >= 0
        self.max_size = max_size
        self.buffer: list[torch.Tensor] = []
        self.prob = prob

    @torch.no_grad()
    def push_and_pop(self, data: torch.Tensor) -> torch.Tensor:
        """data: [B, seq, dim].  Returns same shape, mixing old and new.

        When max_size=0 (disabled) the tensor is returned as-is with no
        buffering, giving vanilla GAN behaviour.
        """
        if self.max_size == 0:
            return data
        to_return = []
        for element in data:
            element = element.unsqueeze(0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(element.cpu())
                to_return.append(element)
            else:
                if torch.rand(1).item() > self.prob:
                    i = torch.randint(0, self.max_size, (1,)).item()
                    to_return.append(self.buffer[i].to(element.device))
                    self.buffer[i] = element.cpu()
                else:
                    to_return.append(element)
        return torch.cat(to_return, dim=0)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _strip_compiled_keys(sd: dict) -> dict:
    prefix = "_orig_mod."
    return {k.replace(prefix, "") if prefix in k else k: v for k, v in sd.items()}


def save_checkpoint(path: str, g_wrapper: MultiGPUWrapper, d_wrapper: MultiGPUWrapper,
                    noise_lerp_val: float, prev_d_loss: float, global_step: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "G": _strip_compiled_keys(g_wrapper.models[0].state_dict()),
        "D": _strip_compiled_keys(d_wrapper.models[0].state_dict()),
        "noise_lerp_val": noise_lerp_val,
        "prev_d_loss": prev_d_loss,
        "global_step": global_step,
    }, path)
    print(f"[ckpt] Saved → {path}")


def load_checkpoint(path: str, g_wrapper: MultiGPUWrapper, d_wrapper: MultiGPUWrapper):
    if not os.path.exists(path):
        print(f"[ckpt] No checkpoint at {path} — starting fresh.")
        return 0, 1.0, 0.693147  # (global_step, noise_lerp_val, prev_d_loss)
    ckpt = torch.load(path, map_location="cpu")
    n_gpus = len(g_wrapper.models)
    for gpu_id in range(n_gpus):
        g_wrapper.models[gpu_id].load_state_dict(ckpt["G"], strict=False)
        d_wrapper.models[gpu_id].load_state_dict(ckpt["D"], strict=False)
    print(f"[ckpt] Loaded ← {path} (step {ckpt.get('global_step', 0)})")
    return (
        ckpt.get("global_step", 0),
        ckpt.get("noise_lerp_val", 1.0),
        ckpt.get("prev_d_loss", 0.693147),
    )


# ---------------------------------------------------------------------------
# Per-GPU forward functions
# ---------------------------------------------------------------------------

def g_forward_fn(
    gpu_id: int,
    g_model: GANGenerator,
    noise: torch.Tensor,
    text_tokens: torch.Tensor | None,
) -> tuple[torch.Tensor, tuple]:
    """Generate fake patches from noise on one GPU.

    Returns:
        (fake_patches [B, seq, output_dim], image_shape (B,C,H,W))
    """
    device = f"cuda:{gpu_id}"
    noise = noise.to(device, non_blocking=True)
    if text_tokens is not None:
        text_tokens = text_tokens.to(device, non_blocking=True)

    image_shape = noise.shape  # (B, C, H, W)

    with torch.autocast("cuda", torch.bfloat16):
        fake_patches = g_model(noise, text_tokens)

    return fake_patches, image_shape


def d_forward_fn(
    gpu_id: int,
    d_model: GANDiscriminator,
    real_patches: torch.Tensor,
    fake_patches_detached: torch.Tensor,
    cls_label: torch.Tensor,
    image_shape: tuple,
    text_tokens: torch.Tensor | None,
    noise_lerp_val: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run D on real and (replayed) fake patches.

    Applies adaptive noise augmentation: lerp inputs towards pure Gaussian
    noise by noise_lerp_val, giving D a tunable signal/noise tradeoff that
    prevents vanishing gradients for G when D is overwhelmingly accurate.

    Returns:
        (d_real_embed [B, disc_embed_dim], d_fake_embed [B, disc_embed_dim])
    """
    device = f"cuda:{gpu_id}"
    real_patches = real_patches.to(device, non_blocking=True)
    fake_patches_detached = fake_patches_detached.to(device, non_blocking=True)
    cls_label = cls_label.to(device, non_blocking=True)
    if text_tokens is not None:
        text_tokens = text_tokens.to(device, non_blocking=True)

    # Adaptive noise augmentation
    real_noisy = torch.lerp(real_patches, torch.randn_like(real_patches), noise_lerp_val)
    fake_noisy = torch.lerp(fake_patches_detached, torch.randn_like(fake_patches_detached), noise_lerp_val)

    with torch.autocast("cuda", torch.bfloat16):
        d_real = d_model(real_noisy, cls_label, image_shape, text_tokens, noise_level=noise_lerp_val)
        d_fake = d_model(fake_noisy, cls_label, image_shape, text_tokens, noise_level=noise_lerp_val)

    return d_real, d_fake


# ---------------------------------------------------------------------------
# Per-GPU backward functions
# ---------------------------------------------------------------------------

def d_backward_fn(
    gpu_id: int,
    d_model: GANDiscriminator,
    output: tuple[torch.Tensor, torch.Tensor],
    disc_embed_dim: int,
    accum_steps: int,
) -> float:
    """Compute relativistic D loss and call .backward().

    Returns scalar D loss (Python float).
    """
    d_real, d_fake = output
    loss = d_relativistic_loss(d_real.float(), d_fake.float(), disc_embed_dim)
    (loss / accum_steps).backward()
    return loss.item()


def g_backward_fn(
    gpu_id: int,
    g_model: GANGenerator,
    d_model: GANDiscriminator,
    noise: torch.Tensor,
    real_patches_detached: torch.Tensor,
    cls_label: torch.Tensor,
    image_shape: tuple,
    text_tokens: torch.Tensor | None,
    noise_lerp_val: float,
    disc_embed_dim: int,
    accum_steps: int,
) -> float:
    """Generate fresh fakes, run D on them (with grad), compute G loss, backward.

    D weights must be frozen (no_grad or optimizer zero_grad before this).
    Returns scalar G loss (Python float).
    """
    device = f"cuda:{gpu_id}"
    noise = noise.to(device, non_blocking=True)
    real_patches_detached = real_patches_detached.to(device, non_blocking=True)
    cls_label = cls_label.to(device, non_blocking=True)
    if text_tokens is not None:
        text_tokens = text_tokens.to(device, non_blocking=True)

    with torch.autocast("cuda", torch.bfloat16):
        # Fresh fakes — with grad flowing through G
        fake_patches = g_model(noise, text_tokens)

        # Adaptive noise on both
        real_noisy  = torch.lerp(real_patches_detached, torch.randn_like(real_patches_detached), noise_lerp_val)
        fake_noisy  = torch.lerp(fake_patches, torch.randn_like(fake_patches), noise_lerp_val)

        # D on real (no grad into D; D params detached via no_grad for D optimizer)
        d_real = d_model(real_noisy.detach(), cls_label, image_shape, text_tokens,
                         noise_level=noise_lerp_val)
        d_fake = d_model(fake_noisy, cls_label, image_shape, text_tokens,
                         noise_level=noise_lerp_val)

    loss = g_relativistic_loss(d_fake.float(), d_real.detach().float(), disc_embed_dim)
    (loss / accum_steps).backward()
    return loss.item()


# ---------------------------------------------------------------------------
# Preview helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def preview_fn(
    gpu_id: int,
    g_model: GANGenerator,
    noise: torch.Tensor,
    text_tokens: torch.Tensor | None,
    n_samples: int,
    patch_size: int,
) -> torch.Tensor:
    """Run G on fixed noise to produce preview images (no grad).

    Returns:
        [n_samples, C, H, W] CPU float in [-1, 1].
    """
    device = f"cuda:{gpu_id}"
    noise = noise[:n_samples].to(device, non_blocking=True)
    B, C, H, W = noise.shape
    if text_tokens is not None:
        text_tokens = text_tokens[:n_samples].to(device, non_blocking=True)

    with torch.autocast("cuda", torch.bfloat16):
        fake_patches = g_model(noise, text_tokens)  # [B, seq, output_dim]

    # Unflatten patches back to image space
    fake_imgs = image_unflatten(fake_patches, (B, C, H, W), shuffle_size=patch_size)
    return fake_imgs.clamp(-1, 1).cpu().float()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict):
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPU(s)")

    # ------------------------------------------------------------------
    # Directories + config copy
    # ------------------------------------------------------------------
    os.makedirs(cfg["ckpt_path"], exist_ok=True)
    os.makedirs(cfg["preview_path"], exist_ok=True)
    _cfg_dest = os.path.join(cfg["ckpt_path"], os.path.basename(CONFIG_PATH))
    shutil.copy(CONFIG_PATH, _cfg_dest)
    print(f"Config saved to {_cfg_dest}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    parquet_cfg = cfg.get("parquet_dataloader")
    if not parquet_cfg:
        raise RuntimeError("No 'parquet_dataloader' config found.")

    dataset = ParquetTextImageDataset(
        batch_size=cfg["batch_size"] * n_gpus,
        parquet_sources=parquet_cfg["parquet_sources"],
        caption_columns=parquet_cfg["caption_columns"],
        filename_column=parquet_cfg.get("filename_column", "url"),
        width_column=parquet_cfg.get("width_column", "image_width"),
        height_column=parquet_cfg.get("height_column", "image_height"),
        loss_weight_column=parquet_cfg.get("loss_weight_column", None),
        image_folder_path=parquet_cfg.get("image_folder_path", ""),
        base_res=parquet_cfg.get("base_resolution", [256]),
        ratio_cutoff=parquet_cfg.get("ratio_cutoff", 2.0),
        resolution_step=parquet_cfg.get("resolution_step", 64),
        shuffle_tags=parquet_cfg.get("shuffle_tags", True),
        tag_drop_percentage=parquet_cfg.get("tag_drop_percentage", 0.0),
        uncond_percentage=0.0,
        seed=cfg.get("seed", 42),
        rank=0,
        num_gpus=1,
        offset=parquet_cfg.get("offset", 0),
        tokenizer=None,
        max_text_len=1,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=parquet_cfg.get("num_workers", 4),
        prefetch_factor=parquet_cfg.get("prefetch_factor", 2),
        pin_memory=True,
        collate_fn=dataset.dummy_collate_fn,
    )

    # ------------------------------------------------------------------
    # Models + MultiGPUWrapper (separate wrappers for G and D)
    # ------------------------------------------------------------------
    g_cfg = cfg["g_model_config"]
    d_cfg = cfg["d_model_config"]

    def g_factory():
        return GANGenerator(**g_cfg)

    def d_factory():
        return GANDiscriminator(**d_cfg)

    lr_g = cfg.get("lr_g", 1e-4)
    lr_d = cfg.get("lr_d", 1e-4)
    warmup = cfg.get("warmup", 1000)

    g_wrapper = MultiGPUWrapper(
        model_factory=g_factory,
        optimizer_factory=lambda params: AdamW(params, lr=lr_g, weight_decay=1e-4, betas=(0.0, 0.99)),
        gradient_accumulation_steps=cfg.get("g_accum", 3),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        scheduler_factory=lambda opt: LinearLR(opt, start_factor=1e-5, end_factor=1.0, total_iters=warmup),
    )
    d_wrapper = MultiGPUWrapper(
        model_factory=d_factory,
        optimizer_factory=lambda params: AdamW(params, lr=lr_d, weight_decay=1e-4, betas=(0.0, 0.99)),
        gradient_accumulation_steps=cfg.get("d_accum", 3),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        scheduler_factory=lambda opt: LinearLR(opt, start_factor=1e-5, end_factor=1.0, total_iters=warmup),
    )
    g_wrapper.setup()
    d_wrapper.setup()

    # Strip torch.compile keys on save
    def _make_stripped_save(wrapper: MultiGPUWrapper):
        def _save(path: str):
            sd = _strip_compiled_keys(wrapper.models[0].state_dict())
            torch.save(sd, path)
            print(f"[MultiGPUWrapper] Saved: {path}")
        wrapper.save_checkpoint = _save

    _make_stripped_save(g_wrapper)
    _make_stripped_save(d_wrapper)

    # Save initial untrained checkpoints
    g_wrapper.save_checkpoint(os.path.join(cfg["ckpt_path"], "G_untrained.pth"))
    d_wrapper.save_checkpoint(os.path.join(cfg["ckpt_path"], "D_untrained.pth"))

    # Optional block compilation (after setup to avoid FX tracing issues)
    if g_cfg.get("compile_blocks", False):
        print("Compiling G blocks...")
        for m in g_wrapper.models:
            m.compile_blocks()
    if d_cfg.get("compile_blocks", False):
        print("Compiling D blocks...")
        for m in d_wrapper.models:
            m.compile_blocks()

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------
    replay_buffer = ReplayBuffer(
        max_size=cfg.get("replay_buffer_size", 100),
        prob=cfg.get("replay_buffer_prob", 0.5),
    )

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------
    disc_embed_dim  = d_cfg["disc_embed_dim"]
    d_accum         = cfg.get("d_accum", 3)
    g_accum         = cfg.get("g_accum", 3)
    target_d_loss   = cfg.get("target_d_loss", 0.693147)  # ln(2) = optimal D
    noise_ema_decay = cfg.get("noise_ema_decay", 0.95)
    patch_size      = g_cfg.get("patch_size", 16)
    preview_spg     = cfg.get("preview_samples_per_gpu", 4)
    preview_qual    = cfg.get("preview_quality", 95)
    eval_interval   = cfg.get("eval_interval", 500)
    save_every      = cfg.get("save_every_n_steps", 2000)
    log_every       = cfg.get("log_every_n_steps", 10)

    global_step     = 0
    # When noise_ema_decay=0.0 (augmentation disabled), keep noise_lerp_val
    # frozen at 0.0 so D sees clean inputs throughout training.
    noise_lerp_val  = 0.0 if noise_ema_decay == 0.0 else 1.0
    prev_d_loss     = target_d_loss

    if cfg.get("resume_checkpoint"):
        global_step, noise_lerp_val, prev_d_loss = load_checkpoint(
            cfg["resume_checkpoint"], g_wrapper, d_wrapper
        )

    # ------------------------------------------------------------------
    # CSV loss log
    # ------------------------------------------------------------------
    csv_path = os.path.join(cfg["ckpt_path"], "loss_log.csv")
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        csv_writer.writerow(["step", "d_loss", "g_loss", "noise_lerp", "lr_g", "lr_d", "time"])
    t0 = time.time()

    master_seed = cfg.get("seed", 42)
    torch.manual_seed(master_seed)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    epoch = 0
    while True:
        epoch += 1
        torch.manual_seed(master_seed + epoch)
        for m in g_wrapper.models:
            m.train()
        for m in d_wrapper.models:
            m.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_data in pbar:
            batch_data = batch_data[0]  # dummy_collate_fn wraps in extra list
            images, captions, _idx, _loss_weights = batch_data[:4]

            # images: [B*n_gpus, C, H, W] CPU
            spg = images.shape[0] // n_gpus
            if spg == 0:
                continue

            image_chunks = [images[i * spg:(i + 1) * spg] for i in range(n_gpus)]
            # Class labels: use dummy zeros if no class info in dataset
            # (replace with real label source when available)
            cls_chunks = [
                torch.zeros(spg, dtype=torch.long)
                for _ in range(n_gpus)
            ]

            # text tokens: currently disabled (use_text_embed=False by default)
            text_chunks = [None] * n_gpus

            # ----------------------------------------------------------
            # Precompute noise chunks (shared across D and G phases)
            # ----------------------------------------------------------
            noise_chunks = [
                torch.randn_like(image_chunks[g]) for g in range(n_gpus)
            ]

            # ===========================================================
            # D PHASE — d_accum micro-steps
            # ===========================================================
            d_loss_accum = 0.0

            for d_step in range(d_accum):
                # 1. Generate fake patches (no G grad needed here)
                with torch.no_grad():
                    g_outputs = g_wrapper.forward(
                        [(noise_chunks[g], text_chunks[g]) for g in range(n_gpus)],
                        forward_fn=g_forward_fn,
                        eval_mode=False,  # keep BN/etc in train mode
                    )

                # g_outputs[g] = (fake_patches [B, seq, dim], image_shape)
                # Replay buffer operates on CPU; push/pop per GPU concatenated
                fake_patches_per_gpu = []
                for g in range(n_gpus):
                    fake_p, _ = g_outputs[g]
                    replayed = replay_buffer.push_and_pop(fake_p.detach().cpu())
                    fake_patches_per_gpu.append(replayed)

                # 2. Flatten real images → patches (on CPU, moved to GPU in d_forward_fn)
                real_patches_per_gpu = []
                for g in range(n_gpus):
                    img = image_chunks[g]
                    patches, _ = image_flatten(img, shuffle_size=patch_size)
                    real_patches_per_gpu.append(patches)

                image_shapes = [image_chunks[g].shape for g in range(n_gpus)]

                # 3. D forward
                d_outputs = d_wrapper.forward(
                    [
                        (
                            real_patches_per_gpu[g],
                            fake_patches_per_gpu[g],
                            cls_chunks[g],
                            image_shapes[g],
                            text_chunks[g],
                            noise_lerp_val,
                        )
                        for g in range(n_gpus)
                    ],
                    forward_fn=d_forward_fn,
                )

                # 4. D backward
                d_results = d_wrapper.run_concurrent(
                    lambda gpu_id: d_backward_fn(
                        gpu_id,
                        d_wrapper.models[gpu_id],
                        d_outputs[gpu_id],
                        disc_embed_dim,
                        d_accum,
                    )
                )
                d_loss_accum += sum(d_results) / n_gpus

            # D optimizer step
            d_wrapper.reduce_grads()
            d_wrapper.clip_grads()
            d_wrapper.optimizer_step()

            avg_d_loss = d_loss_accum / d_accum

            # Update adaptive noise EMA
            # noise_ema_decay=0.0 disables augmentation: noise_lerp_val stays 0.0.
            if noise_ema_decay > 0.0:
                raw_scale = max(0.0, min(1.0, 1.0 - prev_d_loss / target_d_loss))
                noise_lerp_val = noise_ema_decay * noise_lerp_val + (1.0 - noise_ema_decay) * raw_scale
            prev_d_loss = avg_d_loss

            # ===========================================================
            # G PHASE — g_accum micro-steps (only when D is strong enough)
            # ===========================================================
            g_loss_accum = 0.0
            g_updated = False

            if avg_d_loss < target_d_loss:
                g_updated = True

                # Pre-flatten real patches for G's D call (detached)
                real_patches_per_gpu_detached = []
                for g in range(n_gpus):
                    img = image_chunks[g]
                    patches, _ = image_flatten(img, shuffle_size=patch_size)
                    real_patches_per_gpu_detached.append(patches.detach())

                image_shapes = [image_chunks[g].shape for g in range(n_gpus)]

                for _g_step in range(g_accum):
                    # Resample noise each micro-step so G sees varied inputs
                    noise_chunks = [torch.randn_like(image_chunks[g]) for g in range(n_gpus)]

                    g_results = g_wrapper.run_concurrent(
                        lambda gpu_id: g_backward_fn(
                            gpu_id,
                            g_wrapper.models[gpu_id],
                            d_wrapper.models[gpu_id],
                            noise_chunks[gpu_id],
                            real_patches_per_gpu_detached[gpu_id],
                            cls_chunks[gpu_id],
                            image_shapes[gpu_id],
                            text_chunks[gpu_id],
                            noise_lerp_val,
                            disc_embed_dim,
                            g_accum,
                        )
                    )
                    g_loss_accum += sum(g_results) / n_gpus

                # G optimizer step
                g_wrapper.reduce_grads()
                g_wrapper.clip_grads()
                g_wrapper.optimizer_step()

            avg_g_loss = g_loss_accum / g_accum if g_updated else 0.0

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------
            lr_g_val = g_wrapper.last_lr
            lr_d_val = d_wrapper.last_lr
            pbar.set_postfix(
                D=f"{avg_d_loss:.4f}",
                G=f"{avg_g_loss:.4f}",
                noise=f"{noise_lerp_val:.3f}",
                lr_g=f"{lr_g_val:.2e}",
                step=global_step,
            )

            csv_writer.writerow([
                global_step,
                f"{avg_d_loss:.6f}",
                f"{avg_g_loss:.6f}",
                f"{noise_lerp_val:.6f}",
                f"{lr_g_val:.2e}",
                f"{lr_d_val:.2e}",
                f"{time.time() - t0:.1f}",
            ])
            if global_step % log_every == 0:
                csv_file.flush()

            # ----------------------------------------------------------
            # Preview
            # ----------------------------------------------------------
            if global_step % eval_interval == 0:
                for m in g_wrapper.models:
                    m.eval()

                fixed_noise = [
                    torch.randn_like(image_chunks[g]) for g in range(n_gpus)
                ]
                preview_results = g_wrapper.forward(
                    [(fixed_noise[g], text_chunks[g]) for g in range(n_gpus)],
                    forward_fn=lambda gpu_id, model, noise, text: preview_fn(
                        gpu_id, model, noise, text,
                        n_samples=min(preview_spg, noise.shape[0]),
                        patch_size=patch_size,
                    ),
                    eval_mode=True,
                )
                # preview_fn returns CPU tensors
                all_fakes = torch.cat(preview_results, dim=0)  # [N, C, H, W] in [-1,1]
                grid = make_grid((all_fakes + 1) / 2, nrow=preview_spg)
                img_path = os.path.join(cfg["preview_path"], f"step_{global_step}.jpg")
                grid_uint8 = (grid.clamp(0, 1) * 255).to(torch.uint8)
                write_jpeg(grid_uint8, img_path, quality=preview_qual)

                for m in g_wrapper.models:
                    m.train()

            # ----------------------------------------------------------
            # Step checkpoint
            # ----------------------------------------------------------
            if save_every > 0 and global_step > 0 and global_step % save_every == 0:
                save_checkpoint(
                    os.path.join(cfg["ckpt_path"], f"step_{global_step}.pth"),
                    g_wrapper, d_wrapper, noise_lerp_val, prev_d_loss, global_step,
                )

            global_step += 1

        # --- End of epoch ---
        save_checkpoint(
            os.path.join(cfg["ckpt_path"], f"epoch_{epoch}.pth"),
            g_wrapper, d_wrapper, noise_lerp_val, prev_d_loss, global_step,
        )
        dataset.resample()

    csv_file.close()
    g_wrapper.cleanup()
    d_wrapper.cleanup()
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config_gan.json"
    print(f"Loading config from: {CONFIG_PATH}")
    TRAINING_CONFIG = load_config(CONFIG_PATH)
    train(TRAINING_CONFIG)
