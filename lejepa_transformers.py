"""lejepa_transformers.py — LeJEPA self-supervised pre-training.

Implements LeJEPA (Balestriero & LeCun, arXiv:2511.08544) on top of the
existing bucketed parquet dataloader and MultiGPUWrapper infrastructure.

Key differences from pixel_space_transformers.py:
  - No flow-matching, no timestep, no noise, no DINO auxiliary loss.
  - V augmented views per image; SIGReg + invariance loss (LeJEPA objective).
  - PCA of patch token features saved as qualitative eval every eval_interval.
  - Token layout: [CLS, registers, image_patches] — CLS output = embedding.

Run:
    python lejepa_transformers.py
    python lejepa_transformers.py config_lejepa.json

All settings are in config_lejepa.json.
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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader
from torchvision.io import write_jpeg
from torchvision.utils import make_grid
from tqdm import tqdm

from ramtorch.multi_gpu import MultiGPUWrapper

from src.models.lejepa import LeJEPAEncoder, SIGReg
from src.dataloaders.parquet_dataloader import ParquetTextImageDataset

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Profiler helper
# ---------------------------------------------------------------------------

def make_profiler_ctx(cfg: dict, trace_path: str):
    if not cfg.get("profile", False):
        return nullcontext()

    start = cfg.get("profile_start", 20)
    stop  = cfg.get("profile_stop", 23)
    if stop <= start:
        raise ValueError(f"profile_stop ({stop}) must be > profile_start ({start})")

    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=start, warmup=1, active=stop - start, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=lambda p: (
            print(p.key_averages().table(sort_by="cuda_time_total", row_limit=20)),
            p.export_chrome_trace(trace_path),
            print(f"[profiler] Chrome trace saved to {trace_path}"),
        ),
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _strip_compiled_keys(sd: dict) -> dict:
    """Remove the ``_orig_mod.`` prefix added by torch.compile."""
    prefix = "_orig_mod."
    return {
        k.replace(prefix, "") if prefix in k else k: v
        for k, v in sd.items()
    }


# ---------------------------------------------------------------------------
# GPU-side augmentation pipeline
# ---------------------------------------------------------------------------

def build_aug_pipeline(cfg: dict) -> nn.Module:
    """Build a torchvision v2 augmentation pipeline that runs on GPU.

    Images arrive from the dataloader as float32 tensors in [-1, 1].
    The pipeline converts them to [0, 1] for augmentation, then back to [-1, 1].

    RandomResizedCrop is applied *per-image* using the image's own spatial size
    so it works correctly across all bucket resolutions without resizing.
    """
    scale_min   = cfg.get("aug_scale_min", 0.08)
    scale_max   = cfg.get("aug_scale_max", 1.0)
    cj_strength = cfg.get("aug_color_jitter_strength", 0.8)
    blur_p      = cfg.get("aug_blur_prob", 0.5)
    solar_p     = cfg.get("aug_solarize_prob", 0.2)

    # Colour / noise transforms that don't change spatial size — applied to the
    # whole batch at once.
    colour_transforms = v2.Compose([
        v2.RandomApply([v2.ColorJitter(
            brightness=cj_strength,
            contrast=cj_strength,
            saturation=cj_strength,
            hue=cj_strength * 0.25,
        )], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=blur_p),
        v2.RandomApply([v2.RandomSolarize(threshold=0.5)], p=solar_p),
        v2.RandomHorizontalFlip(),
    ])

    def augment(x: torch.Tensor) -> torch.Tensor:
        """Augment a batch [B, C, H, W] in [-1, 1], return same shape in [-1, 1]."""
        # [-1,1] → [0,1]
        x = (x + 1.0) / 2.0
        # Per-image RandomResizedCrop that preserves the original H×W
        H, W = x.shape[-2], x.shape[-1]
        x = v2.RandomResizedCrop(size=(H, W), scale=(scale_min, scale_max),
                                 antialias=True)(x)
        # Colour augmentations (batch-level is fine for these)
        x = colour_transforms(x)
        # [0,1] → [-1,1]
        return x * 2.0 - 1.0

    return augment


# ---------------------------------------------------------------------------
# Per-GPU callables
# ---------------------------------------------------------------------------

def forward_fn(
    gpu_id: int,
    model: LeJEPAEncoder,
    real: torch.Tensor,
    V: int,
    aug_pipeline,
) -> tuple:
    """Forward pass: augment V times on GPU, encode all V*B views.

    Args:
        real:         [B, C, H, W] CPU tensor from the dataloader.
        V:            Number of augmented views per image.
        aug_pipeline: torchvision v2 transform (applied on GPU).

    Returns:
        (emb [V*B, dim], proj [V, B, proj_dim], real_gpu [B, C, H, W])
    """
    device = f"cuda:{gpu_id}"
    real_gpu = real.to(device, non_blocking=True)   # [B, C, H, W]
    B = real_gpu.shape[0]

    # Apply V independent augmentations on GPU, stack → [V*B, C, H, W]
    # aug_pipeline is a plain callable; each call samples fresh random params.
    views = torch.cat([aug_pipeline(real_gpu) for _ in range(V)], dim=0)

    with torch.autocast("cuda", torch.bfloat16):
        emb, proj = model(views)   # emb: [V*B, dim], proj: [V*B, proj_dim]

    proj = proj.reshape(V, B, -1)  # [V, B, proj_dim]
    return emb, proj, real_gpu


def backward_fn(
    gpu_id: int,
    model: LeJEPAEncoder,
    output: tuple,
    sigreg: SIGReg,
    lamb: float,
    accum_steps: int = 1,
) -> tuple[float, float, float]:
    """Backward pass: compute LeJEPA loss and call .backward().

    LeJEPA loss = (1 - λ) * inv_loss + λ * sigreg_loss

    inv_loss:    invariance — each view should match the mean view.
    sigreg_loss: SIGReg — pushes the distribution of projections towards
                 a standard Gaussian (prevents collapse without stop-grad).

    Returns:
        (lejepa_loss, inv_loss, sigreg_loss)  — Python floats
    """
    emb, proj, _ = output
    # proj: [V, B, proj_dim]

    # Invariance: each view should predict the mean of all views
    inv_loss = (proj.mean(0, keepdim=True) - proj).square().mean()

    # SIGReg: characteristic-function regularizer
    sigreg_loss = sigreg(proj)

    lejepa_loss = (1.0 - lamb) * inv_loss + lamb * sigreg_loss

    (lejepa_loss / accum_steps).backward()
    return lejepa_loss.item(), inv_loss.item(), sigreg_loss.item()


# ---------------------------------------------------------------------------
# PCA preview
# ---------------------------------------------------------------------------

@torch.no_grad()
def pca_preview_fn(
    model: LeJEPAEncoder,
    real: torch.Tensor,
    n_samples: int,
    patch_size: int,
    device: str,
) -> torch.Tensor:
    """Compute PCA RGB visualisation of patch token features.

    Projects the top-3 principal components of the patch token features onto
    RGB channels, giving a qualitative view of the learned spatial semantics.
    As training progresses, semantically coherent regions should cluster into
    distinct colours.

    Args:
        model:      LeJEPAEncoder (eval mode, on ``device``).
        real:       [B, C, H, W] CPU tensor (raw dataloader images, [-1, 1]).
        n_samples:  Number of images to visualise.
        patch_size: Patch size in pixels (for upsampling the PCA map).
        device:     CUDA device string.

    Returns:
        grid: float32 CPU tensor [3, H_grid, W_grid] in [0, 1], ready for
              write_jpeg / save_image.  Layout: real images on top row,
              PCA RGB on bottom row.
    """
    x = real[:n_samples].to(device)   # [B, C, H, W]
    B, C, H, W = x.shape
    num_h = H // patch_size
    num_w = W // patch_size

    with torch.autocast("cuda", torch.bfloat16):
        patch_tokens = model.encode_patches(x)   # [B, H*W, dim]

    patch_tokens = patch_tokens.float()          # [B, H*W, dim]

    # --- PCA via truncated SVD ---
    flat = patch_tokens.reshape(B * num_h * num_w, -1)   # [B*H*W, dim]
    flat = flat - flat.mean(0, keepdim=True)              # centre

    # torch.pca_lowrank returns (U, S, V); columns of V are principal directions
    _, _, Vt = torch.pca_lowrank(flat, q=3, niter=4)     # Vt: [dim, 3]
    pca_flat = flat @ Vt                                  # [B*H*W, 3]

    pca = pca_flat.reshape(B, num_h, num_w, 3)           # [B, H, W, 3]

    # Per-image, per-channel min-max normalise → [0, 1]
    pca_min = pca.reshape(B, -1, 3).min(1, keepdim=True).values.unsqueeze(1)  # [B,1,1,3]
    pca_max = pca.reshape(B, -1, 3).max(1, keepdim=True).values.unsqueeze(1)  # [B,1,1,3]
    pca = (pca - pca_min) / (pca_max - pca_min + 1e-8)   # [B, H, W, 3]

    pca = pca.permute(0, 3, 1, 2)                         # [B, 3, H, W]

    # Upsample to pixel space
    pca_up = F.interpolate(pca, scale_factor=patch_size, mode="nearest")  # [B, 3, H*p, W*p]

    # Real images normalised to [0, 1]
    real_vis = (x.clamp(-1, 1) + 1.0) / 2.0              # [B, 3, H*p, W*p]

    # Interleave: [real_0, pca_0, real_1, pca_1, ...]
    pairs = []
    for i in range(B):
        pairs.append(real_vis[i].cpu().float())
        pairs.append(pca_up[i].cpu().float())
    grid = make_grid(torch.stack(pairs), nrow=B)           # [3, H_grid, W_grid]
    return grid


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

    print("Using ParquetTextImageDataset")
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
        shuffle_tags=parquet_cfg.get("shuffle_tags", False),
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
    # Model + MultiGPUWrapper
    # ------------------------------------------------------------------
    V        = cfg.get("V", 4)
    lamb     = cfg.get("lamb", 0.02)
    proj_dim = cfg["model_config"].get("proj_dim", 128)

    def model_factory():
        return LeJEPAEncoder(**cfg["model_config"])

    wrapper = MultiGPUWrapper(
        model_factory=model_factory,
        optimizer_factory=lambda params: AdamW(
            params, lr=cfg["lr"], weight_decay=1e-4, betas=(0.9, 0.95)
        ),
        gradient_accumulation_steps=cfg["accum"],
        max_grad_norm=1.0,
        scheduler_factory=lambda opt: LinearLR(
            opt, start_factor=1e-5, end_factor=1.0, total_iters=cfg["warmup"]
        ),
    )
    wrapper.setup()

    # Patch save_checkpoint to strip torch.compile's _orig_mod. key prefix
    def _save_checkpoint_stripped(path: str):
        sd = wrapper.models[0].state_dict()
        sd = _strip_compiled_keys(sd)
        from safetensors.torch import save_file as _save_file
        if path.endswith((".safetensors", ".sft")):
            _save_file(sd, path)
        else:
            torch.save(sd, path)
        print(f"[MultiGPUWrapper] Saved: {path}")
    wrapper.save_checkpoint = _save_checkpoint_stripped

    if cfg.get("model_checkpoint"):
        wrapper.load_checkpoint(cfg["model_checkpoint"])
    else:
        wrapper.save_checkpoint(os.path.join(cfg["ckpt_path"], "untrained.safetensors"))

    if cfg["model_config"].get("compile_blocks", False):
        print(f"Compiling transformer blocks on {n_gpus} GPU(s)...")
        for m in wrapper.models:
            m.compile_blocks()
        print("  Done.")

    # ------------------------------------------------------------------
    # SIGReg (one per GPU, same weights — no learnable params)
    # ------------------------------------------------------------------
    sigreg_modules = [
        SIGReg(
            knots=cfg.get("sigreg_knots", 17),
            t_max=cfg.get("sigreg_t_max", 3.0),
            n_proj=cfg.get("sigreg_n_proj", 256),
        ).to(f"cuda:{g}")
        for g in range(n_gpus)
    ]

    # ------------------------------------------------------------------
    # Augmentation pipeline (built once, applied on GPU)
    # ------------------------------------------------------------------
    aug_pipeline = build_aug_pipeline(cfg)

    # ------------------------------------------------------------------
    # CSV loss log
    # ------------------------------------------------------------------
    log_every = cfg.get("log_every_n_steps", 10)
    csv_path  = os.path.join(cfg["ckpt_path"], "loss_log.csv")
    csv_file  = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        csv_writer.writerow(["step", "lejepa_loss", "inv_loss", "sigreg_loss", "lr", "time"])
    t0 = time.time()

    global_step = cfg.get("initial_global_step", 0)
    master_seed = cfg.get("seed", 42)
    torch.manual_seed(master_seed)

    with make_profiler_ctx(cfg, "lejepa_trace.json") as prof:
        epoch = 0
        while True:
            epoch += 1
            torch.manual_seed(master_seed + epoch)
            for m in wrapper.models:
                m.train()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

            for batch_idx, batch_data in enumerate(pbar):
                batch_data = batch_data[0]   # dummy_collate_fn wraps in extra list
                images, captions, _idx, loss_weights = batch_data[:4]

                # Split across GPUs
                spg = images.shape[0] // n_gpus
                image_chunks = [images[i * spg:(i + 1) * spg] for i in range(n_gpus)]

                # 1. Forward — chunks are 1-tuples so wrapper unpacks real as the sole positional arg
                with record_function("forward"):
                    outputs = wrapper.forward(
                        [(chunk,) for chunk in image_chunks],
                        forward_fn=forward_fn,
                        V=V,
                        aug_pipeline=aug_pipeline,
                    )

                # 2. Backward — wrapper.backward passes (gpu_id, model, output, **kwargs)
                # SIGReg is per-GPU so we use run_concurrent to pass the right module.
                with record_function("backward"):
                    raw_results = wrapper.run_concurrent(
                        lambda gpu_id: backward_fn(
                            gpu_id, wrapper.models[gpu_id], outputs[gpu_id],
                            sigreg=sigreg_modules[gpu_id],
                            lamb=lamb,
                            accum_steps=cfg["accum"],
                        )
                    )

                lejepa_loss = sum(r[0] for r in raw_results) / n_gpus
                inv_loss    = sum(r[1] for r in raw_results) / n_gpus
                sigreg_loss = sum(r[2] for r in raw_results) / n_gpus

                # 3. Sync + step
                if (batch_idx + 1) % cfg["accum"] == 0:
                    with record_function("reduce_grads"):
                        wrapper.reduce_grads()
                    wrapper.clip_grads()
                    with record_function("optimizer_step"):
                        wrapper.optimizer_step()
                        torch.cuda.synchronize()
                        if cfg.get("empty_cache_on_step", False):
                            torch.cuda.empty_cache()

                lr = wrapper.last_lr
                pbar.set_postfix(
                    lejepa=f"{lejepa_loss:.4f}",
                    inv=f"{inv_loss:.4f}",
                    sig=f"{sigreg_loss:.4f}",
                    lr=f"{lr:.2e}",
                    step=global_step,
                )

                csv_writer.writerow([
                    global_step,
                    f"{lejepa_loss:.6f}",
                    f"{inv_loss:.6f}",
                    f"{sigreg_loss:.6f}",
                    f"{lr:.2e}",
                    f"{time.time() - t0:.1f}",
                ])
                if global_step % log_every == 0:
                    csv_file.flush()

                # --- Step checkpoint ---
                save_every = cfg.get("save_every_n_steps", 0)
                if save_every > 0 and global_step > 0 and global_step % save_every == 0:
                    wrapper.save_checkpoint(
                        os.path.join(cfg["ckpt_path"], f"step_{global_step}.safetensors")
                    )

                # --- PCA preview ---
                if global_step % cfg["eval_interval"] == 0:
                    preview_spg  = cfg.get("preview_samples_per_gpu", 4)
                    preview_qual = cfg.get("preview_quality", 95)

                    # Run PCA on GPU 0 only — use the real images from that GPU's batch
                    real_for_preview = outputs[0][2]   # real_gpu from GPU 0, [B, C, H, W]
                    model_0 = wrapper.models[0]
                    model_0.eval()
                    try:
                        grid = pca_preview_fn(
                            model=model_0,
                            real=real_for_preview.cpu(),
                            n_samples=min(preview_spg, real_for_preview.shape[0]),
                            patch_size=cfg["model_config"]["patch_size"],
                            device="cuda:0",
                        )
                        img_path = f"{cfg['preview_path']}/pca_step_{global_step}.jpg"
                        grid_uint8 = (grid.clamp(0, 1) * 255).to(torch.uint8)
                        write_jpeg(grid_uint8, img_path, quality=preview_qual)
                    finally:
                        model_0.train()

                global_step += 1

                if prof is not None:
                    prof.step()

            # --- End of epoch ---
            wrapper.save_checkpoint(
                os.path.join(cfg["ckpt_path"], f"epoch_{epoch}.safetensors")
            )
            dataset.resample()

    csv_file.close()
    wrapper.cleanup()
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError("pyyaml is required for YAML configs: pip install pyyaml") from e
        with open(path) as f:
            return yaml.safe_load(f)
    else:
        with open(path) as f:
            return json.load(f)


if __name__ == "__main__":
    CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config_lejepa.json"
    print(f"Loading config from: {CONFIG_PATH}")
    TRAINING_CONFIG = load_config(CONFIG_PATH)
    train(TRAINING_CONFIG)
