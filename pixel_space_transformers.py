"""pixel_space_transformers.py — Flow-matching trainer using MultiGPUWrapper.

Supports:
- Parquet-based multi-source image-text dataset
- Qwen tokenizer + learned token embeddings (text conditioning)
- DINOv3 embedding-similarity auxiliary loss
- M-RoPE positional encoding with jitter
- Per-block torch.compile
- Deferred CSV logging

Run:
    python pixel_space_transformers.py
    python pixel_space_transformers.py config.json

All settings are in config.json.
"""
from __future__ import annotations
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # adjust as needed

import copy
import csv
import json
import os
import shutil
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader
from torchvision.io import write_jpeg
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from torch.optim import AdamW
from ramtorch.multi_gpu import MultiGPUWrapper

from src.models.flow import (
    Flow,
    sample_from_distribution,
    create_distribution,
)
from src.dataloaders.parquet_dataloader import ParquetTextImageDataset

# Seed is set at train() entry from config; this is a fallback.
torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Optional: Qwen tokenizer
# ---------------------------------------------------------------------------
try:
    from transformers import AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    print("[warn] transformers not installed — text conditioning disabled")

# ---------------------------------------------------------------------------
# Optional: DINOv3 for embedding-similarity auxiliary loss
# ---------------------------------------------------------------------------
try:
    from transformers import AutoModel as HFAutoModel
    _DINO_AVAILABLE = True
except ImportError:
    _DINO_AVAILABLE = False
    print("[warn] transformers not installed — DINO loss disabled")


# ---------------------------------------------------------------------------
# Profiler helper (same as demo)
# ---------------------------------------------------------------------------


def make_profiler_ctx(cfg: dict, trace_path: str):
    if not cfg.get("profile", False):
        return nullcontext()

    start = cfg.get("profile_start", 20)
    stop = cfg.get("profile_stop", 23)
    if stop <= start:
        raise ValueError(
            f"profile_stop ({stop}) must be greater than profile_start ({start})"
        )

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
# Per-GPU callables (split fwd / bwd pattern)
# ---------------------------------------------------------------------------


def _tokenize_captions(tokenizer, captions: list[str], max_length: int, device: str) -> torch.Tensor:
    """Tokenize a list of captions → int64 token-id tensor [B, max_length]."""
    enc = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device)


def _compute_dino_loss(
    dino_model,
    pred_imgs: torch.Tensor,
    gt_imgs: torch.Tensor,
    timesteps: torch.Tensor,
    threshold: float,
    device: str,
) -> torch.Tensor:
    """DINOv3 embedding-similarity loss (MSE on pooler_output).
    Only applied to samples where t < threshold.
    pred_imgs / gt_imgs: [B, C, H, W] in [-1, 1].
    Returns per-sample loss [B] (zero for samples above threshold).
    """
    B = pred_imgs.shape[0]
    loss = torch.zeros(B, device=device)
    mask = timesteps < threshold
    if not mask.any():
        return loss

    pred_imgs = pred_imgs[mask].float().clamp(-1, 1)
    gt_imgs   = gt_imgs[mask].float().clamp(-1, 1)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    pred_n = ((pred_imgs + 1) / 2 - mean) / std
    gt_n   = ((gt_imgs   + 1) / 2 - mean) / std

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred_feat = dino_model(pixel_values=pred_n).pooler_output
        gt_feat   = dino_model(pixel_values=gt_n).pooler_output

    loss_masked = ((pred_feat - gt_feat) ** 2).mean(dim=-1)
    loss[mask] = loss_masked.to(loss.dtype)
    return loss


def forward_fn(
    gpu_id: int,
    model: Flow,
    real: torch.Tensor,
    captions: list[str],
    tokenizer,
    dino_model,
    dino_threshold: float,
    dino_strength: float,
    max_text_len: int,
    uncond_ratio: float,
    t_mu: float = 0.0,
) -> tuple:
    """Forward only — returns everything needed for backward.
    All tensors are in [B, C, H, W] image space; no patch flattening here.
    """
    device = f"cuda:{gpu_id}"
    x1 = real.to(device)          # [B, C, H, W]
    x0 = torch.randn_like(x1)

    B = x1.shape[0]
    x_dist, probabilities = create_distribution(1000, device=device, mu=t_mu)
    t = sample_from_distribution(x_dist, probabilities, B)[:, None, None, None].to(x1.dtype)

    noisy_image = x0 * t + x1 * (1 - t)

    # Unconditional dropout
    dropped = ["" if (torch.rand(1).item() < uncond_ratio) else c for c in captions]

    text_ids = None
    if tokenizer is not None:
        text_ids = _tokenize_captions(tokenizer, dropped, max_text_len, device)

    with torch.autocast("cuda", torch.bfloat16):
        predicted_image = model(noisy_image, t, text_tokens=text_ids)

    return predicted_image, noisy_image, t, x1, text_ids


def backward_fn(
    gpu_id: int,
    model: Flow,
    output: tuple,
    dino_model,
    dino_threshold: float,
    dino_strength: float,
    accum_steps: int = 1,
) -> tuple[float, float, float]:
    """Backward only — returns (total_loss, mse_loss, dino_loss)."""
    predicted_image, noisy_image, t, x1, text_ids = output
    device = predicted_image.device
    t_scalar = t.view(-1)

    target_velocity    = (noisy_image - x1) / (t + 5e-2)
    predicted_velocity = (noisy_image - predicted_image) / (t + 5e-2)

    mse = F.mse_loss(predicted_velocity, target_velocity)

    if dino_model is not None and dino_strength > 0.0:
        dino_per_sample = _compute_dino_loss(
            dino_model, predicted_image, x1,
            t_scalar, dino_threshold, str(device),
        )
        dino_term = dino_per_sample.mean() * dino_strength
        total = mse + dino_term
    else:
        dino_term = mse.new_zeros(1)
        total = mse

    (total / accum_steps).backward()
    return total.item(), mse.item(), dino_term.item()


def preview_fn(
    gpu_id: int,
    model: Flow,
    real: torch.Tensor,
    text_ids: torch.Tensor | None,
    inference_cfg_and_steps: list,
    n_samples: int,
) -> torch.Tensor:
    """Run euler_cfg for every (cfg_scale, steps) combo on one GPU.

    Returns a float32 CPU tensor [(n_combos+1) * n_samples, C, H, W] in [-1, 1],
    with real samples appended as the last n_samples rows.
    """
    device = f"cuda:{gpu_id}"
    x1 = real[:n_samples].to(device)   # [n_samples, C, H, W]
    z = torch.randn_like(x1)

    text_ids_dev = text_ids[:n_samples].to(device) if text_ids is not None else None

    fake_rows = []
    with torch.autocast("cuda", torch.bfloat16):
        for cfg_scale, steps in inference_cfg_and_steps:
            fake, _ = model.euler_cfg(
                z, pos_cond=None, cfg_scale=cfg_scale,
                num_steps=steps, text_tokens=text_ids_dev,
            )
            fake_rows.append(fake.clamp(-1, 1).cpu().float())

    fake_rows.append(x1.clamp(-1, 1).cpu().float())
    return torch.cat(fake_rows, dim=0)  # [(n_combos+1)*n_samples, C, H, W]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: dict):
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPU(s)")

    # ------------------------------------------------------------------
    # Copy config to ckpt path for experiment tracking
    # ------------------------------------------------------------------
    os.makedirs(cfg["ckpt_path"], exist_ok=True)
    os.makedirs(cfg["preview_path"], exist_ok=True)
    shutil.copy(CONFIG_PATH, os.path.join(cfg["ckpt_path"], "config.json"))
    print(f"Config saved to {cfg['ckpt_path']}/config.json")

    # ------------------------------------------------------------------
    # Tokenizer (shared, CPU-based)
    # ------------------------------------------------------------------
    tokenizer = None
    if cfg.get("qwen_tokenizer_path") and _TRANSFORMERS_AVAILABLE:
        print(f"Loading tokenizer from {cfg['qwen_tokenizer_path']}...")
        tokenizer = AutoTokenizer.from_pretrained(cfg["qwen_tokenizer_path"])
        print(f"  Vocab size: {tokenizer.vocab_size}")

    # ------------------------------------------------------------------
    # DINOv3 (one per GPU, frozen)
    # ------------------------------------------------------------------
    dino_cfg       = cfg.get("dino", {})
    dino_strength  = float(dino_cfg.get("strength", 0.0))
    dino_threshold = float(dino_cfg.get("timestep_threshold", 0.5))
    dino_models    = []
    if dino_strength > 0.0 and _DINO_AVAILABLE:
        model_name = dino_cfg.get("model_name", "facebook/dinov2-base")
        print(f"Loading DINOv3 ({model_name}) on {n_gpus} GPU(s)...")
        base_dino = HFAutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        for gpu_id in range(n_gpus):
            d = copy.deepcopy(base_dino).to(f"cuda:{gpu_id}").eval()
            for p in d.parameters():
                p.requires_grad_(False)
            dino_models.append(d)
        del base_dino
        print("  DINOv3 loaded.")
    else:
        dino_models = [None] * n_gpus
        if dino_strength > 0.0:
            print("[warn] DINO loss requested but transformers unavailable — skipping")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    parquet_cfg = cfg.get("parquet_dataloader")
    if not parquet_cfg:
        raise RuntimeError(
            "No 'parquet_dataloader' config found. "
            "Please add a 'parquet_dataloader' section to config.json."
        )

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
        shuffle_tags=parquet_cfg.get("shuffle_tags", True),
        tag_drop_percentage=parquet_cfg.get("tag_drop_percentage", 0.1),
        uncond_percentage=0.0,  # handled in forward_fn
        seed=42,
        rank=0,
        num_gpus=1,
        offset=parquet_cfg.get("offset", 0),
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
    # Model
    # ------------------------------------------------------------------
    def model_factory():
        return Flow(**cfg["model_config"])

    wrapper = MultiGPUWrapper(
        model_factory=model_factory,
        optimizer_factory=lambda params: AdamW(params, lr=cfg["lr"], weight_decay=1e-4, betas=(0.9, 0.95)),
        gradient_accumulation_steps=cfg["accum"],
        max_grad_norm=1.0,
        scheduler_factory=lambda opt: LinearLR(
            opt, start_factor=1e-5, end_factor=1.0, total_iters=cfg["warmup"]
        ),
    )
    wrapper.setup()

    if cfg.get("model_checkpoint"):
        wrapper.load_checkpoint(cfg["model_checkpoint"])
    else:
        wrapper.save_checkpoint(os.path.join(cfg["ckpt_path"], "untrained.safetensors"))

    # Lazily compile transformer blocks *after* ramtorch has finished its
    # FX-tracing setup, to avoid "FX tracing a dynamo-optimized function".
    if cfg["model_config"].get("compile_blocks", False):
        print(f"Compiling transformer blocks on {n_gpus} GPU(s)...")
        for m in wrapper.models:
            m.compile_blocks()
        print("  Done.")

    # ------------------------------------------------------------------
    # CSV loss log (deferred flush every log_every_n_steps)
    # ------------------------------------------------------------------
    log_every  = cfg.get("log_every_n_steps", 10)
    csv_path   = os.path.join(cfg["ckpt_path"], "loss_log.csv")
    csv_file   = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        csv_writer.writerow(["step", "loss", "mse", "dino", "lr", "time"])
    t0 = time.time()

    global_step  = 0
    uncond_ratio = cfg.get("uncond_ratio", 0.1)
    max_text_len = cfg.get("max_text_len", 128)
    t_mu         = cfg.get("t_mu", 0.0)

    master_seed = cfg.get("seed", 42)
    torch.manual_seed(master_seed)

    with make_profiler_ctx(cfg, "ramtorch_trace.json") as prof:
        epoch = 0
        while True:
            epoch += 1
            # Increment seed each epoch for varied noise/augmentation sampling
            torch.manual_seed(master_seed + epoch)
            for m in wrapper.models:
                m.train()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

            for batch_idx, batch_data in enumerate(pbar):
                # dummy_collate_fn wraps in an extra list
                batch_data = batch_data[0]
                images, captions, _idx, loss_weights = batch_data[:4]

                # Split across GPUs
                spg = images.shape[0] // n_gpus
                image_chunks   = [images[i*spg:(i+1)*spg]   for i in range(n_gpus)]
                caption_chunks = [captions[i*spg:(i+1)*spg] for i in range(n_gpus)]

                # 1. Forward
                with record_function("forward"):
                    outputs = wrapper.forward(
                        list(zip(image_chunks, caption_chunks)),
                        forward_fn=forward_fn,
                        tokenizer=tokenizer,
                        dino_model=None,
                        dino_threshold=dino_threshold,
                        dino_strength=dino_strength,
                        max_text_len=max_text_len,
                        uncond_ratio=uncond_ratio,
                        t_mu=t_mu,
                    )

                # 2. Backward — per-GPU DINO model passed explicitly
                with record_function("backward"):
                    raw_results = wrapper.run_concurrent(
                        lambda gpu_id: backward_fn(
                            gpu_id, wrapper.models[gpu_id], outputs[gpu_id],
                            dino_model=dino_models[gpu_id],
                            dino_threshold=dino_threshold,
                            dino_strength=dino_strength,
                            accum_steps=cfg["accum"],
                        )
                    )

                total_loss = sum(r[0] for r in raw_results) / n_gpus
                mse_loss   = sum(r[1] for r in raw_results) / n_gpus
                dino_loss  = sum(r[2] for r in raw_results) / n_gpus

                # 3. Sync + step
                if (batch_idx + 1) % cfg["accum"] == 0:
                    with record_function("reduce_grads"):
                        wrapper.reduce_grads()
                    wrapper.clip_grads()
                    with record_function("optimizer_step"):
                        wrapper.optimizer_step()
                        torch.cuda.synchronize()  # ensure all GPU work is done before timing or logging
                        torch.cuda.empty_cache()

                lr = wrapper.last_lr
                pbar.set_postfix(loss=f"{total_loss:.4f}", mse=f"{mse_loss:.4f}", dino=f"{dino_loss:.4f}", lr=f"{lr:.2e}", step=global_step)

                # Deferred CSV write
                csv_writer.writerow([global_step, f"{total_loss:.6f}", f"{mse_loss:.6f}", f"{dino_loss:.6f}", f"{lr:.2e}", f"{time.time()-t0:.1f}"])
                if global_step % log_every == 0:
                    csv_file.flush()

                # --- Step checkpoint ---
                save_every = cfg.get("save_every_n_steps", 0)
                if save_every > 0 and global_step > 0 and global_step % save_every == 0:
                    wrapper.save_checkpoint(os.path.join(cfg["ckpt_path"], f"step_{global_step}.safetensors"))

                # --- Preview images (all GPUs in parallel) ---
                if global_step % cfg["eval_interval"] == 0:
                    preview_spg  = cfg.get("preview_samples_per_gpu", 4)
                    preview_qual = cfg.get("preview_quality", 95)
                    use_png      = (preview_qual >= 100)
                    ext          = "png" if use_png else "jpg"

                    preview_chunks = [
                        (outputs[g][3], outputs[g][4])  # (x1, text_ids) per GPU
                        for g in range(n_gpus)
                    ]

                    preview_results = wrapper.forward(
                        preview_chunks,
                        forward_fn=preview_fn,
                        eval_mode=True,
                        inference_cfg_and_steps=cfg["inference_cfg_and_steps"],
                        n_samples=preview_spg,
                    )

                    # Each GPU returns [(n_combos+1)*preview_spg, C, H, W].
                    # Cat across GPUs then lay out as nrow=preview_spg*n_gpus so
                    # each row-group is one CFG combo (+ real at the bottom).
                    all_images = torch.cat(preview_results, dim=0)
                    img_path = f"{cfg['preview_path']}/step_{global_step}.{ext}"
                    grid = make_grid((all_images + 1) / 2, nrow=preview_spg * n_gpus)
                    if use_png:
                        save_image(grid, img_path)
                    else:
                        grid_uint8 = (grid.clamp(0, 1) * 255).to(torch.uint8)
                        write_jpeg(grid_uint8, img_path, quality=preview_qual)

                global_step += 1

                if prof is not None:
                    prof.step()

            # --- End of epoch ---
            wrapper.save_checkpoint(os.path.join(cfg["ckpt_path"], f"epoch_{epoch}.safetensors"))
            dataset.resample()

    csv_file.close()
    wrapper.cleanup()
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config_pixel_space.json"
    print(f"Loading config from: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        TRAINING_CONFIG = json.load(f)

    train(TRAINING_CONFIG)
