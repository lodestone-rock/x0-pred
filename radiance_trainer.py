"""radiance_trainer.py — Flow-matching trainer for Radiance (Chroma/MM-DiT) using MultiGPUWrapper.

Supports:
- Parquet-based multi-source image-text dataset
- T5 text encoder (frozen, one per GPU, loaded via HuggingFace)
- DINOv2 embedding-similarity auxiliary loss
- Per-block torch.compile + grad checkpointing (independent toggles)
- Deferred CSV logging

Run:
    python radiance_trainer.py
    python radiance_trainer.py config_radiance.json

All settings are in config_radiance.json.
"""
from __future__ import annotations

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
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader
from torchvision.io import write_jpeg
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from ramtorch.multi_gpu import MultiGPUWrapper

from src.models.radiance import Radiance, RadianceParams
from src.models.flow import compute_timestep_weights, create_distribution, sample_from_distribution
from src.dataloaders.parquet_dataloader import ParquetTextImageDataset
from src.dataloaders.bucketing_logic import _bucket_generator

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Optional: transformers (T5 text encoder + DINO)
# ---------------------------------------------------------------------------
try:
    from transformers import AutoTokenizer, T5EncoderModel, AutoModel as HFAutoModel
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    print("[warn] transformers not installed — text conditioning and DINO loss disabled")


# ---------------------------------------------------------------------------
# Profiler helper
# ---------------------------------------------------------------------------


def make_profiler_ctx(cfg: dict, trace_path: str):
    if not cfg.get("profile", False):
        return nullcontext()

    start = cfg.get("profile_start", 20)
    stop  = cfg.get("profile_stop",  23)
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
    """Remove ``_orig_mod.`` prefix added by torch.compile."""
    prefix = "_orig_mod."
    return {k.replace(prefix, "") if prefix in k else k: v for k, v in sd.items()}


# ---------------------------------------------------------------------------
# Text encoding (T5)
# ---------------------------------------------------------------------------

def _encode_text(
    text_encoder,
    tokenizer,
    texts: list[str],
    max_seq_len: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a list of strings with HuggingFace T5EncoderModel.

    Returns:
        embeddings: [B, L, hidden_dim]  last hidden state
        mask:       [B, L]              bool attention mask
    """
    inputs = tokenizer(
        texts,
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = text_encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )

    embeddings = outputs.last_hidden_state   # [B, L, hidden_dim]
    mask       = inputs.attention_mask.bool()
    return embeddings, mask


# ---------------------------------------------------------------------------
# DINO auxiliary loss
# ---------------------------------------------------------------------------


def _compute_dino_loss(
    dino_model,
    pred_imgs: torch.Tensor,
    gt_imgs: torch.Tensor,
    timesteps: torch.Tensor,
    threshold: float,
    device: str,
) -> torch.Tensor:
    """DINOv2 embedding-similarity loss (MSE on pooler_output).
    Only applied to samples where t < threshold.
    pred_imgs / gt_imgs: [B, C, H, W] in [-1, 1].
    Returns per-sample loss [B] (zero for samples above threshold).
    """
    B    = pred_imgs.shape[0]
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
    loss[mask]  = loss_masked.to(loss.dtype)
    return loss


# ---------------------------------------------------------------------------
# Per-GPU callables
# ---------------------------------------------------------------------------


def forward_fn(
    gpu_id: int,
    model: Radiance,
    real: torch.Tensor,
    captions: list[str],
    text_encoders: list,   # one per GPU; indexed by gpu_id
    tokenizer,
    dino_model,
    dino_threshold: float,
    dino_strength: float,
    max_seq_len: int,
    uncond_ratio: float,
    neg_embeddings: torch.Tensor | None,  # pre-encoded empty-string [1, L, D]
    neg_mask: torch.Tensor | None,        # pre-encoded empty-string mask [1, L]
    t_mu: float = 0.0,
    t_shift_mode: str = "sampled",
) -> tuple:
    """Forward only — returns everything needed for backward."""
    device = f"cuda:{gpu_id}"
    text_encoder = text_encoders[gpu_id]
    x1 = real.to(device, non_blocking=True)
    x0 = torch.randn_like(x1)
    B  = x1.shape[0]

    # Sample timesteps
    if t_shift_mode == "weighted":
        t_flat    = torch.rand(B, device=device)
        t_weights = compute_timestep_weights(t_flat, mu=t_mu)
        t         = t_flat[:, None, None, None].to(x1.dtype)
    else:
        x_dist, probs = create_distribution(1000, device=device, mu=t_mu)
        t             = sample_from_distribution(x_dist, probs, B)[:, None, None, None].to(x1.dtype)
        t_weights     = None

    noisy_image = x0 * t + x1 * (1 - t)

    # Encode text with T5
    txt_embeds, txt_mask = _encode_text(
        text_encoder, tokenizer, captions, max_seq_len, device
    )  # [B, L, D], [B, L]

    # CFG dropout: replace selected rows with pre-encoded empty-string embedding
    if uncond_ratio > 0.0:
        drop_mask = torch.rand(B, device=device) < uncond_ratio
        if drop_mask.any() and neg_embeddings is not None:
            neg_emb  = neg_embeddings.to(device, non_blocking=True).expand(B, -1, -1)
            neg_msk  = neg_mask.to(device, non_blocking=True).expand(B, -1)
            txt_embeds = torch.where(drop_mask[:, None, None], neg_emb,  txt_embeds)
            txt_mask   = torch.where(drop_mask[:, None],       neg_msk,  txt_mask)

    with torch.autocast("cuda", torch.bfloat16):
        predicted_image = model(noisy_image, t, txt_embeds, txt_mask)

    return predicted_image, noisy_image, t, x1, txt_embeds, txt_mask, t_weights


def backward_fn(
    gpu_id: int,
    model: Radiance,
    output: tuple,
    dino_model,
    dino_threshold: float,
    dino_strength: float,
    accum_steps: int = 1,
) -> tuple[float, float, float]:
    """Backward only — returns (total_loss, mse_loss, dino_loss)."""
    predicted_image, noisy_image, t, x1, txt_embeds, txt_mask, t_weights = output
    device   = predicted_image.device
    t_scalar = t.view(-1)

    # Radiance.forward() returns v-prediction via _apply_x0_residual.
    # Target: v = (noisy - x1) / (t + eps)
    target_velocity    = (noisy_image - x1) / (t + 5e-2)
    predicted_velocity = predicted_image

    if t_weights is not None:
        per_sample_mse = F.mse_loss(predicted_velocity, target_velocity,
                                    reduction="none").mean(dim=[1, 2, 3])
        mse = (per_sample_mse * t_weights.to(device)).mean()
    else:
        mse = F.mse_loss(predicted_velocity, target_velocity)

    if dino_model is not None and dino_strength > 0.0:
        dino_per_sample = _compute_dino_loss(
            dino_model, predicted_image, x1,
            t_scalar, dino_threshold, str(device),
        )
        if t_weights is not None:
            dino_term = (dino_per_sample * t_weights.to(device)).mean() * dino_strength
        else:
            dino_term = dino_per_sample.mean() * dino_strength
        total = mse + dino_term
    else:
        dino_term = mse.new_zeros(1)
        total     = mse

    (total / accum_steps).backward()
    return total.item(), mse.item(), dino_term.item()


def preview_fn(
    gpu_id: int,
    model: Radiance,
    real: torch.Tensor,
    txt_embeds: torch.Tensor | None,
    txt_mask: torch.Tensor | None,
    inference_cfg_and_steps: list,
    n_samples: int,
    text_encoders: list,   # one per GPU; indexed by gpu_id
    tokenizer,
    max_seq_len: int,
    schedule_mu: float | None = None,
) -> torch.Tensor:
    """Run euler_cfg for every combo on one GPU.

    Each entry in ``inference_cfg_and_steps`` may have 2 or 3 elements::

        [cfg_scale, steps]
        [cfg_scale, steps, schedule_mu]   # float override, or null for auto

    Returns a float32 CPU tensor [(n_combos+1) * n_samples, C, H, W] in
    [-1, 1], with real samples appended as the last n_samples rows.
    """
    device       = f"cuda:{gpu_id}"
    text_encoder = text_encoders[gpu_id]
    x1           = real[:n_samples].to(device, non_blocking=True)
    z            = torch.randn_like(x1)

    # Positive conditioning
    pos_embeds = txt_embeds[:n_samples].to(device, non_blocking=True) if txt_embeds is not None else None
    pos_mask   = txt_mask[:n_samples].to(device, non_blocking=True)   if txt_mask   is not None else None

    # Negative conditioning (empty string)
    neg_embeds, neg_mask = _encode_text(
        text_encoder, tokenizer, [""] * n_samples, max_seq_len, device
    )

    fake_rows = []
    with torch.autocast("cuda", torch.bfloat16):
        for combo in inference_cfg_and_steps:
            if len(combo) == 2:
                cfg_scale, steps = combo
                mu = schedule_mu
            elif len(combo) == 3:
                cfg_scale, steps, mu = combo
                if mu is None:
                    mu = schedule_mu
            else:
                raise ValueError(
                    f"inference_cfg_and_steps entry must have 2–3 elements, got {combo!r}"
                )

            fake, _ = model.euler_cfg(
                z,
                cfg_scale=cfg_scale,
                num_steps=steps,
                txt=pos_embeds,
                txt_mask=pos_mask,
                neg_txt=neg_embeds,
                neg_txt_mask=neg_mask,
                schedule_mu=mu,
            )
            fake_rows.append(fake.clamp(-1, 1).cpu().float())

    fake_rows.append(x1.clamp(-1, 1).cpu().float())
    return torch.cat(fake_rows, dim=0)


# ---------------------------------------------------------------------------
# Compile warmup
# ---------------------------------------------------------------------------


def warmup_compiled_model(
    models: list,
    model_cfg: dict,
    base_res: list,
    ratio_cutoff: float,
    resolution_step: int,
    max_seq_len: int,
) -> None:
    """Eagerly trigger torch.compile for every resolution bucket.

    torch.compile is lazy — Dynamo only traces on the first forward call for
    each unique input shape.  When the training loop dispatches work via a
    ThreadPoolExecutor, two threads can hit their first-compilation at the
    same time, causing the intermittent
    "FX tracing a dynamo-optimized function" error.

    Calling this *after* compile_blocks() but *before* the DataLoader is
    constructed runs all compilations sequentially on a single thread, so
    the executor never sees a cold cache.
    """
    in_channels    = model_cfg.get("in_channels", 3)
    context_in_dim = model_cfg.get("context_in_dim", 4096)

    image_shapes: set[tuple[int, int]] = set()
    for res in base_res:
        for w, h in _bucket_generator(res, ratio_cutoff, resolution_step):
            image_shapes.add((h, w))

    n_buckets = len(image_shapes)
    n_gpus    = len(models)
    print(f"  Warming up {n_buckets} resolution bucket(s) on {n_gpus} GPU(s)...")

    for gpu_id, model in enumerate(models):
        device = f"cuda:{gpu_id}"
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for H, W in sorted(image_shapes):
                dummy_img = torch.zeros(1, in_channels, H, W, device=device)
                dummy_t   = torch.tensor([0.5], device=device)
                dummy_txt = torch.zeros(1, max_seq_len, context_in_dim, device=device)
                dummy_msk = torch.ones(1, max_seq_len, dtype=torch.bool, device=device)
                model(dummy_img, dummy_t, dummy_txt, dummy_msk)
                num_patches = (H // 16) * (W // 16)
                print(f"    [GPU {gpu_id}] {W}x{H} ({num_patches} patches)")
        torch.cuda.synchronize(device)
        model.train()

    print("  Warmup complete.")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: dict):
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPU(s)")

    os.makedirs(cfg["ckpt_path"],    exist_ok=True)
    os.makedirs(cfg["preview_path"], exist_ok=True)
    _cfg_dest = os.path.join(cfg["ckpt_path"], os.path.basename(CONFIG_PATH))
    shutil.copy(CONFIG_PATH, _cfg_dest)
    print(f"Config saved to {_cfg_dest}")

    # ------------------------------------------------------------------
    # T5 text encoder (one per GPU, frozen)
    # ------------------------------------------------------------------
    t5_path      = cfg.get("t5_path")
    max_seq_len  = cfg.get("max_seq_len", 512)
    text_encoders = []
    tokenizer     = None

    tokenizer_path = cfg.get("tokenizer_path", t5_path)  # fall back to t5_path if not set

    if t5_path and _TRANSFORMERS_AVAILABLE:
        print(f"Loading T5 tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        print(f"Loading T5 text encoder on {n_gpus} GPU(s)...")
        base_enc = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.bfloat16)
        for gpu_id in range(n_gpus):
            enc = copy.deepcopy(base_enc).to(f"cuda:{gpu_id}").eval()
            for p in enc.parameters():
                p.requires_grad_(False)
            text_encoders.append(enc)
        del base_enc
        print("  T5 encoder loaded.")

        # Pre-encode the empty string once for CFG dropout
        print("  Pre-encoding empty-string negative embedding...")
        neg_embeddings, neg_mask = _encode_text(
            text_encoders[0], tokenizer, [""], max_seq_len, "cuda:0"
        )
        neg_embeddings = neg_embeddings.cpu()  # [1, L, D]
        neg_mask       = neg_mask.cpu()        # [1, L]
        print(f"  Negative embedding shape: {list(neg_embeddings.shape)}")
    else:
        text_encoders  = [None] * n_gpus
        neg_embeddings = None
        neg_mask       = None
        if t5_path:
            print("[warn] transformers not installed — text conditioning disabled")

    # ------------------------------------------------------------------
    # DINOv2 (one per GPU, frozen)
    # ------------------------------------------------------------------
    dino_cfg       = cfg.get("dino", {})
    dino_strength  = float(dino_cfg.get("strength", 0.0))
    dino_threshold = float(dino_cfg.get("timestep_threshold", 0.5))
    dino_models    = []

    if dino_strength > 0.0 and _TRANSFORMERS_AVAILABLE:
        model_name = dino_cfg.get("model_name", "facebook/dinov2-base")
        print(f"Loading DINOv2 ({model_name}) on {n_gpus} GPU(s)...")
        base_dino = HFAutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        for gpu_id in range(n_gpus):
            d = copy.deepcopy(base_dino).to(f"cuda:{gpu_id}").eval()
            for p in d.parameters():
                p.requires_grad_(False)
            dino_models.append(d)
        del base_dino
        print("  DINOv2 loaded.")
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
            "Please add a 'parquet_dataloader' section to config_radiance.json."
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
        tokenizer=None,   # raw strings — text encoding happens in forward_fn
        max_text_len=max_seq_len,
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
        return Radiance(RadianceParams(**cfg["model_config"]))

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
    _orig_save = wrapper.save_checkpoint
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

    # Compile blocks *after* wrapper.setup() to avoid FX tracing conflicts
    if cfg["model_config"].get("_use_compiled", False):
        print(f"Compiling transformer blocks on {n_gpus} GPU(s)...")
        for m in wrapper.models:
            m.compile_blocks()
        warmup_compiled_model(
            wrapper.models,
            cfg["model_config"],
            parquet_cfg.get("base_resolution", [256]),
            parquet_cfg.get("ratio_cutoff", 2.0),
            parquet_cfg.get("resolution_step", 64),
            max_seq_len,
        )
        print("  Done.")

    # ------------------------------------------------------------------
    # CSV loss log
    # ------------------------------------------------------------------
    log_every  = cfg.get("log_every_n_steps", 10)
    csv_path   = os.path.join(cfg["ckpt_path"], "loss_log.csv")
    csv_file   = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        csv_writer.writerow(["step", "loss", "mse", "dino", "lr", "t_mu", "time"])
    t0 = time.time()

    global_step       = cfg.get("initial_global_step", 0)
    uncond_ratio      = cfg.get("uncond_ratio", 0.1)
    t_mu              = cfg.get("t_mu", 0.0)
    t_mu_start        = cfg.get("t_mu_start", t_mu)
    t_mu_end          = cfg.get("t_mu_end", 1.0)
    t_mu_anneal_steps = cfg.get("t_mu_anneal_steps", 0)
    t_shift_mode      = cfg.get("t_shift_mode", "sampled")

    master_seed = cfg.get("seed", 42)
    torch.manual_seed(master_seed)

    with make_profiler_ctx(cfg, "radiance_trace.json") as prof:
        epoch = 0
        while True:
            epoch += 1
            torch.manual_seed(master_seed + epoch)
            for m in wrapper.models:
                m.train()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

            for batch_idx, batch_data in enumerate(pbar):
                batch_data = batch_data[0]
                images, captions, _idx, loss_weights = batch_data[:4]

                spg = images.shape[0] // n_gpus
                image_chunks   = [images[i * spg:(i + 1) * spg]   for i in range(n_gpus)]
                caption_chunks = [captions[i * spg:(i + 1) * spg] for i in range(n_gpus)]

                # Anneal t_mu
                if t_mu_anneal_steps > 0:
                    frac = min(global_step, t_mu_anneal_steps) / t_mu_anneal_steps
                    t_mu = t_mu_start + (t_mu_end - t_mu_start) * frac

                # 1. Forward
                with record_function("forward"):
                    outputs = wrapper.forward(
                        list(zip(image_chunks, caption_chunks)),
                        forward_fn=forward_fn,
                        text_encoders=text_encoders,
                        tokenizer=tokenizer,
                        dino_model=None,
                        dino_threshold=dino_threshold,
                        dino_strength=dino_strength,
                        max_seq_len=max_seq_len,
                        uncond_ratio=uncond_ratio,
                        neg_embeddings=neg_embeddings,
                        neg_mask=neg_mask,
                        t_mu=t_mu,
                        t_shift_mode=t_shift_mode,
                    )

                # 2. Backward
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
                        torch.cuda.synchronize()
                        if cfg.get("empty_cache_on_step", False):
                            torch.cuda.empty_cache()

                lr = wrapper.last_lr
                pbar.set_postfix(
                    loss=f"{total_loss:.4f}", mse=f"{mse_loss:.4f}",
                    dino=f"{dino_loss:.4f}", lr=f"{lr:.2e}",
                    t_mu=f"{t_mu:.3f}", step=global_step,
                )

                csv_writer.writerow([
                    global_step, f"{total_loss:.6f}", f"{mse_loss:.6f}",
                    f"{dino_loss:.6f}", f"{lr:.2e}", f"{t_mu:.4f}",
                    f"{time.time() - t0:.1f}",
                ])
                if global_step % log_every == 0:
                    csv_file.flush()

                # Step checkpoint
                save_every = cfg.get("save_every_n_steps", 0)
                if save_every > 0 and global_step > 0 and global_step % save_every == 0:
                    wrapper.save_checkpoint(
                        os.path.join(cfg["ckpt_path"], f"step_{global_step}.safetensors")
                    )

                # Preview images
                if global_step % cfg["eval_interval"] == 0:
                    preview_spg  = cfg.get("preview_samples_per_gpu", 4)
                    preview_qual = cfg.get("preview_quality", 95)
                    use_png      = (preview_qual >= 100)
                    ext          = "png" if use_png else "jpg"

                    # Grab (x1, txt_embeds, txt_mask) from forward outputs
                    preview_chunks = [
                        (outputs[g][3], outputs[g][4], outputs[g][5])  # x1, txt_embeds, txt_mask
                        for g in range(n_gpus)
                    ]

                    preview_results = wrapper.forward(
                        preview_chunks,
                        forward_fn=preview_fn,
                        eval_mode=True,
                        inference_cfg_and_steps=cfg["inference_cfg_and_steps"],
                        n_samples=preview_spg,
                        text_encoders=text_encoders,
                        tokenizer=tokenizer,
                        max_seq_len=max_seq_len,
                        schedule_mu=t_mu,
                    )

                    # Reorder GPU-major → combo-major so each combo occupies
                    # one contiguous grid row:
                    #   cat shape: [n_gpus*(n_combos+1)*spg, C, H, W]
                    #   -> [n_gpus, n_combos+1, spg, C, H, W]
                    #   -> [n_combos+1, n_gpus, spg, C, H, W]
                    #   -> [(n_combos+1)*n_gpus*spg, C, H, W]
                    n_combos = len(cfg["inference_cfg_and_steps"])
                    _C, _H, _W = preview_results[0].shape[1:]
                    all_images = (
                        torch.cat(preview_results, dim=0)
                        .view(n_gpus, n_combos + 1, preview_spg, _C, _H, _W)
                        .permute(1, 0, 2, 3, 4, 5)
                        .reshape((n_combos + 1) * n_gpus * preview_spg, _C, _H, _W)
                    )
                    total_cols = preview_spg * n_gpus
                    img_path = f"{cfg['preview_path']}/step_{global_step}.{ext}"
                    grid = make_grid((all_images + 1) / 2, nrow=total_cols)

                    # --- Row-axis labels (cfg / steps / shift / ground truth) ---
                    if _PIL_AVAILABLE:
                        _row_labels = []
                        for _combo in cfg["inference_cfg_and_steps"]:
                            _cfg_s = _combo[0]
                            _steps = _combo[1]
                            _mu    = _combo[2] if len(_combo) > 2 else None
                            _shift = "null" if _mu is None else str(_mu)
                            _row_labels.append(f"cfg {_cfg_s}  steps {_steps}  shift {_shift}")
                        _row_labels.append("ground truth")

                        _grid_np  = (grid.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                        _grid_pil = Image.fromarray(_grid_np, mode="RGB")
                        _gw, _gh  = _grid_pil.size

                        _n_rows  = n_combos + 1
                        _row_h   = _gh // _n_rows
                        _font_size = 12

                        _font = None
                        for _fp in [
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                        ]:
                            try:
                                _font = ImageFont.truetype(_fp, _font_size)
                                break
                            except Exception:
                                pass
                        if _font is None:
                            _font = ImageFont.load_default()

                        _draw_tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
                        _label_h  = max(
                            _draw_tmp.textbbox((0, 0), lbl, font=_font)[3]
                            for lbl in _row_labels
                        )
                        _label_w = _label_h + 8

                        _strip = Image.new("RGB", (_label_w, _gh), (30, 30, 30))
                        for _i, _lbl in enumerate(_row_labels):
                            _bb  = _draw_tmp.textbbox((0, 0), _lbl, font=_font)
                            _tw  = _bb[2] - _bb[0]
                            _th  = _bb[3] - _bb[1]
                            _tile = Image.new("RGB", (_row_h, _label_w), (30, 30, 30))
                            ImageDraw.Draw(_tile).text(
                                (_row_h // 2 - _tw // 2, _label_w // 2 - _th // 2),
                                _lbl, fill=(220, 220, 220), font=_font,
                            )
                            _strip.paste(_tile.rotate(90, expand=True), (0, _i * _row_h))

                        _composite = Image.new("RGB", (_label_w + _gw, _gh), (0, 0, 0))
                        _composite.paste(_strip, (0, 0))
                        _composite.paste(_grid_pil, (_label_w, 0))

                        if use_png:
                            _composite.save(img_path)
                        else:
                            import io as _io
                            _buf = _io.BytesIO()
                            _composite.save(_buf, format="JPEG", quality=preview_qual)
                            with open(img_path, "wb") as _f:
                                _f.write(_buf.getvalue())
                    else:
                        if use_png:
                            save_image(grid, img_path)
                        else:
                            grid_uint8 = (grid.clamp(0, 1) * 255).to(torch.uint8)
                            write_jpeg(grid_uint8, img_path, quality=preview_qual)

                global_step += 1

                if prof is not None:
                    prof.step()

            # End of epoch
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
    CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config_radiance.json"
    print(f"Loading config from: {CONFIG_PATH}")
    TRAINING_CONFIG = load_config(CONFIG_PATH)
    train(TRAINING_CONFIG)