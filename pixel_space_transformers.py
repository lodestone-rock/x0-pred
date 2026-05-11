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
    compute_timestep_weights,
)
from src.models.flow_baseline import FlowBaseline
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
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _strip_compiled_keys(sd: dict) -> dict:
    """Remove the ``_orig_mod.`` prefix that torch.compile wraps around keys.

    When blocks are compiled with ``torch.compile``, the state dict keys gain
    a ``_orig_mod.`` prefix (e.g. ``blocks.0._orig_mod.attn.qkv.weight``).
    Stripping it makes checkpoints loadable by the uncompiled model and keeps
    the format consistent regardless of whether compile is enabled.
    """
    prefix = "_orig_mod."
    return {
        k.replace(prefix, "") if prefix in k else k: v
        for k, v in sd.items()
    }


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
    captions: "list[str] | torch.Tensor",
    tokenizer,
    dino_model,
    dino_threshold: float,
    dino_strength: float,
    max_text_len: int,
    uncond_ratio: float,
    t_mu: float = 0.0,
    t_shift_mode: str = "sampled",
    use_mask: bool = False,
    skip_middle: bool = False,
    uncond_token_ids: "torch.Tensor | None" = None,
) -> tuple:
    """Forward only — returns everything needed for backward.
    All tensors are in [B, C, H, W] image space; no patch flattening here.

    Args:
        t_shift_mode: controls how the timestep shift is applied.
            ``"sampled"``  — original behaviour: sample t from the shifted
                distribution (non-uniform t values, no loss weighting).
            ``"weighted"`` — sample t uniformly then weight the per-sample
                loss by the shifted density, keeping t ~ Uniform[0,1].
        use_mask: run the SPRINT sparse-middle path for this batch
            (structured group-wise token drop, [MASK]-token padding,
            dense–sparse residual fusion — arXiv:2510.21986).
        skip_middle: bypass the middle stack entirely (Path-Drop
            training — paper §C.1, 10% probability). Trains the model
            to do PDG inference.
        Both coin-flips are done in the training loop so all GPUs see
        the same decision. ``skip_middle`` takes precedence over
        ``use_mask`` inside ``Flow.forward``.
    """
    device = f"cuda:{gpu_id}"
    x1 = real.to(device, non_blocking=True)          # [B, C, H, W]
    x0 = torch.randn_like(x1)

    B = x1.shape[0]
    if t_shift_mode == "weighted":
        # Uniform t; shift applied via per-sample loss weights in backward_fn.
        t_flat   = torch.rand(B, device=device)
        t_weights = compute_timestep_weights(t_flat, mu=t_mu)
        t = t_flat[:, None, None, None].to(x1.dtype)
    else:
        # Original: sample from the shifted distribution.
        x_dist, probabilities = create_distribution(1000, device=device, mu=t_mu)
        t = sample_from_distribution(x_dist, probabilities, B)[:, None, None, None].to(x1.dtype)
        t_weights = None

    noisy_image = x0 * t + x1 * (1 - t)

    # Unconditional dropout
    is_uncond_mask = torch.zeros(B, dtype=torch.bool, device=device)

    if isinstance(captions, torch.Tensor):
        # Pre-tokenized path: captions is [B, L] int64 from the dataloader worker.
        # Apply CFG dropout by replacing selected rows with the pre-tokenized
        # empty-string ids (uncond_token_ids [1, L]), falling back to an
        # all-zeros row when uncond_token_ids is not provided.
        text_ids = captions.to(device, non_blocking=True)
        if uncond_ratio > 0.0:
            mask = torch.rand(text_ids.shape[0], device=device) < uncond_ratio
            if mask.any():
                if uncond_token_ids is not None:
                    fill = uncond_token_ids.to(device, non_blocking=True).expand(text_ids.shape[0], -1)
                else:
                    fill = torch.zeros_like(text_ids)
                text_ids = torch.where(mask.unsqueeze(1), fill, text_ids)
            is_uncond_mask = mask
    else:
        # Raw-string fallback path (no tokenizer configured on the dataset).
        dropped = ["" if (torch.rand(1).item() < uncond_ratio) else c for c in captions]
        is_uncond_mask = torch.tensor(
            [c == "" for c in dropped], dtype=torch.bool, device=device
        )
        text_ids = None
        if tokenizer is not None:
            text_ids = _tokenize_captions(tokenizer, dropped, max_text_len, device)

    with torch.autocast("cuda", torch.bfloat16):
        predicted_image = model(noisy_image, t, text_tokens=text_ids,
                                use_mask=use_mask, skip_middle=skip_middle)

    return predicted_image, noisy_image, t, x1, text_ids, t_weights, is_uncond_mask


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
    predicted_image, noisy_image, t, x1, text_ids, t_weights, *_ = output
    device = predicted_image.device
    t_scalar = t.view(-1)

    target_velocity    = (noisy_image - x1) / (t + 5e-2)
    predicted_velocity = (noisy_image - predicted_image) / (t + 5e-2)

    if t_weights is not None:
        # Weighted mode: per-sample MSE averaged with importance weights.
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
        total = mse

    (total / accum_steps).backward()
    return total.item(), mse.item(), dino_term.item()


def cfg_amplifying_backward_fn(
    gpu_id: int,
    model: Flow,
    output: tuple,
    teacher_model,
    cfg_scale: float,
    uncond_token_ids: "torch.Tensor | None",
    dino_model,
    dino_threshold: float,
    dino_strength: float,
    accum_steps: int = 1,
) -> tuple[float, float, float]:
    """CFG-amplifying backward pass — returns (total_loss, mse_loss, dino_loss).

    Uncond samples (is_uncond_mask == True) use the standard flow-matching
    target: ``target_velocity = (noisy_image - x1) / (t + 5e-2)``.

    Cond samples (is_uncond_mask == False) chase the teacher model's own CFG
    output (stop-grad), baking the guidance strength into the model weights:

        x0_cfg = x0_neg + cfg_scale * (x0_pos - x0_neg)
        cfg_target_velocity = (noisy_image - x0_cfg) / (t + 5e-2)

    ``teacher_model`` is either an EMA copy of the model (recommended) or the
    current model itself (both under torch.no_grad so no gradients flow back).
    """
    predicted_image, noisy_image, t, x1, text_ids, t_weights, is_uncond_mask = output
    device = predicted_image.device
    t_scalar = t.view(-1)

    B = predicted_image.shape[0]
    eps = 5e-2

    # --- Build mixed target_velocity ---
    # Start with the standard flow target for all samples; overwrite cond rows.
    target_velocity = (noisy_image - x1) / (t + eps)

    cond_idx = (~is_uncond_mask).nonzero(as_tuple=True)[0]
    if cond_idx.numel() > 0:
        noisy_cond = noisy_image[cond_idx]
        t_cond     = t[cond_idx]
        text_cond  = text_ids[cond_idx] if text_ids is not None else None

        # Build uncond fill for the teacher's negative pass.
        if uncond_token_ids is not None:
            uncond_fill = uncond_token_ids.to(device, non_blocking=True).expand(
                cond_idx.numel(), -1
            )
        else:
            uncond_fill = torch.zeros_like(text_cond) if text_cond is not None else None

        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            x0_pos = teacher_model(noisy_cond, t_cond, text_tokens=text_cond)
            x0_neg = teacher_model(noisy_cond, t_cond, text_tokens=uncond_fill)

        # Blend in velocity space (x0 is a point, not a direction), then
        # recover the CFG-guided x0 point and apply the standard training
        # target formula — consistent with backward_fn and safe at t ≈ 0.
        x0_pos = x0_pos.to(noisy_cond.dtype)
        x0_neg = x0_neg.to(noisy_cond.dtype)
        v_pos = (x0_pos - noisy_cond) / t_cond
        v_neg = (x0_neg - noisy_cond) / t_cond
        v_cfg = v_neg + cfg_scale * (v_pos - v_neg)
        # Recover the CFG-guided x0 point from the blended velocity.
        x0_cfg = noisy_cond + v_cfg * t_cond
        cfg_target = (noisy_cond - x0_cfg) / (t_cond + eps)
        target_velocity[cond_idx] = cfg_target

    predicted_velocity = (noisy_image - predicted_image) / (t + eps)

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
    tokenizer=None,
    max_text_len: int = 128,
    schedule_mu: float | None = None,
) -> torch.Tensor:
    """Run euler_cfg for every combo on one GPU.

    Each entry in ``inference_cfg_and_steps`` may have 2, 3, or 4 elements::

        [cfg_scale, steps]
        [cfg_scale, steps, schedule_mu]
        [cfg_scale, steps, schedule_mu, autoguidance_mode]

    ``schedule_mu`` defaults to the function-level ``schedule_mu`` argument
    when omitted (pass ``null`` in JSON for uniform-dt / legacy sampling).

    ``autoguidance_mode`` controls the negative guidance pass (default
    ``"classic"``):
        ``"classic"`` — full forward, empty text (standard CFG).
        ``"pdg"``     — Path-Drop Guidance (SPRINT §3.4,
                        arXiv:2510.21986 Eq. 4): the negative pass bypasses
                        the middle blocks entirely and uses empty text,
                        nearly halving inference FLOPs per guided step.

    Returns a float32 CPU tensor [(n_combos+1) * n_samples, C, H, W] in
    [-1, 1], with real samples appended as the last n_samples rows.
    """
    device = f"cuda:{gpu_id}"
    x1 = real[:n_samples].to(device, non_blocking=True)   # [n_samples, C, H, W]
    z = torch.randn_like(x1)

    text_ids_dev = text_ids[:n_samples].to(device, non_blocking=True) if text_ids is not None else None

    uncond_text_ids = None
    if tokenizer is not None:
        uncond_text_ids = _tokenize_captions(tokenizer, [""] * n_samples, max_text_len, device)

    fake_rows = []
    with torch.autocast("cuda", torch.bfloat16):
        for combo in inference_cfg_and_steps:
            if len(combo) == 2:
                cfg_scale, steps = combo
                mu = schedule_mu
                ag_mode = "classic"
            elif len(combo) == 3:
                cfg_scale, steps, mu = combo
                ag_mode = "classic"
            elif len(combo) == 4:
                cfg_scale, steps, mu, ag_mode = combo
                if mu is None:
                    mu = schedule_mu
            else:
                raise ValueError(
                    f"inference_cfg_and_steps entry must have 2–4 elements, "
                    f"got {combo!r}"
                )
            fake, _ = model.euler_cfg(
                z, cfg_scale=cfg_scale,
                num_steps=steps, text_tokens=text_ids_dev,
                uncond_text_tokens=uncond_text_ids,
                schedule_mu=mu,
                autoguidance_mode=ag_mode,
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
    _cfg_dest = os.path.join(cfg["ckpt_path"], os.path.basename(CONFIG_PATH))
    shutil.copy(CONFIG_PATH, _cfg_dest)
    print(f"Config saved to {_cfg_dest}")

    # ------------------------------------------------------------------
    # Tokenizer (shared, CPU-based)
    # ------------------------------------------------------------------
    tokenizer = None
    uncond_token_ids = None
    if cfg.get("qwen_tokenizer_path") and _TRANSFORMERS_AVAILABLE:
        print(f"Loading tokenizer from {cfg['qwen_tokenizer_path']}...")
        tokenizer = AutoTokenizer.from_pretrained(cfg["qwen_tokenizer_path"])
        print(f"  Vocab size: {tokenizer.vocab_size}")
        # Pre-tokenize the empty string once; used for CFG dropout in forward_fn.
        _enc = tokenizer(
            [""],
            padding="max_length",
            truncation=True,
            max_length=cfg.get("max_text_len", 128),
            return_tensors="pt",
        )
        uncond_token_ids = _enc["input_ids"]  # [1, max_text_len] int64, CPU
        print(f"  Uncond token ids pre-computed (shape {list(uncond_token_ids.shape)}).")

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
        tokenizer=tokenizer,
        max_text_len=cfg.get("max_text_len", 128),
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

    model_cls = {"flow": Flow, "baseline": FlowBaseline}.get(
        cfg.get("model_class", "flow"), Flow
    )

    def model_factory():
        return model_cls(**cfg["model_config"])

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

    # Patch save_checkpoint to strip torch.compile's _orig_mod. key prefix so
    # checkpoints are always loadable by the plain (uncompiled) model.
    _orig_save = wrapper.save_checkpoint
    def _save_checkpoint_stripped(path: str):
        import types
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

    # Lazily compile transformer blocks *after* ramtorch has finished its
    # FX-tracing setup, to avoid "FX tracing a dynamo-optimized function".
    if cfg["model_config"].get("compile_blocks", False):
        print(f"Compiling transformer blocks on {n_gpus} GPU(s)...")
        for m in wrapper.models:
            m.compile_blocks()
        print("  Done.")

    # ------------------------------------------------------------------
    # CFG-amplifying mode setup
    # ------------------------------------------------------------------
    cfg_amp_cfg     = cfg.get("cfg_amplifying", {})
    cfg_amp_enabled = cfg_amp_cfg.get("enabled", False)
    cfg_amp_scale   = float(cfg_amp_cfg.get("cfg_scale", 3.0))
    ema_enabled     = cfg_amp_cfg.get("ema_enabled", True)
    ema_decay       = float(cfg_amp_cfg.get("ema_decay", 0.9999))

    ema_models: list = []
    if cfg_amp_enabled and ema_enabled:
        print(f"CFG-amplifying: building EMA models (decay={ema_decay}) on {n_gpus} GPU(s)...")
        for gpu_id in range(n_gpus):
            ema_m = copy.deepcopy(wrapper.models[gpu_id]).to(f"cuda:{gpu_id}").eval()
            for p in ema_m.parameters():
                p.requires_grad_(False)
            ema_models.append(ema_m)
        print("  EMA models ready.")
    elif cfg_amp_enabled:
        print("CFG-amplifying: EMA disabled — using live model weights as teacher.")

    # ------------------------------------------------------------------
    # CSV loss log (deferred flush every log_every_n_steps)
    # ------------------------------------------------------------------
    log_every  = cfg.get("log_every_n_steps", 10)
    csv_path   = os.path.join(cfg["ckpt_path"], "loss_log.csv")
    csv_file   = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        csv_writer.writerow(["step", "loss", "mse", "dino", "lr", "t_mu", "time"])
    t0 = time.time()

    global_step        = cfg.get("initial_global_step", 0)
    uncond_ratio       = cfg.get("uncond_ratio", 0.1)
    max_text_len       = cfg.get("max_text_len", 128)
    t_mu               = cfg.get("t_mu", 0.0)
    # Linear t_mu annealing: decays from t_mu_start → t_mu_end over
    # t_mu_anneal_steps steps, then holds at t_mu_end.
    # When t_mu_anneal_steps is 0 (default), t_mu is static.
    t_mu_start         = cfg.get("t_mu_start", t_mu)
    t_mu_end           = cfg.get("t_mu_end", 1.0)
    t_mu_anneal_steps  = cfg.get("t_mu_anneal_steps", 0)
    # Timestep shift mode for ablations:
    #   "sampled"  — original: sample t from the shifted distribution (default)
    #   "weighted" — sample t uniformly, weight per-sample loss by shifted density
    t_shift_mode       = cfg.get("t_shift_mode", "sampled")
    # SPRINT training coin-flip probabilities (paper §C.1, arXiv:2510.21986):
    #   sprint_mask_ratio — fraction of steps that run the sparse-middle
    #     (75% token-drop + [MASK] padding) path. Paper uses 1.0 during
    #     pretraining, 0.0 during finetuning.
    #   pdg_drop_ratio    — fraction of steps that bypass the middle stack
    #     entirely (g_theta output replaced with [MASK]). Trains the model
    #     for PDG inference. Paper uses 0.1 throughout both stages.
    sprint_mask_ratio  = cfg.get("sprint_mask_ratio", 1.0)
    pdg_drop_ratio     = cfg.get("pdg_drop_ratio", 0.1)

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
                # captions may be a pre-tokenized Tensor[B, L] (when the dataset
                # has a tokenizer) or a plain list[str] (fallback / no tokenizer).
                if isinstance(captions, torch.Tensor):
                    caption_chunks = [captions[i*spg:(i+1)*spg] for i in range(n_gpus)]
                else:
                    caption_chunks = [captions[i*spg:(i+1)*spg] for i in range(n_gpus)]

                # Anneal t_mu linearly from t_mu_start → t_mu_end.
                if t_mu_anneal_steps > 0:
                    frac = min(global_step, t_mu_anneal_steps) / t_mu_anneal_steps
                    t_mu = t_mu_start + (t_mu_end - t_mu_start) * frac

                # SPRINT coin-flips — one decision shared across all GPUs.
                # skip_middle takes precedence (PDG path-drop training).
                skip_middle = torch.rand(1).item() < pdg_drop_ratio
                use_mask    = (not skip_middle) and torch.rand(1).item() < sprint_mask_ratio

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
                        t_shift_mode=t_shift_mode,
                        use_mask=use_mask,
                        skip_middle=skip_middle,
                        uncond_token_ids=uncond_token_ids,
                    )

                # 2. Backward — per-GPU DINO model passed explicitly
                with record_function("backward"):
                    if cfg_amp_enabled:
                        _teacher_models = ema_models if ema_models else wrapper.models
                        raw_results = wrapper.run_concurrent(
                            lambda gpu_id: cfg_amplifying_backward_fn(
                                gpu_id, wrapper.models[gpu_id], outputs[gpu_id],
                                teacher_model=_teacher_models[gpu_id],
                                cfg_scale=cfg_amp_scale,
                                uncond_token_ids=uncond_token_ids,
                                dino_model=dino_models[gpu_id],
                                dino_threshold=dino_threshold,
                                dino_strength=dino_strength,
                                accum_steps=cfg["accum"],
                            )
                        )
                    else:
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
                        if cfg.get("empty_cache_on_step", False):
                            torch.cuda.empty_cache()  # flush reserved-but-free memory; useful when searching for max batch size, disable once found

                    # EMA update — runs after every optimizer step.
                    if cfg_amp_enabled and ema_models:
                        with torch.no_grad():
                            for gpu_id in range(n_gpus):
                                for ema_p, p in zip(
                                    ema_models[gpu_id].parameters(),
                                    wrapper.models[gpu_id].parameters(),
                                ):
                                    ema_p.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

                lr = wrapper.last_lr
                pbar.set_postfix(loss=f"{total_loss:.4f}", mse=f"{mse_loss:.4f}", dino=f"{dino_loss:.4f}", lr=f"{lr:.2e}", t_mu=f"{t_mu:.3f}", step=global_step)

                # Deferred CSV write
                csv_writer.writerow([global_step, f"{total_loss:.6f}", f"{mse_loss:.6f}", f"{dino_loss:.6f}", f"{lr:.2e}", f"{t_mu:.4f}", f"{time.time()-t0:.1f}"])
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

                    # Default schedule_mu matches the *current* (annealed) t_mu
                    # so previews use the same timestep density the model was
                    # trained on at this point in training.
                    # Set `inference_schedule_mu` in config to override (use
                    # null for uniform-dt / legacy sampling).
                    inference_schedule_mu = cfg.get("inference_schedule_mu", t_mu)

                    preview_results = wrapper.forward(
                        preview_chunks,
                        forward_fn=preview_fn,
                        eval_mode=True,
                        inference_cfg_and_steps=cfg["inference_cfg_and_steps"],
                        n_samples=preview_spg,
                        tokenizer=tokenizer,
                        max_text_len=max_text_len,
                        schedule_mu=inference_schedule_mu,
                    )

                    # Each GPU returns [(n_combos+1)*preview_spg, C, H, W].
                    # Cat across GPUs then lay out as nrow=preview_spg*n_gpus so
                    # each row-group is one CFG combo (+ real at the bottom).
                    all_images = torch.cat(preview_results, dim=0)
                    img_path = f"{cfg['preview_path']}/step_{global_step}.{ext}"
                    grid = make_grid((all_images + 1) / 2, nrow=preview_spg)
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

def load_config(path: str) -> dict:
    """Load a JSON or YAML config file.

    YAML files (``.yaml`` / ``.yml``) support comments (``#``) and are
    otherwise equivalent to JSON for this trainer.  The ``pyyaml`` package
    must be installed (``pip install pyyaml``) to use YAML configs.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "pyyaml is required for YAML configs: pip install pyyaml"
            ) from e
        with open(path) as f:
            return yaml.safe_load(f)
    else:
        with open(path) as f:
            return json.load(f)


if __name__ == "__main__":
    CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config_pixel_space.json"
    print(f"Loading config from: {CONFIG_PATH}")
    TRAINING_CONFIG = load_config(CONFIG_PATH)

    train(TRAINING_CONFIG)
