"""krea2_trainer.py — Flow-matching fine-tuner for Krea-2 (K2) with LoRA.

Architecture:
  - Text encoder:   Qwen3VLConditioner (Qwen3-VL-4B-Instruct, frozen)
  - VAE:            QwenAutoencoder (Qwen-Image f8-16c, frozen)
  - Diffusion model: SingleStreamDiT (K2 MMDiT, LoRA on all nn.Linear)

Training objective:
  Pure flow-matching (RF / rectified flow) in latent-patch token space.
  The model predicts the velocity field v in K2's convention:

    x_t = (1 - t) * x0_clean + t * x0_noise       t ∈ [0, 1]
    dx/dt = x0_noise - x0_clean  =:  v_target

  The Euler update at inference time is:
    x_{t - dt} = x_t + (t_prev - t_curr) * v       (t_prev < t_curr)

  so v points from clean toward noise and the negative-dt step moves the
  latent toward clean image.  This matches the K2 sampling.py convention
  exactly.

  No OT coupling, no perceptual loss.

LoRA strategy:
  - Every nn.Linear inside SingleStreamDiT is replaced with a LoRALinear.
    The frozen base weight lives as a buffer; only lora_A / lora_B are
    Parameters.
  - All other DiT parameters (RMSNorm scales, modulation tensors, last-layer
    bias, positional encoding) are fully fine-tuned.
  - QwenAutoencoder and Qwen3VLConditioner are frozen entirely.

Run:
    python krea2_trainer.py
    python krea2_trainer.py config_krea2.json
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
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from ramtorch.multi_gpu import MultiGPUWrapper

from k2.mmdit import SingleStreamDiT, SingleMMDiTConfig
from k2.encoder import Qwen3VLConditioner, TextEncoderConfig
from k2.autoencoder import QwenAutoencoder
from k2.sampling import prepare, timesteps as k2_timesteps
from k2.lora import inject_lora, lora_state_dict, trainable_param_count

from src.dataloaders.parquet_dataloader import ParquetTextImageDataset
from src.dataloaders.bucketing_logic import _bucket_generator

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Default config path
# ---------------------------------------------------------------------------
CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config_krea2.json"


# ---------------------------------------------------------------------------
# Model configs (mirrors inference.py)
# ---------------------------------------------------------------------------
MMDIT_CONFIGS = {
    "large_wide": SingleMMDiTConfig(
        features=6144,
        tdim=256,
        txtdim=2560,
        heads=48,
        kvheads=12,
        multiplier=4,
        layers=28,
        patch=2,
        channels=16,
        txtheads=20,
        txtkvheads=20,
        txtlayers=12,
    ),
}

ENCODER_CONFIGS = {
    "qwen3_vl_4b": TextEncoderConfig(model_id="Qwen/Qwen3-VL-4B-Instruct"),
}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _strip_compiled_keys(sd: dict) -> dict:
    prefix = "_orig_mod."
    return {k.replace(prefix, "") if prefix in k else k: v for k, v in sd.items()}


def save_lora_checkpoint(model: nn.Module, path: str):
    """Save only trainable parameters (LoRA adapters + unfrozen norms/mods)."""
    sd = {
        k: v.cpu()
        for k, v in model.state_dict().items()
        # Grab LoRA adapter tensors and all other trainable params.
        # base_weight / base_bias are buffers (requires_grad=False) — skip.
        if "lora_A" in k or "lora_B" in k or _param_is_trainable(model, k)
    }
    sd = _strip_compiled_keys(sd)
    if path.endswith((".safetensors", ".sft")):
        save_file(sd, path)
    else:
        torch.save(sd, path)
    print(f"[ckpt] Saved {len(sd)} tensors → {path}")


def _param_is_trainable(model: nn.Module, key: str) -> bool:
    """True if the leaf named_parameter at *key* requires grad."""
    for name, param in model.named_parameters():
        if name == key:
            return param.requires_grad
    return False


def load_lora_checkpoint(model: nn.Module, path: str):
    """Load a LoRA checkpoint (partial state dict — missing keys are fine)."""
    if path.endswith((".safetensors", ".sft")):
        sd = load_file(path, device="cpu")
    else:
        sd = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"[ckpt] {len(unexpected)} unexpected keys: {unexpected[:5]} ...")
    print(f"[ckpt] Loaded LoRA ckpt from {path} "
          f"({len(sd)} tensors, {len(missing)} missing keys).")


# ---------------------------------------------------------------------------
# VAE encode helper (no gradient, bfloat16)
# ---------------------------------------------------------------------------

@torch.no_grad()
def vae_encode(ae: QwenAutoencoder, pixels: torch.Tensor) -> torch.Tensor:
    """Encode a batch of pixel images [-1, 1] to latent space.

    QwenAutoencoder only exposes a decode() method publicly; we call
    the underlying diffusers AutoencoderKLQwenImage for encoding.
    pixels: [B, 3, H, W] float in [-1, 1].
    Returns: [B, 16, H/8, W/8] bfloat16 latent.
    """
    with torch.autocast("cuda", torch.bfloat16):
        # diffusers VAE expects images in [-1, 1]
        x = pixels.to(ae.ae.device, non_blocking=True)
        # The encode path uses the internal diffusers model directly.
        x5d = x.unsqueeze(2)  # [B, C, 1, H, W] for qwen-image AE
        enc = ae.ae.encode(x5d)
        # AutoencoderKLQwenImage returns a distribution; sample the mode.
        latent = enc.latent_dist.mode()          # [B, 16, 1, H/8, W/8]
        latent = latent.squeeze(2)               # [B, 16, H/8, W/8]
        # Normalize with the registered mean/std buffers (same as decode path).
        mean = ae.latents_mean.squeeze(2)        # [1, 16, 1, 1]
        std  = ae.latents_std.squeeze(2)         # [1, 16, 1, 1]
        latent = (latent - mean) / std
        return latent.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Timestep sampling with optional mu-shift (K2 convention: t ∈ [0, 1])
# ---------------------------------------------------------------------------

def sample_timesteps(
    B: int,
    device: torch.device,
    mu: float = 0.0,
) -> torch.Tensor:
    """Sample B timesteps from a logit-normal distribution shifted by mu.

    mu=0  → logit-normal ≈ Beta(1,1) ≈ Uniform (no shift).
    mu>0  → mass shifted toward t=1 (more noisy steps); K2 training default.

    Returns: [B] float32 tensor in (0, 1).
    """
    u = torch.rand(B, device=device)
    # Logit-normal shift: t = sigmoid(mu + sigma * logit(u)), sigma ≈ 1.
    if mu == 0.0:
        return u
    # Clamp to avoid logit singularities at 0/1.
    u = u.clamp(1e-4, 1 - 1e-4)
    logit_u = torch.log(u / (1 - u))
    t = torch.sigmoid(mu + logit_u)
    return t


# ---------------------------------------------------------------------------
# Per-GPU forward pass
# ---------------------------------------------------------------------------

def forward_fn(
    gpu_id: int,
    dit: SingleStreamDiT,
    pixels: torch.Tensor,           # [B, 3, H, W] float32 [-1,1]
    captions: list[str],
    *,
    aes: list,
    encoders: list,
    uncond_ratio: float,
    t_mu: float,
) -> tuple:
    """Encode → patchify → forward → return (v_pred_tokens, v_target_tokens, t, txtmask).

    K2 flow-matching convention:
        x_t = (1 - t) * x0_clean + t * x0_noise
        v_target = x0_noise_tokens - x0_clean_tokens   (dx_t / dt)

    The model predicts v_pred in token/patch space.  Loss = MSE(v_pred, v_target).
    """
    device = f"cuda:{gpu_id}"
    ae = aes[gpu_id]
    # Encoder lives on GPU 0; outputs are moved to the target device below.
    encoder = encoders[0]

    # ---- Text conditioning -----------------------------------------------
    B = pixels.shape[0]
    # CFG uncond dropout
    dropped = [
        "" if (torch.rand(1).item() < uncond_ratio) else c
        for c in captions
    ]
    with torch.no_grad():
        txt, txtmask = encoder(dropped)          # [B, L, 12, 2560], [B, L] bool
        txt = txt.to(device, non_blocking=True)
        txtmask = txtmask.to(device, non_blocking=True)

    # ---- VAE encode -------------------------------------------------------
    x0_clean = vae_encode(ae, pixels.to(device))   # [B, 16, H/8, W/8] bf16

    # ---- Noise + interpolation -------------------------------------------
    x0_noise = torch.randn_like(x0_clean)
    t = sample_timesteps(B, device=device, mu=t_mu)  # [B]

    # x_t = (1-t)*clean + t*noise
    t4 = t[:, None, None, None].to(x0_clean.dtype)
    x_t = (1.0 - t4) * x0_clean + t4 * x0_noise    # [B, 16, H/8, W/8]

    # ---- Patchify (K2 prepare()) -----------------------------------------
    # prepare() patchifies img and builds position ids + combined mask.
    txtlen = txt.shape[1]
    patch  = dit.config.patch

    x_t_tok, pos, mask = prepare(x_t, txtlen, patch, txtmask)  # tokens: [B, hw, C*p*p]
    v_target_tok = rearrange(
        x0_noise - x0_clean,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=patch, pw=patch,
    )  # [B, hw, C*p*p] — velocity target in token space

    # ---- DiT forward -----------------------------------------------------
    with torch.autocast("cuda", torch.bfloat16):
        v_pred_tok = dit(img=x_t_tok, context=txt, t=t, pos=pos, mask=mask)
        # dit() returns output only for image tokens: [B, hw, C*p*p]

    # Also stash x0_clean for the preview (pixels in latent space)
    return v_pred_tok, v_target_tok.to(v_pred_tok.dtype), t, txtmask, x0_clean, list(captions)


# ---------------------------------------------------------------------------
# Per-GPU backward pass
# ---------------------------------------------------------------------------

def backward_fn(
    gpu_id: int,
    dit: SingleStreamDiT,
    fwd_output: tuple,
    accum_steps: int = 1,
) -> float:
    """MSE on velocity tokens → backward → return scalar loss."""
    v_pred, v_target, t, _txtmask, _x0_clean, _captions = fwd_output

    loss = F.mse_loss(v_pred, v_target)
    (loss / accum_steps).backward()
    return loss.item()


# ---------------------------------------------------------------------------
# Preview / inference (Euler + CFG, returns pixel grid)
# ---------------------------------------------------------------------------

@torch.no_grad()
def preview_fn(
    gpu_id: int,
    dit: SingleStreamDiT,
    ae: QwenAutoencoder,
    encoder: Qwen3VLConditioner,
    x0_clean_latent: torch.Tensor,  # [B, 16, H/8, W/8] — already encoded
    captions: list[str],
    steps: int = 28,
    cfg_scale: float = 4.5,
    n_samples: int = 4,
    mu: float | None = None,
    y1: float = 0.5,
    y2: float = 1.15,
    minres: int = 256,
    maxres: int = 1280,
) -> torch.Tensor:
    """Run Euler+CFG sampling and return a float32 CPU tensor [N, 3, H, W]
    in [-1, 1] with generated samples followed by decoded ground-truth latents."""
    device = f"cuda:{gpu_id}"
    patch = dit.config.patch
    n_samples = min(n_samples, x0_clean_latent.shape[0])

    x0_ref = x0_clean_latent[:n_samples].to(device)  # [n, 16, H/8, W/8]
    _, _, latent_h, latent_w = x0_ref.shape

    # Decode ground-truth latents for the reference row.
    gt_pixels = ae.decode(x0_ref.to(torch.bfloat16)).clamp(-1, 1).cpu().float()

    # Encode prompts (conditional + unconditional).
    prompts = list(captions[:n_samples])
    txt,    txtmask    = encoder(prompts)
    untxt,  untxtmask  = encoder([""] * n_samples)
    txt    = txt.to(device)
    txtmask= txtmask.to(device)
    untxt  = untxt.to(device)
    untxtmask = untxtmask.to(device)

    # Start from pure noise in latent space.
    noise = torch.randn_like(x0_ref)

    # Build position/mask for cond and uncond paths.
    img_tok, pos, mask   = prepare(noise, txt.shape[1],   patch, txtmask)
    _,       unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)

    # Resolution-aware timestep schedule (t: 1 → 0).
    x1_res = (minres // (ae.compression * patch)) ** 2
    x2_res = (maxres // (ae.compression * patch)) ** 2
    ts = k2_timesteps(img_tok.shape[1], steps, x1_res, x2_res, y1=y1, y2=y2, mu=mu)

    # Euler integration.
    img = img_tok
    with torch.autocast("cuda", torch.bfloat16):
        for tcurr, tprev in zip(ts[:-1], ts[1:]):
            t_vec = torch.full((n_samples,), tcurr, dtype=img.dtype, device=device)
            cond   = dit(img=img, context=txt,   t=t_vec, pos=pos,   mask=mask)
            uncond = dit(img=img, context=untxt, t=t_vec, pos=unpos, mask=unmask)
            v = uncond + cfg_scale * (cond - uncond)
            img = img + (tprev - tcurr) * v

    # Unpatchify → latent → decode.
    latent = rearrange(
        img,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch, pw=patch,
        h=latent_h // patch,
        w=latent_w // patch,
    )
    pixels_out = ae.decode(latent.to(torch.bfloat16))   # [B, 3, H, W] in [-1,1]
    pixels_out = pixels_out.clamp(-1, 1).cpu().float()

    return torch.cat([pixels_out, gt_pixels], dim=0)  # [2*n_samples, 3, H, W]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict):
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPU(s).")

    os.makedirs(cfg["ckpt_path"], exist_ok=True)
    os.makedirs(cfg["preview_path"], exist_ok=True)
    _cfg_dest = os.path.join(cfg["ckpt_path"], os.path.basename(CONFIG_PATH))
    shutil.copy(CONFIG_PATH, _cfg_dest)
    print(f"Config saved → {_cfg_dest}")

    # ------------------------------------------------------------------
    # Load frozen models (one instance per GPU for ae/encoder)
    # ------------------------------------------------------------------
    dtype = torch.bfloat16

    ae_id        = cfg.get("ae_model_id", "Qwen/Qwen-Image")
    encoder_id   = cfg.get("encoder_model_id", "Qwen/Qwen3-VL-4B-Instruct")
    enc_cfg_name = cfg.get("encoder_config", "qwen3_vl_4b")
    dit_cfg_name = cfg.get("mmdit_config", "large_wide")

    print("Loading VAE...")
    # One VAE per GPU (cheap copy — buffers only).
    aes: list[QwenAutoencoder] = []
    base_ae = QwenAutoencoder()
    base_ae.ae = base_ae.ae.to(dtype).eval().requires_grad_(False)
    for gpu_id in range(n_gpus):
        a = copy.deepcopy(base_ae).to(f"cuda:{gpu_id}").eval()
        a.requires_grad_(False)
        aes.append(a)
    del base_ae
    print("  VAE ready.")

    print("Loading text encoder (Qwen3-VL-4B)...")
    # Encoder is large; keep one on GPU 0, run encoding there, scatter results.
    _enc_cfg = ENCODER_CONFIGS[enc_cfg_name]
    encoder_gpu0 = Qwen3VLConditioner(
        version=encoder_id,
        max_length=_enc_cfg.max_length,
        select_layers=_enc_cfg.select_layers,
    ).to("cuda:0").eval().requires_grad_(False)
    # For multi-GPU we keep the encoder on GPU 0 and move its outputs per-GPU
    # inside forward_fn.
    encoders = [encoder_gpu0] + [None] * (n_gpus - 1)  # only GPU 0 has encoder
    print("  Encoder ready.")

    # ------------------------------------------------------------------
    # DiT — load base weights, inject LoRA
    # ------------------------------------------------------------------
    print("Loading DiT...")
    dit_cfg = MMDIT_CONFIGS[dit_cfg_name]
    with torch.device("meta"):
        base_dit = SingleStreamDiT(dit_cfg)

    mmdit_ckpt = cfg.get("mmdit_checkpoint")
    if mmdit_ckpt:
        print(f"  Loading weights from {mmdit_ckpt}...")
        base_dit.load_state_dict(load_file(mmdit_ckpt), strict=True, assign=True)
    else:
        print("  [warn] No mmdit_checkpoint specified — training from random init.")
        # Move off meta for random init.
        base_dit = base_dit.to("cpu")
        for p in base_dit.parameters():
            nn.init.normal_(p.data, std=0.02)

    # Inject LoRA — replaces all nn.Linear layers in the DiT.
    lora_rank   = cfg.get("lora_rank", 32)
    lora_alpha  = cfg.get("lora_alpha", float(lora_rank))
    lora_exclude = tuple(cfg.get("lora_exclude_prefixes", []))
    inject_lora(base_dit, rank=lora_rank, alpha=lora_alpha, exclude_prefixes=lora_exclude)

    trainable_n, total_n = trainable_param_count(base_dit)
    print(
        f"  DiT: {trainable_n/1e6:.1f}M trainable / "
        f"{total_n/1e6:.1f}M total params "
        f"({100*trainable_n/max(total_n,1):.1f}% trained)."
    )

    def dit_factory():
        """Called once per GPU by MultiGPUWrapper — deep-copy of injected DiT."""
        return copy.deepcopy(base_dit)

    # ------------------------------------------------------------------
    # Optimizer wraps only trainable params (LoRA + norms/mods).
    # MultiGPUWrapper will call .parameters() on the model; we filter
    # requires_grad inside a custom optimizer_factory.
    # ------------------------------------------------------------------
    lr          = cfg.get("lr", 1e-4)
    weight_decay= cfg.get("weight_decay", 1e-4)
    warmup_steps= cfg.get("warmup", 200)
    accum_steps = cfg.get("accum", 4)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

    def optimizer_factory(params):
        # params here is model.parameters() from MultiGPUWrapper;
        # filter to trainable only.
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

    # Move AEs/encoders to match each GPU's DiT dtype (bfloat16).
    for gpu_id in range(n_gpus):
        wrapper.models[gpu_id].to(dtype)

    # ------------------------------------------------------------------
    # Checkpoint save/load
    # ------------------------------------------------------------------
    def _save_checkpoint(path: str):
        sd = lora_state_dict(wrapper.models[0])
        sd = _strip_compiled_keys(sd)
        if path.endswith((".safetensors", ".sft")):
            save_file(sd, path)
        else:
            torch.save(sd, path)
        print(f"[ckpt] Saved {len(sd)} tensors → {path}")

    lora_ckpt = cfg.get("lora_checkpoint")
    if lora_ckpt:
        for gpu_id in range(n_gpus):
            load_lora_checkpoint(wrapper.models[gpu_id], lora_ckpt)
        print(f"  LoRA checkpoint loaded on all {n_gpus} GPU(s).")
    else:
        _save_checkpoint(os.path.join(cfg["ckpt_path"], "untrained_lora.safetensors"))

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
        base_res_weights=parquet_cfg.get("base_resolution_weights", None),
        ratio_cutoff=parquet_cfg.get("ratio_cutoff", 2.0),
        resolution_step=parquet_cfg.get("resolution_step", 64),
        shuffle_tags=parquet_cfg.get("shuffle_tags", True),
        tag_drop_percentage=parquet_cfg.get("tag_drop_percentage", 0.1),
        uncond_percentage=0.0,  # handled in forward_fn
        seed=cfg.get("seed", 42),
        rank=0,
        num_gpus=1,
        offset=parquet_cfg.get("offset", 0),
        tokenizer=None,          # K2 encoder handles tokenization internally
        max_text_len=0,
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
    # Training config
    # ------------------------------------------------------------------
    global_step      = cfg.get("initial_global_step", 0)
    eval_interval    = cfg.get("eval_interval", 200)
    save_every       = cfg.get("save_every_n_steps", 1000)
    log_every        = cfg.get("log_every_n_steps", 10)
    uncond_ratio     = cfg.get("uncond_ratio", 0.1)
    t_mu             = cfg.get("t_mu", 0.0)
    preview_spg      = cfg.get("preview_samples_per_gpu", 4)
    preview_cfg      = cfg.get("preview_cfg_scale", 4.5)
    preview_steps    = cfg.get("preview_steps", 28)
    preview_quality  = cfg.get("preview_quality", 95)
    max_steps        = cfg.get("max_steps", 0)    # 0 = run forever
    master_seed      = cfg.get("seed", 42)

    # ------------------------------------------------------------------
    # CSV loss log
    # ------------------------------------------------------------------
    csv_path   = os.path.join(cfg["ckpt_path"], "loss_log.csv")
    csv_file   = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        csv_writer.writerow(["step", "loss", "lr", "t_mu", "time"])
    t0 = time.time()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    torch.manual_seed(master_seed)
    epoch = 0

    while True:
        epoch += 1
        torch.manual_seed(master_seed + epoch)
        for m in wrapper.models:
            m.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch_data in enumerate(pbar):
            batch_data = batch_data[0]      # dummy_collate_fn wraps in list
            images, captions, _idx, _lw = batch_data[:4]
            # images: [B*n_gpus, 3, H, W], captions: list[str]

            spg = images.shape[0] // n_gpus
            if spg == 0:
                continue

            image_chunks   = [images[i*spg:(i+1)*spg] for i in range(n_gpus)]
            # captions may be list[str] since we pass tokenizer=None above.
            caption_chunks = [captions[i*spg:(i+1)*spg] for i in range(n_gpus)]

            # ---------- Forward -------------------------------------------
            # MultiGPUWrapper.forward() unpacks each tuple chunk as positional
            # args after (gpu_id, model), so signature is:
            #   forward_fn(gpu_id, dit, images, captions, **kwargs)
            outputs = wrapper.forward(
                list(zip(image_chunks, caption_chunks)),
                forward_fn=forward_fn,
                aes=aes,
                encoders=encoders,
                uncond_ratio=uncond_ratio,
                t_mu=t_mu,
            )

            # ---------- Backward ------------------------------------------
            raw_results = wrapper.run_concurrent(
                lambda gpu_id: backward_fn(
                    gpu_id,
                    wrapper.models[gpu_id],
                    outputs[gpu_id],
                    accum_steps=accum_steps,
                )
            )

            total_loss = sum(r for r in raw_results) / n_gpus

            # ---------- Optimizer step ------------------------------------
            if (batch_idx + 1) % accum_steps == 0:
                wrapper.reduce_grads()
                wrapper.clip_grads()
                wrapper.optimizer_step()
                torch.cuda.synchronize()

            lr_now = wrapper.last_lr
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                lr=f"{lr_now:.2e}",
                t_mu=f"{t_mu:.3f}",
                step=global_step,
            )

            csv_writer.writerow([
                global_step,
                f"{total_loss:.6f}",
                f"{lr_now:.2e}",
                f"{t_mu:.4f}",
                f"{time.time() - t0:.1f}",
            ])
            if global_step % log_every == 0:
                csv_file.flush()

            # ---------- Step checkpoint -----------------------------------
            if save_every > 0 and global_step > 0 and global_step % save_every == 0:
                _save_checkpoint(
                    os.path.join(cfg["ckpt_path"], f"lora_step_{global_step}.safetensors")
                )

            # ---------- Preview -------------------------------------------
            if global_step % eval_interval == 0:
                for m in wrapper.models:
                    m.eval()

                all_rows = []
                for gpu_id in range(n_gpus):
                    # outputs[gpu_id] = (v_pred, v_target, t, txtmask, x0_clean, captions)
                    _x0_clean_latent = outputs[gpu_id][4]  # [B,16,H/8,W/8]
                    _captions_gpu    = outputs[gpu_id][5]
                    rows = preview_fn(
                        gpu_id,
                        wrapper.models[gpu_id],
                        aes[gpu_id],
                        encoders[0],
                        _x0_clean_latent,
                        _captions_gpu,
                        steps=preview_steps,
                        cfg_scale=preview_cfg,
                        n_samples=min(preview_spg, spg),
                        mu=cfg.get("preview_mu", None),
                    )
                    all_rows.append(rows)

                combined = torch.cat(all_rows, dim=0)  # [2*n*G, C, H, W]
                grid = make_grid((combined + 1) / 2, nrow=preview_spg * n_gpus)

                ext = "png" if preview_quality >= 100 else "jpg"
                img_path = f"{cfg['preview_path']}/step_{global_step}.{ext}"
                if _PIL_AVAILABLE:
                    _grid_np = (grid.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                    Image.fromarray(_grid_np).save(img_path)
                else:
                    from torchvision.utils import save_image
                    save_image(grid, img_path)
                print(f"[preview] Saved {img_path}")

                for m in wrapper.models:
                    m.train()

            global_step += 1
            if max_steps > 0 and global_step >= max_steps:
                print(f"Reached max_steps={max_steps}. Saving final checkpoint.")
                _save_checkpoint(
                    os.path.join(cfg["ckpt_path"], f"lora_step_{global_step}_final.safetensors")
                )
                csv_file.close()
                return

    csv_file.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    train(cfg)
