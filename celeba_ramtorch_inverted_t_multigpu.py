"""celeba_ramtorch_inverted_t_multigpu.py — CelebA flow-matching trainer using MultiGPUWrapper.

Uses the split forward/backward pattern from demo_multi_gpu_single_thread.py,
based on the model and training logic from celeba_ramtorch_inverted_t.py.

Run:
    python celeba_ramtorch_inverted_t_multigpu.py
    python celeba_ramtorch_inverted_t_multigpu.py config.json

All settings are in config.json.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext

import csv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.optim.lr_scheduler import LinearLR
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from ramtorch import AdamW
# from ramtorch.helpers import replace_linear_with_ramtorch
from ramtorch.multi_gpu import MultiGPUWrapper

import math

torch.manual_seed(0)

# Patch size — set from config at startup before training begins
PATCH_SIZE: int = 16


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
# Kernels & helpers
# ---------------------------------------------------------------------------


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset=0):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_mrope(q, k, position_ids, inv_freq):
    """Apply M-RoPE with 3 independent axes (time, height, width).

    Args:
        q, k:          [B, heads, seq, head_dim]
        position_ids:  [3, B, seq]  — (time, height, width) per token
        inv_freq:      [dim_per_axis // 2]  — shared across all 3 axes

    head_dim is split into 3 equal slices of dim_per_axis each.
    Any remaining dims (head_dim % 3 != 0) are left unrotated.
    """
    head_dim = q.shape[-1]
    dim_per_axis = (head_dim // 3 // 2) * 2  # round down to even

    # position_ids: [3, B, seq] -> per-axis freqs: [B, seq, dim_per_axis]
    # inv_freq: [dim_per_axis // 2]
    def axis_cos_sin(pos):  # pos: [B, seq]
        # [B, seq, 1] x [1, 1, dim//2] -> [B, seq, dim//2]
        freqs = pos.unsqueeze(-1).float() * inv_freq.to(pos.device).unsqueeze(0).unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)  # [B, seq, dim_per_axis]
        return emb.cos(), emb.sin()

    cos_t, sin_t = axis_cos_sin(position_ids[0])  # time
    cos_h, sin_h = axis_cos_sin(position_ids[1])  # height
    cos_w, sin_w = axis_cos_sin(position_ids[2])  # width

    # Stack into [B, seq, 3*dim_per_axis] then unsqueeze heads dim
    cos = torch.cat([cos_t, cos_h, cos_w], dim=-1).unsqueeze(1)  # [B, 1, seq, 3*dim_per_axis]
    sin = torch.cat([sin_t, sin_h, sin_w], dim=-1).unsqueeze(1)

    rotated_dims = 3 * dim_per_axis
    q_rot, q_pass = q[..., :rotated_dims], q[..., rotated_dims:]
    k_rot, k_pass = k[..., :rotated_dims], k[..., rotated_dims:]

    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_out = torch.cat([q_rot, q_pass], dim=-1)
    k_out = torch.cat([k_rot, k_pass], dim=-1)
    return q_out.to(q.dtype), k_out.to(k.dtype)


def build_mrope_position_ids(
    B: int,
    num_h_patches: int,
    num_w_patches: int,
    device,
    n_text: int = 0,
    n_time: int = 1,
    n_class: int = 1,
    n_reg: int = 0,
    pos_jitter: int = 0,
) -> torch.Tensor:
    """Build M-RoPE position ids for the full token sequence.

    Token layout: [text (n_text), time (n_time), class (n_class), register (n_reg), patches (H*W)]

    All prefix tokens sit on the diagonal [i, i, i] in sequence order.
    Image patches start at offset = n_text + n_time + n_class + n_reg + pos_jitter,
    with the time axis frozen at that offset and spatial axes expanding from it.

    pos_jitter: random integer added to the spatial start of patches (training only).

    Returns:
        position_ids: [3, B, seq_len]
    """
    num_patches = num_h_patches * num_w_patches
    n_prefix = n_text + n_time + n_class + n_reg
    seq_len = n_prefix + num_patches

    t_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
    h_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
    w_ids = torch.zeros(seq_len, dtype=torch.long, device=device)

    # All prefix tokens: diagonal [i, i, i]
    diag = torch.arange(n_prefix, dtype=torch.long, device=device)
    t_ids[:n_prefix] = diag
    h_ids[:n_prefix] = diag
    w_ids[:n_prefix] = diag

    # Image patches: time frozen at patch_start, spatial expands from patch_start
    patch_start = n_prefix + pos_jitter
    rows = torch.arange(num_h_patches, device=device).repeat_interleave(num_w_patches)
    cols = torch.arange(num_w_patches, device=device).repeat(num_h_patches)
    t_ids[n_prefix:] = patch_start
    h_ids[n_prefix:] = patch_start + rows
    w_ids[n_prefix:] = patch_start + cols

    # Stack and expand over batch: [3, seq_len] -> [3, B, seq_len]
    position_ids = torch.stack([t_ids, h_ids, w_ids], dim=0)  # [3, seq_len]
    position_ids = position_ids.unsqueeze(1).expand(-1, B, -1)  # [3, B, seq_len]
    return position_ids


def soft_clamp(x, scale, alpha, shift):
    return scale * F.tanh(x * alpha) + shift


class MRoPEEmbedding(nn.Module):
    """Stores inv_freq for a single axis; shared across all 3 M-RoPE axes."""
    def __init__(self, head_dim, max_seq_len=2048):
        super().__init__()
        # dim_per_axis = largest even number <= head_dim // 3
        dim_per_axis = (head_dim // 3 // 2) * 2
        inv_freq = 1.0 / (max_seq_len ** (torch.arange(0, dim_per_axis, 2).float() / dim_per_axis))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, q, k):
        """Apply M-RoPE. position_ids: [3, B, seq]. q,k: [B, heads, seq, head_dim]."""
        return apply_mrope(q, k, position_ids, self.inv_freq)


class PinchedLinear(nn.Module):
    """Low-rank bottleneck projection: in_dim -> pinch_dim -> out_dim.

    The down-projection (in_dim -> pinch_dim) has no bias so the two matrices
    can be merged into a single full-rank weight (W = up @ down) after training.
    The up-projection (pinch_dim -> out_dim) keeps its bias.

    If pinch_dim == -1 this degenerates to a plain nn.Linear.
    """
    def __init__(self, in_dim: int, out_dim: int, pinch_dim: int = -1, bias: bool = True):
        super().__init__()
        self.pinch_dim = pinch_dim
        if pinch_dim == -1:
            self.proj = nn.Linear(in_dim, out_dim, bias=bias)
        else:
            self.down = nn.Linear(in_dim, pinch_dim, bias=False)  # no bias -> mergeable
            self.up   = nn.Linear(pinch_dim, out_dim, bias=bias)

    def forward(self, x):
        if self.pinch_dim == -1:
            return self.proj(x)
        return self.up(self.down(x))

    def merged_weight(self) -> torch.Tensor:
        """Return up.weight @ down.weight — the equivalent full-rank matrix."""
        if self.pinch_dim == -1:
            return self.proj.weight
        return self.up.weight @ self.down.weight


class SoftClamp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.use_compiled = False

    def forward(self, x):
        if self.use_compiled:
            return torch.compile(soft_clamp)(x, self.scale, self.alpha, self.shift)
        else:
            return soft_clamp(x, self.scale, self.alpha, self.shift)


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, max_seq_len=2048, use_rope=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=True)
        self.layer_norm = SoftClamp(dim)
        self.rope = MRoPEEmbedding(self.head_dim, max_seq_len)
        self.q_norm = SoftClamp(dim)
        self.k_norm = SoftClamp(dim)
        self.add_module("layer_norm", self.layer_norm)

        nn.init.zeros_(self.wo.weight)
        self.use_compiled = False
        self.use_rope = use_rope

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, attention_mask=None, position_ids=None):
        residual = x
        x = self.layer_norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope and position_ids is not None:
            q, k = self.rope(position_ids, q, k)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        out = self.wo(out)
        return out + residual


class GLU(nn.Module):
    def __init__(self, dim, exp_fac=4):
        super().__init__()
        self.wi_0 = nn.Linear(dim, dim * exp_fac, bias=False)
        self.wi_1 = nn.Linear(dim, dim * exp_fac, bias=False)
        self.wo = nn.Linear(dim * exp_fac, dim, bias=True)
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        nn.init.zeros_(self.wo.weight)
        self.use_compiled = False

    @property
    def device(self):
        return next(self.parameters()).device

    def _fwd_glu(self, x, residual):
        return self.wo(F.silu(self.wi_0(x)) * self.wi_1(x)) + residual

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        if self.use_compiled:
            return torch.compile(self._fwd_glu)(x, residual)
        else:
            return self._fwd_glu(x, residual)


class _Block(nn.Module):
    """One (attn + glu) transformer block — a clean compile boundary."""
    def __init__(self, attn: AttentionBlock, glu: GLU):
        super().__init__()
        self.attn = attn
        self.glu = glu

    def forward(self, x, attention_mask=None, position_ids=None):
        x = self.attn(x, attention_mask, position_ids=position_ids)
        x = self.glu(x)
        return x


class TransformerNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        dim,
        num_layers,
        num_heads=8,
        exp_fac=4,
        rope_seq_length=2048,
        use_rope=True,
        final_head=True,
        input_proj=True,
        pinch_dim: int = -1,
        compile_blocks: bool = False,
    ):
        super().__init__()
        if input_proj:
            self.input_layer = nn.Linear(input_dim, dim)
        else:
            self.input_layer = nn.Identity()
            input_dim = dim
        blocks = [
            _Block(
                AttentionBlock(dim, num_heads, rope_seq_length, use_rope),
                GLU(dim, exp_fac),
            )
            for _ in range(num_layers)
        ]
        if compile_blocks:
            blocks = [torch.compile(b) for b in blocks]
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = SoftClamp(dim)
        if final_head:
            # out-proj: dim -> (pinch_dim ->) output_dim
            self.output_layer = PinchedLinear(dim, output_dim, pinch_dim=pinch_dim)
        else:
            self.output_layer = nn.Identity()

    def forward(self, x, attention_mask=None, act_ckpt=False, position_ids=None):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x, attention_mask, position_ids=position_ids)
        x = self.out_norm(x)
        x = self.output_layer(x)
        return x


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def image_flatten(latents, shuffle_size=16):
    return (
        rearrange(
            latents,
            "n c (h dh) (w dw) -> n (h w) (c dh dw)",
            dh=shuffle_size,
            dw=shuffle_size,
        ),
        latents.shape,
    )


def image_unflatten(latents, shape, shuffle_size=16):
    n, c, h, w = shape
    return rearrange(
        latents,
        "n (h w) (c dh dw) -> n c (h dh) (w dw)",
        dh=shuffle_size,
        dw=shuffle_size,
        c=c,
        h=h // shuffle_size,
        w=w // shuffle_size,
    )


def sample_from_distribution(x, probabilities, n):
    indices = torch.multinomial(probabilities, n, replacement=True)
    return x[indices]


def create_distribution(num_points, device=None):
    x = torch.linspace(0, 1, num_points, device=device)
    probabilities = -7.7 * ((x - 0.5) ** 2) + 2
    probabilities /= probabilities.sum()
    return x, probabilities


# ---------------------------------------------------------------------------
# Flow model
# ---------------------------------------------------------------------------


class Flow(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        dim,
        num_layers,
        num_heads=8,
        exp_fac=4,
        rope_seq_length=784,
        class_count=10,
        cond_seq_len=40,
        pinch_dim: int = -1,
        # --- token config ---
        n_time_tokens: int = 1,        # timestep tokens: Linear(1, dim*n) -> [B, n, dim]
        n_register_tokens: int = 0,    # learnable register tokens appended before patches
        use_class_embed: bool = True,  # toggle class conditioning token
        use_text_embed: bool = False,  # toggle text token conditioning
        text_dim: int = 768,           # input dim of external text embeddings
        pos_jitter_range: int = 0,     # max random int offset on patch positions (0 = off)
        compile_blocks: bool = False,  # torch.compile each (attn+glu) block
    ):
        super().__init__()
        self.dim = dim
        self.class_count = class_count
        self.n_time_tokens = n_time_tokens
        self.n_register_tokens = n_register_tokens
        self.use_class_embed = use_class_embed
        self.use_text_embed = use_text_embed
        self.pos_jitter_range = pos_jitter_range
        self.cond_seq_len = cond_seq_len

        # in-proj: input_dim -> (pinch_dim ->) dim
        self.input_layer = PinchedLinear(input_dim, dim, pinch_dim=pinch_dim)

        # Timestep: scalar -> dim * n_time_tokens, reshaped to [B, n_time_tokens, dim]
        self.timestep_vector = nn.Linear(1, dim * n_time_tokens)

        # Class conditioning (optional)
        if use_class_embed:
            self.class_embed = nn.Linear(cond_seq_len, dim)
            # self.class_norm = SoftClamp(dim=dim)

        # Text conditioning (optional): project external embeddings -> dim
        if use_text_embed:
            self.text_proj = nn.Linear(text_dim, dim)
            # self.text_norm = SoftClamp(dim=dim)

        # Register tokens: learnable [1, n_reg, dim]
        if n_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, n_register_tokens, dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        self.transformer = TransformerNetwork(
            input_dim=dim,
            output_dim=output_dim,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            exp_fac=exp_fac,
            rope_seq_length=rope_seq_length,
            final_head=True,
            input_proj=False,
            pinch_dim=pinch_dim,
            compile_blocks=compile_blocks,
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.timestep_vector.weight)
        nn.init.zeros_(self.timestep_vector.bias)
        if self.use_class_embed:
            nn.init.zeros_(self.class_embed.weight)
            nn.init.zeros_(self.class_embed.bias)

    @property
    def device(self):
        return torch.cuda.current_device()

    def _build_prefix(self, x, t, condition=None, text_tokens=None):
        """Assemble prefix tokens: [text, time, class, register].
        Returns (prefix, n_text, n_time, n_class, n_reg).
        """
        B = x.shape[0]
        parts = []

        # text tokens (optional, variable length)
        n_text = 0
        if self.use_text_embed and text_tokens is not None:
            txt = self.text_proj(text_tokens.to(self.text_proj.weight.dtype))
            parts.append(txt)
            n_text = txt.shape[1]

        # time tokens: [B, dim*n_time] -> [B, n_time, dim]
        time_vec = self.timestep_vector(t.view(-1, 1)).view(B, self.n_time_tokens, self.dim)
        parts.append(time_vec)

        # class token (optional)
        n_class = 0
        if self.use_class_embed and condition is not None:
            class_vec = self.class_embed(condition.to(self.class_embed.weight.dtype))[:, None, :]
            parts.append(class_vec)
            n_class = 1

        # register tokens
        n_reg = 0
        if self.n_register_tokens > 0:
            parts.append(self.register_tokens.expand(B, -1, -1))
            n_reg = self.n_register_tokens

        prefix = torch.cat(parts, dim=1)  # [B, n_prefix, dim]
        return prefix, n_text, self.n_time_tokens, n_class, n_reg

    def forward(self, x, t, condition=None, attention_mask=None, text_tokens=None):
        B = x.shape[0]
        x_proj = self.input_layer(x)

        prefix, n_text, n_time, n_class, n_reg = self._build_prefix(x, t, condition, text_tokens)
        n_prefix = n_text + n_time + n_class + n_reg

        tokens = torch.cat((prefix, x_proj), dim=1)

        num_patches = x.shape[1]
        num_h = num_w = int(num_patches ** 0.5)

        # Positional jitter: random int in [0, pos_jitter_range] during training only
        jitter = 0
        if self.training and self.pos_jitter_range > 0:
            jitter = int(torch.randint(0, self.pos_jitter_range + 1, (1,)).item())

        position_ids = build_mrope_position_ids(
            B, num_h, num_w, device=x.device,
            n_text=n_text, n_time=n_time, n_class=n_class, n_reg=n_reg,
            pos_jitter=jitter,
        )

        output_tokens = self.transformer(tokens, attention_mask, position_ids=position_ids)
        # Strip prefix — only return patch predictions
        velocity_pred = output_tokens[:, n_prefix:, ...]
        return velocity_pred

    def euler_cfg(
        self,
        x,
        pos_cond=None,
        cfg_scale=4.0,
        num_steps=100,
        skip_last_n=0,
        return_intermediates=False,
        text_tokens=None,
    ):
        if return_intermediates:
            trajectories = [x.cpu()]
        else:
            trajectories = None

        neg_cond = torch.zeros_like(pos_cond) if pos_cond is not None else None
        dt = 1.0 / num_steps
        effective_steps = num_steps - skip_last_n

        for i in tqdm(range(effective_steps), desc="Euler CFG Sampling"):
            with torch.no_grad():
                t_val = 1.0 - (i / num_steps)
                t = torch.ones(x.shape[0], 1).to(self.device, x.dtype) * t_val

                x0_pos = self.forward(x, t, pos_cond, text_tokens=text_tokens)
                x0_neg = self.forward(x, t, neg_cond, text_tokens=text_tokens)

                v_pos = (x0_pos - x) / t.view(-1, 1, 1)
                v_neg = (x0_neg - x) / t.view(-1, 1, 1)

                velocity = v_neg + cfg_scale * (v_pos - v_neg)
                x = x + velocity * dt

            if return_intermediates:
                trajectories.append(x.cpu())

        return x, trajectories


# ---------------------------------------------------------------------------
# Per-GPU callables (split fwd / bwd pattern)
# ---------------------------------------------------------------------------


def forward_fn(
    gpu_id: int,
    model: Flow,
    real: torch.Tensor,
    label: torch.Tensor,
    class_dropout_ratio: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward only — returns (predicted_image, noisy_image, t, image) for backward."""
    device = f"cuda:{gpu_id}"
    real = real.to(device)
    label = label.to(device)

    x1, image_shape = image_flatten(real, shuffle_size=PATCH_SIZE)
    x0 = torch.randn_like(x1)

    B = x1.shape[0]
    num_points = 1000
    x_dist, probabilities = create_distribution(num_points, device=device)
    t = sample_from_distribution(x_dist, probabilities, B)[:, None, None].to(x1.dtype)

    cond_clone = label.clone().float()
    is_dropped = torch.rand(B, device=device) < class_dropout_ratio
    cond_clone[is_dropped] = 0

    noisy_image = x0 * t + x1 * (1 - t)

    with torch.autocast("cuda", torch.bfloat16):
        predicted_image = model(noisy_image, t, condition=cond_clone)

    return predicted_image, noisy_image, t, x1, image_shape, label


def backward_fn(
    gpu_id: int,
    model: Flow,
    output: tuple,
    accum_steps: int = 1,
) -> float:
    """Backward only — receives the output from forward_fn."""
    predicted_image, noisy_image, t, x1, image_shape, label = output

    target_velocity = (noisy_image - x1) / (t.view(-1, 1, 1) + 5e-2)
    predicted_velocity = (noisy_image - predicted_image) / (t.view(-1, 1, 1) + 5e-2)

    loss = F.mse_loss(predicted_velocity, target_velocity)
    (loss / accum_steps).backward()
    return loss.item()


# ---------------------------------------------------------------------------
# Training loop — split fwd/bwd
# ---------------------------------------------------------------------------


def train(cfg: dict):
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPU(s)")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = datasets.CelebA(
        root="celeba/", split="all", transform=transform, download=False
    )
    train_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"] * n_gpus,  # full batch; wrapper splits it
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Build the wrapper
    # ------------------------------------------------------------------
    def model_factory():
        model = Flow(**TRAINING_CONFIG["model_config"])
        # model = replace_linear_with_ramtorch(model)
        return model

    wrapper = MultiGPUWrapper(
        model_factory=model_factory,
        optimizer_factory=lambda params: AdamW(params, lr=cfg["lr"], weight_decay=1e-4),
        gradient_accumulation_steps=cfg["accum"],
        max_grad_norm=1.0,
        scheduler_factory=lambda opt: LinearLR(
            opt, start_factor=1e-5, end_factor=1.0, total_iters=cfg["warmup"]
        ),
    )
    wrapper.setup()

    os.makedirs(TRAINING_CONFIG["ckpt_path"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["preview_path"], exist_ok=True)

    if TRAINING_CONFIG["model_checkpoint"]:
        wrapper.load_checkpoint(TRAINING_CONFIG["model_checkpoint"])
    else:
        wrapper.save_checkpoint(
            os.path.join(TRAINING_CONFIG["ckpt_path"], "untrained.safetensors")
        )

    # ------------------------------------------------------------------
    # CSV loss log
    # ------------------------------------------------------------------
    csv_path = os.path.join(TRAINING_CONFIG["ckpt_path"], "loss_log.csv")
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        csv_writer.writerow(["step", "epoch", "loss", "lr", "time"])
    t0 = time.time()

    global_step = 0

    with make_profiler_ctx(cfg, "celeba_inverted_t_trace.json") as prof:
        for epoch in range(1, cfg["num_epochs"] + 1):
            for m in wrapper.models:
                m.train()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['num_epochs']}")

            for batch_idx, (real, label) in enumerate(pbar):
                label = label.float()
                chunks = wrapper.split_batch(real, label)

                # 1. Forward — all GPUs run concurrently
                with record_function("forward"):
                    outputs = wrapper.forward(chunks, forward_fn=forward_fn, class_dropout_ratio=cfg["class_dropout_ratio"])
                # outputs[gpu_id] == (predicted_image, noisy_image, t, x1, image_shape, label)

                # --- optional: inspect outputs before backward ---
                if global_step % 200 == 0:
                    with torch.no_grad():
                        losses_preview = []
                        for out in outputs:
                            pred, noisy, t_val, x1, _, _ = out
                            tv = (noisy - x1) / (t_val.view(-1, 1, 1) + 5e-2)
                            pv = (noisy - pred) / (t_val.view(-1, 1, 1) + 5e-2)
                            losses_preview.append(F.mse_loss(pv, tv).item())
                        avg_preview = sum(losses_preview) / len(losses_preview)
                        pbar.write(f"  [step {global_step}] preview loss: {avg_preview:.4f}")

                # 2. Backward — all GPUs run concurrently
                with record_function("backward"):
                    loss = wrapper.backward(
                        outputs, backward_fn=backward_fn, accum_steps=cfg["accum"]
                    )

                # 3. Sync + step every accum_steps
                if (batch_idx + 1) % cfg["accum"] == 0:
                    with record_function("reduce_grads"):
                        wrapper.reduce_grads()
                    wrapper.clip_grads()
                    with record_function("optimizer_step"):
                        wrapper.optimizer_step()

                lr = wrapper.last_lr
                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}", step=global_step)

                csv_writer.writerow([global_step, epoch, f"{loss:.6f}", f"{lr:.2e}", f"{time.time()-t0:.1f}"])
                csv_file.flush()

                # --- Evaluation / image saving (GPU 0 only) ---
                if global_step % TRAINING_CONFIG["eval_interval"] == 0:
                    model_0 = wrapper.models[0]
                    model_0.eval()
                    with torch.no_grad():
                        # Grab a sample from the last chunk on GPU 0
                        _, _, _, x1_sample, image_shape, label_sample = outputs[0]
                        z = torch.randn_like(x1_sample)

                        with torch.autocast("cuda", torch.bfloat16):
                            fake_images_list = []
                            class_dropout_ratio = TRAINING_CONFIG["class_dropout_ratio"]
                            pos_cond = torch.zeros_like(label_sample) if class_dropout_ratio >= 1.0 else label_sample
                            for cfg_scale, steps in TRAINING_CONFIG["inference_cfg_and_steps"]:
                                fake_cfg, _ = model_0.euler_cfg(
                                    z, pos_cond, cfg_scale, num_steps=steps
                                )
                                fake_images_list.append(fake_cfg)

                            real_unflattened = image_unflatten(x1_sample, image_shape, shuffle_size=PATCH_SIZE)
                            fake_unflattened = [
                                image_unflatten(img, image_shape, shuffle_size=PATCH_SIZE) for img in fake_images_list
                            ]
                            all_images = torch.cat(fake_unflattened + [real_unflattened], dim=0)

                            img_path = (
                                f"{TRAINING_CONFIG['preview_path']}"
                                f"/epoch_{epoch}_step_{global_step}.jpg"
                            )
                            save_image(
                                make_grid(
                                    (all_images.clip(-1, 1) + 1) / 2,
                                    nrow=TRAINING_CONFIG["batch_size"],
                                ),
                                img_path,
                            )
                    model_0.train()

                global_step += 1

                if prof is not None:
                    prof.step()

            # --- Save checkpoint each epoch ---
            wrapper.save_checkpoint(
                os.path.join(TRAINING_CONFIG["ckpt_path"], f"epoch_{epoch}.safetensors")
            )

    csv_file.close()
    wrapper.cleanup()
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    print(f"Loading config from: {config_path}")
    with open(config_path) as f:
        TRAINING_CONFIG = json.load(f)

    PATCH_SIZE = TRAINING_CONFIG.get("patch_size", 16)

    train(TRAINING_CONFIG)
