"""wan_lora.py — LoRA injection for Wan-2.1-VACE-14B DiT.

Ports the pattern from k2/lora.py to the Wan VACE transformer.  Replaces every
nn.Linear inside the DiT blocks (self-attn, cross-attn, FFN) and VACE blocks
with a LoRALinear drop-in.  The frozen base_weight / base_bias are registered
as buffers; only lora_A / lora_B are trainable Parameters.

Rank-32 per the paper (§3.3.3).  All other parameters (norms, scale_shift_table,
patch embeddings, condition embedder, proj_out) are left frozen.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LoRALinear", "inject_lora", "lora_state_dict", "trainable_param_count"]


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank LoRA delta.

    y = x @ (W + scale * B @ A)^T + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_weight: torch.Tensor,
        base_bias: torch.Tensor | None,
        rank: int,
        alpha: float,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = alpha / rank

        # Frozen base weights — buffers so they ride along with .to() but
        # contribute no gradients.
        self.register_buffer("base_weight", base_weight.detach().clone())
        if base_bias is not None:
            self.register_buffer("base_bias", base_bias.detach().clone())
        else:
            self.base_bias = None

        # Trainable LoRA adapters.
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (self.lora_B @ self.lora_A) * self.scale
        effective_weight = self.base_weight + delta.to(self.base_weight.dtype)
        return F.linear(x, effective_weight, self.base_bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, scale={self.scale:.3f}"
        )


def _is_lora_target(module: nn.Module) -> bool:
    """Return True for nn.Linear modules we want to replace with LoRA."""
    return isinstance(module, nn.Linear)


def inject_lora(
    model: nn.Module,
    rank: int = 32,
    alpha: float | None = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> dict[str, LoRALinear]:
    """Replace all nn.Linear in the model with LoRALinear.

    Args:
        model: The WanVACETransformer3DModel (or any nn.Module).
        rank: LoRA rank (paper uses 32).
        alpha: LoRA alpha; defaults to float(rank).
        exclude_prefixes: Parameter name prefixes to skip (e.g. patch_embedding).

    Returns:
        Dict mapping original parameter names to the new LoRALinear modules.
    """
    if alpha is None:
        alpha = float(rank)

    replaced: dict[str, LoRALinear] = {}

    # Collect (parent, name, module) triples first to avoid mutating during iteration.
    targets: list[tuple[nn.Module, str, nn.Linear]] = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(full_name.startswith(ex) or full_name == ex for ex in exclude_prefixes):
                continue
            if _is_lora_target(child):
                targets.append((parent, child_name, child, full_name))

    for parent, child_name, child, full_name in targets:
        lora_layer = LoRALinear(
            in_features=child.in_features,
            out_features=child.out_features,
            base_weight=child.weight.data,
            base_bias=child.bias.data if child.bias is not None else None,
            rank=rank,
            alpha=alpha,
        )
        # Move to same device/dtype as the original.
        lora_layer = lora_layer.to(
            device=child.weight.device,
            dtype=child.weight.dtype,
        )
        setattr(parent, child_name, lora_layer)
        replaced[full_name] = lora_layer

    # Freeze everything except LoRA params.
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)

    return replaced


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only LoRA adapter tensors (lora_A, lora_B) from the model."""
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            sd[f"{name}.lora_A"] = module.lora_A.data.cpu().clone()
            sd[f"{name}.lora_B"] = module.lora_B.data.cpu().clone()
    return sd


def load_lora_state_dict(model: nn.Module, sd: dict[str, torch.Tensor]):
    """Load LoRA adapter weights into the model (partial, non-strict)."""
    own = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            own[f"{name}.lora_A"] = module.lora_A
            own[f"{name}.lora_B"] = module.lora_B
    loaded = 0
    for k, v in sd.items():
        if k in own:
            own[k].data.copy_(v.to(own[k].device, dtype=own[k].dtype))
            loaded += 1
    print(f"[lora] Loaded {loaded}/{len(sd)} LoRA tensors.")
    return loaded


def trainable_param_count(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
