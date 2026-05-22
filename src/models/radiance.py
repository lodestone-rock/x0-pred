"""radiance.py — Chroma/MM-DiT pixel-space flow-matching model.

Self-contained: owns patchify/unpatchify (via Conv2d/fold), position-ID
construction, and Euler-CFG sampling.  Always predicts x0 then converts to
v-prediction.

Architecture:
  - img_in_patch: Conv2d patchify (patch_size=16, hardcoded)
  - Approximator: generates all AdaLN modulation vectors in one shot (distilled,
    run in torch.no_grad() — intentional; see chroma_tobe_refactored notes)
  - depth × DoubleStreamBlock (MM-DiT, parallel img+txt attention)
  - depth_single_blocks × SingleStreamBlock (merged stream)
  - NerfGLUBlock hypernetwork decoder + NerfFinalLayerConv
  - T5 text encoder (4096-dim) in trainer; context_in_dim configurable

Usage (training):
    model = Radiance(RadianceParams(**cfg["model_config"]))
    v_pred = model(noisy_image, t, txt_embeds, txt_mask)  # [B, 3, H, W]

Usage (inference):
    images, _ = model.euler_cfg(noise, cfg_scale=4.0, num_steps=28,
                                txt=txt_embeds, txt_mask=txt_mask,
                                neg_txt=neg_embeds, neg_txt_mask=neg_mask)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from torch import Tensor
from tqdm import tqdm

from src.models.flow import create_distribution

from functools import lru_cache

# ---------------------------------------------------------------------------
# Patch helpers (pixel-space, (dh dw c) ordering — same as zeta.py)
# ---------------------------------------------------------------------------

PATCH_SIZE = 16  # hardcoded; empirically better than 32


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def modify_mask_to_attend_padding(mask, max_seq_length, num_extra_padding=8):
    """
    Modifies attention mask to allow attention to a few extra padding tokens.

    Args:
        mask: Original attention mask (1 for tokens to attend to, 0 for masked tokens)
        max_seq_length: Maximum sequence length of the model
        num_extra_padding: Number of padding tokens to unmask

    Returns:
        Modified mask
    """
    # Get the actual sequence length from the mask
    seq_length = mask.sum(dim=-1)
    batch_size = mask.shape[0]

    modified_mask = mask.clone()

    for i in range(batch_size):
        current_seq_len = int(seq_length[i].item())

        # Only add extra padding tokens if there's room
        if current_seq_len < max_seq_length:
            # Calculate how many padding tokens we can unmask
            available_padding = max_seq_length - current_seq_len
            tokens_to_unmask = min(num_extra_padding, available_padding)

            # Unmask the specified number of padding tokens right after the sequence
            modified_mask[i, current_seq_len : current_seq_len + tokens_to_unmask] = 1

    return modified_mask

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    # mask should have shape [B, H, L, D]
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, use_compiled: bool = False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.use_compiled = use_compiled

    def _forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale

    def forward(self, x: Tensor):
        return F.rms_norm(x, self.scale.shape, weight=self.scale, eps=1e-6)
        # if self.use_compiled:
        #     return torch.compile(self._forward)(x)
        # else:
        #     return self._forward(x)


def distribute_modulations(tensor: torch.Tensor, depth_single_blocks, depth_double_blocks):
    """
    Distributes slices of the tensor into the block_dict as ModulationOut objects.

    Args:
        tensor (torch.Tensor): Input tensor with shape [batch_size, vectors, dim].
    """
    batch_size, vectors, dim = tensor.shape

    block_dict = {}

    # HARD CODED VALUES! lookup table for the generated vectors
    # TODO: move this into chroma config!
    # Add 38 single mod blocks
    for i in range(depth_single_blocks):
        key = f"single_blocks.{i}.modulation.lin"
        block_dict[key] = None

    # Add 19 image double blocks
    for i in range(depth_double_blocks):
        key = f"double_blocks.{i}.img_mod.lin"
        block_dict[key] = None

    # Add 19 text double blocks
    for i in range(depth_double_blocks):
        key = f"double_blocks.{i}.txt_mod.lin"
        block_dict[key] = None

    # Add the final layer
    block_dict["final_layer.adaLN_modulation.1"] = None
    # 6.2b version
    # block_dict["lite_double_blocks.4.img_mod.lin"] = None
    # block_dict["lite_double_blocks.4.txt_mod.lin"] = None

    idx = 0  # Index to keep track of the vector slices

    for key in block_dict.keys():
        if "single_blocks" in key:
            # Single block: 1 ModulationOut
            block_dict[key] = ModulationOut(
                shift=tensor[:, idx : idx + 1, :],
                scale=tensor[:, idx + 1 : idx + 2, :],
                gate=tensor[:, idx + 2 : idx + 3, :],
            )
            idx += 3  # Advance by 3 vectors

        elif "img_mod" in key:
            # Double block: List of 2 ModulationOut
            double_block = []
            for _ in range(2):  # Create 2 ModulationOut objects
                double_block.append(
                    ModulationOut(
                        shift=tensor[:, idx : idx + 1, :],
                        scale=tensor[:, idx + 1 : idx + 2, :],
                        gate=tensor[:, idx + 2 : idx + 3, :],
                    )
                )
                idx += 3  # Advance by 3 vectors per ModulationOut
            block_dict[key] = double_block

        elif "txt_mod" in key:
            # Double block: List of 2 ModulationOut
            double_block = []
            for _ in range(2):  # Create 2 ModulationOut objects
                double_block.append(
                    ModulationOut(
                        shift=tensor[:, idx : idx + 1, :],
                        scale=tensor[:, idx + 1 : idx + 2, :],
                        gate=tensor[:, idx + 2 : idx + 3, :],
                    )
                )
                idx += 3  # Advance by 3 vectors per ModulationOut
            block_dict[key] = double_block

        elif "final_layer" in key:
            # Final layer: 1 ModulationOut
            block_dict[key] = [
                tensor[:, idx : idx + 1, :],
                tensor[:, idx + 1 : idx + 2, :],
            ]
            idx += 2  # Advance by 3 vectors

    return block_dict



class NerfEmbedder(nn.Module):
    """
    An embedder module that combines input features with a 2D positional
    encoding that mimics the Discrete Cosine Transform (DCT).

    This module takes an input tensor of shape (B, P^2, C), where P is the
    patch size, and enriches it with positional information before projecting
    it to a new hidden size.
    """
    def __init__(self, in_channels, hidden_size_input, max_freqs):
        """
        Initializes the NerfEmbedder.

        Args:
            in_channels (int): The number of channels in the input tensor.
            hidden_size_input (int): The desired dimension of the output embedding.
            max_freqs (int): The number of frequency components to use for both
                             the x and y dimensions of the positional encoding.
                             The total number of positional features will be max_freqs^2.
        """
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        
        # A linear layer to project the concatenated input features and
        # positional encodings to the final output dimension.
        self.embedder = nn.Sequential(
            nn.Linear(in_channels + max_freqs**2, hidden_size_input)
        )

    @lru_cache(maxsize=4)
    def fetch_pos(self, patch_size, device, dtype):
        """
        Generates and caches 2D DCT-like positional embeddings for a given patch size.

        The LRU cache is a performance optimization that avoids recomputing the
        same positional grid on every forward pass.

        Args:
            patch_size (int): The side length of the square input patch.
            device: The torch device to create the tensors on.
            dtype: The torch dtype for the tensors.

        Returns:
            A tensor of shape (1, patch_size^2, max_freqs^2) containing the
            positional embeddings.
        """
        # Create normalized 1D coordinate grids from 0 to 1.
        pos_x = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        
        # Create a 2D meshgrid of coordinates.
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        
        # Reshape positions to be broadcastable with frequencies.
        # Shape becomes (patch_size^2, 1, 1).
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)
        
        # Create a 1D tensor of frequency values from 0 to max_freqs-1.
        freqs = torch.linspace(0, self.max_freqs - 1, self.max_freqs, dtype=dtype, device=device)
        
        # Reshape frequencies to be broadcastable for creating 2D basis functions.
        # freqs_x shape: (1, max_freqs, 1)
        # freqs_y shape: (1, 1, max_freqs)
        freqs_x = freqs[None, :, None]
        freqs_y = freqs[None, None, :]
        
        # A custom weighting coefficient, not part of standard DCT.
        # This seems to down-weight the contribution of higher-frequency interactions.
        coeffs = (1 + freqs_x * freqs_y) ** -1
        
        # Calculate the 1D cosine basis functions for x and y coordinates.
        # This is the core of the DCT formulation.
        dct_x = torch.cos(pos_x * freqs_x * torch.pi)
        dct_y = torch.cos(pos_y * freqs_y * torch.pi)
        
        # Combine the 1D basis functions to create 2D basis functions by element-wise
        # multiplication, and apply the custom coefficients. Broadcasting handles the
        # combination of all (pos_x, freqs_x) with all (pos_y, freqs_y).
        # The result is flattened into a feature vector for each position.
        dct = (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs ** 2)
        
        return dct

    def forward(self, inputs):
        """
        Forward pass for the embedder.

        Args:
            inputs (Tensor): The input tensor of shape (B, P^2, C).

        Returns:
            Tensor: The output tensor of shape (B, P^2, hidden_size_input).
        """
        # Get the batch size, number of pixels, and number of channels.
        B, P2, C = inputs.shape
        # Store the original dtype to cast back to at the end.
        original_dtype = inputs.dtype
        # Force all operations within this module to run in fp32.
        with torch.autocast("cuda", enabled=False):
            # Infer the patch side length from the number of pixels (P^2).
            patch_size = int(P2 ** 0.5)

            inputs = inputs.float()
            # Fetch the pre-computed or cached positional embeddings.
            dct = self.fetch_pos(patch_size, inputs.device, torch.float32)
            
            # Repeat the positional embeddings for each item in the batch.
            dct = dct.repeat(B, 1, 1)
            
            # Concatenate the original input features with the positional embeddings
            # along the feature dimension.
            inputs = torch.cat([inputs, dct], dim=-1)
            
            # Project the combined tensor to the target hidden size.
            inputs = self.embedder.float()(inputs)
        
        return inputs.to(original_dtype)



class NerfGLUBlock(nn.Module):
    """
    A NerfBlock using a Gated Linear Unit (GLU) like MLP.
    """
    def __init__(self, hidden_size_s, hidden_size_x, mlp_ratio, use_compiled):
        super().__init__()
        # The total number of parameters for the MLP is increased to accommodate
        # the gate, value, and output projection matrices.
        # We now need to generate parameters for 3 matrices.
        total_params = 3 * hidden_size_x**2 * mlp_ratio
        self.param_generator = nn.Linear(hidden_size_s, total_params)
        self.norm = RMSNorm(hidden_size_x, use_compiled)
        self.mlp_ratio = mlp_ratio
        # nn.init.zeros_(self.param_generator.weight)
        # nn.init.zeros_(self.param_generator.bias)


    def forward(self, x, s):
        batch_size, num_x, hidden_size_x = x.shape
        mlp_params = self.param_generator(s)

        # Split the generated parameters into three parts for the gate, value, and output projection.
        fc1_gate_params, fc1_value_params, fc2_params = mlp_params.chunk(3, dim=-1)

        # Reshape the parameters into matrices for batch matrix multiplication.
        fc1_gate = fc1_gate_params.view(batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio)
        fc1_value = fc1_value_params.view(batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio)
        fc2 = fc2_params.view(batch_size, hidden_size_x * self.mlp_ratio, hidden_size_x)

        # Normalize the generated weight matrices as in the original implementation.
        fc1_gate = torch.nn.functional.normalize(fc1_gate, dim=-2)
        fc1_value = torch.nn.functional.normalize(fc1_value, dim=-2)
        fc2 = torch.nn.functional.normalize(fc2, dim=-2)

        res_x = x
        x = self.norm(x)

        # Apply the final output projection.
        x = torch.bmm(torch.nn.functional.silu(torch.bmm(x, fc1_gate)) * torch.bmm(x, fc1_value), fc2)
        
        x = x + res_x
        return x


class NerfFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, use_compiled):
        super().__init__()
        self.norm = RMSNorm(hidden_size, use_compiled=use_compiled)
        self.linear = nn.Linear(hidden_size, out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x


class NerfFinalLayerConv(nn.Module):
    def __init__(self, hidden_size, out_channels, use_compiled):
        super().__init__()
        self.norm = RMSNorm(hidden_size, use_compiled=use_compiled)

        # replace nn.Linear with nn.Conv2d since linear is just pointwise conv
        self.conv = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # shape: [N, C, H, W] !
        # RMSNorm normalizes over the last dimension, but our channel dim (C) is at dim=1.
        # So, we permute the dimensions to make the channel dimension the last one.
        x_permuted = x.permute(0, 2, 3, 1)  # Shape becomes [N, H, W, C]

        # Apply normalization on the feature/channel dimension
        x_norm = self.norm(x_permuted)

        # Permute back to the original dimension order for the convolution
        x_norm_permuted = x_norm.permute(0, 3, 1, 2) # Shape becomes [N, C, H, W]

        # Apply the 3x3 convolution
        x = self.conv(x_norm_permuted)
        return x
    

class Approximator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers=4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList(
            [MLPEmbedder(hidden_dim, hidden_dim) for x in range(n_layers)]
        )
        self.norms = nn.ModuleList([RMSNorm(hidden_dim) for x in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)

        for layer, norms in zip(self.layers, self.norms):
            x = x + layer(norms(x))

        x = self.out_proj(x)

        return x


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int, use_compiled: bool = False):
        super().__init__()
        self.query_norm = RMSNorm(dim, use_compiled=use_compiled)
        self.key_norm = RMSNorm(dim, use_compiled=use_compiled)
        self.use_compiled = use_compiled

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        use_compiled: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim, use_compiled=use_compiled)
        self.proj = nn.Linear(dim, dim)
        self.use_compiled = use_compiled

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


def _modulation_shift_scale_fn(x, scale, shift):
    return (1 + scale) * x + shift


def _modulation_gate_fn(x, gate, gate_params):
    return x + gate * gate_params


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
        use_compiled: bool = False,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_compiled=use_compiled,
        )

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_compiled=use_compiled,
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.use_compiled = use_compiled

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def modulation_shift_scale_fn(self, x, scale, shift):
        if self.use_compiled:
            return torch.compile(_modulation_shift_scale_fn)(x, scale, shift)
        else:
            return _modulation_shift_scale_fn(x, scale, shift)

    def modulation_gate_fn(self, x, gate, gate_params):
        if self.use_compiled:
            return torch.compile(_modulation_gate_fn)(x, gate, gate_params)
        else:
            return _modulation_gate_fn(x, gate, gate_params)

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        distill_vec: list[ModulationOut],
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = distill_vec

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        # replaced with compiled fn
        # img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_modulated = self.modulation_shift_scale_fn(
            img_modulated, img_mod1.scale, img_mod1.shift
        )
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        # replaced with compiled fn
        # txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_modulated = self.modulation_shift_scale_fn(
            txt_modulated, txt_mod1.scale, txt_mod1.shift
        )
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe, mask=mask)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        # replaced with compiled fn
        # img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        # img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        img = self.modulation_gate_fn(img, img_mod1.gate, self.img_attn.proj(img_attn))
        img = self.modulation_gate_fn(
            img,
            img_mod2.gate,
            self.img_mlp(
                self.modulation_shift_scale_fn(
                    self.img_norm2(img), img_mod2.scale, img_mod2.shift
                )
            ),
        )

        # calculate the txt bloks
        # replaced with compiled fn
        # txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        # txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        txt = self.modulation_gate_fn(txt, txt_mod1.gate, self.txt_attn.proj(txt_attn))
        txt = self.modulation_gate_fn(
            txt,
            txt_mod2.gate,
            self.txt_mlp(
                self.modulation_shift_scale_fn(
                    self.txt_norm2(txt), txt_mod2.scale, txt_mod2.shift
                )
            ),
        )

        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        use_compiled: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim, use_compiled=use_compiled)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.use_compiled = use_compiled

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def modulation_shift_scale_fn(self, x, scale, shift):
        if self.use_compiled:
            return torch.compile(_modulation_shift_scale_fn)(x, scale, shift)
        else:
            return _modulation_shift_scale_fn(x, scale, shift)

    def modulation_gate_fn(self, x, gate, gate_params):
        if self.use_compiled:
            return torch.compile(_modulation_gate_fn)(x, gate, gate_params)
        else:
            return _modulation_gate_fn(x, gate, gate_params)

    def forward(
        self, x: Tensor, pe: Tensor, distill_vec: list[ModulationOut], mask: Tensor
    ) -> Tensor:
        mod = distill_vec
        # replaced with compiled fn
        # x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        x_mod = self.modulation_shift_scale_fn(self.pre_norm(x), mod.scale, mod.shift)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe, mask=mask)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        # replaced with compiled fn
        # return x + mod.gate * output
        return self.modulation_gate_fn(x, mod.gate, output)


class LastLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        use_compiled: bool = False,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.use_compiled = use_compiled

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def modulation_shift_scale_fn(self, x, scale, shift):
        if self.use_compiled:
            return torch.compile(_modulation_shift_scale_fn)(x, scale, shift)
        else:
            return _modulation_shift_scale_fn(x, scale, shift)

    def forward(self, x: Tensor, distill_vec: list[Tensor]) -> Tensor:
        shift, scale = distill_vec
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        # replaced with compiled fn
        # x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.modulation_shift_scale_fn(
            self.norm_final(x), scale[:, None, :], shift[:, None, :]
        )
        x = self.linear(x)
        return x


def vae_flatten(x: Tensor, patch_size: int = PATCH_SIZE):
    """[B, C, H, W] → ([B, N, P²·C], original_shape)  where N = H/P * W/P"""
    from einops import rearrange
    return (
        rearrange(x, "n c (h dh) (w dw) -> n (h w) (dh dw c)", dh=patch_size, dw=patch_size),
        x.shape,
    )


def vae_unflatten(x: Tensor, shape: tuple, patch_size: int = PATCH_SIZE) -> Tensor:
    """[B, N, P²·C] → [B, C, H, W]"""
    from einops import rearrange
    n, c, h, w = shape
    return rearrange(
        x,
        "n (h w) (dh dw c) -> n c (h dh) (w dw)",
        dh=patch_size,
        dw=patch_size,
        c=c,
        h=h // patch_size,
        w=w // patch_size,
    )


# ---------------------------------------------------------------------------
# Position ID helpers
# ---------------------------------------------------------------------------

def prepare_latent_image_ids(batch_size: int, height: int, width: int,
                              patch_size: int = PATCH_SIZE) -> Tensor:
    """Generate [B, N, 3] RoPE position IDs for image patches.

    Dim 0 = 0 (unused / zero for all img patches, Flux/Chroma convention).
    Dim 1 = patch row index.
    Dim 2 = patch col index.
    """
    h_p = height // patch_size
    w_p = width  // patch_size
    ids = torch.zeros(h_p, w_p, 3)
    ids[..., 1] = torch.arange(h_p)[:, None]
    ids[..., 2] = torch.arange(w_p)[None, :]
    ids = ids.reshape(1, h_p * w_p, 3).expand(batch_size, -1, -1)
    return ids.contiguous()


def make_text_position_ids(batch_size: int, seq_len: int) -> Tensor:
    """Generate [B, L, 3] RoPE position IDs for text tokens.

    Dim 0 = token index 0..L-1.
    Dims 1-2 = 0 (Flux/Chroma convention — text has no spatial position).
    """
    ids = torch.zeros(seq_len, 3)
    ids[:, 0] = torch.arange(seq_len)
    return ids.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


# ---------------------------------------------------------------------------
# Timestep schedule helpers (copied from zeta.py — used in euler_cfg)
# ---------------------------------------------------------------------------

def _time_shift(mu: float, sigma: float, t: Tensor) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    """Build a shifted timestep schedule from t=1→0."""
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        base_seq   = 256
        max_seq    = 4096
        mu         = base_shift + (max_shift - base_shift) * (image_seq_len - base_seq) / (max_seq - base_seq)
        mu         = float(torch.clamp(torch.tensor(mu), min=base_shift, max=max_shift))
        timesteps  = _time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


# ---------------------------------------------------------------------------
# Model params
# ---------------------------------------------------------------------------

@dataclass
class RadianceParams:
    # Image / text
    in_channels: int = 3              # RGB input (patchify done by Conv2d)
    context_in_dim: int = 4096        # T5-XXL hidden dim (or 2560 for Qwen3)

    # Backbone
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19                   # DoubleStreamBlock count
    depth_single_blocks: int = 38     # SingleStreamBlock count
    axes_dim: list[int] = field(default_factory=lambda: [16, 56, 56])
    theta: int = 10_000
    qkv_bias: bool = True

    # Approximator (modulation distillation)
    approximator_in_dim: int = 64
    approximator_depth: int = 5
    approximator_hidden_size: int = 5120

    # NeRF decoder head
    nerf_hidden_size: int = 64
    nerf_mlp_ratio: int = 4
    nerf_depth: int = 4
    nerf_max_freqs: int = 8

    # Training options
    grad_checkpointing: bool = False
    _use_compiled: bool = False       # per-block torch.compile (call compile_blocks() after setup)

    # patch_size is hardcoded to 16; kept here for reference / config serialisation
    patch_size: int = PATCH_SIZE


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class Radiance(nn.Module):
    """Radiance: Chroma/MM-DiT pixel-space flow-matching model.

    forward() accepts [B, 3, H, W] images and returns v-predictions in the
    same shape.  Patchify (Conv2d), position-ID construction, and x0→v
    conversion are all handled internally.
    """

    def __init__(self, params: RadianceParams):
        super().__init__()
        self.params = params
        self.patch_size = PATCH_SIZE   # hardcoded

        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"hidden_size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"sum(axes_dim)={sum(params.axes_dim)} must equal pe_dim={pe_dim}"
            )

        # RoPE positional embedding
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)

        # Image patchify projection: Conv2d zero-init (matches Chroma)
        self.img_in_patch = nn.Conv2d(
            params.in_channels,
            params.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        nn.init.zeros_(self.img_in_patch.weight)
        nn.init.zeros_(self.img_in_patch.bias)

        # Approximator: distills all block modulation vectors from (t, guidance)
        self.distilled_guidance_layer = Approximator(
            params.approximator_in_dim,
            params.hidden_size,
            params.approximator_hidden_size,
            params.approximator_depth,
        )

        # Text projection
        self.txt_in = nn.Linear(params.context_in_dim, params.hidden_size)

        # MM-DiT double-stream blocks (img + txt in parallel)
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(
                params.hidden_size,
                params.num_heads,
                mlp_ratio=params.mlp_ratio,
                qkv_bias=params.qkv_bias,
                use_compiled=params._use_compiled,
            )
            for _ in range(params.depth)
        ])

        # Single-stream blocks (fused img+txt)
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(
                params.hidden_size,
                params.num_heads,
                mlp_ratio=params.mlp_ratio,
                use_compiled=params._use_compiled,
            )
            for _ in range(params.depth_single_blocks)
        ])

        # NeRF decoder head
        self.nerf_image_embedder = NerfEmbedder(
            in_channels=params.in_channels,
            hidden_size_input=params.nerf_hidden_size,
            max_freqs=params.nerf_max_freqs,
        )
        self.nerf_blocks = nn.ModuleList([
            NerfGLUBlock(
                hidden_size_s=params.hidden_size,
                hidden_size_x=params.nerf_hidden_size,
                mlp_ratio=params.nerf_mlp_ratio,
                use_compiled=params._use_compiled,
            )
            for _ in range(params.nerf_depth)
        ])
        self.nerf_final_layer_conv = NerfFinalLayerConv(
            params.nerf_hidden_size,
            out_channels=params.in_channels,
            use_compiled=params._use_compiled,
        )

        # Modulation vector count (matches distribute_modulations layout):
        #   3 per single block + 2×3 per double block (img+txt) + 2 final-layer
        self.mod_index_length = (
            3 * params.depth_single_blocks
            + 2 * 6 * params.depth
            + 2
        )
        self.depth_single_blocks = params.depth_single_blocks
        self.depth_double_blocks  = params.depth
        self.approximator_in_dim  = params.approximator_in_dim

        self.register_buffer(
            "mod_index",
            torch.tensor(list(range(self.mod_index_length))),
            persistent=False,
        )

        self.grad_checkpointing = params.grad_checkpointing

    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def compile_blocks(self) -> None:
        """torch.compile each block.  Call *after* MultiGPUWrapper.setup()."""
        for block in self.double_blocks:
            block.forward = torch.compile(block.forward)
        for block in self.single_blocks:
            block.forward = torch.compile(block.forward)
        for block in self.nerf_blocks:
            block.forward = torch.compile(block.forward)

    # ------------------------------------------------------------------
    # Internal forward (operates on raw [B, C, H, W])
    # ------------------------------------------------------------------

    def _forward(
        self,
        img: Tensor,       # [B, C, H, W]  noisy image
        img_ids: Tensor,   # [B, N, 3]
        txt: Tensor,       # [B, L, context_in_dim]
        txt_ids: Tensor,   # [B, L, 3]
        txt_mask: Tensor,  # [B, L]  bool
        timesteps: Tensor, # [B]  in [0, 1]
    ) -> Tensor:
        """Returns predicted x0  [B, C, H, W]."""
        if img.ndim != 4:
            raise ValueError("img must be [B, C, H, W]")
        B, C, H, W = img.shape

        # --- Extract raw patch pixels for the NeRF decoder head ---
        # unfold → [B, C*P*P, NumPatches] → [B, NumPatches, C*P*P]
        nerf_pixels = F.unfold(img, kernel_size=self.patch_size, stride=self.patch_size)
        nerf_pixels = nerf_pixels.transpose(1, 2)   # [B, N, C*P*P]
        num_patches = nerf_pixels.shape[1]

        # --- Patchify image for transformer backbone ---
        img_hidden = self.img_in_patch(img)          # [B, hidden, H/P, W/P]
        img_hidden = img_hidden.flatten(2).transpose(1, 2)  # [B, N, hidden]

        # --- Text projection ---
        txt_hidden = self.txt_in(txt)                # [B, L, hidden]

        # --- Distill all modulation vectors (Approximator in no_grad) ---
        # Hardcode guidance=0 (pixel-space flow-matching has no guidance distillation)
        with torch.no_grad():
            self.mod_index = self.mod_index.to(img.device)
            distill_timestep = timestep_embedding(timesteps, self.approximator_in_dim // 4)
            distil_guidance  = timestep_embedding(
                torch.zeros(B, device=img.device, dtype=timesteps.dtype),
                self.approximator_in_dim // 4,
            )
            modulation_index = timestep_embedding(self.mod_index, self.approximator_in_dim // 2)
            modulation_index = modulation_index.unsqueeze(0).expand(B, -1, -1)
            timestep_guidance = (
                torch.cat([distill_timestep, distil_guidance], dim=1)
                .unsqueeze(1)
                .expand(-1, self.mod_index_length, -1)
            )
            input_vec  = torch.cat([timestep_guidance, modulation_index], dim=-1)
            mod_vectors = self.distilled_guidance_layer(input_vec.requires_grad_(True))

        mod_vectors_dict = distribute_modulations(
            mod_vectors, self.depth_single_blocks, self.depth_double_blocks
        )

        # --- RoPE position embeddings ---
        ids = torch.cat((txt_ids, img_ids), dim=1)  # [B, L+N, 3]
        pe  = self.pe_embedder(ids)                  # [B, 1, L+N, head_dim]

        # --- Attention mask ---
        max_len = txt_hidden.shape[1]
        with torch.no_grad():
            txt_mask_padded = modify_mask_to_attend_padding(txt_mask, max_len, num_extra_padding=1)
            txt_img_mask = torch.cat(
                [
                    txt_mask_padded,
                    torch.ones(B, num_patches, device=txt_mask.device),
                ],
                dim=1,
            )
            txt_img_mask = txt_img_mask.float().T @ txt_img_mask.float()
            txt_img_mask = (
                txt_img_mask[None, None, ...]
                .expand(B, self.params.num_heads, -1, -1)
                .int()
                .bool()
            )

        _ckpt = lambda fn, *args: ckpt.checkpoint(fn, *args, use_reentrant=False)

        # --- Double-stream blocks ---
        for i, block in enumerate(self.double_blocks):
            img_mod  = mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"]
            txt_mod  = mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"]
            distill_vec = [img_mod, txt_mod]
            if self.grad_checkpointing:
                img_hidden, txt_hidden = _ckpt(
                    block, img_hidden, txt_hidden, pe, distill_vec, txt_img_mask
                )
            else:
                img_hidden, txt_hidden = block(
                    img=img_hidden, txt=txt_hidden,
                    pe=pe, distill_vec=distill_vec, mask=txt_img_mask,
                )

        # --- Merge streams for single-stream blocks ---
        merged = torch.cat((txt_hidden, img_hidden), dim=1)  # [B, L+N, hidden]
        for i, block in enumerate(self.single_blocks):
            single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
            if self.grad_checkpointing:
                merged = _ckpt(block, merged, pe, single_mod, txt_img_mask)
            else:
                merged = block(merged, pe=pe, distill_vec=single_mod, mask=txt_img_mask)

        # Strip text prefix
        img_hidden = merged[:, txt_hidden.shape[1]:, :]  # [B, N, hidden]

        # --- NeRF decoder ---
        # reshape conditioning to [B*N, hidden]
        nerf_cond   = img_hidden.reshape(B * num_patches, self.params.hidden_size)
        # reshape pixels to [B*N, C, P*P] → [B*N, P*P, C]
        nerf_pixels = nerf_pixels.reshape(B * num_patches, C, self.patch_size ** 2)
        nerf_pixels = nerf_pixels.transpose(1, 2)          # [B*N, P*P, C]

        img_dct = self.nerf_image_embedder(nerf_pixels)     # [B*N, P*P, nerf_hidden]
        # reshape to [B*N, 1, hidden] expected by NerfGLUBlock's param_generator
        for block in self.nerf_blocks:
            if self.grad_checkpointing:
                img_dct = _ckpt(block, img_dct, nerf_cond)
            else:
                img_dct = block(img_dct, nerf_cond)

        # --- Reconstruct image via fold ---
        # norm is applied on [B*N, P*P, nerf_hidden]
        img_dct = self.nerf_final_layer_conv.norm(img_dct)
        # → [B*N, C, P*P]
        img_dct = img_dct.transpose(1, 2)
        # → [B, N, C*P*P]
        img_dct = img_dct.reshape(B, num_patches, -1)
        # → [B, C*P*P, N]
        img_dct = img_dct.transpose(1, 2)
        # fold → [B, nerf_hidden, H, W]
        img_dct = F.fold(
            img_dct,
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        # 3×3 conv projection → [B, C, H, W]
        output = self.nerf_final_layer_conv.conv(img_dct)

        return output  # predicted x0

    # ------------------------------------------------------------------
    # x0 → v-prediction conversion
    # ------------------------------------------------------------------

    def _apply_x0_residual(self, predicted: Tensor, noisy: Tensor, t: Tensor) -> Tensor:
        """Convert x0 prediction to v-prediction.

        v = (noisy - x0) / (t + eps)
        eps avoids divide-by-zero at t=0 during training.
        """
        eps = 5e-2 if self.training else 0.0
        return (noisy - predicted) / (t.view(-1, 1, 1, 1) + eps)

    # ------------------------------------------------------------------
    # Public forward: [B, 3, H, W] → [B, 3, H, W] v-prediction
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,         # [B, 3, H, W]  noisy image
        t: Tensor,         # [B] or [B, 1, 1, 1]  timestep in [0, 1]
        txt: Tensor,       # [B, L, context_in_dim]
        txt_mask: Tensor,  # [B, L]  bool
    ) -> Tensor:
        """Returns v-prediction [B, 3, H, W]."""
        t = t.view(-1)
        B, C, H, W = x.shape

        img_ids = prepare_latent_image_ids(B, H, W, self.patch_size).to(x.device)
        txt_ids = make_text_position_ids(B, txt.shape[1]).to(x.device)

        predicted_x0 = self._forward(x, img_ids, txt, txt_ids, txt_mask, t)
        return self._apply_x0_residual(predicted_x0, x, t)

    # ------------------------------------------------------------------
    # Euler CFG sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def euler_cfg(
        self,
        x: Tensor,
        cfg_scale: float,
        num_steps: int,
        txt: Tensor,
        txt_mask: Tensor,
        neg_txt: Tensor,
        neg_txt_mask: Tensor,
        schedule_mu: float | None = None,
        grid_points: int = 1024,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, list | None]:
        """Euler CFG sampler stepping from t=1 (noise) to t=0 (clean).

        Args:
            x:               Initial noise [B, 3, H, W].
            cfg_scale:       Classifier-free guidance scale.
            num_steps:       Number of Euler steps.
            txt / txt_mask:  Positive text conditioning.
            neg_txt / neg_txt_mask: Negative text conditioning.
            schedule_mu:     Timestep shift strength.
                               None  → sequence-length auto-mu (get_schedule).
                               0.0   → uniform linear (no shift).
                               float → shifted via create_distribution CDF inversion.
            grid_points:     CDF grid resolution for schedule_mu path.
            return_intermediates: If True, return CPU tensors at each step.

        Returns:
            (denoised_image, trajectories_or_None)
        """
        B, C, H, W = x.shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)

        # Build timestep schedule
        if schedule_mu is None:
            t_seq = torch.tensor(
                get_schedule(num_steps, num_patches, shift=True),
                device=x.device, dtype=x.dtype,
            )
        elif schedule_mu == 0.0:
            t_seq = torch.linspace(1.0, 0.0, num_steps + 1, device=x.device, dtype=x.dtype)
        else:
            grid_t, grid_p = create_distribution(grid_points, device=x.device, mu=schedule_mu)
            grid_t = grid_t.to(x.dtype)
            grid_p = grid_p.to(x.dtype)

            cdf = torch.cumsum(grid_p, dim=0)
            cdf = cdf / cdf[-1].clamp(min=1e-8)

            q   = torch.linspace(1.0, 0.0, num_steps + 1, device=x.device, dtype=x.dtype)
            idx = torch.searchsorted(cdf, q.clamp(0.0, 1.0)).clamp(1, grid_points - 1)
            cdf_lo, cdf_hi = cdf[idx - 1], cdf[idx]
            t_lo,   t_hi   = grid_t[idx - 1], grid_t[idx]
            frac  = (q - cdf_lo) / (cdf_hi - cdf_lo).clamp(min=1e-8)
            t_seq = t_lo + frac * (t_hi - t_lo)
            t_seq[0]  = 1.0
            t_seq[-1] = 0.0

        trajectories = [x.cpu()] if return_intermediates else None

        for i in tqdm(range(num_steps), desc="Euler CFG"):
            t_curr = t_seq[i]
            t_prev = t_seq[i + 1]
            t_vec  = torch.full((B,), t_curr.item(), dtype=x.dtype, device=x.device)

            v_pos = self.forward(x, t_vec, txt,     txt_mask)
            v_neg = self.forward(x, t_vec, neg_txt, neg_txt_mask)

            velocity = v_neg + cfg_scale * (v_pos - v_neg)
            x = x + (t_prev - t_curr) * velocity

            if return_intermediates:
                trajectories.append(x.cpu())

        return x, trajectories
