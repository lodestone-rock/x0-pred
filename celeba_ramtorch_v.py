import os
import wandb
from tqdm import tqdm
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np

from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms

# from torch.optim import AdamW # Assuming torchastic is your custom optimizer library
from ramtorch import AdamW, Linear
from ramtorch.helpers import move_model_to_device, replace_linear_with_ramtorch
from ramtorch.zero1 import create_zero_param_groups, broadcast_zero_params

from einops import rearrange
import math


# (The rest of your kernel and model code remains the same)
# kernel (if you want to optimize the code, optimize all of these kernel!)
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset=0):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def soft_clamp(x, scale, alpha, shift):
    return scale * F.tanh(x * alpha) + shift


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (max_seq_len ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(self, x, seq_dim=1):
        return (
            self.cos_cached[:, :, : x.shape[seq_dim], :],
            self.sin_cached[:, :, : x.shape[seq_dim], :],
        )


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
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        # enabling bias here so the model has a freedom to shift the activation
        self.wo = nn.Linear(dim, dim, bias=True)
        self.layer_norm = SoftClamp(dim)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

        self.q_norm = SoftClamp(dim)
        self.k_norm = SoftClamp(dim)

        self.add_module("layer_norm", self.layer_norm)

        nn.init.zeros_(self.wo.weight)
        self.use_compiled = False
        self.use_rope = use_rope

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.layer_norm(x)

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(
            1, 2
        )

        cos, sin = self.rope(x, seq_dim=1)
        if self.use_rope:
            if self.use_compiled:
                q, k = torch.compile(apply_rotary_pos_emb)(q, k, cos, sin)
            else:
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask
        )
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        out = self.wo(out)

        return out + residual


class GLU(nn.Module):
    def __init__(self, dim, exp_fac=4):
        super(GLU, self).__init__()
        self.wi_0 = nn.Linear(dim, dim * exp_fac, bias=False)
        self.wi_1 = nn.Linear(dim, dim * exp_fac, bias=False)
        # enabling bias here so the model has a freedom to shift the activation
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
    ):
        super(TransformerNetwork, self).__init__()
        if input_proj:
            self.input_layer = nn.Linear(input_dim, dim)
        else:
            self.input_layer = nn.Identity()
            input_dim = dim
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": AttentionBlock(
                            dim, num_heads, rope_seq_length, use_rope
                        ),
                        "glu": GLU(dim, exp_fac),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = SoftClamp(dim)
        if final_head:
            self.output_layer = nn.Linear(dim, output_dim)
        else:
            self.output_layer = nn.Identity()

    def set_use_compiled(self):
        for name, module in self.named_modules():
            # Check if the module has the 'use_compiled' attribute
            if hasattr(module, "use_compiled"):
                print(f"Setting 'use_compiled' to True in module: {name}")
                setattr(module, "use_compiled", True)

    def forward(self, x, attention_mask=None, act_ckpt=False):
        # just use checkpoint, your GPU is fast enough to recompute the entire thing
        if act_ckpt:
            x = checkpoint(self.input_layer, x)
            for block in self.blocks:
                if type(block) == nn.Identity:
                    continue
                # res = x
                x = checkpoint(
                    lambda x, mask: block["attn"](x, mask), x, attention_mask
                )
                x = checkpoint(block["glu"], x)
                # x = res + x
            x = checkpoint(self.out_norm, x)
            x = checkpoint(self.output_layer, x)

        else:
            x = self.input_layer(x)
            for block in self.blocks:
                if type(block) == nn.Identity:
                    continue
                # res = x
                x = block["attn"](x, attention_mask)
                x = block["glu"](x)
                # x = res + x
            x = self.out_norm(x)
            x = self.output_layer(x)
        return x


def image_flatten(latents, shuffle_size=16):
    # nchw to nhwc then pixel shuffle of arbitrary size then flatten
    # n c h w -> n h w c
    # n (h dh) (w dw) c -> n h w (c dh dw)
    # n h w c -> n (h w) c
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
    # reverse of the flatten operator above
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
    # Probability range on x axis
    x = torch.linspace(0, 1, num_points, device=device)

    # Custom probability density function
    probabilities = -7.7 * ((x - 0.5) ** 2) + 2

    # Normalize to sum to 1
    probabilities /= probabilities.sum()

    return x, probabilities


def repeat_along_dim(tensor, repeats, dim):
    # Move the desired dimension to the front
    permute_order = list(range(tensor.dim()))
    permute_order[dim], permute_order[0] = permute_order[0], permute_order[dim]
    tensor = tensor.permute(permute_order)

    # Unsqueeze to add a new dimension for repetition
    tensor = tensor.unsqueeze(1)

    # Repeat along the new dimension
    repeated_tensor = tensor.repeat(1, repeats, *([1] * (tensor.dim() - 2)))

    # Collapse the repeated dimension
    repeated_tensor = repeated_tensor.view(-1, *repeated_tensor.shape[2:])

    # Move the dimension back to its original order
    permute_order[dim], permute_order[0] = permute_order[0], permute_order[dim]
    repeated_tensor = repeated_tensor.permute(permute_order)

    return repeated_tensor


class Flow(nn.Module):
    """
    A flow-matching model based on a Transformer architecture.

    This model learns the vector field that transports a noise distribution (z)
    to a data distribution (x1) over a normalized time range [0, 1].
    It uses classifier-free guidance for conditional generation and can be informed
    whether it is being used for training or integration via the `integrate` flag.
    """

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
    ):
        super().__init__()
        self.dim = dim
        self.class_count = class_count

        # --- Input & Embedding Layers ---
        self.input_layer = nn.Linear(input_dim, dim)
        self.timestep_vector = nn.Linear(1, dim)
        self.class_embed = nn.Linear(cond_seq_len, dim)
        self.class_norm = SoftClamp(dim=dim)
        self.cond_seq_len = cond_seq_len
        # --- Core Network ---
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
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stability."""
        nn.init.zeros_(self.timestep_vector.weight)
        nn.init.zeros_(self.timestep_vector.bias)
        nn.init.zeros_(self.class_embed.weight)
        nn.init.zeros_(self.class_embed.bias)

    @property
    def device(self):
        return torch.cuda.current_device()  # next(self.parameters()).device

    def forward(self, x, t, condition, attention_mask=None):
        """
        Forward pass for the flow model.

        Args:
            x (torch.Tensor): Input tensor of shape [B, SeqLen, Dim].
            t (torch.Tensor): Time step tensor of shape [B, 1].
            condition (torch.Tensor): Class condition tensor of shape [B].
            integrate (bool): Flag indicating if the model is used for integration (sampling).
            attention_mask (torch.Tensor, optional): Mask for attention.

        Returns:
            torch.Tensor: The predicted velocity vector.
        """
        # 1. Project input patches
        x_proj = self.input_layer(x)

        # 2. Create and project time, class, and integration embeddings
        time_vec = self.timestep_vector(t.view(-1, 1)).unsqueeze(1)
        class_vec = self.class_embed(condition.to(self.class_embed.weight.dtype))[
            :, None, :
        ]
        class_vec = self.class_norm(class_vec)


        # 3. Concatenate tokens: [time, class, integration, sequence]
        tokens = torch.cat((time_vec, class_vec, x_proj), dim=1)

        # 4. Forward through the transformer
        output_tokens = self.transformer(tokens, attention_mask)

        # 5. Return only the output corresponding to the original sequence
        # We slice off the first three tokens (time, class, integration)
        velocity_pred = output_tokens[:, 2:, ...]
        # predicted_x0 = output_tokens[:, 2:, ...]
        return velocity_pred

    def compute_loss(self, image, noise, condition, class_dropout_ratio=0.1, p_integrator=0):
        """
        Calculates the rectified flow loss for a batch.
        During loss computation, `integrate` is always False.
        """
        B = image.shape[0]

        cond_clone = condition.clone()
        is_dropped = torch.rand(B, device=self.device) < class_dropout_ratio
        cond_clone[is_dropped] = 0

        # t = torch.rand((B, 1, 1), device=self.device)
        # target_velocity = image - noise       
        # target_velocity = image - noise
        # x0 noise x1 image
        num_points = 1000  # Number of points in the range
        x, probabilities = create_distribution(num_points, device=self.device)
        t = sample_from_distribution(x, probabilities, B)[:, None, None]
        t = t.to(image.dtype)
        noisy_image = noise * (1 - t) + image * t

        # predicted_image = self.forward(noisy_image, t, cond_clone)

        target_velocity = image - noise # recalibrated
        predicted_velocity = self.forward(noisy_image, t, cond_clone)
        # target_velocity = (image - noisy_image) / (1-t.view(-1, 1, 1) + 5e-2)
        # predicted_velocity = (predicted_image - noisy_image) / (1-t.view(-1, 1, 1) + 5e-2)


        loss = F.mse_loss(predicted_velocity, target_velocity)

        return loss

    def euler_cfg(
        self,
        x,
        pos_cond,
        cfg_scale=4.0,
        num_steps=100,
        skip_last_n=0,
        return_intermediates=False,
    ):
        """
        Euler method sampler with CFG.
        During sampling, `integrate` is always True.
        """
        if return_intermediates:
            trajectories = [x.cpu()]
        else:
            trajectories = None

        # neg_cond = torch.tensor([self.class_count] * pos_cond.shape[0]).to(pos_cond.device)
        neg_cond = torch.zeros_like(pos_cond)
        dt = 1.0 / num_steps
        effective_steps = num_steps - skip_last_n

        for i in tqdm(range(effective_steps), desc="Euler CFG Sampling"):
            with torch.no_grad():
                t_val = i * dt
                t = torch.ones(x.shape[0], 1).to(self.device, x.dtype) * t_val

                v_pos = self.forward(x, t, pos_cond)
                v_neg = self.forward(x, t, neg_cond)
                # x0_pos = self.forward(x, t, pos_cond)
                # x0_neg = self.forward(x, t, neg_cond)

                # v_pos = (x0_pos - x) / (1-t.view(-1, 1, 1))
                # v_neg = (x0_neg - x) / (1-t.view(-1, 1, 1))

                velocity = v_neg + cfg_scale * (v_pos - v_neg)
                x = x + velocity * dt

            if return_intermediates:
                trajectories.append(x.cpu())

        return x, trajectories


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, training_config, model):

    latch = True
    setup(rank, world_size)

    # --- Configuration ---
    torch.manual_seed(0)

    # --- Data Loading ---
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalize to [-1, 1]
        ]
    )

    dataset = datasets.CelebA(
        root="celeba/", split="train", transform=transform, download=True
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        sampler=sampler,
    )

    # --- Model, Optimizer, and Scheduler Initialization ---
    DEVICE = rank


    model = replace_linear_with_ramtorch(model)
    model = move_model_to_device(model)


    # model.to(torch.bfloat16)

    optim = AdamW(model.parameters(), lr=training_config["lr"])
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=1e-5, end_factor=1.0, total_iters=100
    )

    # --- WandB Initialization (only on rank 0) ---
    if rank == 0 and training_config["wandb_project"]:
        wandb.init(
            project=training_config["wandb_project"],
            name=training_config["preview_path"],
        )

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as prof:
        # --- Training Loop ---
        for epoch in range(training_config["num_epochs"]):
            torch.manual_seed(epoch)
            sampler.set_epoch(epoch)  # Important for shuffling
            progress_bar = tqdm(
                total=len(loader), desc="Processing", smoothing=0.1, disable=rank != 0
            )

            for batch_idx, (real, label) in enumerate(loader):
                # --- Data Preparation ---
                real = real.to(DEVICE)#.to(torch.bfloat16)
                label = label.to(DEVICE)#.to(torch.bfloat16)  # Ensure label is float for conditioning

                # Flatten image for the model
                x1, image_shape = image_flatten(real)
                x1 = x1.requires_grad_(True)

                # Initial noise for the flow
                x0 = torch.randn_like(x1)

                # --- Training Step ---
                with torch.autocast("cuda", torch.bfloat16):
                    loss = model.compute_loss(
                        image=x1,
                        noise=x0,
                        condition=label,
                        class_dropout_ratio=training_config["class_dropout_ratio"],
                    )

                loss.backward()

                # --- Manual Gradient All-Reduce ---
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data = param.grad.data.to("cuda")
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        # param.grad.data /= world_size

                optim.step()
                lr_scheduler.step()
                optim.zero_grad()

                # --- Logging (only on rank 0) ---
                if rank == 0:
                    progress_bar.set_description(
                        f"Epoch [{epoch}/{training_config['num_epochs']}] Step [{batch_idx}/{len(loader)}] Loss: {loss:.4f}"
                    )
                    if training_config["wandb_project"]:
                        wandb.log({"Loss": loss, "Epoch": epoch})

                # --- Evaluation and Image Saving (only on rank 0) ---
                if rank == 0 and batch_idx % training_config["eval_interval"] == 0:
                    with torch.no_grad():
                        z = torch.randn_like(x1)

                        with torch.autocast("cuda", torch.bfloat16):
                            fake_images_list = []
                            for cfg, steps in training_config[
                                "inference_cfg_and_steps"
                            ]:
                                fake_images_cfg, _ = model.euler_cfg(
                                    z, label, cfg, num_steps=steps
                                )
                                fake_images_list.append(fake_images_cfg)

                            # Unflatten images for saving
                            real_unflattened = image_unflatten(x1, image_shape)
                            fake_images_unflattened = [
                                image_unflatten(img, image_shape) for img in fake_images_list
                            ]

                            # Concatenate for grid view
                            all_images = torch.cat(
                                fake_images_unflattened + [real_unflattened], dim=0
                            )

                            # Create preview directory if it doesn't exist
                            os.makedirs(training_config["preview_path"], exist_ok=True)
                            img_path = f"{training_config['preview_path']}/epoch_{epoch}_{batch_idx}.jpg"

                            save_image(
                                make_grid(
                                    (all_images.clip(-1, 1) + 1) / 2,
                                    nrow=training_config["batch_size"],
                                ),
                                img_path,
                            )
                            if training_config["wandb_project"]:
                                wandb.log({"example_image": wandb.Image(img_path)})

                if rank == 0:
                    progress_bar.update(1)

                profile_steps = 15
                # Stop profiling but continue training
                if batch_idx == profile_steps and latch:
                    prof.stop()
                    prof.export_chrome_trace(f"trace_{rank}.json")
                    print(f"Stopped profiling and saved trace at step {batch_idx}")
                    latch = False

            # --- Save Checkpoint (only on rank 0) ---
            if rank == 0:
                os.makedirs(training_config["ckpt_path"], exist_ok=True)
                torch.save(
                    model.state_dict(), f"{training_config['ckpt_path']}/{epoch}.pth"
                )

    cleanup()


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--world_size", type=int, default=1, help="Number of GPUs to use"
    # )
    # args = parser.parse_args()
    world_size = torch.cuda.device_count()

    training_config = {
        "batch_size": 64,
        "lr": 5e-5,
        "num_epochs": 1000,
        "eval_interval": 100,
        "preview_path": "/mnt/nvme-3.5tb/lodestone/celeba_flowmatching_DiT-B/2_ramtorch_v",
        "wandb_project": "x0-celeba",
        "ckpt_path": "/mnt/nvme-3.5tb/lodestone/celeba_flowmatching_DiT-B/2_ramtorch_v",
        "class_dropout_ratio": 0.1,
        "model_config": {
            "input_dim": 3 * 256,
            "output_dim": 3 * 256,
            "dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "exp_fac": 4,
            "rope_seq_length": 64**2 + 30,
            "class_count": 40,
        },
        "inference_cfg_and_steps": [
            [1, 1],
            [1, 30],
            [3, 30],
        ],
        "model_checkpoint": None,
    }

    model = Flow(**training_config["model_config"])
    if training_config["model_checkpoint"]:
        model.load_state_dict(torch.load(training_config["model_checkpoint"]))

    mp.spawn(main, args=(world_size, training_config, model), nprocs=world_size)
