"""gan.py — Generator and Discriminator for Relativistic GAN.

Both models share the same FlowBaseline-style TransformerNetwork backbone.

Token layouts
-------------
Generator:
    [text_tokens (optional)] [register_tokens (learned)] [noise_patches]

Discriminator:
    [cls_embed (1)] [text_tokens (optional)] [register_tokens (learned)]
    [timestep_embed (1)] [image_patches]

Discriminator output
--------------------
The final CLS token is fed through a JEPA-style 3-layer MLP projector
(dim → 2048 → 2048 → disc_embed_dim).  The caller computes per-sample
scalar logits as the mean dot-product of the projector output against a
label vector:
    real_vec = ones(disc_embed_dim)   # tinkerable
    fake_vec = zeros(disc_embed_dim)  # tinkerable
    logit = (D_out * label_vec).mean(dim=-1)   # [B]

This lets you later experiment with arbitrary label geometries without
touching the model code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP

from src.models.flow import (
    TransformerNetwork,
    build_mrope_position_ids,
    image_flatten,
    image_unflatten,
)


# ---------------------------------------------------------------------------
# SoftClamp norm (same as lejepa.py, kept local to avoid circular imports)
# ---------------------------------------------------------------------------

class SoftClamp(nn.Module):
    """Pointwise learnable norm: scale * tanh(x * alpha) + shift."""
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(x * self.alpha) + self.shift


# ---------------------------------------------------------------------------
# Label vector helpers
# ---------------------------------------------------------------------------

def real_label_vector(disc_embed_dim: int, device, dtype=torch.float32) -> torch.Tensor:
    """[disc_embed_dim] — all-ones label for real samples."""
    return torch.ones(disc_embed_dim, device=device, dtype=dtype)


def fake_label_vector(disc_embed_dim: int, device, dtype=torch.float32) -> torch.Tensor:
    """[disc_embed_dim] — all-zeros label for fake samples."""
    return torch.zeros(disc_embed_dim, device=device, dtype=dtype)


def embed_to_logit(embed: torch.Tensor, label_vec: torch.Tensor) -> torch.Tensor:
    """Mean dot product of D output against a label vector → [B] scalar logits.

    Args:
        embed:     [B, disc_embed_dim]
        label_vec: [disc_embed_dim]

    Returns:
        [B] per-sample scalar logit.
    """
    return (embed * label_vec.unsqueeze(0)).mean(dim=-1)


# ---------------------------------------------------------------------------
# Relativistic loss helpers
# ---------------------------------------------------------------------------

def d_relativistic_loss(
    d_real: torch.Tensor,
    d_fake: torch.Tensor,
    disc_embed_dim: int,
) -> torch.Tensor:
    """Relativistic discriminator loss (RaGAN).

    Logit = dot(D(real), ones) - dot(D(fake), zeros)
          = mean(D(real), dim=-1) - 0
    But the relativistic twist computes:
        logit = dot(D(real), real_vec) - dot(D(fake), fake_vec)
    then applies softplus(-logit).

    Args:
        d_real: [B, disc_embed_dim]
        d_fake: [B, disc_embed_dim]
        disc_embed_dim: used to build label vectors on the right device.

    Returns:
        scalar loss.
    """
    device, dtype = d_real.device, d_real.dtype
    real_vec = real_label_vector(disc_embed_dim, device, dtype)
    fake_vec = fake_label_vector(disc_embed_dim, device, dtype)
    logit_real = embed_to_logit(d_real, real_vec)   # [B]
    logit_fake = embed_to_logit(d_fake, fake_vec)   # [B]  (all zeros → 0)
    relativistic_logit = logit_real - logit_fake     # [B]
    return F.softplus(-relativistic_logit).mean()


def g_relativistic_loss(
    d_fake: torch.Tensor,
    d_real_detached: torch.Tensor,
    disc_embed_dim: int,
) -> torch.Tensor:
    """Relativistic generator loss (RaGAN).

    G wants D(fake) to look real and D(real) to look fake:
        logit = dot(D(fake), real_vec) - dot(D(real), fake_vec)
              = mean(D(fake), dim=-1) - 0

    Args:
        d_fake:          [B, disc_embed_dim]  — from current G output, with grad
        d_real_detached: [B, disc_embed_dim]  — no grad
        disc_embed_dim:  used to build label vectors.

    Returns:
        scalar loss.
    """
    device, dtype = d_fake.device, d_fake.dtype
    real_vec = real_label_vector(disc_embed_dim, device, dtype)
    fake_vec = fake_label_vector(disc_embed_dim, device, dtype)
    logit_fake_as_real = embed_to_logit(d_fake, real_vec)          # [B]
    logit_real_as_fake = embed_to_logit(d_real_detached, fake_vec) # [B]  (all zeros → 0)
    relativistic_logit = logit_fake_as_real - logit_real_as_fake   # [B]
    return F.softplus(-relativistic_logit).mean()


# ---------------------------------------------------------------------------
# GANGenerator
# ---------------------------------------------------------------------------

class GANGenerator(nn.Module):
    """Transformer-based generator.

    Input token layout:
        [text_tokens (n_text, optional)] [register_tokens (n_reg)] [noise_patches (seq)]

    ``forward`` maps flattened noise patches (same shape as image patches from
    ``image_flatten``) through the transformer and returns predicted patches.
    Caller is responsible for ``image_unflatten`` to recover pixel-space tensors.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dim: int,
        num_layers: int = 12,
        num_heads: int = 8,
        exp_fac: int = 4,
        rope_seq_length: int = 10000,
        n_register_tokens: int = 8,
        use_text_embed: bool = False,
        text_vocab_size: int = 32000,
        patch_size: int = 16,
        compile_blocks: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self._patch_size = patch_size
        self.n_register_tokens = n_register_tokens
        self.use_text_embed = use_text_embed

        # --- Patch projection ---
        self.input_layer = nn.Linear(input_dim, dim)

        # --- Optional text conditioning ---
        if use_text_embed:
            self.token_embed = nn.Embedding(text_vocab_size, dim)
            self.text_norm = nn.RMSNorm(dim, elementwise_affine=True)

        # --- Learned register tokens ---
        if n_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, n_register_tokens, dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        # --- Transformer backbone ---
        # input_proj=False: we project patches ourselves via input_layer.
        # final_head=True:  includes the output linear (dim → output_dim).
        self.transformer = TransformerNetwork(
            input_dim=dim,
            output_dim=output_dim,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            exp_fac=exp_fac,
            rope_seq_length=rope_seq_length,
            input_proj=False,
            final_head=True,
            final_norm=True,
            compile_blocks=compile_blocks,
        )

    def compile_blocks(self) -> None:
        """Compile transformer blocks. Call after MultiGPUWrapper.setup()."""
        self.transformer.compile_blocks()

    @property
    def device(self):
        return next(self.parameters()).device

    def _n_prefix(self, n_text: int) -> int:
        return n_text + self.n_register_tokens

    def _build_position_ids(
        self, B: int, n_text: int, num_h: int, num_w: int, device
    ) -> torch.Tensor:
        """M-RoPE position ids for G token layout.

        Uses n_class=0, n_time=0; text + register tokens occupy the diagonal
        prefix; noise patches expand spatially from patch_start.
        """
        return build_mrope_position_ids(
            B, num_h, num_w, device=device,
            n_text=n_text,
            n_time=0,
            n_class=0,
            n_reg=self.n_register_tokens,
        )

    def forward(
        self,
        noise: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate fake image patches from noise.

        Args:
            noise:       [B, C, H, W] — noise tensor in image shape.
            text_tokens: [B, L] int64 — optional token ids.

        Returns:
            pred_patches: [B, H_p*W_p, output_dim] — flattened patch predictions.
        """
        B, C, H, W = noise.shape
        num_h = H // self._patch_size
        num_w = W // self._patch_size

        noise_patches, _ = image_flatten(noise, shuffle_size=self._patch_size)  # [B, H*W, input_dim]
        noise_proj = self.input_layer(noise_patches)                              # [B, H*W, dim]

        parts = []
        n_text = 0
        if self.use_text_embed and text_tokens is not None:
            txt = self.text_norm(self.token_embed(text_tokens))
            parts.append(txt)
            n_text = txt.shape[1]

        if self.n_register_tokens > 0:
            parts.append(self.register_tokens.expand(B, -1, -1))

        parts.append(noise_proj)
        tokens = torch.cat(parts, dim=1)  # [B, n_prefix + H*W, dim]

        position_ids = self._build_position_ids(B, n_text, num_h, num_w, noise.device)

        output_tokens = self.transformer(tokens, attention_mask=None, position_ids=position_ids)

        n_prefix = self._n_prefix(n_text)
        pred_patches = output_tokens[:, n_prefix:, :]  # [B, H*W, output_dim]
        return pred_patches


# ---------------------------------------------------------------------------
# GANDiscriminator
# ---------------------------------------------------------------------------

class GANDiscriminator(nn.Module):
    """Transformer-based discriminator with JEPA-style projector head.

    Input token layout:
        [cls_embed (1)] [text_tokens (n_text, optional)]
        [register_tokens (n_reg)] [timestep_embed (1)] [image_patches]

    The CLS token output is fed through a 3-layer MLP projector to yield
    a ``[B, disc_embed_dim]`` embedding.  The caller computes the relativistic
    logit via ``embed_to_logit(embed, label_vec)``.
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        num_layers: int = 12,
        num_heads: int = 8,
        exp_fac: int = 4,
        rope_seq_length: int = 10000,
        n_register_tokens: int = 8,
        use_text_embed: bool = False,
        text_vocab_size: int = 32000,
        class_count: int = 1000,
        disc_embed_dim: int = 256,
        proj_norm: str = "batchnorm",
        patch_size: int = 16,
        compile_blocks: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self._patch_size = patch_size
        self.n_register_tokens = n_register_tokens
        self.use_text_embed = use_text_embed
        self.disc_embed_dim = disc_embed_dim

        # --- CLS (class label) embedding ---
        self.cls_embed = nn.Embedding(class_count, dim)
        nn.init.trunc_normal_(self.cls_embed.weight, std=0.02)

        # --- Timestep (noise level) embedding ---
        # Receives noise_lerp_val scalar in [0, 1]
        self.timestep_embed = nn.Linear(1, dim)
        nn.init.zeros_(self.timestep_embed.weight)
        nn.init.zeros_(self.timestep_embed.bias)

        # --- Optional text conditioning ---
        if use_text_embed:
            self.token_embed = nn.Embedding(text_vocab_size, dim)
            self.text_norm = nn.RMSNorm(dim, elementwise_affine=True)

        # --- Learned register tokens ---
        if n_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, n_register_tokens, dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        # --- Patch projection ---
        self.input_layer = nn.Linear(input_dim, dim)

        # --- Transformer backbone ---
        # No final_head: we take raw dim-dimensional CLS output into projector.
        self.transformer = TransformerNetwork(
            input_dim=dim,
            output_dim=dim,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            exp_fac=exp_fac,
            rope_seq_length=rope_seq_length,
            input_proj=False,
            final_head=False,   # keep dim-dimensional output; projector follows
            final_norm=True,
            compile_blocks=compile_blocks,
        )

        # --- JEPA-style projector head ---
        # 3-layer MLP: dim → 2048 → 2048 → disc_embed_dim
        # Norm layer is BatchNorm1d (original LeJEPA design) or SoftClamp.
        _valid = ("batchnorm", "softclamp")
        if proj_norm not in _valid:
            raise ValueError(f"proj_norm must be one of {_valid}, got {proj_norm!r}")
        norm_factory = (
            (lambda c: nn.BatchNorm1d(c))
            if proj_norm == "batchnorm"
            else (lambda c: SoftClamp(c))
        )
        self.projector = MLP(
            in_channels=dim,
            hidden_channels=[2048, 2048, disc_embed_dim],
            norm_layer=norm_factory,
            activation_layer=nn.GELU,
        )

    def compile_blocks(self) -> None:
        """Compile transformer blocks. Call after MultiGPUWrapper.setup()."""
        self.transformer.compile_blocks()

    @property
    def device(self):
        return next(self.parameters()).device

    def _n_prefix(self, n_text: int) -> int:
        # cls(1) + text(n_text) + register(n_reg) + timestep(1)
        return 1 + n_text + self.n_register_tokens + 1

    def _build_position_ids(
        self, B: int, n_text: int, num_h: int, num_w: int, device
    ) -> torch.Tensor:
        """M-RoPE position ids for D token layout.

        n_class=1  → CLS token occupies the first diagonal slot.
        n_time=1   → timestep token occupies the next diagonal slot.
        n_reg      → register tokens follow.
        text tokens follow registers on the diagonal (pass as n_text).
        """
        return build_mrope_position_ids(
            B, num_h, num_w, device=device,
            n_text=n_text,
            n_time=1,       # timestep embed slot
            n_class=1,      # CLS token
            n_reg=self.n_register_tokens,
        )

    def forward(
        self,
        image_patches: torch.Tensor,
        cls_label: torch.Tensor,
        image_shape: tuple,
        text_tokens: torch.Tensor | None = None,
        noise_level: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Discriminate a batch of (possibly noisy) image patches.

        Args:
            image_patches: [B, H_p*W_p, input_dim] — flattened patches.
            cls_label:     [B] int64 — class label indices.
            image_shape:   (B, C, H, W) of the original image (for num_h/num_w).
            text_tokens:   [B, L] int64 — optional.
            noise_level:   scalar or [B] float — noise_lerp_val for the timestep embed.

        Returns:
            [B, disc_embed_dim] — discriminator embedding.
        """
        B, C, H, W = image_shape
        num_h = H // self._patch_size
        num_w = W // self._patch_size

        # --- Build prefix tokens ---
        # 1. CLS embed
        cls_vec = self.cls_embed(cls_label)[:, None, :]  # [B, 1, dim]

        # 2. Optional text
        parts = [cls_vec]
        n_text = 0
        if self.use_text_embed and text_tokens is not None:
            txt = self.text_norm(self.token_embed(text_tokens))
            parts.append(txt)
            n_text = txt.shape[1]

        # 3. Register tokens
        if self.n_register_tokens > 0:
            parts.append(self.register_tokens.expand(B, -1, -1))

        # 4. Timestep embed (noise_lerp_val → dim)
        if noise_level is None:
            ts = torch.zeros(B, 1, device=image_patches.device, dtype=image_patches.dtype)
        elif isinstance(noise_level, (float, int)):
            ts = torch.full((B, 1), noise_level, device=image_patches.device, dtype=image_patches.dtype)
        else:
            # tensor: scalar or [B]
            ts = noise_level.float().view(B, 1).to(image_patches.device)
        timestep_vec = self.timestep_embed(ts).unsqueeze(1)  # [B, 1, dim]
        parts.append(timestep_vec)

        # 5. Image patches
        img_proj = self.input_layer(image_patches)  # [B, H*W, dim]
        parts.append(img_proj)

        tokens = torch.cat(parts, dim=1)  # [B, n_prefix + H*W, dim]

        position_ids = self._build_position_ids(B, n_text, num_h, num_w, image_patches.device)

        out = self.transformer(tokens, attention_mask=None, position_ids=position_ids)
        # out: [B, n_prefix + H*W, dim]

        cls_out = out[:, 0, :]           # [B, dim] — CLS token
        embed = self.projector(cls_out)  # [B, disc_embed_dim]
        return embed
