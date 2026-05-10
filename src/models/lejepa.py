"""lejepa.py — LeJEPA encoder for self-supervised pre-training.

Architecture:
  - Token layout: [cls(1), register_tokens(n_reg), image_patches(H*W)]
  - CLS token output is used as the image embedding (emb).
  - A 3-layer MLP projector (with BatchNorm1d, critical per empirical testing)
    maps emb → proj for the SIGReg + invariance loss.
  - M-RoPE positional encoding with optional pos_jitter for sequence-length
    equivariance across bucketed resolutions.
  - encode_patches() exposes raw patch token features for PCA visualization.

References:
  LeJEPA: Balestriero & LeCun, arXiv:2511.08544
  MINIMAL.md: https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.ops import MLP

from src.models.flow import (
    TransformerNetwork,
    build_mrope_position_ids,
    image_flatten,
)


# ---------------------------------------------------------------------------
# SIGReg — Characteristic-function regularizer
# ---------------------------------------------------------------------------

class SIGReg(nn.Module):
    """Spectral Independence Gaussian Regularizer (SIGReg).

    Minimises the distance between the empirical characteristic function of
    the projected embeddings and that of a standard Gaussian, integrated over
    t ∈ [0, t_max] using the trapezoidal rule.

    The symmetric ECF property is exploited: integrating on [0, t_max] and
    doubling the weights is equivalent to integrating on [-t_max, t_max].

    Args:
        knots:   Number of quadrature points (default 17, as in the paper).
        t_max:   Upper integration limit (default 3.0).
        n_proj:  Number of random projection directions (default 256).
    """

    def __init__(self, knots: int = 17, t_max: float = 3.0, n_proj: int = 256):
        super().__init__()
        self.n_proj = n_proj
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        # Trapezoidal weights, doubled for the symmetric trick
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        # Gaussian window: E[e^{itX}] for X~N(0,1) is e^{-t²/2}
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss.

        Args:
            proj: [V, B, proj_dim]  (V views, B samples per view)

        Returns:
            Scalar loss.
        """
        # Flatten views into the batch dimension: [V*B, proj_dim]
        V, B, D = proj.shape
        flat = proj.reshape(V * B, D)

        # Random unit-norm projection directions: [D, n_proj]
        A = torch.randn(D, self.n_proj, device=flat.device, dtype=flat.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp(min=1e-8)

        # Projected scalars scaled by quadrature points: [V*B, n_proj, knots]
        x_t = (flat @ A).unsqueeze(-1) * self.t  # broadcast over knots

        # Empirical CF vs Gaussian CF
        err = (
            (x_t.cos().mean(0) - self.phi).square()   # real part
            + x_t.sin().mean(0).square()               # imaginary part
        )  # [n_proj, knots]

        statistic = (err @ self.weights) * (V * B)     # [n_proj]
        return statistic.mean()


# ---------------------------------------------------------------------------
# LeJEPA Encoder
# ---------------------------------------------------------------------------

class LeJEPAEncoder(nn.Module):
    """LeJEPA image encoder.

    Token layout (sequence order):
        [CLS (1)] [register tokens (n_register_tokens)] [image patches (H*W)]

    The CLS token output is used as the image-level embedding.  A 3-layer MLP
    projector (with BatchNorm1d) maps it to the projection space used by the
    SIGReg + invariance loss.

    ``encode_patches`` returns the raw patch token features (after the
    transformer's output norm, before the projector) for PCA visualisation.
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
        pos_jitter_range: int = 32,
        compile_blocks: bool = False,
        patch_size: int = 16,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self._patch_size = patch_size
        self.n_register_tokens = n_register_tokens
        self.pos_jitter_range = pos_jitter_range

        # --- Special tokens ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if n_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, n_register_tokens, dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        # --- Patch projection ---
        self.input_layer = nn.Linear(input_dim, dim)

        # --- Transformer backbone ---
        self.transformer = TransformerNetwork(
            input_dim=dim,
            output_dim=dim,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            exp_fac=exp_fac,
            rope_seq_length=rope_seq_length,
            input_proj=False,   # we project patches ourselves above
            final_head=False,   # keep dim-dimensional output
            final_norm=True,
            compile_blocks=compile_blocks,
        )

        # --- Projector head (original LeJEPA design) ---
        # BatchNorm1d is empirically critical — do not replace with LayerNorm.
        self.projector = MLP(
            in_channels=dim,
            hidden_channels=[2048, 2048, proj_dim],
            norm_layer=lambda c: nn.BatchNorm1d(c),
            activation_layer=nn.GELU,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def compile_blocks(self) -> None:
        """Compile transformer blocks with torch.compile.

        Call *after* any FX-based model wrapping (e.g. MultiGPUWrapper.setup())
        to avoid the "FX tracing a dynamo-optimized function" error.
        """
        self.transformer.compile_blocks()

    @property
    def device(self):
        return next(self.parameters()).device

    def _n_prefix(self) -> int:
        """Number of prefix tokens: CLS + registers."""
        return 1 + self.n_register_tokens

    def _build_tokens(self, x: torch.Tensor):
        """Flatten image → patches, project, prepend CLS + registers.

        Args:
            x: [B, C, H, W]

        Returns:
            tokens:  [B, n_prefix + H*W, dim]
            num_h:   number of patch rows
            num_w:   number of patch cols
        """
        B, C, H, W = x.shape
        patches, _ = image_flatten(x, shuffle_size=self._patch_size)  # [B, H*W, input_dim]
        patches_proj = self.input_layer(patches)                       # [B, H*W, dim]

        cls = self.cls_token.expand(B, -1, -1)                        # [B, 1, dim]
        parts = [cls]
        if self.n_register_tokens > 0:
            parts.append(self.register_tokens.expand(B, -1, -1))
        parts.append(patches_proj)

        tokens = torch.cat(parts, dim=1)                               # [B, n_prefix+H*W, dim]
        num_h = H // self._patch_size
        num_w = W // self._patch_size
        return tokens, num_h, num_w

    def _build_position_ids(self, B: int, num_h: int, num_w: int, device, jitter: int = 0):
        """Build M-RoPE position IDs.

        CLS occupies the n_class=1 slot (diagonal [0,0,0]).
        Registers follow on the diagonal.
        Image patches start at patch_start = n_prefix + jitter.
        """
        return build_mrope_position_ids(
            B, num_h, num_w, device=device,
            n_text=0,
            n_time=0,
            n_class=1,          # CLS token
            n_reg=self.n_register_tokens,
            pos_jitter=jitter,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of images.

        Args:
            x: [B, C, H, W]  — pixel values in any range (no normalisation
               assumed here; caller is responsible for augmentation/normalisation)

        Returns:
            emb:  [B, dim]       — CLS token output (image-level embedding)
            proj: [B, proj_dim]  — MLP projector output (used for LeJEPA loss)
        """
        B = x.shape[0]
        tokens, num_h, num_w = self._build_tokens(x)

        jitter = 0
        if self.training and self.pos_jitter_range > 0:
            jitter = int(torch.randint(0, self.pos_jitter_range + 1, (1,)).item())

        position_ids = self._build_position_ids(B, num_h, num_w, x.device, jitter)

        out = self.transformer(tokens, attention_mask=None, position_ids=position_ids)
        # out: [B, n_prefix + H*W, dim]

        emb = out[:, 0, :]          # CLS token output: [B, dim]
        proj = self.projector(emb)  # [B, proj_dim]
        return emb, proj

    # ------------------------------------------------------------------
    # PCA visualisation helper
    # ------------------------------------------------------------------

    def encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw patch token features for PCA visualisation.

        Runs the full transformer forward pass but returns only the image-patch
        portion of the output (positions [n_prefix:]), after the transformer's
        output norm and before the projector.

        Args:
            x: [B, C, H, W]

        Returns:
            patch_tokens: [B, H*W, dim]
        """
        B = x.shape[0]
        tokens, num_h, num_w = self._build_tokens(x)
        position_ids = self._build_position_ids(B, num_h, num_w, x.device, jitter=0)

        out = self.transformer(tokens, attention_mask=None, position_ids=position_ids)
        n_prefix = self._n_prefix()
        return out[:, n_prefix:, :]   # [B, H*W, dim]
