"""wan_text_encoder.py — UMT5 text encoder wrapper for Wan VACE training.

Loads the UMT5-XXL encoder + tokenizer, encodes text prompts into
[B, 512, 4096] hidden states (matching the Wan VACE transformer's text_dim).
Frozen, BF16.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["WanTextEncoder"]


class WanTextEncoder(nn.Module):
    """Frozen UMT5-XXL text encoder + tokenizer.

    Args:
        model_id: HF repo id (default ``linoyts/Wan-VACE-14B-diffusers``).
        max_length: max token sequence length (Wan default 512).
    """

    TEXT_DIM = 4096

    def __init__(
        self,
        model_id: str = "linoyts/Wan-VACE-14B-diffusers",
        max_length: int = 512,
    ):
        super().__init__()
        from transformers import UMT5EncoderModel, T5TokenizerFast

        self.tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
        self.encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder")
        self.encoder.eval().requires_grad_(False)
        self.max_length = max_length

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @property
    def dtype(self):
        return next(self.encoder.parameters()).dtype

    @torch.no_grad()
    def encode(self, prompts: list[str]) -> Tensor:
        """Encode a list of text prompts into hidden states.

        Args:
            prompts: list of caption strings.

        Returns:
            [B, max_length, 4096] float tensor on the encoder's device.
        """
        # Clean prompts (strip excessive whitespace, same as diffusers prompt_clean)
        prompts = [" ".join(p.split()) for p in prompts]
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)

        enc_dtype = self.dtype
        with torch.autocast("cuda", enc_dtype, enabled=(enc_dtype != torch.float32)):
            prompt_embeds = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        return prompt_embeds.to(enc_dtype)
