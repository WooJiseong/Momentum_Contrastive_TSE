"""Frozen PN loader for the sibling SIA Flow and t-predictor trainers."""

from __future__ import annotations

import torch
from torch import nn

from improved_pn import load_trainable_pn_encoder


def _ensure_channel(wave: torch.Tensor) -> torch.Tensor:
    return wave.unsqueeze(1) if wave.ndim == 2 else wave


class FrozenImprovedPNEncoder(nn.Module):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.model = load_trainable_pn_encoder(
            path,
            num_blocks=1,
            head_layers=2,
            binaural=False,
            strict=True,
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def encode(self, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        pos = _ensure_channel(pos)
        neg = _ensure_channel(neg)
        pos_emb = self.model.encoder(pos.transpose(1, 2))
        neg_emb = self.model.encoder(neg.transpose(1, 2))
        cond = self.model.encoder_head(pos_emb, neg_emb)
        return cond[:, :, :pos_emb.shape[2]]


def load_pn_encoder(ckpt_path: str, device="cpu") -> nn.Module:
    """Reference-compatible loader: eval mode, frozen weights, .encode()."""
    return FrozenImprovedPNEncoder(ckpt_path).to(device)

