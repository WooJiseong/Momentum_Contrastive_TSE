from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from pn_attn_mococo.soft_moco import SoftMomentumContrastivePNLearner


class LightweightAttentionPooling(nn.Module):
    """Windowed average pooling followed by a learned scalar gate per token."""

    def __init__(
        self,
        in_ch: int,
        in_freq: int,
        pooling_size: int = 40,
        stride: int = 40,
        gate_hidden_dim: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()
        if pooling_size <= 0 or stride <= 0:
            raise ValueError("pooling_size and stride must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.in_ch = int(in_ch)
        self.in_freq = int(in_freq)
        self.pooling_size = int(pooling_size)
        self.stride = int(stride)
        self.temperature = float(temperature)
        self.gate = nn.Sequential(
            nn.Linear(self.in_ch, int(gate_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(gate_hidden_dim), 1),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        if emb.ndim != 4:
            raise ValueError(f"Expected [B,C,T,F], got {tuple(emb.shape)}")
        batch, channels, frames, freqs = emb.shape
        if channels != self.in_ch or freqs != self.in_freq:
            raise ValueError(
                "Projection input shape mismatch: "
                f"expected C/F=({self.in_ch},{self.in_freq}), got ({channels},{freqs})"
            )
        if frames < self.pooling_size:
            raise ValueError(
                f"Need at least {self.pooling_size} temporal frames, got {frames}"
            )

        # Pool local temporal windows first. The gate then scores only the
        # resulting tokens instead of constructing a T-by-T attention matrix.
        tokens = emb.float().permute(0, 1, 3, 2).reshape(batch, channels * freqs, frames)
        tokens = F.avg_pool1d(tokens, kernel_size=self.pooling_size, stride=self.stride)
        token_count = tokens.shape[-1]
        tokens = tokens.transpose(1, 2).reshape(batch, token_count, channels, freqs)

        gate_input = tokens.mean(dim=3)  # [B, token_count, C]
        scores = self.gate(gate_input).squeeze(-1) / self.temperature
        weights = torch.softmax(scores, dim=1)
        return (tokens * weights[:, :, None, None]).sum(dim=1)


class LightweightAttentionProjectionHead(nn.Module):
    """Projection head compatible with the original 256-D MoCo output."""

    def __init__(
        self,
        in_ch: int,
        in_freq: int,
        hidden_dim: int,
        emb_dim: int,
        nlayers: int = 3,
        use_bn: bool = False,
        pooling_size: int = 40,
        pooling_stride: int = 40,
        gate_hidden_dim: int = 64,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        if nlayers <= 1:
            layers.append(nn.Linear(in_ch * in_freq, emb_dim))
        else:
            layers.append(nn.Linear(in_ch * in_freq, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, emb_dim))
        self.pool = LightweightAttentionPooling(
            in_ch=in_ch,
            in_freq=in_freq,
            pooling_size=pooling_size,
            stride=pooling_stride,
            gate_hidden_dim=gate_hidden_dim,
            temperature=gate_temperature,
        )
        self.net = nn.Sequential(*layers)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(emb).flatten(1)
        return F.normalize(self.net(pooled), dim=-1)


class AttnSoftMomentumContrastivePNLearner(SoftMomentumContrastivePNLearner):
    """Soft_MOCOCO with 40-frame Lightweight Attention Projection Pooling."""

    def __init__(self, cfg: dict):
        projection_cfg = dict(cfg.get("contrastive", {}).get("projection", {}))
        base_cfg = copy.deepcopy(cfg)
        base_projection_cfg = dict(projection_cfg)
        for key in (
            "pooling_size",
            "pooling_stride",
            "gate_hidden_dim",
            "gate_temperature",
        ):
            base_projection_cfg.pop(key, None)
        base_cfg.setdefault("contrastive", {})["projection"] = base_projection_cfg
        super().__init__(base_cfg)
        self.student_head = LightweightAttentionProjectionHead(**projection_cfg)
        self.momentum_head = copy.deepcopy(self.student_head)
        for parameter in self.momentum_head.parameters():
            parameter.requires_grad = False
