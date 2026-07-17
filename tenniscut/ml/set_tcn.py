"""Tiny Set-TCN rally decoder (player-set encoder + temporal conv)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional at import time
    torch = None
    nn = object


class PlayerSetEncoder(nn.Module if torch else object):
    """Encode variable player sets per frame with mean/max + near/far summaries."""

    def __init__(self, player_dim: int, hidden: int = 32):
        if torch is None:
            raise ImportError("torch required for Set-TCN")
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(player_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden * 4 + 1, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, P, F)
        b, t, p, f = x.shape
        flat = x.reshape(b * t, p, f)
        mask = (flat.abs().sum(dim=-1) > 0).float()
        emb = self.mlp(flat)
        emb = emb * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_pool = emb.sum(dim=1) / denom
        max_pool = emb.max(dim=1).values
        # near/far: assume last 2 dims of player features include role flags
        near_mask = flat[..., -3] > 0.5
        far_mask = flat[..., -2] > 0.5
        near_emb = (emb * near_mask.unsqueeze(-1)).sum(1) / near_mask.sum(1, keepdim=True).clamp(min=1.0)
        far_emb = (emb * far_mask.unsqueeze(-1)).sum(1) / far_mask.sum(1, keepdim=True).clamp(min=1.0)
        count = mask.sum(dim=1, keepdim=True) / float(p)
        scene = torch.cat([mean_pool, max_pool, near_emb, far_emb, count], dim=-1)
        scene = self.out(scene)
        return scene.reshape(b, t, -1)


class TCNBlock(nn.Module if torch else object):
    def __init__(self, channels: int, kernel: int, dilation: int):
        if torch is None:
            raise ImportError("torch required for Set-TCN")
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return F.relu(self.bn(self.conv(x)))


class SetTCNRallyDecoder(nn.Module if torch else object):
    def __init__(
        self,
        player_dim: int,
        hidden: int = 32,
        dilations: Optional[List[int]] = None,
    ):
        if torch is None:
            raise ImportError("torch required for Set-TCN")
        super().__init__()
        dilations = dilations or [1, 2, 4, 8, 16]
        self.encoder = PlayerSetEncoder(player_dim, hidden=hidden)
        self.tcn = nn.ModuleList([TCNBlock(hidden, kernel=3, dilation=d) for d in dilations])
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, P, F) -> logits (B, T)
        scene = self.encoder(x)
        y = scene.transpose(1, 2)
        for block in self.tcn:
            y = block(y)
        logits = self.head(y).squeeze(1)
        return logits


def temporal_smoothness_loss(probs: torch.Tensor, weight: float = 0.15) -> torch.Tensor:
    if probs.shape[1] < 2:
        return probs.new_tensor(0.0)
    diff = probs[:, 1:] - probs[:, :-1]
    return weight * (diff ** 2).mean()


@dataclass
class SetTCNConfig:
    player_dim: int
    hidden: int = 32
    dilations: Optional[List[int]] = None


def save_set_tcn(model: "SetTCNRallyDecoder", path: Path, config: SetTCNConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config.__dict__}, path)


def load_set_tcn(path: Path) -> tuple["SetTCNRallyDecoder", SetTCNConfig]:
    if torch is None:
        raise ImportError("torch required")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = SetTCNConfig(**payload["config"])
    model = SetTCNRallyDecoder(
        player_dim=config.player_dim,
        hidden=config.hidden,
        dilations=config.dilations,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config


def predict_sequence(model: "SetTCNRallyDecoder", frames: np.ndarray) -> np.ndarray:
    """frames: (T, P, F) -> probabilities (T,)"""
    if torch is None:
        raise ImportError("torch required")
    with torch.no_grad():
        x = torch.from_numpy(frames).float().unsqueeze(0)
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    return probs
