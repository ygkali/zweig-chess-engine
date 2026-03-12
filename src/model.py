"""
CNN/ResNet architecture for chess move prediction.

Uses 3x3 convolutions to learn spatial piece relationships (pins, forks, pawn chains)
while preserving 8x8 board geometry through residual blocks. Only flattened at the
policy head for final move classification.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    LEGACY_CHANNELS, 
    MAIA2_CHANNELS, 
    ELO_EMBEDDING_DIM, 
    ELO_BUCKETS,
    ELO_MIN,
)


class ResBlock(nn.Module):
    """Residual block with two 3x3 convolutions and skip connection."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

class Maia1_Legacy(nn.Module):
    """
    14-channel Legacy architecture. Input: [B, 14, 8, 8].
    conv_in (14->256) -> 12 ResBlocks -> Policy Head -> move class logits.
    """
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(LEGACY_CHANNELS, 256, 3, 1, 1)
        self.bn_in = nn.BatchNorm2d(256)
        self.res_blocks = nn.Sequential(*[ResBlock(256) for _ in range(12)])
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, vocab_size),  # ~4208 UCI move vocabulary
        )

    def forward(
        self, 
        x: torch.Tensor, 
        my_elo: Optional[torch.Tensor] = None, 
        opp_elo: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # my_elo/opp_elo kept for interface compatibility, ignored in legacy
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        return self.policy_head(x)

class Maia2_New(nn.Module):
    """
    19-channel Maia-2 architecture with Skill-Aware Gating.
    Input: [B, 19, 8, 8] + my_elo, opp_elo (raw ELO values, converted internally).
    ELO embeddings modulate board features via a sigmoid gate, allowing the model
    to produce skill-dependent move predictions.
    """
    def __init__(self, vocab_size: int, channels: int = 19) -> None:
        if vocab_size is None:
            raise ValueError("vocab_size must be provided and cannot be None")
        super().__init__()
        self.conv_in = nn.Conv2d(channels, 256, 3, 1, 1)
        self.bn_in = nn.BatchNorm2d(256)
        self.res_blocks = nn.Sequential(*[ResBlock(256) for _ in range(12)])
        
        self.elo_emb = nn.Embedding(ELO_BUCKETS, ELO_EMBEDDING_DIM)
        self.skill_proj = nn.Linear(256, 256)
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, vocab_size),
        )

    def _elo_to_index(self, elo: torch.Tensor) -> torch.Tensor:
        """Convert ELO value to embedding index (vectorized)."""
        return (elo - ELO_MIN).clamp(0, ELO_BUCKETS - 1)

    def forward(
        self, 
        x: torch.Tensor, 
        my_elo: torch.Tensor, 
        opp_elo: torch.Tensor
    ) -> torch.Tensor:
        feat = F.relu(self.bn_in(self.conv_in(x)))
        feat = self.res_blocks(feat)
        
        my_elo_idx = self._elo_to_index(my_elo)
        opp_elo_idx = self._elo_to_index(opp_elo)
        
        e1 = self.elo_emb(my_elo_idx)
        e2 = self.elo_emb(opp_elo_idx)
        
        combined_elo = torch.cat([e1, e2], dim=1)  # [B, 256]
        b, c, h, w = feat.shape
        skill_gate = torch.sigmoid(self.skill_proj(combined_elo)).view(b, c, 1, 1)
        
        return self.policy_head(feat * skill_gate)