"""
CNN/ResNet Architecture: How the model "sees" the board.

=== SCANNING: "Flashlight" TECHNIQUE (Conv2d) ===
The model slides small 3x3 windows (kernels) across the board.
- Looks not at a single square, but at that square + its direct neighbors.
- Learns: "If queen on square X and opponent king on diagonal -> strong position"
- Learns piece relationships (pins, forks, pawn chains).

=== SPATIAL MEMORY (Spatial Preservation) ===
Throughout ResBlock layers, data remains in [B, C, 8, 8] shape.
- Board geometry is preserved until the final stage of training.
- "This info comes from top-right corner of the board" knowledge is retained.
- Only flattened in Policy Head for move class selection.
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
    """
    Residual Block: Scans the board with 3x3 convolution.
    Each layer learns more abstract patterns (pawn chain -> fianchetto -> pin).
    """
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
    14-channel Legacy architecture. Input: [Batch, 14, 8, 8]

    Flow: conv_in (14->256) -> 12 ResBlock (8x8 preserved) -> Policy Head (flatten -> move class)
    Model doesn't hardcode chess rules; learns implicitly from millions of examples.
    """
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(LEGACY_CHANNELS, 256, 3, 1, 1)
        self.bn_in = nn.BatchNorm2d(256)
        self.res_blocks = nn.Sequential(*[ResBlock(256) for _ in range(12)])
        
        # Policy Head: Only flatten the 8x8 map here, select move class
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
    Input: [Batch, 19, 8, 8] + my_elo, opp_elo

    Maia-2 revolution: Doesn't just look at the board, also asks "Who is playing?"
    - ELO 1100: Focuses on short-term threats (knight forks).
    - ELO 2500: Focuses on long-term positional advantages.
    - skill_gate: Multiplies board features with skill vector (feat * skill_gate).
    
    NOTE: ELO values are given directly (400-3200), converted to index inside the model.
    """
    def __init__(self, vocab_size: int, channels: int = 19) -> None:
        if vocab_size is None:
            raise ValueError("vocab_size must be provided and cannot be None")
        super().__init__()
        self.conv_in = nn.Conv2d(channels, 256, 3, 1, 1)
        self.bn_in = nn.BatchNorm2d(256)
        self.res_blocks = nn.Sequential(*[ResBlock(256) for _ in range(12)])
        
        # Skill-Aware: ELO -> Skill Vector -> Channel Gate
        self.elo_emb = nn.Embedding(ELO_BUCKETS, ELO_EMBEDDING_DIM)
        self.skill_proj = nn.Linear(256, 256)  # ELO embeddings -> channel gate
        
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
        
        # ELO -> Index -> Skill Vector (Embedding)
        my_elo_idx = self._elo_to_index(my_elo)
        opp_elo_idx = self._elo_to_index(opp_elo)
        
        e1 = self.elo_emb(my_elo_idx)
        e2 = self.elo_emb(opp_elo_idx)
        
        # Skill Gate: Modulates board features with skill
        combined_elo = torch.cat([e1, e2], dim=1)  # [Batch, 256]
        b, c, h, w = feat.shape
        
        # Reshape for broadcasting: [Batch, Channels, 1, 1]
        skill_gate = torch.sigmoid(self.skill_proj(combined_elo)).view(b, c, 1, 1)
        
        return self.policy_head(feat * skill_gate)