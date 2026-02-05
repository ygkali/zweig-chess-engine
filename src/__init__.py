"""
Chess AI - Modular Package

This package provides PyTorch implementations of human-like chess move prediction
at different skill levels, inspired by Microsoft's Maia Chess project.
"""
from __future__ import annotations

from .model import Maia1_Legacy, Maia2_New
from .utils import create_vocab, mirror_move, mirror_move_uci, get_inverse_vocab

# Dynamically calculate vocabulary size (cached after first call)
VOCAB_SIZE: int = len(create_vocab())

__all__ = [
    "Maia1_Legacy",
    "Maia2_New",
    "create_vocab",
    "get_inverse_vocab",
    "mirror_move",
    "mirror_move_uci",
    "VOCAB_SIZE",
]

__version__ = "0.2.0"
