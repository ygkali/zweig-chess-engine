"""
Move vocabulary and board-to-tensor conversion utilities.

Board state is encoded as multi-channel bitboards (14ch or 19ch) where each channel
represents the presence of a specific piece type or game state feature on the 8x8 grid.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Tuple

import chess
import numpy as np
import numpy.typing as npt

from src.config import LEGACY_CHANNELS, MAIA2_CHANNELS


# --- VOCABULARY ---
_VOCAB_CACHE: Optional[Dict[str, int]] = None


def create_vocab() -> Dict[str, int]:
    """Build and cache the UCI move vocabulary (~4208 move classes)."""
    global _VOCAB_CACHE
    if _VOCAB_CACHE is not None:
        return _VOCAB_CACHE
    
    _VOCAB_CACHE = _build_vocab()
    return _VOCAB_CACHE


def _build_vocab() -> Dict[str, int]:
    """Build vocabulary dictionary (internal)."""
    moves = []
    for f in range(64):
        for t in range(64):
            if f == t:
                continue
            moves.append(chess.Move(f, t).uci())
    
    promotions = ['q', 'r', 'b', 'n']
    for f in range(8):
        for d in [-1, 0, 1]:
            t = f + d
            if 0 <= t <= 7:
                # White promotion (rank 6 -> 7)
                moves.extend([
                    chess.Move(
                        chess.square(f, 6),
                        chess.square(t, 7),
                        promotion=chess.Piece.from_symbol(p).piece_type,
                    ).uci()
                    for p in promotions
                ])
                # Black promotion (rank 1 -> 0)
                moves.extend([
                    chess.Move(
                        chess.square(f, 1),
                        chess.square(t, 0),
                        promotion=chess.Piece.from_symbol(p).piece_type,
                    ).uci()
                    for p in promotions
                ])
    
    return {m: i for i, m in enumerate(sorted(set(moves)))}


@lru_cache(maxsize=1)
def get_inverse_vocab() -> Dict[int, str]:
    """Index -> UCI mapping (cached)."""
    return {idx: uci for uci, idx in create_vocab().items()}


# --- MOVE UTILITIES ---
def mirror_move(move: chess.Move) -> chess.Move:
    """Vertically mirror a move (black's perspective to white's)."""
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion
    )


def mirror_move_uci(move: chess.Move) -> str:
    """Return the UCI string of the mirrored move."""
    return mirror_move(move).uci()


# --- BOARD TO TENSOR ---
_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def board_to_tensor_14ch(board: chess.Board) -> npt.NDArray[np.float32]:
    """
    14-channel Legacy bitboard. Shape: (14, 8, 8).

    Ch 0-5:  White P, N, B, R, Q, K
    Ch 6-11: Black P, N, B, R, Q, K
    Ch 12:   Repetition flag
    Ch 13:   Normalized move count
    """
    tensor = np.zeros((LEGACY_CHANNELS, 8, 8), dtype=np.float32)
    
    for i, piece_type in enumerate(_PIECES):
        for sq in board.pieces(piece_type, chess.WHITE):
            r, c = divmod(sq, 8)
            tensor[i, 7 - r, c] = 1.0
        for sq in board.pieces(piece_type, chess.BLACK):
            r, c = divmod(sq, 8)
            tensor[i + 6, 7 - r, c] = 1.0
    
    if board.is_repetition(2):
        tensor[12, :, :] = 1.0
    tensor[13, :, :] = min(board.fullmove_number, 100) / 100.0
    
    return tensor


def board_to_tensor_19ch(board: chess.Board) -> npt.NDArray[np.float32]:
    """
    19-channel Maia-2 bitboard. Shape: (19, 8, 8).

    Ch 0-5:   White P, N, B, R, Q, K
    Ch 6-11:  Black P, N, B, R, Q, K
    Ch 12:    Repetition flag
    Ch 13-16: Castling rights (WK, WQ, BK, BQ)
    Ch 17:    En passant square
    Ch 18:    Normalized move count
    """
    tensor = np.zeros((MAIA2_CHANNELS, 8, 8), dtype=np.float32)
    
    for i, piece_type in enumerate(_PIECES):
        for sq in board.pieces(piece_type, chess.WHITE):
            r, c = divmod(sq, 8)
            tensor[i, 7 - r, c] = 1.0
        for sq in board.pieces(piece_type, chess.BLACK):
            r, c = divmod(sq, 8)
            tensor[i + 6, 7 - r, c] = 1.0
    
    if board.is_repetition(2):
        tensor[12, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0
    if board.ep_square:
        r, c = divmod(board.ep_square, 8)
        tensor[17, 7 - r, c] = 1.0
    tensor[18, :, :] = min(board.fullmove_number, 200) / 200.0
    
    return tensor
