"""
Helper functions for Chess AI: move vocabulary, board-to-tensor conversion.

=== HOW DO WE SEE THE BOARD? (Input Representation) ===

The model doesn't read chess rules as text. It sees the board like "an image made of pixels".
This "Mapping" operation is performed by board_to_tensor functions.

Technique: Channel-based Bitboard / One-Hot Encoding
- The board is split into 14 or 19 "transparent acetate layers".
- Each layer marks only the position of one piece type (1=present, 0=absent).
- This allows the model to learn geometric relationships of pieces (diagonal neighbors, pawn chains)
  through mathematical matrices. This relationship would be lost in a flat list.
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
    """
    UCI move vocabulary (~4208 different move types).
    Model doesn't output coordinates; it selects a class from this vocabulary (Idx 0: a2a3, Idx 1: a2a4...).
    
    Result is cached - not recalculated on subsequent calls.
    """
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
    """
    Convert move from black's perspective to white's perspective (for board mirroring).
    
    Args:
        move: Original move
        
    Returns:
        Mirrored move
    """
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
    Each channel = an "acetate layer": square is 1 if piece is present, 0 otherwise.

    Channels 0-5:   White Pawn, Knight, Bishop, Rook, Queen, King
    Channels 6-11:  Black Pawn, Knight, Bishop, Rook, Queen, King
    Channel 12:     Repetition state (draw check)
    Channel 13:     Normalized move count (fullmove_number / 100)
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
    Same as 14ch channels 0-12 + additional channels:

    Channels 0-5:   White pieces | Channels 6-11: Black pieces
    Channel 12:     Repetition state
    Channels 13-16: Castling rights (WK, WQ, BK, BQ)
    Channel 17:     En passant square (1 if exists)
    Channel 18:     Normalized move count (opening vs endgame context)
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
    # Channel 18: Normalized Move Count (Opening vs Endgame context)
    tensor[18, :, :] = min(board.fullmove_number, 200) / 200.0
    
    return tensor
