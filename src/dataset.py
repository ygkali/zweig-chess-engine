"""
Chess dataset implementations for training and evaluation.
Supports PGN and ZST compressed files with worker sharding.
"""
from __future__ import annotations

import io
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional, TextIO, BinaryIO

import chess
import chess.pgn
import torch
from torch.utils.data import IterableDataset

from src.utils import (
    create_vocab, 
    board_to_tensor_14ch, 
    board_to_tensor_19ch,
    mirror_move_uci,
)
from src.config import (
    MAIA2_CHANNELS, 
    DEFAULT_MIN_PLY, 
    DEFAULT_SHUFFLE_BUFFER,
)

logger = logging.getLogger(__name__)

# Optional Zstandard import
try:
    import zstandard as zstd
except ImportError:
    zstd = None

class BaseChessDataset(IterableDataset, ABC):
    """
    Abstract Base Class for PGN Datasets.
    Handles file reading, worker sharding, and shuffling buffer.
    Specific tensor conversion logic is delegated to subclasses.
    """

    def __init__(self, file_path: str, vocab: Dict[str, int], config: Dict[str, Any]) -> None:
        self.file_path = file_path
        self.vocab = vocab
        self.min_ply = config.get("min_ply", DEFAULT_MIN_PLY)
        self.buffer_size = config.get("shuffle_buffer_size", DEFAULT_SHUFFLE_BUFFER)
        self.debug_mode = config.get("debug", False)
        
        # Resource tracking
        self._raw_file: Optional[BinaryIO] = None
        self._parse_errors: int = 0

    def _get_file_handle(self) -> TextIO:
        """Handles plain text or zstd compressed files with proper resource management."""
        if self.file_path.endswith(".zst"):
            if zstd is None:
                raise ImportError("zstandard module missing. pip install zstandard")
            self._raw_file = open(self.file_path, "rb")
            dctx = zstd.ZstdDecompressor()
            return io.TextIOWrapper(dctx.stream_reader(self._raw_file), encoding="utf-8")
        return open(self.file_path, "r", encoding="utf-8")
    
    def _close_resources(self) -> None:
        """Close any open file handles."""
        if self._raw_file is not None:
            try:
                self._raw_file.close()
            except Exception:
                pass
            self._raw_file = None

    @abstractmethod
    def process_game(self, game: chess.pgn.Game) -> Iterator[Dict[str, Any]]:
        """Subclasses must implement this to yield tensors from a game."""
        pass

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        buffer: list[Dict[str, Any]] = []
        self._parse_errors = 0
        
        try:
            with self._get_file_handle() as f:
                game_idx = 0
                while True:
                    try:
                        game = chess.pgn.read_game(f)
                    except ValueError as e:
                        self._parse_errors += 1
                        if self.debug_mode:
                            logger.warning(f"PGN Parse Error #{self._parse_errors}: {e}")
                        continue
                    
                    if game is None:
                        break

                    # Worker Sharding Logic
                    if worker_info is not None:
                        if game_idx % worker_info.num_workers != worker_info.id:
                            game_idx += 1
                            continue
                    game_idx += 1

                    # Delegate specific processing to child class
                    for item in self.process_game(game):
                        if len(buffer) < self.buffer_size:
                            buffer.append(item)
                        else:
                            idx = random.randint(0, self.buffer_size - 1)
                            yield buffer[idx]
                            buffer[idx] = item

            # Flush remaining buffer
            random.shuffle(buffer)
            for item in buffer:
                yield item
                
            if self._parse_errors > 0:
                logger.info(f"Dataset iteration complete. Parse errors: {self._parse_errors}")
        finally:
            self._close_resources()


class LegacyDataset(BaseChessDataset):
    """Dataset for Maia-1 (Legacy) 14-channel architecture."""
    
    def process_game(self, game: chess.pgn.Game) -> Iterator[Dict[str, Any]]:
        board = game.board()
        ply_count = 0
        
        for move in game.mainline_moves():
            ply_count += 1
            uci = move.uci()
            
            # Filters
            if ply_count <= self.min_ply:
                board.push(move)
                continue
            if uci not in self.vocab:
                board.push(move)
                continue

            # Data Generation (White/Black Mirroring)
            is_black = (board.turn == chess.BLACK)
            
            if is_black:
                m_board = board.mirror()
                m_move_uci = mirror_move_uci(move)
                
                if m_move_uci in self.vocab:
                    yield {
                        "board": board_to_tensor_14ch(m_board),
                        "target": self.vocab[m_move_uci]
                    }
            else:
                yield {
                    "board": board_to_tensor_14ch(board),
                    "target": self.vocab[uci]
                }
            
            board.push(move)


class Maia2Dataset(BaseChessDataset):
    """Dataset for Maia-2 (New) 19-channel architecture with ELO metadata."""

    def process_game(self, game: chess.pgn.Game) -> Iterator[Dict[str, Any]]:
        try:
            w_elo = int(game.headers.get("WhiteElo", 1500))
            b_elo = int(game.headers.get("BlackElo", 1500))
        except (ValueError, TypeError):
            w_elo, b_elo = 1500, 1500

        board = game.board()
        ply_count = 0

        for move in game.mainline_moves():
            ply_count += 1
            uci = move.uci()

            if ply_count <= self.min_ply or uci not in self.vocab:
                board.push(move)
                continue

            current_elo = w_elo if board.turn == chess.WHITE else b_elo
            opp_elo = b_elo if board.turn == chess.WHITE else w_elo
            
            is_black = (board.turn == chess.BLACK)

            if is_black:
                m_board = board.mirror()
                m_move_uci = mirror_move_uci(move)

                if m_move_uci in self.vocab:
                    tensor = board_to_tensor_19ch(m_board)
                    assert tensor.shape == (MAIA2_CHANNELS, 8, 8), f"Shape Mismatch: {tensor.shape}"
                         
                    yield {
                        "board": tensor,
                        "target": self.vocab[m_move_uci],
                        "my_elo": current_elo,
                        "opp_elo": opp_elo
                    }
            else:
                tensor = board_to_tensor_19ch(board)
                assert tensor.shape == (MAIA2_CHANNELS, 8, 8), f"Shape Mismatch: {tensor.shape}"

                yield {
                    "board": tensor,
                    "target": self.vocab[uci],
                    "my_elo": current_elo,
                    "opp_elo": opp_elo
                }

            board.push(move)


class HybridTestDataset(IterableDataset):
    """
    Evaluation dataset: ZST or PGN, ELO filtering, 14ch or 19ch.
    Uses the SAME board_to_tensor functions (consistency with training).
    """

    def __init__(
        self, 
        path: str, 
        max_samples: int, 
        min_elo: int, 
        max_elo: int, 
        is_legacy: bool, 
        vocab: Optional[Dict[str, int]] = None
    ) -> None:
        self.path = path
        self.max_samples = max_samples
        self.min_elo = min_elo
        self.max_elo = max_elo
        self.is_legacy = is_legacy
        self.vocab = vocab if vocab is not None else create_vocab()
        self._raw_file: Optional[BinaryIO] = None

    def _open_stream(self) -> TextIO:
        """Open file with proper resource management."""
        if self.path.lower().endswith(".zst"):
            if zstd is None:
                raise ImportError("zstandard required for .zst: pip install zstandard")
            self._raw_file = open(self.path, "rb")
            dctx = zstd.ZstdDecompressor()
            return io.TextIOWrapper(dctx.stream_reader(self._raw_file), encoding="utf-8")
        return open(self.path, "r", encoding="utf-8")
    
    def _close_resources(self) -> None:
        """Close any open file handles."""
        if self._raw_file is not None:
            try:
                self._raw_file.close()
            except Exception:
                pass
            self._raw_file = None

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        tensor_func = board_to_tensor_14ch if self.is_legacy else board_to_tensor_19ch
        vocab = self.vocab
        sample_count = 0

        try:
            with self._open_stream() as text:
                while sample_count < self.max_samples:
                    try:
                        game = chess.pgn.read_game(text)
                    except Exception:
                        continue
                    if game is None:
                        break
                    try:
                        w_elo = int(game.headers.get("WhiteElo", 0))
                        b_elo = int(game.headers.get("BlackElo", 0))
                    except (ValueError, TypeError):
                        continue

                    board = game.board()
                    for move in game.mainline_moves():
                        uci = move.uci()
                        if uci not in vocab:
                            board.push(move)
                            continue

                        current_elo = w_elo if board.turn == chess.WHITE else b_elo
                        opp_elo = b_elo if board.turn == chess.WHITE else w_elo
                        if not (self.min_elo <= current_elo <= self.max_elo):
                            board.push(move)
                            continue

                        is_black = board.turn == chess.BLACK
                        if is_black:
                            m_board = board.mirror()
                            m_move_uci = mirror_move_uci(move)
                            if m_move_uci in vocab:
                                yield {
                                    "board": tensor_func(m_board),
                                    "target": vocab[m_move_uci],
                                    "my_elo": current_elo,
                                    "opp_elo": opp_elo,
                                }
                                sample_count += 1
                        else:
                            yield {
                                "board": tensor_func(board),
                                "target": vocab[uci],
                                "my_elo": current_elo,
                                "opp_elo": opp_elo,
                            }
                            sample_count += 1

                        if sample_count >= self.max_samples:
                            return
                        board.push(move)
        finally:
            self._close_resources()