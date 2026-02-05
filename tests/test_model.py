"""
Unit tests for model architectures and utilities.
"""
from __future__ import annotations

import pytest
import torch
import chess

from src import Maia1_Legacy, Maia2_New, VOCAB_SIZE, create_vocab, mirror_move, mirror_move_uci
from src.config import LEGACY_CHANNELS, MAIA2_CHANNELS, ELO_MIN, ELO_MAX, elo_to_index
from src.utils import board_to_tensor_14ch, board_to_tensor_19ch


class TestModels:
    """Test model architectures."""
    
    def test_legacy_model_forward(self):
        """Test Legacy model forward pass."""
        model = Maia1_Legacy(vocab_size=VOCAB_SIZE)
        batch_size = 4
        x = torch.randn(batch_size, LEGACY_CHANNELS, 8, 8)

        output = model(x)

        assert output.shape == (batch_size, VOCAB_SIZE)
        assert not torch.isnan(output).any()

    def test_maia2_model_forward(self):
        """Test Maia-2 model forward pass with real ELO values."""
        model = Maia2_New(vocab_size=VOCAB_SIZE)
        batch_size = 4
        x = torch.randn(batch_size, MAIA2_CHANNELS, 8, 8)
        # Use real ELO values (400-3200 range)
        my_elo = torch.randint(ELO_MIN, ELO_MAX, (batch_size,))
        opp_elo = torch.randint(ELO_MIN, ELO_MAX, (batch_size,))

        output = model(x, my_elo, opp_elo)

        assert output.shape == (batch_size, VOCAB_SIZE)
        assert not torch.isnan(output).any()

    def test_maia2_elo_edge_cases(self):
        """Test Maia-2 with edge case ELO values."""
        model = Maia2_New(vocab_size=VOCAB_SIZE)
        x = torch.randn(2, MAIA2_CHANNELS, 8, 8)
        
        # Test with min and max ELO
        my_elo = torch.tensor([ELO_MIN, ELO_MAX - 1])
        opp_elo = torch.tensor([ELO_MAX - 1, ELO_MIN])
        
        output = model(x, my_elo, opp_elo)
        assert output.shape == (2, VOCAB_SIZE)
        assert not torch.isnan(output).any()

    def test_model_parameters(self):
        """Test that models have trainable parameters."""
        legacy_model = Maia1_Legacy(vocab_size=VOCAB_SIZE)
        maia2_model = Maia2_New(vocab_size=VOCAB_SIZE)

        legacy_params = sum(p.numel() for p in legacy_model.parameters())
        maia2_params = sum(p.numel() for p in maia2_model.parameters())

        assert legacy_params > 0
        assert maia2_params > 0
        # Maia-2 should have more parameters due to ELO embeddings
        assert maia2_params > legacy_params

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = Maia2_New(vocab_size=VOCAB_SIZE)
        x = torch.randn(2, MAIA2_CHANNELS, 8, 8)
        my_elo = torch.randint(ELO_MIN, ELO_MAX, (2,))
        opp_elo = torch.randint(ELO_MIN, ELO_MAX, (2,))
        target = torch.randint(0, VOCAB_SIZE, (2,))

        output = model(x, my_elo, opp_elo)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check that at least some gradients are non-zero
        has_grad = any(
            param.grad is not None and torch.abs(param.grad).sum() > 0
            for param in model.parameters()
        )
        assert has_grad, "No gradients computed"


class TestUtils:
    """Test utility functions."""
    
    def test_vocab_creation(self):
        """Test vocabulary is created correctly."""
        vocab = create_vocab()
        assert len(vocab) > 4000  # Should have ~4208 moves
        assert "e2e4" in vocab
        assert "e7e5" in vocab
        
    def test_vocab_caching(self):
        """Test vocabulary is cached."""
        vocab1 = create_vocab()
        vocab2 = create_vocab()
        assert vocab1 is vocab2  # Same object (cached)
        
    def test_mirror_move(self):
        """Test move mirroring."""
        move = chess.Move.from_uci("e2e4")
        mirrored = mirror_move(move)
        assert mirrored.uci() == "e7e5"
        
        # Test promotion
        promo_move = chess.Move.from_uci("e7e8q")
        mirrored_promo = mirror_move(promo_move)
        assert mirrored_promo.uci() == "e2e1q"
        
    def test_mirror_move_uci(self):
        """Test UCI string mirroring."""
        move = chess.Move.from_uci("a2a4")
        assert mirror_move_uci(move) == "a7a5"
        
    def test_elo_to_index(self):
        """Test ELO to index conversion."""
        assert elo_to_index(ELO_MIN) == 0
        assert elo_to_index(ELO_MAX - 1) == ELO_MAX - ELO_MIN - 1
        # Test clamping
        assert elo_to_index(0) == 0  # Below min
        assert elo_to_index(5000) == ELO_MAX - ELO_MIN - 1  # Above max


class TestTensorConversion:
    """Test board to tensor conversion."""
    
    def test_14ch_shape(self):
        """Test 14-channel tensor shape."""
        board = chess.Board()
        tensor = board_to_tensor_14ch(board)
        assert tensor.shape == (14, 8, 8)
        assert tensor.dtype.name == "float32"
        
    def test_19ch_shape(self):
        """Test 19-channel tensor shape."""
        board = chess.Board()
        tensor = board_to_tensor_19ch(board)
        assert tensor.shape == (19, 8, 8)
        assert tensor.dtype.name == "float32"
        
    def test_piece_encoding(self):
        """Test that pieces are correctly encoded."""
        board = chess.Board()
        tensor = board_to_tensor_14ch(board)
        
        # White pawns should be on rank 2 (index 6 in tensor due to flip)
        # Channel 0 is white pawns
        assert tensor[0, 6, :].sum() == 8  # 8 white pawns
        
    def test_castling_rights(self):
        """Test castling rights encoding in 19ch."""
        board = chess.Board()
        tensor = board_to_tensor_19ch(board)
        
        # All castling rights should be 1 at start
        assert tensor[13, 0, 0] == 1.0  # White kingside
        assert tensor[14, 0, 0] == 1.0  # White queenside
        assert tensor[15, 0, 0] == 1.0  # Black kingside
        assert tensor[16, 0, 0] == 1.0  # Black queenside
