#!/usr/bin/env python3
"""
Inference: Run move predictions with Zweig chess models.

Usage:
  python scripts/inference.py --model maia_05 --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
  python scripts/inference.py --model maia_11 --interactive
  python scripts/inference.py --download maia_05  # Download model from HuggingFace
  python scripts/inference.py --list  # List available models
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import chess

from src import Maia1_Legacy, Maia2_New, create_vocab, VOCAB_SIZE
from src.utils import board_to_tensor_14ch, board_to_tensor_19ch, mirror_move
from src.config import (
    DEFAULT_SAVE_DIR, 
    DEFAULT_CHECKPOINT_DIR,
    setup_logging,
    HF_MODEL_REPO,
    HF_MODEL_URL,
    HF_MODEL_FILES,
)

logger = logging.getLogger(__name__)


def download_model_from_hf(model_name: str, output_dir: str = None) -> Optional[str]:
    """
    Download a model from HuggingFace Hub.
    
    Args:
        model_name: Model name (e.g., 'maia_05') or filename
        output_dir: Target directory. Default: checkpoints/
        
    Returns:
        Local path to downloaded model, or None if failed
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return None
    
    if output_dir is None:
        output_dir = DEFAULT_CHECKPOINT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Resolve model name to filename
    if model_name in HF_MODEL_FILES:
        filename = HF_MODEL_FILES[model_name]
    elif model_name.endswith(".pth"):
        filename = model_name
    else:
        # Try to find matching file
        try:
            files = list_repo_files(HF_MODEL_REPO)
            matches = [f for f in files if model_name in f and f.endswith(".pth")]
            if matches:
                filename = matches[0]
            else:
                logger.error(f"Model not found: {model_name}")
                return None
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return None
    
    logger.info(f"Downloading from HuggingFace: {filename}")
    
    try:
        local_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=filename,
            local_dir=output_dir,
        )
        logger.info(f"Downloaded: {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def list_available_models() -> None:
    """List available models on HuggingFace."""
    print(f"\n=== Available Zweig Models ===\n")
    print(f"HuggingFace: {HF_MODEL_URL}\n")
    
    print("Pre-configured models:")
    for name, filename in HF_MODEL_FILES.items():
        print(f"  {name:12} -> {filename}")
    
    print(f"\nDownload: python scripts/inference.py --download <model_name>")
    print(f"Example:  python scripts/inference.py --download maia_05\n")


def resolve_model_path(model_arg: str) -> Optional[str]:
    """
    Resolve model argument to a local file path.
    Downloads from HuggingFace if needed.
    """
    # Check if it's a local path
    if os.path.exists(model_arg):
        return model_arg
    
    # Check in checkpoints directory
    checkpoint_path = os.path.join(DEFAULT_CHECKPOINT_DIR, model_arg)
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    
    # Check if it's a model name with .pth
    if not model_arg.endswith(".pth"):
        if model_arg in HF_MODEL_FILES:
            filename = HF_MODEL_FILES[model_arg]
            checkpoint_path = os.path.join(DEFAULT_CHECKPOINT_DIR, filename)
            if os.path.exists(checkpoint_path):
                return checkpoint_path
    
    # Check in finetuned_models directory
    finetuned_path = os.path.join(DEFAULT_SAVE_DIR, model_arg)
    if os.path.exists(finetuned_path):
        return finetuned_path
    
    # Try to download from HuggingFace
    logger.info(f"Model not found locally, attempting HuggingFace download...")
    return download_model_from_hf(model_arg)


def load_model(
    path: str, 
    device: torch.device, 
    is_legacy: bool = True
) -> Tuple[Optional[torch.nn.Module], Optional[Dict[str, int]]]:
    """Load model from checkpoint."""
    vocab = create_vocab()
    model = Maia1_Legacy(vocab_size=VOCAB_SIZE).to(device) if is_legacy else Maia2_New(vocab_size=VOCAB_SIZE).to(device)
    
    try:
        # Try secure load first
        try:
            cp = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            logger.warning("weights_only load failed, falling back to full load")
            cp = torch.load(path, map_location=device)
            
        if isinstance(cp, dict) and "model_state_dict" in cp:
            sd = cp["model_state_dict"]
        elif isinstance(cp, dict) and "model" in cp:
            sd = cp["model"]
        else:
            sd = cp
        sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model, vocab
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        return None, None


def get_top_moves(
    model: torch.nn.Module, 
    board: chess.Board, 
    vocab: Dict[str, int], 
    device: torch.device, 
    top_k: int = 5, 
    elo: int = 1500, 
    is_legacy: bool = True
) -> List[Tuple[str, float]]:
    """
    Return top moves for a given position.
    
    Args:
        model: Loaded model
        board: python-chess Board (from perspective of side to move)
        vocab: UCI -> index mapping
        device: Torch device
        top_k: Number of moves to return
        elo: Player ELO (for Maia2)
        is_legacy: Is this legacy architecture?
        
    Returns:
        List of (uci_move, probability) tuples
    """
    inv_vocab = {idx: uci for uci, idx in vocab.items()}
    tensor_func = board_to_tensor_14ch if is_legacy else board_to_tensor_19ch

    # Mirror if black's turn (model always learns from "our" perspective)
    if board.turn == chess.BLACK:
        board_t = tensor_func(board.mirror())
    else:
        board_t = tensor_func(board)

    # [1, C, 8, 8]
    x = torch.from_numpy(board_t).unsqueeze(0).float().to(device)

    with torch.no_grad():
        if is_legacy:
            logits = model(x, my_elo=None, opp_elo=None)
        else:
            # ELO value is given directly, converted to index inside the model
            elo_t = torch.tensor([elo], dtype=torch.long, device=device)
            logits = model(x, elo_t, elo_t)

    probs = torch.softmax(logits[0], dim=0)
    top_probs, top_indices = torch.topk(probs, min(top_k, len(vocab)))

    moves: List[Tuple[str, float]] = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        uci = inv_vocab.get(idx, "????")
        # Mirror the move if it's black's turn
        if board.turn == chess.BLACK and uci != "????":
            try:
                m = chess.Move.from_uci(uci)
                uci = mirror_move(m).uci()
            except Exception:
                pass
        moves.append((uci, prob))
    return moves


def main() -> None:
    """CLI entry point."""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Zweig Chess Engine - Move Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python scripts/inference.py --model maia_05 --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
  python scripts/inference.py --model maia_11 --interactive
  python scripts/inference.py --download maia_05
  python scripts/inference.py --list

Models: {HF_MODEL_URL}
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g., maia_05) or path to .pth file",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="legacy",
        choices=["legacy", "maia2"],
        help="Architecture: legacy (14ch) or maia2 (19ch+ELO)",
    )
    parser.add_argument(
        "--fen",
        type=str,
        default=None,
        help="Position in FEN format",
    )
    parser.add_argument(
        "--elo",
        type=int,
        default=1500,
        help="Player ELO for Maia2 (400-3200)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top moves to show",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: input FEN, get predictions",
    )
    parser.add_argument(
        "--download",
        type=str,
        default=None,
        metavar="MODEL",
        help="Download model from HuggingFace (e.g., maia_05)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_available_models()
        sys.exit(0)

    # Handle --download
    if args.download:
        path = download_model_from_hf(args.download)
        if path:
            print(f"\n✅ Model downloaded: {path}")
            sys.exit(0)
        else:
            sys.exit(1)

    # Require --model for inference
    if not args.model:
        parser.error("--model is required for inference. Use --list to see available models.")

    # Resolve model path (download if needed)
    path = resolve_model_path(args.model)
    if not path:
        logger.error(f"Model not found: {args.model}")
        logger.info(f"Use --list to see available models or --download to fetch from HuggingFace")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_legacy = args.arch == "legacy"

    model, vocab = load_model(path, device, is_legacy)
    if model is None:
        sys.exit(1)

    print(f"\n=== Zweig Chess Engine ===\n")
    print(f"  Model: {os.path.basename(path)}")
    print(f"  Architecture: {'14ch Legacy' if is_legacy else '19ch Maia-2'}")
    if not is_legacy:
        print(f"  ELO: {args.elo}")
    print()

    if args.interactive:
        print("Interactive mode. Enter FEN (empty = starting position), 'q' = quit\n")
        while True:
            try:
                fen = input("FEN> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if fen.lower() == "q":
                break
            if not fen:
                fen = chess.STARTING_FEN
            try:
                board = chess.Board(fen)
            except Exception:
                print("  Invalid FEN")
                continue
            moves = get_top_moves(model, board, vocab, device, args.top, args.elo, is_legacy)
            print(f"  Turn: {'Black' if board.turn == chess.BLACK else 'White'}")
            for i, (uci, prob) in enumerate(moves, 1):
                print(f"  {i}. {uci} ({prob*100:.1f}%)")
            print()

    elif args.fen:
        try:
            board = chess.Board(args.fen)
        except Exception:
            logger.error("Invalid FEN")
            sys.exit(1)
        moves = get_top_moves(model, board, vocab, device, args.top, args.elo, is_legacy)
        print(f"Position: {args.fen[:60]}...")
        print(f"Turn: {'Black' if board.turn == chess.BLACK else 'White'}\n")
        print("Predicted moves:")
        for i, (uci, prob) in enumerate(moves, 1):
            print(f"  {i}. {uci} ({prob*100:.1f}%)")

    else:
        logger.error("--fen or --interactive required")
        sys.exit(1)


if __name__ == "__main__":
    main()
