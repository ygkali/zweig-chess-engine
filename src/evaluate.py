"""
Benchmarking: load checkpoint, run inference, report Top-1/Top-3 accuracy.

Use Lichess 2022 (Jan-Mar) unseen data for evaluation - do not rely on train accuracy (~54%).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from src import Maia1_Legacy, Maia2_New, create_vocab, VOCAB_SIZE
from src.dataset import HybridTestDataset

logger = logging.getLogger(__name__)


def load_model_safe(
    path: str, 
    vocab_size: int, 
    device: torch.device, 
    is_legacy: bool = True
) -> Optional[torch.nn.Module]:
    """
    Load checkpoint with flexible format handling.
    Supports: {"model_state_dict": ...}, {"model": ...}, raw state_dict.
    """
    if not os.path.exists(path):
        return None
    
    model = Maia1_Legacy(vocab_size).to(device) if is_legacy else Maia2_New(vocab_size).to(device)
    
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
        elif isinstance(cp, dict) and "state_dict" in cp:
            sd = cp["state_dict"]
        else:
            sd = cp
            
        if not isinstance(sd, dict):
            logger.warning(f"Unexpected checkpoint format at {path}")
            return None
            
        clean_sd = {
            k.replace("_orig_mod.", "").replace("module.", ""): v
            for k, v in sd.items()
        }
        missing, unexpected = model.load_state_dict(clean_sd, strict=False)
        if missing:
            logger.warning(f"Missing keys: {len(missing)}")
        if unexpected:
            logger.warning(f"Unexpected keys: {len(unexpected)}")
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Load error: {e}")
        return None


def evaluate_single_model(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    device: torch.device, 
    is_legacy: bool = True
) -> tuple[float, float, int]:
    """Run inference, return (acc_top1, acc_top3, total)."""
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            b = batch["board"].to(device)
            t = batch["target"].to(device)

            if is_legacy:
                logits = model(b, my_elo=None, opp_elo=None)
            else:
                # ELO values are given directly, converted to index inside the model
                my_elo = batch["my_elo"].to(device).long()
                opp_elo = batch["opp_elo"].to(device).long()
                logits = model(b, my_elo, opp_elo)

            preds = torch.argmax(logits, dim=1)
            correct_top1 += (preds == t).sum().item()
            _, top3 = torch.topk(logits, 3, dim=1)
            correct_top3 += (top3 == t.unsqueeze(1)).any(dim=1).sum().item()
            total += b.size(0)

    acc1 = (correct_top1 / total) * 100 if total > 0 else 0.0
    acc3 = (correct_top3 / total) * 100 if total > 0 else 0.0
    return acc1, acc3, total


def run_benchmark(
    model_configs: List[Dict[str, Any]],
    test_data_path: str,
    test_limit: int = 20000,
    batch_size: int = 4096,
    device: Optional[torch.device] = None,
    num_workers: int = 2,
) -> List[Dict[str, Any]]:
    """
    Evaluate multiple models on 2022 test data.
    
    Args:
        model_configs: list of dicts with keys: path, min_elo, max_elo, is_legacy, name
        test_data_path: Path to test PGN/ZST file
        test_limit: Maximum samples to evaluate
        batch_size: Batch size for evaluation
        device: Torch device (auto-detected if None)
        num_workers: DataLoader workers
        
    Returns:
        List of result dicts with name, acc_top1, acc_top3, total
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = create_vocab()
    results: List[Dict[str, Any]] = []

    for config in model_configs:
        name = config.get("name", config.get("path", "unknown"))
        path = config["path"]
        min_elo = config["min_elo"]
        max_elo = config["max_elo"]
        is_legacy = config.get("is_legacy", True)

        logger.info(f"Testing: {name}")
        logger.info(f"  Architecture: {'14ch Legacy' if is_legacy else '19ch Maia-2'}")
        logger.info(f"  ELO range: {min_elo}-{max_elo}")

        if not os.path.exists(path):
            logger.warning(f"Model not found: {path}")
            continue

        model = load_model_safe(path, VOCAB_SIZE, device, is_legacy)
        if model is None:
            logger.error("Load error")
            continue

        ds = HybridTestDataset(
            test_data_path, test_limit, min_elo, max_elo, is_legacy, vocab=vocab
        )
        dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

        acc1, acc3, total = evaluate_single_model(model, dl, device, is_legacy)
        logger.info(f"  Top-1: {acc1:.2f}% | Top-3: {acc3:.2f}% (n={total})")
        
        if acc1 < 2.0 and total > 100:
            logger.warning("Very low accuracy - model may be untrained or wrong checkpoint.")

        results.append({
            "name": name,
            "acc_top1": acc1,
            "acc_top3": acc3,
            "total": total,
        })

        del model
        torch.cuda.empty_cache()

    return results
