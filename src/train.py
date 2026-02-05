"""
Training module for Maia chess models.
Supports both Legacy (14ch) and Maia-2 (19ch + ELO) architectures.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# Models and Datasets
from src import Maia1_Legacy, Maia2_New, create_vocab, VOCAB_SIZE
from src.dataset import LegacyDataset, Maia2Dataset
from src.config import (
    get_config_by_name, 
    TrainConfig, 
    DEFAULT_DATA_DIR, 
    DEFAULT_SAVE_DIR,
    DEFAULT_MIN_PLY,
    DEFAULT_SHUFFLE_BUFFER,
    setup_logging,
)

logger = logging.getLogger(__name__)

@dataclass
class TrainArgs:
    """Training arguments - for programmatic API."""
    config: str
    arch: str = "legacy"
    data_dir: str = DEFAULT_DATA_DIR
    save_dir: str = DEFAULT_SAVE_DIR
    base_model: Optional[str] = None
    sandbox_dir: Optional[str] = None
    num_workers: int = 4
    force: bool = False


def _save_checkpoint(
    model: torch.nn.Module, 
    path: str, 
    step: int, 
    arch: str, 
    vocab_size: int
) -> None:
    """
    Consistent checkpoint format: Both intermediate and final saves use the same structure.
    This ensures evaluate.py correctly loads both formats.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "arch": arch,
        "vocab_size": vocab_size,
    }
    torch.save(checkpoint, path)
    logger.debug(f"Checkpoint saved: {path}")


def prepare_file(filename: str, source_dir: str, sandbox_dir: Optional[str] = None) -> str:
    """Prepare PGN file, optionally copying to sandbox directory."""
    src_path = os.path.join(source_dir, filename)
    
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"PGN file not found: {src_path}")
        
    if sandbox_dir:
        os.makedirs(sandbox_dir, exist_ok=True)
        dst_path = os.path.join(sandbox_dir, filename)
        if not os.path.exists(dst_path):
            logger.info(f"Transferring to sandbox: {filename}")
            shutil.copy2(src_path, dst_path)
        return dst_path
    
    return src_path

def train_runner(cfg: TrainConfig, args: TrainArgs) -> None:
    """Run training for a single configuration."""
    logger.info(f"Job: {cfg.name} | Arch: {args.arch} | ELO: {cfg.elo_min}-{cfg.elo_max}")
    
    # 1. Prepare Data Path
    pgn_path = prepare_file(cfg.pgn_file, args.data_dir, args.sandbox_dir)
    save_path = os.path.join(args.save_dir, f"{cfg.name}_{args.arch}.pth")
    
    if os.path.exists(save_path) and not args.force:
        logger.warning(f"Skipping: {save_path} already exists. Use --force to overwrite.")
        return

    # 2. Setup Device & Model Architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = create_vocab()
    
    # Architecture selection now comes from arguments (not Config)
    if args.arch == "maia2":
        model = Maia2_New(vocab_size=VOCAB_SIZE).to(device)
        DatasetClass = Maia2Dataset
        logger.info("Architecture Selected: Maia-2 (19ch + ELO Metadata)")
    else:
        model = Maia1_Legacy(vocab_size=VOCAB_SIZE).to(device)
        DatasetClass = LegacyDataset
        logger.info("Architecture Selected: Legacy (14ch Transfer)")

    # 3. Load Checkpoint (Transfer Learning)
    if args.base_model and os.path.exists(args.base_model):
        logger.info(f"Loading Base Model: {args.base_model}")
        try:
            # weights_only=True for security (prevents arbitrary code execution)
            cp = torch.load(args.base_model, map_location=device, weights_only=True)
        except Exception:
            # Fallback for older checkpoints that may have non-tensor data
            logger.warning("weights_only load failed, falling back to full load")
            cp = torch.load(args.base_model, map_location=device)
            
        if isinstance(cp, dict) and "model_state_dict" in cp:
            sd = cp["model_state_dict"]
        elif isinstance(cp, dict) and "model" in cp:
            sd = cp["model"]
        elif hasattr(cp, "state_dict"):
            sd = cp.state_dict()
        else:
            sd = cp
        sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
        
        try:
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            logger.warning(f"Warning during load: {e}")

    # 4. Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = GradScaler("cuda")
    
    # Dataset Init
    ds = DatasetClass(pgn_path, vocab, {
        "min_ply": DEFAULT_MIN_PLY, 
        "shuffle_buffer_size": DEFAULT_SHUFFLE_BUFFER
    })
    dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=args.num_workers, pin_memory=True)

    # 5. Training Loop
    model.train()
    step = 0
    running_loss = 0.0
    start_t = time.time()
    
    logger.info(f"Training started... Target: {cfg.total_steps} steps")
    
    try:
        iter_dl = iter(dl)
        while step < cfg.total_steps:
            try:
                batch = next(iter_dl)
            except StopIteration:
                iter_dl = iter(dl)
                batch = next(iter_dl)
            
            step += 1
            boards = batch["board"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type="cuda"):
                # Forward pass varies by architecture
                if args.arch == "maia2":
                    my_elo = batch["my_elo"].to(device, non_blocking=True)
                    opp_elo = batch["opp_elo"].to(device, non_blocking=True)
                    logits = model(boards, my_elo, opp_elo)
                else:
                    logits = model(boards)
                
                loss = F.cross_entropy(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()

            # Logging
            if step % cfg.log_every == 0:
                avg_loss = running_loss / cfg.log_every
                elapsed = time.time() - start_t
                speed = (cfg.log_every * cfg.batch_size) / elapsed if elapsed > 0 else 0
                logger.info(f"[Step {step}] Loss: {avg_loss:.4f} | Speed: {speed:.0f} pos/s")
                running_loss = 0.0
                start_t = time.time()

            # Intermediate Checkpoint
            if step % cfg.save_every == 0:
                ckpt_path = os.path.join(args.save_dir, f"ckpt_{cfg.name}_{args.arch}_step_{step}.pth")
                _save_checkpoint(model, ckpt_path, step, args.arch, VOCAB_SIZE)

        # Final Save
        _save_checkpoint(model, save_path, step, args.arch, VOCAB_SIZE)
        logger.info(f"Done! Saved: {save_path} (step={step}, trained weights)")

    except KeyboardInterrupt:
        logger.warning("Interrupted. Saving emergency checkpoint...")
        _save_checkpoint(
            model,
            os.path.join(args.save_dir, f"{cfg.name}_{args.arch}_INTERRUPTED.pth"),
            step,
            args.arch,
            VOCAB_SIZE,
        )


def train_with_args(
    config: str,
    arch: str = "legacy",
    data_dir: str = DEFAULT_DATA_DIR,
    save_dir: str = DEFAULT_SAVE_DIR,
    base_model: Optional[str] = None,
    sandbox_dir: Optional[str] = None,
    num_workers: int = 4,
    force: bool = False,
) -> None:
    """
    Programmatic training API - no dependency on sys.argv.
    
    Args:
        config: Scenario index (1-12) or name (maia_01)
        arch: Model architecture - "legacy" or "maia2"
        data_dir: Directory containing PGN files
        save_dir: Model save directory
        base_model: Base model path for transfer learning
        sandbox_dir: Optional fast temp directory
        num_workers: DataLoader worker count
        force: Overwrite existing model
    """
    cfg = get_config_by_name(config)
    if not cfg:
        raise ValueError(f"Config '{config}' not found.")
    
    os.makedirs(save_dir, exist_ok=True)
    
    args = TrainArgs(
        config=config,
        arch=arch,
        data_dir=data_dir,
        save_dir=save_dir,
        base_model=base_model,
        sandbox_dir=sandbox_dir,
        num_workers=num_workers,
        force=force,
    )
    
    train_runner(cfg, args)

def main() -> None:
    """CLI entry point."""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Train Maia chess models")
    parser.add_argument("--config", type=str, required=True, help="Index (1-12) or Name (maia_01)")
    parser.add_argument("--arch", type=str, default="legacy", choices=["legacy", "maia2"], help="Model Architecture")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Path to 'data/processed'")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--sandbox_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    
    args = parser.parse_args()
    
    try:
        train_with_args(
            config=args.config,
            arch=args.arch,
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            base_model=args.base_model,
            sandbox_dir=args.sandbox_dir,
            num_workers=args.num_workers,
            force=args.force,
        )
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()