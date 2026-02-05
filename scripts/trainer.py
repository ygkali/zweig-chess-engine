#!/usr/bin/env python3
"""
Trainer: Orchestrates the training process.

First prepare data with Data Pipeline, then:
  python scripts/trainer.py --config 1 --arch legacy
  python scripts/trainer.py --config maia_02 --arch maia2 --base_model checkpoints/maia_base.pth

Windows: num_workers=0 or 2 recommended.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from src.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_SAVE_DIR,
    SCENARIOS,
    get_config_by_name,
    setup_logging,
)

logger = logging.getLogger(__name__)


def preflight_check(args: argparse.Namespace) -> bool:
    """Pre-training validation checks."""
    logger.info("=== Pre-Flight Check ===")

    # 1. Data directory
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.info("First run: python scripts/data_pipeline.py --source huggingface")
        return False
    logger.info(f"OK: Data directory exists: {args.data_dir}")

    # 2. PGN file
    cfg = get_config_by_name(args.config)
    if not cfg:
        logger.error(f"Config not found: {args.config}")
        return False
    pgn_path = os.path.join(args.data_dir, cfg.pgn_file)
    if not os.path.exists(pgn_path):
        logger.error(f"PGN file not found: {cfg.pgn_file}")
        return False
    logger.info(f"OK: PGN exists: {cfg.pgn_file}")

    # 3. Base model (optional)
    if args.base_model and not os.path.exists(args.base_model):
        logger.warning(f"Base model not found: {args.base_model}")
    elif args.base_model:
        logger.info(f"OK: Base model: {args.base_model}")

    # 4. GPU
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"OK: GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("No GPU available, CPU training will be slow")
    except ImportError:
        logger.error("torch is not installed")
        return False

    # 5. Save directory
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f"OK: Save directory: {args.save_dir}")

    logger.info("Pre-Flight OK. Starting training...")
    return True


def list_configs() -> None:
    """List available training scenarios."""
    print("\n=== Training Scenarios ===\n")
    for i, s in enumerate(SCENARIOS, 1):
        print(f"  {i:2}. {s['name']:12} | ELO {s['elo_min']}-{s['elo_max']} | {s['pgn_file']}")
    print()


def main() -> None:
    """CLI entry point."""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Trainer: Zweig chess model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/trainer.py --config 1 --arch legacy
  python scripts/trainer.py --config maia_02 --arch maia2 --base_model checkpoints/maia_base.pth
  python scripts/trainer.py --list
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Scenario index (1-12) or name (maia_01)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="legacy",
        choices=["legacy", "maia2"],
        help="Architecture: legacy (14ch) or maia2 (19ch+ELO)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing PGN files",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        help="Model save directory",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model path for transfer learning",
    )
    parser.add_argument(
        "--sandbox_dir",
        type=str,
        default=None,
        help="Optional: Fast temp directory (SSD/Ramdisk)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker count (Windows: 0 or 2)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing model",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List scenarios and exit",
    )
    parser.add_argument(
        "--skip_check",
        action="store_true",
        help="Skip pre-flight check",
    )

    args = parser.parse_args()

    if args.list:
        list_configs()
        sys.exit(0)

    if not args.config:
        logger.error("--config required. Use --list to see available scenarios.")
        sys.exit(1)

    if not args.skip_check and not preflight_check(args):
        sys.exit(1)

    # Use programmatic API - no sys.argv manipulation
    from src.train import train_with_args
    
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
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
