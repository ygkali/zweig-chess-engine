#!/usr/bin/env python3
"""
Data Pipeline: Download dataset from HuggingFace or local sources.

Usage:
  python scripts/data_pipeline.py --source huggingface
  python scripts/data_pipeline.py --source huggingface --bucket 5
  python scripts/data_pipeline.py --source local --source_dir /path/to/pgns
  python scripts/data_pipeline.py --verify
"""
import argparse
import os
import shutil
import sys
from typing import Optional, List

from src.config import (
    PROJECT_ROOT,
    DEFAULT_DATA_DIR,
    DEFAULT_RAW_DATA_DIR,
    SCENARIOS,
    HF_DATASET_REPO,
    HF_DATASET_URL,
    setup_logging,
)

import logging
logger = logging.getLogger(__name__)


def ensure_dirs() -> None:
    """Create required directories."""
    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
    os.makedirs(DEFAULT_RAW_DATA_DIR, exist_ok=True)
    logger.info(f"Target directory: {DEFAULT_DATA_DIR}")


def download_from_huggingface(output_dir: str, buckets: Optional[List[int]] = None) -> bool:
    """
    Download dataset from HuggingFace Hub.
    
    Args:
        output_dir: Target directory for downloaded files
        buckets: List of bucket IDs (1-12) to download. None = all buckets.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    
    logger.info(f"Downloading from HuggingFace: {HF_DATASET_URL}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # List available files
        files = list_repo_files(HF_DATASET_REPO, repo_type="dataset")
        pgn_files = [f for f in files if f.endswith(".pgn")]
        
        if not pgn_files:
            logger.warning("No .pgn files found in repository")
            return False
        
        # Filter by bucket if specified
        if buckets:
            bucket_strs = [f"{b:02d}" for b in buckets]
            pgn_files = [f for f in pgn_files if any(f"train_{b}" in f or f"_{b}_" in f for b in bucket_strs)]
        
        logger.info(f"Found {len(pgn_files)} files to download")
        
        for filename in pgn_files:
            logger.info(f"  Downloading: {filename}")
            local_path = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=output_dir,
            )
            logger.info(f"  Saved: {os.path.basename(local_path)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Download error: {e}")
        return False


def copy_local(source_dir: str, dest_dir: str, pattern: str = "*.pgn") -> int:
    """Copy PGN files from local directory."""
    import glob
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    for src in glob.glob(os.path.join(source_dir, pattern)):
        if os.path.isfile(src):
            dst = os.path.join(dest_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            logger.info(f"  Copied: {os.path.basename(src)}")
            count += 1
    return count


def verify_data_dir(data_dir: str) -> bool:
    """Verify expected PGN files exist."""
    expected = [s["pgn_file"] for s in SCENARIOS]
    missing = []
    found = []
    for f in expected:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            found.append(f)
        else:
            missing.append(f)
    
    if found:
        logger.info(f"Found {len(found)} files")
    if missing:
        logger.warning(f"Missing {len(missing)} files: {missing[:3]}{'...' if len(missing) > 3 else ''}")
        return False
    return True


def list_available_buckets() -> None:
    """List available ELO buckets."""
    print("\n=== Available ELO Buckets ===\n")
    for i, s in enumerate(SCENARIOS, 1):
        print(f"  {i:2}. {s['name']:12} | ELO {s['elo_min']:4}-{s['elo_max']:4} | {s['pgn_file']}")
    print(f"\nDataset: {HF_DATASET_URL}")
    print()


def main() -> None:
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Data Pipeline: Download dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python scripts/data_pipeline.py --source huggingface
  python scripts/data_pipeline.py --source huggingface --bucket 5 8 11
  python scripts/data_pipeline.py --source local --source_dir ./my_data
  python scripts/data_pipeline.py --verify
  python scripts/data_pipeline.py --list

Dataset URL: {HF_DATASET_URL}
        """,
    )
    parser.add_argument(
        "--source",
        choices=["huggingface", "hf", "local", "copy"],
        default="huggingface",
        help="Data source: huggingface/hf=HuggingFace Hub, local/copy=local directory",
    )
    parser.add_argument(
        "--bucket",
        type=int,
        nargs="+",
        default=None,
        help="Specific bucket IDs to download (1-12). Default: all",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Target data directory",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Source directory (for --source local)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing data, no download",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available buckets and exit",
    )

    args = parser.parse_args()

    if args.list:
        list_available_buckets()
        sys.exit(0)

    print(f"\n=== Zweig Chess Engine - Data Pipeline ===\n")

    if args.verify:
        ensure_dirs()
        ok = verify_data_dir(args.data_dir)
        sys.exit(0 if ok else 1)

    ensure_dirs()

    if args.source in ["huggingface", "hf"]:
        ok = download_from_huggingface(args.data_dir, args.bucket)
        if not ok:
            sys.exit(1)

    elif args.source in ["local", "copy"]:
        src = args.source_dir
        if not src or not os.path.exists(src):
            logger.error("--source_dir required and must exist")
            sys.exit(1)
        count = copy_local(src, args.data_dir)
        logger.info(f"{count} files copied")

    ok = verify_data_dir(args.data_dir)
    print("\n  Pipeline complete." if ok else "\n  Warning: Some files may be missing.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
