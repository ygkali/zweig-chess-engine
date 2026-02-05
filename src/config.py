"""
Configuration management.
Single Source of Truth for Zweig Chess Engine.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging
import os

# --- PROJECT INFO ---
PROJECT_NAME = "zweig-chess-engine"
PROJECT_VERSION = "0.2.0"

# --- HUGGING FACE ---
HF_MODEL_REPO = "ygkla/zweig-chess-engine-models"
HF_DATASET_REPO = "ygkla/zweig-chess-engine-processed"
HF_MODEL_URL = f"https://huggingface.co/{HF_MODEL_REPO}"
HF_DATASET_URL = f"https://huggingface.co/datasets/{HF_DATASET_REPO}"

# --- LOGGING ---
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the entire project."""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

# --- CONSTANTS ---
LEGACY_CHANNELS = 14
MAIA2_CHANNELS = 19

# ELO Configuration - min/max values for index calculation
ELO_MIN = 400
ELO_MAX = 3200
ELO_BUCKETS = ELO_MAX - ELO_MIN  # 2800 buckets (400-3199 range)
ELO_EMBEDDING_DIM = 128

# Dataset defaults
DEFAULT_MIN_PLY = 10
DEFAULT_SHUFFLE_BUFFER = 20000

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DEFAULT_SAVE_DIR = os.path.join(PROJECT_ROOT, "finetuned_models")
DEFAULT_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# --- HF MODEL FILES ---
# Model filenames (as stored on HuggingFace)
HF_MODEL_FILES: Dict[str, str] = {
    "maia_base": "maia_base.pth",
    "maia_01": "maia_finetuned_train_01_400-1000.pth",
    "maia_02": "maia_finetuned_train_02_1001-1200.pth",
    "maia_03": "maia_finetuned_train_03_1201-1325.pth",
    "maia_05": "maia_finetuned_train_05_1426-1500.pth",
    "maia_08": "maia_finetuned_train_08_1651-1750.pth",
    "maia_09": "maia_finetuned_train_09_1751-1875.pth",
    "maia_10": "maia_finetuned_train_10_1876-2100.pth",
    "maia_11": "maia_finetuned_train_11_2101-2400.pth",
    "maia_12": "maia_finetuned_train_12_2401-PLUS.pth",
    "maia_12_aggressive": "maia_finetuned_train_12_2401-PLUS_AGGRESSIVE.pth",
}

@dataclass
class TrainConfig:
    """Training scenario configuration."""
    name: str
    pgn_file: str
    elo_min: int
    elo_max: int
    
    # Training Hyperparams
    total_steps: int = 30000     
    batch_size: int = 8192       
    lr: float = 2e-5             
    save_every: int = 5000
    log_every: int = 100
    
    def __post_init__(self) -> None:
        """Validation."""
        if self.elo_min >= self.elo_max:
            raise ValueError(f"elo_min ({self.elo_min}) must be < elo_max ({self.elo_max})")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
    
    @property
    def elo_range(self) -> Tuple[int, int]:
        """Return ELO range as tuple."""
        return (self.elo_min, self.elo_max)

# --- SCENARIOS ---
SCENARIOS: List[Dict] = [
    {"name": "maia_01", "pgn_file": "train_01_400-1000.pgn",   "elo_min": 400,  "elo_max": 1000},
    {"name": "maia_02", "pgn_file": "train_02_1001-1200.pgn",  "elo_min": 1001, "elo_max": 1200},
    {"name": "maia_03", "pgn_file": "train_03_1201-1325.pgn",  "elo_min": 1201, "elo_max": 1325},
    {"name": "maia_04", "pgn_file": "train_04_1326-1425.pgn",  "elo_min": 1326, "elo_max": 1425},
    {"name": "maia_05", "pgn_file": "train_05_1426-1500.pgn",  "elo_min": 1426, "elo_max": 1500},
    {"name": "maia_06", "pgn_file": "train_06_1501-1575.pgn",  "elo_min": 1501, "elo_max": 1575},
    {"name": "maia_07", "pgn_file": "train_07_1576-1650.pgn",  "elo_min": 1576, "elo_max": 1650},
    {"name": "maia_08", "pgn_file": "train_08_1651-1750.pgn",  "elo_min": 1651, "elo_max": 1750},
    {"name": "maia_09", "pgn_file": "train_09_1751-1875.pgn",  "elo_min": 1751, "elo_max": 1875},
    {"name": "maia_10", "pgn_file": "train_10_1876-2100.pgn",  "elo_min": 1876, "elo_max": 2100},
    {"name": "maia_11", "pgn_file": "train_11_2101-2400.pgn",  "elo_min": 2101, "elo_max": 2400},
    {"name": "maia_12", "pgn_file": "train_12_2401-PLUS.pgn",  "elo_min": 2401, "elo_max": 3200},
]

def get_config_by_name(name_or_idx: str) -> Optional[TrainConfig]:
    """Get config by name or index."""
    target = None
    if name_or_idx.isdigit():
        idx = int(name_or_idx)
        if 1 <= idx <= len(SCENARIOS):
            target = SCENARIOS[idx - 1]
        elif idx == 0 and len(SCENARIOS) > 0:
            target = SCENARIOS[0]
    else:
        for s in SCENARIOS:
            if s["name"] == name_or_idx:
                target = s
                break
    
    if target:
        return TrainConfig(**target)
    return None


def elo_to_index(elo: int) -> int:
    """Convert ELO value to embedding index (0-based)."""
    return max(0, min(elo - ELO_MIN, ELO_BUCKETS - 1))


def index_to_elo(index: int) -> int:
    """Convert embedding index to ELO value."""
    return index + ELO_MIN