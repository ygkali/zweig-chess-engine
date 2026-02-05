# Data Directory

This directory contains training data for the Zweig Chess Engine.

## Download Dataset

You can download the processed dataset directly from HuggingFace using the data pipeline script:

```bash
# Download all ELO buckets
python scripts/data_pipeline.py --source huggingface

# Download specific buckets (e.g., 5, 8, 11)
python scripts/data_pipeline.py --source huggingface --bucket 5 8 11

# List available buckets
python scripts/data_pipeline.py --list

# Verify downloaded data
python scripts/data_pipeline.py --verify
```

**Dataset URL**: https://huggingface.co/datasets/ygkla/zweig-chess-engine-processed

## Structure

```
data/
├── raw/          # Raw PGN files (not tracked in git)
└── processed/    # Processed training data by ELO bracket
```

## ELO Buckets

| Bucket | ELO Range | Status | File |
|--------|-----------|--------|------|
| 01 | 400-1000 | ✅ Available | `train_01_400-1000.pgn` |
| 02 | 1001-1200 | ✅ Available | `train_02_1001-1200.pgn` |
| 03 | 1201-1325 | ✅ Available | `train_03_1201-1325.pgn` |
| 04 | 1326-1425 | ✅ Available | `train_04_1326-1425.pgn` |
| 05 | 1426-1500 | ✅ Available | `train_05_1426-1500.pgn` |
| 06 | 1501-1575 | ✅ Available | `train_06_1501-1575.pgn` |
| 07 | 1576-1650 | ✅ Available | `train_07_1576-1650.pgn` |
| 08 | 1651-1750 | ✅ Available | `train_08_1651-1750.pgn` |
| 09 | 1751-1875 | ✅ Available | `train_09_1751-1875.pgn` |
| 10 | 1876-2100 | ✅ Available | `train_10_1876-2100.pgn` |
| 11 | 2101-2400 | ✅ Available | `train_11_2101-2400.pgn` |
| 12 | 2401+ | ✅ Available | `train_12_2401-PLUS.pgn` |

## Git Tracking

Data files are excluded from git via `.gitignore` due to their large size.
