# Model Checkpoints

Pre-trained models are hosted on **HuggingFace**:

**🤗 https://huggingface.co/ygkla/zweig-chess-engine-models**

## Download Models

You can download models directly using the inference script:

```bash
# List available models
python scripts/inference.py --list

# Download a specific model
python scripts/inference.py --download maia_05

# Download and run inference
python scripts/inference.py --model maia_05 --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
```

## Available Models

| Model | ELO Range | Status | Filename |
|-------|-----------|--------|----------|
| maia_01 | 400-1000 | ✅ Available | `maia_finetuned_train_01_400-1000.pth` |
| maia_02 | 1001-1200 | ✅ Available | `maia_finetuned_train_02_1001-1200.pth` |
| maia_03 | 1201-1325 | ✅ Available | `maia_finetuned_train_03_1201-1325.pth` |
| maia_04 | 1326-1425 | 🔄 Training in progress | - |
| maia_05 | 1426-1500 | ✅ Available | `maia_finetuned_train_05_1426-1500.pth` |
| maia_06 | 1501-1575 | 🔄 Training in progress | - |
| maia_07 | 1576-1650 | 🔄 Training in progress | - |
| maia_08 | 1651-1750 | ✅ Available | `maia_finetuned_train_08_1651-1750.pth` |
| maia_09 | 1751-1875 | ✅ Available | `maia_finetuned_train_09_1751-1875.pth` |
| maia_10 | 1876-2100 | ✅ Available | `maia_finetuned_train_10_1876-2100.pth` |
| maia_11 | 2101-2400 | ✅ Available | `maia_finetuned_train_11_2101-2400.pth` |
| maia_12 | 2401+ | ✅ Available | `maia_finetuned_train_12_2401-PLUS.pth` |

## Git Tracking

Model files are excluded from git via `.gitignore` due to their large size.
