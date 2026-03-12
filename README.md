# Zweig Chess Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Models-yellow.svg)](https://huggingface.co/ygkla/zweig-chess-engine-models)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-blue.svg)](https://huggingface.co/datasets/ygkla/zweig-chess-engine-processed)

A human-aligned chess engine that predicts moves a human player at a specific ELO rating would likely make. Unlike traditional engines that find the "best" move, Zweig understands how humans actually play.

## Overview

This project implements two generations of neural network architectures for predicting human chess moves:

- **Legacy Model (Gen-1)**: CNN-based architecture with 14-channel input (pieces + repetition)
- **Maia-2 Model (Gen-2)**: Enhanced architecture with 19-channel input including castling rights and skill-aware gating

The models are trained on millions of real Lichess games across 12 different ELO brackets (400-3200+).

## Models & Dataset

**🤗 HuggingFace Resources:**
- **Models**: https://huggingface.co/ygkla/zweig-chess-engine-models
- **Dataset**: https://huggingface.co/datasets/ygkla/zweig-chess-engine-processed

The dataset contains 11.15M human chess games from Lichess, bucketed into 12 ELO ranges (400-3200+).

## Features

- **ELO-Specific Predictions**: Separate models trained for 12 different skill levels
- **Skill-Aware Architecture**: Maia-2 incorporates player ELO through sigmoid gating (feature modulation)
- **Rule-Aware Input**: Full encoding of castling rights, en-passant, and repetition
- **Scalable Data Pipeline**: Automated downloading and processing of Lichess game databases
- **ResNet-Based**: 12-layer residual network preserving spatial information

## Architecture

### Input Representation

**Legacy (14 channels)**:
- Channels 0-5: White pieces (P, N, B, R, Q, K)
- Channels 6-11: Black pieces (P, N, B, R, Q, K)
- Channel 12: Repetition flag
- Channel 13: Normalized move count

**Maia-2 (19 channels)**:
- Channels 0-5: White pieces (P, N, B, R, Q, K)
- Channels 6-11: Black pieces (P, N, B, R, Q, K)
- Channel 12: Repetition flag
- Channels 13-16: Castling rights (WK, WQ, BK, BQ)
- Channel 17: En passant square
- Channel 18: Normalized move count

### Network Architecture

```
Input [Batch, 19, 8, 8]
    ↓
Conv2d (19→256) + BatchNorm + ReLU
    ↓
12× ResBlock (256 channels, 3×3 kernel)
    ↓
Skill-Aware Gating (ELO embeddings)
    ↓
Policy Head → [Batch, vocab_size]
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- python-chess
- zstandard

### Setup

```bash
# Clone the repository
git clone https://github.com/ygkali/zweig-chess-engine.git
cd zweig-chess-engine

# Install the package (recommended)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Project Structure

```
zweig-chess-engine/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration and ELO brackets
│   ├── model.py           # Neural network architectures
│   ├── dataset.py         # Data loading and preprocessing
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Evaluation metrics
│   └── utils.py           # Helper functions
├── scripts/
│   ├── data_pipeline.py   # Automated data download from Lichess
│   ├── trainer.py         # Training orchestration
│   └── inference.py       # Model inference
├── docs/
│   ├── 01-Architecture-and-Design.md
│   ├── 02-Data-Engineering.md
│   ├── 03-Training-Strategy.md
│   └── 04-Benchmarks-and-Results.md
├── data/
│   ├── raw/              # Raw PGN files (not tracked)
│   └── processed/        # Processed training data
├── checkpoints/          # Model checkpoints (not tracked)
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Download Models (from HuggingFace)

```bash
# List available models
python scripts/inference.py --list

# Download a specific model
python scripts/inference.py --download maia_05
```

### 2. Download Dataset (from HuggingFace)

```bash
# Download all ELO buckets
python scripts/data_pipeline.py --source huggingface

# Download specific buckets (e.g., 5, 8, 11)
python scripts/data_pipeline.py --source huggingface --bucket 5 8 11

# List available buckets
python scripts/data_pipeline.py --list
```

### 3. Inference

```bash
# Predict moves for a position
python scripts/inference.py --model maia_05 --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

# Interactive mode
python scripts/inference.py --model maia_11 --interactive
```

### 4. Training (Optional)

```bash
# Train a model for a specific ELO bracket
python scripts/trainer.py --config maia_06 --arch maia2
```

## ELO Brackets

| Model | ELO Range | Status |
|-------|-----------|--------|
| maia_01 | 400-1000 | ✅ Available |
| maia_02 | 1001-1200 | ✅ Available |
| maia_03 | 1201-1325 | ✅ Available |
| maia_04 | 1326-1425 | 🔄 Training in progress |
| maia_05 | 1426-1500 | ✅ Available |
| maia_06 | 1501-1575 | 🔄 Training in progress |
| maia_07 | 1576-1650 | 🔄 Training in progress |
| maia_08 | 1651-1750 | ✅ Available |
| maia_09 | 1751-1875 | ✅ Available |
| maia_10 | 1876-2100 | ✅ Available |
| maia_11 | 2101-2400 | ✅ Available |
| maia_12 | 2401+ | ✅ Available |

## Training Configuration

Default hyperparameters:
- **Batch Size**: 8192 positions
- **Learning Rate**: 2e-5
- **Total Steps**: 30,000
- **Optimizer**: AdamW (weight_decay=1e-4)
- **Loss**: Cross-Entropy

## Documentation

Detailed documentation is available in the `docs/` directory:

- [01-Architecture-and-Design.md](docs/01-Architecture-and-Design.md) - Model architecture details
- [02-Data-Engineering.md](docs/02-Data-Engineering.md) - Data pipeline and processing
- [03-Training-Strategy.md](docs/03-Training-Strategy.md) - Training methodology
- [04-Benchmarks-and-Results.md](docs/04-Benchmarks-and-Results.md) - Performance metrics

## Key Differences from Stockfish

| Feature | This Project | Stockfish |
|---------|-------------|-----------|
| Goal | Predict human-like moves | Find objectively best moves |
| Method | Supervised learning | Minimax + alpha-beta pruning |
| Input | ELO-aware board state | Position + depth search |
| Output | Move probability distribution | Best move + evaluation |
| Speed | Fast (single forward pass) | Slow (deep tree search) |

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by [Microsoft's Maia Chess](https://maiachess.com/)
- Game data from [Lichess Open Database](https://database.lichess.org/)
- Based on AlphaZero/Leela Chess Zero architecture principles

## Citation

If you use this project in your research, please cite:

```bibtex
@software{zweig-chess-engine,
  title={Zweig Chess Engine: Human-Like Chess Move Prediction},
  author={[ygkali]},
  year={2026},
  url={https://github.com/ygkali/zweig-chess-engine}
}
```

## References

- McIlroy-Young, R., Sen, S., Kleinberg, J., & Anderson, A. (2020). Aligning Superhuman AI with Human Behavior: Chess as a Model System. *KDD '20*.
- Silver, D., et al. (2017). Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm. *arXiv:1712.01815*.
- Lichess Open Database: https://database.lichess.org/

## Contact

For questions or issues, please open an issue on GitHub.
