# 🏋️ Training Strategy & Methodology

This document outlines the computational resources, time budget, and specific training protocols used to develop the chess AI.

The project explores two distinct training paradigms to optimize human-like move prediction: **Transfer Learning** (Legacy) and **Isolated Training** (Maia-2).

## ⏱️ Computational Budget & Timeline

Developing a production-grade AI model requires significant resource allocation. Below is the breakdown of the time investment for this project on NVIDIA GPUs (A100/L4).

| Phase | Duration | Details |
| :--- | :--- | :--- |
| **Data Processing** | ~20 Hours | Parsing, filtering, and "bucketing" **11.15 million** games (2025 Data). |
| **Base Model Training** | 33 Hours | Pre-training the generic "Legacy" model on mixed data (Phase 0). |
| **Phase 1: Rapid Production** | ~1 Hour | Fast adaptation per bucket (5,000 steps). |
| **Phase 2: Scientific Deepening**| ~6.5 Hours | Deep fine-tuning per bucket (30,000 steps). |
| **Scratch Training** | ~7 Hours | Training Maia-2 (2400+) from zero (300k steps). |

---

## 🧪 Evaluation Protocol: Cross-Temporal Validation

To ensure the model is not simply memorizing games, we strictly separated training and testing data by **different years** to guarantee zero leakage.

* **Training Data:** **Lichess 2025 Dataset (Jan - Dec)**.
  * *Goal:* To train the model on the most current chess theory and modern opening trends.
* **Test Data:** **Lichess 2022 Dataset (Jan, Feb, Mar)**.
  * *Goal:* To evaluate performance on a completely unseen historical dataset.

> **Why this matters:** Using distinct years guarantees that no game in the training set (2025) appears in the test set (2022). This proves that the model's accuracy stems from **generalization** of chess principles rather than memorization.

---

## 🔄 Experiment A: Transfer Learning (Legacy Strategy)

This approach mimics the "Pre-training + Fine-tuning" paradigm used in Large Language Models (LLMs). We divided the development into two strategic eras:

### Era 1: Rapid Adaptation (5,000 Steps)
In the initial phase, the goal was to adapt the robust "Base Model" to different ELO buckets with minimal compute cost.

* **Configuration:** Constant Learning Rate, 5,000 Steps.
* **Goal:** Style Transfer (shifting the model's bias without destroying general knowledge).
* **Outcome:** Models stabilized at **~1.50 Loss**. Accuracy ranged between **49% - 51%**. This established the baseline for mass production.

### Era 2: Scientific Deepening (30,000 Steps)
Following the initial success, we extended the training duration by 6x for Elite buckets (Master & GM) to maximize accuracy.

* **Configuration:** 30,000 Steps using **MultiStepLR** (Learning Rate Decay).
* **Technical Detail:** Decaying the learning rate allowed the model to settle into sharper local minima.
* **Outcome:**
    * **Loss Improvement:** Dropped from ~1.50 to **1.38**.
    * **Performance:** These "Scientific" models achieved state-of-the-art accuracy (**~51.01%**), significantly outperforming the Rapid models.

---

## ⚡ Experiment B: Isolated Training (Maia-2)

For the advanced **19-channel architecture**, we abandoned transfer learning in favor of training from scratch.

* **Target:** 2400+ ELO Bucket.
* **Configuration:** 300,000 Steps, 19-Channel Input (Rule-Aware).
* **Outcome:** While conceptually superior, the Isolated Training approach yielded lower accuracy (~44%) compared to the Transfer Learning approach (~55%) due to the lack of massive pre-training.

---

## 📝 Hyperparameter Log

| Parameter | Rapid Models (5k) | Scientific Models (30k) |
| :--- | :--- | :--- |
| **Steps** | 5,000 | **30,000** |
| **LR Scheduler** | Constant | **MultiStepLR** |
| **Target Loss** | ~1.50 | **~1.38** |
| **Training Time** | ~1 Hour | ~6.5 Hours |

### Final Configuration (Best Performing Scientific Model)
* **Architecture:** ResNet-9 (14 Channels)
* **Batch Size:** 9216
* **Optimizer:** SGD with Momentum (0.9)
* **Loss Function:** CrossEntropyLoss