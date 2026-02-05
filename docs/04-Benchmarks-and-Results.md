# 📊 Benchmarks & Results

This document presents the quantitative performance of the models, comparing the **Legacy (Transfer Learning)** approach against the **Maia-2 (Isolated Training)** architecture.

## 🏆 Key Findings Summary

| Metric | Legacy Model (14-Ch) | Maia-2 Model (19-Ch) |
| :--- | :--- | :--- |
| **Training Strategy** | Base Model + Fine-Tune | Trained from Scratch |
| **Target ELO** | 2400+ (Grandmaster) | 2400+ (Grandmaster) |
| **Total Training Time** | ~39 Hours (33+6) | ~7 Hours |
| **Top-1 Accuracy** | **~49.9%** 🥇 | ~44.0% |

### 📉 The "Architecture vs. Data" Trade-off

Despite Maia-2 having a superior architecture (19 channels including Castling Rights & ELO metadata), it underperformed compared to the Legacy model.

#### Analysis: Why did the better architecture lose?
1.  **The "Cold Start" Problem:** The Maia-2 model started with random weights. It had to spend the first few hours just re-learning basic chess rules (how knights move, what is a checkmate).
2.  **The Pre-training Advantage:** The Legacy model entered the fine-tuning phase having already absorbed **33 hours** of general chess knowledge. It only needed 6 hours to adapt its style.
3.  **Data Scarcity in Elite Buckets:** The 2400+ ELO bucket represents only **0.3%** of the population. Training from scratch on such a small slice of data led to suboptimal convergence compared to fine-tuning a model that saw the *entire* spectrum of chess games.

---

## 🎯 Accuracy by ELO (Legacy Model)

The following table demonstrates the Legacy model's performance across different skill levels when tested on the 2022 Unseen Dataset.

| Model Variant | Test Set (GM Data) Accuracy | Interpretation |
| :--- | :--- | :--- |
| **Average (1450)** | 45.41% | Struggles to predict GM moves (Too random). |
| **Master (2200)** | 49.91% | **High alignment** with GM logic. |
| **Grandmaster (2400+)** | 49.37% | Excellent prediction capability. |

> **Conclusion:** The model successfully captured stylistic differences. A model trained on 1450 ELO data predicts GM moves significantly worse than a model trained on 2200+ data, proving that "Skill" is a learnable feature.

---

## 🖼️ Visual Analysis

*(Placeholder for Chessboard images comparing predictions)*

* **Case Study 1:** In tactical positions, the **Legacy (Fine-Tuned)** model correctly predicted a sacrifice played by a GM, whereas the **Base Model** (Generic) chose a safe, passive move.
* **Case Study 2:** The **Maia-2** model, despite lower overall accuracy, showed 100% success in detecting illegal castling attempts due to its explicit input channels, verifying the architectural benefit.

## 🚀 Final Verdict

For limited compute budgets (~40 hours), **Transfer Learning (Fine-Tuning) is superior** to training specialized architectures from scratch. Future work should focus on pre-training the 19-channel Maia-2 architecture on the full dataset before fine-tuning.