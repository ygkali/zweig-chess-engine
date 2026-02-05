# ⚙️ Data Engineering & Pipeline Optimization

This document details the complete data lifecycle: from acquiring raw PGNs to segregating them into skill-based "buckets" and optimizing the I/O pipeline for high-performance training.

## 1. Data Collection & Partitioning Strategy

The core hypothesis of this project is that a generic chess model cannot capture the stylistic nuances of specific rating groups. To address this, we implemented a strict **"Data Bucketing"** strategy using Lichess databases.

### The Source: 2025 Dataset
To ensure the model learns the most current "meta" and opening trends, we utilized the most recent available data.

* **Origin:** Lichess Open Database (**Year 2025**).
* **Coverage:** Data was aggregated from **all 12 months of 2025** (Jan-Dec). This comprehensive collection ensures that seasonal variations in player activity and the latest theoretical novelties are represented in the training set.
* **Selection Criteria:** Standard time controls (Blitz, Rapid, Classical). Bullet and Hyper-Bullet games were excluded to reduce "noise."

### Methodology: Academic Standard
Rather than using arbitrary linear steps (e.g., every 100 points), the data partitioning follows a **cumulative distribution strategy** aligned with academic standards and the actual Lichess player population curve.

This approach ensures that each bucket captures a statistically significant segment of the player base, from novices to the top 0.3% of elite players.

### The "Bucket" Definitions
We processed the raw data and split it into 12 distinct buckets based on the following JSON configuration:

| ID | ELO Range | Population % | Description |
| :--- | :--- | :--- | :--- |
| **01** | 400 - 1050 | 8.3% | Novice |
| **02** | 1051 - 1200 | 9.6% | Beginner |
| **03** | 1201 - 1325 | 11.6% | Casual Player |
| **04** | 1326 - 1425 | 11.4% | Lower Intermediate |
| **05** | 1426 - 1500 | 9.1% | Intermediate |
| **06** | 1501 - 1575 | 9.1% | Upper Intermediate |
| **07** | 1576 - 1650 | 8.5% | Advanced Intermediate |
| **08** | 1651 - 1750 | 10.2% | Club Player |
| **09** | 1751 - 1875 | 9.8% | Strong Club Player |
| **10** | 1876 - 2100 | 8.8% | Expert |
| **11** | 2101 - 2400 | 3.3% | Master Level |
| **12** | 2401 - 3000 | 0.3% | Elite / Top Tier |



> **Note on Ratings:** This project uses Lichess ratings (Glicko-2 system), which are typically inflated compared to FIDE (OTB) ratings. For context, a "2400+" rating here represents the top 0.3% of the online pool (super-expert level), though not necessarily implying a formal Grandmaster title.

> **Insight:** As seen in the table and distribution graph, Bucket 12 represents the "Elite" outliers. This makes training for this specific style computationally challenging due to data scarcity compared to the populous middle ranges (Buckets 3-4).

### Dataset Volume & Balancing
To prevent the model from simply memorizing the most common rating ranges (Gaussian peak at ~1500), we applied a **balancing strategy** across the buckets.

* **Total Dataset Size:** ~11,150,000 Games processed.
* **Standard Buckets (01-11):** We targeted approximately **1,000,000 games per bucket**. This ensures that the "Novice" model and the "Expert" model receive similar amounts of training exposure.
* **The Elite Constraint (Bucket 12):** Due to the extreme rarity of 2400+ games in the public domain, we utilized all available data for this range, which amounted to **~150,000 games**.

> **Critical Implication:** The Elite model (2400+) was trained on roughly **15%** of the data volume compared to other buckets. This inherent data scarcity is identified as a primary factor contributing to the lower convergence rates observed in the top-tier model benchmarks.

---


## 2. Preprocessing & Encoding

Before training, raw PGN moves are converted into a format the Neural Network can understand.

1.  **Parsing:** Using `python-chess` to iterate through moves.
2.  **Filtering:**
    * Games shorter than 10 moves are discarded.
3.  **Encoding:**
    * **Input:** The board state is serialized into Bitboards (See Architecture doc).
    * **Label:** The human move is encoded as an index (0-1968) representing the specific move in the UCI format (e.g., `e2e4`).

---