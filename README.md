# Business Entity Resolution Pipeline

High-precision entity resolution framework across heterogeneous data sources ($S_1, S_2, S_3$) evaluated under macro-averaged $F_{0.5}$.

---

## 📁 Repository Structure

```text
amazon_ml_27/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── output/
│   ├── candidate_pairs.tsv       # Scored candidate pool (audit file)
│   └── matching_results.tsv      # Leaderboard submission file
├── src/
│   ├── __init__.py
│   ├── eda.py                   # Stage 0: Automated data auditing & statistics
│   ├── normalization.py         # Stage 1: NFKD accent folding, legal suffix isolation, address parsing
│   ├── blocking.py              # Stage 2: High-recall Multi-Blocker (Char TF-IDF kNN, rare-tokens, locality keys)
│   ├── features.py              # Stage 3: Pairwise fuzzy, IDF-mass, structural, & rank context features
│   ├── model.py                 # Stage 4: LightGBM GBDT with GroupKFold CV & threshold optimization
│   ├── postprocessing.py        # Stage 5: One-owner global assignment & singleton formatting
│   ├── metrics.py               # Official Macro F0.5 evaluation metric with singleton handling
│   └── pipeline.py              # Master pipeline orchestrator
├── utils/
│   ├── generate_synthetic_data.py  # Realistic multi-country synthetic data generator
│   └── validate_submission.py      # Hard constraint checker matching challenge specs
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. When Dataset Arrives
Place the competition files into:
- `dataset/train/` (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`)
- `dataset/test/` (`test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`)

### 3. Run Complete End-to-End Pipeline
```bash
python3 src/pipeline.py --train-dir dataset/train --test-dir dataset/test --output-dir output
```

This single command will:
1. Run automated EDA profiling on training sources and verify data integrity.
2. Preprocess and normalize multi-view text fields (names, addresses, landmarks, postal codes).
3. Fit unsupervised TF-IDF on combined corpus (`train + test`) to support unseen countries (France).
4. Run the high-recall multi-blocker to generate candidate pairs and save `output/candidate_pairs.tsv`.
5. Extract pairwise similarity and context features.
6. Train LightGBM matcher using GroupKFold (grouped by $S_1$ entity ID to prevent leakage).
7. Optimize decision threshold $t^*$ directly on Macro $F_{0.5}$.
8. Apply one-owner global consistency postprocessing.
9. Generate `output/matching_results.tsv`.
10. Automatically run `utils/validate_submission.py` to ensure submission compliance.

---

## 🔍 Validation Check
You can independently validate your submission files anytime:
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
