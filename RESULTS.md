# Validation results

All numbers are measured with `src/validate.py`: train and evaluate S1 samples come from disjoint sides of a fixed 80/20 split of the training data (crc32(entity_id) % 5), and every S1 entity is matched against the FULL S2/S3 pool of its country. The threshold is tuned on out-of-fold predictions of the training sample only.

| run | code | train → eval | scope | recall ceiling | avg cands | macro F0.5 | singleton acc | matches F0.5 | CV F0.5 (OOF) | minutes |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_original | original | 30k all → 30k all | India | 64.56% | 49.6 | 0.6679 | 0.3437 | 0.6862 |  | 93 |
| baseline_original | original | 30k all → 30k all | US | 69.74% | 49.7 | 0.6887 | 0.2844 | 0.7129 |  | 93 |
| baseline_original | original | 30k all → 30k all | ALL | 67.66% | 49.7 | 0.6803 | 0.3074 | 0.7022 | 0.8903 | 93 |

Notes on `baseline_original`: the original logic (commit 57fc86f) including its 3%-distractor training pool; only
run-enabling shims were added (queries in batches of 25 rows over 4 forked workers, float32 TF-IDF,
features built on candidate targets only). CV F0.5 is measured on that easy training pool, which is why it is
far above the full-pool holdout score. PYTHONHASHSEED=0 (the original truncates a set in hash order).
