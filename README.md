# Business Entity Resolution (Amazon ML Challenge)

For every Source 1 (S1) business, find its matching records in Source 2 / Source 3, scored with
macro F0.5 per S1 entity. Everything runs locally on the provided data: no external data, APIs or
pretrained models.

## Layout

```
src/
  normalization.py   text normalization: Indic-script transliteration (own rule table), accents,
                     legal forms, abbreviations, website/junk cleanup, leading zeros, house numbers
  features.py        vectorized pair features (rapidfuzz cpdist + sparse token/IDF/number overlaps)
  pipeline_core.py   per-country blocking index, sibling expansion, rival (same-name S1) features,
                     two-stage LightGBM, stage-1 top-k filter
  decision.py        one-owner rule + decision rule, macro F0.5 identical to src/metrics.py
  metrics.py         macro F0.5 (reference implementation)
  infer_test.py      train + full test inference + official validator
  validate.py        holdout validation harness (see below)
experiments/         blocking / transliteration / error-analysis scripts used to choose settings
utils/               data sampling helper, older local validator
student_resource/    challenge files; data expected in student_resource/dataset/{train,test}
```

## How it works

1. **Per country.** The country label only partitions records (training data has no cross-country
   matches); it is never a feature. Each country, including the unseen France, gets its own
   TF-IDF / IDF statistics fitted on its own unlabeled text.
2. **Blocking.** Word TF-IDF on name+address and on name (max_df=0.01), top-40 / top-15 per S1,
   plus *sibling expansion*: records sharing the exact normalized address or compact name with a
   high-confidence candidate.
3. **Features.** Name / address string similarities, IDF-weighted token overlap, address
   components, house / all-number overlap, compact names, blocking cosines, per-S1 rank/gap
   context, and *rival* features: other S1 entities with the same name and how well the candidate
   fits them.
4. **Stage 1** LightGBM scores all candidates and keeps the top 12 per S1 (this is the candidate
   set written to `candidate_pairs.tsv`). **Stage 2** LightGBM adds *sibling* features (similarity
   to the S1's other confident candidates) and gives the final probability.
5. **Decision.** One-owner rule per country (each S2/S3 record goes to at most one S1), then a
   threshold / top-score / relative-to-best rule tuned for macro F0.5 on out-of-fold predictions.

## Reproduce

```bash
pip install -r requirements.txt
python3 src/infer_test.py --n-train 30000 --prune-k 12 --rival-drop 0.2
```

Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (one row per test S1) and
runs the official validator. Runtime on a 10-core / 16 GB laptop is roughly 1.5–2.5 hours.

## Validation

```bash
python3 src/validate.py --adapter current --code-dir . --name myrun --prune-k 12 \
    --rival-drop-train 0.2 --rival-drop-eval 0.2
```

S1 entities are split 80/20 by `crc32(entity_id) % 5`; 30k training-side S1 train the model and
30k holdout S1 are scored, both matched against the **full** S2/S3 pool of their country.
Decision rules are tuned only on out-of-fold training predictions. `--rival-drop-eval 0.2`
simulates the test set, where about 20% of businesses have S2/S3 records but no S1 entity.
Results are appended to `RESULTS.md`; `experiments/error_analysis.py <run>` breaks the loss down.
