# Fix Plan: Business Entity Resolution

Hand-off document for the agent doing the fixes. Every claim below was checked against the
code or the data (see "Evidence"). "Expected effect" describes what the fix should do. Those
effects are **not measured yet**, so measure each one on the validation setup in section 1
and record it in `RESULTS.md`.

---

## 0. Hard rules (never violate)

- No external data, APIs, geocoding or internet lookups. Any model must be MIT/Apache 2.0 and ≤8B params (no Llama/Gemma).
- Don't hard-code, filter or one-hot countries. Partitioning by the `country` string is fine; France is unseen in training.
- Every test S1 entity gets exactly one output row. `candidate_pairs.tsv` = exactly the pairs the model scores, and matches must be a subset of it.
- `src/metrics.py` is verified correct: **do not change its logic.**
- Tune thresholds/rules on out-of-fold predictions from the *training* portion, never on the holdout, never on test.
- The machine has 16 GB RAM and ~7.5 GB free disk. Don't write big caches or pickles of whole pools.

## Data facts the fixes rely on

| Fact | Value |
|---|---|
| Train S1 / S2 / S3 | 2,206,821 / 5,034,616 / 5,285,603 (US + India) |
| Test S1 / S2 / S3 | 1,732,544 / 4,887,273 / 5,082,316 (India 810k, US 663k, France 259k S1) |
| Singletons | 5.6% (same in both countries) |
| Matches per S1 | mean 3.46, max 11 (≈1.9 from S2, ≈2.0 from S3) |
| Share of S2/S3 records that are a true match of some S1 | **74%**, so the pool is mostly near-duplicates of *other* entities |
| Target owned by >1 S1 | **0** (one-owner rule is valid) |
| Cross-country matches | **0** (per-country partitioning is safe) |
| Indic script in India S2 / S3 names | **23% / 13%** (addresses ~23%). S1 is always Latin |

---

## 1. Chunking and sample size (30k)

- **Validation:** split train S1 80/20 by entity, stratified by country, fixed seed, and save the IDs.
  Train on 30k S1 from the 80% and evaluate on 30k S1 from the 20%.
  **Both must be matched against the full S2/S3 pool of their country (4–6M records), never a sampled pool.**
- **Test:** process S1 in chunks of 30k, **but never chunk or sample S2/S3.**
  Build the target index (TF-IDF matrix, inverted index) once per country and query it chunk by chunk.
- **One-owner rule:** apply it once per country after *all* chunks of that country are scored.
  To do that, keep only compact arrays `(s1_idx int32, cand_idx int32, prob float32)` per pair; roughly 0.6 GB for the whole test set.
- Chunk results are written to the output files in streaming fashion only after the country's one-owner step.

Report per country and overall: blocking recall ceiling, avg/max candidates per S1, macro F0.5,
singleton accuracy, matches-only F0.5, runtime. Also run a **France proxy** (train US → eval India, and the reverse).
Assert that the ground truth covers every train S1. Record the baseline of the current code first.

---

## 2. All issues, ranked

Impact: **Critical** = pipeline is wrong or can't run at scale · **High** = large score change · **Medium** · **Low** = hygiene.

### Priority 0: make evaluation truthful and the pipeline runnable

#### P0-1 (bug 1) Training uses only 3% of the non-matching S2/S3 records. Impact: Critical
- **Where:** `src/infer_test.py:71,77` (also `utils/create_chunks.py:61`, so `dataset_sample_50k/` and `output_50k/` are affected too).
- **Problem:** The training pool is ~20× sparser than at test. Because 74% of the pool consists of *other* entities' records, most of the hard negatives are removed. The model learns that "the most similar record is almost always the match". Rank/gap features and the threshold are calibrated for an easy world, which gives false merges at test time.
- **Fix:** Match training S1 against the full country pool. Don't reuse `dataset_sample_50k` for any score.
- **Expected effect:** Train distribution matches test, and the threshold becomes trustworthy. The CV score will probably *drop* (it's honest now) while the real test score rises.
- **Depends on:** P0-2 (full-pool blocking is infeasible until the matrix is sparse) and P1-6 (more pairs).

#### P0-2 (bugs 8 + 5) Blocking cosine matrix is dense: OOM and ~100 h runtime. Impact: Critical
- **Where:** `src/blocking.py:54` (`s1_vecs.dot(self.target_vecs_T)`), vectorizer config `:96-109`, fit in `infer_test.py:87-96`.
- **Evidence (measured):** 1,000 S1 × 300k targets gives an **82–99% dense** result, 14 s and ~3.5 GB. A full country pool (~4.7M) would be ~50 MB per S1 row. A 50k batch cannot fit; even 1–2k batches are ~100 GB.
- **Fix:**
  1. Normalize the test text exactly like train (the current fit uses raw `business_name + business_address` for test).
  2. Set `max_df` (e.g. drop char n-grams present in >0.5–2% of the pool; tune for recall), with a larger or unlimited vocabulary instead of `max_features=50000`, which keeps the *most common* n-grams.
  3. Query in batches of ~1–2k S1 and take top-k per row with `argpartition`.
  4. Fit IDF per country on that country's own pool (S1+S2+S3 text, unlabeled) at both train and test time. That keeps train and test consistent and gives France French IDF.
- **Expected effect:** Makes the full run feasible at all; also improves candidate quality because common n-grams ("road", " na", "ltd") stop dominating the cosine.
- **Unblocks:** P0-1, P1-2, P1-5, the full test run.

#### P0-3 (bug 9) Feature extraction won't scale to tens of millions of pairs. Impact: Critical (runtime)
- **Where:** `src/features.py:189-190` (`to_dict` of the full 4–5M-row target frame per batch), `:222` (DataFrame per S1), `:14-31` (pure-Python LCS), `:242` (Python loop for gaps); `src/postprocessing.py:54` (`iterrows`); `src/model.py:155` (81 thresholds × `iterrows`).
- **Fix:** Build pairs as index arrays `(s1_idx, cand_idx)`. Compute string metrics with `rapidfuzz.process.cpdist(..., scorer=..., workers=-1)` (installed, v3.14.6). Compute rank/gap features with `groupby` + `rank`/`transform`. Replace LCS with `rapidfuzz.distance.LCSseq`/`Indel`, or drop it if it's redundant. Vectorize postprocessing with `sort_values` + `drop_duplicates` + `groupby`, and vectorize the threshold sweep.
- **Check:** On a 2k-S1 sample, new features must equal the old ones (except the intentionally replaced LCS).
- **Expected effect:** 50–100× faster, so a 30k chunk takes minutes instead of hours.

#### P0-4 (bug 10) The validator used isn't the official one. Impact: Medium
- **Where:** `src/infer_test.py:28,243`, `src/pipeline.py:31,191`.
- **Correction to the original claim:** the import does work (`utils/validate_submission.py:28` defines it), but that file is a home-made copy. The official validator is `student_resource/utils/validate_submission.py` and has no such function.
- **Fix:** `subprocess.run([sys.executable, "student_resource/utils/validate_submission.py", "--matching", ..., "--candidate", ..., "--test-dir", ...])` and check the exit code.
- **Expected effect:** Submission checks match the real scorer's rules; also avoids loading ~10M IDs into pandas.

### Priority 1: directly change which pairs are matched (largest score impact)

#### P1-1 (bug 4) One-owner rule applied per 50k batch, not globally. Impact: High
- **Where:** `src/infer_test.py:205`.
- **Problem:** A single S2/S3 record can be assigned to S1 entities in different batches. The data guarantees each target has at most one owner, so every duplicate assignment includes at least one false positive. That's precision loss, and a 0.0 if it hits a singleton.
- **Fix:** Apply it once per country after all chunks are scored (see section 1).
- **Relation:** Changes the effective threshold, so re-tune the threshold after this and after P0-1.

#### P1-2 (bug 2) Candidates dropped at random. Impact: High
- **Where:** `src/blocking.py:88`: `list(cands)[:top_k*2]` on a `set`.
- **Problem:** The union of the three blocking passes can exceed 100 ids and is cut to 50 in hash order. Python string hashing is randomized per process, so which true matches survive changes **between runs** (results are also non-reproducible).
- **Fix:** Keep an ordered list: B1 by cosine score descending first, then B2, then B3, with de-duplication; truncate that list.
- **Expected effect:** Higher and deterministic blocking recall.
- **Relation:** Needs the cosine scores kept, which feeds P1-5.

#### P1-3 (new N1) Indic-script names/addresses are destroyed. Impact: High (India ≈ 47% of test S1)
- **Where:** `src/normalization.py:14-18,98`. NFKD `combining` removal plus `[^\w\s]` strip vowel signs (matras) and viramas.
- **Evidence:** `राम मार्केटिंग प्राइवेट लिमिटेड` becomes `र म म रक ट ग पर इव ट ल म ट ड`. 23% of India S2 names are in Indic script, while S1 is always Latin, so these records can never match on the name.
- **Fix:** Write our own rule-based transliteration to Latin. The Unicode blocks for Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and Malayalam share the same layout (same offset = same sound), so one ~80-entry table covers all nine. Apply it **before** accent stripping. Optionally map common phonetic spellings (`praivet`→`private`, `limited`, `el el pi`→`llp`). No external library or data.
- **Expected effect:** Much higher blocking recall and name similarity for India.
- **Relation:** Changes TF-IDF input, so re-measure blocking recall afterwards.

#### P1-4 (new N2) Rare-token blocker keeps the wrong records. Impact: Medium–High
- **Where:** `src/blocking.py:33-35`, `:77-79`.
- **Problem:** The comment says over-saturated tokens are skipped, but the code keeps the first 1,000 postings in *file order* and queries take the first 25, which are effectively random records for common tokens. The `idf > 2.0` cutoff (smooth IDF) admits any token in <~37% of documents, i.e. almost all.
- **Fix:** Skip tokens whose document frequency is above a cap, instead of truncating. Keep the full posting list for rare tokens. Rank the returned targets (e.g. by number or IDF-mass of shared rare tokens) before taking top-k.
- **Relation:** Uses the IDF from P0-2 and the ordered list from P1-2.

#### P1-5 (bug 11) Blocking cosine not used as a feature or for ranking. Impact: Medium–High
- **Where:** `src/blocking.py:61-71` (computed, then discarded); `src/features.py:224-231` ranks by a hand-weighted composite score.
- **Fix:** Return the cosine per candidate (0 for candidates from B2/B3 only), add it to `FEATURE_COLS`, and compute `cand_rank`, gap-to-best and gap-to-second from it (and/or from a first-pass model score).
- **Relation:** Needs P1-2 (keep scores) and P0-2.

#### P1-6 (bug 5, part b) Rare tokens get IDF 1.0. Impact: Medium
- **Where:** `src/features.py:119-120` (`self.idf_dict.get(tok, 1.0)`) combined with `max_features=50000`.
- **Problem:** Tokens outside the vocabulary are exactly the rarest, most distinctive ones, but they receive the *lowest* possible weight. `sum_shared_idf`, `max_shared_idf` and `idf_coverage_ratio` are therefore inverted for the most informative tokens.
- **Fix:** Use the per-country word IDF from P0-2 with no `max_features` cap. For unseen tokens, default to the maximum IDF.

### Priority 2: normalization and feature quality

#### P2-1 (bug 7) Generic words stripped as "legal suffixes". Impact: Medium
- **Where:** `src/normalization.py:22-29`.
- **Evidence:** `"Summit Solutions Group"` → core `"summit"`; `"ABC Services Pvt Ltd"` → `"abc"`. Distinct businesses collapse to the same core, and `name_core_exact` fires wrongly, causing false merges.
- **Fix:** Strip only true legal forms (pvt/private limited, ltd, llc, llp, inc, corp, plc, gmbh, sarl, sas, sa, eurl, sasu, sci, ei, …). Keep services/solutions/group/enterprises/company/holding(s) in the core.

#### P2-2 (bug 6) French legal forms missing. Impact: Medium (France ≈ 15% of test)
- **Evidence:** Share of test France names: SARL 26%, SAS 19%, **EURL 6.4%, SASU 4.2%, SCI 3.4%, EI 1.5%**.
- **Fix:** Add `eurl, sasu, sci, ei, snc, selarl` (plus dotted variants) to the legal-form list. `ei` must only match as a whole word.
- **Relation:** Do this together with P2-1 (same list).

#### P2-3 (bug 3) "Postal code" is actually the house number. Impact: Medium
- **Where:** `src/normalization.py:78,135-138`; `src/features.py:107-113`; `src/blocking.py:39-44,82-86`.
- **Evidence:** 5–6 digit numbers appear in India 0.27%, France 0.4%, US 10.8%. In the US, 8.8% are followed by a street word (`17560 Ellis Road`); only 0.01% are at the end like a ZIP.
- **Fix:** Replace with:
  - house/street number: the leading number of the street component, with exact-match, mismatch and missing flags;
  - overlap of comma-separated address components, compared order-independently (addresses are often reordered, e.g. `IA, Iowa City, 1064 Newton Rd`), e.g. the share of components with a fuzzy match ≥ 0.9;
  - city/locality token overlap.
- **B3 blocker:** key on (house number + first street token) or (locality + name prefix) instead of "postal".

#### P2-4 (new N5) No missing-address flag. Impact: Low–Medium
- ~3% of S2/S3 addresses are empty, so address similarity is 0, the same as a real mismatch. Add `addr_missing`, and set address similarities to NaN (LightGBM handles NaN natively).

#### P2-5 (new N6) Landmark text dropped. Impact: Low
- `src/normalization.py:128-132` removes `near …` phrases from the address, and the extracted `landmark` column is never used. Either keep the text in the address or add a landmark similarity feature. Measure both.

#### P2-6 Address abbreviation collisions. Impact: Low
- `\br\b→rue`, `\bst\b→street` (also "Saint"), `\bste\b→suite` (also "Sainte"), `\bdr\b→drive` (also "Doctor"). These are applied symmetrically, so the damage is small. French forms like `r.`, `bd`, `av`, `ets`, `cie` are missing. Low priority; measure before and after.

### Priority 3: model and hygiene

#### P3-1 (bug 12) Row subsampling silently ignored. Impact: Low
- `src/model.py:73`: add `"subsample_freq": 1`. Consider early stopping with a GroupKFold validation fold.

#### P3-2 (new N3) `pipeline.py` can't run at scale, and the README points to it
- It loads all train and test data at once and repeats the dense blocking. Make `infer_test.py` the single entry point, turn `pipeline.py` into a thin wrapper or delete it, and fix the README paths (`student_resource/dataset/...`).

#### P3-3 (new N4) Docs describe features that don't exist
- `src/model.py:5-7` (isotonic calibration, focal loss, hold-one-country-out), `src/postprocessing.py:5` (margin pruning), unused `IsotonicRegression` import. Remove the claims, or implement and measure them.

#### P3-4 (new N7) `requirements.txt` not pinned
- Pin the versions actually used: `pandas==3.0.2 numpy==1.26.4 scipy==1.17.1 scikit-learn==1.8.0 lightgbm==4.7.0 rapidfuzz==3.14.6 tqdm==<installed>`.

#### P3-5 (new N8) Pool density differs between train and test (watch item, not a bug)
- Train has 4.7 S2/S3 records per S1; test has 5.5–5.8. The US pool is smaller in test (3.8M vs 6.2M), and India's is larger (4.7M vs 4.1M). Rank/gap features may shift. Compare prediction-rate statistics (matches per S1, share of empty predictions) between the validation run and the test run.

---

## 3. Dependency map

```
P0-2 (sparse blocking, normalized TF-IDF, per-country IDF)
  ├─> P0-1 (full-pool training)      ── needs P0-3 too (more pairs)
  ├─> P1-2 (ordered candidates) ──> P1-5 (cosine feature + ranking)
  ├─> P1-4 (rare-token blocker)
  └─> P1-6 (IDF for rare tokens)

Normalization changes: P1-3, P2-1, P2-2, P2-3, P2-6
  └─> change TF-IDF input: re-measure blocking recall after each

P0-1 and P1-1 (global one-owner) both shift calibration: re-tune the threshold after each

Independent: P0-4, P3-1, P3-2, P3-3, P3-4
```

## 4. Recommended execution order

Do these one at a time. After each, run validation (30k train / 30k holdout, full pool) and
add a before/after row to `RESULTS.md`. Keep what helps and revert what doesn't.

0. Commit the current code as-is. Build the validation harness and record the **baseline** (current logic, with only the minimal changes needed to run: small matmul batches, features built on candidate targets only).
1. P0-4 validator, P3-1 `subsample_freq` (trivial, independent)
2. P0-2 blocking sparsity + TF-IDF fit
3. P0-3 vectorized features/postprocessing (check that features are identical)
4. P1-2 ordered candidates → P1-4 rare-token blocker
5. P0-1 full-pool training
6. P1-1 global one-owner
7. P2-1 + P2-2 legal forms
8. P1-3 Indic transliteration
9. P2-3 house number + address components, P2-4, P2-5
10. P1-5 cosine + rank/gap features, P1-6 IDF
11. Decision layer: threshold, a separate "no-match" threshold on the top score, and a relative-to-best rule. Tune on OOF, then confirm on the holdout.
12. P3-2, P3-3, P3-4 docs/entry point/pins. Then time a 30k test chunk, estimate the full run, run it, and run the official validator on both files.

Targets: blocking recall ceiling ≥ 97% at a manageable candidate count. The France-proxy score should be close to the in-country score; report the gap.
