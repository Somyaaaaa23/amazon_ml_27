# Business Entity Resolution Challenge: Approach Documentation

---

## 1. Problem Statement

### 1.1 What the problem is
Business identity data comes from **3 independent sources**. Each source holds partial, noisy records about real-world businesses, and the sources share **no common identifier**. Deciding which records refer to the same business is **Entity Resolution (ER)**.

### 1.2 The task
- **Source 1** is the *deduplicated reference* source (each business appears once).
- For **every Source 1 entity**, find **all** matching records in **Source 2** and **Source 3**.
- A Source 1 entity may match **zero, one, or many** records across S2 and S3.

### 1.3 Input data
| File | Purpose |
|---|---|
| `train_source1/2/3.tsv` | Training records (US and India) |
| `train_ground_truth.tsv` | `source1_entity_id` → comma-separated matched S2/S3 IDs (empty = singleton) |
| `test_source1/2/3.tsv` | Test records (US, India **and France**, which is unseen in training) |

Columns in each source file: `entity_id` (prefix S1-/S2-/S3- tells the source), `business_name`, `business_address`, `country`.

> All files are **tab-separated**. Always read with `sep="\t"`. Do not let a default comma parse silently collapse the columns.

### 1.4 Required output (two files in `output/`)
| File | Columns | Scored? |
|---|---|---|
| `matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` | **Yes**, the only leaderboard file |
| `candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` | No, used to audit blocking quality |

`candidate_pairs.tsv` must be the **final candidate set the model actually scored** (the last filtering stage before model inference). Matches must be a **subset** of the candidates.

### 1.5 Noise to expect
- **Names:** abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited), legal-suffix inconsistencies, DBA/trade names, `&` vs `and`, word-order swaps, typos, transliterations.
- **Addresses:** abbreviations (Rd/Road, St/Street), transliteration variants, missing components (no PIN, no state), landmark references ("Near SBI ATM"), municipal numbering formats, component reordering.

---

## 2. Marking Scheme

### 2.1 Metric: F0.5 (precision-heavy)

```
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

- **Macro-average:** F0.5 is computed **per Source 1 entity**, then averaged over all S1 entities in the evaluation set.
- **Singletons count.** For an entity with no true matches:
  - predict empty list → **1.0**
  - predict anything → **0.0**
- Precision is weighted 2× over recall. Merging two different businesses hurts more than missing a link.

### 2.2 Worked example (from the PS)
Prediction: `[S2-00047, S2-00193, S3-00812]`, truth: `[S2-00047, S3-00812]`
Precision = 2/3, Recall = 2/2 = 1.0 → F0.5 = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) ≈ **0.714**

### 2.3 What this means for design
1. One wrong ID on an entity with 1 true match drops that entity to about 0.83 or lower. A wrong ID on a singleton drops it to **0**.
2. The decision rule is "add a candidate only if we are confident". A missed match costs less than a false one.
3. The score depends on **each entity**, not each pair. Entities with many matches do not dominate the average.
4. The share of singletons in the data (check it in EDA) sets how conservative the system needs to be.

### 2.4 Leaderboard
- **Public leaderboard:** scored on a subset of the test set during the challenge.
- **Private leaderboard:** scored on the remaining test data after the challenge. It decides the **final ranking**.
- Always submit predictions for the **full** test set.

### 2.5 Hard constraints (submission is rejected or disqualified if violated)
| Rule | Detail |
|---|---|
| Every S1 test entity has **exactly one row** | Missing or duplicate `source1_entity_id` → rejected |
| Empty list for no matches | Leave the field blank |
| No duplicate IDs inside a list | Rejected |
| IDs must be **S2- or S3-** and exist in the test set | No self-matches to S1, no unknown IDs |
| Matches ⊆ candidates | Validator warns otherwise |
| **Final model: MIT/Apache 2.0 licensed, ≤ 8B parameters** | Licence check applies to any pretrained model used |
| **No external lookups** | No geocoding APIs, business registries, entity-resolution services or internet augmentation → **disqualification** |
| Final zip | `output/`, `code/business_entity_resolution/` (`src/`, `README.md`, `requirements.txt`), `Documentation_template.md` |

Validate before every upload:
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

---

## 3. Edge Cases and Related Scenarios

| # | Scenario | Why it matters | Handling |
|---|---|---|---|
| 1 | **Unseen country (France)** | Country-specific rules or one-hot features break at test time | Treat `country` as an open string. Use it only for partitioning and as a same/different flag. Avoid country-specific vocab. Simulate with a hold-one-country-out validation |
| 2 | **Singletons** | Each is worth 1.0 or 0.0 | Conservative threshold; an optional margin rule; track singleton accuracy separately in validation |
| 3 | **One S1 entity → many matches** | Top-1 matching would lose recall | Return every candidate above the threshold, not just the best |
| 4 | **One S2/S3 record claimed by several S1 entities** | Usually means a false merge (S1 is deduplicated) | Verify in train ground truth whether a record maps to only one S1. If yes, assign it to its highest-scoring S1 only |
| 5 | **Generic address or landmark** ("Near SBI ATM") | Creates false matches between unrelated businesses | Frequency-weighted (TF-IDF) similarity; strip landmark phrases from the address view |
| 6 | **Generic name tokens** ("Store", "Traders", "Corp", "Rue") | Same as above | IDF downweighting; legal suffix stripping |
| 7 | **Chain or franchise businesses** (same name, different addresses) | Name matches but the entity differs | Address features must count; require both name and address evidence |
| 8 | **Same address, different businesses** (shopping complex, co-working space) | Address matches but the entity differs | Require name evidence; never match on address alone |
| 9 | **Missing address components** (no PIN, no state) | Address similarity becomes unreliable | Missingness flags as features; fall back on name-heavy evidence; component-aware address comparison |
| 10 | **Transliteration variants** (Indian names, French accents) | Exact token match fails | Accent folding, character n-grams, phonetic keys as a secondary blocker |
| 11 | **Word-order transposition** | Sequence-based metrics fail | Token-set and token-sort similarities |
| 12 | **DBA / trade name differs from legal name** | Names can share little text | Low name score but strong address score is a valid path. The model learns the interaction |
| 13 | **Typos in rare tokens** | Rare tokens carry the most weight | Fuzzy token matching plus character n-grams, not exact rare-token vetoes |
| 14 | **Blocking miss** | Caps recall permanently | Union several blockers; measure candidate recall on train; K about 20 to 30 |
| 15 | **Empty or NaN fields, odd encodings** | Crashes or silent errors | Fill with empty string, Unicode-normalize, explicit tab parsing |
| 16 | **Very large files** (hundreds of MB) | Memory and time | Chunked, sparse, per-country processing; polars or pandas with dtypes; never build a dense N×M matrix |
| 17 | **Distribution shift in test** | Threshold tuned on train may not transfer | Fit TF-IDF on train + test text (unlabeled), use scale-free features, avoid absolute-count features |
| 18 | **Duplicates within S2 or S3** | Many true matches for one S1 entity | Expected: allowed by the task |

---

## 4. Approach Overview

**Idea:** first get every plausible candidate cheaply (high recall), then score each candidate carefully and keep only the confident ones (high precision).

```
 S1 / S2 / S3 raw TSVs
          │
          ▼
 [0] Load + clean + normalize (name view, address view)
          │
          ▼
 [1] Blocking (per country partition):
     TF-IDF char n-gram kNN  ∪  rare-token overlap  ∪  postcode+name-prefix key
          │  → top-K candidates per S1 entity  ──►  candidate_pairs.tsv
          ▼
 [2] Pair feature extraction (name / address / structural / rank features)
          │
          ▼
 [3] Gradient-boosted classifier (LightGBM), P(match)
          │
          ▼
 [4] One-owner assignment (each S2/S3 record → best S1 only), if validated by EDA
          │
          ▼
 [5] F0.5-tuned decision threshold (+ optional margin rule)
          │
          ▼
 [6] Write matching_results.tsv → validate → package
```

---

## 5. Method, Stage by Stage

### Stage 0: Exploratory Data Analysis (do this first)
Answer these on train before designing anything:
1. Row counts per source per country; share of S1 singletons.
2. Distribution of matches per S1 entity (0, 1, 2, 3+; from S2 vs S3).
3. Does any S2/S3 ID appear under more than one S1 entity? (decides Stage 4)
4. 30 to 50 sampled true pairs per country, to see the actual noise types.
5. Most frequent name tokens and address tokens (the generic-token list).
6. Fraction of records with missing PIN/state/city.
7. Text length statistics.

### Stage 1: Normalization
Build two cleaned views per record:
- **Name view:** lowercase → Unicode NFKD accent folding → `&` → `and` → strip punctuation → collapse whitespace → expand abbreviations (pvt → private, ltd → limited, corp → corporation) → **legal-suffix-stripped variant** as a separate field.
- **Address view:** lowercase → accent folding → standardize road/street/floor/etc. tokens → extract PIN/postcode/city if present → **remove landmark phrases** ("near ...", "opp ...", "behind ...") into a separate field rather than deleting silently.

Keep both raw and normalized versions. Normalization rules must be generic. Any abbreviation dictionary should be learned or written from train data patterns, and must not break on French.

### Stage 2: Blocking (candidate generation), goal: recall
Process **within each country** to cut the search space.

| Blocker | How | Catches |
|---|---|---|
| **B1: Char n-gram TF-IDF** | Char 3-grams (or 3 to 4) on name + address, sparse cosine top-K per S1 record | Typos, transliteration, abbreviations |
| **B2: Rare-token overlap** | Inverted index on the rarest name tokens (by IDF) | Word-order shifts, same distinctive word |
| **B3: Postcode / locality + name prefix** | Composite key | Cases where names differ but location is exact |

- Candidate set = **union** of B1, B2 and B3, capped to about K = 20 to 30 per S1 entity per source.
- IDF is fitted on the text of **train + test** (labels are not used, so this is allowed). It adapts to France's own common words without any labels.
- **Success criterion:** candidate recall on the train ground truth (share of true pairs present in candidates). Tune K and blockers until recall is very high while candidates per entity stay manageable.
- This final candidate set is what goes into `candidate_pairs.tsv`.

### Stage 3: Pair Feature Engineering
Compute features separately for **name** and **address**:

| Group | Features |
|---|---|
| Name similarity | Jaro-Winkler, normalized Levenshtein, token-set Jaccard, token-sort ratio, TF-IDF cosine (word and char), suffix-stripped exact match, longest common substring ratio |
| Address similarity | Same family of metrics on the address view; PIN equal / both present / one missing; city equal; landmark-stripped similarity |
| Rare-token evidence | Sum of IDF of shared name tokens; max IDF among shared tokens; fraction of the S1 name's IDF mass covered by the candidate |
| Structural | Length ratios, token count difference, missingness flags |
| Candidate context | Blocking score, **rank** within the S1 entity's candidates, **gap to best/second-best score**, number of candidates above a similarity level |
| Source | Whether candidate is from S2 or S3 (source identity is not country-specific, so it is safe) |

Do **not** use: country one-hot, raw counts that scale with dataset size, or any feature tied to specific US/India strings.

### Stage 4: Matching Model
- **Model:** LightGBM binary classifier (CatBoost is a good alternative), on features above.
- **Training data:** candidates produced by **your own blocker** on train. True pairs = ground truth, everything else = hard negatives. This mirrors test-time inference.
- **Class imbalance:** handle with scale weighting or negative subsampling. Check calibration afterward.
- **Output:** `P(match)` per candidate pair.
- Optional: isotonic calibration on a validation fold so probabilities are more trustworthy for thresholding.

### Stage 5: Consistency and Decision
1. **One-owner assignment** (only if EDA shows a record belongs to at most one S1 entity): for each S2/S3 record, keep it only under the S1 entity where its score is highest.
2. **Threshold** `t`: keep candidates with `P ≥ t`.
3. **Optional margin rule:** for an S1 entity, drop a candidate if a competing candidate from a *different* S1 entity scores close to it (ambiguity → abstain).
4. Anything left uncertain is dropped, and an entity with nothing above `t` becomes a singleton (empty list).

### Stage 6: Threshold Tuning
- Sweep `t` (and margin parameters) on a validation split and compute **macro F0.5 exactly as the PS defines it**, including singletons.
- Pick `t` that maximizes the validation score. Do not pick 0.95 or 0.98 in advance. Choose from data, then check stability with nearby values.
- Report separately: F0.5 on entities with matches, and singleton accuracy.

### Stage 7: Output and Validation
1. Ensure one row per S1 test entity, empty string where no matches, no duplicates.
2. Write `candidate_pairs.tsv` (final scored candidates) and `matching_results.tsv`.
3. Run `utils/validate_submission.py`, and fix any failures before uploading.

---

## 6. Model Layers and Criteria

| Layer | Component | Purpose | Criterion / metric |
|---|---|---|---|
| L0 | Normalization | Remove formatting noise | Manual audit on sampled pairs |
| L1 | Blocking (B1 ∪ B2 ∪ B3) | Recall ceiling | **Candidate recall** on train (maximize) and candidates per entity (keep small) |
| L2 | Feature extractor | Turn a pair into evidence numbers | Feature importance; no country-specific features |
| L3 | LightGBM matcher | Score P(match) | AUC / PR-AUC on hard negatives; calibration |
| L4 | One-owner assignment (+ optional margin rule) | Remove conflicting claims | Precision gain on validation |
| L5 | Threshold `t` | Precision-recall trade-off | **Macro F0.5** on validation (includes singletons) |
| L6 | Format validation | Avoid rejection | Validator returns PASS |

**Acceptance criteria before submitting:**
- Candidate recall on validation is high (reported).
- Macro F0.5 on a held-out S1 split is stable across nearby thresholds.
- Hold-one-country-out score is close to the in-country score (proxy for France).
- Validator prints PASS.

---

## 7. Main Algorithm (Pseudocode)

```text
INPUT: S1, S2, S3 (train or test), ground truth (train only)

# ---- Preprocess ----
for each record r in S1 ∪ S2 ∪ S3:
    r.name_norm, r.name_core = normalize_name(r.business_name)      # core = legal suffix stripped
    r.addr_norm, r.landmark, r.pin = normalize_address(r.business_address)

fit TF-IDF (char n-gram and word) on all records of train + test     # unlabeled → allowed

# ---- Stage 1: Blocking ----
candidates = {}
for country c in countries(S1):
    S1c, S2c, S3c = records with country == c
    for source in (S2c, S3c):
        C1 = topK_cosine(tfidf(S1c), tfidf(source), K)               # sparse kNN, chunked
        C2 = rare_token_matches(S1c, source)
        C3 = postcode_and_prefix_matches(S1c, source)
        candidates[s1] += C1 ∪ C2 ∪ C3     # cap to K per source
save candidates → candidate_pairs.tsv                                # final scored set

# ---- Stage 2 & 3: Features + model ----
X = pair_features(s1, cand) for each (s1, cand) in candidates
if TRAIN:
    y = 1 if cand in ground_truth[s1] else 0
    model = LightGBM.fit(X, y)      # split by S1 entity, never by pair
p = model.predict_proba(X)          # optional: isotonic calibration

# ---- Stage 4: consistency ----
if one_owner_rule_holds_in_train:
    for each candidate record q: keep only argmax over s1 of p(s1, q)

# ---- Stage 5: decision ----
matches[s1] = { q : p(s1, q) >= t and passes_margin_rule }
# entities with empty sets are singletons

# ---- Tune t (on validation) ----
t* = argmax over t of  mean_over_S1( F0.5(matches_t[s1], truth[s1]) )   # singletons: 1.0 iff empty

# ---- Output ----
write matching_results.tsv (one row per S1 entity, comma-joined IDs, blank if none)
run validator
```

**Per-entity scoring function (used in tuning):**
```text
def f05(pred, truth):
    if not truth and not pred: return 1.0
    if not truth and pred:     return 0.0
    if not pred:               return 0.0
    tp = |pred ∩ truth|
    P = tp/|pred|;  R = tp/|truth|
    return 0.0 if tp == 0 else 1.25*P*R / (0.25*P + R)
```

---

## 8. Validation Strategy

1. **Split by S1 entity** (a pair-level split leaks). Use a large enough holdout to include many singletons.
2. **Score with the exact metric**: macro F0.5, with singleton rules.
3. **Hold-one-country-out:** train on US, test on India (and the reverse) as a France proxy.
4. **Blocking audit:** recall of candidates before the model; the model can never recover missed pairs.
5. **Error analysis:** inspect false positives first (they hurt most): generic-address collisions, chain businesses, same-address different-name.
6. **Reproducibility:** fixed seeds, pinned `requirements.txt`, README with exact run order.

---

## 9. Compliance Checklist

- [ ] No external API, registry, geocoder or web lookup anywhere in code
- [ ] All pretrained models (if used) are MIT/Apache 2.0 and ≤ 8B parameters, with licence noted in the docs
- [ ] `country` never hard-coded to {US, India}
- [ ] Every S1 test entity present exactly once; no duplicate IDs; only S2-/S3- IDs
- [ ] `matching_results.tsv` ⊆ `candidate_pairs.tsv`
- [ ] Validator PASS
- [ ] Zip structure matches the required layout; README and pinned requirements included

---

## 10. Future Scope

| Upgrade | What it adds | Note |
|---|---|---|
| **Cross-encoder reranker** (e.g., mDeBERTa-v3 or bge-reranker, licence to be verified) | Better handling of semantic and transliteration cases | Run only on borderline pairs (score near `t`) to control compute |
| **Multilingual embedding blocker/feature** (e.g., multilingual MiniLM, licence to be verified) | Higher recall where character overlap is low | Use as a second blocker or as an added feature |
| **Probability calibration** (isotonic/Platt) | More reliable thresholding across countries | Cheap to add |
| **Cluster-level consistency checks** | Removes contradictions across S2/S3 matches for the same S1 entity | Lighter than a GNN |
| **Graph-based methods (GNN)** | Transitivity and global structure | Likely overkill unless the baseline plateaus |
| **Fine-tuned small LLM judge** (Apache/MIT, ≤ 8B; check licence carefully) | Hardest ambiguous cases | Only on a small slice of pairs |
| **Country-adaptive thresholds** | Handles distribution shift | Only if hold-one-country-out shows a consistent gap |
| **Semi-supervised adaptation on unlabeled test text** | Adapts IDF and features to France | Unlabeled test text only, never external data |

---

## 11. Mapping to `Documentation_template.md`

| Template section | Fill it from |
|---|---|
| Methodology | Sections 4 and 5 |
| Candidate generation / blocking | Section 5, Stage 2 (report candidate recall and average candidates per entity) |
| Model architecture and feature engineering | Section 5, Stages 3 and 4; Section 6 |
| Threshold and precision strategy | Section 5, Stage 6 |
| Generalization to unseen country | Section 3 (#1), Section 8 (#3) |
| Compliance / licences | Section 9 |
| Results and error analysis | Section 8 |
