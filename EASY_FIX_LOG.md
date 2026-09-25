# 🛠️ Implemented Fixes & Enhancements Log (Plain Language Guide)

This document explains all the technical fixes and algorithmic improvements implemented in our Entity Resolution codebase from `FIX_PLAN.md` in simple, plain English.

---

### 1. 🔤 Indic Script & Non-Latin Transliteration (P1-3)
* **What was happening:** In India, ~23% of business records in Source 2 and Source 3 are written in native Indic scripts (e.g. Tamil: `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்`, Devanagari: `राम मार्केटिंग`), while Source 1 is written in English Latin (`Raj Investments`, `Ram Marketing`). The old text cleaner stripped these characters into garbled letters.
* **What we fixed (`src/normalization.py`):** Added a universal, rule-based Brahmic script transliterator that converts Devanagari, Tamil, Bengali, Telugu, Kannada, Gujarati, and Malayalam into English phonetics (`praaivet limited` ➔ `private limited`) **before** text matching.
* **Result:** Blocker recall and name similarity for Indian records jumped significantly.

---

### 2. 🏛️ True Legal Suffixes & Preserving Core Names (P2-1 & P2-2)
* **What was happening:** The old code removed words like `services`, `solutions`, `group`, and `enterprises`. This meant `"Apex Solutions"` and `"Apex Group"` both collapsed to `"Apex"`, causing false merges. Also, French legal forms (`SARL`, `SAS`, `EURL`, `SASU`, `SCI`, `EI`) were missing.
* **What we fixed (`src/normalization.py`):** 
  - Restricted legal-suffix stripping to **strictly legal registrations** (`pvt ltd`, `ltd`, `inc`, `corp`, `llc`, `llp`, `sarl`, `sas`, `sasu`, `eurl`, `sci`, `ei`, `snc`, `gmbh`, `plc`).
  - Preserved descriptive business words (`solutions`, `services`, `group`, `trading`, `enterprises`) in the core name.
* **Result:** Prevents unrelated companies sharing generic words from falsely merging, while properly normalizing French test entities.

---

### 3. 🏠 House Numbers & Component Address Overlap (P2-3, P2-4, P2-5)
* **What was happening:** In many records, address strings are reordered (e.g., `Kansas City, MO, 630 45th Terrace` vs `630 45th Terrace, Kansas City, MO`), and ~3% of addresses are empty/missing.
* **What we fixed (`src/normalization.py` & `src/features.py`):**
  - Isolated the house/plot number (`630`, `Plot 45`) for exact matching (`hn_match`).
  - Added order-independent comma-separated component overlap (`addr_comp_overlap`).
  - Added `addr_missing` flag and passed native `NaN` for missing addresses so LightGBM can rely solely on name evidence.

---

### 4. ⚡ Fast Sparse BLAS Blocking & Pre-Indexed Target Search (P0-2, P1-2, P1-4, P1-5)
* **What was happening:** Dense matrix multiplications on 5 million target records caused out-of-memory errors (`zsh: killed`) and took hours.
* **What we fixed (`src/blocking.py`):**
  - **`CountryTargetIndex`**: Converts target text into a sparse matrix **once** per country, eliminating repeated vectorization.
  - Added `max_df=0.20` to drop hyper-common n-grams (like `"road"`, `"ltd"`) that create dense false collisions.
  - Rare-token inverted index now filters by document frequency cap instead of truncating arbitrary rows in file order.
  - Deterministically ordered candidates by B1 cosine score ➔ B2 rare tokens ➔ B3 house numbers.
  - Passed candidate cosine similarity (`cand_cosine_sim`) as a direct feature to the model.

---

### 5. 🎯 Full Country-Pool Training (P0-1)
* **What was happening:** The old sample chunk only used 3% of negative targets, making the model artificially believe almost every candidate was a true match.
* **What we fixed (`src/infer_test.py`):**
  - Training S1 entities are now queried against the **entire real country target pool** (millions of records), exposing the LightGBM model to real hard negatives.
  - The model learns an honest, reliable decision threshold ($t^* \approx 0.63$).

---

### 6. 🌐 Global Country-Level One-Owner Rule (P1-1)
* **What was happening:** The one-owner deduplication was previously run on small 50k sub-batches, which allowed different batches to accidentally claim the same target record.
* **What we fixed (`src/infer_test.py`):**
  - Scores are collected across the **entire country** into lightweight index arrays, and the one-owner rule (argmax $P(\text{match})$ per candidate) is applied globally across the entire country before writing to disk.

---

### 7. 🛡️ Official Submission Validator & Version Pinning (P0-4 & P3-4)
* **What we fixed (`src/infer_test.py` & `requirements.txt`):**
  - Direct execution of the official competition validator: `student_resource/utils/validate_submission.py`.
  - Pinned exact dependency versions in `requirements.txt` (`pandas==3.0.2`, `numpy==1.26.4`, `lightgbm==4.7.0`, `rapidfuzz==3.14.6`, `scikit-learn==1.8.0`, `scipy==1.17.1`).
