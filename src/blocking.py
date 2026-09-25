"""
Stage 2: Blocking / Candidate Generation module.
Implements:
- P0-2: Sparse BLAS matrix multiplication with max_df filtering and per-country TF-IDF.
- P1-2: Deterministic candidate ordering (B1 cosine descending -> B2 rare tokens -> B3 locality/house).
- P1-4: Document-frequency-capped rare-token inverted index.
- P1-5: Returns candidate cosine similarities for downstream ranking & model features.
"""

from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class CountryTargetIndex:
    """
    Pre-indexes target records for a country partition to enable ultra-fast sub-second batch querying.
    """
    def __init__(self, df_target: pd.DataFrame, blocker: 'MultiBlocker', max_doc_freq: int = 5000):
        self.target_ids = df_target["entity_id"].values
        self.n_targets = len(df_target)

        # 1. Pre-transform target vectors ONCE (sparse matrix CSR)
        self.target_vecs_T = blocker.tfidf_char.transform(df_target["combined_text"].fillna("")).T.tocsr()

        # 2. Inverted index on rare tokens with document frequency cap (P1-4)
        raw_token_postings = defaultdict(list)
        for idx, name in enumerate(df_target["name_core"].fillna("").astype(str)):
            tokens = set(name.split())
            for tok in tokens:
                if tok in blocker.idf_dict and blocker.idf_dict[tok] > 2.5:
                    raw_token_postings[tok].append(idx)

        # Keep full posting list for truly rare tokens; discard over-frequent tokens
        self.inverted_index = {tok: postings for tok, postings in raw_token_postings.items() if len(postings) <= max_doc_freq}

        # 3. Locality / House number index (P2-3)
        self.locality_index = defaultdict(list)
        house_numbers = df_target["house_number"].values if "house_number" in df_target.columns else [""] * self.n_targets
        for idx, (hn, core) in enumerate(zip(house_numbers, df_target["name_core"].fillna("").astype(str))):
            h = str(hn).strip()
            pre = core[:3].strip().lower()
            if h and len(h) >= 2 and pre:
                self.locality_index[f"{h}_{pre}"].append(idx)

    def query_batch(
        self,
        df_s1: pd.DataFrame,
        blocker: 'MultiBlocker',
        top_k: int = 25
    ) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float]]:
        """
        Queries a batch of S1 records against the pre-indexed country targets.
        Returns:
        - cand_dict: s1_id -> ordered list of candidate target IDs
        - cosine_sims: (s1_id, target_id) -> float cosine similarity
        """
        if df_s1.empty or self.n_targets == 0:
            return {sid: [] for sid in df_s1["entity_id"]}, {}

        s1_ids = df_s1["entity_id"].values
        s1_vecs = blocker.tfidf_char.transform(df_s1["combined_text"].fillna(""))

        # Sparse BLAS dot product across batch
        sim_matrix = s1_vecs.dot(self.target_vecs_T)

        cand_dict = {}
        cosine_sims = {}

        for row_idx, s1_id in enumerate(s1_ids):
            ordered_cands = []
            seen = set()

            # B1: TF-IDF Cosine Top-K (ordered strictly by score descending) (P1-2, P1-5)
            row_data = sim_matrix[row_idx]
            if row_data.nnz > 0:
                col_indices = row_data.indices
                col_values = row_data.data
                if len(col_values) > top_k:
                    top_local = np.argpartition(col_values, -top_k)[-top_k:]
                    sorted_top = top_local[np.argsort(-col_values[top_local])]
                else:
                    sorted_top = np.argsort(-col_values)

                for local_idx in sorted_top:
                    t_idx = col_indices[local_idx]
                    t_id = self.target_ids[t_idx]
                    score = float(col_values[local_idx])
                    if t_id not in seen and score > 0.05:
                        seen.add(t_id)
                        ordered_cands.append(t_id)
                        cosine_sims[(s1_id, t_id)] = score

            # B2: Rare token overlap (ranked by IDF) (P1-4)
            s1_name_core = str(df_s1.iloc[row_idx].get("name_core", "")).strip()
            tokens = set(s1_name_core.split())
            sorted_tokens = sorted([t for t in tokens if t in blocker.idf_dict], key=lambda x: blocker.idf_dict[x], reverse=True)
            for tok in sorted_tokens[:3]:
                for t_idx in self.inverted_index.get(tok, [])[:top_k]:
                    t_id = self.target_ids[t_idx]
                    if t_id not in seen:
                        seen.add(t_id)
                        ordered_cands.append(t_id)
                        cosine_sims[(s1_id, t_id)] = cosine_sims.get((s1_id, t_id), 0.0)

            # B3: House / Locality key (P2-3)
            hn = str(df_s1.iloc[row_idx].get("house_number", "")).strip()
            pre = s1_name_core[:3].strip().lower()
            if hn and len(hn) >= 2 and pre:
                for t_idx in self.locality_index.get(f"{hn}_{pre}", []):
                    t_id = self.target_ids[t_idx]
                    if t_id not in seen:
                        seen.add(t_id)
                        ordered_cands.append(t_id)
                        cosine_sims[(s1_id, t_id)] = cosine_sims.get((s1_id, t_id), 0.0)

            # Truncate deterministically to top_k * 2
            cand_dict[s1_id] = ordered_cands[: top_k * 2]

        return cand_dict, cosine_sims


class MultiBlocker:
    def __init__(self, top_k_per_source: int = 25):
        self.top_k_per_source = top_k_per_source
        # max_df drops hyper-common n-grams (e.g. 'road', 'ltd', 'inc') that create dense collisions (P0-2)
        self.tfidf_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=2,
            max_df=0.20,
            sublinear_tf=True
        )
        self.tfidf_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1),
            min_df=2,
            token_pattern=r"(?u)\b\w+\b"
        )
        self.is_fitted = False
        self.idf_dict = {}
        self.max_idf = 10.0

    def fit_vectorizers(self, all_texts: List[str]):
        """
        Fits TF-IDF vectorizers on unlabeled corpus.
        """
        valid_texts = [t if isinstance(t, str) and t.strip() else "missing" for t in all_texts]
        self.tfidf_char.fit(valid_texts)
        self.tfidf_word.fit(valid_texts)
        self.is_fitted = True

        feature_names = self.tfidf_word.get_feature_names_out()
        self.idf_dict = dict(zip(feature_names, self.tfidf_word.idf_))
        self.max_idf = float(np.max(self.tfidf_word.idf_)) if len(self.tfidf_word.idf_) > 0 else 10.0

    def generate_candidates(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame
    ) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float]]:
        """
        Executes fast candidate generation across country partitions.
        """
        if not self.is_fitted:
            raise ValueError("Blocker TF-IDF vectorizers must be fitted before running generate_candidates!")

        all_candidates = defaultdict(list)
        all_cosine_sims = {}
        all_s1_ids = set(df_s1["entity_id"])

        col_name = "country_clean" if "country_clean" in df_s1.columns else "country"
        countries = df_s1[col_name].unique()

        for country in countries:
            s1_c = df_s1[df_s1[col_name] == country]
            s2_c = df_s2[df_s2[col_name] == country]
            s3_c = df_s3[df_s3[col_name] == country]

            targets_c = pd.concat([s2_c, s3_c], ignore_index=True)
            if targets_c.empty or s1_c.empty:
                continue

            target_index = CountryTargetIndex(targets_c, self)
            cands_c, cos_c = target_index.query_batch(s1_c, self, top_k=self.top_k_per_source)

            for s1_id, cand_list in cands_c.items():
                all_candidates[s1_id].extend(cand_list)
            all_cosine_sims.update(cos_c)

        result = {}
        for s1_id in all_s1_ids:
            # Deduplicate preserving order
            cands = []
            seen = set()
            for c in all_candidates.get(s1_id, []):
                if c not in seen:
                    seen.add(c)
                    cands.append(c)
            result[s1_id] = cands

        return result, all_cosine_sims


def evaluate_blocking_recall(
    candidates_dict: Dict[str, List[str]],
    ground_truth_dict: Dict[str, List[str]]
) -> Dict[str, float]:
    total_true_pairs = 0
    captured_pairs = 0
    candidate_counts = []

    for s1_id, true_list in ground_truth_dict.items():
        true_set = {x.strip() for x in true_list if x.strip()}
        cand_set = set(candidates_dict.get(s1_id, []))
        candidate_counts.append(len(cand_set))

        total_true_pairs += len(true_set)
        captured_pairs += len(true_set.intersection(cand_set))

    recall = captured_pairs / total_true_pairs if total_true_pairs > 0 else 1.0
    avg_cands = float(np.mean(candidate_counts)) if candidate_counts else 0.0

    return {
        "candidate_recall": recall,
        "captured_pairs": captured_pairs,
        "total_true_pairs": total_true_pairs,
        "avg_candidates_per_s1": avg_cands,
        "max_candidates": max(candidate_counts) if candidate_counts else 0,
        "min_candidates": min(candidate_counts) if candidate_counts else 0,
    }
