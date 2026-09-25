"""
Stage 2: Blocking / Candidate Generation module.
Combines multiple blocking strategies to achieve maximum recall:
- B1: Character n-gram TF-IDF Sparse Cosine kNN (catches typos, transliterations, variations)
- B2: Rare-token Inverted Index (catches word-order shifts and distinctive tokens)
- B3: Locality/Postal-code + Name Prefix Composite Key
Operates per-country partition with CountryTargetIndex for high-throughput vectorized querying.
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class CountryTargetIndex:
    """
    Pre-indexes target records for a country partition to enable ultra-fast sub-second batch querying.
    """
    def __init__(self, df_target: pd.DataFrame, blocker: 'MultiBlocker'):
        self.target_ids = df_target["entity_id"].values
        self.n_targets = len(df_target)

        # 1. Pre-transform target vectors ONCE (sparse matrix transpose)
        self.target_vecs_T = blocker.tfidf_char.transform(df_target["combined_text"].fillna("")).T.tocsr()

        # 2. Inverted index on rare tokens
        self.inverted_index = defaultdict(list)
        for idx, name in enumerate(df_target["name_core"].fillna("").astype(str)):
            tokens = set(name.split())
            for tok in tokens:
                if tok in blocker.idf_dict and blocker.idf_dict[tok] > 2.0:
                    if len(self.inverted_index[tok]) < 1000:  # Skip over-saturated tokens
                        self.inverted_index[tok].append(idx)

        # 3. Locality / Postal prefix index
        self.locality_index = defaultdict(list)
        postal_codes = df_target["postal_code"].values if "postal_code" in df_target.columns else [""] * self.n_targets
        for idx, (postal, core) in enumerate(zip(postal_codes, df_target["name_core"].fillna("").astype(str))):
            p = str(postal).strip()
            pre = core[:3].strip().lower()
            if p and len(p) >= 4 and pre:
                self.locality_index[f"{p}_{pre}"].append(idx)

    def query_batch(self, df_s1: pd.DataFrame, blocker: 'MultiBlocker', top_k: int = 25) -> Dict[str, List[str]]:
        if df_s1.empty or self.n_targets == 0:
            return {sid: [] for sid in df_s1["entity_id"]}

        s1_ids = df_s1["entity_id"].values
        s1_vecs = blocker.tfidf_char.transform(df_s1["combined_text"].fillna(""))

        # Sparse BLAS dot product across batch
        sim_matrix = s1_vecs.dot(self.target_vecs_T)

        result = {}
        for row_idx, s1_id in enumerate(s1_ids):
            cands = set()

            # B1: TF-IDF Cosine top-K
            row_data = sim_matrix[row_idx]
            if row_data.nnz > 0:
                col_indices = row_data.indices
                col_values = row_data.data
                if len(col_values) > top_k:
                    top_local = np.argpartition(col_values, -top_k)[-top_k:]
                    top_target_indices = col_indices[top_local]
                else:
                    top_target_indices = col_indices
                for t_idx in top_target_indices:
                    cands.add(self.target_ids[t_idx])

            # B2: Rare tokens
            s1_name_core = str(df_s1.iloc[row_idx].get("name_core", "")).strip()
            tokens = set(s1_name_core.split())
            sorted_tokens = sorted([t for t in tokens if t in blocker.idf_dict], key=lambda x: blocker.idf_dict[x], reverse=True)
            for tok in sorted_tokens[:2]:
                for t_idx in self.inverted_index.get(tok, [])[:top_k]:
                    cands.add(self.target_ids[t_idx])

            # B3: Locality prefix
            postal = str(df_s1.iloc[row_idx].get("postal_code", "")).strip()
            pre = s1_name_core[:3].strip().lower()
            if postal and len(postal) >= 4 and pre:
                for t_idx in self.locality_index.get(f"{postal}_{pre}", []):
                    cands.add(self.target_ids[t_idx])

            result[s1_id] = list(cands)[: top_k * 2]

        return result


class MultiBlocker:
    def __init__(self, top_k_per_source: int = 25):
        self.top_k_per_source = top_k_per_source
        self.tfidf_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=2,
            max_features=50000,
            sublinear_tf=True
        )
        self.tfidf_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1),
            min_df=2,
            max_features=50000,
            token_pattern=r"(?u)\b\w+\b"
        )
        self.is_fitted = False

    def fit_vectorizers(self, all_texts: List[str]):
        """
        Fits TF-IDF vectorizers on unlabeled corpus (train + test text).
        """
        valid_texts = [t if isinstance(t, str) and t.strip() else "missing" for t in all_texts]
        self.tfidf_char.fit(valid_texts)
        self.tfidf_word.fit(valid_texts)
        self.is_fitted = True

        # Build vocabulary IDF lookup for rare token blocker
        feature_names = self.tfidf_word.get_feature_names_out()
        self.idf_dict = dict(zip(feature_names, self.tfidf_word.idf_))

    def generate_candidates(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Executes fast candidate generation across country partitions using CountryTargetIndex.
        """
        if not self.is_fitted:
            raise ValueError("Blocker TF-IDF vectorizers must be fitted before running generate_candidates!")

        all_candidates = defaultdict(set)
        all_s1_ids = set(df_s1["entity_id"])

        countries = df_s1["country_clean"].unique() if "country_clean" in df_s1.columns else df_s1["country"].unique()

        for country in countries:
            col_name = "country_clean" if "country_clean" in df_s1.columns else "country"
            s1_c = df_s1[df_s1[col_name] == country]
            s2_c = df_s2[df_s2[col_name] == country]
            s3_c = df_s3[df_s3[col_name] == country]

            targets_c = pd.concat([s2_c, s3_c], ignore_index=True)
            if targets_c.empty or s1_c.empty:
                continue

            target_index = CountryTargetIndex(targets_c, self)
            cands_c = target_index.query_batch(s1_c, self, top_k=self.top_k_per_source)

            for s1_id, cand_list in cands_c.items():
                all_candidates[s1_id].update(cand_list)

        result = {}
        for s1_id in all_s1_ids:
            result[s1_id] = sorted(list(all_candidates.get(s1_id, set())))

        return result


def evaluate_blocking_recall(
    candidates_dict: Dict[str, List[str]],
    ground_truth_dict: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Measures blocking candidate recall on ground truth.
    """
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
