"""
Stage 3: Pair Feature Engineering module.
Extracts deep pairwise similarity, rare-token mass, structural, and rank-context features.
Engineered strictly without country one-hot encoding or dataset-scale features for maximum generalization.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler


def longest_common_substring_len(s1: str, s2: str) -> int:
    if not s1 or not s2:
        return 0
    m, n = len(s1), len(s2)
    # Use 1D array DP for speed
    prev = [0] * (n + 1)
    max_len = 0
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > max_len:
                    max_len = curr[j]
            else:
                curr[j] = 0
        prev = curr
    return max_len


def jaccard_similarity(tokens1: set, tokens2: set) -> float:
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0


def char_ngrams(text: str, n: int = 3) -> set:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


class PairFeatureExtractor:
    def __init__(self, idf_dict: Dict[str, float] = None):
        self.idf_dict = idf_dict or {}

    def extract_pair_features(
        self,
        r1: Dict[str, Any],
        r2: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extracts comprehensive pairwise similarity metrics between S1 and target record.
        """
        s1_name_norm = str(r1.get("name_norm", ""))
        s2_name_norm = str(r2.get("name_norm", ""))
        s1_name_core = str(r1.get("name_core", ""))
        s2_name_core = str(r2.get("name_core", ""))

        s1_addr_norm = str(r1.get("addr_norm", ""))
        s2_addr_norm = str(r2.get("addr_norm", ""))
        s1_postal = str(r1.get("postal_code", "")).strip()
        s2_postal = str(r2.get("postal_code", "")).strip()

        target_id = str(r2.get("entity_id", ""))

        # ----------------------------------------------------
        # 1. Name Similarities
        # ----------------------------------------------------
        name_jw = JaroWinkler.similarity(s1_name_norm, s2_name_norm)
        name_core_jw = JaroWinkler.similarity(s1_name_core, s2_name_core)
        name_lev = Levenshtein.normalized_similarity(s1_name_norm, s2_name_norm)
        name_token_set = fuzz.token_set_ratio(s1_name_norm, s2_name_norm) / 100.0
        name_token_sort = fuzz.token_sort_ratio(s1_name_norm, s2_name_norm)
        name_partial = fuzz.partial_ratio(s1_name_norm, s2_name_norm) / 100.0
        name_core_exact = 1.0 if s1_name_core and s1_name_core == s2_name_core else 0.0

        s1_name_tokens = set(s1_name_norm.split())
        s2_name_tokens = set(s2_name_norm.split())
        name_word_jaccard = jaccard_similarity(s1_name_tokens, s2_name_tokens)
        name_char_jaccard = jaccard_similarity(char_ngrams(s1_name_norm, 3), char_ngrams(s2_name_norm, 3))

        lcs_len = longest_common_substring_len(s1_name_norm, s2_name_norm)
        max_len = max(len(s1_name_norm), len(s2_name_norm), 1)
        name_lcs_ratio = lcs_len / max_len

        # ----------------------------------------------------
        # 2. Address Similarities
        # ----------------------------------------------------
        addr_jw = JaroWinkler.similarity(s1_addr_norm, s2_addr_norm)
        addr_lev = Levenshtein.normalized_similarity(s1_addr_norm, s2_addr_norm)
        addr_token_set = fuzz.token_set_ratio(s1_addr_norm, s2_addr_norm) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(s1_addr_norm, s2_addr_norm)

        s1_addr_tokens = set(s1_addr_norm.split())
        s2_addr_tokens = set(s2_addr_norm.split())
        addr_word_jaccard = jaccard_similarity(s1_addr_tokens, s2_addr_tokens)
        addr_char_jaccard = jaccard_similarity(char_ngrams(s1_addr_norm, 3), char_ngrams(s2_addr_norm, 3))

        # Postal code comparison
        if s1_postal and s2_postal:
            postal_match = 1.0 if s1_postal == s2_postal else -1.0
            postal_missing = 0.0
        else:
            postal_match = 0.0
            postal_missing = 1.0

        # ----------------------------------------------------
        # 3. Rare Token / IDF Mass Evidence
        # ----------------------------------------------------
        shared_tokens = s1_name_tokens.intersection(s2_name_tokens)
        shared_idfs = [self.idf_dict.get(tok, 1.0) for tok in shared_tokens]
        s1_all_idfs = [self.idf_dict.get(tok, 1.0) for tok in s1_name_tokens]

        sum_shared_idf = sum(shared_idfs)
        max_shared_idf = max(shared_idfs) if shared_idfs else 0.0
        total_s1_idf = sum(s1_all_idfs) if s1_all_idfs else 1.0
        idf_coverage_ratio = sum_shared_idf / total_s1_idf

        # ----------------------------------------------------
        # 4. Structural & Length Ratios
        # ----------------------------------------------------
        len_s1_n, len_s2_n = len(s1_name_norm), len(s2_name_norm)
        name_len_ratio = min(len_s1_n, len_s2_n) / max(len_s1_n, len_s2_n, 1)

        len_s1_a, len_s2_a = len(s1_addr_norm), len(s2_addr_norm)
        addr_len_ratio = min(len_s1_a, len_s2_a) / max(len_s1_a, len_s2_a, 1)

        token_cnt_diff_name = abs(len(s1_name_tokens) - len(s2_name_tokens))
        token_cnt_diff_addr = abs(len(s1_addr_tokens) - len(s2_addr_tokens))

        # ----------------------------------------------------
        # 5. Combined / Interaction Features
        # ----------------------------------------------------
        name_addr_geometric_mean = np.sqrt(max(0.0, name_jw * addr_jw))
        name_addr_min = min(name_jw, addr_jw)
        is_source_2 = 1.0 if target_id.startswith("S2-") else 0.0

        features = {
            "name_jw": name_jw,
            "name_core_jw": name_core_jw,
            "name_lev": name_lev,
            "name_token_set": name_token_set,
            "name_token_sort": name_token_sort / 100.0,
            "name_partial": name_partial,
            "name_core_exact": name_core_exact,
            "name_word_jaccard": name_word_jaccard,
            "name_char_jaccard": name_char_jaccard,
            "name_lcs_ratio": name_lcs_ratio,
            "addr_jw": addr_jw,
            "addr_lev": addr_lev,
            "addr_token_set": addr_token_set,
            "addr_token_sort": addr_token_sort / 100.0,
            "addr_word_jaccard": addr_word_jaccard,
            "addr_char_jaccard": addr_char_jaccard,
            "postal_match": postal_match,
            "postal_missing": postal_missing,
            "sum_shared_idf": sum_shared_idf,
            "max_shared_idf": max_shared_idf,
            "idf_coverage_ratio": idf_coverage_ratio,
            "name_len_ratio": name_len_ratio,
            "addr_len_ratio": addr_len_ratio,
            "token_cnt_diff_name": float(token_cnt_diff_name),
            "token_cnt_diff_addr": float(token_cnt_diff_addr),
            "name_addr_geometric_mean": name_addr_geometric_mean,
            "name_addr_min": name_addr_min,
            "is_source_2": is_source_2,
        }

        return features

    def build_feature_table(
        self,
        df_s1: pd.DataFrame,
        df_target: pd.DataFrame,
        candidates_dict: Dict[str, List[str]],
        ground_truth_dict: Dict[str, List[str]] = None
    ) -> pd.DataFrame:
        """
        Builds complete pairwise feature DataFrame with candidate context and rank features.
        """
        s1_map = df_s1.set_index("entity_id").to_dict(orient="index")
        target_map = df_target.set_index("entity_id").to_dict(orient="index")

        rows = []

        for s1_id, cand_ids in candidates_dict.items():
            if s1_id not in s1_map or not cand_ids:
                continue

            r1 = s1_map[s1_id]
            s1_cand_rows = []

            # Determine true matches if ground truth provided
            true_set = set(ground_truth_dict.get(s1_id, [])) if ground_truth_dict is not None else set()

            for cand_id in cand_ids:
                if cand_id not in target_map:
                    continue
                r2 = target_map[cand_id]

                feat = self.extract_pair_features(r1, r2)
                feat["source1_entity_id"] = s1_id
                feat["candidate_entity_id"] = cand_id

                if ground_truth_dict is not None:
                    feat["label"] = 1 if cand_id in true_set else 0

                s1_cand_rows.append(feat)

            if not s1_cand_rows:
                continue

            # Compute candidate pool context & rank features per S1 entity
            cand_df = pd.DataFrame(s1_cand_rows)
            # Base composite score to determine ranking
            composite_score = (
                cand_df["name_jw"] * 0.4 +
                cand_df["name_token_set"] * 0.2 +
                cand_df["addr_jw"] * 0.3 +
                cand_df["idf_coverage_ratio"] * 0.1
            )
            cand_df["composite_score"] = composite_score
            cand_df = cand_df.sort_values(by="composite_score", ascending=False).reset_index(drop=True)

            num_cands = len(cand_df)
            best_score = cand_df["composite_score"].iloc[0]
            second_best_score = cand_df["composite_score"].iloc[1] if num_cands > 1 else 0.0

            cand_df["cand_rank"] = np.arange(1, num_cands + 1)
            cand_df["cand_pool_size"] = num_cands

            # Gap to best/second-best
            gaps = []
            for rank_val, comp_val in zip(cand_df["cand_rank"], cand_df["composite_score"]):
                if rank_val == 1:
                    gaps.append(comp_val - second_best_score)
                else:
                    gaps.append(best_score - comp_val)
            cand_df["cand_score_gap"] = gaps

            rows.append(cand_df)

        if not rows:
            return pd.DataFrame()

        full_df = pd.concat(rows, ignore_index=True)
        return full_df
