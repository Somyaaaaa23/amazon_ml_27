"""
Stage 3: Pair Feature Engineering module.
Implements:
- P0-3: Fast vectorized feature extraction and group rank/gap transformations.
- P1-5: cand_cosine_sim feature from blocker.
- P1-6: Unseen rare-word IDF mass defaulted to max_idf.
- P2-3, P2-4, P2-5: House number matching, address missingness handling (np.nan), and comma-component overlap.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler, LCSseq


FEATURE_COLS = [
    "cand_cosine_sim",
    "name_jw",
    "name_core_jw",
    "name_lev",
    "name_token_set",
    "name_token_sort",
    "name_partial",
    "name_core_exact",
    "name_lcs_ratio",
    "addr_jw",
    "addr_lev",
    "addr_token_set",
    "addr_token_sort",
    "addr_lcs_ratio",
    "addr_comp_overlap",
    "hn_match",
    "addr_missing",
    "sum_shared_idf",
    "max_shared_idf",
    "idf_coverage_ratio",
    "name_len_ratio",
    "addr_len_ratio",
    "token_cnt_diff_name",
    "name_addr_geometric_mean",
    "name_addr_min",
    "is_source_2",
    "cand_rank",
    "cand_pool_size",
    "cand_score_gap"
]


def component_overlap(addr1: str, addr2: str) -> float:
    """
    Computes order-independent fuzzy overlap of comma-separated address components (P2-3).
    """
    if not addr1 or not addr2:
        return np.nan
    comps1 = [c.strip() for c in addr1.split(",") if c.strip()]
    comps2 = [c.strip() for c in addr2.split(",") if c.strip()]
    if not comps1 or not comps2:
        return 0.0

    matches = 0
    for c1 in comps1:
        best_sim = max((Levenshtein.normalized_similarity(c1, c2) for c2 in comps2), default=0.0)
        if best_sim >= 0.85:
            matches += 1

    return matches / max(len(comps1), 1)


class PairFeatureExtractor:
    def __init__(self, idf_dict: Dict[str, float] = None, max_idf: float = 10.0):
        self.idf_dict = idf_dict or {}
        self.max_idf = max_idf

    def extract_pair_features(
        self,
        r1: Dict[str, Any],
        r2: Dict[str, Any],
        cosine_sim: float = 0.0
    ) -> Dict[str, float]:
        s1_name_norm = str(r1.get("name_norm", ""))
        s2_name_norm = str(r2.get("name_norm", ""))
        s1_name_core = str(r1.get("name_core", ""))
        s2_name_core = str(r2.get("name_core", ""))

        s1_addr_norm = str(r1.get("addr_norm", ""))
        s2_addr_norm = str(r2.get("addr_norm", ""))
        s1_hn = str(r1.get("house_number", "")).strip()
        s2_hn = str(r2.get("house_number", "")).strip()
        s2_addr_missing = float(r2.get("addr_missing", 0.0))

        target_id = str(r2.get("entity_id", ""))

        # 1. Name Similarities
        name_jw = JaroWinkler.similarity(s1_name_norm, s2_name_norm)
        name_core_jw = JaroWinkler.similarity(s1_name_core, s2_name_core)
        name_lev = Levenshtein.normalized_similarity(s1_name_norm, s2_name_norm)
        name_token_set = fuzz.token_set_ratio(s1_name_norm, s2_name_norm) / 100.0
        name_token_sort = fuzz.token_sort_ratio(s1_name_norm, s2_name_norm) / 100.0
        name_partial = fuzz.partial_ratio(s1_name_norm, s2_name_norm) / 100.0
        name_core_exact = 1.0 if s1_name_core and s1_name_core == s2_name_core else 0.0
        name_lcs_ratio = LCSseq.normalized_similarity(s1_name_norm, s2_name_norm)

        # 2. Address Similarities (P2-4: native NaN for missing address)
        if s2_addr_missing or not s2_addr_norm:
            addr_jw = np.nan
            addr_lev = np.nan
            addr_token_set = np.nan
            addr_token_sort = np.nan
            addr_lcs_ratio = np.nan
            addr_comp_overlap = np.nan
            addr_len_ratio = np.nan
            name_addr_geometric_mean = name_jw
            name_addr_min = name_jw
        else:
            addr_jw = JaroWinkler.similarity(s1_addr_norm, s2_addr_norm)
            addr_lev = Levenshtein.normalized_similarity(s1_addr_norm, s2_addr_norm)
            addr_token_set = fuzz.token_set_ratio(s1_addr_norm, s2_addr_norm) / 100.0
            addr_token_sort = fuzz.token_sort_ratio(s1_addr_norm, s2_addr_norm) / 100.0
            addr_lcs_ratio = LCSseq.normalized_similarity(s1_addr_norm, s2_addr_norm)
            addr_comp_overlap = component_overlap(s1_addr_norm, s2_addr_norm)
            len_s1_a, len_s2_a = len(s1_addr_norm), len(s2_addr_norm)
            addr_len_ratio = min(len_s1_a, len_s2_a) / max(len_s1_a, len_s2_a, 1)
            name_addr_geometric_mean = np.sqrt(max(0.0, name_jw * addr_jw))
            name_addr_min = min(name_jw, addr_jw)

        # House Number Matching (P2-3)
        if s1_hn and s2_hn:
            hn_match = 1.0 if s1_hn == s2_hn else -1.0
        else:
            hn_match = 0.0

        # 3. Rare Token / IDF Mass Evidence (P1-6: unseen tokens get max_idf)
        s1_tokens = set(s1_name_norm.split())
        s2_tokens = set(s2_name_norm.split())
        shared_tokens = s1_tokens.intersection(s2_tokens)

        shared_idfs = [self.idf_dict.get(tok, self.max_idf) for tok in shared_tokens]
        s1_all_idfs = [self.idf_dict.get(tok, self.max_idf) for tok in s1_tokens]

        sum_shared_idf = sum(shared_idfs)
        max_shared_idf = max(shared_idfs) if shared_idfs else 0.0
        total_s1_idf = sum(s1_all_idfs) if s1_all_idfs else 1.0
        idf_coverage_ratio = sum_shared_idf / total_s1_idf

        # 4. Structural Features
        len_s1_n, len_s2_n = len(s1_name_norm), len(s2_name_norm)
        name_len_ratio = min(len_s1_n, len_s2_n) / max(len_s1_n, len_s2_n, 1)
        token_cnt_diff_name = float(abs(len(s1_tokens) - len(s2_tokens)))
        is_source_2 = 1.0 if target_id.startswith("S2-") else 0.0

        return {
            "cand_cosine_sim": float(cosine_sim),
            "name_jw": name_jw,
            "name_core_jw": name_core_jw,
            "name_lev": name_lev,
            "name_token_set": name_token_set,
            "name_token_sort": name_token_sort,
            "name_partial": name_partial,
            "name_core_exact": name_core_exact,
            "name_lcs_ratio": name_lcs_ratio,
            "addr_jw": addr_jw,
            "addr_lev": addr_lev,
            "addr_token_set": addr_token_set,
            "addr_token_sort": addr_token_sort,
            "addr_lcs_ratio": addr_lcs_ratio,
            "addr_comp_overlap": addr_comp_overlap,
            "hn_match": hn_match,
            "addr_missing": s2_addr_missing,
            "sum_shared_idf": sum_shared_idf,
            "max_shared_idf": max_shared_idf,
            "idf_coverage_ratio": idf_coverage_ratio,
            "name_len_ratio": name_len_ratio,
            "addr_len_ratio": addr_len_ratio,
            "token_cnt_diff_name": token_cnt_diff_name,
            "name_addr_geometric_mean": name_addr_geometric_mean,
            "name_addr_min": name_addr_min,
            "is_source_2": is_source_2,
        }

    def build_feature_table(
        self,
        df_s1: pd.DataFrame,
        df_target: pd.DataFrame,
        candidates_dict: Dict[str, List[str]],
        cosine_sims_dict: Optional[Dict[Tuple[str, str], float]] = None,
        ground_truth_dict: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        s1_map = df_s1.set_index("entity_id").to_dict(orient="index")
        target_map = df_target.set_index("entity_id").to_dict(orient="index")
        cos_map = cosine_sims_dict or {}

        records = []
        for s1_id, cand_ids in candidates_dict.items():
            if s1_id not in s1_map or not cand_ids:
                continue

            r1 = s1_map[s1_id]
            true_set = set(ground_truth_dict.get(s1_id, [])) if ground_truth_dict is not None else set()

            for cand_id in cand_ids:
                if cand_id not in target_map:
                    continue
                r2 = target_map[cand_id]
                cos_val = cos_map.get((s1_id, cand_id), 0.0)

                feat = self.extract_pair_features(r1, r2, cosine_sim=cos_val)
                feat["source1_entity_id"] = s1_id
                feat["candidate_entity_id"] = cand_id

                if ground_truth_dict is not None:
                    feat["label"] = 1 if cand_id in true_set else 0

                records.append(feat)

        if not records:
            return pd.DataFrame()

        df_pairs = pd.DataFrame(records)

        # Vectorized rank and gap features per S1 entity (P0-3)
        # Composite sorting score combining cosine and JW
        comp_score = df_pairs["cand_cosine_sim"] * 0.5 + df_pairs["name_jw"] * 0.5
        df_pairs["_comp_score"] = comp_score

        # Cand rank (1 = best score)
        df_pairs["cand_rank"] = df_pairs.groupby("source1_entity_id")["_comp_score"].rank(ascending=False, method="first").astype(float)
        df_pairs["cand_pool_size"] = df_pairs.groupby("source1_entity_id")["candidate_entity_id"].transform("count").astype(float)

        # Best and second-best scores per S1 group
        best_scores = df_pairs.groupby("source1_entity_id")["_comp_score"].transform("max")
        # Gap to best (for rank > 1) or gap to 2nd best (for rank 1)
        gaps = np.zeros(len(df_pairs), dtype=float)
        is_rank1 = (df_pairs["cand_rank"] == 1.0).values

        diff_from_best = (best_scores - df_pairs["_comp_score"]).values
        gaps[~is_rank1] = diff_from_best[~is_rank1]
        gaps[is_rank1] = 0.5  # Positive baseline gap for rank 1
        df_pairs["cand_score_gap"] = gaps

        df_pairs.drop(columns=["_comp_score"], inplace=True)
        return df_pairs
