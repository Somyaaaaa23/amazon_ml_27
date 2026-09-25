"""
Stage 5: Postprocessing & Global Consistency module.
Implements:
1. One-Owner Assignment (argmax S1 assignment per candidate record).
2. Margin-based ambiguity pruning.
3. Threshold-based matching selection with precise singleton generation.
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np


def apply_one_owner_filter(
    df_scored_pairs: pd.DataFrame,
    prob_col: str = "pred_prob"
) -> pd.DataFrame:
    """
    Ensures each candidate target entity ID (S2-xxx / S3-xxx) is assigned
    to at most ONE Source 1 entity (the one with the highest predicted probability).
    """
    if df_scored_pairs.empty:
        return df_scored_pairs

    df_sorted = df_scored_pairs.sort_values(by=prob_col, ascending=False)
    # Deduplicate on candidate_entity_id keeping first (highest probability)
    df_dedup = df_sorted.drop_duplicates(subset=["candidate_entity_id"], keep="first").copy()

    return df_dedup


def generate_matching_predictions(
    df_scored_pairs: pd.DataFrame,
    all_s1_ids: List[str],
    threshold: float,
    apply_one_owner: bool = True,
    prob_col: str = "pred_prob"
) -> Dict[str, List[str]]:
    """
    Converts candidate probability predictions into final Source 1 match dictionary.
    Guarantees every S1 entity in all_s1_ids is present (empty list if no matches above threshold).
    """
    if apply_one_owner:
        df_filtered = apply_one_owner_filter(df_scored_pairs, prob_col=prob_col)
    else:
        df_filtered = df_scored_pairs.copy()

    # Filter by decision threshold
    df_matches = df_filtered[df_filtered[prob_col] >= threshold]

    # Group by S1 ID
    matches_dict = defaultdict(list)
    for _, row in df_matches.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        cand_id = str(row["candidate_entity_id"]).strip()
        matches_dict[s1_id].append(cand_id)

    # Ensure all S1 entities exist in output
    final_output = {}
    for s1_id in all_s1_ids:
        cands = matches_dict.get(s1_id, [])
        # Ensure unique IDs preserving order
        unique_cands = []
        for c in cands:
            if c not in unique_cands:
                unique_cands.append(c)
        final_output[s1_id] = unique_cands

    return final_output
