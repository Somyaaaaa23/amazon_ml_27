"""
Evaluation metrics for the Entity Resolution challenge.
Implements the macro-averaged F0.5 metric with precise singleton handling as defined in Section 2.
"""

from typing import Dict, Set, Iterable, Any
import numpy as np


def compute_f05_per_entity(pred_ids: Set[str], true_ids: Set[str]) -> float:
    """
    Computes F0.5 score for a single Source 1 entity.
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

    Singleton handling:
    - If truth is empty and pred is empty -> 1.0
    - If truth is empty and pred is not empty -> 0.0
    - If truth is not empty and pred is empty -> 0.0
    """
    # Normalize sets
    pred = set(pred_ids) if pred_ids else set()
    truth = set(true_ids) if true_ids else set()

    if not truth and not pred:
        return 1.0
    if not truth and pred:
        return 0.0
    if truth and not pred:
        return 0.0

    tp = len(pred.intersection(truth))
    if tp == 0:
        return 0.0

    precision = tp / len(pred)
    recall = tp / len(truth)

    denom = (0.25 * precision) + recall
    if denom == 0:
        return 0.0

    return (1.25 * precision * recall) / denom


def evaluate_macro_f05(
    predictions: Dict[str, Iterable[str]],
    ground_truth: Dict[str, Iterable[str]]
) -> Dict[str, float]:
    """
    Computes macro F0.5 across all Source 1 entities in ground_truth.
    Also provides detailed breakdowns (singletons vs non-singletons).
    """
    scores = []
    singleton_scores = []
    match_scores = []

    for s1_id, true_list in ground_truth.items():
        true_set = set(true_list) if true_list else set()
        # Filter empty strings if parsed from blank TSV
        true_set = {x for x in true_set if x.strip()}
        
        pred_list = predictions.get(s1_id, set())
        pred_set = set(pred_list) if pred_list else set()
        pred_set = {x for x in pred_set if x.strip()}

        score = compute_f05_per_entity(pred_set, true_set)
        scores.append(score)

        if not true_set:
            singleton_scores.append(score)
        else:
            match_scores.append(score)

    return {
        "macro_f05": float(np.mean(scores)) if scores else 0.0,
        "singleton_accuracy": float(np.mean(singleton_scores)) if singleton_scores else 1.0,
        "matches_macro_f05": float(np.mean(match_scores)) if match_scores else 0.0,
        "total_s1_entities": len(ground_truth),
        "total_singletons": len(singleton_scores),
        "total_with_matches": len(match_scores),
    }
