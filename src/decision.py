"""
Stage 5: decision layer (vectorized).

Given scored pairs (s1 index, candidate id, probability), decide the final match set per S1:
  1. one-owner: each candidate keeps only its highest-probability S1 (the training data has
     no S2/S3 record shared by two S1 entities),
  2. keep a pair if  p >= t,  p >= rel * best_p(S1),  and  best_p(S1) >= t_top.
All parameters are tuned for macro F0.5 on out-of-fold predictions of the training sample.
`macro_f05_arrays` reproduces src/metrics.evaluate_macro_f05 exactly (checked in tests).
"""

import itertools

import numpy as np
import pandas as pd


def one_owner(s1_idx, cand, prob):
    """Boolean mask keeping, for every candidate id, only its highest-probability pair."""
    order = np.lexsort((-prob, cand))
    keep = np.zeros(len(prob), dtype=bool)
    first = np.r_[True, cand[order][1:] != cand[order][:-1]]
    keep[order[first]] = True
    return keep


def decide(s1_idx, prob, owner_mask, t=0.5, t_top=0.0, rel=0.0):
    """Mask of pairs predicted as matches. prob/s1_idx are row-aligned, s1_idx are ints >= 0."""
    p = np.where(owner_mask, prob, -1.0)
    best = np.full(s1_idx.max() + 1 if len(s1_idx) else 1, -1.0)
    np.maximum.at(best, s1_idx, p)
    b = best[s1_idx]
    return owner_mask & (p >= t) & (p >= rel * b) & (b >= t_top)


def macro_f05_arrays(s1_idx, label, pred, n_true, n_s1):
    """Macro F0.5 over n_s1 entities. n_true[i] = number of true matches of entity i (all of them,
    including ones blocking missed). Entities with no pairs count with empty predictions."""
    tp = np.bincount(s1_idx, weights=(label & pred), minlength=n_s1)
    npred = np.bincount(s1_idx, weights=pred, minlength=n_s1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(npred > 0, tp / npred, 0.0)
        rec = np.where(n_true > 0, tp / n_true, 0.0)
        f = np.where(tp > 0, 1.25 * prec * rec / (0.25 * prec + rec), 0.0)
    f = np.where((n_true == 0) & (npred == 0), 1.0, f)
    single = n_true == 0
    return {
        "macro_f05": float(f.mean()),
        "singleton_accuracy": float(f[single].mean()) if single.any() else 1.0,
        "matches_macro_f05": float(f[~single].mean()) if (~single).any() else 0.0,
    }


def tune(s1_idx, cand, prob, label, n_true, n_s1,
         ts=np.round(np.arange(0.20, 0.91, 0.02), 3),
         t_tops=(0.0,), rels=(0.0,)):
    """Grid search of (t, t_top, rel) for macro F0.5. Returns best params, best metrics, full table."""
    own = one_owner(s1_idx, cand, prob)
    rows = []
    for t, tt, r in itertools.product(ts, t_tops, rels):
        if tt and tt < t:
            continue  # t_top below t has no effect
        m = macro_f05_arrays(s1_idx, label, decide(s1_idx, prob, own, t, tt, r), n_true, n_s1)
        rows.append({"t": t, "t_top": tt, "rel": r, **m})
    table = pd.DataFrame(rows).sort_values("macro_f05", ascending=False)
    best = table.iloc[0]
    return {"t": float(best["t"]), "t_top": float(best["t_top"]), "rel": float(best["rel"])}, best.to_dict(), table
