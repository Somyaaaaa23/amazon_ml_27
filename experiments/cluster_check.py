"""For true matches the pipeline missed (rejected or never blocked): how often does the missed record
look like another true match of the same S1 that WAS predicted (an 'anchor')?"""
import os, sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.normalization import normalize_name, normalize_address
run = sys.argv[1]
D = os.path.join(ROOT, "student_resource/dataset/train")
ev = pd.read_csv(os.path.join(ROOT, "output_val", run, "eval_predictions.tsv"), sep="\t", dtype=str, keep_default_na=False)
gt_df = pd.read_csv(f"{D}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = {k: set(x for x in v.split(",") if x) for k, v in zip(gt_df.source1_entity_id, gt_df.matched_entity_ids)}
rec = {}
for f in ["train_source2.tsv", "train_source3.tsv"]:
    d = pd.read_csv(f"{D}/{f}", sep="\t", dtype=str, keep_default_na=False)
    rec.update(zip(d.entity_id, zip(d.business_name, d.business_address)))
norm = {}
def N(t):
    if t not in norm:
        n, a = rec[t]
        norm[t] = (normalize_name(n)[0], normalize_address(a)[0])
    return norm[t]
stats = {"rejected": [], "never_blocked": []}
for k, m, c in zip(ev.source1_entity_id, ev.matched_entity_ids, ev.candidate_entity_ids):
    pred, cand, true = set(m.split(",")) - {""}, set(c.split(",")) - {""}, gt[k]
    anchors = pred & true
    for kind, missed in [("rejected", (true & cand) - pred), ("never_blocked", true - cand)]:
        for t in missed:
            if not anchors:
                stats[kind].append((0, 0.0, 0.0)); continue
            nt, at = N(t)
            best_n = max(fuzz.token_set_ratio(nt, N(a)[0]) for a in anchors)
            best_a = max((fuzz.token_set_ratio(at, N(a)[1]) if at and N(a)[1] else 0) for a in anchors)
            stats[kind].append((1, best_n, best_a))
for kind, rows in stats.items():
    a = np.array(rows)
    has = a[:, 0] == 1
    print(f"{kind}: {len(a):,} missed; with >=1 predicted true sibling: {has.mean():.3f}")
    for thr in (90, 95, 100):
        print(f"   sibling name token_set>={thr}: {(has & (a[:,1] >= thr)).mean():.3f}   address token_set>={thr}: {(has & (a[:,2] >= thr)).mean():.3f}   either: {(has & ((a[:,1] >= thr) | (a[:,2] >= thr))).mean():.3f}")
