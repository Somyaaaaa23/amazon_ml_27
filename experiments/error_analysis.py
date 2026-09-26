"""
Where does macro F0.5 go? Reads output_val/<run>/eval_predictions.tsv (holdout predictions and
candidates) and the training ground truth, and reports per country:
  - actual macro F0.5 and counterfactual scores that fix one error bucket at a time
      * no FP        : drop every false match (keep true ones)
      * no cand-FN   : add every true match that was among the candidates
      * oracle       : perfect decisions within the candidates (= best possible given blocking)
      * perfect block: oracle + blocking misses fixed (= 1.0)
  - singleton outcomes and where false matches come from (owned by another S1 / owned by nobody)

  python3 experiments/error_analysis.py current_v1
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.metrics import compute_f05_per_entity  # noqa: E402

run = sys.argv[1]
ev = pd.read_csv(os.path.join(ROOT, "output_val", run, "eval_predictions.tsv"), sep="\t", dtype=str, keep_default_na=False)
D = os.path.join(ROOT, "student_resource/dataset/train")
gt_df = pd.read_csv(f"{D}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = {k: set(x for x in v.split(",") if x) for k, v in zip(gt_df.source1_entity_id, gt_df.matched_entity_ids)}
owner = {t: k for k, v in gt.items() for t in v}
s1 = pd.read_csv(f"{D}/train_source1.tsv", sep="\t", dtype=str, usecols=["entity_id", "country"])
country = dict(zip(s1.entity_id, s1.country))

rows = []
for k, m, c in zip(ev.source1_entity_id, ev.matched_entity_ids, ev.candidate_entity_ids):
    pred, cand, true = set(x for x in m.split(",") if x), set(x for x in c.split(",") if x), gt[k]
    tp, fp = pred & true, pred - true
    fn_cand, fn_block = (true & cand) - pred, true - cand
    rows.append(dict(
        country=country[k], singleton=not true,
        f=compute_f05_per_entity(pred, true),
        f_noFP=compute_f05_per_entity(tp, true),
        f_noCandFN=compute_f05_per_entity(pred | fn_cand, true),
        f_oracle=compute_f05_per_entity(true & cand, true),
        n_fp=len(fp), n_fp_other_owner=sum(1 for t in fp if t in owner), n_fp_no_owner=sum(1 for t in fp if t not in owner),
        n_fn_cand=len(fn_cand), n_fn_block=len(fn_block), n_true=len(true), n_pred=len(pred), n_cand=len(cand)))
df = pd.DataFrame(rows)

pd.set_option("display.width", 200)
for scope, d in list(df.groupby("country")) + [("ALL", df)]:
    s, ns = d[d.singleton], d[~d.singleton]
    print(f"\n=== {scope}: {len(d):,} S1 ({len(s):,} singletons) ===")
    print(f"macro F0.5 actual        {d.f.mean():.4f}")
    print(f"  if no false matches    {d.f_noFP.mean():.4f}   (+{d.f_noFP.mean() - d.f.mean():.4f})")
    print(f"  if no candidate misses {d.f_noCandFN.mean():.4f}   (+{d.f_noCandFN.mean() - d.f.mean():.4f})")
    print(f"  oracle within cands    {d.f_oracle.mean():.4f}   (+{d.f_oracle.mean() - d.f.mean():.4f})   <- best possible with this blocking")
    print(f"  perfect blocking too   1.0000   (+{1 - d.f_oracle.mean():.4f} lost to blocking)")
    print(f"singletons: {len(s):,}, correct (empty) {(s.n_pred == 0).mean():.4f}; wrong ones carry {s.n_pred[s.n_pred > 0].mean() if (s.n_pred > 0).any() else 0:.2f} matches on average")
    print(f"loss share: singletons {(1 - s.f).sum() / max((1 - d.f).sum(), 1e-9):.1%} of total loss, non-singletons {(1 - ns.f).sum() / max((1 - d.f).sum(), 1e-9):.1%}")
    print(f"false matches: {d.n_fp.sum():,} total; owned by another S1 {d.n_fp_other_owner.sum():,}, owned by nobody {d.n_fp_no_owner.sum():,}")
    print(f"missed true matches: {d.n_fn_cand.sum():,} rejected by the model, {d.n_fn_block.sum():,} never blocked (of {d.n_true.sum():,})")
    print(f"non-singletons with 0 predictions: {(ns.n_pred == 0).mean():.4f}; exact set right: {(ns.f == 1).mean():.4f}")
