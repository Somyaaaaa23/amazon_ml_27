"""Print examples of (a) true matches never blocked, (b) true matches rejected by the model,
(c) false matches, from output_val/<run>/eval_predictions.tsv."""
import os, sys, random
import pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
run, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12
D = os.path.join(ROOT, "student_resource/dataset/train")
ev = pd.read_csv(os.path.join(ROOT, "output_val", run, "eval_predictions.tsv"), sep="\t", dtype=str, keep_default_na=False)
gt_df = pd.read_csv(f"{D}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = {k: [x for x in v.split(",") if x] for k, v in zip(gt_df.source1_entity_id, gt_df.matched_entity_ids)}
owner = {t: k for k, v in gt.items() for t in v}
rec = {}
for f in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
    d = pd.read_csv(f"{D}/{f}", sep="\t", dtype=str, keep_default_na=False)
    rec.update(zip(d.entity_id, zip(d.business_name, d.business_address, d.country)))
buckets = {"never_blocked": [], "rejected": [], "false_match": []}
for k, m, c in zip(ev.source1_entity_id, ev.matched_entity_ids, ev.candidate_entity_ids):
    pred, cand, true = set(m.split(",")) - {""}, set(c.split(",")) - {""}, set(gt[k])
    buckets["never_blocked"] += [(k, t) for t in true - cand]
    buckets["rejected"] += [(k, t) for t in (true & cand) - pred]
    buckets["false_match"] += [(k, t) for t in pred - true]
random.seed(1)
fmt = lambda r: f"{r[0][:55]:55s} | {r[1][:85]}"
for b, items in buckets.items():
    print(f"\n######## {b}: {len(items):,}")
    for k, t in random.sample(items, min(n, len(items))):
        print(f"[{rec[k][2]}] S1  {fmt(rec[k])}")
        extra = f"  (belongs to {owner[t]}: {rec[owner[t]][0][:40]})" if b == "false_match" and t in owner else ""
        print(f"     {t[:2]}  {fmt(rec[t])}{extra}")
        if b != "false_match":
            others = [x for x in gt[k] if x != t][:2]
            for o in others:
                print(f"     (other true {o[:2]}: {fmt(rec[o])})")
        print()
