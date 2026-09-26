"""Like show_misses.py but streams the source files and keeps only the needed records (low memory)."""
import os, sys, random
import pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
run, n, bucket = sys.argv[1], int(sys.argv[2]), sys.argv[3]
D = os.path.join(ROOT, "student_resource/dataset/train")
ev = pd.read_csv(os.path.join(ROOT, "output_val", run, "eval_predictions.tsv"), sep="\t", dtype=str, keep_default_na=False)
gt_df = pd.read_csv(f"{D}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
ids = set(ev.source1_entity_id)
gt = {k: [x for x in v.split(",") if x] for k, v in zip(gt_df.source1_entity_id, gt_df.matched_entity_ids) if k in ids}
items = []
for k, m, c in zip(ev.source1_entity_id, ev.matched_entity_ids, ev.candidate_entity_ids):
    pred, cand, true = set(m.split(",")) - {""}, set(c.split(",")) - {""}, set(gt[k])
    if bucket == "never_blocked":
        items += [(k, t, cand & true) for t in true - cand]
    elif bucket == "rejected":
        items += [(k, t, pred & true) for t in (true & cand) - pred]
random.seed(7)
sample = random.sample(items, min(n, len(items)))
need = {k for k, _, _ in sample} | {t for _, t, _ in sample} | {x for _, _, s in sample for x in s}
rec = {}
for f in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
    for ch in pd.read_csv(f"{D}/{f}", sep="\t", dtype=str, keep_default_na=False, chunksize=500000):
        ch = ch[ch.entity_id.isin(need)]
        rec.update(zip(ch.entity_id, zip(ch.business_name, ch.business_address, ch.country)))
fmt = lambda r: f"{r[0][:50]:50s} | {r[1][:90]}"
print(f"{bucket}: {len(items):,}")
for k, t, found in sample:
    print(f"[{rec[k][2]}] S1 {fmt(rec[k])}\n     MISSED {t[:2]} {fmt(rec[t])}")
    for x in list(found)[:2]:
        print(f"     found  {x[:2]} {fmt(rec[x])}")
    print()
