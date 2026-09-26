"""How many blocking misses of a holdout run could a 'twin closure' recover?

For every true match that never made the candidate list, check whether it is an exact twin
(same compact name / name key / address key) of one of the S1's final candidates, and whether
that candidate is itself a true match. Streams the source files (low memory).

  python3 experiments/closure_check.py v12a_digitfix
"""
import os
import sys

import pandas as pd
from rapidfuzz import fuzz

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src import pipeline_core as pc  # noqa: E402

run = sys.argv[1]
D = os.path.join(ROOT, "student_resource/dataset/train")
ev = pd.read_csv(os.path.join(ROOT, "output_val", run, "eval_predictions.tsv"), sep="\t", dtype=str, keep_default_na=False)
gt_df = pd.read_csv(f"{D}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
ids = set(ev.source1_entity_id)
gt = {k: set(x for x in v.split(",") if x) for k, v in zip(gt_df.source1_entity_id, gt_df.matched_entity_ids) if k in ids}

rows = []   # (s1, missed target, candidates, true candidates)
for k, c in zip(ev.source1_entity_id, ev.candidate_entity_ids):
    cand = set(c.split(",")) - {""}
    for t in gt[k] - cand:
        rows.append((k, t, cand, cand & gt[k]))
need = {r[0] for r in rows} | {r[1] for r in rows} | {x for r in rows for x in r[2]}
parts = []
for f in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
    for ch in pd.read_csv(f"{D}/{f}", sep="\t", dtype=str, chunksize=500000):
        parts.append(ch[ch.entity_id.isin(need)])
rec = pc.normalize(pd.concat(parts, ignore_index=True)).set_index("entity_id")
key = {col: rec[col].to_dict() for col in ["name_compact", "name_key", "addr_key", "addr_norm", "name_norm"]}
country = rec["country"].to_dict()

stats = {}
for s, t, cand, true_c in rows:
    c_ = country[s]
    st = stats.setdefault(c_, dict(missed=0, twin_any=0, twin_true=0, twin_true_addr80=0, twin_true_name90=0))
    st["missed"] += 1
    hits = []
    for x in cand:
        same = any(key[col][x] and key[col][x] == key[col][t] and len(key[col][x]) >= 6
                   for col in ["name_compact", "name_key", "addr_key"])
        if same:
            hits.append(x)
    if not hits:
        continue
    st["twin_any"] += 1
    th = [x for x in hits if x in true_c]
    if th:
        st["twin_true"] += 1
        a80 = any(fuzz.token_set_ratio(key["addr_norm"][s], key["addr_norm"][x]) >= 80 for x in th)
        n90 = any(fuzz.token_set_ratio(key["name_norm"][s], key["name_norm"][x]) >= 90 for x in th)
        st["twin_true_addr80"] += a80
        st["twin_true_name90"] += n90

print(f"{run}: blocking misses that are exact twins (compact name / name key / address key) of a final candidate")
for c_, st in stats.items():
    m = st["missed"]
    print(f"  {c_:6s} missed {m:5d} | twin of any candidate {st['twin_any']:5d} ({st['twin_any'] / m:.1%})"
          f" | twin of a TRUE candidate {st['twin_true']:5d} ({st['twin_true'] / m:.1%})"
          f" | ...whose address fits S1 (ts>=80) {st['twin_true_addr80']:5d}"
          f" | ...whose name fits S1 (ts>=90) {st['twin_true_name90']:5d}")
