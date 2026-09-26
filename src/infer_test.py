"""
Full test inference: train on the training data, then match every test S1 entity.

  python3 src/infer_test.py --n-train 30000

1. Train: a stratified sample of training S1 entities is matched against the FULL S2/S3 pool of
   its country; LightGBM is trained with GroupKFold and the decision rule is tuned on its
   out-of-fold predictions (macro F0.5).
2. Test: per country (taken from the data, never hard-coded), one index over that country's full
   test pool; S1 scored in chunks; the one-owner rule and decision rule are applied once per
   country over all its scored pairs; one output row per test S1 entity.
3. The official validator (student_resource/utils/validate_submission.py) checks both files.
"""

import argparse
import gc
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import pipeline_core as pc  # noqa: E402
from src.pipeline_core import log  # noqa: E402


def read(path):
    return pd.read_csv(path, sep="\t", dtype=str)


def pools_by_country(s2, s3, countries):
    return {c: pd.concat([s2[s2["country"] == c], s3[s3["country"] == c]], ignore_index=True) for c in countries}


def train(train_dir, n_train, prune_k=None, rival_drop=0.0, hop2=False, views=None, seed=0):
    s1, s2, s3 = (read(os.path.join(train_dir, f"train_source{i}.tsv")) for i in (1, 2, 3))
    gt_df = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", dtype=str, keep_default_na=False)
    gt = {k: [x for x in v.split(",") if x] for k, v in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"])}
    assert set(s1["entity_id"]) <= set(gt), "ground truth does not cover all training S1"
    tr = s1.groupby("country", group_keys=False).sample(frac=min(1.0, n_train / len(s1)), random_state=seed)
    tr_n = pc.normalize(tr)
    log(f"train S1: {len(tr_n):,} {tr_n['country'].value_counts().to_dict()}")
    tr_key = {k: i for i, k in enumerate(tr_n["entity_id"])}
    parts, offset = [], 0
    pools = pools_by_country(s2, s3, sorted(tr_n["country"].unique()))
    s1_by_c = {c: s1[s1["country"] == c] for c in pools}
    del s1, s2, s3
    for c in sorted(pools, key=lambda c: len(pools[c])):
        pool = pc.normalize(pools.pop(c))
        tr_c = tr_n[tr_n["country"] == c].reset_index(drop=True)
        idx = pc.CountryIndex(pool, extra=[tr_c], hop2=hop2, views=views)
        # rival pool: the country's training S1, minus a random share of the non-sampled ones to
        # simulate the test set (where ~20% of businesses have S2/S3 records but no S1 entity)
        s1c = s1_by_c.pop(c)
        keep = s1c["entity_id"].isin(set(tr_c["entity_id"])).to_numpy() | (np.random.default_rng(1).random(len(s1c)) >= rival_drop)
        rivals = pc.RivalIndex(pc.normalize(s1c[keep]))
        pairs = idx.candidates(tr_c)
        feats = pc.build_features(idx, tr_c, pairs, rivals=rivals)
        s1_ids = tr_c["entity_id"].to_numpy()[feats["s1_idx"].to_numpy()]
        cand_ids = idx.pool["entity_id"].to_numpy()[feats["t_idx"].to_numpy()]
        gt_set = {k: set(gt[k]) for k in set(s1_ids)}
        feats["label"] = np.fromiter((t in gt_set[s] for s, t in zip(s1_ids, cand_ids)), dtype=np.int8, count=len(s1_ids))
        feats["s1_key"] = pd.Series(s1_ids).map(tr_key).to_numpy()
        feats["cand_key"] = feats["t_idx"].to_numpy().astype(np.int64) + offset
        offset += len(pool) + 1
        n_pos = sum(len(gt[k]) for k in tr_c["entity_id"])
        log(f"train [{c}]: pool {len(pool):,}; {len(feats):,} pairs; blocking recall {feats['label'].sum() / max(n_pos, 1):.4f}")
        parts.append(feats)
        del idx, pool
        gc.collect()
    train_df = pd.concat(parts, ignore_index=True)
    n_true = np.array([len(gt[k]) for k in tr_n["entity_id"]])
    model, _ = pc.train_model(train_df, n_true, len(tr_n), prune_k=prune_k)
    return model


def predict_test(model, test_dir, output_dir, chunk, hop2=False, views=None):
    s1, s2, s3 = (read(os.path.join(test_dir, f"test_source{i}.tsv")) for i in (1, 2, 3))
    n_s1 = len(s1)
    log(f"test S1 {n_s1:,} {s1['country'].value_counts(dropna=False).to_dict()}")
    os.makedirs(output_dir, exist_ok=True)
    match_path = os.path.join(output_dir, "matching_results.tsv")
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    matches, cands = {}, {}
    countries = [c for c in s1["country"].dropna().unique()]
    pools = pools_by_country(s2, s3, countries)
    del s2, s3
    for c in sorted(pools, key=lambda c: len(pools[c])):
        s1_c = pc.normalize(s1[s1["country"] == c])
        pool = pc.normalize(pools.pop(c))
        log(f"test [{c}]: {len(s1_c):,} S1, pool {len(pool):,}; building index")
        if pool.empty:
            continue
        idx = pc.CountryIndex(pool, extra=[s1_c], hop2=hop2, views=views)
        rivals = pc.RivalIndex(s1_c)                             # every test S1 of the country
        pool_ids = idx.pool["entity_id"].to_numpy()
        S, T, P = [], [], []
        for lo in range(0, len(s1_c), chunk):
            part = s1_c.iloc[lo:lo + chunk].reset_index(drop=True)
            t0 = time.time()
            pairs = idx.candidates(part)
            t1 = time.time()
            feats = pc.build_features(idx, part, pairs, rivals=rivals)
            t2 = time.time()
            prob, kept = model.predict(feats)     # kept = pairs the final model scored
            t3 = time.time()
            timing = f"blocking {t1 - t0:.0f}s, features {t2 - t1:.0f}s, model {t3 - t2:.0f}s"
            S.append(feats["s1_idx"].to_numpy().astype(np.int64)[kept] + lo)
            T.append(feats["t_idx"].to_numpy().astype(np.int64)[kept])
            P.append(prob[kept])
            log(f"test [{c}]: scored {min(lo + chunk, len(s1_c)):,}/{len(s1_c):,} S1 "
                f"({len(feats) / len(part):.1f} blocked, {kept.sum() / len(part):.1f} scored by final model per S1; {timing})")
            del feats, pairs
        S, T, P = np.concatenate(S), np.concatenate(T), np.concatenate(P)
        own = pc.one_owner(S, T, P)               # global over the whole country
        keep = pc.decide(S, P, own, **model.decision)
        ids = s1_c["entity_id"].to_numpy()
        order = np.argsort(S, kind="stable")
        for arr, dst in [(order, cands), (order[keep[order]], matches)]:
            s_sorted, t_sorted = S[arr], T[arr]
            bounds = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1], True])
            for a, b in zip(bounds[:-1], bounds[1:]):
                dst[ids[s_sorted[a]]] = pool_ids[t_sorted[a:b]]
        log(f"test [{c}]: {int(keep.sum()):,} matches for {len(s1_c):,} S1")
        del idx, pool, S, T, P
        gc.collect()

    # exactly one row per test S1, in the order of test_source1.tsv
    with open(match_path, "w", encoding="utf-8") as fm, open(cand_path, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for k in s1["entity_id"]:
            fm.write(f"{k}\t{','.join(matches.get(k, ()))}\n")
            fc.write(f"{k}\t{','.join(cands.get(k, ()))}\n")
    rows = sum(1 for _ in open(match_path)) - 1
    assert rows == n_s1, f"wrote {rows} rows, expected {n_s1}"
    empty = sum(1 for k in s1["entity_id"] if not len(matches.get(k, ())))
    log(f"wrote {rows:,} rows; {empty:,} with no match ({empty / rows:.1%})")
    return match_path, cand_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=os.path.join(PROJECT_ROOT, "student_resource/dataset/train"))
    ap.add_argument("--test-dir", default=os.path.join(PROJECT_ROOT, "student_resource/dataset/test"))
    ap.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "output"))
    ap.add_argument("--n-train", type=int, default=30000)
    ap.add_argument("--chunk", type=int, default=30000)
    ap.add_argument("--prune-k", type=int, default=None, help="stage 1 keeps the top-k candidates per S1 for the final model")
    ap.add_argument("--rival-drop", type=float, default=0.0, help="share of non-sampled training S1 left out of the rival pool")
    ap.add_argument("--hop2", action="store_true", help="second-hop retrieval from confident candidates")
    ap.add_argument("--views", default="default", help="blocking view set (pipeline_core.VIEW_SETS)")
    args = ap.parse_args()
    views = pc.VIEW_SETS[args.views]

    model = train(args.train_dir, args.n_train, args.prune_k, args.rival_drop, args.hop2, views)
    log(f"model: decision {model.decision}, OOF {model.cv}")
    match_path, cand_path = predict_test(model, args.test_dir, args.output_dir, args.chunk, args.hop2, views)
    validator = os.path.join(PROJECT_ROOT, "student_resource", "utils", "validate_submission.py")
    r = subprocess.run([sys.executable, validator, "--matching", match_path, "--candidate", cand_path, "--test-dir", args.test_dir])
    log("VALIDATOR PASS" if r.returncode == 0 else "VALIDATOR FAIL")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
