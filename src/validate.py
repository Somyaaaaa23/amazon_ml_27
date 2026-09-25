"""
Validation harness: scores a pipeline version on a held-out split of the TRAINING data,
matching against the FULL S2/S3 pool of each country (same density as the test set).

Split: an S1 entity is in the holdout iff crc32(entity_id) % 5 == 0 (~20%, deterministic,
independent of country, so it is stratified by construction; proportions are asserted).
Train / eval S1 sets are stratified samples (fixed seed) from each side of the split.
Thresholds are only ever tuned inside the adapter on training data (OOF); the holdout is
used for reporting only.

Usage:
  python3 src/validate.py --adapter original --code-dir .worktrees/original --name baseline
  python3 src/validate.py --adapter antigravity --code-dir . --name antigravity
  # France proxy: train on one country, evaluate on the other
  python3 src/validate.py ... --train-countries US --eval-countries India
"""

import argparse
import gc
import importlib
import json
import os
import sys
import time
import zlib
from collections import defaultdict
from itertools import chain

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "test")
RESULTS_MD = os.path.join(PROJECT_ROOT, "RESULTS.md")
VAL_OUT = os.path.join(PROJECT_ROOT, "output_val")

T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


# -----------------------------------------------------------------------------
# Data, split and sampling
# -----------------------------------------------------------------------------
def is_holdout(entity_id: str) -> bool:
    return zlib.crc32(entity_id.encode("utf-8")) % 5 == 0


def load_train():
    # Same NA handling as the pipelines (default read_csv), dtype=str so ids stay strings
    rd = lambda f: pd.read_csv(os.path.join(TRAIN_DIR, f), sep="\t", dtype=str)
    s1, s2, s3 = rd("train_source1.tsv"), rd("train_source2.tsv"), rd("train_source3.tsv")
    gt_df = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t", dtype=str, keep_default_na=False)
    gt = {k.strip(): [x.strip() for x in v.split(",") if x.strip()]
          for k, v in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"])}
    assert gt_df["source1_entity_id"].is_unique, "duplicate S1 ids in ground truth"
    missing = set(s1["entity_id"]) - set(gt)
    assert not missing, f"ground truth misses {len(missing)} train S1 entities"
    return s1, s2, s3, gt


def stratified_sample(df, n, seed):
    if n >= len(df):
        return df
    frac = n / len(df)
    return df.groupby("country", group_keys=False).sample(frac=frac, random_state=seed)


def make_split(s1, n_train, n_eval, train_countries, eval_countries, seed=0):
    hold = s1["entity_id"].map(is_holdout)
    by_c = s1.assign(h=hold).groupby("country")["h"].mean()
    log(f"holdout share by country: {by_c.round(4).to_dict()}")
    assert ((by_c - 0.2).abs() < 0.01).all(), "holdout split not ~20% in every country"
    tr_pool, ev_pool = s1[~hold], s1[hold]
    if train_countries:
        tr_pool = tr_pool[tr_pool["country"].isin(train_countries)]
    if eval_countries:
        ev_pool = ev_pool[ev_pool["country"].isin(eval_countries)]
    tr = stratified_sample(tr_pool, n_train, seed)
    ev = stratified_sample(ev_pool, n_eval, seed)
    assert not set(tr["entity_id"]) & set(ev["entity_id"])
    return tr.reset_index(drop=True), ev.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def score(cands, preds, gt, s1_ids, evaluate_macro_f05):
    sub_gt = {k: gt[k] for k in s1_ids}
    true_pairs = sum(len(v) for v in sub_gt.values())
    captured = sum(len(set(v) & set(cands.get(k, []))) for k, v in sub_gt.items())
    n_c = np.array([len(cands.get(k, [])) for k in s1_ids])
    bad = [k for k in s1_ids if not set(preds.get(k, [])) <= set(cands.get(k, []))]
    assert not bad, f"{len(bad)} entities have matches outside their candidates"
    m = evaluate_macro_f05({k: preds.get(k, []) for k in s1_ids}, sub_gt)
    return {
        "n_s1": len(s1_ids),
        "recall_ceiling": captured / max(true_pairs, 1),
        "avg_cands": float(n_c.mean()),
        "max_cands": int(n_c.max()),
        "macro_f05": m["macro_f05"],
        "singleton_acc": m["singleton_accuracy"],
        "matches_f05": m["matches_macro_f05"],
        "n_singletons": m["total_singletons"],
    }


# -----------------------------------------------------------------------------
# Helpers shared by adapters
# -----------------------------------------------------------------------------
def load_modules(code_dir):
    code_dir = os.path.abspath(code_dir)
    for k in [k for k in sys.modules if k == "src" or k.startswith("src.") or k.startswith("utils")]:
        del sys.modules[k]
    sys.path.insert(0, code_dir)
    mods = {n: importlib.import_module(f"src.{n}") for n in ["normalization", "blocking", "features", "model", "postprocessing"]}
    sys.path.pop(0)
    return mods


def raw_test_text(nrows):
    out = []
    for f in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
        d = pd.read_csv(os.path.join(TEST_DIR, f), sep="\t", nrows=nrows)
        out.extend(d["business_name"].fillna("").astype(str) + " " + d["business_address"].fillna("").astype(str))
    return out


_QUERY_FN = None


def _run_query(batch):
    return _QUERY_FN(batch)


def parallel_query(fn, s1_df, qbatch, workers=4):
    """Runs fn over row batches of s1_df in forked workers (the index is shared copy-on-write).
    Pure harness speed-up: results are identical to a sequential loop over the same batches."""
    global _QUERY_FN
    batches = [s1_df.iloc[i:i + qbatch] for i in range(0, len(s1_df), qbatch)]
    out = {}
    if workers <= 1 or len(batches) < 2 * workers:
        for b in batches:
            out.update(fn(b))
        return out
    import multiprocessing as mp
    _QUERY_FN = fn
    with mp.get_context("fork").Pool(workers) as pool:
        for i, r in enumerate(pool.imap(_run_query, batches, chunksize=1)):
            out.update(r)
            if (i + 1) % 200 == 0:
                log(f"    queried {min((i + 1) * qbatch, len(s1_df)):,}/{len(s1_df):,}")
    _QUERY_FN = None
    return out


# -----------------------------------------------------------------------------
# Adapter: ORIGINAL code (commit 57fc86f), logic of the original src/infer_test.py.
# Only run-enabling shims: query in batches of `qbatch` rows (the original multiplies a whole
# 50k batch against the pool at once -> OOM), float32 TF-IDF (halves memory), and feature
# building against the candidate targets only (the original converts the whole pool to a
# dict per batch). Model/threshold/postprocessing logic is untouched.
# -----------------------------------------------------------------------------
def adapter_original(mods, s1, s2, s3, gt, tr, ev, top_k=25, qbatch=25):
    N, B, F, M, P = (mods[k] for k in ["normalization", "blocking", "features", "model", "postprocessing"])
    tr_ids = list(tr["entity_id"])
    gt_tr = {k: gt[k] for k in tr_ids}
    needed = set(chain.from_iterable(gt_tr.values()))

    def sampled_pool(df):  # exactly the original: all needed targets + 3% of the rest, per 250k chunk
        parts = []
        for i in range(0, len(df), 250000):
            ch = df.iloc[i:i + 250000]
            parts.append(pd.concat([ch[ch["entity_id"].isin(needed)],
                                    ch[~ch["entity_id"].isin(needed)].sample(frac=0.03, random_state=42)]))
        return pd.concat(parts, ignore_index=True)

    log("original: preprocessing training data (3% distractor pool, as in original)")
    tr_n = N.preprocess_dataframe(tr)
    s2_tr, s3_tr = N.preprocess_dataframe(sampled_pool(s2)), N.preprocess_dataframe(sampled_pool(s3))
    targets_tr = pd.concat([s2_tr, s3_tr], ignore_index=True)
    log(f"original: training pool {len(targets_tr):,} targets")

    blocker = B.MultiBlocker(top_k_per_source=top_k)
    blocker.tfidf_char.set_params(dtype=np.float32)
    corpus = list(tr_n["combined_text"]) + list(targets_tr["combined_text"]) + raw_test_text(50000)
    blocker.fit_vectorizers(corpus)
    log("original: vectorizers fitted")

    def query_all(index, s1_df):
        return parallel_query(lambda b: index.query_batch(b, blocker, top_k=top_k), s1_df, qbatch)

    # generate_candidates, batched
    train_cands = defaultdict(set)
    for c in tr_n["country_clean"].unique():
        tgt_c = targets_tr[targets_tr["country_clean"] == c].reset_index(drop=True)
        idx = B.CountryTargetIndex(tgt_c, blocker)
        for k, v in query_all(idx, tr_n[tr_n["country_clean"] == c]).items():
            train_cands[k].update(v)
        del idx
    train_cands = {k: sorted(train_cands.get(k, set())) for k in tr_ids}
    log("original: training candidates done")

    extractor = F.PairFeatureExtractor(idf_dict=blocker.idf_dict)
    pairs = extractor.build_feature_table(df_s1=tr_n, df_target=targets_tr, candidates_dict=train_cands, ground_truth_dict=gt_tr)
    log(f"original: {len(pairs):,} training pairs ({int(pairs['label'].sum()):,} positive)")
    matcher = M.EntityMatcherModel(n_estimators=300, learning_rate=0.05, max_depth=6)
    cv = matcher.train_cv(df_train_pairs=pairs, ground_truth_dict=gt_tr, n_splits=5)
    best_t = matcher.best_threshold
    cv_info = {"threshold": best_t, "cv_macro_f05": cv["best_metrics"]["macro_f05"]}
    log(f"original: trained, t*={best_t:.3f}, CV macro F0.5={cv_info['cv_macro_f05']:.4f}")
    del pairs, targets_tr, s2_tr, s3_tr, cv
    gc.collect()

    # Evaluation: each country against its FULL pool, smallest pool first (memory)
    pools = {c: pd.concat([s2[s2["country"] == c], s3[s3["country"] == c]], ignore_index=True) for c in ev["country"].unique()}
    cands_all, preds_all = {}, {}
    for c in sorted(pools, key=lambda c: len(pools[c])):
        pool = N.preprocess_dataframe(pools.pop(c))
        pool = pool.drop(columns=["business_name", "business_address", "country", "landmark"])
        log(f"original eval [{c}]: pool {len(pool):,}, building index")
        idx = B.CountryTargetIndex(pool, blocker)
        ev_c = N.preprocess_dataframe(ev[ev["country"] == c])
        log(f"original eval [{c}]: querying {len(ev_c):,} S1")
        cands = query_all(idx, ev_c)
        del idx
        gc.collect()
        need = set(chain.from_iterable(cands.values()))
        pool_small = pool[pool["entity_id"].isin(need)]
        del pool
        gc.collect()
        log(f"original eval [{c}]: features")
        for i in range(0, len(ev_c), 50000):  # original chunk_size=50000, one-owner per chunk
            batch = ev_c.iloc[i:i + 50000]
            ids = list(batch["entity_id"])
            pairs = extractor.build_feature_table(df_s1=batch, df_target=pool_small,
                                                  candidates_dict={k: cands[k] for k in ids})
            if pairs.empty:
                continue
            scored = matcher.predict_pairs(pairs)
            preds_all.update(P.generate_matching_predictions(scored, ids, threshold=best_t, apply_one_owner=True, prob_col="pred_prob"))
            cands_all.update(pairs.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict())
        log(f"original eval [{c}]: done")
    return cands_all, preds_all, cv_info


# -----------------------------------------------------------------------------
# Adapter: ANTIGRAVITY batch (commit a8bd013), logic of its src/infer_test.py.
# Same run-enabling shims as above (batched queries, float32 TF-IDF, candidate-only target
# frame). Each country's full-pool index is built once and used for both the training and the
# evaluation queries (identical inputs to building it twice). Its one-owner rule is global over
# the evaluated S1 of a country, then threshold, as in its code.
# -----------------------------------------------------------------------------
def adapter_antigravity(mods, s1, s2, s3, gt, tr, ev, top_k=20, qbatch=25):
    N, B, F, M = (mods[k] for k in ["normalization", "blocking", "features", "model"])
    gt_tr = {k: gt[k] for k in tr["entity_id"]}
    tr_n = N.preprocess_dataframe(tr)
    ev_n = N.preprocess_dataframe(ev)

    blocker = B.MultiBlocker(top_k_per_source=top_k)
    blocker.tfidf_char.set_params(dtype=np.float32)
    corpus = list(tr_n["combined_text"])
    for f in ["train_source2.tsv", "train_source3.tsv"]:
        d = pd.read_csv(os.path.join(TRAIN_DIR, f), sep="\t", nrows=25000)
        corpus.extend(d["business_name"].fillna("").astype(str) + " " + d["business_address"].fillna("").astype(str))
    corpus.extend(raw_test_text(25000))
    blocker.fit_vectorizers(corpus)
    extractor = F.PairFeatureExtractor(idf_dict=blocker.idf_dict, max_idf=blocker.max_idf)
    log("antigravity: vectorizers fitted")

    def query_all(index, s1_df):
        def fn(b):
            c, cos = index.query_batch(b, blocker, top_k=top_k)
            return {k: (v, {t: cos.get((k, t), 0.0) for t in v}) for k, v in c.items()}
        res = parallel_query(fn, s1_df, qbatch)
        cands = {k: v[0] for k, v in res.items()}
        cos = {(k, t): s for k, v in res.items() for t, s in v[1].items()}
        return cands, cos

    train_pairs, ev_pairs, ev_cands = [], {}, {}
    countries = sorted(set(tr["country"]) | set(ev["country"]))
    pools = {c: pd.concat([s2[s2["country"] == c], s3[s3["country"] == c]], ignore_index=True) for c in countries}
    for c in sorted(pools, key=lambda c: len(pools[c])):
        pool = N.preprocess_dataframe(pools.pop(c))
        pool = pool.drop(columns=["business_name", "business_address", "country", "landmark"])
        log(f"antigravity [{c}]: pool {len(pool):,}, building index")
        idx = B.CountryTargetIndex(pool, blocker)
        tr_c, ev_c = tr_n[tr_n["country"] == c], ev_n[ev_n["country"] == c]
        log(f"antigravity [{c}]: querying {len(tr_c):,} train + {len(ev_c):,} eval S1")
        tc, tcos = query_all(idx, tr_c) if len(tr_c) else ({}, {})
        ec, ecos = query_all(idx, ev_c) if len(ev_c) else ({}, {})
        del idx
        gc.collect()
        need = set(chain.from_iterable(tc.values())) | set(chain.from_iterable(ec.values()))
        pool_small = pool[pool["entity_id"].isin(need)]
        del pool
        gc.collect()
        if tc:
            train_pairs.append(extractor.build_feature_table(tr_c, pool_small, tc, tcos, gt_tr))
        if ec:
            ev_pairs[c] = extractor.build_feature_table(ev_c, pool_small, ec, ecos, None)
            ev_cands.update(ec)
        log(f"antigravity [{c}]: features done")

    pairs = pd.concat(train_pairs, ignore_index=True)
    log(f"antigravity: {len(pairs):,} training pairs ({int(pairs['label'].sum()):,} positive)")
    matcher = M.EntityMatcherModel(n_estimators=300, learning_rate=0.05, max_depth=6)
    cv = matcher.train_cv(df_train_pairs=pairs, ground_truth_dict=gt_tr, n_splits=5)
    best_t = matcher.best_threshold
    cv_info = {"threshold": best_t, "cv_macro_f05": cv["best_metrics"]["macro_f05"]}
    log(f"antigravity: trained, t*={best_t:.3f}, CV macro F0.5={cv_info['cv_macro_f05']:.4f}")

    preds = {}
    for c, p in ev_pairs.items():
        if p.empty:
            continue
        sc = matcher.predict_pairs(p)[["source1_entity_id", "candidate_entity_id", "pred_prob"]]
        sc = sc.sort_values("pred_prob", ascending=False).drop_duplicates("candidate_entity_id")
        sc = sc[sc["pred_prob"] >= best_t]
        for k, t in zip(sc["source1_entity_id"], sc["candidate_entity_id"]):
            preds.setdefault(k, []).append(t)
    return ev_cands, preds, cv_info


ADAPTERS = {"original": adapter_original, "antigravity": adapter_antigravity}


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def append_results(name, args, res, cv_info, minutes):
    new = not os.path.exists(RESULTS_MD)
    with open(RESULTS_MD, "a", encoding="utf-8") as f:
        if new:
            f.write("# Validation results\n\n"
                    "All numbers are measured with `src/validate.py`: train and evaluate S1 samples come from "
                    "disjoint sides of a fixed 80/20 split of the training data (crc32(entity_id) % 5), and "
                    "every S1 entity is matched against the FULL S2/S3 pool of its country. "
                    "The threshold is tuned on out-of-fold predictions of the training sample only.\n\n"
                    "| run | code | train → eval | scope | recall ceiling | avg cands | macro F0.5 | singleton acc | matches F0.5 | CV F0.5 (OOF) | minutes |\n"
                    "|---|---|---|---|---|---|---|---|---|---|---|\n")
        tr_desc = f"{args.n_train//1000}k {'+'.join(args.train_countries or ['all'])} → {args.n_eval//1000}k {'+'.join(args.eval_countries or ['all'])}"
        cv_txt = f"{cv_info.get('cv_macro_f05', float('nan')):.4f}" if cv_info else "-"
        for scope, r in res.items():
            f.write(f"| {name} | {args.adapter} | {tr_desc} | {scope} | {r['recall_ceiling']*100:.2f}% | {r['avg_cands']:.1f} | "
                    f"{r['macro_f05']:.4f} | {r['singleton_acc']:.4f} | {r['matches_f05']:.4f} | "
                    f"{cv_txt if scope == 'ALL' else ''} | {minutes:.0f} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-train", type=int, default=30000)
    ap.add_argument("--n-eval", type=int, default=30000)
    ap.add_argument("--train-countries", nargs="*")
    ap.add_argument("--eval-countries", nargs="*")
    ap.add_argument("--no-results-md", action="store_true", help="smoke tests: do not append to RESULTS.md")
    args = ap.parse_args()

    sys.path.insert(0, PROJECT_ROOT)
    from src.metrics import evaluate_macro_f05  # verified metric, shared by all versions

    log(f"run '{args.name}': adapter={args.adapter} code={args.code_dir}")
    s1, s2, s3, gt = load_train()
    log(f"loaded train: S1 {len(s1):,}  S2 {len(s2):,}  S3 {len(s3):,}; GT covers all S1")
    tr, ev = make_split(s1, args.n_train, args.n_eval, args.train_countries, args.eval_countries)
    log(f"train S1 {len(tr):,} {tr['country'].value_counts().to_dict()} | eval S1 {len(ev):,} {ev['country'].value_counts().to_dict()}")
    del s1
    mods = load_modules(args.code_dir)

    cands, preds, cv_info = ADAPTERS[args.adapter](mods, None, s2, s3, gt, tr, ev)
    minutes = (time.time() - T0) / 60

    res = {}
    for c in sorted(ev["country"].unique()):
        res[c] = score(cands, preds, gt, list(ev.loc[ev["country"] == c, "entity_id"]), evaluate_macro_f05)
    res["ALL"] = score(cands, preds, gt, list(ev["entity_id"]), evaluate_macro_f05)

    out_dir = os.path.join(VAL_OUT, args.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"args": vars(args), "cv": cv_info, "results": res, "minutes": minutes}, f, indent=2)
    with open(os.path.join(out_dir, "eval_predictions.tsv"), "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\tcandidate_entity_ids\n")
        for k in ev["entity_id"]:
            f.write(f"{k}\t{','.join(preds.get(k, []))}\t{','.join(cands.get(k, []))}\n")

    print("\n" + "=" * 100)
    print(f"RESULTS '{args.name}' ({minutes:.1f} min)  CV: {cv_info}")
    for scope, r in res.items():
        print(f"  {scope:6s} n={r['n_s1']:6d}  recall_ceiling={r['recall_ceiling']*100:6.2f}%  avg_cands={r['avg_cands']:5.1f}  "
              f"max_cands={r['max_cands']:3d}  macroF0.5={r['macro_f05']:.4f}  singleton_acc={r['singleton_acc']:.4f} "
              f"(n={r['n_singletons']})  matchesF0.5={r['matches_f05']:.4f}")
    print("=" * 100)
    if not args.no_results_md:
        append_results(args.name, args, res, cv_info, minutes)


if __name__ == "__main__":
    main()
