"""
Blocking design experiments (not part of the pipeline).
Uses only the TRAINING side of the validation split, so blocking choices are not tuned on the holdout.
For each variant: fit per-country TF-IDF on the country's full pool, query sampled train S1,
report density (nnz/row), query time, and recall@k of true matches; plus unions of variants.

  python3 experiments/blocking_lab.py --country India --n-query 3000
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.validate import load_train, make_split, log  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--country", default="India")
ap.add_argument("--n-query", type=int, default=3000)
ap.add_argument("--norm-dir", default=os.path.join(ROOT, ".worktrees", "antigravity"))
args = ap.parse_args()

sys.path.insert(0, args.norm_dir)
for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
    del sys.modules[k]
from src.normalization import preprocess_dataframe  # noqa: E402

s1, s2, s3, gt = load_train()
tr, _ = make_split(s1, 30000, 30000, None, None)
q = tr[tr["country"] == args.country].head(args.n_query)
pool = pd.concat([s2[s2["country"] == args.country], s3[s3["country"] == args.country]], ignore_index=True)
del s1, s2, s3
log(f"{args.country}: pool {len(pool):,}, queries {len(q):,}; normalizing")
pool = preprocess_dataframe(pool)
q = preprocess_dataframe(q)
tid = {t: i for i, t in enumerate(pool["entity_id"])}
truth = [np.array([tid[t] for t in gt[k]], dtype=np.int64) for k in q["entity_id"]]
n_true = sum(len(t) for t in truth)
log(f"normalized; {n_true:,} true pairs among queries")

KS = [5, 10, 20, 30, 50, 100]


def topk_lists(Q, T, k=100, bs=250):
    """Returns per-query arrays of target indices sorted by score desc (top k), plus stats."""
    out, nnz, t0 = [], 0, time.time()
    for i in range(0, Q.shape[0], bs):
        S = (Q[i:i + bs] @ T).tocsr()
        nnz += S.nnz
        for r in range(S.shape[0]):
            a, b = S.indptr[r], S.indptr[r + 1]
            d, ix = S.data[a:b], S.indices[a:b]
            if len(d) > k:
                p = np.argpartition(-d, k)[:k]
                d, ix = d[p], ix[p]
            out.append(ix[np.argsort(-d, kind="stable")])
    return out, nnz / Q.shape[0], (time.time() - t0) / Q.shape[0] * 1000


def recall_at(lists, ks=KS):
    return {k: sum(len(np.intersect1d(t, l[:k])) for t, l in zip(truth, lists)) / n_true for k in ks}


def union_recall(list_groups):
    """list_groups: [(lists, k), ...] -> recall and avg size of the union."""
    cap, size = 0, 0
    for i, t in enumerate(truth):
        u = np.unique(np.concatenate([lg[i][:k] for lg, k in list_groups]))
        cap += len(np.intersect1d(t, u))
        size += len(u)
    return cap / n_true, size / len(truth)


def variant(name, analyzer, field, ngram, max_df, min_df=2):
    t0 = time.time()
    v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=min_df, max_df=max_df,
                        sublinear_tf=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b")
    corpus = pd.concat([pool[field], q[field]]).fillna("")
    v.fit(corpus)
    T = v.transform(pool[field].fillna("")).T.tocsr()
    Q = v.transform(q[field].fillna(""))
    fit_s = time.time() - t0
    lists, nnz, ms = topk_lists(Q, T)
    r = recall_at(lists)
    log(f"{name:38s} vocab={len(v.vocabulary_):>8,} fit+transform={fit_s:5.0f}s nnz/row={nnz:>10,.0f} "
        f"query={ms:6.1f} ms/S1 | " + " ".join(f"R@{k}={r[k]*100:5.1f}" for k in KS))
    return lists


res = {}
res["char34_comb_df.01"] = variant("char_wb(3,4) combined, max_df=0.01", "char_wb", "combined_text", (3, 4), 0.01)
res["char34_comb_df.002"] = variant("char_wb(3,4) combined, max_df=0.002", "char_wb", "combined_text", (3, 4), 0.002)
res["word_comb_df.01"] = variant("word combined, max_df=0.01", "word", "combined_text", (1, 1), 0.01)
res["word_name_df.01"] = variant("word name_norm, max_df=0.01", "word", "name_norm", (1, 1), 0.01)
res["char3_name_df.01"] = variant("char_wb(3,3) name_norm, max_df=0.01", "char_wb", "name_norm", (3, 3), 0.01)
res["char3_addr_df.01"] = variant("char_wb(3,3) addr_norm, max_df=0.01", "char_wb", "addr_norm", (3, 3), 0.01)

for combo in [
    [("char34_comb_df.01", 30)],
    [("char34_comb_df.01", 30), ("word_name_df.01", 20)],
    [("char34_comb_df.01", 20), ("char3_name_df.01", 20), ("char3_addr_df.01", 20)],
    [("char34_comb_df.01", 30), ("char3_name_df.01", 20), ("word_comb_df.01", 20)],
    [("char34_comb_df.002", 30), ("char3_name_df.01", 20), ("char3_addr_df.01", 20)],
]:
    r, sz = union_recall([(res[n], k) for n, k in combo])
    log(f"UNION {' + '.join(f'{n}@{k}' for n, k in combo):90s} recall={r*100:6.2f}%  avg size={sz:5.1f}")
