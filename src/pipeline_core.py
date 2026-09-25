"""
Core of the matching pipeline, shared by training, validation (src/validate.py) and test
inference (src/infer_test.py). Everything works per country: the country label is only used to
partition records (the training data has no cross-country matches), never as a feature, so an
unseen country (France) simply gets its own index fitted on its own text.

  normalize()      -> normalized frame (parallel)
  CountryIndex     -> per-country TF-IDF blocking views, fitted on that country's pool + S1 text
  CountryIndex.candidates() -> candidate pairs with the cosine of EVERY view
  build_features() -> pair features + per-S1 context features
  train_model()    -> LightGBM with GroupKFold OOF predictions + tuned decision rule
"""

import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

from src.decision import decide, macro_f05_arrays, one_owner, tune
from src.features import TokenSpace, add_group_features, pair_features
from src.normalization import preprocess_dataframe

NORM_COLS = ["entity_id", "country", "name_norm", "name_core", "addr_norm", "house_number"]
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------
def _norm_chunk(df):
    out = preprocess_dataframe(df)
    return out[NORM_COLS].assign(combined=out["name_norm"] + " " + out["addr_norm"].str.replace(",", " ", regex=False))


def normalize(df: pd.DataFrame, workers: int = 6, chunk: int = 100_000) -> pd.DataFrame:
    if len(df) <= chunk:
        return _norm_chunk(df).reset_index(drop=True)
    parts = [df.iloc[i:i + chunk] for i in range(0, len(df), chunk)]
    with mp.get_context("fork").Pool(workers) as pool:
        out = pool.map(_norm_chunk, parts)
    return pd.concat(out, ignore_index=True)


# -----------------------------------------------------------------------------
# Blocking
# -----------------------------------------------------------------------------
@dataclass
class View:
    name: str
    field: str            # column of the normalized frame
    analyzer: str         # "char_wb" or "word"
    ngram: tuple
    max_df: float         # n-grams in more than this share of documents are dropped (keeps the product sparse)
    k: int                # top-k candidates taken from this view


DEFAULT_VIEWS = [
    View("comb", "combined", "char_wb", (3, 4), 0.01, 30),
    View("name", "name_norm", "char_wb", (3, 3), 0.01, 20),
    View("addr", "addr_norm", "char_wb", (3, 3), 0.01, 20),
]


_QFN = None


def _qrun(args):
    return _QFN(*args)


class CountryIndex:
    def __init__(self, pool: pd.DataFrame, extra: Sequence[pd.DataFrame] = (), views: List[View] = None):
        """pool: normalized S2+S3 records of one country. extra: normalized S1 frames of the same
        country, used only as unlabeled text for fitting IDF statistics."""
        self.pool = pool.reset_index(drop=True)
        self.views = views or DEFAULT_VIEWS
        self.vecs, self.T = {}, {}
        for v in self.views:
            vec = TfidfVectorizer(analyzer=v.analyzer, ngram_range=v.ngram, min_df=2, max_df=v.max_df,
                                  sublinear_tf=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b")
            vec.fit(pd.concat([self.pool[v.field]] + [e[v.field] for e in extra]))
            self.vecs[v.name] = vec
            self.T[v.name] = vec.transform(self.pool[v.field]).tocsr()          # rows = targets
        self.TT = {n: m.T.tocsr() for n, m in self.T.items()}                  # for Q @ TT
        self.space = TokenSpace().fit(self.pool, *extra)
        self.t_mats = self.space.transform(self.pool)
        self.is_s2 = self.pool["entity_id"].str.startswith("S2-").to_numpy()

    def _query_batch(self, Q: Dict[str, sp.csr_matrix], lo: int, hi: int):
        s_list, t_list = [], []
        for v in self.views:
            S = (Q[v.name][lo:hi] @ self.TT[v.name]).tocsr()
            for r in range(S.shape[0]):
                a, b = S.indptr[r], S.indptr[r + 1]
                if a == b:
                    continue
                d, ix = S.data[a:b], S.indices[a:b]
                if len(d) > v.k:
                    ix = ix[np.argpartition(-d, v.k)[:v.k]]
                s_list.append(np.full(len(ix), lo + r, dtype=np.int32))
                t_list.append(ix.astype(np.int32))
        if not s_list:
            return np.empty(0, np.int32), np.empty(0, np.int32)
        return np.concatenate(s_list), np.concatenate(t_list)

    def candidates(self, s1: pd.DataFrame, batch: int = 200, workers: int = 4) -> pd.DataFrame:
        """Union of the top-k of every view; returns pairs with the cosine of every view."""
        Q = {v.name: self.vecs[v.name].transform(s1[v.field]).tocsr() for v in self.views}
        spans = [(lo, min(lo + batch, len(s1))) for lo in range(0, len(s1), batch)]
        global _QFN
        _QFN = lambda lo, hi: self._query_batch(Q, lo, hi)
        if workers > 1 and len(spans) >= 2 * workers:
            with mp.get_context("fork").Pool(workers) as pool:
                res = pool.map(_qrun, spans, chunksize=1)
        else:
            res = [_qrun(sp_) for sp_ in spans]
        _QFN = None
        s_idx = np.concatenate([r[0] for r in res]) if res else np.empty(0, np.int32)
        t_idx = np.concatenate([r[1] for r in res]) if res else np.empty(0, np.int32)
        key = np.unique(s_idx.astype(np.int64) * (len(self.pool) + 1) + t_idx)
        s_idx, t_idx = (key // (len(self.pool) + 1)).astype(np.int32), (key % (len(self.pool) + 1)).astype(np.int32)
        pairs = pd.DataFrame({"s1_idx": s_idx, "t_idx": t_idx})
        for v in self.views:  # cosine of every view for every candidate (vectors are L2-normalized)
            cos = np.empty(len(pairs), dtype=np.float32)
            for lo in range(0, len(pairs), 500_000):
                a, b = s_idx[lo:lo + 500_000], t_idx[lo:lo + 500_000]
                cos[lo:lo + len(a)] = np.asarray(Q[v.name][a].multiply(self.T[v.name][b]).sum(axis=1)).ravel()
            pairs[f"cos_{v.name}"] = cos
        return pairs


# -----------------------------------------------------------------------------
# Features
# -----------------------------------------------------------------------------
def build_features(index: CountryIndex, s1: pd.DataFrame, pairs: pd.DataFrame, chunk: int = 1_000_000) -> pd.DataFrame:
    s1_mats = index.space.transform(s1)
    out = []
    for lo in range(0, len(pairs), chunk):
        p = pairs.iloc[lo:lo + chunk]
        f = pair_features(s1, index.pool, s1_mats, index.t_mats, index.space,
                          p["s1_idx"].to_numpy(), p["t_idx"].to_numpy())
        f.index = p.index
        out.append(f)
    feats = pd.concat([pairs] + [pd.concat(out)], axis=1) if out else pairs
    cos_cols = [c for c in pairs.columns if c.startswith("cos_")]
    feats["cos_max"] = feats[cos_cols].max(axis=1)
    feats["cos_mean"] = feats[cos_cols].mean(axis=1)
    return add_group_features(feats, "s1_idx", score_cols=("cos_comb", "cos_name", "cos_addr", "name_tset", "addr_tset"))


def feature_columns(df: pd.DataFrame) -> List[str]:
    skip = {"s1_idx", "t_idx", "label", "s1_key", "cand_key", "prob", "fold"}
    return [c for c in df.columns if c not in skip]


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  n_estimators=2000, random_state=42, verbose=-1, n_jobs=8)


@dataclass
class TrainedModel:
    booster: lgb.LGBMClassifier
    features: List[str]
    decision: Dict[str, float]
    cv: Dict[str, float] = field(default_factory=dict)

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        return self.booster.predict_proba(feats[self.features].to_numpy(np.float32))[:, 1]


def train_model(feats: pd.DataFrame, n_true: np.ndarray, n_s1: int, n_splits: int = 5,
                t_tops=(0.0,), rels=(0.0,)):
    """feats: training pairs with 'label', 's1_key' (0..n_s1-1, unique over countries) and 'cand_key'
    (unique target id). n_true[s1_key] = number of true matches (including ones blocking missed).
    Returns (TrainedModel, oof probabilities)."""
    cols = feature_columns(feats)
    X = feats[cols].to_numpy(np.float32)
    y = feats["label"].to_numpy()
    groups = feats["s1_key"].to_numpy()
    oof = np.zeros(len(y), dtype=np.float64)
    best_iters = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        best_iters.append(m.best_iteration_)
        log(f"  fold {fold + 1}/{n_splits}: best_iter={m.best_iteration_}")
    params, best, _ = tune(groups, feats["cand_key"].to_numpy(), oof, y.astype(bool), n_true, n_s1,
                           t_tops=t_tops, rels=rels)
    n_est = int(np.mean(best_iters) * 1.1)
    final = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": n_est})
    final.fit(X, y)
    cv = {"oof_macro_f05": best["macro_f05"], "oof_singleton_acc": best["singleton_accuracy"],
          "oof_matches_f05": best["matches_macro_f05"], "n_estimators": n_est, **params}
    log(f"  OOF macro F0.5={best['macro_f05']:.4f} with {params}; final model {n_est} trees")
    return TrainedModel(final, cols, params, cv), oof
