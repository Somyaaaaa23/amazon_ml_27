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


# Chosen with experiments/blocking_lab.py on the TRAINING side (India, 3k S1 vs the full 4.1M pool):
# word TF-IDF on name+address, max_df=0.01: recall@30 91.0%, @50 92.3%, 2.6 ms/S1 (char 3-4 grams:
# 87.7% @30 at 19.2 ms/S1). The name-only view adds candidates whose address differs.
DEFAULT_VIEWS = [
    View("comb", "combined", "word", (1, 1), 0.01, 40),
    View("name", "name_norm", "word", (1, 1), 0.01, 15),
]


_QFN = None
_TVEC = None


def _transform_chunk(texts):
    return _TVEC.transform(texts)


def parallel_transform(vec, texts: pd.Series, workers: int = 6, chunk: int = 200_000) -> sp.csr_matrix:
    global _TVEC
    parts = [texts.iloc[i:i + chunk] for i in range(0, len(texts), chunk)]
    if len(parts) < 2:
        return vec.transform(texts).tocsr()
    _TVEC = vec
    with mp.get_context("fork").Pool(workers) as pool:
        mats = pool.map(_transform_chunk, parts)
    _TVEC = None
    return sp.vstack(mats).tocsr()


def _qrun(args):
    return _QFN(*args)


class CountryIndex:
    def __init__(self, pool: pd.DataFrame, extra: Sequence[pd.DataFrame] = (), views: List[View] = None,
                 fit_sample: int = 1_000_000):
        """pool: normalized S2+S3 records of one country. extra: normalized S1 frames of the same
        country, used only as unlabeled text for fitting IDF statistics."""
        self.pool = pool.reset_index(drop=True)
        self.views = views or DEFAULT_VIEWS
        self.vecs, self.TT = {}, {}
        for v in self.views:
            extra_kw = {"token_pattern": r"(?u)\b\w+\b"} if v.analyzer == "word" else {}
            vec = TfidfVectorizer(analyzer=v.analyzer, ngram_range=v.ngram, min_df=2, max_df=v.max_df,
                                  sublinear_tf=True, dtype=np.float32, **extra_kw)
            corpus = pd.concat([self.pool[v.field]] + [e[v.field] for e in extra], ignore_index=True)
            if len(corpus) > fit_sample:  # vocabulary + IDF from a fixed random sample (speed)
                corpus = corpus.sample(fit_sample, random_state=0)
            vec.fit(corpus)
            self.vecs[v.name] = vec
            self.TT[v.name] = parallel_transform(vec, self.pool[v.field]).T.tocsr()  # features x targets, stored once
        self.space = TokenSpace().fit(self.pool, *extra)
        self.t_mats = self.space.transform(self.pool)

    def _query_batch(self, Q: Dict[str, sp.csr_matrix], lo: int, hi: int):
        """Top-k of every view, unioned per S1; the cosine of every view for every union member is
        read from that view's product row (0 when the two records share no kept n-gram)."""
        prods = []
        for v in self.views:
            S = (Q[v.name][lo:hi] @ self.TT[v.name]).tocsr()
            S.sort_indices()
            prods.append(S)
        s_out, t_out, cos_out = [], [], [[] for _ in self.views]
        for r in range(hi - lo):
            picks = []
            for v, S in zip(self.views, prods):
                a, b = S.indptr[r], S.indptr[r + 1]
                if b - a > v.k:
                    d = S.data[a:b]
                    picks.append(S.indices[a:b][np.argpartition(-d, v.k)[:v.k]])
                else:
                    picks.append(S.indices[a:b])
            u = np.unique(np.concatenate(picks))
            if len(u) == 0:
                continue
            s_out.append(np.full(len(u), lo + r, dtype=np.int32))
            t_out.append(u.astype(np.int32))
            for j, S in enumerate(prods):
                a, b = S.indptr[r], S.indptr[r + 1]
                ix = S.indices[a:b]
                pos = np.searchsorted(ix, u)
                pos_c = np.minimum(pos, max(len(ix) - 1, 0))
                hit = (pos < len(ix)) & (ix[pos_c] == u) if len(ix) else np.zeros(len(u), bool)
                c = np.zeros(len(u), dtype=np.float32)
                c[hit] = S.data[a:b][pos_c[hit]]
                cos_out[j].append(c)
        if not s_out:
            return np.empty(0, np.int32), np.empty(0, np.int32), [np.empty(0, np.float32) for _ in self.views]
        return np.concatenate(s_out), np.concatenate(t_out), [np.concatenate(c) for c in cos_out]

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
        pairs = pd.DataFrame({"s1_idx": np.concatenate([r[0] for r in res]),
                              "t_idx": np.concatenate([r[1] for r in res])})
        for j, v in enumerate(self.views):
            pairs[f"cos_{v.name}"] = np.concatenate([r[2][j] for r in res])
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
    return add_group_features(feats, "s1_idx", score_cols=("cos_comb", "cos_name", "name_tset", "addr_tset"))


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


DECISION_GRID = dict(ts=np.round(np.arange(0.20, 0.91, 0.04), 2),
                     t_tops=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9), rels=(0.0, 0.3, 0.5, 0.7))


def train_model(feats: pd.DataFrame, n_true: np.ndarray, n_s1: int, n_splits: int = 5,
                t_tops=DECISION_GRID["t_tops"], rels=DECISION_GRID["rels"]):
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
        m.fit(X[tr], y[tr], eval_X=(X[va],), eval_y=(y[va],), callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        best_iters.append(m.best_iteration_)
        log(f"  fold {fold + 1}/{n_splits}: best_iter={m.best_iteration_}")
    params, best, _ = tune(groups, feats["cand_key"].to_numpy(), oof, y.astype(bool), n_true, n_s1,
                           ts=DECISION_GRID["ts"], t_tops=t_tops, rels=rels)
    n_est = int(np.mean(best_iters) * 1.1)
    final = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": n_est})
    final.fit(X, y)
    cv = {"oof_macro_f05": best["macro_f05"], "oof_singleton_acc": best["singleton_accuracy"],
          "oof_matches_f05": best["matches_macro_f05"], "n_estimators": n_est, **params}
    log(f"  OOF macro F0.5={best['macro_f05']:.4f} with {params}; final model {n_est} trees")
    return TrainedModel(final, cols, params, cv), oof
