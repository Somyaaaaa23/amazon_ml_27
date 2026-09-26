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
from rapidfuzz import fuzz, process
from src.features import Records, TokenSpace, add_group_features, pair_features
from src.normalization import preprocess_dataframe

NORM_COLS = ["entity_id", "country", "name_norm", "name_core", "name_compact", "addr_norm", "house_number"]  # + addr_key, name_key
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------
def _sorted_key(text: str) -> str:
    return " ".join(sorted(set(text.replace(",", " ").split())))


def _norm_chunk(df):
    out = preprocess_dataframe(df)
    out = out[NORM_COLS].assign(combined=out["name_norm"] + " " + out["addr_norm"].str.replace(",", " ", regex=False))
    # order-invariant keys: "BIG LAKE, MN, 1373 MANITOU ST" == "1373 MANITOU ST, BIG LAKE, MN"
    out["addr_key"] = [_sorted_key(a) for a in out["addr_norm"]]
    out["name_key"] = [_sorted_key(n) for n in out["name_core"]]
    return out


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
# + character 3-grams of the name: tolerates typos ("Mfedia", "Endccorinology") that word views miss
VIEW_SETS = {
    "default": DEFAULT_VIEWS,
    "charname": DEFAULT_VIEWS + [View("cname", "name_norm", "char_wb", (3, 3), 0.01, 15)],
}


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
                 fit_sample: Optional[int] = None, expand: bool = True, hop2: bool = False):
        """pool: normalized S2+S3 records of one country. extra: normalized S1 frames of the same
        country, used only as unlabeled text for fitting IDF statistics."""
        self.pool = pool.reset_index(drop=True)
        extra = [e for e in extra if len(e)]
        self.views = views or DEFAULT_VIEWS
        self.vecs, self.TT = {}, {}
        for v in self.views:
            extra_kw = {"token_pattern": r"(?u)\b\w+\b"} if v.analyzer == "word" else {}
            vec = TfidfVectorizer(analyzer=v.analyzer, ngram_range=v.ngram, min_df=2, max_df=v.max_df,
                                  sublinear_tf=True, dtype=np.float32, **extra_kw)
            corpus = pd.concat([self.pool[v.field]] + [e[v.field] for e in extra], ignore_index=True)
            if fit_sample and len(corpus) > fit_sample:  # optional: vocabulary + IDF from a random sample
                corpus = corpus.sample(fit_sample, random_state=0)
            vec.fit(corpus)
            self.vecs[v.name] = vec
            self.TT[v.name] = parallel_transform(vec, self.pool[v.field]).T.tocsr()  # features x targets, stored once
        self.T = {n: m.T.tocsr() for n, m in self.TT.items()}   # targets x features (word views are small)
        self.space = TokenSpace().fit(self.pool, *extra)
        self.t_mats = self.space.transform(self.pool)
        self.rec = Records(self.pool)                           # column arrays computed once
        # exact-twin groups for sibling expansion (records of one business often share the exact
        # normalized address or compact name); large groups are generic and skipped
        self.expand = expand
        self.hop2 = hop2
        self.addr_groups = _twin_groups(self.pool["addr_key"], min_len=10)
        self.compact_groups = _twin_groups(self.pool["name_compact"], min_len=6)
        self.namekey_groups = _twin_groups(self.pool["name_key"], min_len=6)

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
        pairs["expanded"] = np.float32(0)
        if self.expand and len(pairs):
            pairs = self._expand(s1, Q, pairs)
        if self.hop2 and len(pairs):
            pairs = self._second_hop(s1, Q, pairs, workers=workers)
        return pairs

    def _second_hop(self, s1: pd.DataFrame, Q, pairs: pd.DataFrame, n_anchor: int = 2, k2: int = 10,
                    max_new: int = 15, workers: int = 4, batch: int = 400) -> pd.DataFrame:
        """Records of one business resemble each other more than they resemble S1: use each S1's most
        reliable candidates (by name+address similarity) as queries and add their nearest neighbours."""
        s_idx, t_idx = pairs["s1_idx"].to_numpy(), pairs["t_idx"].to_numpy()
        name_ts = process.cpdist(s1["name_norm"].to_numpy(dtype=object)[s_idx], self.rec.obj["name_norm"][t_idx],
                                 scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        addr_ts = process.cpdist(s1["addr_norm"].to_numpy(dtype=object)[s_idx], self.rec.obj["addr_norm"][t_idx],
                                 scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        score = np.where((name_ts >= 80) | (addr_ts >= 80), name_ts + addr_ts, -1.0)
        rank = group_rank(s_idx, score)
        sel = (rank <= n_anchor) & (score >= 0)
        sa, ta = s_idx[sel], t_idx[sel]
        if not len(sa):
            return pairs
        A = self.T["comb"][ta]
        spans = [(lo, min(lo + batch, len(ta))) for lo in range(0, len(ta), batch)]

        def run(lo, hi):
            S = (A[lo:hi] @ self.TT["comb"]).tocsr()
            rows, cols = [], []
            for r in range(S.shape[0]):
                a, b = S.indptr[r], S.indptr[r + 1]
                d, ix = S.data[a:b], S.indices[a:b]
                if len(d) > k2:
                    ix = ix[np.argpartition(-d, k2)[:k2]]
                rows.append(np.full(len(ix), lo + r, dtype=np.int64))
                cols.append(ix.astype(np.int64))
            return (np.concatenate(rows) if rows else np.empty(0, np.int64),
                    np.concatenate(cols) if cols else np.empty(0, np.int64))

        global _QFN
        _QFN = run
        if workers > 1 and len(spans) >= 2 * workers:
            with mp.get_context("fork").Pool(workers) as pool:
                res = pool.map(_qrun, spans, chunksize=1)
        else:
            res = [_qrun(sp_) for sp_ in spans]
        _QFN = None
        arow = np.concatenate([r[0] for r in res])
        nbr = np.concatenate([r[1] for r in res])
        N = len(self.pool) + 1
        key_new = np.unique(sa[arow].astype(np.int64) * N + nbr)
        key_new = key_new[~np.isin(key_new, s_idx.astype(np.int64) * N + t_idx)]
        ns, nt = (key_new // N).astype(np.int32), (key_new % N).astype(np.int32)
        first = np.r_[0, np.flatnonzero(ns[1:] != ns[:-1]) + 1] if len(ns) else np.empty(0, int)
        rk = np.arange(len(ns)) - np.repeat(first, np.diff(np.r_[first, len(ns)])) if len(ns) else np.empty(0, int)
        keep = rk < max_new
        ns, nt = ns[keep], nt[keep]
        add = pd.DataFrame({"s1_idx": ns, "t_idx": nt, **self._cos(Q, ns, nt)})
        add["expanded"] = np.float32(0)
        pairs = pairs.assign(hop2=np.float32(0))
        add["hop2"] = np.float32(1)
        return pd.concat([pairs, add], ignore_index=True)

    def _cos(self, Q, s_idx, t_idx):
        out = {}
        for v in self.views:
            c = np.empty(len(s_idx), dtype=np.float32)
            for lo in range(0, len(s_idx), 500_000):
                a, b = s_idx[lo:lo + 500_000], t_idx[lo:lo + 500_000]
                c[lo:lo + len(a)] = np.asarray(Q[v.name][a].multiply(self.T[v.name][b]).sum(axis=1)).ravel()
            out[f"cos_{v.name}"] = c
        return out

    def _expand(self, s1: pd.DataFrame, Q, pairs: pd.DataFrame, max_new: int = 20) -> pd.DataFrame:
        """Add exact twins (same normalized address or same compact name) of 'anchor' candidates,
        i.e. candidates whose name and address both closely match the S1 record."""
        s_idx, t_idx = pairs["s1_idx"].to_numpy(), pairs["t_idx"].to_numpy()
        n1 = s1["name_norm"].to_numpy(dtype=object)[s_idx]
        n2 = self.rec.obj["name_norm"][t_idx]
        a1 = s1["addr_norm"].to_numpy(dtype=object)[s_idx]
        a2 = self.rec.obj["addr_norm"][t_idx]
        name_ts = process.cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        addr_ts = process.cpdist(a1, a2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        anchor = (name_ts >= 90) & (addr_ts >= 80)
        sa, ta = s_idx[anchor], t_idx[anchor]
        new_s, new_t = [], []
        for codes, order, ptr in [self.addr_groups, self.compact_groups, self.namekey_groups]:
            c = codes[ta]
            ok = c >= 0
            if not ok.any():
                continue
            c, ss = c[ok], sa[ok]
            lens = ptr[c + 1] - ptr[c]
            rep_s = np.repeat(ss, lens)
            offs = np.repeat(ptr[c], lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
            new_s.append(rep_s)
            new_t.append(order[offs])
        if not new_s:
            return pairs
        N = len(self.pool) + 1
        key_new = np.unique(np.concatenate(new_s).astype(np.int64) * N + np.concatenate(new_t))
        key_old = s_idx.astype(np.int64) * N + t_idx
        key_new = key_new[~np.isin(key_new, key_old)]
        ns, nt = (key_new // N).astype(np.int32), (key_new % N).astype(np.int32)
        # cap additions per S1 (keys are sorted by S1, so keep the first max_new of each)
        first = np.r_[0, np.flatnonzero(ns[1:] != ns[:-1]) + 1]
        rank = np.arange(len(ns)) - np.repeat(first, np.diff(np.r_[first, len(ns)]))
        keep = rank < max_new
        ns, nt = ns[keep], nt[keep]
        add = pd.DataFrame({"s1_idx": ns, "t_idx": nt, **self._cos(Q, ns, nt)})
        add["expanded"] = np.float32(1)
        return pd.concat([pairs, add], ignore_index=True)


def _twin_groups(values: pd.Series, min_len: int, max_size: int = 20):
    """codes[i] = group id of record i (-1 if its value is short, unique or in a group > max_size);
    (order, ptr) list the members of each group: order[ptr[g]:ptr[g+1]]."""
    codes, _ = pd.factorize(values.where(values.str.len() >= min_len))   # missing -> -1
    if (codes >= 0).any():
        sizes = np.bincount(codes[codes >= 0])
        good = (sizes >= 2) & (sizes <= max_size)
        codes = np.where((codes >= 0) & good[np.maximum(codes, 0)], codes, -1)
    return _compact_codes(codes)


def _compact_codes(codes):
    keep = codes >= 0
    uniq, inv = np.unique(codes[keep], return_inverse=True)
    out = np.full(len(codes), -1, dtype=np.int64)
    out[keep] = inv
    idx = np.flatnonzero(keep)
    order = idx[np.argsort(inv, kind="stable")]
    ptr = np.r_[0, np.cumsum(np.bincount(inv, minlength=len(uniq)))]
    return out, order, ptr


# -----------------------------------------------------------------------------
# Features
# -----------------------------------------------------------------------------
class RivalIndex:
    """All S1 records of one country (unlabeled), grouped by name, to tell how ambiguous an S1 name is
    and whether a candidate fits a rival S1 with the same name better. At training time this is every
    training S1 of the country, at test time every test S1 of the country."""

    def __init__(self, s1_all: pd.DataFrame, max_rivals: int = 20):
        self.ids = {k: i for i, k in enumerate(s1_all["entity_id"])}
        self.addr = s1_all["addr_norm"].to_numpy(dtype=object)
        self.name = s1_all["name_norm"].to_numpy(dtype=object)
        tokkey = s1_all["name_core"].map(lambda x: " ".join(sorted(set(x.split()))))
        self.max_rivals = max_rivals
        self.groups = {}
        for key, values in [("compact", s1_all["name_compact"]), ("tokens", tokkey)]:
            codes, _ = pd.factorize(values.where(values.str.len() > 0))
            self.groups[key] = _compact_codes(codes)

    def features(self, s1: pd.DataFrame, s1_idx: np.ndarray, t_name: np.ndarray, t_addr: np.ndarray,
                 self_addr_tset: np.ndarray) -> pd.DataFrame:
        rows = np.array([self.ids.get(k, -1) for k in s1["entity_id"]])[s1_idx]
        n = len(s1_idx)
        f = {}
        for key, (codes, order, ptr) in self.groups.items():
            c = np.where(rows >= 0, codes[np.maximum(rows, 0)], -1)
            size = np.where(c >= 0, ptr[np.maximum(c, 0) + 1] - ptr[np.maximum(c, 0)], 1)
            f[f"n_rival_{key}"] = (size - 1).astype(np.float32)
        # address / name of the candidate vs rival S1s sharing the compact name
        codes, order, ptr = self.groups["compact"]
        c = np.where(rows >= 0, codes[np.maximum(rows, 0)], -1)
        lens = np.where(c >= 0, np.minimum(ptr[np.maximum(c, 0) + 1] - ptr[np.maximum(c, 0)], self.max_rivals + 1), 0)
        has = lens > 1
        pr = np.flatnonzero(has)
        rep = np.repeat(pr, lens[pr])
        offs = np.repeat(ptr[c[pr]], lens[pr]) + (np.arange(lens[pr].sum()) - np.repeat(np.cumsum(lens[pr]) - lens[pr], lens[pr]))
        riv = order[offs]
        not_self = riv != rows[rep]
        rep, riv = rep[not_self], riv[not_self]
        addr_best = np.full(n, np.nan, dtype=np.float32)
        name_best = np.full(n, np.nan, dtype=np.float32)
        if len(rep):
            a_sim = process.cpdist(t_addr[rep], self.addr[riv], scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100
            a_sim[pd.Series(t_addr[rep]).str.len().to_numpy() == 0] = np.nan
            n_sim = process.cpdist(t_name[rep], self.name[riv], scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100
            tmp = np.full(n, -1.0, dtype=np.float32)
            np.fmax.at(tmp, rep, np.nan_to_num(a_sim, nan=-1.0))
            addr_best = np.where(tmp >= 0, tmp, np.nan).astype(np.float32)
            tmp = np.full(n, -1.0, dtype=np.float32)
            np.fmax.at(tmp, rep, n_sim)
            name_best = np.where(tmp >= 0, tmp, np.nan).astype(np.float32)
        f["rival_addr_best"] = addr_best
        f["rival_name_best"] = name_best
        f["self_minus_rival_addr"] = (np.nan_to_num(self_addr_tset, nan=0.0) - addr_best).astype(np.float32)
        return pd.DataFrame(f)


def build_features(index: CountryIndex, s1: pd.DataFrame, pairs: pd.DataFrame, chunk: int = 1_000_000,
                   rivals: Optional["RivalIndex"] = None) -> pd.DataFrame:
    s1_mats = index.space.transform(s1)
    s1_rec = Records(s1)
    out = []
    for lo in range(0, len(pairs), chunk):
        p = pairs.iloc[lo:lo + chunk]
        f = pair_features(s1_rec, index.rec, s1_mats, index.t_mats, index.space,
                          p["s1_idx"].to_numpy(), p["t_idx"].to_numpy())
        f.index = p.index
        out.append(f)
    feats = pd.concat([pairs] + [pd.concat(out)], axis=1) if out else pairs
    cos_cols = [c for c in pairs.columns if c.startswith("cos_")]
    t = feats["t_idx"].to_numpy()
    for col, src in [("_t_name", "name_norm"), ("_t_addr", "addr_norm"), ("_t_compact", "name_compact"),
                     ("_t_addrkey", "addr_key"), ("_t_namekey", "name_key"), ("_t_hn", "house_number")]:
        feats[col] = index.rec.obj[src][t]
    feats["cos_max"] = feats[cos_cols].max(axis=1)
    feats["cos_mean"] = feats[cos_cols].mean(axis=1)
    # how many pool records share the candidate's exact compact name (size of its twin group)
    tc = index.compact_groups[0][t]
    feats["t_compact_twins"] = np.where(tc >= 0, np.diff(index.compact_groups[2])[np.maximum(tc, 0)], 1).astype(np.float32)
    if rivals is not None:
        rf = rivals.features(s1, feats["s1_idx"].to_numpy(), feats["_t_name"].to_numpy(dtype=object),
                             feats["_t_addr"].to_numpy(dtype=object), feats["addr_tset"].to_numpy())
        rf.index = feats.index
        feats = pd.concat([feats, rf], axis=1)
    return add_group_features(feats, "s1_idx", score_cols=("cos_comb", "cos_name", "name_tset", "addr_tset"))


def feature_columns(df: pd.DataFrame) -> List[str]:
    skip = {"s1_idx", "t_idx", "label", "s1_key", "cand_key", "prob", "fold"}
    return [c for c in df.columns if c not in skip and not c.startswith("_") and pd.api.types.is_numeric_dtype(df[c])]


# -----------------------------------------------------------------------------
# Stage-2 "sibling" features: the S2/S3 records of one business resemble each other, so a candidate
# that looks like the entity's most confident OTHER candidates is likely a match too (and one that
# looks like none of them is suspicious). Uses stage-1 probabilities p1 (OOF on training pairs).
# -----------------------------------------------------------------------------
def sibling_features(key: np.ndarray, p1: np.ndarray, t_name: np.ndarray, t_addr: np.ndarray,
                     t_compact: np.ndarray, is_s2: np.ndarray, n_slots: int = 4,
                     t_addrkey: Optional[np.ndarray] = None, t_namekey: Optional[np.ndarray] = None,
                     t_hn: Optional[np.ndarray] = None) -> pd.DataFrame:
    n = len(key)
    order = np.lexsort((-p1, key))
    sk = key[order]
    starts = np.r_[0, np.flatnonzero(sk[1:] != sk[:-1]) + 1]
    sizes = np.diff(np.r_[starts, n])
    gid = np.repeat(np.arange(len(starts)), sizes)
    pos = np.arange(n)
    rank = pos - starts[gid]                                   # 0 = most probable in its group

    slots = starts[gid][:, None] + np.arange(n_slots + 1)[None, :]
    valid = (np.arange(n_slots + 1)[None, :] < sizes[gid][:, None]) & (slots != pos[:, None])
    r_pos = np.repeat(pos, valid.sum(axis=1))
    a_pos = slots[valid]
    r, a = order[r_pos], order[a_pos]                          # original row ids: row, anchor
    w = p1[a]

    name_sim = process.cpdist(t_name[r], t_name[a], scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100
    has_addr = pd.Series(t_addr).str.len().to_numpy() > 0
    has_compact = pd.Series(t_compact).str.len().to_numpy() > 0
    addr_ok = has_addr[r] & has_addr[a]
    addr_sim = process.cpdist(t_addr[r], t_addr[a], scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100
    addr_sim[~addr_ok] = 0.0
    addr_eq = addr_ok & (t_addr[r] == t_addr[a])
    comp_eq = (t_compact[r] == t_compact[a]) & has_compact[r]
    conf = w >= 0.5

    def agg_max(vals):
        out = np.full(n, 0.0, dtype=np.float32)
        np.maximum.at(out, r, vals.astype(np.float32))
        return out

    def agg_sum(vals):
        return np.bincount(r, weights=vals.astype(np.float64), minlength=n).astype(np.float32)

    f = {
        "p1": p1.astype(np.float32),
        "p1_rank": np.empty(n, dtype=np.float32),
        "sib_name_wmax": agg_max(w * name_sim),
        "sib_addr_wmax": agg_max(w * addr_sim),
        "sib_name_conf_max": agg_max(np.where(conf, name_sim, 0)),
        "sib_addr_conf_max": agg_max(np.where(conf, addr_sim, 0)),
        "sib_addr_eq_conf": agg_sum(conf & addr_eq),
        "sib_compact_eq_conf": agg_sum(conf & comp_eq),
        "sib_name_hi_conf": agg_sum(conf & (name_sim >= 0.9)),
        "sib_addr_hi_conf": agg_sum(conf & (addr_sim >= 0.9)),
    }
    for name, arr in [("addrkey", t_addrkey), ("namekey", t_namekey), ("hn", t_hn)]:
        if arr is None:
            continue
        ok = pd.Series(arr).str.len().to_numpy() > 0
        eq = ok[r] & ok[a] & (arr[r] == arr[a])
        f[f"sib_{name}_eq_conf"] = agg_sum(conf & eq)
        f[f"sib_{name}_eq_wmax"] = agg_max(w * eq)
        if name == "hn":   # house number differs from a confident sibling that has one
            f["sib_hn_conflict_conf"] = agg_sum(conf & ok[r] & ok[a] & (arr[r] != arr[a]))
    f["p1_rank"][order] = rank + 1
    # group-level context of p1
    first = order[starts]                                      # most probable row of each group
    second_p = np.where(sizes > 1, p1[order[np.minimum(starts + 1, n - 1)]], 0.0)
    best_other = np.where(order[starts][gid] == order, second_p[gid], p1[first][gid])
    bo = np.empty(n, dtype=np.float32)
    bo[order] = best_other
    f["p1_best_other"] = bo
    f["p1_gap_best_other"] = (p1 - bo).astype(np.float32)
    g_conf = np.bincount(gid, weights=(p1[order] >= 0.5), minlength=len(starts))
    g_sum = np.bincount(gid, weights=p1[order], minlength=len(starts))
    g_conf_s2 = np.bincount(gid, weights=(p1[order] >= 0.5) & is_s2[order], minlength=len(starts))
    for name, g in [("n_conf", g_conf), ("sum_p1", g_sum), ("n_conf_s2", g_conf_s2), ("n_conf_s3", g_conf - g_conf_s2)]:
        v = np.empty(n, dtype=np.float32)
        v[order] = g[gid]
        f[name] = v
    same_src = np.where(is_s2, f["n_conf_s2"], f["n_conf_s3"]) - (p1 >= 0.5)
    f["n_conf_same_source_other"] = same_src.astype(np.float32)
    return pd.DataFrame(f)


def sibling_inputs(feats: pd.DataFrame, key_col: str, p1: np.ndarray) -> pd.DataFrame:
    opt = lambda c: feats[c].to_numpy(dtype=object) if c in feats else None
    return sibling_features(feats[key_col].to_numpy(), p1, feats["_t_name"].to_numpy(dtype=object),
                            feats["_t_addr"].to_numpy(dtype=object), feats["_t_compact"].to_numpy(dtype=object),
                            feats["is_s2"].to_numpy() > 0.5, t_addrkey=opt("_t_addrkey"),
                            t_namekey=opt("_t_namekey"), t_hn=opt("_t_hn"))


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  n_estimators=2000, random_state=42, verbose=-1, n_jobs=8)

DECISION_GRID = dict(ts=np.round(np.arange(0.20, 0.91, 0.04), 2),
                     t_tops=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9), rels=(0.0, 0.3, 0.5, 0.7))


def fit_oof(X, y, groups, n_splits=5, tag=""):
    """GroupKFold OOF probabilities + a final model on all rows (trees = 1.1 x mean best iteration)."""
    oof = np.zeros(len(y), dtype=np.float64)
    best_iters = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(X[tr], y[tr], eval_X=(X[va],), eval_y=(y[va],), callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        best_iters.append(m.best_iteration_)
        log(f"  {tag} fold {fold + 1}/{n_splits}: best_iter={m.best_iteration_}")
    n_est = int(np.mean(best_iters) * 1.1)
    final = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": n_est})
    final.fit(X, y)
    return oof, final, n_est


def group_rank(key: np.ndarray, p: np.ndarray) -> np.ndarray:
    """1-based rank of p (descending) within each key group."""
    order = np.lexsort((-p, key))
    sk = key[order]
    starts = np.r_[0, np.flatnonzero(sk[1:] != sk[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(sk)])
    rank = np.empty(len(p), dtype=np.int64)
    rank[order] = np.arange(len(p)) - np.repeat(starts, sizes) + 1
    return rank


@dataclass
class TrainedModel:
    stage1: lgb.LGBMClassifier
    cols1: List[str]
    stage2: Optional[lgb.LGBMClassifier]
    cols2: List[str]
    decision: Dict[str, float]
    cv: Dict[str, float] = field(default_factory=dict)
    prune_k: Optional[int] = None     # stage 1 keeps the top-k candidates per S1 for stage 2

    def predict(self, feats: pd.DataFrame, key_col: str = "s1_idx"):
        """Returns (probability, kept): kept marks the pairs the final model scored (the candidate
        set written to candidate_pairs.tsv); probability is 0 for pruned pairs."""
        X1 = feats[self.cols1].to_numpy(np.float32)
        p1 = self.stage1.predict_proba(X1)[:, 1]
        kept = np.ones(len(p1), dtype=bool)
        if self.stage2 is None:
            return p1, kept
        if self.prune_k:
            kept = group_rank(feats[key_col].to_numpy(), p1) <= self.prune_k
        sub = feats[kept]
        sib = sibling_inputs(sub, key_col, p1[kept])
        p = np.zeros(len(p1))
        p[kept] = self.stage2.predict_proba(np.hstack([X1[kept], sib[self.cols2].to_numpy(np.float32)]))[:, 1]
        return p, kept


def train_model(feats: pd.DataFrame, n_true: np.ndarray, n_s1: int, n_splits: int = 5, two_stage: bool = True,
                prune_k: Optional[int] = None):
    """feats: training pairs with 'label', 's1_key' (0..n_s1-1, unique over countries), 'cand_key'
    (unique target id) and the _t_* target strings. n_true[s1_key] = number of true matches
    (including ones blocking missed). The decision rule is tuned on the OOF probabilities of the
    last stage. Returns (TrainedModel, oof probabilities)."""
    cols1 = feature_columns(feats)
    X1 = feats[cols1].to_numpy(np.float32)
    y = feats["label"].to_numpy()
    groups = feats["s1_key"].to_numpy()
    cand = feats["cand_key"].to_numpy()
    oof1, m1, n1 = fit_oof(X1, y, groups, n_splits, "stage1")
    params1, best1, _ = tune(groups, cand, oof1, y.astype(bool), n_true, n_s1, ts=DECISION_GRID["ts"],
                             t_tops=DECISION_GRID["t_tops"], rels=DECISION_GRID["rels"])
    log(f"  stage1 OOF macro F0.5={best1['macro_f05']:.4f} singleton_acc={best1['singleton_accuracy']:.4f} with {params1}")
    cv = {"stage1_oof_macro_f05": best1["macro_f05"], "stage1_oof_singleton_acc": best1["singleton_accuracy"]}
    if not two_stage:
        cv.update({"oof_macro_f05": best1["macro_f05"], "oof_singleton_acc": best1["singleton_accuracy"],
                   "oof_matches_f05": best1["matches_macro_f05"], **params1})
        return TrainedModel(m1, cols1, None, [], params1, cv), oof1
    keep = np.ones(len(y), dtype=bool)
    if prune_k:
        keep = group_rank(groups, oof1) <= prune_k
        log(f"  prune to top-{prune_k} by stage-1 OOF: keeps {keep.mean():.3f} of pairs and "
            f"{y[keep].sum() / max(y.sum(), 1):.4f} of blocked true pairs")
    feats_k, X1k, yk, gk, ck = feats[keep], X1[keep], y[keep], groups[keep], cand[keep]
    sib = sibling_inputs(feats_k, "s1_key", oof1[keep])
    cols2 = list(sib.columns)
    X2 = np.hstack([X1k, sib.to_numpy(np.float32)])
    oof2, m2, n2 = fit_oof(X2, yk, gk, n_splits, "stage2")
    params2, best2, _ = tune(gk, ck, oof2, yk.astype(bool), n_true, n_s1, ts=DECISION_GRID["ts"],
                             t_tops=DECISION_GRID["t_tops"], rels=DECISION_GRID["rels"])
    log(f"  stage2 OOF macro F0.5={best2['macro_f05']:.4f} singleton_acc={best2['singleton_accuracy']:.4f} with {params2}")
    cv.update({"oof_macro_f05": best2["macro_f05"], "oof_singleton_acc": best2["singleton_accuracy"],
               "oof_matches_f05": best2["matches_macro_f05"], "n_estimators": [n1, n2], **params2})
    oof = np.zeros(len(y))
    oof[keep] = oof2
    return TrainedModel(m1, cols1, m2, cols2, params2, cv, prune_k), oof
