"""
Stage 3: vectorized pair features.

Pairs are given as index arrays (s1_idx, t_idx) into a normalized S1 frame and a normalized
target-pool frame of ONE country. String similarities use rapidfuzz.process.cpdist (C++,
multi-threaded); token / IDF / address-component overlaps use sparse binary matrices, so no
Python loop runs per pair. A similarity is NaN when either side is empty (LightGBM treats NaN
as missing), instead of a misleading 0 or 1.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, LCSseq, Levenshtein
from sklearn.feature_extraction.text import HashingVectorizer

TOKEN_PATTERN = r"(?u)\b\w+\b"


def _components(addr: str) -> List[str]:
    return [c.strip() for c in addr.split(",") if c.strip()]


class TokenSpace:
    """Binary hashed token matrices + IDF for one country, with document frequencies counted on that
    country's own text (pool + S1), so an unseen country (France) gets its own statistics.
    Hashing (2^22 buckets) needs no vocabulary pass, so transforms run in parallel."""

    FIELDS = {"name": ("name_norm", TOKEN_PATTERN), "addr": ("addr_norm", TOKEN_PATTERN),
              "comp": ("addr_norm", None), "num": ("addr_norm", r"\b\d+[a-z]?\b")}

    def __init__(self, n_features: int = 2 ** 22):
        self.vecs = {}
        for k, (_, pat) in self.FIELDS.items():
            kw = {"analyzer": _components} if pat is None else {"token_pattern": pat}
            self.vecs[k] = HashingVectorizer(n_features=n_features, binary=True, norm=None, alternate_sign=False,
                                             lowercase=False, dtype=np.float32, **kw)

    def fit(self, *frames: pd.DataFrame) -> "TokenSpace":
        from src.pipeline_core import parallel_transform
        frames = [f for f in frames if len(f)]      # e.g. a country with no training S1 in a proxy run
        n_docs = sum(len(f) for f in frames)
        for k, field in [("name", "name_norm"), ("addr", "addr_norm")]:
            df = sum(np.asarray(parallel_transform(self.vecs[k], f[field]).sum(axis=0)).ravel() for f in frames)
            setattr(self, f"{k}_idf", (np.log((1.0 + n_docs) / (1.0 + df)) + 1.0).astype(np.float32))
        return self

    def transform(self, frame: pd.DataFrame) -> Dict[str, sp.csr_matrix]:
        from src.pipeline_core import parallel_transform
        return {k: parallel_transform(self.vecs[k], frame[field]) for k, (field, _) in self.FIELDS.items()}


def _pairwise(scorer, a, b, scale=1.0):
    out = process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
    return out / scale if scale != 1.0 else out


def _weighted_overlap(A, B, idf):
    """For binary row-aligned matrices A, B: IDF mass shared, IDF mass of A, of B, max shared IDF."""
    inter = A.multiply(B).tocsr()
    shared = inter @ idf
    mass_a, mass_b = A @ idf, B @ idf
    w = inter.multiply(idf[None, :]).tocsr()
    max_shared = np.zeros(A.shape[0], dtype=np.float32)
    nz = np.diff(w.indptr) > 0
    if nz.any():
        max_shared[nz] = np.maximum.reduceat(w.data, w.indptr[:-1][nz])
    return shared.astype(np.float32), mass_a.astype(np.float32), mass_b.astype(np.float32), max_shared


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan).astype(np.float32)


class Records:
    """Column arrays + string lengths of a normalized frame, computed once. Building these per call
    for a 4-6M record pool was the main per-chunk cost of test inference."""
    COLS = ("entity_id", "name_norm", "name_core", "name_compact", "addr_norm", "house_number")

    def __init__(self, df: pd.DataFrame):
        self.obj = {c: df[c].to_numpy(dtype=object) for c in self.COLS if c in df}
        self.len = {c: df[c].str.len().to_numpy() for c in self.COLS if c in df and c != "entity_id"}
        self.is_s2 = df["entity_id"].str.startswith("S2-").to_numpy()
        self.n = len(df)


def pair_features(s1, tgt, s1_mats, t_mats, space: TokenSpace,
                  s1_idx: np.ndarray, t_idx: np.ndarray) -> pd.DataFrame:
    """Row-aligned features for pairs (s1_idx[i], t_idx[i]). s1 / tgt: DataFrame or Records.
    Group/rank features are added separately."""
    f = {}
    r1 = s1 if isinstance(s1, Records) else Records(s1)
    r2 = tgt if isinstance(tgt, Records) else Records(tgt)
    recs = {id(s1): r1, id(tgt): r2}
    col = lambda df, c, idx: recs[id(df)].obj[c][idx]
    ln = lambda df, c, idx: recs[id(df)].len[c][idx]
    n1, n2 = col(s1, "name_norm", s1_idx), col(tgt, "name_norm", t_idx)
    c1, c2 = col(s1, "name_core", s1_idx), col(tgt, "name_core", t_idx)
    a1, a2 = col(s1, "addr_norm", s1_idx), col(tgt, "addr_norm", t_idx)

    name_empty = (ln(s1, "name_norm", s1_idx) == 0) | (ln(tgt, "name_norm", t_idx) == 0)
    core_empty = (ln(s1, "name_core", s1_idx) == 0) | (ln(tgt, "name_core", t_idx) == 0)
    addr_empty = (ln(s1, "addr_norm", s1_idx) == 0) | (ln(tgt, "addr_norm", t_idx) == 0)

    # 1. name strings
    f["name_jw"] = _pairwise(JaroWinkler.normalized_similarity, n1, n2)
    f["name_lev"] = _pairwise(Levenshtein.normalized_similarity, n1, n2)
    f["name_lcs"] = _pairwise(LCSseq.normalized_similarity, n1, n2)
    f["name_tset"] = _pairwise(fuzz.token_set_ratio, n1, n2, 100.0)
    f["name_tsort"] = _pairwise(fuzz.token_sort_ratio, n1, n2, 100.0)
    f["name_partial"] = _pairwise(fuzz.partial_ratio, n1, n2, 100.0)
    for k in ["name_jw", "name_lev", "name_lcs", "name_tset", "name_tsort", "name_partial"]:
        f[k][name_empty] = np.nan
    f["core_jw"] = _pairwise(JaroWinkler.normalized_similarity, c1, c2)
    f["core_tset"] = _pairwise(fuzz.token_set_ratio, c1, c2, 100.0)
    f["core_jw"][core_empty] = np.nan
    f["core_tset"][core_empty] = np.nan
    f["core_exact"] = ((c1 == c2) & ~core_empty).astype(np.float32)
    # compact core name (spaces removed) catches "highlandintelligence(.com)" vs "Highland Intelligence Inc"
    k1, k2 = col(s1, "name_compact", s1_idx), col(tgt, "name_compact", t_idx)
    f["compact_partial"] = _pairwise(fuzz.partial_ratio, k1, k2, 100.0)
    f["compact_ratio"] = _pairwise(fuzz.ratio, k1, k2, 100.0)
    f["compact_partial"][core_empty] = np.nan
    f["compact_ratio"][core_empty] = np.nan
    f["compact_exact"] = ((k1 == k2) & ~core_empty).astype(np.float32)

    # 2. address strings
    f["addr_jw"] = _pairwise(JaroWinkler.normalized_similarity, a1, a2)
    f["addr_lcs"] = _pairwise(LCSseq.normalized_similarity, a1, a2)
    f["addr_tset"] = _pairwise(fuzz.token_set_ratio, a1, a2, 100.0)
    f["addr_tsort"] = _pairwise(fuzz.token_sort_ratio, a1, a2, 100.0)
    for k in ["addr_jw", "addr_lcs", "addr_tset", "addr_tsort"]:
        f[k][addr_empty] = np.nan
    f["addr_missing"] = (ln(tgt, "addr_norm", t_idx) == 0).astype(np.float32)

    # 3. house / street number: +1 same, -1 different, 0 when either side has none
    h1, h2 = col(s1, "house_number", s1_idx), col(tgt, "house_number", t_idx)
    has = (ln(s1, "house_number", s1_idx) > 0) & (ln(tgt, "house_number", t_idx) > 0)
    f["hn_match"] = np.where(has, np.where(h1 == h2, 1.0, -1.0), 0.0).astype(np.float32)

    # 4. IDF-weighted token overlap (names, address words) and address-component overlap
    A, B = s1_mats["name"][s1_idx], t_mats["name"][t_idx]
    shared, m1, m2, mx = _weighted_overlap(A, B, space.name_idf)
    f["name_idf_shared"] = shared
    f["name_idf_max_shared"] = mx
    f["name_idf_cov_s1"] = _safe_div(shared, m1)
    f["name_idf_cov_t"] = _safe_div(shared, m2)
    f["name_idf_miss_s1"] = (m1 - shared).astype(np.float32)
    f["name_idf_miss_t"] = (m2 - shared).astype(np.float32)

    A, B = s1_mats["addr"][s1_idx], t_mats["addr"][t_idx]
    shared, m1, m2, mx = _weighted_overlap(A, B, space.addr_idf)
    f["addr_idf_cov_s1"] = _safe_div(shared, m1)
    f["addr_idf_cov_t"] = _safe_div(shared, m2)
    f["addr_idf_max_shared"] = mx

    A, B = s1_mats["comp"][s1_idx], t_mats["comp"][t_idx]
    inter = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    f["comp_ov_s1"] = _safe_div(inter, np.asarray(A.sum(axis=1)).ravel())
    f["comp_ov_t"] = _safe_div(inter, np.asarray(B.sum(axis=1)).ravel())
    # all numbers in the two addresses (house, plot, unit, street numbers)
    A, B = s1_mats["num"][s1_idx], t_mats["num"][t_idx]
    inter = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    na, nb = np.asarray(A.sum(axis=1)).ravel(), np.asarray(B.sum(axis=1)).ravel()
    f["num_jacc"] = _safe_div(inter, na + nb - inter)
    f["num_conflict"] = ((na > 0) & (nb > 0) & (inter == 0)).astype(np.float32)
    for k in ["addr_idf_cov_s1", "addr_idf_cov_t", "addr_idf_max_shared", "comp_ov_s1", "comp_ov_t", "num_jacc"]:
        f[k][addr_empty] = np.nan

    # 5. structure
    l1, l2 = ln(s1, "name_norm", s1_idx), ln(tgt, "name_norm", t_idx)
    f["name_len_ratio"] = _safe_div(np.minimum(l1, l2), np.maximum(l1, l2))
    l1, l2 = ln(s1, "addr_norm", s1_idx), ln(tgt, "addr_norm", t_idx)
    f["addr_len_ratio"] = _safe_div(np.minimum(l1, l2), np.maximum(l1, l2))
    f["addr_len_ratio"][addr_empty] = np.nan
    f["is_s2"] = r2.is_s2[t_idx].astype(np.float32)
    return pd.DataFrame(f)


def add_group_features(df: pd.DataFrame, key: str = "s1_idx",
                       score_cols=("cos_comb", "name_tset", "addr_tset")) -> pd.DataFrame:
    """Context of a pair among all candidates of the same S1 entity: rank, gap to best / second best."""
    keys = df[key].to_numpy()
    df["n_cands"] = df.groupby(key, sort=False)[key].transform("size").astype(np.float32)
    for c in score_cols:
        if c not in df:
            continue
        v = df[c].fillna(-1.0).to_numpy(dtype=np.float64)
        order = np.lexsort((-v, keys))                      # by key, then score descending
        sk, sv = keys[order], v[order]
        starts = np.r_[0, np.flatnonzero(sk[1:] != sk[:-1]) + 1]
        sizes = np.diff(np.r_[starts, len(sk)])
        best_g = sv[starts]
        second_g = np.where(sizes > 1, sv[np.minimum(starts + 1, len(sv) - 1)], -1.0)
        best, second = np.empty_like(v), np.empty_like(v)
        best[order] = np.repeat(best_g, sizes)
        second[order] = np.repeat(second_g, sizes)
        df[f"{c}_rank"] = pd.Series(v).groupby(keys).rank(ascending=False, method="min").to_numpy(dtype=np.float32)
        df[f"{c}_gap_best"] = (best - v).astype(np.float32)
        df[f"{c}_gap_second"] = np.where(v >= best, v - second, v - best).astype(np.float32)
    return df
