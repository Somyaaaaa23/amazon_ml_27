"""Proxy check for Indic transliteration: name similarity between Indic-script S2/S3 records and
their true S1 match, using only S1 entities on the TRAINING side of the validation split."""
import sys, os, importlib, pandas as pd, numpy as np
from rapidfuzz import fuzz
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.validate import is_holdout
D = os.path.join(ROOT, "student_resource/dataset/train")
parts = []
for f in ["train_source2.tsv", "train_source3.tsv"]:
    d = pd.read_csv(f"{D}/{f}", sep="\t", dtype=str, keep_default_na=False, nrows=300000)
    parts.append(d[d.business_name.map(lambda s: any(0x900 <= ord(c) <= 0xD7F for c in s))])
nl = pd.concat(parts)
ids = set(nl.entity_id)
gt = pd.read_csv(f"{D}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
owner = {t: k for k, v in zip(gt.source1_entity_id, gt.matched_entity_ids) for t in v.split(",") if t in ids and not is_holdout(k)}
s1 = pd.read_csv(f"{D}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
s1n = dict(zip(s1.entity_id, s1.business_name))
nl = nl[nl.entity_id.isin(owner)]
for label, path in [("antigravity", os.path.join(ROOT, ".worktrees/antigravity")), ("new", ROOT)]:
    for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
        del sys.modules[k]
    sys.path.insert(0, path)
    nn = importlib.import_module("src.normalization").normalize_name
    sys.path.pop(0)
    a = [nn(s1n[owner[t]])[0] for t in nl.entity_id]
    b = [nn(n)[0] for n in nl.business_name]
    ts = np.array([fuzz.token_set_ratio(x, y) for x, y in zip(a, b)])
    rs = np.array([fuzz.ratio(x, y) for x, y in zip(a, b)])
    print(f"{label:12s} n={len(ts)}  token_set mean={ts.mean():.1f}  share>=80={np.mean(ts>=80):.3f}  ratio mean={rs.mean():.1f}")
