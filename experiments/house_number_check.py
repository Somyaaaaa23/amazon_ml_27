"""House-number extraction check on TRAINING-side true pairs vs random same-country pairs."""
import sys, os, re, pandas as pd, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.validate import is_holdout
from src.normalization import HOUSE_NUMBER_PATTERN, strip_accents, transliterate_indic

def old(a):
    if not isinstance(a, str) or not a.strip(): return ""
    m = HOUSE_NUMBER_PATTERN.search(strip_accents(transliterate_indic(a)).lower())
    return (m.group(1) or m.group(2) or "") if m else ""

EXPLICIT = re.compile(r"\b(?:plot|door|house|flat|shop|h|d|kh|khasra|survey|sy|s)\s*(?:no|n|number|#)?\s*[\.:#-]*\s*(\d+[a-z]?(?:\s*[/-]\s*\d+[a-z]?)*)", re.I)
NO_NUM = re.compile(r"(?:\bno|#|\bn[o°º])\s*[\.:#-]*\s*(\d+[a-z]?(?:\s*[/-]\s*\d+[a-z]?)*)", re.I)
COMP_NUM = re.compile(r"^\s*([a-z]{0,2}[-/]?\d+[a-z]?(?:\s*[/-]\s*\d+[a-z]?)*)(?:\s+(?:bis|ter))?\s+[a-z]", re.I)
def new(a):
    if not isinstance(a, str) or not a.strip(): return ""
    t = strip_accents(transliterate_indic(a)).lower()
    for pat in (EXPLICIT, NO_NUM):
        m = pat.search(t)
        if m: return re.sub(r"\s+", "", m.group(1))
    for comp in t.split(","):
        m = COMP_NUM.match(comp)
        if m: return re.sub(r"\s+", "", m.group(1))
    return ""

D = os.path.join(ROOT, "student_resource/dataset/train")
s1 = pd.read_csv(f"{D}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
s1 = s1[~s1.entity_id.map(is_holdout)].sample(40000, random_state=0)
s2 = pd.read_csv(f"{D}/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = pd.read_csv(f"{D}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("source1_entity_id")
addr2 = dict(zip(s2.entity_id, s2.business_address)); c2 = s2.groupby("country").business_address.apply(list)
rows = []
for eid, a1, c in zip(s1.entity_id, s1.business_address, s1.country):
    for t in gt.loc[eid, "matched_entity_ids"].split(","):
        if t in addr2: rows.append((c, a1, addr2[t], 1))
rng = np.random.default_rng(0)
for eid, a1, c in zip(s1.entity_id[:20000], s1.business_address[:20000], s1.country[:20000]):
    rows.append((c, a1, c2[c][rng.integers(len(c2[c]))], 0))
df = pd.DataFrame(rows, columns=["country", "a1", "a2", "true"])
for name, fn in [("old", old), ("new", new)]:
    h1, h2 = df.a1.map(fn), df.a2.map(fn)
    both = (h1 != "") & (h2 != "")
    for c in sorted(df.country.unique()):
        for tv in (1, 0):
            m = (df.country == c) & (df.true == tv)
            print(f"{name} {c:6s} {'true ' if tv else 'random'} pairs: both have number {both[m].mean():.3f} | equal given both {(h1 == h2)[m & both].mean():.3f}")
print(df[df.true == 1].sample(12, random_state=3).assign(h1=lambda d: d.a1.map(new), h2=lambda d: d.a2.map(new))[["a1", "a2", "h1", "h2"]].to_string())
