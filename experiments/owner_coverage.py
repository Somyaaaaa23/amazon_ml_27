"""Share of S2/S3 pool records whose compact name equals the compact name of SOME S1 record of the
same country: train vs test. A lower test share suggests test pools contain more records whose
business has no S1 entity (pure distractors)."""
import os, sys
import pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.pipeline_core import normalize
for split in ["train", "test"]:
    D = os.path.join(ROOT, "student_resource/dataset", split)
    s1 = pd.read_csv(f"{D}/{split}_source1.tsv", sep="\t", dtype=str)
    s2 = pd.read_csv(f"{D}/{split}_source2.tsv", sep="\t", dtype=str)
    for c in sorted(s1.country.unique()):
        a = normalize(s1[s1.country == c])
        b = normalize(s2[s2.country == c].sample(150000, random_state=0))
        names = set(a.name_compact[a.name_compact.str.len() > 0])
        share = b.name_compact.isin(names).mean()
        print(f"{split:5s} {c:7s} S1={len(a):>9,}  S2 pool={int((s2.country == c).sum()):>9,}  "
              f"pool/S1={(s2.country == c).sum() / len(a):.2f}  share of S2 with exact-name S1 twin={share:.3f}", flush=True)
