"""Quick look for structure the top scores might exploit: file order, id relationships, cluster sizes."""
import numpy as np, pandas as pd
D = "student_resource/dataset/train"
s1 = pd.read_csv(f"{D}/train_source1.tsv", sep="\t", dtype=str, usecols=["entity_id", "country"])
s2 = pd.read_csv(f"{D}/train_source2.tsv", sep="\t", dtype=str, usecols=["entity_id"])
s3 = pd.read_csv(f"{D}/train_source3.tsv", sep="\t", dtype=str, usecols=["entity_id"])
gt = pd.read_csv(f"{D}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
pos1 = dict(zip(s1.entity_id, range(len(s1))))
pos2 = dict(zip(s2.entity_id, range(len(s2)))); pos3 = dict(zip(s3.entity_id, range(len(s3))))
rows = []
for k, v in zip(gt.source1_entity_id.head(200000), gt.matched_entity_ids.head(200000)):
    for t in v.split(","):
        if t:
            p = pos2.get(t, pos3.get(t))
            rows.append((pos1[k], p, t[:2], int(k[3:]), int(t[3:])))
d = pd.DataFrame(rows, columns=["p1", "pt", "src", "id1", "idt"])
for src in ["S2", "S3"]:
    x = d[d.src == src]
    n = len(s2) if src == "S2" else len(s3)
    print(src, "Spearman(file position S1, file position target) =", round(x[["p1", "pt"]].corr(method="spearman").iloc[0, 1], 4),
          "| Spearman(id S1, id target) =", round(x[["id1", "idt"]].corr(method="spearman").iloc[0, 1], 4))
# within one S1: are its S2 records adjacent in the S2 file?
g = d[d.src == "S2"].groupby("p1").pt.agg(["min", "max", "size"])
g = g[g["size"] > 1]
print("S1s with >=2 S2 matches: median position span", int((g["max"] - g["min"]).median()), "of", len(s2), "(random would be ~", len(s2) // 3, ")")
print("S1 file sorted by id?", s1.entity_id.str[3:].astype(int).is_monotonic_increasing, "| S2 sorted by id?", s2.entity_id.str[3:].astype(int).is_monotonic_increasing)
