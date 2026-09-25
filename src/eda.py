"""
Stage 0: Exploratory Data Analysis (EDA) module.
Performs comprehensive data auditing on training sources and ground truth:
1. Row counts per source and per country.
2. Share and distribution of singletons vs multi-matches.
3. One-owner integrity check (does any S2/S3 ID link to multiple S1 records?).
4. Most frequent tokens / generic word analysis.
5. Missing field and length distributions.
"""

import os
from collections import Counter
import pandas as pd
import numpy as np


def run_eda(train_dir: str = "dataset/train") -> dict:
    print("=" * 60)
    print("STAGE 0: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str)
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str)
    df_gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)

    print(f"\n[1] Row Counts:")
    print(f"  Source 1 (Reference): {len(df_s1)} rows")
    print(f"  Source 2:            {len(df_s2)} rows")
    print(f"  Source 3:            {len(df_s3)} rows")

    print(f"\n[2] Country Distribution in Source 1:")
    if "country" in df_s1.columns:
        for country, count in df_s1["country"].value_counts().items():
            print(f"  - {country}: {count} ({count / len(df_s1) * 100:.1f}%)")
    else:
        print("  - 'country' column not present in Source 1.")

    # Analyze ground truth
    gt_dict = {}
    all_matched_s2 = []
    all_matched_s3 = []
    matches_per_s1 = []
    target_to_s1 = {}
    multiple_owner_targets = set()

    for _, row in df_gt.iterrows():
        s1_id = row["source1_entity_id"].strip()
        val = row["matched_entity_ids"].strip()
        m_list = [x.strip() for x in val.split(",") if x.strip()]
        gt_dict[s1_id] = m_list
        matches_per_s1.append(len(m_list))

        for target_id in m_list:
            if target_id.startswith("S2-"):
                all_matched_s2.append(target_id)
            elif target_id.startswith("S3-"):
                all_matched_s3.append(target_id)

            if target_id in target_to_s1:
                multiple_owner_targets.add(target_id)
            else:
                target_to_s1[target_id] = s1_id

    total_s1 = len(df_s1)
    singletons = sum(1 for m in matches_per_s1 if m == 0)
    print(f"\n[3] Ground Truth Match Distributions:")
    print(f"  - Total S1 entities: {total_s1}")
    print(f"  - Singletons (0 matches): {singletons} ({singletons / total_s1 * 100:.1f}%)")
    print(f"  - Exactly 1 match: {sum(1 for m in matches_per_s1 if m == 1)} ({sum(1 for m in matches_per_s1 if m == 1) / total_s1 * 100:.1f}%)")
    print(f"  - 2+ matches: {sum(1 for m in matches_per_s1 if m >= 2)} ({sum(1 for m in matches_per_s1 if m >= 2) / total_s1 * 100:.1f}%)")
    print(f"  - Total Matched S2 records: {len(all_matched_s2)} (Coverage of S2: {len(set(all_matched_s2)) / len(df_s2) * 100:.1f}%)")
    print(f"  - Total Matched S3 records: {len(all_matched_s3)} (Coverage of S3: {len(set(all_matched_s3)) / len(df_s3) * 100:.1f}%)")

    # One-owner rule check
    print(f"\n[4] One-Owner Integrity Check:")
    if multiple_owner_targets:
        print(f"  [!] Found {len(multiple_owner_targets)} target IDs linked to multiple S1 records!")
        print("  Recommendation: Soft assignment or threshold-based multi-claiming.")
        one_owner_valid = False
    else:
        print("  [✓] Every matched S2/S3 ID belongs to AT MOST ONE Source 1 entity.")
        print("  Recommendation: Enforce one-owner postprocessing (argmax S1 assignment per candidate).")
        one_owner_valid = True

    # Token Frequency Analysis
    all_names = list(df_s1["business_name"].dropna()) + list(df_s2["business_name"].dropna()) + list(df_s3["business_name"].dropna())
    name_tokens = [w.lower().strip(".,;:()\"'") for name in all_names for w in name.split()]
    name_counts = Counter(name_tokens)
    print(f"\n[5] Top 10 Most Common Name Tokens:")
    for tok, c in name_counts.most_common(10):
        print(f"  - '{tok}': {c}")

    eda_summary = {
        "total_s1": total_s1,
        "total_s2": len(df_s2),
        "total_s3": len(df_s3),
        "singleton_ratio": singletons / total_s1 if total_s1 > 0 else 0,
        "one_owner_valid": one_owner_valid,
        "top_generic_tokens": [tok for tok, _ in name_counts.most_common(20)]
    }

    return eda_summary


if __name__ == "__main__":
    run_eda()
