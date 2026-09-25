#!/usr/bin/env python3
"""
Create consistent, stratified subsample chunks from student_resource dataset.
Preserves ground-truth linkage integrity, country balance, and singleton ratio.
"""

import os
import argparse
import pandas as pd
import numpy as np


def create_sample_dataset(
    src_train_dir: str = "student_resource/dataset/train",
    src_test_dir: str = "student_resource/dataset/test",
    output_dir: str = "dataset_sample_50k",
    sample_s1_train: int = 50000,
    sample_s1_test: int = 25000,
    random_state: int = 42
):
    np.random.seed(random_state)
    out_train = os.path.join(output_dir, "train")
    out_test = os.path.join(output_dir, "test")
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)

    print(f"=== Creating Sample Dataset in '{output_dir}' ===")
    print(f"Sampling {sample_s1_train:,} Train S1 entities and {sample_s1_test:,} Test S1 entities...")

    # 1. Load Train S1 and sample
    df_s1_tr = pd.read_csv(os.path.join(src_train_dir, "train_source1.tsv"), sep="\t")
    if sample_s1_train < len(df_s1_tr):
        frac = sample_s1_train / len(df_s1_tr)
        df_s1_tr_sampled = df_s1_tr.groupby("country", group_keys=False).sample(frac=frac, random_state=random_state).reset_index(drop=True)
    else:
        df_s1_tr_sampled = df_s1_tr

    sampled_s1_ids = set(df_s1_tr_sampled["entity_id"])
    print(f"Sampled Train S1: {len(df_s1_tr_sampled):,} rows")

    # 2. Extract matching ground truth
    df_gt = pd.read_csv(os.path.join(src_train_dir, "train_ground_truth.tsv"), sep="\t", keep_default_na=False)
    df_gt_sampled = df_gt[df_gt["source1_entity_id"].isin(sampled_s1_ids)].copy()
    print(f"Sampled Ground Truth: {len(df_gt_sampled):,} rows")

    target_ids_needed = set()
    for val in df_gt_sampled["matched_entity_ids"]:
        for item in val.split(","):
            if item.strip():
                target_ids_needed.add(item.strip())
    print(f"Total True Target Matches needed: {len(target_ids_needed):,}")

    # 3. Stream and extract S2 and S3 (true matches + distractors)
    for source_name, fname in [("S2", "train_source2.tsv"), ("S3", "train_source3.tsv")]:
        fpath = os.path.join(src_train_dir, fname)
        kept_chunks = []
        for chunk in pd.read_csv(fpath, sep="\t", chunksize=200000):
            # Keep all needed true matches
            matched = chunk[chunk["entity_id"].isin(target_ids_needed)]
            # Also keep 3% random distractors
            distractors = chunk[~chunk["entity_id"].isin(target_ids_needed)].sample(frac=0.03, random_state=random_state)
            kept_chunks.append(pd.concat([matched, distractors]))
        df_target_sampled = pd.concat(kept_chunks, ignore_index=True)
        out_fpath = os.path.join(out_train, fname)
        df_target_sampled.to_csv(out_fpath, sep="\t", index=False)
        print(f"Saved {source_name} to {out_fpath} ({len(df_target_sampled):,} rows)")

    # Save S1 and GT
    df_s1_tr_sampled.to_csv(os.path.join(out_train, "train_source1.tsv"), sep="\t", index=False)
    df_gt_sampled.to_csv(os.path.join(out_train, "train_ground_truth.tsv"), sep="\t", index=False)

    # 4. Sample Test S1
    df_s1_te = pd.read_csv(os.path.join(src_test_dir, "test_source1.tsv"), sep="\t")
    if sample_s1_test < len(df_s1_te):
        frac_te = sample_s1_test / len(df_s1_te)
        df_s1_te_sampled = df_s1_te.groupby("country", group_keys=False).sample(frac=frac_te, random_state=random_state).reset_index(drop=True)
    else:
        df_s1_te_sampled = df_s1_te

    df_s1_te_sampled.to_csv(os.path.join(out_test, "test_source1.tsv"), sep="\t", index=False)
    print(f"Sampled Test S1: {len(df_s1_te_sampled):,} rows")

    # Sample Test S2 and S3
    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        fpath = os.path.join(src_test_dir, fname)
        kept_chunks = []
        for chunk in pd.read_csv(fpath, sep="\t", chunksize=200000):
            kept_chunks.append(chunk.sample(frac=0.05, random_state=random_state))
        df_sub = pd.concat(kept_chunks, ignore_index=True)
        out_fpath = os.path.join(out_test, fname)
        df_sub.to_csv(out_fpath, sep="\t", index=False)
        print(f"Saved {fname} to {out_fpath} ({len(df_sub):,} rows)")

    print(f"\n>>> Subsample dataset ready in '{output_dir}'! <<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create sample chunk of entity resolution dataset.")
    parser.add_argument("--sample-s1-train", type=int, default=50000, help="Number of S1 train rows to sample")
    parser.add_argument("--sample-s1-test", type=int, default=25000, help="Number of S1 test rows to sample")
    parser.add_argument("--output-dir", default="dataset_sample_50k", help="Destination folder")

    args = parser.parse_args()
    create_sample_dataset(
        sample_s1_train=args.sample_s1_train,
        sample_s1_test=args.sample_s1_test,
        output_dir=args.output_dir
    )
