"""
Production-Grade Full Test Inference Engine.
Implements:
- P0-1: Training S1 matched against full country target pool for honest hard negatives.
- P0-2: Sparse BLAS blocking with per-country IDF fit.
- P0-3: Fast vectorized feature extraction.
- P0-4: Official validator subprocess invocation (student_resource/utils/validate_submission.py).
- P1-1: Global one-owner postprocessing across the entire country pool.
- P1-2, P1-4, P1-5, P1-6: Ordered candidates, DF-capped rare tokens, and cosine feature pass-through.
"""

import os
import sys
import time
import subprocess
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.normalization import preprocess_dataframe
from src.blocking import MultiBlocker, CountryTargetIndex, evaluate_blocking_recall
from src.features import PairFeatureExtractor, FEATURE_COLS
from src.model import EntityMatcherModel
from src.postprocessing import apply_one_owner_filter


def run_full_inference(
    train_dir: str = "student_resource/dataset/train",
    test_dir: str = "student_resource/dataset/test",
    output_dir: str = "output",
    sample_train_s1: int = 35000,
    top_k: int = 20,
    chunk_size: int = 30000
):
    print("=" * 75)
    print("      ENTITY RESOLUTION: HIGH-PERFORMANCE INFERENCE PIPELINE")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    # -------------------------------------------------------------------------
    # 1. MODEL TRAINING (P0-1: Train S1 against full country target pools)
    # -------------------------------------------------------------------------
    print("\n[Step 1/3] Training GBDT Matcher against Full Country Pools...")
    df_s1_tr = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t")
    if sample_train_s1 < len(df_s1_tr):
        frac = sample_train_s1 / len(df_s1_tr)
        df_s1_tr_sample = df_s1_tr.groupby("country", group_keys=False).sample(frac=frac, random_state=42).reset_index(drop=True)
    else:
        df_s1_tr_sample = df_s1_tr

    sampled_s1_ids = set(df_s1_tr_sample["entity_id"])
    print(f"  Training S1 sample: {len(df_s1_tr_sample):,} entities across US and India.")

    df_gt = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", keep_default_na=False)
    gt_dict = {
        row["source1_entity_id"].strip(): [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
        for _, row in df_gt[df_gt["source1_entity_id"].isin(sampled_s1_ids)].iterrows()
    }

    # Preprocess training S1
    df_s1_tr_sample = preprocess_dataframe(df_s1_tr_sample)

    # We will build training pairs by country
    train_pairs_list = []
    training_blocker = MultiBlocker(top_k_per_source=top_k)

    # Collect training text to fit vectorizers
    train_text_sample = list(df_s1_tr_sample["combined_text"])
    for f in ["train_source2.tsv", "train_source3.tsv", "test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
        p = os.path.join(train_dir, f) if "train" in f else os.path.join(test_dir, f)
        if os.path.exists(p):
            sub = pd.read_csv(p, sep="\t", nrows=25000)
            train_text_sample.extend(sub["business_name"].fillna("").astype(str) + " " + sub["business_address"].fillna("").astype(str))

    training_blocker.fit_vectorizers(train_text_sample)
    extractor = PairFeatureExtractor(idf_dict=training_blocker.idf_dict, max_idf=training_blocker.max_idf)

    for country in ["US", "India"]:
        print(f"  Building training pairs for {country} against full country target pool...")
        s1_c = df_s1_tr_sample[df_s1_tr_sample["country"] == country].copy()
        if s1_c.empty:
            continue

        # Load full S2 and S3 for this country
        s2_list, s3_list = [], []
        for chunk in pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", chunksize=500000):
            sub = chunk[chunk["country"] == country]
            if not sub.empty:
                s2_list.append(sub)
        for chunk in pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", chunksize=500000):
            sub = chunk[chunk["country"] == country]
            if not sub.empty:
                s3_list.append(sub)

        targets_c = pd.concat(s2_list + s3_list, ignore_index=True)
        targets_c = preprocess_dataframe(targets_c)
        print(f"    Loaded {len(targets_c):,} target records for {country}. Pre-indexing...")

        target_index = CountryTargetIndex(targets_c, training_blocker)
        cands_c, cos_c = target_index.query_batch(s1_c, training_blocker, top_k=top_k)

        b_metrics = evaluate_blocking_recall(cands_c, gt_dict)
        print(f"    {country} Training Blocker Recall: {b_metrics['candidate_recall']*100:.2f}% ({b_metrics['captured_pairs']:,}/{b_metrics['total_true_pairs']:,} true pairs)")

        pairs_c = extractor.build_feature_table(
            df_s1=s1_c,
            df_target=targets_c,
            candidates_dict=cands_c,
            cosine_sims_dict=cos_c,
            ground_truth_dict=gt_dict
        )
        print(f"    {country} Training Samples: {len(pairs_c):,} pairs (Positives: {pairs_c['label'].sum():,})")
        train_pairs_list.append(pairs_c)

        del targets_c, target_index, s2_list, s3_list

    df_train_pairs = pd.concat(train_pairs_list, ignore_index=True)
    print(f"\n  Total Training Pairs: {len(df_train_pairs):,} (Positives: {df_train_pairs['label'].sum():,})")

    matcher = EntityMatcherModel(n_estimators=300, learning_rate=0.05, max_depth=6)
    cv_res = matcher.train_cv(df_train_pairs=df_train_pairs, ground_truth_dict=gt_dict, n_splits=5)
    best_t = matcher.best_threshold
    print(f"  >>> Model Trained! Best Macro F0.5 Threshold: t* = {best_t:.3f}")

    del df_train_pairs, train_pairs_list, df_s1_tr, df_s1_tr_sample

    # -------------------------------------------------------------------------
    # 2. FULL TEST INFERENCE (P1-1: Global Country-Level One-Owner Execution)
    # -------------------------------------------------------------------------
    print("\n[Step 2/3] Executing Full Test Inference across Country Partitions...")
    df_s1_te_all = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t")
    total_test_s1_count = len(df_s1_te_all)
    print(f"  Total Test S1 Records: {total_test_s1_count:,}")

    matching_out_path = os.path.join(output_dir, "matching_results.tsv")
    cand_out_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # Initialize output files with headers
    with open(matching_out_path, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
    with open(cand_out_path, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    countries = ["France", "US", "India"]
    total_matches_written = 0
    total_s1_written = 0

    for country in countries:
        c_name = str(country)
        print(f"\n=======================================================")
        print(f"  Processing Country Partition: {c_name.upper()}")
        print(f"=======================================================")

        df_s1_c = df_s1_te_all[df_s1_te_all["country"] == country].copy()
        if df_s1_c.empty:
            continue

        print(f"  Loading Test Targets for {c_name}...")
        s2_list, s3_list = [], []
        for chunk in pd.read_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t", chunksize=500000):
            sub = chunk[chunk["country"] == country]
            if not sub.empty:
                s2_list.append(sub)
        for chunk in pd.read_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t", chunksize=500000):
            sub = chunk[chunk["country"] == country]
            if not sub.empty:
                s3_list.append(sub)

        targets_c = pd.concat(s2_list + s3_list, ignore_index=True)
        print(f"  {c_name} Pool: {len(df_s1_c):,} S1 entities, {len(targets_c):,} Target records.")

        df_s1_c = preprocess_dataframe(df_s1_c)
        targets_c = preprocess_dataframe(targets_c)

        print(f"  Building pre-index for {c_name}...")
        target_index = CountryTargetIndex(targets_c, training_blocker)

        # Scored pair accumulators for global country-level one-owner rule (P1-1)
        country_scored_s1 = []
        country_scored_cand = []
        country_scored_prob = []
        country_candidate_map = defaultdict(list)

        n_s1 = len(df_s1_c)
        for i in range(0, n_s1, chunk_size):
            s1_batch = df_s1_c.iloc[i : i + chunk_size].copy()
            batch_s1_ids = list(s1_batch["entity_id"].values)

            batch_cands, batch_cos = target_index.query_batch(s1_batch, training_blocker, top_k=top_k)
            for sid, c_list in batch_cands.items():
                country_candidate_map[sid].extend(c_list)

            df_batch_pairs = extractor.build_feature_table(
                df_s1=s1_batch,
                df_target=targets_c,
                candidates_dict=batch_cands,
                cosine_sims_dict=batch_cos,
                ground_truth_dict=None
            )

            if not df_batch_pairs.empty:
                df_batch_scored = matcher.predict_pairs(df_batch_pairs)
                country_scored_s1.extend(df_batch_scored["source1_entity_id"].values)
                country_scored_cand.extend(df_batch_scored["candidate_entity_id"].values)
                country_scored_prob.extend(df_batch_scored["pred_prob"].values)

            print(f"    Scored {min(i + chunk_size, n_s1):,}/{n_s1:,} [{c_name}]")

        # Global One-Owner Postprocessing across the ENTIRE Country (P1-1)
        print(f"  Applying Global One-Owner Rule across {c_name}...")
        if country_scored_s1:
            df_country_pairs = pd.DataFrame({
                "source1_entity_id": country_scored_s1,
                "candidate_entity_id": country_scored_cand,
                "pred_prob": country_scored_prob
            })

            # Sort by predicted probability descending and deduplicate by candidate_entity_id
            df_country_pairs = df_country_pairs.sort_values(by="pred_prob", ascending=False)
            df_country_dedup = df_country_pairs.drop_duplicates(subset=["candidate_entity_id"], keep="first")

            # Filter above decision threshold
            df_matches = df_country_dedup[df_country_dedup["pred_prob"] >= best_t]

            country_matches_dict = defaultdict(list)
            for s1_id, cand_id in zip(df_matches["source1_entity_id"], df_matches["candidate_entity_id"]):
                country_matches_dict[s1_id].append(cand_id)
        else:
            country_matches_dict = defaultdict(list)

        # Stream country results to disk
        print(f"  Writing {c_name} predictions to output files...")
        with open(matching_out_path, "a", encoding="utf-8") as f_match:
            for s1_id in df_s1_c["entity_id"]:
                m_list = country_matches_dict.get(s1_id, [])
                f_match.write(f"{s1_id}\t{','.join(m_list)}\n")
                total_matches_written += len(m_list)

        with open(cand_out_path, "a", encoding="utf-8") as f_cand:
            for s1_id in df_s1_c["entity_id"]:
                c_list = country_candidate_map.get(s1_id, [])
                f_cand.write(f"{s1_id}\t{','.join(c_list)}\n")

        total_s1_written += len(df_s1_c)
        print(f"  {c_name} Complete! Total S1 Written: {total_s1_written:,}/{total_test_s1_count:,}")

        del df_s1_c, targets_c, target_index, country_scored_s1, country_scored_cand, country_scored_prob

    # -------------------------------------------------------------------------
    # 3. OFFICIAL SUBMISSION VALIDATION (P0-4)
    # -------------------------------------------------------------------------
    print("\n[Step 3/3] Running Official Challenge Validator...")
    official_val_script = os.path.join(PROJECT_ROOT, "student_resource", "utils", "validate_submission.py")
    if os.path.exists(official_val_script):
        val_res = subprocess.run([
            sys.executable,
            official_val_script,
            "--matching", matching_out_path,
            "--candidate", cand_out_path,
            "--test-dir", test_dir
        ])
        is_valid = (val_res.returncode == 0)
    else:
        from utils.validate_submission import validate_submission
        is_valid = validate_submission(matching_out_path, cand_out_path, test_dir)

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"\n=======================================================")
    print(f" FULL INFERENCE COMPLETED IN {elapsed_min:.1f} MINUTES!")
    print(f" Total S1 Entities: {total_s1_written:,}")
    print(f" Total Matches:     {total_matches_written:,}")
    print(f" Submission Ready:  {matching_out_path}")
    print(f" Candidate File:    {cand_out_path}")
    print(f" Status: {'PASSED - READY FOR UPLOAD' if is_valid else 'VALIDATION FAILED'}")
    print(f"=======================================================")

    return is_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Performance Full Test Inference Engine.")
    parser.add_argument("--train-dir", default="student_resource/dataset/train", help="Train dataset directory")
    parser.add_argument("--test-dir", default="student_resource/dataset/test", help="Test dataset directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--sample-train-s1", type=int, default=35000, help="Number of S1 train entities for model fitting")
    parser.add_argument("--top-k", type=int, default=20, help="Top K candidates per source")
    parser.add_argument("--chunk-size", type=int, default=30000, help="Streaming batch size")

    args = parser.parse_args()
    success = run_full_inference(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        sample_train_s1=args.sample_train_s1,
        top_k=args.top_k,
        chunk_size=args.chunk_size
    )
    sys.exit(0 if success else 1)
