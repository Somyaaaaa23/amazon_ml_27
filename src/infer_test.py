"""
Full-Scale Test Inference Engine for Entity Resolution Challenge.
Trains the GBDT matcher on the representative training dataset and runs
high-throughput, memory-safe streaming inference on the full test set
(1.73M Test S1 entities across India, US, and France).
Outputs:
- output/matching_results.tsv (Scored submission file)
- output/candidate_pairs.tsv (Candidate audit file)
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.normalization import preprocess_dataframe
from src.blocking import MultiBlocker, CountryTargetIndex, evaluate_blocking_recall
from src.features import PairFeatureExtractor
from src.model import EntityMatcherModel
from src.postprocessing import generate_matching_predictions
from utils.validate_submission import validate_submission


def run_full_inference(
    train_dir: str = "student_resource/dataset/train",
    test_dir: str = "student_resource/dataset/test",
    output_dir: str = "output",
    sample_train_s1: int = 60000,
    top_k: int = 25,
    chunk_size: int = 50000
):
    print("=" * 75)
    print("      ENTITY RESOLUTION: FULL TEST SET INFERENCE ENGINE")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    # -------------------------------------------------------------------------
    # 1. MODEL TRAINING ON STRATIFIED TRAINING DATASET
    # -------------------------------------------------------------------------
    print("\n[Step 1/4] Preparing Training Data & Fitting LightGBM Matcher...")
    df_s1_tr = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t")
    if sample_train_s1 < len(df_s1_tr):
        frac = sample_train_s1 / len(df_s1_tr)
        df_s1_tr = df_s1_tr.groupby("country", group_keys=False).sample(frac=frac, random_state=42).reset_index(drop=True)

    sampled_s1_ids = set(df_s1_tr["entity_id"])
    print(f"  Training S1 entities: {len(df_s1_tr):,} (US: {(df_s1_tr['country']=='US').sum():,}, India: {(df_s1_tr['country']=='India').sum():,})")

    # Load matching ground truth
    df_gt = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", keep_default_na=False)
    df_gt_tr = df_gt[df_gt["source1_entity_id"].isin(sampled_s1_ids)].copy()
    gt_dict = {row["source1_entity_id"].strip(): [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()] for _, row in df_gt_tr.iterrows()}

    needed_targets = set()
    for targets in gt_dict.values():
        needed_targets.update(targets)

    # Load S2 and S3 for training
    s2_chunks, s3_chunks = [], []
    for chunk in pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", chunksize=250000):
        m = chunk[chunk["entity_id"].isin(needed_targets)]
        d = chunk[~chunk["entity_id"].isin(needed_targets)].sample(frac=0.03, random_state=42)
        s2_chunks.append(pd.concat([m, d]))
    df_s2_tr = pd.concat(s2_chunks, ignore_index=True)

    for chunk in pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", chunksize=250000):
        m = chunk[chunk["entity_id"].isin(needed_targets)]
        d = chunk[~chunk["entity_id"].isin(needed_targets)].sample(frac=0.03, random_state=42)
        s3_chunks.append(pd.concat([m, d]))
    df_s3_tr = pd.concat(s3_chunks, ignore_index=True)

    print("  Preprocessing training text...")
    df_s1_tr = preprocess_dataframe(df_s1_tr)
    df_s2_tr = preprocess_dataframe(df_s2_tr)
    df_s3_tr = preprocess_dataframe(df_s3_tr)
    df_targets_tr = pd.concat([df_s2_tr, df_s3_tr], ignore_index=True)

    # Fit multi-blocker vectorizers on train + test unlabeled text (to learn French vocabulary)
    print("  Fitting TF-IDF vectorizers on combined train + test unlabeled corpus...")
    blocker = MultiBlocker(top_k_per_source=top_k)
    sample_test_texts = []
    for f in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
        df_sub = pd.read_csv(os.path.join(test_dir, f), sep="\t", nrows=50000)
        sample_test_texts.extend(df_sub["business_name"].fillna("").astype(str) + " " + df_sub["business_address"].fillna("").astype(str))

    all_corpus_text = list(df_s1_tr["combined_text"]) + list(df_targets_tr["combined_text"]) + sample_test_texts
    blocker.fit_vectorizers(all_corpus_text)

    # Generate training candidates
    train_cands = blocker.generate_candidates(df_s1_tr, df_s2_tr, df_s3_tr)
    b_metrics = evaluate_blocking_recall(train_cands, gt_dict)
    print(f"  Training Blocker Recall: {b_metrics['candidate_recall']*100:.2f}% ({b_metrics['captured_pairs']:,}/{b_metrics['total_true_pairs']:,} true pairs)")

    # Extract training features and train CV
    extractor = PairFeatureExtractor(idf_dict=blocker.idf_dict)
    df_train_pairs = extractor.build_feature_table(
        df_s1=df_s1_tr,
        df_target=df_targets_tr,
        candidates_dict=train_cands,
        ground_truth_dict=gt_dict
    )
    print(f"  Extracted {len(df_train_pairs):,} training pairs (Positives: {df_train_pairs['label'].sum():,})")

    matcher = EntityMatcherModel(n_estimators=300, learning_rate=0.05, max_depth=6)
    cv_res = matcher.train_cv(df_train_pairs=df_train_pairs, ground_truth_dict=gt_dict, n_splits=5)
    best_t = matcher.best_threshold
    print(f"  >>> Model Trained! Optimal Decision Threshold: t* = {best_t:.3f}")

    # Free training memory
    del df_s1_tr, df_s2_tr, df_s3_tr, df_targets_tr, df_train_pairs, train_cands

    # -------------------------------------------------------------------------
    # 2. LOAD TEST DATA & PARTITION BY COUNTRY
    # -------------------------------------------------------------------------
    print("\n[Step 2/4] Loading Full Test Dataset (1.73M S1 entities)...")
    df_s1_te_all = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t")
    print(f"  Total Test S1 Records: {len(df_s1_te_all):,}")
    print(f"  Country Distribution in Test:\n{df_s1_te_all['country'].value_counts().to_string()}")

    matching_out_path = os.path.join(output_dir, "matching_results.tsv")
    cand_out_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # Initialize output files with headers
    with open(matching_out_path, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
    with open(cand_out_path, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    # -------------------------------------------------------------------------
    # 3. STREAMING TEST INFERENCE PER COUNTRY PARTITION
    # -------------------------------------------------------------------------
    print("\n[Step 3/4] Streaming Test Inference by Country Partitions...")
    countries = list(df_s1_te_all["country"].unique())

    total_test_scored = 0
    total_matches_found = 0

    for country in countries:
        c_name = str(country)
        print(f"\n=======================================================")
        print(f"  Processing Country Partition: {c_name.upper()}")
        print(f"=======================================================")

        # Load country subset for S1
        df_s1_c = df_s1_te_all[df_s1_te_all["country"] == country].copy()
        print(f"  Loading Test Targets for {c_name}...")

        # Load S2 and S3 filtered by country
        s2_list = []
        for chunk in pd.read_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t", chunksize=500000):
            sub = chunk[chunk["country"] == country]
            if not sub.empty:
                s2_list.append(sub)
        df_s2_c = pd.concat(s2_list, ignore_index=True) if s2_list else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

        s3_list = []
        for chunk in pd.read_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t", chunksize=500000):
            sub = chunk[chunk["country"] == country]
            if not sub.empty:
                s3_list.append(sub)
        df_s3_c = pd.concat(s3_list, ignore_index=True) if s3_list else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

        print(f"  {c_name} Counts: {len(df_s1_c):,} S1, {len(df_s2_c):,} S2, {len(df_s3_c):,} S3")

        # Preprocess
        df_s1_c = preprocess_dataframe(df_s1_c)
        df_s2_c = preprocess_dataframe(df_s2_c)
        df_s3_c = preprocess_dataframe(df_s3_c)
        df_targets_c = pd.concat([df_s2_c, df_s3_c], ignore_index=True)

        print(f"  Building fast pre-index for {c_name} targets ({len(df_targets_c):,} records)...")
        country_index = CountryTargetIndex(df_targets_c, blocker)
        print(f"  Index built! Streaming S1 batches...")

        # Process S1 in streaming batches
        n_s1 = len(df_s1_c)
        for i in range(0, n_s1, chunk_size):
            s1_batch = df_s1_c.iloc[i : i + chunk_size].copy()
            batch_s1_ids = list(s1_batch["entity_id"].values)

            # Candidate generation using fast pre-index
            batch_cands = country_index.query_batch(s1_batch, blocker, top_k=top_k)

            # Feature extraction
            df_batch_pairs = extractor.build_feature_table(
                df_s1=s1_batch,
                df_target=df_targets_c,
                candidates_dict=batch_cands,
                ground_truth_dict=None
            )

            if not df_batch_pairs.empty:
                # Predict
                df_batch_scored = matcher.predict_pairs(df_batch_pairs)
                # Postprocess with one-owner rule and threshold
                batch_preds = generate_matching_predictions(
                    df_scored_pairs=df_batch_scored,
                    all_s1_ids=batch_s1_ids,
                    threshold=best_t,
                    apply_one_owner=True,
                    prob_col="pred_prob"
                )
                cand_dict_scored = df_batch_pairs.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()
            else:
                batch_preds = {sid: [] for sid in batch_s1_ids}
                cand_dict_scored = {sid: [] for sid in batch_s1_ids}

            # Append to output files directly
            with open(matching_out_path, "a", encoding="utf-8") as f_match:
                for sid in batch_s1_ids:
                    m_list = batch_preds.get(sid, [])
                    f_match.write(f"{sid}\t{','.join(m_list)}\n")
                    total_matches_found += len(m_list)

            with open(cand_out_path, "a", encoding="utf-8") as f_cand:
                for sid in batch_s1_ids:
                    c_list = cand_dict_scored.get(sid, [])
                    f_cand.write(f"{sid}\t{','.join(c_list)}\n")

            total_test_scored += len(batch_s1_ids)
            print(f"    Processed {min(i + chunk_size, n_s1):,}/{n_s1:,} [{c_name}] | Total S1 Complete: {total_test_scored:,}/{len(df_s1_te_all):,}")

        # Free country memory
        del df_s1_c, df_s2_c, df_s3_c, df_targets_c, country_index

    print(f"\n[Step 3 Complete] Total Test S1 Processed: {total_test_scored:,}")
    print(f"Total Matches Identified: {total_matches_found:,}")
    print(f"Files written to:\n  - {matching_out_path}\n  - {cand_out_path}")

    # -------------------------------------------------------------------------
    # 4. SUBMISSION VALIDATION
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] Validating Full Submission with Challenge Validator...")
    is_valid = validate_submission(
        matching_file=matching_out_path,
        candidate_file=cand_out_path,
        test_dir=test_dir
    )

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"\n=======================================================")
    print(f" FULL INFERENCE COMPLETED IN {elapsed_min:.1f} MINUTES!")
    print(f" Submission Status: {'READY FOR PORTAL UPLOAD' if is_valid else 'VALIDATION FAILED'}")
    print(f"=======================================================")

    return is_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Test Inference Engine.")
    parser.add_argument("--train-dir", default="student_resource/dataset/train", help="Train dataset directory")
    parser.add_argument("--test-dir", default="student_resource/dataset/test", help="Test dataset directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--sample-train-s1", type=int, default=60000, help="Number of S1 train entities for model fitting")
    parser.add_argument("--top-k", type=int, default=25, help="Top K candidates per source")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Streaming batch size")

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
