"""
Master End-to-End Pipeline for Entity Resolution.
Coordinates:
Stage 0: EDA & Validation Check
Stage 1: Multi-view Text Normalization
Stage 2: High-Recall Multi-Blocker Candidate Generation
Stage 3: Pairwise Feature Engineering
Stage 4: LightGBM GBDT Training & GroupKFold Validation
Stage 5 & 6: Global Consistency (One-Owner Rule) & Decision Threshold Optimization
Stage 7: Output Generation & Automatic Submission Validation
"""

import os
import sys
import argparse

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

from src.eda import run_eda
from src.normalization import preprocess_dataframe
from src.blocking import MultiBlocker, evaluate_blocking_recall
from src.features import PairFeatureExtractor
from src.model import EntityMatcherModel
from src.postprocessing import generate_matching_predictions
from utils.validate_submission import validate_submission


def run_pipeline(
    train_dir: str = "student_resource/dataset/train",
    test_dir: str = "student_resource/dataset/test",
    output_dir: str = "output",
    top_k: int = 25,
    sample_train_size: int = 150000
):
    print("=" * 70)
    print("      ENTITY RESOLUTION END-TO-END PIPELINE")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # STAGE 0: EDA
    # -------------------------------------------------------------------------
    eda_summary = run_eda(train_dir)

    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------
    print("\n[Loading Datasets...]")
    df_s1_tr = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", dtype=str)
    df_s2_tr = pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", dtype=str)
    df_s3_tr = pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", dtype=str)
    df_gt_tr = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", dtype=str, keep_default_na=False)

    df_s1_te = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", dtype=str)
    df_s2_te = pd.read_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t", dtype=str)
    df_s3_te = pd.read_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t", dtype=str)

    # Ground truth mapping: source1_entity_id -> list of matched IDs
    gt_dict = {}
    for _, row in df_gt_tr.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        val = str(row["matched_entity_ids"]).strip()
        gt_dict[s1_id] = [x.strip() for x in val.split(",") if x.strip()]

    # -------------------------------------------------------------------------
    # STAGE 1: NORMALIZATION
    # -------------------------------------------------------------------------
    print("\n[Stage 1: Preprocessing & Normalizing Text Fields...]")
    df_s1_tr = preprocess_dataframe(df_s1_tr)
    df_s2_tr = preprocess_dataframe(df_s2_tr)
    df_s3_tr = preprocess_dataframe(df_s3_tr)

    df_s1_te = preprocess_dataframe(df_s1_te)
    df_s2_te = preprocess_dataframe(df_s2_te)
    df_s3_te = preprocess_dataframe(df_s3_te)

    # Combine targets
    df_targets_tr = pd.concat([df_s2_tr, df_s3_tr], ignore_index=True)
    df_targets_te = pd.concat([df_s2_te, df_s3_te], ignore_index=True)

    # -------------------------------------------------------------------------
    # STAGE 2: BLOCKING (CANDIDATE GENERATION)
    # -------------------------------------------------------------------------
    print("\n[Stage 2: Candidate Generation / Multi-Blocking...]")
    all_corpus_texts = (
        list(df_s1_tr["combined_text"]) + list(df_targets_tr["combined_text"]) +
        list(df_s1_te["combined_text"]) + list(df_targets_te["combined_text"])
    )

    blocker = MultiBlocker(top_k_per_source=top_k)
    blocker.fit_vectorizers(all_corpus_texts)

    print("  Generating training candidates...")
    train_cands = blocker.generate_candidates(df_s1_tr, df_s2_tr, df_s3_tr)
    blocking_metrics = evaluate_blocking_recall(train_cands, gt_dict)
    print(f"  >>> Training Candidate Recall: {blocking_metrics['candidate_recall']*100:.2f}%")
    print(f"      Captured: {blocking_metrics['captured_pairs']}/{blocking_metrics['total_true_pairs']} true pairs")
    print(f"      Avg candidates per S1: {blocking_metrics['avg_candidates_per_s1']:.1f}")

    print("  Generating test candidates...")
    test_cands = blocker.generate_candidates(df_s1_te, df_s2_te, df_s3_te)

    # -------------------------------------------------------------------------
    # STAGE 3: PAIR FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    print("\n[Stage 3: Extracting Pairwise Features...]")
    extractor = PairFeatureExtractor(idf_dict=blocker.idf_dict)

    print("  Extracting training pair features...")
    df_train_pairs = extractor.build_feature_table(
        df_s1=df_s1_tr,
        df_target=df_targets_tr,
        candidates_dict=train_cands,
        ground_truth_dict=gt_dict
    )
    print(f"  Generated {len(df_train_pairs)} training pair samples (Positives: {df_train_pairs['label'].sum()}).")

    print("  Extracting test pair features...")
    df_test_pairs = extractor.build_feature_table(
        df_s1=df_s1_te,
        df_target=df_targets_te,
        candidates_dict=test_cands,
        ground_truth_dict=None
    )
    print(f"  Generated {len(df_test_pairs)} test pair samples to score.")

    # Write candidate_pairs.tsv directly from the final scored table
    cand_out_path = os.path.join(output_dir, "candidate_pairs.tsv")
    cand_dict_scored = df_test_pairs.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()
    cand_rows = []
    for s1_id in df_s1_te["entity_id"]:
        cand_list = cand_dict_scored.get(s1_id, [])
        cand_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cand_list)
        })
    df_cand_out = pd.DataFrame(cand_rows)
    df_cand_out.to_csv(cand_out_path, sep="\t", index=False)
    print(f"  [Saved] {cand_out_path}")

    # -------------------------------------------------------------------------
    # STAGE 4: MODEL TRAINING & THRESHOLD OPTIMIZATION
    # -------------------------------------------------------------------------
    print("\n[Stage 4: Training GBDT Model & Optimizing Decision Threshold...]")
    matcher = EntityMatcherModel(n_estimators=300, learning_rate=0.05, max_depth=6)
    cv_results = matcher.train_cv(
        df_train_pairs=df_train_pairs,
        ground_truth_dict=gt_dict,
        n_splits=5
    )

    # -------------------------------------------------------------------------
    # STAGE 5 & 6: PREDICTION & POSTPROCESSING
    # -------------------------------------------------------------------------
    print("\n[Stage 5 & 6: Test Inference & Consistency Postprocessing...]")
    df_test_scored = matcher.predict_pairs(df_test_pairs)

    all_test_s1_ids = list(df_s1_te["entity_id"].values)
    final_predictions = generate_matching_predictions(
        df_scored_pairs=df_test_scored,
        all_s1_ids=all_test_s1_ids,
        threshold=matcher.best_threshold,
        apply_one_owner=eda_summary["one_owner_valid"],
        prob_col="pred_prob"
    )

    # Write matching_results.tsv
    matching_out_path = os.path.join(output_dir, "matching_results.tsv")
    match_rows = []
    for s1_id in all_test_s1_ids:
        matched_list = final_predictions.get(s1_id, [])
        match_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched_list)
        })
    df_match_out = pd.DataFrame(match_rows)
    df_match_out.to_csv(matching_out_path, sep="\t", index=False)
    print(f"  [Saved] {matching_out_path}")

    # -------------------------------------------------------------------------
    # STAGE 7: SUBMISSION VALIDATION
    # -------------------------------------------------------------------------
    print("\n[Stage 7: Validating Submission Against Hard Constraints...]")
    is_valid = validate_submission(
        matching_file=matching_out_path,
        candidate_file=cand_out_path,
        test_dir=test_dir
    )

    if is_valid:
        print("\n=======================================================")
        print(" PIPELINE EXECUTION SUCCESSFUL: ALL CHECKS PASSED!")
        print("=======================================================")
    else:
        print("\n[!] WARNING: Validation issues detected. Check errors above.")

    return is_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete Entity Resolution pipeline.")
    parser.add_argument("--train-dir", default="student_resource/dataset/train", help="Path to training directory")
    parser.add_argument("--test-dir", default="student_resource/dataset/test", help="Path to test directory")
    parser.add_argument("--output-dir", default="output", help="Path to output directory")
    parser.add_argument("--top-k", type=int, default=25, help="Top K candidates per source blocker")

    args = parser.parse_args()
    success = run_pipeline(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        top_k=args.top_k
    )
    sys.exit(0 if success else 1)
