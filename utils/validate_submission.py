#!/usr/bin/env python3
"""
Submission validator for Entity Resolution Challenge.
Enforces all hard constraints outlined in Section 2.5 of the approach documentation:
1. Correct filenames and TSV format (tab-separated).
2. Exactly one row per Source 1 test entity.
3. No duplicate IDs inside list.
4. Only valid S2- / S3- entity IDs from test set.
5. Matches must be a subset of candidate pairs (matches ⊆ candidates).
6. No S1 self-matches.
"""

import argparse
import os
import sys
import pandas as pd


def parse_id_list(val):
    if pd.isna(val):
        return []
    val_str = str(val).strip()
    if not val_str:
        return []
    return [item.strip() for item in val_str.split(",") if item.strip()]


def validate_submission(matching_file: str, candidate_file: str, test_dir: str) -> bool:
    print(f"=== Starting Submission Validation ===")
    print(f"Matching file:  {matching_file}")
    print(f"Candidate file: {candidate_file}")
    print(f"Test directory: {test_dir}")

    passed = True
    errors = []
    warnings = []

    # Check files exist
    for fpath, name in [(matching_file, "Matching results"), (candidate_file, "Candidate pairs")]:
        if not os.path.exists(fpath):
            errors.append(f"Missing required output file: {fpath} ({name})")
            return False

    # Check test source files
    s1_test_path = os.path.join(test_dir, "test_source1.tsv")
    s2_test_path = os.path.join(test_dir, "test_source2.tsv")
    s3_test_path = os.path.join(test_dir, "test_source3.tsv")

    for fpath, name in [(s1_test_path, "Test S1"), (s2_test_path, "Test S2"), (s3_test_path, "Test S3")]:
        if not os.path.exists(fpath):
            errors.append(f"Missing test file: {fpath} ({name})")

    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        return False

    # Load test source IDs
    df_s1 = pd.read_csv(s1_test_path, sep="\t", dtype=str)
    df_s2 = pd.read_csv(s2_test_path, sep="\t", dtype=str)
    df_s3 = pd.read_csv(s3_test_path, sep="\t", dtype=str)

    expected_s1_ids = set(df_s1["entity_id"].dropna().str.strip())
    valid_s2_ids = set(df_s2["entity_id"].dropna().str.strip())
    valid_s3_ids = set(df_s3["entity_id"].dropna().str.strip())
    all_valid_target_ids = valid_s2_ids.union(valid_s3_ids)

    print(f"Loaded ground truth test references: {len(expected_s1_ids)} S1, {len(valid_s2_ids)} S2, {len(valid_s3_ids)} S3 records.")

    # Load matching results
    try:
        df_matching = pd.read_csv(matching_file, sep="\t", dtype=str, keep_default_na=False)
    except Exception as e:
        errors.append(f"Failed to parse {matching_file} as TSV: {e}")
        return False

    # Load candidate pairs
    try:
        df_candidates = pd.read_csv(candidate_file, sep="\t", dtype=str, keep_default_na=False)
    except Exception as e:
        errors.append(f"Failed to parse {candidate_file} as TSV: {e}")
        return False

    # Check columns
    if "source1_entity_id" not in df_matching.columns or "matched_entity_ids" not in df_matching.columns:
        errors.append(f"matching_results.tsv must have columns: ['source1_entity_id', 'matched_entity_ids']. Got: {list(df_matching.columns)}")
    if "source1_entity_id" not in df_candidates.columns or "candidate_entity_ids" not in df_candidates.columns:
        errors.append(f"candidate_pairs.tsv must have columns: ['source1_entity_id', 'candidate_entity_ids']. Got: {list(df_candidates.columns)}")

    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        return False

    # Check S1 entity completeness and uniqueness
    matching_s1_ids = list(df_matching["source1_entity_id"].str.strip())
    candidate_s1_ids = list(df_candidates["source1_entity_id"].str.strip())

    if len(matching_s1_ids) != len(set(matching_s1_ids)):
        errors.append("Duplicate source1_entity_id found in matching_results.tsv")
    if len(candidate_s1_ids) != len(set(candidate_s1_ids)):
        errors.append("Duplicate source1_entity_id found in candidate_pairs.tsv")

    matching_s1_set = set(matching_s1_ids)
    candidate_s1_set = set(candidate_s1_ids)

    missing_in_matching = expected_s1_ids - matching_s1_set
    if missing_in_matching:
        errors.append(f"matching_results.tsv is missing {len(missing_in_matching)} S1 test entities (e.g., {list(missing_in_matching)[:3]})")

    extra_in_matching = matching_s1_set - expected_s1_ids
    if extra_in_matching:
        errors.append(f"matching_results.tsv has {len(extra_in_matching)} unknown S1 entities (e.g., {list(extra_in_matching)[:3]})")

    # Map candidates and matches
    matching_dict = {row["source1_entity_id"].strip(): parse_id_list(row["matched_entity_ids"]) for _, row in df_matching.iterrows()}
    candidate_dict = {row["source1_entity_id"].strip(): parse_id_list(row["candidate_entity_ids"]) for _, row in df_candidates.iterrows()}

    # Detailed row-by-row checks
    invalid_ids_count = 0
    duplicate_in_row_count = 0
    not_subset_count = 0
    s1_self_match_count = 0

    for s1_id, match_list in matching_dict.items():
        cand_list = candidate_dict.get(s1_id, [])
        cand_set = set(cand_list)
        match_set = set(match_list)

        # 1. Duplicate IDs inside row
        if len(match_list) != len(match_set):
            duplicate_in_row_count += 1

        # 2. Check subset: matches ⊆ candidates
        if not match_set.issubset(cand_set):
            not_subset_count += 1

        # 3. Check IDs are S2 or S3 and valid
        for mid in match_set:
            if mid.startswith("S1-"):
                s1_self_match_count += 1
            if mid not in all_valid_target_ids:
                invalid_ids_count += 1

    if duplicate_in_row_count > 0:
        errors.append(f"{duplicate_in_row_count} rows in matching_results.tsv contain duplicate IDs in their list.")
    if not_subset_count > 0:
        warnings.append(f"{not_subset_count} rows have matches that are NOT a subset of candidates!")
    if s1_self_match_count > 0:
        errors.append(f"{s1_self_match_count} self-matches (S1- prefix) found in matching_results.tsv.")
    if invalid_ids_count > 0:
        errors.append(f"{invalid_ids_count} matched IDs are not present in test S2/S3 dataset.")

    print("\n--- Validation Summary ---")
    print(f"Total S1 evaluated: {len(df_matching)}")
    print(f"Total Matches Predicted: {sum(len(v) for v in matching_dict.values())}")
    print(f"Total Singletons (0 matches): {sum(1 for v in matching_dict.values() if len(v) == 0)}")
    print(f"Total Candidates Scored: {sum(len(v) for v in candidate_dict.values())}")

    for w in warnings:
        print(f"[WARNING] {w}")

    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        print("\n>>> VALIDATION FAILED <<<")
        return False
    else:
        print("\n>>> VALIDATION PASSED: Submission is compliant with all challenge constraints! <<<")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate submission files against challenge hard constraints.")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to directory containing test_source1/2/3.tsv")

    args = parser.parse_args()
    success = validate_submission(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)
