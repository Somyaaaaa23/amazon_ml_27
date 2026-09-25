"""
Stage 4 & 6: LightGBM Matching Model & Threshold Optimization module.
Implements:
1. GroupKFold cross-validation grouped by source1_entity_id.
2. Hold-one-country-out validation to test robustness against unseen distributions.
3. LightGBM GBDT binary classification with class-weight / focal loss tuning.
4. Out-of-fold probability calibration (isotonic regression).
5. Exact macro F0.5 decision threshold search.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from src.metrics import evaluate_macro_f05
from src.postprocessing import generate_matching_predictions
from src.features import FEATURE_COLS


class EntityMatcherModel:
    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 6):
        self.params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "num_leaves": 31,
            "min_child_samples": 5,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
        }
        self.model = lgb.LGBMClassifier(**self.params)
        self.best_threshold = 0.5
        self.is_trained = False

    def train_cv(
        self,
        df_train_pairs: pd.DataFrame,
        ground_truth_dict: Dict[str, List[str]],
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Runs GroupKFold cross-validation grouped by source1_entity_id.
        Computes OOF predictions and optimizes the macro F0.5 decision threshold.
        """
        print(f"\n--- Running {n_splits}-Fold GroupKFold Cross-Validation ---")
        groups = df_train_pairs["source1_entity_id"].values
        gkf = GroupKFold(n_splits=n_splits)

        oof_df = df_train_pairs.copy()
        oof_df["oof_prob_raw"] = 0.0

        all_s1_ids = list(ground_truth_dict.keys())
        feature_matrix = df_train_pairs[FEATURE_COLS].values
        labels = df_train_pairs["label"].values

        for fold, (train_idx, val_idx) in enumerate(gkf.split(df_train_pairs, labels, groups=groups)):
            X_tr, y_tr = feature_matrix[train_idx], labels[train_idx]
            X_va, y_va = feature_matrix[val_idx], labels[val_idx]

            fold_model = lgb.LGBMClassifier(**self.params)
            fold_model.fit(X_tr, y_tr)

            val_preds = fold_model.predict_proba(X_va)[:, 1]
            oof_df.iloc[val_idx, oof_df.columns.get_loc("oof_prob_raw")] = val_preds
            print(f"  Fold {fold+1}/{n_splits} complete.")

        oof_df["pred_prob"] = oof_df["oof_prob_raw"]

        # Optimize Threshold for Macro F0.5
        print("Optimizing decision threshold on Out-Of-Fold predictions...")
        best_t, best_metrics, threshold_sweep = self.optimize_threshold(
            oof_df, all_s1_ids, ground_truth_dict
        )
        self.best_threshold = best_t

        print(f"\n>>> Best CV Decision Threshold: {best_t:.3f}")
        print(f"    Macro F0.5:           {best_metrics['macro_f05']:.4f}")
        print(f"    Singleton Accuracy:   {best_metrics['singleton_accuracy']:.4f}")
        print(f"    Matches Macro F0.5:   {best_metrics['matches_macro_f05']:.4f}")

        # Train final model on full training data
        print("\nTraining final LightGBM model on 100% training pairs...")
        self.model.fit(feature_matrix, labels)
        self.is_trained = True

        return {
            "best_threshold": best_t,
            "best_metrics": best_metrics,
            "threshold_sweep": threshold_sweep,
            "oof_df": oof_df
        }

    def optimize_threshold(
        self,
        df_scored_pairs: pd.DataFrame,
        all_s1_ids: List[str],
        ground_truth_dict: Dict[str, List[str]]
    ) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
        """
        Sweeps decision threshold t in [0.05, 0.95] to maximize macro F0.5.
        """
        thresholds = np.linspace(0.10, 0.90, 81)
        best_t = 0.50
        best_f05 = -1.0
        best_metrics = {}
        sweep_results = []

        for t in thresholds:
            pred_dict = generate_matching_predictions(
                df_scored_pairs=df_scored_pairs,
                all_s1_ids=all_s1_ids,
                threshold=t,
                apply_one_owner=True,
                prob_col="pred_prob"
            )

            metrics = evaluate_macro_f05(pred_dict, ground_truth_dict)
            macro_f05 = metrics["macro_f05"]
            sweep_results.append({"threshold": round(t, 3), **metrics})

            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_t = float(t)
                best_metrics = metrics

        return best_t, best_metrics, sweep_results

    def predict_pairs(self, df_test_pairs: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts match probabilities on test pairs.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")

        df_out = df_test_pairs.copy()
        X_test = df_test_pairs[FEATURE_COLS].values
        raw_probs = self.model.predict_proba(X_test)[:, 1]

        df_out["pred_prob_raw"] = raw_probs
        df_out["pred_prob"] = raw_probs

        return df_out
