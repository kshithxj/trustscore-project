"""Tests for Phase 5: Base Rates, SHAP Explanations, and Fairness Mitigation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.utils import load_config
from src.features.model_b_features import prepare_model_b_datasets
from src.fairness.mitigation import Float64XGBWrapper

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPhase5Shap:
    """Verifies SHAP sampling and explanation integrity."""

    def test_shap_sample_ids_unique_and_subset_of_test(self):
        config = load_config()
        sample_ids_path = _PROJECT_ROOT / config["paths"]["reports"] / "tables" / "shap_sample_ids.csv"
        test_ids_path = _PROJECT_ROOT / config["paths"]["splits"] / "test_ids.csv"

        assert sample_ids_path.exists(), f"Missing {sample_ids_path}"
        assert test_ids_path.exists(), f"Missing {test_ids_path}"

        sample_df = pd.read_csv(sample_ids_path)
        test_df = pd.read_csv(test_ids_path)

        # Check uniqueness
        assert sample_df["SK_ID_CURR"].nunique() == len(sample_df), "Sample IDs must be unique"
        assert len(sample_df) == 4940, f"Expected 4940 samples (2500 full + 2000 thin + 440 no), got {len(sample_df)}"

        # Check subset of test split
        test_set = set(test_df["SK_ID_CURR"])
        assert set(sample_df["SK_ID_CURR"]).issubset(test_set), "All sample IDs must belong to the test split"

    def test_shap_summary_and_figures_exist(self):
        config = load_config()
        summary_path = _PROJECT_ROOT / config["paths"]["metrics"] / "shap_summary.json"
        assert summary_path.exists(), "shap_summary.json must exist"

        with open(summary_path, "r") as f:
            summary = json.load(f)

        assert summary["metadata"]["method"] == "shap.TreeExplainer"
        assert summary["metadata"]["scale"] == "margin (log-odds)"
        assert summary["metadata"]["sample_sizes"]["total"] == 4940

        figures_dir = _PROJECT_ROOT / config["paths"]["figures"]
        required_figs = [
            "shap_importance_full_file.png",
            "shap_importance_thin_file.png",
            "shap_behavioral_share_by_segment.png",
            "shap_by_gender.png",
            "shap_by_binary_age.png",
            "shap_dependence_top_behavioral.png",
            "shap_local_declined_thin_file.png",
        ]
        for fig_name in required_figs:
            p = figures_dir / fig_name
            assert p.exists(), f"Missing expected figure: {fig_name}"


class TestPhase5Mitigation:
    """Verifies ThresholdOptimizer setup, score fidelity, and tradeoff structure."""

    def test_wrapped_estimator_scores_match_saved_model_b(self):
        config = load_config()
        X_train, y_train, X_val, y_val, X_test, y_test, meta_tr, meta_va, meta_te, feature_names, cat_maps = (
            prepare_model_b_datasets(config)
        )

        model_path = _PROJECT_ROOT / "models" / "model_b.json"
        raw_model = xgb.XGBClassifier()
        raw_model.load_model(str(model_path))
        wrapped_model = Float64XGBWrapper(raw_model)

        val_preds_saved = pd.read_parquet(_PROJECT_ROOT / config["paths"]["predictions"] / "model_b_val.parquet")
        test_preds_saved = pd.read_parquet(_PROJECT_ROOT / config["paths"]["predictions"] / "model_b_test.parquet")

        val_scores = wrapped_model.predict_proba(X_val)[:, 1]
        test_scores = wrapped_model.predict_proba(X_test)[:, 1]

        val_merged = pd.merge(val_preds_saved, pd.DataFrame({"SK_ID_CURR": meta_va["SK_ID_CURR"].values, "score_w": val_scores}), on="SK_ID_CURR")
        test_merged = pd.merge(test_preds_saved, pd.DataFrame({"SK_ID_CURR": meta_te["SK_ID_CURR"].values, "score_w": test_scores}), on="SK_ID_CURR")

        val_diff = np.max(np.abs(val_merged["score"] - val_merged["score_w"]))
        test_diff = np.max(np.abs(test_merged["score"] - test_merged["score_w"]))

        assert val_diff < 1e-5, f"Validation scores mismatch: max abs diff {val_diff}"
        assert test_diff < 1e-5, f"Test scores mismatch: max abs diff {test_diff}"

    def test_tradeoff_structure_and_constraints(self):
        config = load_config()
        tradeoff_path = _PROJECT_ROOT / config["paths"]["reports"] / "tables" / "tradeoff.csv"
        assert tradeoff_path.exists(), "tradeoff.csv must exist"

        df = pd.read_csv(tradeoff_path)
        assert len(df) == 6, f"Expected 6 rows (2 attributes x 3 policies), got {len(df)}"

        for attr in ["gender", "binary_age"]:
            sub = df[df["sensitive_feature"] == attr]
            policies = set(sub["policy"])
            assert policies == {"unmitigated", "equalized_odds", "demographic_parity"}

            # AUC must be identical across rows for the same underlying score
            aucs = sub["auc"].round(6).unique()
            assert len(aucs) == 1, f"AUC must be identical across policies for {attr}, got {aucs}"

    def test_xna_rows_excluded_and_objective_recorded(self):
        config = load_config()
        mitigation_path = _PROJECT_ROOT / config["paths"]["metrics"] / "mitigation.json"
        assert mitigation_path.exists(), "mitigation.json must exist"

        with open(mitigation_path, "r") as f:
            data = json.load(f)

        meta = data["metadata"]
        assert meta["objective"] == "balanced_accuracy_score"
        assert meta["prefit_supported"] is True
        assert meta["test_xna_count"] == 1
        assert "disclaimers" in meta and len(meta["disclaimers"]) >= 5


class TestPhase5BaseRates:
    """Verifies descriptive base rates table on the test split."""

    def test_base_rates_table(self):
        config = load_config()
        base_rates_path = _PROJECT_ROOT / config["paths"]["reports"] / "tables" / "base_rates.csv"
        assert base_rates_path.exists(), "base_rates.csv must exist"

        df = pd.read_csv(base_rates_path)
        dims = set(df["dimension"])
        expected_dims = {"gender", "binary_age", "age_bin", "segment", "segment_x_gender", "segment_x_binary_age"}
        assert expected_dims.issubset(dims), f"Expected dimensions {expected_dims}, found {dims}"

        # All groups must have size >= 100
        assert (df["size"] >= 100).all(), "All groups in base_rates.csv must have size >= 100"
