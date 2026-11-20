"""Phase 3 tests: Behavioral features, Model B feature matrix, paired evaluation, and artifacts."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config
from src.features.model_a_features import FORBIDDEN_COLS, validate_feature_list
from src.features.behavioral import ALL_BEHAVIORAL_COLS

_CFG = load_config()
_METRICS = _PROJECT_ROOT / _CFG["paths"]["metrics"]
_PREDICTIONS = _PROJECT_ROOT / _CFG["paths"]["predictions"]
_SPLITS = _PROJECT_ROOT / _CFG["paths"]["splits"]
_PROCESSED = _PROJECT_ROOT / _CFG["paths"]["processed"]


class TestModelBFeatures:
    def test_forbidden_features_absent(self):
        feat_path = _METRICS / "model_b_features.txt"
        assert feat_path.exists(), "model_b_features.txt missing!"
        with open(feat_path, "r", encoding="utf-8") as f:
            features = [line.strip() for line in f if line.strip()]

        intersection = set(features).intersection(FORBIDDEN_COLS)
        assert len(intersection) == 0, f"Forbidden features found: {intersection}"
        validate_feature_list(features)

    def test_model_b_contains_every_model_a_feature(self):
        feat_a_path = _METRICS / "model_a_features.txt"
        feat_b_path = _METRICS / "model_b_features.txt"

        with open(feat_a_path, "r", encoding="utf-8") as f:
            feats_a = [line.strip() for line in f if line.strip()]
        with open(feat_b_path, "r", encoding="utf-8") as f:
            feats_b = [line.strip() for line in f if line.strip()]

        for fa in feats_a:
            assert fa in feats_b, f"Model A feature {fa} missing in Model B!"

        assert len(feats_b) == len(feats_a) + len(ALL_BEHAVIORAL_COLS)

    def test_all_behavioral_features_present(self):
        feat_b_path = _METRICS / "model_b_features.txt"
        with open(feat_b_path, "r", encoding="utf-8") as f:
            feats_b = [line.strip() for line in f if line.strip()]

        for col in ALL_BEHAVIORAL_COLS:
            assert col in feats_b, f"Missing behavioral column: {col}"


class TestBehavioralMathAndIntegrity:
    def test_days_late_sign_tiny_handmade_frame(self):
        # Due at day -10, paid at day -7 -> paid 3 days late -> days_late = (-7) - (-10) = +3
        days_instalment = -10.0
        days_entry_payment = -7.0
        days_late = days_entry_payment - days_instalment
        assert days_late == 3.0, f"Expected +3 for payment 3 days after due, got {days_late}"

        # Due at day -10, paid at day -12 -> paid 2 days early -> days_late = (-12) - (-10) = -2
        days_entry_early = -12.0
        days_early = days_entry_early - days_instalment
        assert days_early == -2.0, f"Expected -2 for payment 2 days early, got {days_early}"

    def test_join_keeps_applicant_row_count(self):
        base_df = pd.read_parquet(_PROCESSED / "applicant_base.parquet", columns=["SK_ID_CURR"])
        beh_df = pd.read_parquet(_PROCESSED / "behavioral_features.parquet")
        assert len(base_df) == len(beh_df) == 307511
        assert beh_df["SK_ID_CURR"].is_unique

    def test_no_file_behavioral_columns_all_nan(self):
        base_df = pd.read_parquet(_PROCESSED / "applicant_base.parquet", columns=["SK_ID_CURR", "segment"])
        beh_df = pd.read_parquet(_PROCESSED / "behavioral_features.parquet")
        merged = base_df.merge(beh_df, on="SK_ID_CURR", how="left")

        no_file_df = merged[merged["segment"] == "no_file"][ALL_BEHAVIORAL_COLS]
        assert len(no_file_df) == 2200
        assert no_file_df.isna().all().all(), "Found non-NaN behavioral values for no_file applicants!"


class TestModelBPredictions:
    def test_predictions_cover_every_split_id_exactly_once(self):
        val_pred = pd.read_parquet(_PREDICTIONS / "model_b_val.parquet")
        test_pred = pd.read_parquet(_PREDICTIONS / "model_b_test.parquet")

        orig_val = pd.read_csv(_SPLITS / "valid_ids.csv")
        orig_test = pd.read_csv(_SPLITS / "test_ids.csv")

        assert len(val_pred) == len(orig_val) == 61502
        assert len(test_pred) == len(orig_test) == 61503

        assert set(val_pred["SK_ID_CURR"]) == set(orig_val["SK_ID_CURR"])
        assert set(test_pred["SK_ID_CURR"]) == set(orig_test["SK_ID_CURR"])

        assert val_pred["SK_ID_CURR"].is_unique
        assert test_pred["SK_ID_CURR"].is_unique

    def test_scores_in_valid_range_no_nan(self):
        val_pred = pd.read_parquet(_PREDICTIONS / "model_b_val.parquet")
        test_pred = pd.read_parquet(_PREDICTIONS / "model_b_test.parquet")

        for df, name in [(val_pred, "val"), (test_pred, "test")]:
            assert not df["score"].isna().any(), f"NaN found in Model B {name} score!"
            assert (df["score"] >= 0.0).all(), f"Negative score in Model B {name}!"
            assert (df["score"] <= 1.0).all(), f"Score > 1.0 in Model B {name}!"


class TestComparativeArtifacts:
    def test_a_vs_b_json_expected_keys(self):
        path = _METRICS / "a_vs_b.json"
        assert path.exists(), "a_vs_b.json not found"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required = {"comparison", "validation_auc", "test_auc"}
        assert required.issubset(set(data.keys()))
        assert "overall" in data["test_auc"]
        assert "by_segment" in data["test_auc"]
        assert "paired_bootstrap" in data["test_auc"]["overall"]

    def test_model_b_auc_in_valid_range(self):
        path = _METRICS / "model_b.json"
        assert path.exists()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        auc = data["test_results"]["overall"]["auc"]
        assert 0.60 <= auc <= 0.85, f"Model B test AUC {auc} out of valid range!"
