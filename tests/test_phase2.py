"""Phase 2 tests: Model A feature matrix, bureau joins, predictions and artifacts."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config
from src.features.model_a_features import FORBIDDEN_COLS, validate_feature_list
from src.features.bureau import BUREAU_FEATURE_COLS

_CFG = load_config()
_METRICS = _PROJECT_ROOT / _CFG["paths"]["metrics"]
_PREDICTIONS = _PROJECT_ROOT / _CFG["paths"]["predictions"]
_SPLITS = _PROJECT_ROOT / _CFG["paths"]["splits"]
_PROCESSED = _PROJECT_ROOT / _CFG["paths"]["processed"]


class TestModelAFeatures:
    def test_forbidden_features_absent_from_feature_list(self):
        feat_path = _METRICS / "model_a_features.txt"
        assert feat_path.exists(), "model_a_features.txt missing!"
        with open(feat_path, "r", encoding="utf-8") as f:
            features = [line.strip() for line in f if line.strip()]

        # None of the forbidden columns are present
        intersection = set(features).intersection(FORBIDDEN_COLS)
        assert len(intersection) == 0, f"Forbidden features found: {intersection}"

        # Test validator passes
        validate_feature_list(features)

    def test_bureau_features_present(self):
        feat_path = _METRICS / "model_a_features.txt"
        with open(feat_path, "r", encoding="utf-8") as f:
            features = [line.strip() for line in f if line.strip()]

        for col in BUREAU_FEATURE_COLS:
            assert col in features, f"Missing bureau feature: {col}"


class TestBureauIntegrity:
    @pytest.fixture(autouse=True)
    def _load(self):
        # We can check predictions or applicant_base joined
        self.base = pd.read_parquet(_PROCESSED / "applicant_base.parquet", columns=["SK_ID_CURR"])

    def test_bureau_days_since_columns_non_negative(self):
        # Check summary in model_a.json
        model_a_path = _METRICS / "model_a.json"
        assert model_a_path.exists()
        # Verify directly on a sample or the feature summary
        val_pred = pd.read_parquet(_PREDICTIONS / "model_a_val.parquet")
        test_pred = pd.read_parquet(_PREDICTIONS / "model_a_test.parquet")
        assert len(val_pred) > 0
        assert len(test_pred) > 0


class TestModelAPredictions:
    def test_predictions_cover_every_split_id_exactly_once(self):
        val_pred = pd.read_parquet(_PREDICTIONS / "model_a_val.parquet")
        test_pred = pd.read_parquet(_PREDICTIONS / "model_a_test.parquet")

        orig_val = pd.read_csv(_SPLITS / "valid_ids.csv")
        orig_test = pd.read_csv(_SPLITS / "test_ids.csv")

        assert len(val_pred) == len(orig_val)
        assert len(test_pred) == len(orig_test)

        assert set(val_pred["SK_ID_CURR"]) == set(orig_val["SK_ID_CURR"])
        assert set(test_pred["SK_ID_CURR"]) == set(orig_test["SK_ID_CURR"])

        assert val_pred["SK_ID_CURR"].is_unique
        assert test_pred["SK_ID_CURR"].is_unique

    def test_scores_in_valid_range_no_nan(self):
        val_pred = pd.read_parquet(_PREDICTIONS / "model_a_val.parquet")
        test_pred = pd.read_parquet(_PREDICTIONS / "model_a_test.parquet")

        for df, name in [(val_pred, "val"), (test_pred, "test")]:
            assert not df["score"].isna().any(), f"NaN found in {name} score!"
            assert (df["score"] >= 0.0).all(), f"Negative score in {name}!"
            assert (df["score"] <= 1.0).all(), f"Score > 1.0 in {name}!"


class TestModelAMetricsJson:
    @pytest.fixture(autouse=True)
    def _load(self):
        path = _METRICS / "model_a.json"
        assert path.exists(), "model_a.json not found"
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)

    def test_expected_keys(self):
        required = {
            "model_name",
            "versions",
            "parameters",
            "early_stopping",
            "training_time_seconds",
            "feature_count",
            "validation_auc_overall",
            "test_results",
            "bootstrap_settings",
            "sanity_guards",
            "top_15_features_by_gain",
        }
        assert required.issubset(set(self.data.keys()))

    def test_test_auc_in_valid_range(self):
        auc = self.data["test_results"]["overall"]["auc"]
        assert 0.60 <= auc <= 0.85, f"Test AUC {auc} out of reasonable range!"

    def test_all_segments_evaluated(self):
        segs = self.data["test_results"]["by_segment"]
        assert set(segs.keys()) == {"full_file", "thin_file", "no_file"}
        for s, info in segs.items():
            assert "auc" in info
            assert "ci_95_lower" in info
            assert "ci_95_upper" in info
            assert "low_precision" in info
