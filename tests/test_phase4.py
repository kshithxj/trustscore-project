"""Phase 4 tests: Fairness policy, metric definitions, audit structure, and proxy-leakage feature sets."""

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
from src.features.model_a_features import FORBIDDEN_COLS
from src.fairness.policy import compute_threshold_from_validation, apply_decision_policy
from src.fairness.audit import compute_group_metrics, compute_fairness_gaps_hand

_CFG = load_config()
_METRICS = _PROJECT_ROOT / _CFG["paths"]["metrics"]
_PREDICTIONS = _PROJECT_ROOT / _CFG["paths"]["predictions"]
_SPLITS = _PROJECT_ROOT / _CFG["paths"]["splits"]
_TABLES = _PROJECT_ROOT / "reports" / "tables"


class TestFairnessPolicy:
    def test_thresholds_computed_on_validation_only_and_differ(self):
        val_a = pd.read_parquet(_PREDICTIONS / "model_a_val.parquet")
        val_b = pd.read_parquet(_PREDICTIONS / "model_b_val.parquet")

        thresh_a = compute_threshold_from_validation(val_a["score"].values, decline_rate=0.15)
        thresh_b = compute_threshold_from_validation(val_b["score"].values, decline_rate=0.15)

        assert 0.0 < thresh_a < 1.0
        assert 0.0 < thresh_b < 1.0
        assert thresh_a != thresh_b, "Model A and Model B should have separate numeric thresholds!"

    def test_realized_test_decline_rate_within_0_02_of_0_15(self):
        fairness_path = _METRICS / "fairness.json"
        assert fairness_path.exists(), "fairness.json missing!"
        with open(fairness_path, "r", encoding="utf-8") as f:
            fairness_data = json.load(f)

        decl_a = fairness_data["policy"]["realized_test_decline_a"]
        decl_b = fairness_data["policy"]["realized_test_decline_b"]

        assert abs(decl_a - 0.15) <= 0.02, f"Model A test decline rate {decl_a} not within 0.02 of 0.15"
        assert abs(decl_b - 0.15) <= 0.02, f"Model B test decline rate {decl_b} not within 0.02 of 0.15"


class TestFairnessMetricDefinitions:
    def test_dp_ratio_equals_min_div_max_approval_on_handmade_frame(self):
        # 10 applicants: 5 in Group 1, 5 in Group 2
        # Group 1: 4 approved (1 declined) -> approval_rate = 0.8
        # Group 2: 2 approved (3 declined) -> approval_rate = 0.4
        y_dummy = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        # decline: Group 1 has 1 decline, Group 2 has 3 declines
        decline = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1])

        g1_metrics = compute_group_metrics(y_dummy[:5], decline[:5])
        g2_metrics = compute_group_metrics(y_dummy[5:], decline[5:])

        assert g1_metrics["approval_rate"] == 0.8
        assert g2_metrics["approval_rate"] == 0.4

        gaps = compute_fairness_gaps_hand({"G1": g1_metrics, "G2": g2_metrics})
        expected_ratio = 0.4 / 0.8  # 0.5
        expected_diff = 0.8 - 0.4   # 0.4

        assert abs(gaps["dp_ratio"] - expected_ratio) < 1e-6
        assert abs(gaps["dp_difference"] - expected_diff) < 1e-6


class TestFairnessAuditIntegrity:
    @pytest.fixture(autouse=True)
    def _load(self):
        with open(_METRICS / "fairness.json", "r", encoding="utf-8") as f:
            self.audit = json.load(f)
        self.test_ids = pd.read_csv(_SPLITS / "test_ids.csv")

    def test_group_sizes_sum_to_test_split_excluding_xna(self):
        n_test = len(self.test_ids)  # 61,503
        xna_count = self.audit["policy"]["test_xna_count"]

        # Gender: F + M should equal n_test - xna_count
        gender_metrics = self.audit["overall"]["gender"]["model_a"]["group_metrics"]
        gender_total = gender_metrics["F"]["n"] + gender_metrics["M"]["n"]
        assert gender_total == n_test - xna_count

        # Binary age: below + above median should equal n_test
        age_metrics = self.audit["overall"]["binary_age"]["model_a"]["group_metrics"]
        age_total = age_metrics["<median"]["n"] + age_metrics[">=median"]["n"]
        assert age_total == n_test

        # Four age bins: sum of 4 bins should equal n_test
        bins_metrics = self.audit["overall"]["four_age_bins"]["model_a"]["group_metrics"]
        bins_total = sum(bins_metrics[b]["n"] for b in bins_metrics)
        assert bins_total == n_test

    def test_train_median_age_used(self):
        # Should be approximately 43.12
        med = self.audit["policy"]["train_median_age"]
        assert 40.0 <= med <= 46.0


class TestProxyPurity:
    def test_proxy_feature_sets_have_no_forbidden_columns(self):
        with open(_METRICS / "model_a_features.txt", "r", encoding="utf-8") as f:
            feats_a = [line.strip() for line in f if line.strip()]

        s1_cols = [f for f in feats_a if not f.startswith("bureau_")]

        from src.features.behavioral import ALL_BEHAVIORAL_COLS
        s2_cols = list(ALL_BEHAVIORAL_COLS)
        s3_cols = s1_cols + s2_cols

        for s_name, s_cols in [("S1", s1_cols), ("S2", s2_cols), ("S3", s3_cols)]:
            intersection = set(s_cols).intersection(FORBIDDEN_COLS)
            assert len(intersection) == 0, f"Forbidden columns found in {s_name}: {intersection}"

            # Check GENDER/AGE substring (only OWN_CAR_AGE allowed)
            for c in s_cols:
                if "GENDER" in c.upper() or "AGE" in c.upper():
                    assert c == "OWN_CAR_AGE", f"Unexpected column {c} in {s_name}"
