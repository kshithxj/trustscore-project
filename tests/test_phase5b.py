"""Tests for Phase 5b: Matched-Decline-Rate Control for Mitigation Comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils import load_config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPhase5bMatchedControl:
    """Verifies fidelity and integrity of the matched-rate control experiments."""

    def test_reproduction_check_recorded_and_near_zero(self):
        config = load_config()
        json_path = _PROJECT_ROOT / config["paths"]["metrics"] / "mitigation_control.json"
        assert json_path.exists(), f"Missing {json_path}"

        with open(json_path, "r") as f:
            data = json.load(f)

        max_diffs = data["metadata"]["max_reproduction_abs_diff"]
        assert max_diffs["decline_rate"] < 1e-5, f"Decline rate reproduction diff too large: {max_diffs['decline_rate']}"
        assert max_diffs["dp_difference"] < 1e-5, f"DP diff reproduction diff too large: {max_diffs['dp_difference']}"

        for run_name, check in data["reproduction_verification"].items():
            assert check["abs_diff_decline"] < 1e-5
            assert check["abs_diff_dp"] < 1e-5

    def test_control_validation_decline_rate_matches_mitigated_within_0_01(self):
        config = load_config()
        csv_path = _PROJECT_ROOT / config["paths"]["reports"] / "tables" / "tradeoff_matched.csv"
        assert csv_path.exists(), f"Missing {csv_path}"

        df = pd.read_csv(csv_path)

        for (sens, constraint), sub in df.groupby(["sensitive_feature", "constraint"]):
            assert len(sub) == 2, f"Expected 2 rows for ({sens}, {constraint}), got {len(sub)}"
            mit_row = sub[sub["policy_variant"] == "mitigated"].iloc[0]
            ctrl_row = sub[sub["policy_variant"] == "control"].iloc[0]

            val_decline_mit = mit_row["validation_decline_rate"]
            val_decline_ctrl = ctrl_row["validation_decline_rate"]
            assert abs(val_decline_mit - val_decline_ctrl) < 0.01, (
                f"Validation decline mismatch for ({sens}, {constraint}): "
                f"mit={val_decline_mit}, ctrl={val_decline_ctrl}"
            )

            # Test decline rate of control should also be close (within 0.01) of validation target
            test_decline_ctrl = ctrl_row["decline_rate"]
            assert abs(test_decline_ctrl - val_decline_mit) < 0.01, (
                f"Test decline rate for control deviates by >0.01 from target: "
                f"target={val_decline_mit}, test_ctrl={test_decline_ctrl}"
            )

    def test_control_uses_single_threshold(self):
        config = load_config()
        csv_path = _PROJECT_ROOT / config["paths"]["reports"] / "tables" / "tradeoff_matched.csv"
        df = pd.read_csv(csv_path)

        ctrl_rows = df[df["policy_variant"] == "control"]
        assert len(ctrl_rows) == 4, f"Expected 4 control rows, got {len(ctrl_rows)}"
        assert ctrl_rows["control_threshold"].notna().all(), "Control threshold must be recorded for all control rows"
        assert (ctrl_rows["control_threshold"] > 0).all(), "Control threshold must be positive"

        # Mitigated rows should not use a single scalar threshold
        mit_rows = df[df["policy_variant"] == "mitigated"]
        assert mit_rows["control_threshold"].isna().all(), "Mitigated rows should have NaN control_threshold"

    def test_matched_table_has_all_four_runs_with_both_rows(self):
        config = load_config()
        csv_path = _PROJECT_ROOT / config["paths"]["reports"] / "tables" / "tradeoff_matched.csv"
        df = pd.read_csv(csv_path)

        assert len(df) == 8, f"Expected 8 rows (4 runs x 2 variants), got {len(df)}"
        expected_runs = {
            ("gender", "equalized_odds"),
            ("gender", "demographic_parity"),
            ("binary_age", "equalized_odds"),
            ("binary_age", "demographic_parity"),
        }
        actual_runs = set(zip(df["sensitive_feature"], df["constraint"]))
        assert actual_runs == expected_runs, f"Expected {expected_runs}, got {actual_runs}"

    def test_xna_excluded_in_control_json(self):
        config = load_config()
        json_path = _PROJECT_ROOT / config["paths"]["metrics"] / "mitigation_control.json"
        with open(json_path, "r") as f:
            data = json.load(f)

        assert data["metadata"]["test_xna_count"] == 1
        assert data["metadata"]["val_xna_count"] == 1
