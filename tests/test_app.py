"""Unit and integration tests for TrustScore Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_app_module_imports():
    """Verify that src.app modules import without error."""
    import src.app
    import src.app.helpers as helpers
    import src.app.streamlit_app as app
    import src.app.precompute as precompute

    assert helpers is not None
    assert app is not None
    assert precompute is not None


def test_every_file_app_reads_exists():
    """Verify that every file the Streamlit app reads exists on disk."""
    required_files = [
        _PROJECT_ROOT / "reports" / "metrics" / "fairness.json",
        _PROJECT_ROOT / "reports" / "metrics" / "model_a.json",
        _PROJECT_ROOT / "reports" / "metrics" / "model_b.json",
        _PROJECT_ROOT / "reports" / "metrics" / "a_vs_b.json",
        _PROJECT_ROOT / "reports" / "metrics" / "extras_stability.json",
        _PROJECT_ROOT / "reports" / "metrics" / "extras_flips.json",
        _PROJECT_ROOT / "reports" / "tables" / "base_rates.csv",
        _PROJECT_ROOT / "reports" / "tables" / "app_frontier.csv",
        _PROJECT_ROOT / "reports" / "tables" / "extras_calibration_bins.csv",
        _PROJECT_ROOT / "reports" / "metrics" / "extras_calibration.json",
        _PROJECT_ROOT / "reports" / "metrics" / "extras_auc_by_group.json",
        _PROJECT_ROOT / "reports" / "metrics" / "proxy.json",
        _PROJECT_ROOT / "reports" / "metrics" / "shap_summary.json",
        _PROJECT_ROOT / "reports" / "metrics" / "mitigation_control.json",
        _PROJECT_ROOT / "reports" / "metrics" / "extras_ablation.json",
        _PROJECT_ROOT / "reports" / "figures" / "extras_ablation.png",
        _PROJECT_ROOT / "reports" / "figures" / "shap_local_declined_thin_file.png",
        _PROJECT_ROOT / "reports" / "shap_notes.md",
        _PROJECT_ROOT / "docs" / "limitations.md",
        _PROJECT_ROOT / "reports" / "predictions" / "extras" / "app_shap_sample.parquet",
        _PROJECT_ROOT / "reports" / "predictions" / "model_a_test.parquet",
        _PROJECT_ROOT / "reports" / "predictions" / "model_b_test.parquet",
        _PROJECT_ROOT / "reports" / "predictions" / "model_a_val.parquet",
        _PROJECT_ROOT / "reports" / "predictions" / "model_b_val.parquet",
        _PROJECT_ROOT / "data" / "processed" / "eval_attrs.parquet",
        _PROJECT_ROOT / "data" / "splits" / "train_ids.csv",
    ]

    for p in required_files:
        assert p.exists(), f"Required file missing: {p}"


def test_helper_metrics_at_15_match_fairness_json():
    """Verify that helper's live metrics at 15% equal fairness.json within 1e-6 for A and B, gender and binary age."""
    from src.app.helpers import check_consistency_at_15

    res = check_consistency_at_15()
    assert res["passed"] is True, f"Consistency check failed: max difference is {res['max_diff']:.2e} > 1e-6"
    assert res["max_diff"] <= 1e-6, f"Max difference {res['max_diff']:.2e} exceeds 1e-6"


def test_gender_xna_excluded_from_gender_metrics():
    """Verify that applicants with gender == 'XNA' are strictly excluded from gender analyses."""
    from src.app.helpers import compute_live_policy, load_test_applicants_data

    df = load_test_applicants_data()
    xna_count = (df["gender"] == "XNA").sum()
    assert xna_count > 0, "Test data should contain XNA rows to test exclusion"

    res = compute_live_policy("Model A", "gender", 0.15)
    assert "XNA" not in res["group_stats"], "XNA should be excluded from gender group stats"
    assert set(res["group_stats"].keys()) == {"F", "M"}


def test_frontier_file_row_counts():
    """Verify that app_frontier.csv exists and has exactly 26 rows per model."""
    frontier_path = _PROJECT_ROOT / "reports" / "tables" / "app_frontier.csv"
    assert frontier_path.exists(), "app_frontier.csv does not exist"

    df = pd.read_csv(frontier_path)
    assert len(df) == 52, f"Total rows in app_frontier.csv should be 52, got {len(df)}"

    counts = df["model"].value_counts()
    assert counts.get("Model A", 0) == 26, f"Model A rows: {counts.get('Model A', 0)} != 26"
    assert counts.get("Model B", 0) == 26, f"Model B rows: {counts.get('Model B', 0)} != 26"


def test_shap_local_accuracy():
    """Verify that for every applicant in app_shap_sample.parquet, sigmoid(bias + sum of contributions) == Model B test score within 1e-5."""
    sample_path = _PROJECT_ROOT / "reports" / "predictions" / "extras" / "app_shap_sample.parquet"
    assert sample_path.exists(), "app_shap_sample.parquet missing"

    df = pd.read_parquet(sample_path)
    assert len(df) == 4940, f"Expected 4940 rows, got {len(df)}"

    shap_cols = [c for c in df.columns if c.startswith("shap_")]
    assert len(shap_cols) == 144, f"Expected 144 SHAP columns, got {len(shap_cols)}"

    bias = df["bias"].values
    sum_contribs = df[shap_cols].sum(axis=1).values
    margin = bias + sum_contribs
    computed_prob = 1.0 / (1.0 + np.exp(-margin))

    actual_prob = df["score_b"].values
    diff = np.abs(computed_prob - actual_prob)
    max_diff = float(np.max(diff))

    assert max_diff <= 1e-5, f"Local accuracy check failed: max difference {max_diff:.2e} > 1e-5"


def test_streamlit_pages_render_without_exception():
    """Test that all 10 pages in streamlit_app.py render without exception using AppTest."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1.AppTest not available in this environment")

    app_path = str(_PROJECT_ROOT / "src" / "app" / "streamlit_app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    assert not at.exception, f"Initial run raised exception: {at.exception}"

    pages = [
        "Guided tour",
        "Does behavior help?",
        "Flip explorer",
        "Policy simulator",
        "Fairness and calibration",
        "Proxy leakage",
        "Explanations",
        "Mitigation lab",
        "Which features matter",
        "Limitations",
    ]

    for p in pages:
        at.sidebar.radio[0].set_value(p).run(timeout=30)
        assert not at.exception, f"Page '{p}' raised exception: {at.exception}"
