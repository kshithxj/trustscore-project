"""Tests for Phase 8 Stage 4: New applicants page and live scoring core."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.app.scoring import (
    load_scoring_resources,
    score_applicants,
    explain_applicant,
    run_round_trip_check,
)
from src.features.behavioral import ALL_BEHAVIORAL_COLS
from src.features.bureau import BUREAU_FEATURE_COLS


def test_round_trip_predictions_300():
    """(a) Round trip: scores equal saved predictions within 1e-6 for 300 random test applicants, both models."""
    diff_a, diff_b = run_round_trip_check(n_samples=300, seed=42)
    assert diff_a < 1e-6, f"Model A round trip failed: max diff {diff_a:.2e} >= 1e-6"
    assert diff_b < 1e-6, f"Model B round trip failed: max diff {diff_b:.2e} >= 1e-6"


def test_only_test_split_ids_accepted_and_train_rejected():
    """(b) Only test-split ids can be selected, and an id from the train split is rejected."""
    train_ids_path = _PROJECT_ROOT / "data" / "splits" / "train_ids.csv"
    train_df = pd.read_csv(train_ids_path, nrows=5)
    train_id = int(train_df.iloc[0]["SK_ID_CURR"])

    # Attempting to score a train ID must raise ValueError
    with pytest.raises(ValueError, match="not in the held-out test split"):
        score_applicants(train_id)

    # Attempting to explain a train ID must raise ValueError
    with pytest.raises(ValueError, match="not in the held-out test split"):
        explain_applicant(train_id)

    # Valid test ID must succeed
    test_ids_path = _PROJECT_ROOT / "data" / "splits" / "test_ids.csv"
    test_df = pd.read_csv(test_ids_path, nrows=1)
    test_id = int(test_df.iloc[0]["SK_ID_CURR"])

    res = score_applicants(test_id)
    assert len(res.score_a) == 1
    assert len(res.score_b) == 1


def test_ablation_switches_all_nans_and_score_changes_for_thin_file_with_lender():
    """(c) With both switches on, all bureau_* and behavioral columns are NaN and scores change for thin_file with lender history."""
    resources = load_scoring_resources()
    meta = resources["meta_test"]
    base_df = pd.read_parquet(_PROJECT_ROOT / "data" / "processed" / "applicant_base.parquet").set_index("SK_ID_CURR")

    thin_ids = meta[meta["segment"] == "thin_file"].index
    thin_with_lender = base_df.loc[thin_ids]
    thin_with_lender_ids = thin_with_lender[thin_with_lender["has_lender_history"] == True].index.values

    assert len(thin_with_lender_ids) > 0, "Expected thin_file applicants with lender history"
    sample_id = int(thin_with_lender_ids[0])

    orig = score_applicants(sample_id)
    abl = score_applicants(sample_id, overrides={"hide_bureau": True, "hide_behavioral": True})

    # Verify that all bureau columns in X_a and X_b are NaN
    bureau_cols_a = [c for c in abl.X_a.columns if c.startswith("bureau_") or c in BUREAU_FEATURE_COLS]
    bureau_cols_b = [c for c in abl.X_b.columns if c.startswith("bureau_") or c in BUREAU_FEATURE_COLS]
    assert len(bureau_cols_a) > 0
    assert len(bureau_cols_b) > 0
    assert abl.X_a[bureau_cols_a].isna().all().all(), "All bureau columns in ablated X_a must be NaN"
    assert abl.X_b[bureau_cols_b].isna().all().all(), "All bureau columns in ablated X_b must be NaN"

    # Verify that all behavioral columns in X_b are NaN
    beh_set = set(ALL_BEHAVIORAL_COLS)
    beh_cols_b = [c for c in abl.X_b.columns if c in beh_set]
    assert len(beh_cols_b) == 20
    assert abl.X_b[beh_cols_b].isna().all().all(), "All behavioral columns in ablated X_b must be NaN"

    # Verify that score changes
    diff_b = abs(float(orig.score_b[0]) - float(abl.score_b[0]))
    assert diff_b > 1e-4, f"Model B score should change under behavioral ablation, got diff {diff_b}"


def test_thresholds_come_from_fairness_json():
    """(d) Thresholds are strictly loaded from reports/metrics/fairness.json."""
    fairness_path = _PROJECT_ROOT / "reports" / "metrics" / "fairness.json"
    with open(fairness_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_thresh_a = float(data["policy"]["threshold_a"])
    expected_thresh_b = float(data["policy"]["threshold_b"])

    resources = load_scoring_resources()
    assert resources["threshold_a"] == expected_thresh_a
    assert resources["threshold_b"] == expected_thresh_b


def test_local_accuracy_20_random_applicants():
    """(e) The TreeSHAP contributions satisfy the local-accuracy check for 20 random test applicants."""
    resources = load_scoring_resources()
    test_ids = list(resources["test_ids_set"])
    rng = np.random.default_rng(123)
    sample_ids = rng.choice(test_ids, size=20, replace=False).tolist()

    max_diff = 0.0
    for sid in sample_ids:
        exp = explain_applicant(sid)
        bias = float(exp["bias"])
        contribs = exp["contributions"]
        margin = bias + float(np.sum(contribs.values))
        prob = 1.0 / (1.0 + np.exp(-margin))
        score = float(exp["score"])
        diff = abs(prob - score)
        max_diff = max(max_diff, diff)
        assert diff < 1e-5, f"Local accuracy failed for applicant {sid}: diff {diff:.2e} >= 1e-5"

    assert max_diff < 1e-5


def test_no_gender_or_age_field_on_page():
    """(f) No gender or age field appears on the page (check widget labels)."""
    import streamlit.testing.v1 as st_test

    app_path = _PROJECT_ROOT / "src" / "app" / "streamlit_app.py"
    at = st_test.AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    # Navigate to 'New applicants'
    at.sidebar.radio[0].set_value("New applicants").run()
    assert len(at.exception) == 0

    # Collect all widget labels across page
    widget_labels: List[str] = []
    for elem in at.selectbox:
        if elem.label:
            widget_labels.append(elem.label)
    for elem in at.slider:
        if elem.label:
            widget_labels.append(elem.label)
    for elem in at.text_input:
        if elem.label:
            widget_labels.append(elem.label)
    for elem in at.button:
        if elem.label:
            widget_labels.append(elem.label)
    for elem in at.checkbox:
        if elem.label:
            widget_labels.append(elem.label)

    for label in widget_labels:
        label_lower = label.lower()
        assert "gender" not in label_lower, f"Forbidden 'gender' found in widget label: '{label}'"
        assert "age" not in label_lower, f"Forbidden 'age' found in widget label: '{label}'"


def test_new_applicants_page_renders_with_apptest():
    """Verify that both tabs and Guided Tour step 6 render cleanly with AppTest."""
    import streamlit.testing.v1 as st_test

    app_path = _PROJECT_ROOT / "src" / "app" / "streamlit_app.py"
    at = st_test.AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    # Verify 'New applicants' page renders
    at.sidebar.radio[0].set_value("New applicants").run()
    assert len(at.exception) == 0, f"Exceptions on New applicants page: {at.exception}"
    assert len(at.tabs) == 2
    assert at.tabs[0].label == "One applicant"
    assert at.tabs[1].label == "A batch of new applicants"

    # Verify Guided Tour step 6 renders
    at.sidebar.radio[0].set_value("Guided tour").run()
    at.select_slider[0].set_value("6. Applicant Explanation").run()
    assert len(at.exception) == 0, f"Exceptions on Guided tour step 6: {at.exception}"
