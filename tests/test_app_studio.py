"""Tests for Phase 8 Stage 6: TrustScore Studio, Case files, Visuals, and App navigation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.app.scoring import load_scoring_resources, score_applicants
from src.app.visuals import get_gauge_max_score, decision_badge, risk_gauge, score_histogram
from src.app.profile_scoring import score_with_information_level, INFORMATION_LEVEL_MAP
from src.app.pages.case_files import get_case_files_data


def test_studio_module_imports():
    """Verify that all Phase 8 Stage 6 modules import without error."""
    import src.app.visuals as visuals
    import src.app.profile_scoring as profile_scoring
    import src.app.pages.studio as studio
    import src.app.pages.case_files as case_files
    import src.app.pages.check_applicant as check_applicant
    import src.app.pages.new_applicants as new_applicants
    import src.app.streamlit_app as streamlit_app

    assert visuals is not None
    assert profile_scoring is not None
    assert studio is not None
    assert case_files is not None
    assert check_applicant is not None
    assert new_applicants is not None
    assert streamlit_app is not None


def test_case_files_deterministic_and_belong_to_test_split():
    """Verify that the three case applicants are deterministic across two runs and belong to the TEST split."""
    run1 = get_case_files_data()["cases"]
    run2 = get_case_files_data()["cases"]

    resources = load_scoring_resources()
    test_ids_set = resources["test_ids_set"]

    for category in ["Rescued", "Caught", "Both wrong"]:
        assert run1[category] is not None, f"Category '{category}' should not be empty"
        assert run2[category] is not None, f"Category '{category}' should not be empty"
        assert run1[category]["SK_ID_CURR"] == run2[category]["SK_ID_CURR"], f"Non-deterministic selection in {category}"

        sid = run1[category]["SK_ID_CURR"]
        assert sid in test_ids_set, f"Applicant {sid} in '{category}' is not in the held-out test split!"


def test_case_files_category_conditions():
    """Verify category conditions hold: decisions by each model at 15% thresholds and actual outcomes."""
    data = get_case_files_data()
    cases = data["cases"]
    ta = data["threshold_a"]
    tb = data["threshold_b"]

    # 1. Rescued: thin_file, declined A, approved B, target == 0
    rescued = cases["Rescued"]
    assert rescued["segment"] == "thin_file"
    assert rescued["score_a"] >= ta, "Rescued applicant must be declined by Model A"
    assert rescued["score_b"] < tb, "Rescued applicant must be approved by Model B"
    assert rescued["TARGET"] == 0, "Rescued applicant must have repaid (TARGET == 0)"

    # 2. Caught: thin_file, approved A, declined B, target == 1
    caught = cases["Caught"]
    assert caught["segment"] == "thin_file"
    assert caught["score_a"] < ta, "Caught applicant must be approved by Model A"
    assert caught["score_b"] >= tb, "Caught applicant must be declined by Model B"
    assert caught["TARGET"] == 1, "Caught applicant must have defaulted (TARGET == 1)"

    # 3. Both wrong: declined A, declined B, target == 0
    both_wrong = cases["Both wrong"]
    assert both_wrong["score_a"] >= ta, "Both wrong applicant must be declined by Model A"
    assert both_wrong["score_b"] >= tb, "Both wrong applicant must be declined by Model B"
    assert both_wrong["TARGET"] == 0, "Both wrong applicant must have repaid (TARGET == 0, false decline)"


def test_case_files_scores_equal_saved_predictions_within_1e6():
    """Verify live scores of case applicants match saved test predictions within 1e-6."""
    data = get_case_files_data()
    cases = data["cases"]

    pred_a = pd.read_parquet(_PROJECT_ROOT / "reports" / "predictions" / "model_a_test.parquet").set_index("SK_ID_CURR")
    pred_b = pd.read_parquet(_PROJECT_ROOT / "reports" / "predictions" / "model_b_test.parquet").set_index("SK_ID_CURR")

    for cat, info in cases.items():
        sid = info["SK_ID_CURR"]
        live_res = score_applicants(sid)
        expected_a = float(pred_a.loc[sid, "score"])
        expected_b = float(pred_b.loc[sid, "score"])

        assert abs(live_res.score_a[0] - expected_a) < 1e-6, f"Model A score mismatch for {cat} ({sid})"
        assert abs(live_res.score_b[0] - expected_b) < 1e-6, f"Model B score mismatch for {cat} ({sid})"


def test_thresholds_come_from_fairness_json():
    """Verify thresholds match reports/metrics/fairness.json exactly."""
    fairness_path = _PROJECT_ROOT / "reports" / "metrics" / "fairness.json"
    with open(fairness_path, "r", encoding="utf-8") as f:
        fj = json.load(f)

    expected_ta = float(fj["policy"]["threshold_a"])
    expected_tb = float(fj["policy"]["threshold_b"])

    data = get_case_files_data()
    assert data["threshold_a"] == expected_ta
    assert data["threshold_b"] == expected_tb


def test_gauge_maximum_computed_from_test_scores():
    """Verify risk gauge maximum is computed as the 99th percentile of test scores."""
    pa = pd.read_parquet(_PROJECT_ROOT / "reports" / "predictions" / "model_a_test.parquet")["score"]
    pb = pd.read_parquet(_PROJECT_ROOT / "reports" / "predictions" / "model_b_test.parquet")["score"]
    expected_max = round(float(max(np.percentile(pa, 99), np.percentile(pb, 99))), 4)

    assert get_gauge_max_score() == expected_max


def test_no_gender_or_age_in_widget_labels():
    """Verify that neither gender nor age appears as a widget label anywhere in the app."""
    import streamlit.testing.v1 as st_test

    app_path = _PROJECT_ROOT / "src" / "app" / "streamlit_app.py"
    at = st_test.AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    # Collect widget labels on default page
    all_widgets = list(at.selectbox) + list(at.slider) + list(at.text_input) + list(at.button) + list(at.checkbox) + list(at.radio)
    for elem in all_widgets:
        if elem.label:
            lbl = elem.label.lower()
            assert "gender" not in lbl, f"Forbidden 'gender' in widget label: {elem.label}"
            assert "age" not in lbl, f"Forbidden 'age' in widget label: {elem.label}"


def test_every_page_renders_with_apptest():
    """Verify that every page in Demo and Research sections renders cleanly without exception."""
    import streamlit.testing.v1 as st_test
    from src.app.streamlit_app import PAGE_NAMES

    app_path = _PROJECT_ROOT / "src" / "app" / "streamlit_app.py"
    at = st_test.AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"Initial run exception: {at.exception}"

    # Verify each page renders without uncaught exceptions
    for p in PAGE_NAMES:
        at.sidebar.radio[0].set_value(p).run(timeout=30)
        assert len(at.exception) == 0, f"Page '{p}' raised exception: {at.exception}"
