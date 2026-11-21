"""Page: Check an applicant - Interactive evaluation across information levels.

Phase 8 Stage 6: Profile inspection with information level segmented control and gauges.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

from src.app.scoring import (
    load_scoring_resources,
    score_applicants,
    explain_applicant,
)
from src.app.profile_scoring import score_with_information_level, INFORMATION_LEVEL_MAP
from src.app.visuals import risk_gauge, decision_badge, score_histogram
from src.app.pages.new_applicants import build_contributions_chart


def render_page_check_applicant():
    """Main renderer for Check an Applicant page."""
    st.title("🔍 Check an Applicant: Information Level Thought Experiment")
    st.markdown(
        """
        Inspect how different data sources (traditional credit bureau records vs internal lender repayment history) 
        alter Model A and Model B risk evaluations for a held-out historical test applicant.
        """
    )
    st.caption(
        "Notice: Every applicant shown is an anonymized historical test borrower from data/splits/test_ids.csv. "
        "Scores represent model outputs, not predictions about real people and not financial advice."
    )

    resources = load_scoring_resources()
    meta_test = resources["meta_test"]
    thresh_a = resources["threshold_a"]
    thresh_b = resources["threshold_b"]
    test_ids_set = resources["test_ids_set"]

    if "check_applicant_id" not in st.session_state:
        st.session_state["check_applicant_id"] = 333107

    col_btn, col_inp = st.columns([1, 2])
    with col_btn:
        st.write("")
        st.write("")
        if st.button("Draw random test applicant", key="btn_draw_check"):
            random_id = int(np.random.choice(list(test_ids_set)))
            st.session_state["check_applicant_id"] = random_id
            st.rerun()

    with col_inp:
        typed_val = st.text_input(
            "Enter held-out test applicant SK_ID_CURR",
            value=str(st.session_state["check_applicant_id"]),
            help="Must be a valid test split ID from data/splits/test_ids.csv",
        )
        if typed_val:
            try:
                val_int = int(typed_val.strip())
                if val_int in test_ids_set:
                    if val_int != st.session_state["check_applicant_id"]:
                        st.session_state["check_applicant_id"] = val_int
                        st.rerun()
                else:
                    st.error(f"ID {val_int} is not in the held-out test split. Only test split IDs can be evaluated.")
            except ValueError:
                st.error("Please enter a valid numeric SK_ID_CURR.")

    curr_id = st.session_state["check_applicant_id"]
    meta_row = meta_test.loc[curr_id]
    segment = str(meta_row["segment"])

    st.markdown(f"### Profile: Applicant `{curr_id}`")
    st.markdown(f"**Credit Segment**: `{segment}` (anonymized historical record)")

    # Segmented control for Information used
    st.markdown("#### Information Used")
    info_options = ["Full information", "Bureau only", "Lender history only", "No history"]
    info_level = st.radio(
        "Select information availability",
        options=info_options,
        index=0,
        horizontal=True,
        help="Simulates data availability constraints on the same borrower profile.",
    )

    # Score profile with selected information level
    res = score_with_information_level(curr_id, information_level=info_level)
    sa = float(res.score_a[0])
    sb = float(res.score_b[0])

    dec_a = "declined" if sa >= thresh_a else "approved"
    dec_b = "declined" if sb >= thresh_b else "approved"

    # Side-by-side gauges and badges
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### Model A (Bureau-only)")
        decision_badge(dec_a)
        fig_a = risk_gauge(sa, thresh_a, "Model A")
        st.pyplot(fig_a, use_container_width=True)
        st.altair_chart(score_histogram(sa, thresh_a, "Model A"), use_container_width=True)

    with g2:
        st.markdown("#### Model B (Bureau + Behavioral)")
        decision_badge(dec_b)
        fig_b = risk_gauge(sb, thresh_b, "Model B")
        st.pyplot(fig_b, use_container_width=True)
        st.altair_chart(score_histogram(sb, thresh_b, "Model B"), use_container_width=True)

    st.caption(
        "A thought experiment on the same person: it shows what each information source contributes to the score. "
        "It reproduces the full_file, thin_file and no_file situations. It is not a prediction about anyone and not advice."
    )
    st.caption(
        f"Validation policy 15% thresholds: Model A = {thresh_a:.4f}, Model B = {thresh_b:.4f}. "
        "Source: reports/metrics/fairness.json and live feature masking."
    )

    # Top contributions chart
    st.subheader("Top Model B Feature Contributions (Full Profile Live TreeSHAP)")
    try:
        exp = explain_applicant(curr_id)
        st.altair_chart(build_contributions_chart(exp), use_container_width=True)
        st.caption(
            "Feature contributions on margin log-odds scale from live TreeSHAP attribution. "
            "Source: reports/predictions/model_b_test.parquet and src/explainability/shap_analysis.py."
        )
    except Exception as e:
        st.caption(f"Explanation unavailable: {e}")
