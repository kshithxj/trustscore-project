"""Page: Case files - Canonical test cases chosen by deterministic rule.

Phase 8 Stage 6: Three rule-selected case profiles from held-out test split.
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
from src.app.visuals import risk_gauge, decision_badge
from src.app.pages.new_applicants import compute_verdict, build_contributions_chart


@st.cache_data
def get_case_files_data() -> Dict[str, Any]:
    """Deterministically identifies three case applicants from held-out test split by exact rule.
    
    Rules:
    (1) Rescued: thin_file, declined by A, approved by B (at 15% validation thresholds), TARGET == 0.
        Selects applicant closest to the median Model B score of the eligible set.
    (2) Caught: thin_file, approved by A, declined by B, TARGET == 1.
        Selects applicant closest to the median Model B score of the eligible set.
    (3) Both wrong: declined by both models, TARGET == 0 (false decline).
        Selects applicant closest to the median Model B score of the eligible set.
    """
    fairness_path = _PROJECT_ROOT / "reports" / "metrics" / "fairness.json"
    with open(fairness_path, "r", encoding="utf-8") as f:
        fj = json.load(f)
    thresh_a = float(fj["policy"]["threshold_a"])
    thresh_b = float(fj["policy"]["threshold_b"])

    pred_dir = _PROJECT_ROOT / "reports" / "predictions"
    df_a = pd.read_parquet(pred_dir / "model_a_test.parquet")
    df_b = pd.read_parquet(pred_dir / "model_b_test.parquet")

    df = df_a[["SK_ID_CURR", "segment", "TARGET"]].copy()
    df["score_a"] = df_a["score"].values
    df["score_b"] = df_b["score"].values

    cases: Dict[str, Any] = {}

    # 1. Rescued
    c1_mask = (df["segment"] == "thin_file") & (df["score_a"] >= thresh_a) & (df["score_b"] < thresh_b) & (df["TARGET"] == 0)
    c1_df = df[c1_mask].copy()
    if len(c1_df) > 0:
        c1_med = float(c1_df["score_b"].median())
        c1_df["dist"] = (c1_df["score_b"] - c1_med).abs()
        c1_chosen = c1_df.sort_values(["dist", "SK_ID_CURR"]).iloc[0]
        cases["Rescued"] = {
            "SK_ID_CURR": int(c1_chosen["SK_ID_CURR"]),
            "segment": str(c1_chosen["segment"]),
            "score_a": float(c1_chosen["score_a"]),
            "score_b": float(c1_chosen["score_b"]),
            "decision_a": "declined",
            "decision_b": "approved",
            "TARGET": int(c1_chosen["TARGET"]),
            "eligible_count": len(c1_df),
            "median_score_b": c1_med,
            "rule": "Segment thin-file, declined by Model A, approved by Model B at 15% policy, did not default (TARGET == 0). Closest to median Model B score.",
        }
    else:
        cases["Rescued"] = None

    # 2. Caught
    c2_mask = (df["segment"] == "thin_file") & (df["score_a"] < thresh_a) & (df["score_b"] >= thresh_b) & (df["TARGET"] == 1)
    c2_df = df[c2_mask].copy()
    if len(c2_df) > 0:
        c2_med = float(c2_df["score_b"].median())
        c2_df["dist"] = (c2_df["score_b"] - c2_med).abs()
        c2_chosen = c2_df.sort_values(["dist", "SK_ID_CURR"]).iloc[0]
        cases["Caught"] = {
            "SK_ID_CURR": int(c2_chosen["SK_ID_CURR"]),
            "segment": str(c2_chosen["segment"]),
            "score_a": float(c2_chosen["score_a"]),
            "score_b": float(c2_chosen["score_b"]),
            "decision_a": "approved",
            "decision_b": "declined",
            "TARGET": int(c2_chosen["TARGET"]),
            "eligible_count": len(c2_df),
            "median_score_b": c2_med,
            "rule": "Segment thin-file, approved by Model A, declined by Model B at 15% policy, defaulted (TARGET == 1). Closest to median Model B score.",
        }
    else:
        cases["Caught"] = None

    # 3. Both wrong (false decline)
    c3_mask = (df["score_a"] >= thresh_a) & (df["score_b"] >= thresh_b) & (df["TARGET"] == 0)
    c3_df = df[c3_mask].copy()
    if len(c3_df) > 0:
        c3_med = float(c3_df["score_b"].median())
        c3_df["dist"] = (c3_df["score_b"] - c3_med).abs()
        c3_chosen = c3_df.sort_values(["dist", "SK_ID_CURR"]).iloc[0]
        cases["Both wrong"] = {
            "SK_ID_CURR": int(c3_chosen["SK_ID_CURR"]),
            "segment": str(c3_chosen["segment"]),
            "score_a": float(c3_chosen["score_a"]),
            "score_b": float(c3_chosen["score_b"]),
            "decision_a": "declined",
            "decision_b": "declined",
            "TARGET": int(c3_chosen["TARGET"]),
            "eligible_count": len(c3_df),
            "median_score_b": c3_med,
            "rule": "Declined by both models at 15% policy, did not default (TARGET == 0, false decline). Closest to median Model B score.",
        }
    else:
        cases["Both wrong"] = None

    return {
        "cases": cases,
        "threshold_a": thresh_a,
        "threshold_b": thresh_b,
    }


def render_case_card(category_name: str, case_data: Dict[str, Any], thresh_a: float, thresh_b: float):
    """Renders a single case profile card with gauges, pre-reveal contributions, and outcome reveal."""
    if case_data is None:
        st.warning(f"No eligible test applicants found for category: '{category_name}'.")
        return

    sid = case_data["SK_ID_CURR"]
    seg = case_data["segment"]
    sa = case_data["score_a"]
    sb = case_data["score_b"]
    target = case_data["TARGET"]
    count_elig = case_data["eligible_count"]

    dec_a = "declined" if sa >= thresh_a else "approved"
    dec_b = "declined" if sb >= thresh_b else "approved"

    st.markdown(f"### Case {category_name}: Applicant `{sid}`")
    st.markdown(f"**Credit Segment**: `{seg}` | **Eligible Group Size**: `{count_elig:,}` applicants")
    st.caption(f"Rule: {case_data['rule']}")

    # Gauges side by side
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("**Model A (Bureau-only)**")
        decision_badge(dec_a)
        fig_a = risk_gauge(sa, thresh_a, "Model A")
        st.pyplot(fig_a, use_container_width=True)
    with col_g2:
        st.markdown("**Model B (Bureau + Behavioral)**")
        decision_badge(dec_b)
        fig_b = risk_gauge(sb, thresh_b, "Model B")
        st.pyplot(fig_b, use_container_width=True)

    # Top 8 Model B TreeSHAP contributions
    st.markdown("#### Top 8 Model B Risk Contributions (Log-Odds Scale)")
    try:
        exp = explain_applicant(sid)
        # Select top 8 contributions
        top_8_df = exp.top_10.head(8).copy()
        exp_8 = exp
        st.altair_chart(build_contributions_chart(exp_8), use_container_width=True)
        st.caption(
            "Feature contributions on margin log-odds scale from live TreeSHAP. "
            "Source: reports/predictions/model_b_test.parquet and src/explainability/shap_analysis.py."
        )
    except Exception as e:
        st.caption(f"TreeSHAP explanation unavailable: {e}")

    # Reveal button
    rev_key = f"reveal_case_{category_name}_{sid}"
    if rev_key not in st.session_state:
        st.session_state[rev_key] = False

    if not st.session_state[rev_key]:
        if st.button(f"Reveal what actually happened for {category_name}", key=f"btn_{rev_key}"):
            st.session_state[rev_key] = True
            st.rerun()

    if st.session_state[rev_key]:
        actual_str = "Defaulted" if target == 1 else "Repaid (Non-default)"
        verdict_a = compute_verdict(dec_a, target)
        verdict_b = compute_verdict(dec_b, target)

        st.markdown("---")
        st.markdown("#### Actual Historical Outcome & Model Verdicts")
        o1, o2, o3 = st.columns(3)
        with o1:
            st.metric("Actual Outcome", actual_str)
        with o2:
            st.metric("Model A Verdict", verdict_a)
        with o3:
            st.metric("Model B Verdict", verdict_b)

    st.markdown("---")


def render_page_case_files():
    """Main renderer for Case Files page."""
    st.title("📁 Case Files: Canonical Real-World Test Profiles")
    st.markdown(
        """
        Three anonymized held-out test applicants chosen **strictly by mathematical rule** from `data/splits/test_ids.csv`.
        These cases demonstrate where alternative behavioral history flips underwriting decisions or where models make shared errors.
        """
    )
    st.caption(
        "Notice: These cases are illustrative examples chosen by an explicit median-distance rule, not typical applicants. "
        "Every applicant is an anonymized historical test borrower whom the models never saw during training. "
        "Source: reports/metrics/fairness.json and reports/predictions/model_{a,b}_test.parquet."
    )

    data = get_case_files_data()
    cases = data["cases"]
    ta = data["threshold_a"]
    tb = data["threshold_b"]

    tab_r, tab_c, tab_w = st.tabs(["(1) Rescued", "(2) Caught", "(3) Both wrong"])

    with tab_r:
        render_case_card("Rescued", cases["Rescued"], ta, tb)

    with tab_c:
        render_case_card("Caught", cases["Caught"], ta, tb)

    with tab_w:
        render_case_card("Both wrong", cases["Both wrong"], ta, tb)
