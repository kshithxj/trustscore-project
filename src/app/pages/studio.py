"""Landing Page: TrustScore Studio.

Phase 8 Stage 6: Demo-first landing page with study overview, entry cards, and live metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

from src.app.helpers import (
    load_model_a_json,
    load_model_b_json,
    load_base_rates_csv,
    load_fairness_json,
)


def render_page_studio():
    """Renders the TrustScore Studio landing page."""
    # Banner
    st.markdown(
        """
        <div style="background-color: #f0f2f6; border-left: 5px solid #1f77b4; padding: 10px 15px; border-radius: 4px; margin-bottom: 20px;">
            <span style="font-weight: 600; color: #1e293b;">
                Academic proof-of-concept on the Home Credit dataset. Not a lending system.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title("🎯 TrustScore Studio")

    # One sentence research question (no numbers)
    st.markdown(
        """
        #### Can a lender's internal behavioural repayment records improve default risk prediction for thin-file borrowers without amplifying demographic disparities?
        """
    )
    st.markdown(
        "*Explore how alternative behavioral credit features influence underwriting decisions on held-out historical applicants.*"
    )

    st.markdown("---")

    # Four big navigation cards/buttons
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; min-height: 140px;">
                <h4 style="margin-top:0;">🔍 Check an applicant</h4>
                <p style="font-size: 0.9rem; color: #475569;">Inspect individual test applicants under 4 distinct information availability levels.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Check an Applicant", key="btn_nav_check", use_container_width=True):
            st.session_state["nav_target"] = "Check an applicant"
            st.rerun()

    with c2:
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; min-height: 140px;">
                <h4 style="margin-top:0;">📁 Case files</h4>
                <p style="font-size: 0.9rem; color: #475569;">Rule-selected canonical test cases: Rescued, Caught, and Both-wrong false declines.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Case Files", key="btn_nav_cases", use_container_width=True):
            st.session_state["nav_target"] = "Case files"
            st.rerun()

    with c3:
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; min-height: 140px;">
                <h4 style="margin-top:0;">👤 New applicants</h4>
                <p style="font-size: 0.9rem; color: #475569;">Live scoring and batch draws of anonymized historical applicants the models never saw.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open New Applicants", key="btn_nav_new", use_container_width=True):
            st.session_state["nav_target"] = "New applicants"
            st.rerun()

    with c4:
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; min-height: 140px;">
                <h4 style="margin-top:0;">🔬 Research results</h4>
                <p style="font-size: 0.9rem; color: #475569;">Detailed research findings: lift stability, fairness audits, proxy tests, and mitigations.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Research Results", key="btn_nav_research", use_container_width=True):
            st.session_state["nav_target"] = "Guided tour"
            st.rerun()

    st.markdown("---")

    # Live metrics row read from files
    ma = load_model_a_json()
    mb = load_model_b_json()
    base_df = load_base_rates_csv()
    fj = load_fairness_json()

    n_test = int(fj["overall"]["gender"]["model_a"]["group_metrics"]["F"]["n"] + fj["overall"]["gender"]["model_a"]["group_metrics"]["M"]["n"] + fj["policy"].get("test_xna_count", 1))
    n_segments = int(base_df[base_df["dimension"] == "segment"]["group"].nunique())
    auc_a = float(ma["test_results"]["overall"]["auc"])
    auc_b = float(mb["test_results"]["overall"]["auc"])

    st.subheader("Study Scope & Baseline Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Test Applicants", f"{n_test:,}", help="Held-out historical test split applicants")
    with m2:
        st.metric("Credit Segments", f"{n_segments}", help="Full-file, Thin-file, and No-file")
    with m3:
        st.metric("Model A Test AUC", f"{auc_a:.4f}", help="Bureau-only baseline model")
    with m4:
        st.metric("Model B Test AUC", f"{auc_b:.4f}", delta=f"{auc_b - auc_a:+.4f}", help="Behavioral + bureau model")

    st.caption(
        "Metrics source: reports/metrics/model_a.json, reports/metrics/model_b.json, reports/tables/base_rates.csv, and reports/metrics/fairness.json."
    )

    st.markdown(
        """
        > **Honesty Notice**: Demo of a research model on one lender's historical data.
        > Model scores represent statistical default probabilities from historical data, not advice or predictions about real people.
        """
    )
