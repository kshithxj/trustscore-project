"""Page: New applicants - Live scoring of held-out TEST applicants.

Phase 8 Stage 4: Live scoring of historical held-out test applicants.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# Set writable MPLCONFIGDIR for headless Windows execution
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.app.scoring import (
    load_scoring_resources,
    score_applicants,
    explain_applicant,
    ScoringResult,
    ExplanationResult,
)
from src.app.helpers import load_flips_json, load_fairness_json

PAGE_DISCLAIMER = (
    "**Notice**: Every profile on this page is an anonymized historical held-out test applicant "
    "from data/splits/test_ids.csv whom the models never saw during training or validation. "
    "Each model score describes statistical patterns in one lender's historical dataset. "
    "These model scores are not advice, are not predictions about any real person, and do not convey how to obtain credit."
)


def compute_verdict(decision: str, target: int) -> str:
    """Computes standard risk verdict based on decision ('declined'/'approved') and target (1=default, 0=repaid)."""
    is_declined = (decision == "declined")
    if is_declined and target == 1:
        return "correct decline"
    elif is_declined and target == 0:
        return "false decline"
    elif not is_declined and target == 0:
        return "correct approval"
    else:
        return "missed default"


def build_contributions_chart(exp: ExplanationResult) -> alt.Chart:
    """Builds an Altair horizontal bar chart of top 10 Model B contributions on margin log-odds scale."""
    top_10 = exp.top_10.copy()
    top_10["direction"] = np.where(top_10["shap_value"] >= 0, "Increases risk (+)", "Decreases risk (-)")
    top_10["abs_shap_round"] = top_10["abs_shap"].round(4)
    top_10["shap_value_round"] = top_10["shap_value"].round(4)

    # Format feature value for display
    def _fmt_val(v):
        if pd.isna(v):
            return "Missing"
        if isinstance(v, (float, np.floating)):
            return f"{v:,.2f}"
        return str(v)

    top_10["display_value"] = top_10["feature_value"].apply(_fmt_val)
    top_10["feature_value"] = top_10["feature_value"].astype(str)

    chart = (
        alt.Chart(top_10)
        .mark_bar()
        .encode(
            x=alt.X("shap_value:Q", title="Contribution to log-odds margin (Model B score)"),
            y=alt.Y("feature:N", sort=alt.EncodingSortField(field="abs_shap", order="descending"), title="Feature"),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Increases risk (+)", "Decreases risk (-)"], range=["#d95f02", "#1b9e77"]),
                legend=alt.Legend(title="Direction"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("display_value:N", title="Raw Applicant Value"),
                alt.Tooltip("shap_value_round:Q", title="Margin Contribution (log-odds)"),
            ],
        )
        .properties(height=320)
    )
    return chart


def render_single_applicant_card(applicant_id: int, is_guided_tour: bool = False):
    """Renders the single held-out test applicant card."""
    resources = load_scoring_resources()
    test_ids_set = resources["test_ids_set"]

    if applicant_id not in test_ids_set:
        st.error(
            f"Applicant ID {applicant_id} is not in data/splits/test_ids.csv. "
            "Only held-out test applicants whom the models never saw can appear on this page."
        )
        return

    meta_row = resources["meta_test"].loc[applicant_id]
    segment = str(meta_row["segment"])
    actual_target = int(meta_row["TARGET"])
    actual_outcome_str = "defaulted" if actual_target == 1 else "repaid"

    threshold_a = resources["threshold_a"]
    threshold_b = resources["threshold_b"]

    # Original live score
    base_res = score_applicants(applicant_id)
    score_a = float(base_res.score_a[0])
    score_b = float(base_res.score_b[0])

    decision_a = "declined" if score_a >= threshold_a else "approved"
    decision_b = "declined" if score_b >= threshold_b else "approved"

    st.markdown(f"### Held-Out Test Applicant: `{applicant_id}`")
    st.markdown(f"**Segment**: `{segment}` (anonymized historical record)")

    # 1. Scores & Decisions
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Model A Score", f"{score_a:.4f}", help="Default probability from Model A")
    with c2:
        st.metric("Model A Decision", decision_a, help=f"15% policy threshold: {threshold_a:.4f}")
    with c3:
        st.metric("Model B Score", f"{score_b:.4f}", help="Default probability from Model B")
    with c4:
        st.metric("Model B Decision", decision_b, help=f"15% policy threshold: {threshold_b:.4f}")

    st.caption(
        f"Model scores are default probabilities computed live. "
        f"Validation 15% policy thresholds: Model A = {threshold_a:.4f}, Model B = {threshold_b:.4f}. "
        f"Source: reports/metrics/fairness.json."
    )

    # 2. Top 10 Model B contributions
    st.subheader("Top 10 Model B Contributions (Live TreeSHAP)")
    try:
        exp = explain_applicant(applicant_id)
        st.altair_chart(build_contributions_chart(exp), use_container_width=True)
        st.caption(
            "Top 10 Model B contributions on margin (log-odds scale). "
            f"Bias baseline = {exp.bias:+.4f}. "
            "Source: live TreeSHAP attribution matching src/explainability/shap_analysis.py."
        )
    except Exception as e:
        st.warning(f"Live explanation not available: {e}")

    # 3. Reveal Actual Outcome
    st.subheader("Actual Historical Outcome")
    reveal_key = f"reveal_{applicant_id}"
    if is_guided_tour:
        revealed = True
    else:
        if reveal_key not in st.session_state:
            st.session_state[reveal_key] = False
        if not st.session_state[reveal_key]:
            if st.button("Reveal what actually happened", key=f"btn_rev_{applicant_id}"):
                st.session_state[reveal_key] = True
                st.rerun()
        revealed = st.session_state[reveal_key]

    if revealed:
        verdict_a = compute_verdict(decision_a, actual_target)
        verdict_b = compute_verdict(decision_b, actual_target)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Actual Outcome", actual_outcome_str)
        with r2:
            st.metric("Model A Verdict", verdict_a)
        with r3:
            st.metric("Model B Verdict", verdict_b)

        st.caption("Actual historical outcome from data/splits/test_ids.csv. Source: Home Credit historical record.")

    # 4. Thought Experiment: Information Source Ablation Switches
    st.subheader("Information Source Thought Experiment")
    st.markdown("Toggle switches to observe the impact of removing specific data sources:")

    col_sw1, col_sw2 = st.columns(2)
    with col_sw1:
        hide_bureau = st.checkbox("Hide bureau record", value=False, key=f"sw_bur_{applicant_id}")
    with col_sw2:
        hide_lender = st.checkbox("Hide lender history", value=False, key=f"sw_len_{applicant_id}")

    if hide_bureau or hide_lender:
        overrides = {"hide_bureau": hide_bureau, "hide_behavioral": hide_lender}
        abl_res = score_applicants(applicant_id, overrides=overrides)
        abl_score_a = float(abl_res.score_a[0])
        abl_score_b = float(abl_res.score_b[0])
        abl_dec_a = "declined" if abl_score_a >= threshold_a else "approved"
        abl_dec_b = "declined" if abl_score_b >= threshold_b else "approved"

        diff_a = abl_score_a - score_a
        diff_b = abl_score_b - score_b

        st.markdown("**Rescored Under Information Ablation:**")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.metric("Ablated Model A Score", f"{abl_score_a:.4f}", delta=f"{diff_a:+.4f}")
        with a2:
            st.metric("Ablated Model A Decision", abl_dec_a)
        with a3:
            st.metric("Ablated Model B Score", f"{abl_score_b:.4f}", delta=f"{diff_b:+.4f}")
        with a4:
            st.metric("Ablated Model B Decision", abl_dec_b)

    st.caption(
        "A thought experiment on the same person: it shows how much each information source contributes "
        "to the score. It reproduces the full_file, thin_file and no_file situations. "
        "It is not a prediction about anyone and not advice."
    )


def render_tab_1_one_applicant():
    """Tab 1: One held-out test applicant interactive exploration."""
    st.markdown("Select or draw an anonymized historical held-out test applicant.")

    resources = load_scoring_resources()
    meta_test = resources["meta_test"]
    threshold_b = resources["threshold_b"]

    # Filter controls
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        seg_choice = st.selectbox(
            "Filter by Segment",
            options=["All segments", "full_file", "thin_file", "no_file"],
            index=0,
        )
    with col_f2:
        dec_choice = st.selectbox(
            "Filter by Model B Decision",
            options=["All decisions", "declined", "approved"],
            index=0,
        )
    with col_f3:
        out_choice = st.selectbox(
            "Filter by Actual Outcome",
            options=["All outcomes", "default", "no default"],
            index=0,
        )

    # Initialize session state for chosen applicant
    if "selected_applicant_id" not in st.session_state:
        # Default to a representative thin_file applicant
        st.session_state["selected_applicant_id"] = 333107

    # Draw button
    col_btn, col_inp = st.columns([1, 2])
    with col_btn:
        st.write("")
        st.write("")
        if st.button("Draw a new applicant"):
            # Filter pool according to chosen criteria
            pool_mask = pd.Series(True, index=meta_test.index)
            if seg_choice != "All segments":
                pool_mask = pool_mask & (meta_test["segment"] == seg_choice)

            # Load Model B test scores to filter by Model B decision quickly
            test_b = pd.read_parquet(_PROJECT_ROOT / "reports" / "predictions" / "model_b_test.parquet").set_index("SK_ID_CURR")
            if dec_choice == "declined":
                decl_mask = (test_b.loc[meta_test.index, "score"] >= threshold_b).values
                pool_mask = pool_mask & decl_mask
            elif dec_choice == "approved":
                appr_mask = (test_b.loc[meta_test.index, "score"] < threshold_b).values
                pool_mask = pool_mask & appr_mask

            if out_choice == "default":
                pool_mask = pool_mask & (meta_test["TARGET"] == 1)
            elif out_choice == "no default":
                pool_mask = pool_mask & (meta_test["TARGET"] == 0)

            matching_ids = meta_test[pool_mask].index.values
            if len(matching_ids) > 0:
                drawn_id = int(np.random.choice(matching_ids))
                st.session_state["selected_applicant_id"] = drawn_id
                st.session_state[f"reveal_{drawn_id}"] = False
                st.rerun()
            else:
                st.warning("No test applicants match the selected criteria.")

    with col_inp:
        typed_id_str = st.text_input(
            "Or enter a held-out test applicant SK_ID_CURR",
            value=str(st.session_state["selected_applicant_id"]),
            help="Must be an ID from data/splits/test_ids.csv",
        )
        if typed_id_str:
            try:
                typed_id = int(typed_id_str.strip())
                if typed_id in resources["test_ids_set"]:
                    if typed_id != st.session_state["selected_applicant_id"]:
                        st.session_state["selected_applicant_id"] = typed_id
                        st.session_state[f"reveal_{typed_id}"] = False
                        st.rerun()
                else:
                    st.error(
                        f"ID {typed_id} is not in the held-out test split (data/splits/test_ids.csv). "
                        "Train and validation applicants cannot be viewed here."
                    )
            except ValueError:
                st.error("Please enter a valid numeric SK_ID_CURR.")

    st.markdown("---")
    curr_id = st.session_state.get("selected_applicant_id", 333107)
    render_single_applicant_card(curr_id)


def render_tab_2_batch_applicants():
    """Tab 2: A batch of new applicants live evaluation and comparison."""
    st.markdown("Evaluate a randomly drawn sample of held-out test applicants against full-test-split benchmarks.")

    resources = load_scoring_resources()
    meta_test = resources["meta_test"]
    threshold_a = resources["threshold_a"]
    threshold_b = resources["threshold_b"]

    flips_data = load_flips_json()
    fairness_data = load_fairness_json()

    # Controls
    col_c1, col_c2, col_c3 = st.columns([2, 2, 1])
    with col_c1:
        n_batch = st.slider("Number of applicants in batch", min_value=20, max_value=500, value=100, step=10)
    with col_c2:
        seg_filter = st.selectbox(
            "Batch Segment Filter",
            options=["All segments", "full_file", "thin_file", "no_file"],
            index=0,
            key="batch_seg_filter",
        )
    with col_c3:
        st.write("")
        st.write("")
        draw_batch_clicked = st.button("Draw batch", key="btn_draw_batch")

    if "batch_ids" not in st.session_state or draw_batch_clicked:
        eligible_mask = pd.Series(True, index=meta_test.index)
        if seg_filter != "All segments":
            eligible_mask = (meta_test["segment"] == seg_filter)

        eligible_ids = meta_test[eligible_mask].index.values
        chosen_n = min(n_batch, len(eligible_ids))
        st.session_state["batch_ids"] = np.random.choice(eligible_ids, size=chosen_n, replace=False).tolist()

    batch_ids = st.session_state["batch_ids"]

    # Score batch live
    res = score_applicants(batch_ids)
    sub_meta = meta_test.loc[batch_ids]

    scores_a = res.score_a
    scores_b = res.score_b
    targets = sub_meta["TARGET"].values
    segments = sub_meta["segment"].values

    decl_a = scores_a >= threshold_a
    decl_b = scores_b >= threshold_b

    dec_a_str = np.where(decl_a, "declined", "approved")
    dec_b_str = np.where(decl_b, "declined", "approved")
    target_str = np.where(targets == 1, "defaulted", "repaid")

    verdict_a_list = [compute_verdict(dec_a_str[i], targets[i]) for i in range(len(batch_ids))]
    verdict_b_list = [compute_verdict(dec_b_str[i], targets[i]) for i in range(len(batch_ids))]

    # Table of applicants
    table_df = pd.DataFrame({
        "SK_ID_CURR": batch_ids,
        "Segment": segments,
        "Model A Score": np.round(scores_a, 4),
        "Model A Decision": dec_a_str,
        "Model B Score": np.round(scores_b, 4),
        "Model B Decision": dec_b_str,
        "Actual Outcome": target_str,
        "Model A Verdict": verdict_a_list,
        "Model B Verdict": verdict_b_list,
    })

    st.subheader(f"Batch Table ({len(table_df)} held-out test applicants)")
    st.dataframe(table_df, use_container_width=True, height=280)
    st.caption("Live scoring on drawn batch. Decisions at 15% validation policy. Source: live scoring on data/splits/test_ids.csv.")

    # Live summary counts for batch
    n_total = len(batch_ids)
    n_decl_a = int(np.sum(decl_a))
    n_appr_a = n_total - n_decl_a
    n_def_decl_a = int(np.sum(decl_a & (targets == 1)))
    n_def_appr_a = int(np.sum((~decl_a) & (targets == 1)))
    n_good_decl_a = int(np.sum(decl_a & (targets == 0)))

    n_decl_b = int(np.sum(decl_b))
    n_appr_b = n_total - n_decl_b
    n_def_decl_b = int(np.sum(decl_b & (targets == 1)))
    n_def_appr_b = int(np.sum((~decl_b) & (targets == 1)))
    n_good_decl_b = int(np.sum(decl_b & (targets == 0)))

    # Flips in batch
    rescued = int(np.sum(decl_a & (~decl_b) & (targets == 0)))
    let_in = int(np.sum(decl_a & (~decl_b) & (targets == 1)))
    newly_declined_good = int(np.sum((~decl_a) & decl_b & (targets == 0)))
    newly_caught = int(np.sum((~decl_a) & decl_b & (targets == 1)))

    # Full test-split reference metrics
    slice_key = "overall" if seg_filter == "All segments" else seg_filter
    if slice_key == "overall":
        ref_flips = flips_data["slices"]["overall"]
        ref_counts = ref_flips["counts"]
        n_ref = ref_flips["n"]
    else:
        ref_flips = flips_data["slices"]["segment"][slice_key]
        ref_counts = ref_flips["counts"]
        n_ref = ref_flips["n"]

    ref_decl_a = ref_counts["both_declined_default"] + ref_counts["both_declined_good"] + ref_counts["rescued"] + ref_counts["let_in"]
    ref_appr_a = n_ref - ref_decl_a
    ref_def_decl_a = ref_counts["both_declined_default"] + ref_counts["let_in"]
    ref_def_appr_a = ref_counts["both_approved_default"] + ref_counts["newly_caught"]
    ref_good_decl_a = ref_counts["both_declined_good"] + ref_counts["rescued"]

    ref_decl_b = ref_counts["both_declined_default"] + ref_counts["both_declined_good"] + ref_counts["newly_declined_good"] + ref_counts["newly_caught"]
    ref_appr_b = n_ref - ref_decl_b
    ref_def_decl_b = ref_counts["both_declined_default"] + ref_counts["newly_caught"]
    ref_def_appr_b = ref_counts["both_approved_default"] + ref_counts["let_in"]
    ref_good_decl_b = ref_counts["both_declined_good"] + ref_counts["newly_declined_good"]

    ref_rescued = ref_counts["rescued"]
    ref_let_in = ref_counts["let_in"]
    ref_newly_decl_good = ref_counts["newly_declined_good"]
    ref_newly_caught = ref_counts["newly_caught"]

    # Comparative Summary Cards
    st.subheader("Live Batch Metrics vs Full Test-Split Benchmark")

    st.markdown("#### Model A vs Model B Decisions")
    summary_df = pd.DataFrame([
        {
            "Quantity": "Declined",
            "Batch Model A": f"{n_decl_a} ({n_decl_a / n_total:.1%})",
            "Full Split Model A": f"{ref_decl_a:,} ({ref_decl_a / n_ref:.1%})",
            "Batch Model B": f"{n_decl_b} ({n_decl_b / n_total:.1%})",
            "Full Split Model B": f"{ref_decl_b:,} ({ref_decl_b / n_ref:.1%})",
        },
        {
            "Quantity": "Approved",
            "Batch Model A": f"{n_appr_a} ({n_appr_a / n_total:.1%})",
            "Full Split Model A": f"{ref_appr_a:,} ({ref_appr_a / n_ref:.1%})",
            "Batch Model B": f"{n_appr_b} ({n_appr_b / n_total:.1%})",
            "Full Split Model B": f"{ref_appr_b:,} ({ref_appr_b / n_ref:.1%})",
        },
        {
            "Quantity": "Defaulters Declined (Capture)",
            "Batch Model A": f"{n_def_decl_a}",
            "Full Split Model A": f"{ref_def_decl_a:,}",
            "Batch Model B": f"{n_def_decl_b}",
            "Full Split Model B": f"{ref_def_decl_b:,}",
        },
        {
            "Quantity": "Defaulters Approved (Missed)",
            "Batch Model A": f"{n_def_appr_a}",
            "Full Split Model A": f"{ref_def_appr_a:,}",
            "Batch Model B": f"{n_def_appr_b}",
            "Full Split Model B": f"{ref_def_appr_b:,}",
        },
        {
            "Quantity": "Good Applicants Declined (False Declines)",
            "Batch Model A": f"{n_good_decl_a}",
            "Full Split Model A": f"{ref_good_decl_a:,}",
            "Batch Model B": f"{n_good_decl_b}",
            "Full Split Model B": f"{ref_good_decl_b:,}",
        },
    ])
    st.table(summary_df)

    st.markdown("#### Decisions Differing Between A and B (Flips)")
    flips_summary_df = pd.DataFrame([
        {
            "Flip Type": "Rescued (declined A, approved B, good)",
            "Batch Count": f"{rescued} ({rescued / n_total:.1%})",
            "Full Split Benchmark": f"{ref_rescued:,} ({ref_rescued / n_ref:.1%})",
        },
        {
            "Flip Type": "Let-in (declined A, approved B, default)",
            "Batch Count": f"{let_in} ({let_in / n_total:.1%})",
            "Full Split Benchmark": f"{ref_let_in:,} ({ref_let_in / n_ref:.1%})",
        },
        {
            "Flip Type": "Newly declined good (approved A, declined B, good)",
            "Batch Count": f"{newly_declined_good} ({newly_declined_good / n_total:.1%})",
            "Full Split Benchmark": f"{ref_newly_decl_good:,} ({ref_newly_decl_good / n_ref:.1%})",
        },
        {
            "Flip Type": "Newly caught (approved A, declined B, default)",
            "Batch Count": f"{newly_caught} ({newly_caught / n_total:.1%})",
            "Full Split Benchmark": f"{ref_newly_caught:,} ({ref_newly_caught / n_ref:.1%})",
        },
    ])
    st.table(flips_summary_df)

    st.caption(
        "Small batches are noisy: results vary with every draw. "
        "The full test-split results are on the other pages."
    )
    st.caption("Thresholds source: reports/metrics/fairness.json. Benchmark source: reports/metrics/extras_flips.json.")


def render_page_new_applicants():
    """Main renderer for the 'New applicants' page."""
    st.title("👤 New Applicants: Live Scoring of Held-Out Test Applicants")
    st.markdown(PAGE_DISCLAIMER)

    tab1, tab2 = st.tabs(["One applicant", "A batch of new applicants"])

    with tab1:
        render_tab_1_one_applicant()

    with tab2:
        render_tab_2_batch_applicants()
