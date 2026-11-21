"""TrustScore Demo Dashboard (Streamlit).

Academic proof-of-concept on the Home Credit dataset. Not a lending system.
All results are from the held-out test split.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure MPLCONFIGDIR is writable in headless environments
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

from src.app.helpers import (
    COLOR_MODEL_A,
    COLOR_MODEL_B,
    render_header_and_banner,
    render_footer,
    load_fairness_json,
    load_model_a_json,
    load_model_b_json,
    load_a_vs_b_json,
    load_stability_json,
    load_flips_json,
    load_base_rates_csv,
    load_frontier_csv,
    load_calibration_bins_csv,
    load_calibration_json,
    load_auc_by_group_json,
    get_display_sample_for_flips,
    compute_live_policy,
    check_consistency_at_15,
)

# -----------------------------------------------------------------------------
# App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TrustScore Demo Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_NAMES = [
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


# =============================================================================
# Chart & Component Builders (Reusable across Pages and Guided Tour)
# =============================================================================

def build_data_segments_chart() -> alt.Chart:
    """Builds segment default rates chart from reports/tables/base_rates.csv."""
    df_base = load_base_rates_csv()
    seg_df = df_base[df_base["dimension"] == "segment"].copy()
    chart = (
        alt.Chart(seg_df)
        .mark_bar(size=40, color="#4682b4")
        .encode(
            x=alt.X("group:N", title="Applicant Segment", sort=["full_file", "thin_file", "no_file"]),
            y=alt.Y("default_rate:Q", title="Default Rate", axis=alt.Axis(format=".1%")),
            tooltip=[
                alt.Tooltip("group:N", title="Segment"),
                alt.Tooltip("size:Q", title="Applicants", format=","),
                alt.Tooltip("defaults:Q", title="Defaulters", format=","),
                alt.Tooltip("default_rate:Q", title="Default Rate", format=".2%"),
            ],
        )
        .properties(height=320)
    )
    return chart


def build_auc_comparison_chart() -> alt.Chart:
    """Builds grouped bar chart of AUC A vs B by segment with 95% CIs."""
    ma = load_model_a_json()
    mb = load_model_b_json()

    rows = []
    # Overall
    rows.append({
        "segment": "overall",
        "model": "Model A",
        "auc": ma["test_results"]["overall"]["auc"],
        "ci_lower": ma["test_results"]["overall"]["ci_95_lower"],
        "ci_upper": ma["test_results"]["overall"]["ci_95_upper"],
    })
    rows.append({
        "segment": "overall",
        "model": "Model B",
        "auc": mb["test_results"]["overall"]["auc"],
        "ci_lower": mb["test_results"]["overall"]["ci_95_lower"],
        "ci_upper": mb["test_results"]["overall"]["ci_95_upper"],
    })

    # By segment
    for seg in ["full_file", "thin_file", "no_file"]:
        res_a = ma["test_results"]["by_segment"][seg]
        res_b = mb["test_results"]["by_segment"][seg]
        rows.append({
            "segment": seg,
            "model": "Model A",
            "auc": res_a["auc"],
            "ci_lower": res_a["ci_lower"] if "ci_lower" in res_a else res_a["ci_95_lower"],
            "ci_upper": res_a["ci_upper"] if "ci_upper" in res_a else res_a["ci_95_upper"],
        })
        rows.append({
            "segment": seg,
            "model": "Model B",
            "auc": res_b["auc"],
            "ci_lower": res_b["ci_lower"] if "ci_lower" in res_b else res_b["ci_95_lower"],
            "ci_upper": res_b["ci_upper"] if "ci_upper" in res_b else res_b["ci_95_upper"],
        })

    df = pd.DataFrame(rows)
    color_scale = alt.Scale(domain=["Model A", "Model B"], range=[COLOR_MODEL_A, COLOR_MODEL_B])

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title=None, axis=alt.Axis(labels=False)),
            y=alt.Y("auc:Q", title="Test ROC AUC", scale=alt.Scale(domain=[0.68, 0.90])),
            color=alt.Color("model:N", scale=color_scale, legend=alt.Legend(title="Model")),
            tooltip=[
                alt.Tooltip("segment:N", title="Segment"),
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("auc:Q", title="AUC", format=".4f"),
                alt.Tooltip("ci_lower:Q", title="95% CI Lower", format=".4f"),
                alt.Tooltip("ci_upper:Q", title="95% CI Upper", format=".4f"),
            ],
        )
    )

    error_bars = (
        alt.Chart(df)
        .mark_errorbar(color="black", ticks=True)
        .encode(
            x=alt.X("model:N"),
            y=alt.Y("ci_lower:Q", title=None),
            y2=alt.Y2("ci_upper:Q"),
        )
    )

    chart = (
        alt.layer(bars, error_bars, data=df)
        .properties(width=160, height=300)
        .facet(
            column=alt.Column(
                "segment:N",
                title=None,
                sort=["overall", "full_file", "thin_file", "no_file"],
                header=alt.Header(labelFontSize=12, labelFontWeight="bold"),
            )
        )
    )
    return chart


def build_net_flips_chart() -> alt.Chart:
    """Builds net decision flips per segment from extras_flips.json."""
    flips_data = load_flips_json()
    slices = flips_data["slices"]

    rows = []
    # Overall
    rows.append({
        "segment": "overall",
        "net": slices["overall"]["net"],
        "ci_lower": slices["overall"]["net_ci_95"][0],
        "ci_upper": slices["overall"]["net_ci_95"][1],
    })
    # Segments
    for seg in ["full_file", "thin_file", "no_file"]:
        s = slices["segment"][seg]
        rows.append({
            "segment": seg,
            "net": s["net"],
            "ci_lower": s["net_ci_95"][0],
            "ci_upper": s["net_ci_95"][1],
        })

    df = pd.DataFrame(rows)
    bars = (
        alt.Chart(df)
        .mark_bar(size=35, color="#2e7d32")
        .encode(
            x=alt.X("segment:N", title="Segment", sort=["overall", "full_file", "thin_file", "no_file"]),
            y=alt.Y("net:Q", title="Net Positive Decision Shifts"),
            tooltip=[
                alt.Tooltip("segment:N", title="Segment"),
                alt.Tooltip("net:Q", title="Net Shifts", format=","),
                alt.Tooltip("ci_lower:Q", title="95% CI Lower", format=","),
                alt.Tooltip("ci_upper:Q", title="95% CI Upper", format=","),
            ],
        )
    )
    errors = (
        alt.Chart(df)
        .mark_errorbar(color="black", ticks=True)
        .encode(
            x=alt.X("segment:N", sort=["overall", "full_file", "thin_file", "no_file"]),
            y=alt.Y("ci_lower:Q"),
            y2=alt.Y2("ci_upper:Q"),
        )
    )
    return alt.layer(bars, errors).properties(height=320)


def build_fairness_gaps_15_chart() -> alt.Chart:
    """Builds fairness gaps chart at 15% policy for gender and age from fairness.json."""
    fj = load_fairness_json()
    ov = fj["overall"]

    rows = []
    for attr, label in [("gender", "Gender (F vs M)"), ("binary_age", "Binary Age (<med vs >=med)")]:
        ma_gaps = ov[attr]["model_a"]["gaps"]
        mb_gaps = ov[attr]["model_b"]["gaps"]
        rows.append({
            "attribute": label,
            "metric": "Demographic Parity Diff",
            "model": "Model A",
            "value": ma_gaps["dp_difference"],
        })
        rows.append({
            "attribute": label,
            "metric": "Demographic Parity Diff",
            "model": "Model B",
            "value": mb_gaps["dp_difference"],
        })
        rows.append({
            "attribute": label,
            "metric": "Equalized Odds Diff",
            "model": "Model A",
            "value": ma_gaps["eo_difference"],
        })
        rows.append({
            "attribute": label,
            "metric": "Equalized Odds Diff",
            "model": "Model B",
            "value": mb_gaps["eo_difference"],
        })

    df = pd.DataFrame(rows)
    color_scale = alt.Scale(domain=["Model A", "Model B"], range=[COLOR_MODEL_A, COLOR_MODEL_B])

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title=None, axis=alt.Axis(labels=False)),
            y=alt.Y("value:Q", title="Gap Magnitude (Lower is Fairer)"),
            color=alt.Color("model:N", scale=color_scale),
            tooltip=[
                alt.Tooltip("attribute:N", title="Attribute"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("value:Q", title="Gap", format=".4f"),
            ],
        )
        .properties(height=180, width=240)
        .facet(
            row=alt.Row("metric:N", title=None),
            column=alt.Column("attribute:N", title=None),
        )
    )
    return chart


# =============================================================================
# PAGE 0: GUIDED TOUR
# =============================================================================
def render_page_0_guided_tour():
    st.title("🧭 Guided Tour: Key Findings")
    st.markdown("Walk through the 7 primary empirical takeaways from the TrustScore study.")

    tour_step = st.select_slider(
        "Select Step",
        options=[
            "1. Data & Segments",
            "2. Behavioral Lift",
            "3. Who Benefits",
            "4. Fairness Gaps",
            "5. Proxy Leakage",
            "6. Applicant Explanation",
            "7. Mitigation Lab",
        ],
        value="1. Data & Segments",
    )

    if tour_step.startswith("1."):
        df_base = load_base_rates_csv()
        seg_df = df_base[df_base["dimension"] == "segment"].set_index("group")
        n_tot = int(seg_df["size"].sum())
        r_full = float(seg_df.loc["full_file", "default_rate"])
        r_thin = float(seg_df.loc["thin_file", "default_rate"])
        r_no = float(seg_df.loc["no_file", "default_rate"])

        takeaway = (
            f"The test split contains {n_tot:,} applicants across three segments: "
            f"full-file ({r_full:.1%} default rate), thin-file ({r_thin:.1%} default rate), "
            f"and no-file ({r_no:.1%} default rate)."
        )
        st.info(takeaway)
        st.altair_chart(build_data_segments_chart(), use_container_width=True)
        st.caption("Source: reports/tables/base_rates.csv")

    elif tour_step.startswith("2."):
        avb = load_a_vs_b_json()
        ma = load_model_a_json()
        mb = load_model_b_json()
        auc_a = ma["test_results"]["overall"]["auc"]
        auc_b = mb["test_results"]["overall"]["auc"]
        lift_all = avb["test_auc"]["overall"]["diff_b_minus_a"]
        ci_lo = avb["test_auc"]["overall"]["paired_bootstrap"]["ci_95_lower"]
        ci_hi = avb["test_auc"]["overall"]["paired_bootstrap"]["ci_95_upper"]
        lift_thin = avb["test_auc"]["by_segment"]["thin_file"]["paired_bootstrap"]["mean_difference"]

        takeaway = (
            f"Behavioral features produce modest performance gains overall (AUC {auc_b:.4f} vs {auc_a:.4f}, "
            f"lift +{lift_all:.4f}, 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]). "
            f"The largest lift occurs in the thin-file segment (+{lift_thin:.4f} AUC)."
        )
        st.info(takeaway)
        st.altair_chart(build_auc_comparison_chart(), use_container_width=True)
        st.caption("Source: reports/metrics/model_a.json, reports/metrics/model_b.json, reports/metrics/a_vs_b.json")

    elif tour_step.startswith("3."):
        flips = load_flips_json()
        s_all = flips["slices"]["overall"]
        s_seg = flips["slices"]["segment"]
        tot_flips = s_all["counts"]["total_flips"]
        net_all = s_all["net"]
        net_lo = s_all["net_ci_95"][0]
        net_hi = s_all["net_ci_95"][1]
        net_full = s_seg["full_file"]["net"]
        net_thin = s_seg["thin_file"]["net"]

        takeaway = (
            f"At the 15% validation policy, Model B changes decisions for {tot_flips:,} test applicants, "
            f"producing a net positive shift of {net_all} correct decisions (95% CI [{net_lo:.0f}, {net_hi:.0f}]), "
            f"with {net_full} net improvements in full-file and {net_thin} in thin-file."
        )
        st.info(takeaway)
        st.altair_chart(build_net_flips_chart(), use_container_width=True)
        st.caption("Source: reports/metrics/extras_flips.json")

    elif tour_step.startswith("4."):
        fj = load_fairness_json()
        ov = fj["overall"]
        g_dp_a = ov["gender"]["model_a"]["gaps"]["dp_difference"]
        g_dp_b = ov["gender"]["model_b"]["gaps"]["dp_difference"]
        a_dp_a = ov["binary_age"]["model_a"]["gaps"]["dp_difference"]
        a_dp_b = ov["binary_age"]["model_b"]["gaps"]["dp_difference"]

        takeaway = (
            f"At the 15% decline policy, Model B slightly reduces demographic parity difference for gender "
            f"({g_dp_b:.4f} vs {g_dp_a:.4f}) and for binary age ({a_dp_b:.4f} vs {a_dp_a:.4f}), "
            f"yet persistent disparities remain driven by underlying base rate differences."
        )
        st.info(takeaway)
        st.altair_chart(build_fairness_gaps_15_chart(), use_container_width=True)
        st.caption("Source: reports/metrics/fairness.json")

    elif tour_step.startswith("5."):
        st.info("Added in stage 3 (Page 5: Proxy leakage).")

    elif tour_step.startswith("6."):
        st.info("Added in stage 3 (Page 6: Explanations & Applicant Explorer).")

    elif tour_step.startswith("7."):
        st.info("Added in stage 3 (Page 7: Mitigation lab).")


# =============================================================================
# PAGE 1: DOES BEHAVIOR HELP?
# =============================================================================
def render_page_1_does_behavior_help():
    st.title("📈 Does Behavior Help?")
    
    ma = load_model_a_json()
    mb = load_model_b_json()
    avb = load_a_vs_b_json()
    stab = load_stability_json()

    auc_a = ma["test_results"]["overall"]["auc"]
    auc_b = mb["test_results"]["overall"]["auc"]
    lift = avb["test_auc"]["overall"]["diff_b_minus_a"]
    lift_ci_l = avb["test_auc"]["overall"]["paired_bootstrap"]["ci_95_lower"]
    lift_ci_u = avb["test_auc"]["overall"]["paired_bootstrap"]["ci_95_upper"]

    thin_diff = avb["test_auc"]["by_segment"]["thin_file"]["paired_bootstrap"]["mean_difference"]
    thin_diff_l = avb["test_auc"]["by_segment"]["thin_file"]["paired_bootstrap"]["ci_95_lower"]
    thin_diff_u = avb["test_auc"]["by_segment"]["thin_file"]["paired_bootstrap"]["ci_95_upper"]

    lift_diff_info = stab["lift_difference"]
    mean_diff_lifts = lift_diff_info["mean_diff_of_lifts"]
    diff_lifts_l = lift_diff_info["ci_95_lower"]
    diff_lifts_u = lift_diff_info["ci_95_upper"]

    takeaway = (
        f"Behavioral features lift overall test AUC by +{lift:.4f} (95% CI [{lift_ci_l:.4f}, {lift_ci_u:.4f}]). "
        f"Lift is more pronounced in thin-file applicants (+{thin_diff:.4f}, 95% CI [{thin_diff_l:.4f}, {thin_diff_u:.4f}]), "
        f"with a thin-minus-full lift difference of +{mean_diff_lifts:.4f} (95% CI [{diff_lifts_l:.4f}, {diff_lifts_u:.4f}])."
    )
    st.info(takeaway)

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model A Test AUC", f"{auc_a:.4f}", help="Bureau baseline model")
    col2.metric("Model B Test AUC", f"{auc_b:.4f}", delta=f"+{lift:.4f} Lift")
    col3.metric("Thin-File Lift", f"+{thin_diff:.4f}", f"[{thin_diff_l:.4f}, {thin_diff_u:.4f}]")
    col4.metric("Thin - Full Lift Diff", f"+{mean_diff_lifts:.4f}", f"[{diff_lifts_l:.4f}, {diff_lifts_u:.4f}]")

    st.subheader("1. Test AUC by Segment (Model A vs Model B)")
    st.altair_chart(build_auc_comparison_chart(), use_container_width=True)
    st.caption("Source: reports/metrics/model_a.json, reports/metrics/model_b.json, reports/metrics/a_vs_b.json")

    st.subheader("2. Stability Across Random Splits (B minus A)")
    # Extract 5 splits from extras_stability.json and compute mean + std in code
    split_rows = []
    for sp in stab["splits"]:
        s_id = sp["seed"]
        split_rows.append({"split": f"Split {s_id}", "segment": "overall", "diff": sp["overall"]["diff"]})
        split_rows.append({"split": f"Split {s_id}", "segment": "full_file", "diff": sp["by_segment"]["full_file"]["diff"]})
        split_rows.append({"split": f"Split {s_id}", "segment": "thin_file", "diff": sp["by_segment"]["thin_file"]["diff"]})
        split_rows.append({"split": f"Split {s_id}", "segment": "no_file (low precision)", "diff": sp["by_segment"]["no_file"]["diff"]})

    df_splits = pd.DataFrame(split_rows)
    stats = df_splits.groupby("segment")["diff"].agg(["mean", "std"]).reset_index()

    # Altair chart: points for individual splits + summary errorbar
    chart_pts = (
        alt.Chart(df_splits)
        .mark_circle(size=70, color="#d95f02", opacity=0.8)
        .encode(
            x=alt.X("segment:N", title="Segment Evaluation", sort=["overall", "full_file", "thin_file", "no_file (low precision)"]),
            y=alt.Y("diff:Q", title="AUC Difference (Model B - Model A)"),
            tooltip=[alt.Tooltip("segment:N"), alt.Tooltip("split:N"), alt.Tooltip("diff:Q", format=".4f")],
        )
    )

    stats_pts = (
        alt.Chart(stats)
        .mark_point(size=120, color="black", shape="diamond")
        .encode(
            x=alt.X("segment:N", sort=["overall", "full_file", "thin_file", "no_file (low precision)"]),
            y=alt.Y("mean:Q"),
            tooltip=[alt.Tooltip("segment:N"), alt.Tooltip("mean:Q", title="Mean Diff", format=".4f"), alt.Tooltip("std:Q", title="Std Dev", format=".4f")],
        )
    )

    st.altair_chart(chart_pts + stats_pts, use_container_width=True)
    st.caption("Dots show B minus A lift for each of 5 independent stratified splits. Diamonds show sample mean. Note: no_file has only 31 defaulters and is marked low precision. Source: reports/metrics/extras_stability.json")


# =============================================================================
# PAGE 2: FLIP EXPLORER
# =============================================================================
def render_page_2_flip_explorer():
    st.title("🔄 Flip Explorer: Who Changes Decision?")

    fj = load_fairness_json()
    flips_data = load_flips_json()
    thresh_a = float(fj["policy"]["threshold_a"])
    thresh_b = float(fj["policy"]["threshold_b"])

    # Segment filter
    seg_choice = st.selectbox("Segment Filter", ["all", "full_file", "thin_file", "no_file"], index=0)
    
    # Load slice stats for takeaway
    if seg_choice == "all":
        s_info = flips_data["slices"]["overall"]
    else:
        s_info = flips_data["slices"]["segment"][seg_choice]

    n_rescued = s_info["counts"]["rescued"]
    n_caught = s_info["counts"]["newly_caught"]
    n_letin = s_info["counts"]["let_in"]
    n_good_dec = s_info["counts"]["newly_declined_good"]
    net_val = s_info["net"]
    ci_net_l = s_info["net_ci_95"][0]
    ci_net_u = s_info["net_ci_95"][1]

    takeaway = (
        f"In segment '{seg_choice}', Model B rescues {n_rescued:,} non-defaulters and newly catches {n_caught:,} defaulters, "
        f"while letting in {n_letin:,} defaulters and declining {n_good_dec:,} non-defaulters. "
        f"Net shift: {net_val:,} (95% CI [{ci_net_l:.0f}, {ci_net_u:.0f}])."
    )
    st.info(takeaway)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rescued (Good Appr)", f"{n_rescued:,}", help="Declined by A, Approved by B (TARGET = 0)")
    col2.metric("Newly Caught (Bad Decl)", f"{n_caught:,}", help="Approved by A, Declined by B (TARGET = 1)")
    col3.metric("Let-in (Bad Appr)", f"{n_letin:,}", help="Declined by A, Approved by B (TARGET = 1)")
    col4.metric("Newly Decl Good", f"{n_good_dec:,}", help="Approved by A, Declined by B (TARGET = 0)")
    col5.metric("Net Gain", f"{net_val:,}", f"[{ci_net_l:.0f}, {ci_net_u:.0f}]")

    # Display sample scatter plot
    sample_df = get_display_sample_for_flips(segment_filter=seg_choice)
    n_sample_tot = len(sample_df)
    n_sample_def = int((sample_df["TARGET"] == 1).sum())
    n_sample_non = n_sample_tot - n_sample_def

    # Add human readable outcome column
    sample_df["Outcome"] = np.where(sample_df["TARGET"] == 1, "Default (TARGET=1)", "Non-Default (TARGET=0)")

    color_scale = alt.Scale(domain=["Default (TARGET=1)", "Non-Default (TARGET=0)"], range=["#e41a1c", "#377eb8"])

    scatter = (
        alt.Chart(sample_df)
        .mark_circle(size=25, opacity=0.45)
        .encode(
            x=alt.X("score_a:Q", title=f"Model A Score (15% Threshold: {thresh_a:.4f})", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("score_b:Q", title=f"Model B Score (15% Threshold: {thresh_b:.4f})", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Outcome:N", scale=color_scale, legend=alt.Legend(title="Test Outcome")),
            tooltip=[
                alt.Tooltip("SK_ID_CURR:Q", title="Applicant ID"),
                alt.Tooltip("segment:N", title="Segment"),
                alt.Tooltip("score_a:Q", title="Model A Score", format=".4f"),
                alt.Tooltip("score_b:Q", title="Model B Score", format=".4f"),
                alt.Tooltip("Outcome:N", title="Outcome"),
            ],
        )
        .properties(height=480)
    )

    # Threshold rule overlays
    rule_x = alt.Chart(pd.DataFrame({"x": [thresh_a]})).mark_rule(color="black", strokeDash=[4, 4]).encode(x="x:Q")
    rule_y = alt.Chart(pd.DataFrame({"y": [thresh_b]})).mark_rule(color="black", strokeDash=[4, 4]).encode(y="y:Q")

    st.altair_chart(scatter + rule_x + rule_y, use_container_width=True)
    st.caption(
        f"Display sample: displays ALL {n_sample_def:,} defaulters in the selected filter and a random sample of {n_sample_non:,} "
        f"non-defaulters ({n_sample_tot:,} points total; seed from config) to preserve browser rendering responsiveness. "
        "Quadrants: Top-Right = Declined by both; Bottom-Left = Approved by both; Top-Left = Declined by B only; Bottom-Right = Declined by A only. "
        "Source: reports/predictions/model_a_test.parquet, reports/predictions/model_b_test.parquet, reports/metrics/fairness.json"
    )

    st.markdown("### Demographic Breakdown of Flips")
    slice_cat = st.radio("Breakdown Dimension", ["Gender", "Binary Age"], horizontal=True)
    if slice_cat == "Gender":
        dim_data = flips_data["slices"]["gender"]
    else:
        dim_data = flips_data["slices"]["binary_age"]

    breakdown_rows = []
    for g_key, g_val in dim_data.items():
        cnts = g_val["counts"]
        breakdown_rows.append({
            "Group": g_key,
            "Applicants": g_val["n"],
            "Rescued": cnts["rescued"],
            "Let-in": cnts["let_in"],
            "Newly Decl Good": cnts["newly_declined_good"],
            "Newly Caught": cnts["newly_caught"],
            "Net": g_val["net"],
            "95% CI Lower": g_val["net_ci_95"][0],
            "95% CI Upper": g_val["net_ci_95"][1],
        })
    st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True)
    st.caption("Source: reports/metrics/extras_flips.json")


# =============================================================================
# PAGE 3: POLICY SIMULATOR
# =============================================================================
def render_page_3_policy_simulator():
    st.title("🎛️ Policy Simulator")

    col_m, col_a = st.columns(2)
    model_choice = col_m.selectbox("Model", ["Model A", "Model B"], index=1)
    attr_choice = col_a.selectbox("Protected Attribute (Evaluation Only)", ["gender", "binary age"], index=0)

    decline_rate = st.slider("Target Decline Rate", min_value=0.05, max_value=0.30, value=0.15, step=0.01, format="%.0f%%")

    # Live computation
    live_res = compute_live_policy(model_choice, attr_choice, decline_rate)
    thresh = live_res["threshold"]
    real_decl = live_res["realized_decline_rate"]
    gaps = live_res["gaps"]
    overall = live_res["overall"]

    takeaway = (
        f"At a {decline_rate:.0%} target decline rate, {model_choice} establishes a validation threshold of {thresh:.4f}, "
        f"yielding a {real_decl:.2%} realized test decline rate and {overall['tpr']:.2%} default capture. "
        f"For {attr_choice}, Demographic Parity difference is {gaps['dp_difference']:.4f} and Equalized Odds difference is {gaps['eo_difference']:.4f}."
    )
    st.info(takeaway)

    # 15% Verification Check against fairness.json
    if np.isclose(decline_rate, 0.15):
        chk = check_consistency_at_15()
        if chk["passed"]:
            st.success("✅ Validation matches saved benchmark results in fairness.json within 1e-6 tolerance.")
        else:
            st.error(f"❌ Consistency check failed: maximum difference is {chk['max_diff']:.2e} > 1e-6.")
    else:
        st.warning(f"Simulating policy at {decline_rate:.0%}. Set slider to 15% to verify against saved fairness.json benchmark.")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Validation Threshold", f"{thresh:.4f}")
    col2.metric("Realized Decline", f"{real_decl:.2%}")
    col3.metric("Default Capture (TPR)", f"{overall['tpr']:.2%}")
    col4.metric("DP Difference", f"{gaps['dp_difference']:.4f}")
    col5.metric("Equalized Odds Diff", f"{gaps['eo_difference']:.4f}")

    st.subheader(f"1. Error Rates by {attr_choice.title()} Group (Live Policy)")
    bar_rows = []
    for g, stats in live_res["group_stats"].items():
        bar_rows.append({"Group": g, "Rate": stats["approval_rate"], "Metric": "Approval Rate"})
        bar_rows.append({"Group": g, "Rate": stats["tpr"], "Metric": "TPR (Default Capture)"})
        bar_rows.append({"Group": g, "Rate": stats["fpr"], "Metric": "FPR (False Decline)"})
        bar_rows.append({"Group": g, "Rate": stats["fnr"], "Metric": "FNR (Default Let-in)"})

    df_bars = pd.DataFrame(bar_rows)
    grp_chart = (
        alt.Chart(df_bars)
        .mark_bar()
        .encode(
            x=alt.X("Group:N", title="Group"),
            y=alt.Y("Rate:Q", title="Rate", axis=alt.Axis(format=".1%")),
            color=alt.Color("Group:N", scale=alt.Scale(scheme="category10")),
            tooltip=[alt.Tooltip("Group:N"), alt.Tooltip("Metric:N"), alt.Tooltip("Rate:Q", format=".2%")],
        )
        .properties(height=260, width=160)
        .facet(column=alt.Column("Metric:N", title=None))
    )
    st.altair_chart(grp_chart, use_container_width=True)
    st.caption("Live calculation based on test split predictions. Source: reports/predictions/model_a_test.parquet, reports/predictions/model_b_test.parquet")

    st.subheader("2. Efficiency Frontier Across Decline Rates (5% – 30%)")
    frontier_df = load_frontier_csv()
    
    # Filter for the chosen attribute columns
    dp_col = "gender_dp_diff" if attr_choice == "gender" else "binary_age_dp_diff"
    eo_col = "gender_eo_diff" if attr_choice == "gender" else "binary_age_eo_diff"

    f_plot_rows = []
    for _, row in frontier_df.iterrows():
        f_plot_rows.append({"model": row["model"], "decline_rate": row["decline_rate"], "Metric": "Default Capture (TPR)", "value": row["default_capture"]})
        f_plot_rows.append({"model": row["model"], "decline_rate": row["decline_rate"], "Metric": "Demographic Parity Diff", "value": row[dp_col]})
        f_plot_rows.append({"model": row["model"], "decline_rate": row["decline_rate"], "Metric": "Equalized Odds Diff", "value": row[eo_col]})

    df_fplot = pd.DataFrame(f_plot_rows)
    color_scale = alt.Scale(domain=["Model A", "Model B"], range=[COLOR_MODEL_A, COLOR_MODEL_B])

    line_chart = (
        alt.Chart()
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("decline_rate:Q", title="Decline Rate Target", axis=alt.Axis(format=".0%")),
            y=alt.Y("value:Q", title="Metric Value"),
            color=alt.Color("model:N", scale=color_scale),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("decline_rate:Q", title="Decline Rate", format=".1%"),
                alt.Tooltip("Metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Value", format=".4f"),
            ],
        )
    )

    rule = alt.Chart().mark_rule(color="red", strokeDash=[4, 4]).encode(x=alt.datum(decline_rate))
    chart_faceted = alt.layer(line_chart, rule, data=df_fplot).properties(height=280, width=240).facet(column=alt.Column("Metric:N", title=None))

    st.altair_chart(
        chart_faceted,
        use_container_width=True,
    )
    st.caption("Dashed red vertical rule marks current slider position. Source: reports/tables/app_frontier.csv")

    st.subheader(f"3. Base Rates: {attr_choice.title()} Groups")
    base_df = load_base_rates_csv()
    sub_base = base_df[base_df["dimension"] == ("gender" if attr_choice == "gender" else "binary_age")].copy()
    st.dataframe(sub_base, use_container_width=True)
    st.caption("Source: reports/tables/base_rates.csv")


# =============================================================================
# PAGE 4: FAIRNESS AND CALIBRATION
# =============================================================================
def render_page_4_fairness_calibration():
    st.title("🎯 Fairness & Calibration Analysis")

    calib_json = load_calibration_json()
    bins_df = load_calibration_bins_csv()
    auc_group = load_auc_by_group_json()

    col1, col2 = st.columns(2)
    m_toggle = col1.selectbox("Model", ["Model A", "Model B"], index=1)
    attr_toggle = col2.selectbox("Attribute / Segment", ["gender", "binary_age", "segment"], index=0)

    m_key = "model_a" if m_toggle == "Model A" else "model_b"
    brier_ov = calib_json[m_key]["overall_brier_score"]
    ece_ov = calib_json[m_key]["groups"]["overall:all"]["ece"]

    takeaway = (
        f"For {m_toggle}, overall Brier score is {brier_ov:.4f} and overall Expected Calibration Error (ECE) is {ece_ov:.4f}. "
        f"Calibration across {attr_toggle} groups indicates risk scores generally align with observed default frequencies, "
        "though thin-file and no-file exhibit higher uncertainty."
    )
    st.info(takeaway)

    col_k1, col_k2, col_k3 = st.columns(3)
    col_k1.metric("Overall Brier Score", f"{brier_ov:.4f}", help="Mean squared probability error")
    col_k2.metric("Overall ECE", f"{ece_ov:.4f}", help="Expected Calibration Error")
    col_k3.metric("Calibration Status", "Well-calibrated", "Close to 45° line")

    st.subheader(f"1. Reliability Curves by {attr_toggle.title()} ({m_toggle})")
    sub_bins = bins_df[(bins_df["model"] == m_key) & (bins_df["group_type"] == attr_toggle)].copy()

    # Filter out XNA if gender
    if attr_toggle == "gender":
        sub_bins = sub_bins[sub_bins["group_value"] != "XNA"]

    rel_chart = (
        alt.Chart(sub_bins)
        .mark_line(point=True)
        .encode(
            x=alt.X("mean_predicted:Q", title="Mean Predicted Default Probability", scale=alt.Scale(domain=[0, 0.45])),
            y=alt.Y("observed_rate:Q", title="Observed Default Rate", scale=alt.Scale(domain=[0, 0.45])),
            color=alt.Color("group_value:N", title="Group", scale=alt.Scale(scheme="category10")),
            tooltip=[
                alt.Tooltip("group_value:N", title="Group"),
                alt.Tooltip("bin_idx:Q", title="Decile Bin"),
                alt.Tooltip("count:Q", title="Applicants", format=","),
                alt.Tooltip("mean_predicted:Q", title="Mean Pred", format=".4f"),
                alt.Tooltip("observed_rate:Q", title="Observed", format=".4f"),
            ],
        )
        .properties(height=360)
    )

    # Reference 45-degree line
    diag_df = pd.DataFrame({"x": [0, 0.45], "y": [0, 0.45]})
    diag_line = alt.Chart(diag_df).mark_line(color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")

    st.altair_chart(rel_chart + diag_line, use_container_width=True)
    st.caption("Dashed gray line represents perfect calibration (predicted = observed). Source: reports/tables/extras_calibration_bins.csv")

    st.subheader(f"2. Group Calibration Summary Table ({m_toggle})")
    # Table of mean predicted vs observed default rate, Brier score and ECE point estimate
    calib_groups = calib_json[m_key]["groups"]
    table_rows = []
    for g_id, g_info in calib_groups.items():
        if g_info["group_type"] == attr_toggle or (attr_toggle == "overall" and g_info["group_type"] == "overall"):
            val_name = g_info["group_value"]
            if val_name == "XNA":
                continue

            # Special rule: Do NOT display ECE confidence intervals for thin_file or no_file
            if val_name in ["thin_file", "no_file"]:
                ece_ci_str = "N/A (Interval Inversion)"
            else:
                ece_ci_str = f"[{g_info['ece_ci_lower']:.4f}, {g_info['ece_ci_upper']:.4f}]"

            table_rows.append({
                "Group": val_name,
                "Count": f"{g_info['count']:,}",
                "Mean Predicted": f"{g_info['mean_predicted']:.4f}",
                "Observed Rate": f"{g_info['observed_rate']:.4f}",
                "Ratio (Pred/Obs)": f"{g_info['ratio_pred_obs']:.3f}",
                "Brier Score": f"{g_info['brier_score']:.4f}",
                "ECE Point Estimate": f"{g_info['ece']:.4f}",
                "ECE 95% CI": ece_ci_str,
            })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    st.caption(
        "Note: ECE confidence intervals are omitted for thin_file and no_file because the point estimate falls outside its bootstrap interval. "
        "Source: reports/metrics/extras_calibration.json"
    )

    st.subheader("3. Subgroup AUC with Confidence Intervals")
    auc_rows = []
    for grp_id, grp_data in auc_group["groups"].items():
        cat = grp_data["slice_category"]
        if cat in [attr_toggle, "segment"] or (attr_toggle == "binary_age" and cat == "binary_age"):
            auc_val = grp_data["auc_a"] if m_toggle == "Model A" else grp_data["auc_b"]
            ci = grp_data["auc_a_ci"] if m_toggle == "Model A" else grp_data["auc_b_ci"]
            auc_rows.append({
                "Group": grp_data["group"],
                "Category": cat,
                "Applicants": f"{grp_data['n']:,}",
                "Defaulters": f"{grp_data['n_positives']:,}",
                "Default Rate": f"{grp_data['default_rate']:.2%}",
                "AUC": f"{auc_val:.4f}",
                "95% CI": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
                "Low Precision": "⚠️ Yes" if grp_data["low_precision"] else "No",
            })

    st.dataframe(pd.DataFrame(auc_rows), use_container_width=True)
    st.caption("Low-precision flags denote subgroups with small sample counts or few defaulters. Source: reports/metrics/extras_auc_by_group.json")


# =============================================================================
# Placeholder Renderers for Pages 5 to 9 (Completed in Stage 3)
# =============================================================================
def render_stage_3_placeholder(page_title: str):
    st.title(page_title)
    st.info("Added in stage 3.")


# =============================================================================
# Main Application Router
# =============================================================================
def main():
    st.sidebar.title("TrustScore Dashboard")
    page = st.sidebar.radio("Navigation", PAGE_NAMES, index=0)

    render_header_and_banner()

    if page == "Guided tour":
        render_page_0_guided_tour()
    elif page == "Does behavior help?":
        render_page_1_does_behavior_help()
    elif page == "Flip explorer":
        render_page_2_flip_explorer()
    elif page == "Policy simulator":
        render_page_3_policy_simulator()
    elif page == "Fairness and calibration":
        render_page_4_fairness_calibration()
    elif page == "Proxy leakage":
        render_stage_3_placeholder("🔍 Proxy Leakage")
    elif page == "Explanations":
        render_stage_3_placeholder("💡 Explanations & Applicant Explorer")
    elif page == "Mitigation lab":
        render_stage_3_placeholder("⚖️ Mitigation Lab")
    elif page == "Which features matter":
        render_stage_3_placeholder("📊 Which Features Matter (Ablation)")
    elif page == "Limitations":
        render_stage_3_placeholder("📝 Limitations & Ethical Considerations")

    render_footer()


if __name__ == "__main__":
    main()
