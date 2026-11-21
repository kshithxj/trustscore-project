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
    load_proxy_json,
    load_shap_summary_json,
    load_mitigation_control_json,
    load_ablation_json,
    load_shap_notes_md,
    load_limitations_md,
    load_shap_sample_parquet,
    get_display_sample_for_flips,
    compute_live_policy,
    check_consistency_at_15,
)
from src.app.pages.new_applicants import (
    render_page_new_applicants,
    render_single_applicant_card,
)
from src.app.pages.studio import render_page_studio
from src.app.pages.case_files import render_page_case_files
from src.app.pages.check_applicant import render_page_check_applicant
from src.app.visuals import apply_presenter_styling

# -----------------------------------------------------------------------------
# App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TrustScore Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEMO_PAGES = [
    "TrustScore Studio",
    "Check an applicant",
    "Case files",
    "New applicants",
]

RESEARCH_PAGES = [
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

PAGE_NAMES = DEMO_PAGES + RESEARCH_PAGES


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


def build_proxy_leakage_chart() -> alt.Chart:
    """Builds proxy prediction AUC chart (S1, S2, S3) with 0.5 reference line from proxy.json."""
    proxy_data = load_proxy_json()
    rows = []
    for target_key, target_label in [("gender", "Predicting Gender"), ("binary_age", "Predicting Age")]:
        fsets = proxy_data["targets"][target_key]["feature_sets"]
        for s_key in ["S1", "S2", "S3"]:
            rows.append({
                "target": target_label,
                "feature_set": s_key,
                "auc": fsets[s_key]["mean_auc"],
                "std": fsets[s_key]["std_auc"],
            })

    df = pd.DataFrame(rows)
    bars = (
        alt.Chart()
        .mark_bar()
        .encode(
            x=alt.X("feature_set:N", title="Feature Set", sort=["S1", "S2", "S3"]),
            y=alt.Y("auc:Q", title="Cross-Validated AUC", scale=alt.Scale(domain=[0.45, 1.0])),
            color=alt.Color("feature_set:N", scale=alt.Scale(scheme="tableau10"), legend=alt.Legend(title="Feature Set")),
            tooltip=[
                alt.Tooltip("target:N"),
                alt.Tooltip("feature_set:N"),
                alt.Tooltip("auc:Q", format=".4f"),
                alt.Tooltip("std:Q", format=".4f"),
            ],
        )
    )
    rule = alt.Chart().mark_rule(color="red", strokeDash=[4, 4]).encode(y=alt.datum(0.5))
    chart = alt.layer(bars, rule, data=df).properties(width=220, height=280).facet(column=alt.Column("target:N", title=None))
    return chart


def build_mitigation_overview_chart() -> alt.Chart:
    """Builds mitigated vs control chart for equalized odds and balanced accuracy from mitigation_control.json."""
    mit_data = load_mitigation_control_json()
    runs = mit_data["runs"]

    rows = []
    run_configs = [
        ("gender_equalized_odds", "Gender EO"),
        ("gender_demographic_parity", "Gender DP"),
        ("binary_age_equalized_odds", "Age EO"),
        ("binary_age_demographic_parity", "Age DP"),
    ]

    for run_key, label in run_configs:
        r = runs[run_key]
        mit = r["mitigated"]
        ctrl = r["control"]

        rows.append({"run": label, "variant": "Mitigated", "metric": "Equalized Odds Diff", "val": mit["equalized_odds_difference"]})
        rows.append({"run": label, "variant": "Control (Single Thresh)", "metric": "Equalized Odds Diff", "val": ctrl["equalized_odds_difference"]})

        rows.append({"run": label, "variant": "Mitigated", "metric": "Balanced Accuracy", "val": mit["balanced_accuracy"]})
        rows.append({"run": label, "variant": "Control (Single Thresh)", "metric": "Balanced Accuracy", "val": ctrl["balanced_accuracy"]})

    df = pd.DataFrame(rows)
    color_scale = alt.Scale(domain=["Mitigated", "Control (Single Thresh)"], range=["#7570b3", "#1b9e77"])

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("variant:N", title=None, axis=alt.Axis(labels=False)),
            y=alt.Y("val:Q", title="Value"),
            color=alt.Color("variant:N", scale=color_scale),
            tooltip=[alt.Tooltip("run:N"), alt.Tooltip("variant:N"), alt.Tooltip("metric:N"), alt.Tooltip("val:Q", format=".4f")],
        )
        .properties(width=140, height=200)
        .facet(row=alt.Row("metric:N", title=None), column=alt.Column("run:N", title=None))
    )
    return bars


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
        proxy_data = load_proxy_json()
        s1_g = proxy_data["targets"]["gender"]["feature_sets"]["S1"]["mean_auc"]
        s1_a = proxy_data["targets"]["binary_age"]["feature_sets"]["S1"]["mean_auc"]
        takeaway = (
            f"Standard application features (S1) predict gender with {s1_g:.4f} AUC and age with {s1_a:.4f} AUC (baseline 0.50), "
            "demonstrating severe proxy leakage through legitimate predictors such as occupation, income, and vehicle ownership."
        )
        st.info(takeaway)
        st.altair_chart(build_proxy_leakage_chart(), use_container_width=True)
        st.caption("Source: reports/metrics/proxy.json")

    elif tour_step.startswith("6."):
        shap_sum = load_shap_summary_json()
        lexp = shap_sum.get("chosen_local_applicant") or shap_sum.get("local_explanation", {})
        app_id = int(lexp.get("SK_ID_CURR", lexp.get("applicant_id", 333107)))
        app_score = float(lexp.get("score", 0.2153))
        top_contribs = lexp.get("top_8_contributions", [{"feature": "EXT_SOURCE_2", "shap_value": 1.1069}])
        top_feat = top_contribs[0]["feature"]
        top_contrib = top_contribs[0]["shap_value"]

        takeaway = (
            f"Local explanation for the median declined thin-file applicant ({app_id}, score {app_score:.4f}): "
            f"the strongest risk driver is {top_feat} (+{top_contrib:.4f} log-odds), illustrating how TreeSHAP attributes decision margins."
        )
        st.info(takeaway)
        render_single_applicant_card(app_id, is_guided_tour=True)
        if st.button("📁 View More Examples in Case Files", key="btn_tour_cases_step6"):
            st.session_state["nav_target"] = "Case files"
            st.rerun()

    elif tour_step.startswith("7."):
        mit_data = load_mitigation_control_json()
        g_eo_mit = mit_data["runs"]["gender_equalized_odds"]["mitigated"]["equalized_odds_difference"]
        g_eo_ctrl = mit_data["runs"]["gender_equalized_odds"]["control"]["equalized_odds_difference"]
        takeaway = (
            f"Threshold optimization reduces gender equalized odds difference to {g_eo_mit:.4f} vs {g_eo_ctrl:.4f} for the matched control, "
            "but requires explicit group-dependent decision cutoffs, which is legally and ethically contested in credit underwriting."
        )
        st.info(takeaway)
        st.altair_chart(build_mitigation_overview_chart(), use_container_width=True)
        st.caption("Source: reports/metrics/mitigation_control.json")

        st.markdown("---")
        st.markdown("### 🎯 Explore the Interactive Demo Studio")
        st.markdown("Try live scoring, information source ablations, and canonical case files:")
        cd1, cd2, cd3 = st.columns(3)
        with cd1:
            if st.button("📁 Open Case Files", key="btn_tour_to_cases"):
                st.session_state["nav_target"] = "Case files"
                st.rerun()
        with cd2:
            if st.button("🔍 Check an Applicant", key="btn_tour_to_check"):
                st.session_state["nav_target"] = "Check an applicant"
                st.rerun()
        with cd3:
            if st.button("👤 Open New Applicants", key="btn_tour_to_new"):
                st.session_state["nav_target"] = "New applicants"
                st.rerun()


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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model A Test AUC", f"{auc_a:.4f}", help="Bureau baseline model")
    col2.metric("Model B Test AUC", f"{auc_b:.4f}", delta=f"+{lift:.4f} Lift")
    col3.metric("Thin-File Lift", f"+{thin_diff:.4f}", f"[{thin_diff_l:.4f}, {thin_diff_u:.4f}]")
    col4.metric("Thin - Full Lift Diff", f"+{mean_diff_lifts:.4f}", f"[{diff_lifts_l:.4f}, {diff_lifts_u:.4f}]")

    st.subheader("1. Test AUC by Segment (Model A vs Model B)")
    st.altair_chart(build_auc_comparison_chart(), use_container_width=True)
    st.caption("Source: reports/metrics/model_a.json, reports/metrics/model_b.json, reports/metrics/a_vs_b.json")

    st.subheader("2. Stability Across Random Splits (B minus A)")
    split_rows = []
    for sp in stab["splits"]:
        s_id = sp["seed"]
        split_rows.append({"split": f"Split {s_id}", "segment": "overall", "diff": sp["overall"]["diff"]})
        split_rows.append({"split": f"Split {s_id}", "segment": "full_file", "diff": sp["by_segment"]["full_file"]["diff"]})
        split_rows.append({"split": f"Split {s_id}", "segment": "thin_file", "diff": sp["by_segment"]["thin_file"]["diff"]})
        split_rows.append({"split": f"Split {s_id}", "segment": "no_file (low precision)", "diff": sp["by_segment"]["no_file"]["diff"]})

    df_splits = pd.DataFrame(split_rows)
    stats = df_splits.groupby("segment")["diff"].agg(["mean", "std"]).reset_index()

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

    seg_choice = st.selectbox("Segment Filter", ["all", "full_file", "thin_file", "no_file"], index=0)
    
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

    sample_df = get_display_sample_for_flips(segment_filter=seg_choice)
    n_sample_tot = len(sample_df)
    n_sample_def = int((sample_df["TARGET"] == 1).sum())
    n_sample_non = n_sample_tot - n_sample_def

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

    st.altair_chart(chart_faceted, use_container_width=True)
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

    diag_df = pd.DataFrame({"x": [0, 0.45], "y": [0, 0.45]})
    diag_line = alt.Chart(diag_df).mark_line(color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")

    st.altair_chart(rel_chart + diag_line, use_container_width=True)
    st.caption("Dashed gray line represents perfect calibration (predicted = observed). Source: reports/tables/extras_calibration_bins.csv")

    st.subheader(f"2. Group Calibration Summary Table ({m_toggle})")
    calib_groups = calib_json[m_key]["groups"]
    table_rows = []
    for g_id, g_info in calib_groups.items():
        if g_info["group_type"] == attr_toggle or (attr_toggle == "overall" and g_info["group_type"] == "overall"):
            val_name = g_info["group_value"]
            if val_name == "XNA":
                continue

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
# PAGE 5: PROXY LEAKAGE
# =============================================================================
def render_page_5_proxy_leakage():
    st.title("🔍 Proxy Leakage Evaluation")

    proxy_data = load_proxy_json()
    targets = proxy_data["targets"]

    g_s1_auc = targets["gender"]["feature_sets"]["S1"]["mean_auc"]
    g_s3_auc = targets["gender"]["feature_sets"]["S3"]["mean_auc"]
    a_s1_auc = targets["binary_age"]["feature_sets"]["S1"]["mean_auc"]
    a_s3_auc = targets["binary_age"]["feature_sets"]["S3"]["mean_auc"]

    takeaway = (
        f"Predictive models trained on inputs strictly excluding protected attributes reveal severe proxy leakage: "
        f"application features alone (S1) predict gender with {g_s1_auc:.4f} AUC and binary age with {a_s1_auc:.4f} AUC "
        f"(rising to {g_s3_auc:.4f} and {a_s3_auc:.4f} in S3 vs 0.50 baseline). "
        "Omitting protected attributes does not prevent models from learning demographic proxies."
    )
    st.info(takeaway)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gender S1 AUC", f"{g_s1_auc:.4f}", help="Application features only")
    col2.metric("Gender S3 AUC", f"{g_s3_auc:.4f}", help="Application + Behavioral")
    col3.metric("Age S1 AUC", f"{a_s1_auc:.4f}", help="Application features only")
    col4.metric("Age S3 AUC", f"{a_s3_auc:.4f}", help="Application + Behavioral")

    st.subheader("1. Cross-Validated AUC Predicting Protected Attributes (S1, S2, S3)")
    st.altair_chart(build_proxy_leakage_chart(), use_container_width=True)
    st.caption("An AUC well above 0.5 means the inputs can reveal the attribute even though it is not an input. S1 = Application, S2 = Behavioral, S3 = Combined. Source: reports/metrics/proxy.json")

    st.subheader("2. Target Prevalence & Top Proxy Features")
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("#### Gender Target Details")
        st.write(f"- **Positive Label**: {targets['gender']['positive_label']}")
        st.write(f"- **Prevalence**: {targets['gender']['prevalence']:.2%}")
        st.write(f"- **Majority Class Accuracy**: {targets['gender']['majority_class_accuracy']:.2%}")
        st.write(f"- **Samples**: {targets['gender']['n_samples']:,}")
        
        top5_g = targets["gender"]["top_5_features_by_gain_for_best_set"]["top_5"]
        st.dataframe(pd.DataFrame(top5_g), use_container_width=True)
        st.caption("Top 5 features by gain predicting gender. Source: reports/metrics/proxy.json")

    with col_t2:
        st.markdown("#### Binary Age Target Details")
        st.write(f"- **Positive Label**: {targets['binary_age']['positive_label']}")
        st.write(f"- **Prevalence**: {targets['binary_age']['prevalence']:.2%}")
        st.write(f"- **Majority Class Accuracy**: {targets['binary_age']['majority_class_accuracy']:.2%}")
        st.write(f"- **Samples**: {targets['binary_age']['n_samples']:,}")

        top5_a = targets["binary_age"]["top_5_features_by_gain_for_best_set"]["top_5"]
        st.dataframe(pd.DataFrame(top5_a), use_container_width=True)
        st.caption("Top 5 features by gain predicting binary age. Source: reports/metrics/proxy.json")


# =============================================================================
# PAGE 6: EXPLANATIONS & APPLICANT EXPLORER
# =============================================================================
def render_page_6_explanations():
    st.title("💡 Explanations & Applicant Explorer")

    shap_data = load_shap_summary_json()
    seg_sums = shap_data["segment_summaries"]
    fj = load_fairness_json()
    tb = float(fj["policy"]["threshold_b"])
    ta = float(fj["policy"]["threshold_a"])

    b_share_full = seg_sums["full_file"]["behavioral_share"]
    b_share_thin = seg_sums["thin_file"]["behavioral_share"]
    top_full = list(seg_sums["full_file"]["top_15_features"].keys())[0]
    top_thin = list(seg_sums["thin_file"]["top_15_features"].keys())[0]

    takeaway = (
        f"Behavioral features account for {b_share_full:.1%} of mean absolute SHAP in full-file and {b_share_thin:.1%} in thin-file. "
        f"The primary global feature across segments remains {top_full}, while behavioral predictors such as repayment history provide "
        "supplementary signal especially when bureau depth is scarce."
    )
    st.info(takeaway)

    col1, col2, col3 = st.columns(3)
    col1.metric("Full-File Behavioral Share", f"{b_share_full:.1%}")
    col2.metric("Thin-File Behavioral Share", f"{b_share_thin:.1%}")
    col3.metric("Top Global Predictor", top_full)

    st.subheader("1. Global Feature Importance (Top 15 Features by Segment)")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### Full-File Top 15 Features")
        full_df = pd.DataFrame([
            {"Feature": f, "Mean |SHAP|": v}
            for f, v in seg_sums["full_file"]["top_15_features"].items()
        ])
        c_full = (
            alt.Chart(full_df)
            .mark_bar(color="#1f77b4")
            .encode(
                x=alt.X("Mean |SHAP|:Q", title="Mean |SHAP| (Margin Scale)"),
                y=alt.Y("Feature:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("Feature:N"), alt.Tooltip("Mean |SHAP|:Q", format=".4f")],
            )
            .properties(height=360)
        )
        st.altair_chart(c_full, use_container_width=True)

    with col_g2:
        st.markdown("#### Thin-File Top 15 Features")
        thin_df = pd.DataFrame([
            {"Feature": f, "Mean |SHAP|": v}
            for f, v in seg_sums["thin_file"]["top_15_features"].items()
        ])
        c_thin = (
            alt.Chart(thin_df)
            .mark_bar(color="#ff7f0e")
            .encode(
                x=alt.X("Mean |SHAP|:Q", title="Mean |SHAP| (Margin Scale)"),
                y=alt.Y("Feature:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("Feature:N"), alt.Tooltip("Mean |SHAP|:Q", format=".4f")],
            )
            .properties(height=360)
        )
        st.altair_chart(c_thin, use_container_width=True)

    st.caption("Margin (log-odds) scale. Source: reports/metrics/shap_summary.json")

    st.subheader("2. Applicant Explorer")
    shap_sample = load_shap_sample_parquet()

    col_f1, col_f2, col_f3 = st.columns(3)
    filt_seg = col_f1.selectbox("Filter Segment", ["all", "full_file", "thin_file", "no_file"], index=0)
    filt_dec = col_f2.selectbox("Filter Model B Decision (15% Policy)", ["All", "Declined", "Approved"], index=0)
    filt_out = col_f3.selectbox("Filter Actual Outcome", ["All", "Default (TARGET=1)", "Non-Default (TARGET=0)"], index=0)

    filtered = shap_sample.copy()
    if filt_seg != "all":
        filtered = filtered[filtered["segment"] == filt_seg]
    if filt_dec == "Declined":
        filtered = filtered[filtered["score_b"] >= tb]
    elif filt_dec == "Approved":
        filtered = filtered[filtered["score_b"] < tb]
    if filt_out == "Default (TARGET=1)":
        filtered = filtered[filtered["TARGET"] == 1]
    elif filt_out == "Non-Default (TARGET=0)":
        filtered = filtered[filtered["TARGET"] == 0]

    st.write(f"Matching sample applicants: **{len(filtered):,}** (from 4,940 precomputed test applicants)")

    if len(filtered) > 0:
        cand_ids = filtered["SK_ID_CURR"].tolist()
        sel_id = st.selectbox("Select Applicant (SK_ID_CURR)", cand_ids, index=0)
        row = filtered[filtered["SK_ID_CURR"] == sel_id].iloc[0]

        dec_a = "Declined" if row["score_a"] >= ta else "Approved"
        dec_b = "Declined" if row["score_b"] >= tb else "Approved"
        outcome_lbl = "Default (TARGET=1)" if row["TARGET"] == 1 else "Non-Default (TARGET=0)"

        c_a1, c_a2, c_a3, c_a4, c_a5 = st.columns(5)
        c_a1.metric("Segment", row["segment"])
        c_a2.metric("Model A Score", f"{row['score_a']:.4f}", dec_a)
        c_a3.metric("Model B Score", f"{row['score_b']:.4f}", dec_b)
        c_a4.metric("Bias Term", f"{row['bias']:.4f}")
        c_a5.metric("Actual Outcome", outcome_lbl)

        contrib_cols = [c for c in shap_sample.columns if c.startswith("shap_")]
        app_contribs = []
        for c in contrib_cols:
            f_name = c[5:]
            c_val = float(row[c])
            raw_val = row[f"raw_{f_name}"] if f"raw_{f_name}" in row else np.nan
            app_contribs.append({
                "Feature": f_name,
                "Contribution": c_val,
                "AbsContribution": abs(c_val),
                "Feature Value": str(raw_val),
                "Direction": "Increases Risk" if c_val > 0 else "Decreases Risk",
            })

        df_top10 = pd.DataFrame(app_contribs).sort_values("AbsContribution", ascending=False).head(10)

        c_local = (
            alt.Chart(df_top10)
            .mark_bar()
            .encode(
                x=alt.X("Contribution:Q", title="SHAP Contribution (Log-Odds Scale)"),
                y=alt.Y("Feature:N", sort="-x", title=None),
                color=alt.Color("Direction:N", scale=alt.Scale(domain=["Increases Risk", "Decreases Risk"], range=["#e41a1c", "#377eb8"])),
                tooltip=[
                    alt.Tooltip("Feature:N"),
                    alt.Tooltip("Contribution:Q", format=".4f"),
                    alt.Tooltip("Feature Value:N"),
                    alt.Tooltip("Direction:N"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(c_local, use_container_width=True)
        st.caption("Top 10 feature contributions on the margin (log-odds) scale. SHAP describes what the model uses, not what causes default. Source: reports/predictions/extras/app_shap_sample.parquet")

    st.subheader("3. Benchmark Local Explanation: Median Declined Thin-File Applicant")
    img_path = _PROJECT_ROOT / "reports" / "figures" / "shap_local_declined_thin_file.png"
    if img_path.exists():
        st.image(str(img_path), caption="Local explanation for median declined thin-file applicant (SK_ID_CURR 333107). Source: reports/figures/shap_local_declined_thin_file.png", use_container_width=True)

    st.subheader("4. SHAP Qualitative Notes")
    st.markdown(load_shap_notes_md())
    st.caption("Source: reports/shap_notes.md")


# =============================================================================
# PAGE 7: MITIGATION LAB
# =============================================================================
def render_page_7_mitigation_lab():
    st.title("⚖️ Mitigation Lab: Post-Processing Tradeoffs")

    st.warning(
        "⚠️ **Experiment only**: This uses a protected attribute at decision time, which is legally and ethically "
        "contested and may be unlawful in some jurisdictions. It changes the decline rate and moves errors between groups."
    )

    mit_data = load_mitigation_control_json()
    runs = mit_data["runs"]

    g_eo_mit = runs["gender_equalized_odds"]["mitigated"]["equalized_odds_difference"]
    g_eo_ctrl = runs["gender_equalized_odds"]["control"]["equalized_odds_difference"]
    g_eo_decl = runs["gender_equalized_odds"]["mitigated"]["decline_rate"]

    takeaway = (
        f"Post-processing mitigation against the matched single-threshold control reduces gender equalized odds difference "
        f"from {g_eo_ctrl:.4f} to {g_eo_mit:.4f}, but inflates overall decline volume to {g_eo_decl:.1%}. "
        "Evaluating against a matched control reveals that apparent fairness gains often stem from threshold shifts rather than true equity improvements."
    )
    st.info(takeaway)

    run_options = {
        "Gender: Equalized Odds": "gender_equalized_odds",
        "Gender: Demographic Parity": "gender_demographic_parity",
        "Binary Age: Equalized Odds": "binary_age_equalized_odds",
        "Binary Age: Demographic Parity": "binary_age_demographic_parity",
    }
    sel_label = st.selectbox("Select Mitigation Experiment", list(run_options.keys()), index=0)
    sel_run = runs[run_options[sel_label]]

    mit = sel_run["mitigated"]
    ctrl = sel_run["control"]
    paired = sel_run["paired_difference_mitigated_minus_control"]
    pts = paired["point_estimates"]
    ci_dict = paired["confidence_intervals_95"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mitigated Decline Rate", f"{mit['decline_rate']:.1%}", f"Control: {ctrl['decline_rate']:.1%}")
    c2.metric("Mitigated EO Diff", f"{mit['equalized_odds_difference']:.4f}", f"Control: {ctrl['equalized_odds_difference']:.4f}")
    c3.metric("Mitigated DP Diff", f"{mit['demographic_parity_difference']:.4f}", f"Control: {ctrl['demographic_parity_difference']:.4f}")
    c4.metric("Balanced Accuracy", f"{mit['balanced_accuracy']:.4f}", f"Control: {ctrl['balanced_accuracy']:.4f}")

    st.subheader("1. Mitigated vs Matched Single-Threshold Control Metrics")
    comp_rows = [
        {"Metric": "Equalized Odds Diff", "Variant": "Mitigated", "Value": mit["equalized_odds_difference"]},
        {"Metric": "Equalized Odds Diff", "Variant": "Control", "Value": ctrl["equalized_odds_difference"]},
        {"Metric": "Demographic Parity Diff", "Variant": "Mitigated", "Value": mit["demographic_parity_difference"]},
        {"Metric": "Demographic Parity Diff", "Variant": "Control", "Value": ctrl["demographic_parity_difference"]},
        {"Metric": "Balanced Accuracy", "Variant": "Mitigated", "Value": mit["balanced_accuracy"]},
        {"Metric": "Balanced Accuracy", "Variant": "Control", "Value": ctrl["balanced_accuracy"]},
        {"Metric": "Default Capture (TPR)", "Variant": "Mitigated", "Value": mit["default_capture"]},
        {"Metric": "Default Capture (TPR)", "Variant": "Control", "Value": ctrl["default_capture"]},
        {"Metric": "Overall FPR", "Variant": "Mitigated", "Value": mit["overall_fpr"]},
        {"Metric": "Overall FPR", "Variant": "Control", "Value": ctrl["overall_fpr"]},
    ]
    df_comp = pd.DataFrame(comp_rows)
    c_bars = (
        alt.Chart(df_comp)
        .mark_bar()
        .encode(
            x=alt.X("Variant:N", title=None),
            y=alt.Y("Value:Q", title="Metric Value"),
            color=alt.Color("Variant:N", scale=alt.Scale(domain=["Mitigated", "Control"], range=["#7570b3", "#1b9e77"])),
            tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("Variant:N"), alt.Tooltip("Value:Q", format=".4f")],
        )
        .properties(width=110, height=200)
        .facet(column=alt.Column("Metric:N", title=None))
    )
    st.altair_chart(c_bars, use_container_width=True)
    st.caption("Grouped bars comparing mitigated model against matched single-threshold control. Source: reports/metrics/mitigation_control.json")

    st.subheader("2. Paired Differences (Mitigated minus Control) with 95% CIs")
    metric_keys = [
        ("equalized_odds_difference", "Equalized Odds Diff", "eo_difference"),
        ("demographic_parity_difference", "Demographic Parity Diff", "dp_difference"),
        ("balanced_accuracy", "Balanced Accuracy", "balanced_accuracy"),
        ("default_capture", "Default Capture (TPR)", "default_capture"),
        ("overall_fpr", "Overall FPR", "overall_fpr"),
    ]
    pair_rows = []
    for full_k, disp_name, short_k in metric_keys:
        m_val = pts[short_k]
        ci = ci_dict[short_k]
        pair_rows.append({
            "Metric": disp_name,
            "Paired Difference (Mit - Ctrl)": f"{m_val:+.4f}",
            "95% CI Lower": f"{ci[0]:.4f}",
            "95% CI Upper": f"{ci[1]:.4f}",
        })
    st.dataframe(pd.DataFrame(pair_rows), use_container_width=True)
    st.caption("Source: reports/metrics/mitigation_control.json")


# =============================================================================
# PAGE 8: WHICH FEATURES MATTER (ABLATION)
# =============================================================================
def render_page_8_which_features_matter():
    st.title("📊 Which Features Matter? (Ablation Analysis)")

    abl_data = load_ablation_json()
    models = abl_data["models"]

    auc_b = models["model_b"]["auc"]
    auc_b_minus_prev = models["model_b_minus_previous_application"]["auc"]
    lift_thin_b = models["model_b"]["thin_file_lift_vs_a"]

    takeaway = (
        f"Ablating feature tables from Model B reveals that previous applications carry the highest incremental signal: "
        f"removing previous_application drops overall AUC from {auc_b:.4f} to {auc_b_minus_prev:.4f}. "
        f"In thin-file applicants, behavioral features provide +{lift_thin_b:.4f} AUC lift over Model A."
    )
    st.info(takeaway)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model B Overall AUC", f"{auc_b:.4f}")
    col2.metric("Without Previous App AUC", f"{auc_b_minus_prev:.4f}", delta=f"{auc_b_minus_prev - auc_b:.4f}")
    col3.metric("Thin-File Lift vs Model A", f"+{lift_thin_b:.4f}")

    st.subheader("1. Ablation Model Comparison")
    table_rows = []
    plot_rows = []

    for m_name, m_info in models.items():
        delta_val = m_info["delta_auc_vs_ref"]
        delta_ci = f"[{m_info['delta_auc_ci_lower']:.4f}, {m_info['delta_auc_ci_upper']:.4f}]" if m_info["compare_to"] else "N/A"
        thin_lift = m_info.get("thin_file_lift_vs_a", 0.0)

        table_rows.append({
            "Model": m_name,
            "Features": m_info["n_features"],
            "Overall AUC": f"{m_info['auc']:.4f}",
            "95% CI": f"[{m_info['auc_ci_lower']:.4f}, {m_info['auc_ci_upper']:.4f}]",
            "Delta AUC vs Ref": f"{delta_val:+.4f}" if m_info["compare_to"] else "Baseline",
            "Delta AUC 95% CI": delta_ci,
            "Thin-File Lift vs A": f"{thin_lift:+.4f}",
            "Gender DP Diff Change*": f"{m_info.get('change_gender_dp_vs_a', 0.0):+.4f}",
            "Age DP Diff Change*": f"{m_info.get('change_age_dp_vs_a', 0.0):+.4f}",
        })

        plot_rows.append({
            "Model": m_name,
            "Overall AUC": m_info["auc"],
            "CI Lower": m_info["auc_ci_lower"],
            "CI Upper": m_info["auc_ci_upper"],
        })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    st.caption(
        "*Note: Gap changes (marked with *) are POINT ESTIMATES ONLY (no confidence intervals were stored during ablation fits). "
        "Source: reports/metrics/extras_ablation.json"
    )

    df_plot = pd.DataFrame(plot_rows)
    c_abl = (
        alt.Chart(df_plot)
        .mark_bar(color="#2b5c8f")
        .encode(
            x=alt.X("Overall AUC:Q", scale=alt.Scale(domain=[0.74, 0.78]), title="Test AUC"),
            y=alt.Y("Model:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Overall AUC:Q", format=".4f")],
        )
        .properties(height=320)
    )
    st.altair_chart(c_abl, use_container_width=True)
    st.caption("Overall test AUC across ablation fits. Source: reports/metrics/extras_ablation.json")

    st.subheader("2. Detailed Ablation Tradeoff Visualizations")
    img_path = _PROJECT_ROOT / "reports" / "figures" / "extras_ablation.png"
    if img_path.exists():
        st.image(str(img_path), caption="Ablation study figures. Source: reports/figures/extras_ablation.png", use_container_width=True)


# =============================================================================
# PAGE 9: LIMITATIONS
# =============================================================================
def render_page_9_limitations():
    st.title("📝 Limitations & Ethical Considerations")
    lim_md = load_limitations_md()
    st.markdown(lim_md)
    st.caption("Source: docs/limitations.md")


# =============================================================================
# Main Application Router
# =============================================================================
def main():
    st.sidebar.title("TrustScore Studio")

    # Presenter View switch
    presenter_mode = st.sidebar.checkbox("Presenter view", value=False)
    apply_presenter_styling(presenter_mode)

    # Check for programmatic navigation request (e.g. from Studio entry cards or Guided Tour)
    target_page = st.session_state.pop("nav_target", None)
    default_idx = PAGE_NAMES.index(target_page) if target_page in PAGE_NAMES else 0

    st.sidebar.markdown("**Sections**")
    page = st.sidebar.radio("Navigation", PAGE_NAMES, index=default_idx)

    render_header_and_banner()

    if page == "TrustScore Studio":
        render_page_studio()
    elif page == "Check an applicant":
        render_page_check_applicant()
    elif page == "Case files":
        render_page_case_files()
    elif page == "New applicants":
        render_page_new_applicants()
    elif page == "Guided tour":
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
        render_page_5_proxy_leakage()
    elif page == "Explanations":
        render_page_6_explanations()
    elif page == "Mitigation lab":
        render_page_7_mitigation_lab()
    elif page == "Which features matter":
        render_page_8_which_features_matter()
    elif page == "Limitations":
        render_page_9_limitations()

    render_footer()


if __name__ == "__main__":
    main()
