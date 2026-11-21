"""Helpers and cached data loaders for TrustScore Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.utils import load_config
from src.fairness.policy import compute_threshold_from_validation, apply_decision_policy
from src.fairness.audit import compute_group_metrics, compute_fairness_gaps_hand

COLOR_MODEL_A = "#1f77b4"
COLOR_MODEL_B = "#ff7f0e"

BANNER_TEXT = (
    "Academic proof-of-concept on the Home Credit dataset. "
    "Not a lending system. All results are from the held-out test split."
)

FOOTER_TEXT = (
    "Gender and age are evaluation attributes only. "
    "Definitions: positive class = default (TARGET = 1); prediction = decline; "
    "approval (score < threshold) is the favorable outcome."
)


def render_header_and_banner():
    """Renders the top banner on every page."""
    st.markdown(
        f"""
        <div style="background-color: #f0f2f6; border-left: 5px solid #1f77b4; padding: 10px 15px; border-radius: 4px; margin-bottom: 20px;">
            <span style="font-weight: 600; color: #1e293b;">{BANNER_TEXT}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    """Renders the standard evaluation and policy definition footer."""
    st.markdown("---")
    st.caption(f"**Policy & Evaluation Notice**: {FOOTER_TEXT}")


@st.cache_data
def get_config() -> Dict[str, Any]:
    return load_config()


@st.cache_data
def load_fairness_json() -> Dict[str, Any]:
    path = _PROJECT_ROOT / "reports" / "metrics" / "fairness.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_model_a_json() -> Dict[str, Any]:
    path = _PROJECT_ROOT / "reports" / "metrics" / "model_a.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_model_b_json() -> Dict[str, Any]:
    path = _PROJECT_ROOT / "reports" / "metrics" / "model_b.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_a_vs_b_json() -> Dict[str, Any]:
    path = _PROJECT_ROOT / "reports" / "metrics" / "a_vs_b.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_stability_json() -> Dict[str, Any]:
    path = _PROJECT_ROOT / "reports" / "metrics" / "extras_stability.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_flips_json() -> Dict[str, Any]:
    path = _PROJECT_ROOT / "reports" / "metrics" / "extras_flips.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_base_rates_csv() -> pd.DataFrame:
    path = _PROJECT_ROOT / "reports" / "tables" / "base_rates.csv"
    return pd.read_csv(path)


@st.cache_data
def load_frontier_csv() -> pd.DataFrame:
    path = _PROJECT_ROOT / "reports" / "tables" / "app_frontier.csv"
    return pd.read_csv(path)


@st.cache_data
def load_calibration_bins_csv() -> pd.DataFrame:
    path = _PROJECT_ROOT / "reports" / "tables" / "extras_calibration_bins.csv"
    return pd.read_csv(path)


@st.cache_data
def load_calibration_json() -> Dict[str, Any]:
    path = _PROJECT_ROOT / "reports" / "metrics" / "extras_calibration.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_auc_by_group_json() -> Dict[str, Any]:
    path = _PROJECT_ROOT / "reports" / "metrics" / "extras_auc_by_group.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_proxy_json() -> Dict[str, Any]:
    path = _PROJECT_ROOT / "reports" / "metrics" / "proxy.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_shap_summary_json() -> Dict[str, Any]:
    path = _PROJECT_ROOT / "reports" / "metrics" / "shap_summary.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_mitigation_control_json() -> Dict[str, Any]:
    path = _PROJECT_ROOT / "reports" / "metrics" / "mitigation_control.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_ablation_json() -> Dict[str, Any]:
    path = _PROJECT_ROOT / "reports" / "metrics" / "extras_ablation.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_shap_notes_md() -> str:
    path = _PROJECT_ROOT / "reports" / "shap_notes.md"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@st.cache_data
def load_limitations_md() -> str:
    path = _PROJECT_ROOT / "docs" / "limitations.md"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@st.cache_data
def load_shap_sample_parquet() -> pd.DataFrame:
    path = _PROJECT_ROOT / "reports" / "predictions" / "extras" / "app_shap_sample.parquet"
    return pd.read_parquet(path)



@st.cache_data
def load_val_scores() -> Tuple[np.ndarray, np.ndarray]:
    cfg = get_config()
    pdir = _PROJECT_ROOT / cfg["paths"]["predictions"]
    va = pd.read_parquet(pdir / "model_a_val.parquet")["score"].values
    vb = pd.read_parquet(pdir / "model_b_val.parquet")["score"].values
    return va, vb


@st.cache_data
def load_test_applicants_data() -> pd.DataFrame:
    cfg = get_config()
    pdir = _PROJECT_ROOT / cfg["paths"]["predictions"]
    procdir = _PROJECT_ROOT / cfg["paths"]["processed"]
    splitsdir = _PROJECT_ROOT / cfg["paths"]["splits"]

    test_a = pd.read_parquet(pdir / "model_a_test.parquet")
    test_b = pd.read_parquet(pdir / "model_b_test.parquet")
    eval_attrs = pd.read_parquet(procdir / "eval_attrs.parquet")

    train_ids = pd.read_csv(splitsdir / "train_ids.csv", usecols=["SK_ID_CURR"])
    train_eval = eval_attrs[eval_attrs["SK_ID_CURR"].isin(set(train_ids["SK_ID_CURR"]))]
    train_median_age = round(float(train_eval["age_years"].median()), 2)

    df = test_a[["SK_ID_CURR", "segment", "TARGET"]].copy()
    df["score_a"] = test_a["score"].values
    df["score_b"] = test_b["score"].values
    df = df.merge(eval_attrs[["SK_ID_CURR", "gender", "age_years", "age_bin"]], on="SK_ID_CURR", how="left")
    df["binary_age"] = np.where(df["age_years"] >= train_median_age, ">=median", "<median")
    return df


@st.cache_data
def get_display_sample_for_flips(segment_filter: str = "all") -> pd.DataFrame:
    """Samples applicants for flip explorer: keeps ALL defaulters and samples non-defaulters up to 8,000 points."""
    cfg = get_config()
    seed = int(cfg.get("seed", 42))
    df = load_test_applicants_data()

    if segment_filter != "all":
        df = df[df["segment"] == segment_filter].copy()

    defaulters = df[df["TARGET"] == 1]
    non_defaulters = df[df["TARGET"] == 0]

    n_def = len(defaulters)
    n_remain = max(0, 8000 - n_def)

    if len(non_defaulters) > n_remain:
        sampled_non = non_defaulters.sample(n=n_remain, random_state=seed)
    else:
        sampled_non = non_defaulters

    sample_df = pd.concat([defaulters, sampled_non]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sample_df


def compute_live_policy(
    model_choice: str,
    attribute_choice: str,
    decline_rate: float,
) -> Dict[str, Any]:
    """Computes live policy metrics using existing fairness functions."""
    val_a, val_b = load_val_scores()
    test_df = load_test_applicants_data()

    if model_choice in ["Model A", "model_a"]:
        val_scores = val_a
        score_col = "score_a"
    else:
        val_scores = val_b
        score_col = "score_b"

    thresh = compute_threshold_from_validation(val_scores, decline_rate=decline_rate)
    scores = test_df[score_col].values
    decl, app = apply_decision_policy(scores, thresh)

    temp = test_df.copy()
    temp["decline"] = decl

    # Group metrics
    if attribute_choice == "gender":
        clean_df = temp[temp["gender"] != "XNA"]
        groups = ["F", "M"]
        col = "gender"
    else:
        clean_df = temp
        groups = ["<median", ">=median"]
        col = "binary_age"

    group_stats = {}
    for g in groups:
        sub = clean_df[clean_df[col] == g]
        group_stats[g] = compute_group_metrics(sub["TARGET"].values, sub["decline"].values)

    gaps = compute_fairness_gaps_hand(group_stats)
    overall_stats = compute_group_metrics(test_df["TARGET"].values, decl)

    return {
        "threshold": thresh,
        "realized_decline_rate": float(np.mean(decl)),
        "group_stats": group_stats,
        "gaps": gaps,
        "overall": overall_stats,
    }


def check_consistency_at_15() -> Dict[str, Any]:
    """Computes max difference between live values and fairness.json at 15% decline rate."""
    fair_json = load_fairness_json()
    max_diff = 0.0
    details = {}

    for m_choice, m_key in [("Model A", "model_a"), ("Model B", "model_b")]:
        for attr_choice, attr_key in [("gender", "gender"), ("binary age", "binary_age")]:
            res = compute_live_policy(m_choice, attr_choice, 0.15)
            thresh_key = "threshold_a" if m_key == "model_a" else "threshold_b"
            real_key = "realized_test_decline_a" if m_key == "model_a" else "realized_test_decline_b"

            d_thresh = abs(res["threshold"] - fair_json["policy"][thresh_key])
            d_real = abs(res["realized_decline_rate"] - fair_json["policy"][real_key])

            gaps_json = fair_json["overall"][attr_key][m_key]["gaps"]
            d_dp_diff = abs(res["gaps"]["dp_difference"] - gaps_json["dp_difference"])
            d_dp_ratio = abs(res["gaps"]["dp_ratio"] - gaps_json["dp_ratio"])
            d_eo_diff = abs(res["gaps"]["eo_difference"] - gaps_json["eo_difference"])

            pair_max = max(d_thresh, d_real, d_dp_diff, d_dp_ratio, d_eo_diff)
            max_diff = max(max_diff, pair_max)
            details[f"{m_choice}_{attr_choice}"] = {
                "d_thresh": d_thresh,
                "d_real": d_real,
                "d_dp_diff": d_dp_diff,
                "d_dp_ratio": d_dp_ratio,
                "d_eo_diff": d_eo_diff,
                "max": pair_max,
            }

    return {"max_diff": max_diff, "details": details, "passed": bool(max_diff <= 1e-6)}
