"""Shared visual components and styling for TrustScore Studio.

Phase 8 Stage 6: Badges, semicircular risk gauges, score histograms, and presenter view.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Optional, Tuple

# Set writable MPLCONFIGDIR for headless Windows execution
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Fixed model palette across all visual components
COLOR_MODEL_A = "#1f77b4"
COLOR_MODEL_B = "#ff7f0e"
COLOR_APPROVED = "#15803d"
COLOR_DECLINED = "#b91c1c"


@st.cache_data
def get_gauge_max_score() -> float:
    """Computes the maximum gauge scale as the 99th percentile of held-out test scores.
    
    Computed dynamically from saved test predictions.
    """
    pred_dir = _PROJECT_ROOT / "reports" / "predictions"
    pa = pd.read_parquet(pred_dir / "model_a_test.parquet")["score"]
    pb = pd.read_parquet(pred_dir / "model_b_test.parquet")["score"]
    max_99 = float(max(np.percentile(pa, 99), np.percentile(pb, 99)))
    return round(max_99, 4)


def decision_badge(decision: str):
    """Renders a large rounded badge (DECLINED in red or APPROVED in green)."""
    dec_upper = str(decision).strip().upper()
    is_declined = "DECLINE" in dec_upper

    if is_declined:
        bg_color = "#fee2e2"
        border_color = "#ef4444"
        text_color = COLOR_DECLINED
        label = "DECLINED"
    else:
        bg_color = "#dcfce7"
        border_color = "#22c55e"
        text_color = COLOR_APPROVED
        label = "APPROVED"

    badge_html = f"""
    <div style="
        display: inline-block;
        padding: 6px 16px;
        background-color: {bg_color};
        border: 2px solid {border_color};
        color: {text_color};
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.05em;
        text-align: center;
        margin: 4px 0;
    ">
        {label}
    </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)


def risk_gauge(probability: float, threshold: float, model_name: str = "Model B") -> plt.Figure:
    """Renders a semicircular risk gauge with decline zone and probability display.
    
    Scale runs from 0 to the 99th percentile of held-out test scores.
    """
    max_val = get_gauge_max_score()
    prob_clamped = min(max(float(probability), 0.0), max_val)
    thresh_clamped = min(max(float(threshold), 0.0), max_val)

    fig, ax = plt.subplots(figsize=(4.2, 2.4), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    # Orientation: semicircular top arc (pi on left to 0 on right)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    theta_0 = np.pi
    theta_thresh = np.pi * (1.0 - thresh_clamped / max_val)
    theta_prob = np.pi * (1.0 - prob_clamped / max_val)

    # Background approved arc (grey/slate)
    theta_appr = np.linspace(theta_thresh, theta_0, 80)
    ax.fill_between(theta_appr, 0.65, 1.0, color="#f1f5f9", alpha=0.9)

    # Background declined arc (soft red)
    theta_decl = np.linspace(0, theta_thresh, 80)
    ax.fill_between(theta_decl, 0.65, 1.0, color="#fee2e2", alpha=0.9)

    # Active fill up to current score
    active_color = COLOR_MODEL_B if "B" in model_name.upper() else COLOR_MODEL_A
    theta_active = np.linspace(theta_prob, theta_0, 80)
    ax.fill_between(theta_active, 0.70, 0.95, color=active_color, alpha=0.9)

    # Threshold dashed red boundary
    ax.plot([theta_thresh, theta_thresh], [0.60, 1.05], color="#dc2626", linewidth=2.5, linestyle="--")

    ax.set_ylim(0, 1.1)
    ax.axis("off")

    # Center text annotations
    plt.text(
        0, -0.15,
        f"{probability:.2%}",
        ha="center", va="center",
        fontsize=18, fontweight="bold",
        color="#0f172a",
        transform=ax.transAxes
    )
    plt.text(
        0, -0.32,
        f"15% Threshold: {threshold:.2%}",
        ha="center", va="center",
        fontsize=9, color="#64748b",
        transform=ax.transAxes
    )
    plt.text(
        0, 1.08,
        model_name,
        ha="center", va="center",
        fontsize=11, fontweight="bold",
        color=active_color,
        transform=ax.transAxes
    )

    plt.tight_layout()
    return fig


@st.cache_data
def load_test_score_distributions() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads held-out test predictions for Model A and Model B."""
    pred_dir = _PROJECT_ROOT / "reports" / "predictions"
    df_a = pd.read_parquet(pred_dir / "model_a_test.parquet")[["score"]].copy()
    df_b = pd.read_parquet(pred_dir / "model_b_test.parquet")[["score"]].copy()
    return df_a, df_b


def score_histogram(probability: float, threshold: float, model_name: str = "Model B") -> alt.Chart:
    """Renders score distribution chart on held-out test split with markers for score and threshold."""
    df_a, df_b = load_test_score_distributions()
    scores = df_b["score"] if "B" in model_name.upper() else df_a["score"]

    counts, bin_edges = np.histogram(scores, bins=35, range=(0.0, float(np.percentile(scores, 99.5))))
    hist_df = pd.DataFrame({
        "bin_start": bin_edges[:-1],
        "bin_end": bin_edges[1:],
        "bin_mid": (bin_edges[:-1] + bin_edges[1:]) / 2.0,
        "count": counts,
    })

    model_color = COLOR_MODEL_B if "B" in model_name.upper() else COLOR_MODEL_A

    base = alt.Chart(hist_df).mark_bar(color=model_color, opacity=0.45).encode(
        x=alt.X("bin_mid:Q", title=f"{model_name} Score (Default Probability)"),
        y=alt.Y("count:Q", title="Test Applicants"),
        tooltip=[alt.Tooltip("bin_mid:Q", title="Score"), alt.Tooltip("count:Q", title="Count")],
    )

    # Threshold rule
    thresh_df = pd.DataFrame([{"threshold": threshold}])
    rule_thresh = alt.Chart(thresh_df).mark_rule(color="#dc2626", strokeDash=[4, 4], size=2).encode(
        x="threshold:Q"
    )

    # Current score rule
    score_df = pd.DataFrame([{"current_score": probability}])
    rule_score = alt.Chart(score_df).mark_rule(color="#0f172a", size=2.5).encode(
        x="current_score:Q"
    )

    chart = (base + rule_thresh + rule_score).properties(height=180)
    return chart


def apply_presenter_styling(enabled: bool):
    """Applies clean card styling and optional Presenter View enlarged font and clean chrome."""
    base_css = """
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700;
    }
    .studio-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    </style>
    """
    st.markdown(base_css, unsafe_allow_html=True)

    if enabled:
        presenter_css = """
        <style>
        html, body, [class*="css"] {
            font-size: 17.5px !important;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
        st.markdown(presenter_css, unsafe_allow_html=True)
