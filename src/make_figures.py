"""Generate publication-ready poster figures for TrustScore (Phase 6).

Generates:
a) auc_by_segment.png: Grouped bars of test AUC for Model A vs Model B across segments with 95% CIs
   and explicit low-precision callout for no_file (31 defaults).
b) fairness_gaps.png: Grouped bars comparing Demographic Parity and Equalized Odds differences
   between Model A and Model B under the 15% policy with 95% CIs.
c) mitigation_matched.png: Side-by-side comparison of mitigated policy vs matched single-threshold control
   on Equalized Odds difference and Balanced Accuracy across all 4 runs with paired differences.

All figures saved to reports/figures/ at 300 DPI, reading numbers strictly from JSON/CSV files.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Headless matplotlib configuration
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_figures():
    metrics_dir = _PROJECT_ROOT / "reports" / "metrics"
    tables_dir = _PROJECT_ROOT / "reports" / "tables"
    figures_dir = _PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "model_a.json", "r") as f:
        ma = json.load(f)
    with open(metrics_dir / "model_b.json", "r") as f:
        mb = json.load(f)
    with open(metrics_dir / "a_vs_b.json", "r") as f:
        avsb = json.load(f)
    with open(metrics_dir / "fairness.json", "r") as f:
        fair = json.load(f)
    with open(metrics_dir / "mitigation_control.json", "r") as f:
        mit_ctrl = json.load(f)

    fair_gaps_df = pd.read_csv(tables_dir / "fairness_gaps.csv")
    base_rates_df = pd.read_csv(tables_dir / "base_rates.csv")

    no_file_defaults = int(base_rates_df[(base_rates_df["dimension"] == "segment") & (base_rates_df["group"] == "no_file")]["defaults"].iloc[0])

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
    })

    # =========================================================================
    # Figure A: auc_by_segment.png
    # =========================================================================
    print("[make_figures] Generating auc_by_segment.png...")
    segments = ["Overall", "full_file", "thin_file", "no_file"]
    
    # Extract point estimates and CIs
    a_aucs, a_err_low, a_err_high = [], [], []
    b_aucs, b_err_low, b_err_high = [], [], []

    # Overall
    a_ov = ma["test_results"]["overall"]
    b_ov = mb["test_results"]["overall"]
    a_aucs.append(a_ov["auc"])
    a_err_low.append(a_ov["auc"] - a_ov["ci_95_lower"])
    a_err_high.append(a_ov["ci_95_upper"] - a_ov["auc"])

    b_aucs.append(b_ov["auc"])
    b_err_low.append(b_ov["auc"] - b_ov["ci_95_lower"])
    b_err_high.append(b_ov["ci_95_upper"] - b_ov["auc"])

    for s in ["full_file", "thin_file", "no_file"]:
        a_s = ma["test_results"]["by_segment"][s]
        b_s = mb["test_results"]["by_segment"][s]
        a_aucs.append(a_s["auc"])
        a_err_low.append(a_s["auc"] - a_s["ci_95_lower"])
        a_err_high.append(a_s["ci_95_upper"] - a_s["auc"])

        b_aucs.append(b_s["auc"])
        b_err_low.append(b_s["auc"] - b_s["ci_95_lower"])
        b_err_high.append(b_s["ci_95_upper"] - b_s["auc"])

    x = np.arange(len(segments))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars1 = ax.bar(x - width/2, a_aucs, width, yerr=[a_err_low, a_err_high], capsize=4, label="Model A (Bureau Baseline)", color="#4c72b0", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, b_aucs, width, yerr=[b_err_low, b_err_high], capsize=4, label="Model B (TrustScore)", color="#55a868", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Test ROC-AUC by Segment: Model A vs Model B (with 95% Bootstrap CIs)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Overall\n(All Test)", "full_file\n(Bureau+Lender)", "thin_file\n(Lender Only)", "no_file\n(No Prior History)"])
    ax.set_ylim(0.55, 0.82)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    # Annotate values and low precision
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 14), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 14), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.annotate(
        f"* Low precision: no_file test split has only {no_file_defaults} default events (wide CIs)",
        xy=(3, 0.56), xytext=(2.2, 0.58),
        arrowprops=dict(facecolor="red", arrowstyle="->", lw=1),
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffeedd", ec="red", lw=1),
        fontsize=8,
    )

    fig.tight_layout()
    fig_a_path = figures_dir / "auc_by_segment.png"
    plt.savefig(fig_a_path, dpi=300)
    plt.close()

    # =========================================================================
    # Figure B: fairness_gaps.png
    # =========================================================================
    print("[make_figures] Generating fairness_gaps.png...")
    # Gender DP & EO, Binary Age DP & EO
    attrs = ["Gender", "Binary Age"]
    g_row = fair_gaps_df[fair_gaps_df["attribute"] == "gender"].iloc[0]
    a_row = fair_gaps_df[fair_gaps_df["attribute"] == "binary_age"].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    # Subplot 1: Demographic Parity Difference
    g_dp = fair_gaps_df[(fair_gaps_df["scope"] == "overall") & (fair_gaps_df["attribute"] == "gender") & (fair_gaps_df["metric"] == "dp_difference")].iloc[0]
    a_dp = fair_gaps_df[(fair_gaps_df["scope"] == "overall") & (fair_gaps_df["attribute"] == "binary_age") & (fair_gaps_df["metric"] == "dp_difference")].iloc[0]

    dp_a = [g_dp["model_a_gap"], a_dp["model_a_gap"]]
    dp_a_err = [
        [g_dp["model_a_gap"] - g_dp["model_a_ci_lower"], a_dp["model_a_gap"] - a_dp["model_a_ci_lower"]],
        [g_dp["model_a_ci_upper"] - g_dp["model_a_gap"], a_dp["model_a_ci_upper"] - a_dp["model_a_gap"]],
    ]
    dp_b = [g_dp["model_b_gap"], a_dp["model_b_gap"]]
    dp_b_err = [
        [g_dp["model_b_gap"] - g_dp["model_b_ci_lower"], a_dp["model_b_gap"] - a_dp["model_b_ci_lower"]],
        [g_dp["model_b_ci_upper"] - g_dp["model_b_gap"], a_dp["model_b_ci_upper"] - a_dp["model_b_gap"]],
    ]

    x = np.arange(len(attrs))
    ax1.bar(x - width/2, dp_a, width, yerr=dp_a_err, capsize=4, label="Model A", color="#4c72b0", edgecolor="black", lw=0.5)
    ax1.bar(x + width/2, dp_b, width, yerr=dp_b_err, capsize=4, label="Model B", color="#55a868", edgecolor="black", lw=0.5)
    ax1.set_title("Demographic Parity Difference\n(Approval Rate Gap; Lower is Better)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(attrs)
    ax1.set_ylabel("Gap (Absolute Difference)")
    ax1.set_ylim(0, 0.22)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # Subplot 2: Equalized Odds Difference
    g_eo = fair_gaps_df[(fair_gaps_df["scope"] == "overall") & (fair_gaps_df["attribute"] == "gender") & (fair_gaps_df["metric"] == "eo_difference")].iloc[0]
    a_eo = fair_gaps_df[(fair_gaps_df["scope"] == "overall") & (fair_gaps_df["attribute"] == "binary_age") & (fair_gaps_df["metric"] == "eo_difference")].iloc[0]

    eo_a = [g_eo["model_a_gap"], a_eo["model_a_gap"]]
    eo_a_err = [
        [g_eo["model_a_gap"] - g_eo["model_a_ci_lower"], a_eo["model_a_gap"] - a_eo["model_a_ci_lower"]],
        [g_eo["model_a_ci_upper"] - g_eo["model_a_gap"], a_eo["model_a_ci_upper"] - a_eo["model_a_gap"]],
    ]
    eo_b = [g_eo["model_b_gap"], a_eo["model_b_gap"]]
    eo_b_err = [
        [g_eo["model_b_gap"] - g_eo["model_b_ci_lower"], a_eo["model_b_gap"] - a_eo["model_b_ci_lower"]],
        [g_eo["model_b_ci_upper"] - g_eo["model_b_gap"], a_eo["model_b_ci_upper"] - a_eo["model_b_gap"]],
    ]

    ax2.bar(x - width/2, eo_a, width, yerr=eo_a_err, capsize=4, label="Model A", color="#4c72b0", edgecolor="black", lw=0.5)
    ax2.bar(x + width/2, eo_b, width, yerr=eo_b_err, capsize=4, label="Model B", color="#55a868", edgecolor="black", lw=0.5)
    ax2.set_title("Equalized Odds Difference\n(max(TPR gap, FPR gap); Lower is Better)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(attrs)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")

    fig.suptitle("Fairness Disparity Metrics at Fixed 15% Validation Decline Policy (Test Split)", fontweight="bold")
    fig.tight_layout()
    fig_b_path = figures_dir / "fairness_gaps.png"
    plt.savefig(fig_b_path, dpi=300)
    plt.close()

    # =========================================================================
    # Figure C: mitigation_matched.png
    # =========================================================================
    print("[make_figures] Generating mitigation_matched.png...")
    runs = [
        ("gender_equalized_odds", "Gender\n(EO)"),
        ("gender_demographic_parity", "Gender\n(DP)"),
        ("binary_age_equalized_odds", "Age\n(EO)"),
        ("binary_age_demographic_parity", "Age\n(DP)"),
    ]

    eo_mit, eo_ctrl = [], []
    eo_mit_err, eo_ctrl_err = [[], []], [[], []]

    ba_mit, ba_ctrl = [], []
    ba_mit_err, ba_ctrl_err = [[], []], [[], []]

    for r_key, _ in runs:
        r_info = mit_ctrl["runs"][r_key]
        m = r_info["mitigated"]
        c = r_info["control"]
        m_ci = m["confidence_intervals_95"]
        c_ci = c["confidence_intervals_95"]

        eo_mit.append(m["equalized_odds_difference"])
        eo_mit_err[0].append(m["equalized_odds_difference"] - m_ci["eo_difference"][0])
        eo_mit_err[1].append(m_ci["eo_difference"][1] - m["equalized_odds_difference"])

        eo_ctrl.append(c["equalized_odds_difference"])
        eo_ctrl_err[0].append(c["equalized_odds_difference"] - c_ci["eo_difference"][0])
        eo_ctrl_err[1].append(c_ci["eo_difference"][1] - c["equalized_odds_difference"])

        ba_mit.append(m["balanced_accuracy"])
        ba_mit_err[0].append(m["balanced_accuracy"] - m_ci["balanced_accuracy"][0])
        ba_mit_err[1].append(m_ci["balanced_accuracy"][1] - m["balanced_accuracy"])

        ba_ctrl.append(c["balanced_accuracy"])
        ba_ctrl_err[0].append(c["balanced_accuracy"] - c_ci["balanced_accuracy"][0])
        ba_ctrl_err[1].append(c_ci["balanced_accuracy"][1] - c["balanced_accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(runs))

    # Panel 1: Equalized Odds Difference
    ax1.bar(x - width/2, eo_ctrl, width, yerr=eo_ctrl_err, capsize=4, label="Matched Single-Threshold Control", color="#c44e52", edgecolor="black", lw=0.5)
    ax1.bar(x + width/2, eo_mit, width, yerr=eo_mit_err, capsize=4, label="Mitigated (ThresholdOptimizer)", color="#8172b3", edgecolor="black", lw=0.5)
    ax1.set_title("Equalized Odds Difference (Lower is Better)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([r[1] for r in runs])
    ax1.set_ylabel("Equalized Odds Difference")
    ax1.set_ylim(0, 0.22)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Panel 2: Balanced Accuracy
    ax2.bar(x - width/2, ba_ctrl, width, yerr=ba_ctrl_err, capsize=4, label="Matched Single-Threshold Control", color="#c44e52", edgecolor="black", lw=0.5)
    ax2.bar(x + width/2, ba_mit, width, yerr=ba_mit_err, capsize=4, label="Mitigated (ThresholdOptimizer)", color="#8172b3", edgecolor="black", lw=0.5)
    ax2.set_title("Balanced Accuracy (Higher is Better)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([r[1] for r in runs])
    ax2.set_ylabel("Balanced Accuracy")
    ax2.set_ylim(0.65, 0.72)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    fig.suptitle(
        "Fairness Mitigation vs Matched-Decline-Rate Control (Test Split)\n"
        "*Experimental simulation: ThresholdOptimizer requires protected attributes at decision time*",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig_c_path = figures_dir / "mitigation_matched.png"
    plt.savefig(fig_c_path, dpi=300)
    plt.close()

    print("[make_figures] All three publication figures generated successfully.")


if __name__ == "__main__":
    make_figures()
