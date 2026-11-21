"""Calibration by group analysis for TrustScore (Phase 7 Part C).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config, set_seed
from src.extras.common import compute_auc_ci


def compute_calibration_bins(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile"
) -> Tuple[List[Dict[str, Any]], float]:
    """Compute 10-bin reliability table and Expected Calibration Error (ECE).
    
    Using quantile strategy to ensure every bin contains data in skewed credit datasets.
    """
    n = len(y_true)
    if n == 0:
        return [], float("nan")

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(y_score, quantiles)
        # Ensure distinct edges
        bin_edges[0] = -1e-8
        bin_edges[-1] = 1.0 + 1e-8
        for i in range(1, len(bin_edges) - 1):
            if bin_edges[i] <= bin_edges[i - 1]:
                bin_edges[i] = bin_edges[i - 1] + 1e-6
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)

    bin_assignments = np.digitize(y_score, bin_edges) - 1
    # Clip to valid bin range
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    bins_info = []
    ece = 0.0

    for b in range(n_bins):
        mask = (bin_assignments == b)
        cnt = int(mask.sum())
        if cnt > 0:
            mean_pred = float(np.mean(y_score[mask]))
            obs_rate = float(np.mean(y_true[mask]))
            bin_lower = float(bin_edges[b])
            bin_upper = float(bin_edges[b + 1])
            ece += (cnt / n) * abs(mean_pred - obs_rate)
        else:
            mean_pred = float("nan")
            obs_rate = float("nan")
            bin_lower = float(bin_edges[b])
            bin_upper = float(bin_edges[b + 1])

        bins_info.append({
            "bin_idx": b + 1,
            "bin_lower": round(bin_lower, 6),
            "bin_upper": round(bin_upper, 6),
            "count": cnt,
            "mean_predicted": round(mean_pred, 6) if not np.isnan(mean_pred) else None,
            "observed_rate": round(obs_rate, 6) if not np.isnan(obs_rate) else None,
        })

    return bins_info, round(float(ece), 6)


def bootstrap_ece_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
    n_bootstraps: int = 200,
    seed: int = 42,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """Compute 95% bootstrap CI for ECE."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_eces = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        sample_y = y_true[idx]
        sample_s = y_score[idx]
        _, ece = compute_calibration_bins(sample_y, sample_s, n_bins=n_bins, strategy="quantile")
        if not np.isnan(ece):
            boot_eces.append(ece)

    if len(boot_eces) == 0:
        return float("nan"), float("nan")

    lower = float(np.percentile(boot_eces, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_eces, 100 * (1.0 - alpha / 2.0)))
    return round(lower, 6), round(upper, 6)


def run_calibration() -> None:
    config = load_config()
    seed = set_seed(config=config)
    bootstrap_n = int(config.get("bootstrap_n", 200))

    out_metrics_dir = _PROJECT_ROOT / "reports" / "metrics"
    out_tables_dir = _PROJECT_ROOT / "reports" / "tables"
    out_figures_dir = _PROJECT_ROOT / "reports" / "figures"
    processed_dir = _PROJECT_ROOT / config["paths"]["processed"]
    preds_dir = _PROJECT_ROOT / "reports" / "predictions"

    # Load test predictions for Model A and Model B
    df_pred_a = pd.read_parquet(preds_dir / "model_a_test.parquet")
    df_pred_b = pd.read_parquet(preds_dir / "model_b_test.parquet")

    # Load eval_attrs
    eval_attrs = pd.read_parquet(processed_dir / "eval_attrs.parquet")
    test_eval = df_pred_a.merge(eval_attrs, on="SK_ID_CURR", how="left")
    
    # Binary age definition (< 43.12 vs >= 43.12)
    median_age = 43.12
    test_eval["binary_age"] = np.where(test_eval["age_years"] < median_age, "below_median", "above_median")

    y_test = test_eval["TARGET"].values
    score_a = df_pred_a["score"].values
    score_b = df_pred_b["score"].values

    models = {
        "model_a": {"name": "Model A (Bureau)", "score": score_a},
        "model_b": {"name": "Model B (Behavioral)", "score": score_b},
    }

    # Group definitions
    # Filter gender XNA for gender analyses
    gender_valid_mask = (test_eval["gender_xna"] == False).values

    eval_slices = [
        ("overall", "all", np.ones(len(y_test), dtype=bool)),
        ("gender", "M", (test_eval["gender"] == "M").values & gender_valid_mask),
        ("gender", "F", (test_eval["gender"] == "F").values & gender_valid_mask),
        ("binary_age", "below_median", (test_eval["binary_age"] == "below_median").values),
        ("binary_age", "above_median", (test_eval["binary_age"] == "above_median").values),
        ("segment", "full_file", (test_eval["segment"] == "full_file").values),
        ("segment", "thin_file", (test_eval["segment"] == "thin_file").values),
        ("segment", "no_file", (test_eval["segment"] == "no_file").values),
    ]

    metrics_res = {}
    table_rows = []

    for m_key, m_info in models.items():
        s = m_info["score"]
        m_dict = {
            "model_name": m_info["name"],
            "overall_brier_score": round(float(brier_score_loss(y_test, s)), 6),
            "groups": {},
        }

        for grp_type, grp_val, mask in eval_slices:
            y_sub = y_test[mask]
            s_sub = s[mask]
            n_sub = len(y_sub)
            if n_sub == 0:
                continue

            brier = float(brier_score_loss(y_sub, s_sub))
            mean_pred = float(np.mean(s_sub))
            obs_rate = float(np.mean(y_sub))
            
            # Calibration bins & ECE
            bins, ece = compute_calibration_bins(y_sub, s_sub, n_bins=10, strategy="quantile")
            ece_lower, ece_upper = bootstrap_ece_ci(y_sub, s_sub, n_bins=10, n_bootstraps=bootstrap_n, seed=seed)

            # AUC with CI (skip no_file if degenerate)
            if len(np.unique(y_sub)) >= 2:
                auc, auc_lower, auc_upper = compute_auc_ci(y_sub, s_sub, n_bootstraps=bootstrap_n, seed=seed)
            else:
                auc, auc_lower, auc_upper = float("nan"), float("nan"), float("nan")

            m_dict["groups"][f"{grp_type}:{grp_val}"] = {
                "group_type": grp_type,
                "group_value": grp_val,
                "count": n_sub,
                "brier_score": round(brier, 6),
                "mean_predicted": round(mean_pred, 6),
                "observed_rate": round(obs_rate, 6),
                "ratio_pred_obs": round(mean_pred / obs_rate, 6) if obs_rate > 0 else float("nan"),
                "diff_pred_obs": round(mean_pred - obs_rate, 6),
                "ece": ece,
                "ece_ci_lower": ece_lower,
                "ece_ci_upper": ece_upper,
                "auc": auc,
                "auc_ci_lower": auc_lower,
                "auc_ci_upper": auc_upper,
            }

            # Add to bins table
            for b in bins:
                table_rows.append({
                    "model": m_key,
                    "group_type": grp_type,
                    "group_value": grp_val,
                    "bin_idx": b["bin_idx"],
                    "bin_lower": b["bin_lower"],
                    "bin_upper": b["bin_upper"],
                    "count": b["count"],
                    "mean_predicted": b["mean_predicted"],
                    "observed_rate": b["observed_rate"],
                })

        metrics_res[m_key] = m_dict

    # Save metrics JSON
    with open(out_metrics_dir / "extras_calibration.json", "w", encoding="utf-8") as f:
        json.dump(metrics_res, f, indent=2)
    print(f"[calibration] Saved {out_metrics_dir / 'extras_calibration.json'}")

    # Save bins table CSV
    df_bins = pd.DataFrame(table_rows)
    df_bins.to_csv(out_tables_dir / "extras_calibration_bins.csv", index=False)
    print(f"[calibration] Saved {out_tables_dir / 'extras_calibration_bins.csv'}")

    # Generate Figure: Reliability curves for gender and binary age (Model B)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Subplot 1: Gender
    for grp_val, color, marker in [("M", "#1f77b4", "o"), ("F", "#e377c2", "s")]:
        sub_df = df_bins[(df_bins["model"] == "model_b") & (df_bins["group_type"] == "gender") & (df_bins["group_value"] == grp_val)]
        ax1.plot(sub_df["mean_predicted"], sub_df["observed_rate"], marker=marker, label=f"Model B ({grp_val})", color=color, linewidth=1.5)

    # Reference diagonal
    max_val_1 = max(df_bins[(df_bins["model"] == "model_b") & (df_bins["group_type"] == "gender")]["mean_predicted"].max(), 0.35)
    ax1.plot([0, max_val_1], [0, max_val_1], "k--", alpha=0.6, label="Perfect calibration")
    ax1.set_xlabel("Mean Predicted Default Probability", fontsize=10)
    ax1.set_ylabel("Observed Default Rate", fontsize=10)
    ax1.set_title("Reliability Curve by Gender (Model B)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9)

    # Subplot 2: Binary Age
    for grp_val, color, marker in [("below_median", "#ff7f0e", "^"), ("above_median", "#2ca02c", "v")]:
        sub_df = df_bins[(df_bins["model"] == "model_b") & (df_bins["group_type"] == "binary_age") & (df_bins["group_value"] == grp_val)]
        label_str = "< 43.1y" if grp_val == "below_median" else ">= 43.1y"
        ax2.plot(sub_df["mean_predicted"], sub_df["observed_rate"], marker=marker, label=f"Model B ({label_str})", color=color, linewidth=1.5)

    max_val_2 = max(df_bins[(df_bins["model"] == "model_b") & (df_bins["group_type"] == "binary_age")]["mean_predicted"].max(), 0.35)
    ax2.plot([0, max_val_2], [0, max_val_2], "k--", alpha=0.6, label="Perfect calibration")
    ax2.set_xlabel("Mean Predicted Default Probability", fontsize=10)
    ax2.set_ylabel("Observed Default Rate", fontsize=10)
    ax2.set_title("Reliability Curve by Binary Age (Model B)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_figures_dir / "extras_calibration.png")
    plt.close(fig)
    print(f"[calibration] Saved {out_figures_dir / 'extras_calibration.png'}")


if __name__ == "__main__":
    run_calibration()
