"""Phase 7 Part A: Winners and Losers (Decision Flips).

Cross-tabulates Model A vs Model B decisions on the TEST split under the 15% decline policy:
- rescued: declined by A, approved by B, TARGET == 0
- let-in: declined by A, approved by B, TARGET == 1
- newly declined good: approved by A, declined by B, TARGET == 0
- newly caught: approved by A, declined by B, TARGET == 1
- net: (rescued + newly caught) - (let-in + newly declined good)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# Headless matplotlib configuration
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config, set_seed
from src.fairness.policy import compute_threshold_from_validation, apply_decision_policy


def compute_flips_metrics(
    df: pd.DataFrame,
    bootstrap_n: int = 200,
    seed: int = 42
) -> Dict[str, Any]:
    """Compute flip counts, shares, and bootstrap CI for net."""
    n = len(df)
    if n == 0:
        return {
            "n": 0, "unreliable": True,
            "counts": {}, "shares": {},
            "net": 0, "net_ci_95": [0, 0]
        }

    dec_a = df["decline_a"].values
    dec_b = df["decline_b"].values
    y = df["TARGET"].values

    rescued = int(np.sum((dec_a == 1) & (dec_b == 0) & (y == 0)))
    let_in = int(np.sum((dec_a == 1) & (dec_b == 0) & (y == 1)))
    newly_declined_good = int(np.sum((dec_a == 0) & (dec_b == 1) & (y == 0)))
    newly_caught = int(np.sum((dec_a == 0) & (dec_b == 1) & (y == 1)))

    both_declined_default = int(np.sum((dec_a == 1) & (dec_b == 1) & (y == 1)))
    both_declined_good = int(np.sum((dec_a == 1) & (dec_b == 1) & (y == 0)))
    both_approved_good = int(np.sum((dec_a == 0) & (dec_b == 0) & (y == 0)))
    both_approved_default = int(np.sum((dec_a == 0) & (dec_b == 0) & (y == 1)))

    total_flips = rescued + let_in + newly_declined_good + newly_caught
    net = (rescued + newly_caught) - (let_in + newly_declined_good)

    # Bootstrap for net CI
    rng = np.random.RandomState(seed)
    boot_nets = []
    for _ in range(bootstrap_n):
        idx = rng.randint(0, n, size=n)
        b_dec_a = dec_a[idx]
        b_dec_b = dec_b[idx]
        b_y = y[idx]
        b_res = np.sum((b_dec_a == 1) & (b_dec_b == 0) & (b_y == 0))
        b_let = np.sum((b_dec_a == 1) & (b_dec_b == 0) & (b_y == 1))
        b_dec_g = np.sum((b_dec_a == 0) & (b_dec_b == 1) & (b_y == 0))
        b_cgt = np.sum((b_dec_a == 0) & (b_dec_b == 1) & (b_y == 1))
        boot_nets.append((b_res + b_cgt) - (b_let + b_dec_g))

    ci_lower = float(np.percentile(boot_nets, 2.5))
    ci_upper = float(np.percentile(boot_nets, 97.5))

    return {
        "n": n,
        "unreliable": n < 100,
        "counts": {
            "rescued": rescued,
            "let_in": let_in,
            "newly_declined_good": newly_declined_good,
            "newly_caught": newly_caught,
            "both_declined_default": both_declined_default,
            "both_declined_good": both_declined_good,
            "both_approved_good": both_approved_good,
            "both_approved_default": both_approved_default,
            "total_flips": total_flips,
        },
        "shares_of_group": {
            "rescued": round(rescued / n, 6),
            "let_in": round(let_in / n, 6),
            "newly_declined_good": round(newly_declined_good / n, 6),
            "newly_caught": round(newly_caught / n, 6),
            "total_flips": round(total_flips / n, 6),
        },
        "shares_of_flips": {
            "rescued": round(rescued / total_flips, 6) if total_flips > 0 else 0.0,
            "let_in": round(let_in / total_flips, 6) if total_flips > 0 else 0.0,
            "newly_declined_good": round(newly_declined_good / total_flips, 6) if total_flips > 0 else 0.0,
            "newly_caught": round(newly_caught / total_flips, 6) if total_flips > 0 else 0.0,
        },
        "net": net,
        "net_ci_95": [round(ci_lower, 2), round(ci_upper, 2)],
    }


def run_flips_analysis(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Execute Part A winners and losers analysis."""
    if config is None:
        config = load_config()

    seed = set_seed(config=config)
    bootstrap_n = int(config.get("bootstrap_n", 200))
    decline_rate = float(config.get("decline_rate", 0.15))

    val_a = pd.read_parquet(_PROJECT_ROOT / "reports" / "predictions" / "model_a_val.parquet")
    val_b = pd.read_parquet(_PROJECT_ROOT / "reports" / "predictions" / "model_b_val.parquet")
    test_a = pd.read_parquet(_PROJECT_ROOT / "reports" / "predictions" / "model_a_test.parquet")
    test_b = pd.read_parquet(_PROJECT_ROOT / "reports" / "predictions" / "model_b_test.parquet")
    eval_attrs = pd.read_parquet(_PROJECT_ROOT / "data" / "processed" / "eval_attrs.parquet")

    # Policy thresholds on validation
    th_a = compute_threshold_from_validation(val_a["score"].values, decline_rate=decline_rate)
    th_b = compute_threshold_from_validation(val_b["score"].values, decline_rate=decline_rate)

    # Combine test data
    df = test_a[["SK_ID_CURR", "segment", "TARGET"]].copy()
    df["score_a"] = test_a["score"]
    df["score_b"] = test_b["score"]
    df["decline_a"] = (df["score_a"] >= th_a).astype(int)
    df["decline_b"] = (df["score_b"] >= th_b).astype(int)

    # Merge demographic attributes
    df = df.merge(eval_attrs[["SK_ID_CURR", "gender", "gender_xna", "age_years", "age_bin"]], on="SK_ID_CURR", how="left")

    train_median_age = 43.12
    df["binary_age"] = np.where(df["age_years"] < train_median_age, "<median", ">=median")

    # Define slice analyses
    slices_data: Dict[str, Dict[str, Any]] = {}
    table_rows: List[Dict[str, Any]] = []

    # 1. Overall
    ov = compute_flips_metrics(df, bootstrap_n=bootstrap_n, seed=seed)
    slices_data["overall"] = ov
    table_rows.append({
        "slice_type": "overall", "group": "all", "n": ov["n"],
        "rescued": ov["counts"]["rescued"], "let_in": ov["counts"]["let_in"],
        "newly_declined_good": ov["counts"]["newly_declined_good"], "newly_caught": ov["counts"]["newly_caught"],
        "total_flips": ov["counts"]["total_flips"], "net": ov["net"],
        "net_ci_lower": ov["net_ci_95"][0], "net_ci_upper": ov["net_ci_95"][1],
        "rescued_share": ov["shares_of_group"]["rescued"], "let_in_share": ov["shares_of_group"]["let_in"],
        "newly_declined_good_share": ov["shares_of_group"]["newly_declined_good"], "newly_caught_share": ov["shares_of_group"]["newly_caught"],
        "unreliable": ov["unreliable"]
    })

    # 2. Segments
    slices_data["segment"] = {}
    for seg in ["full_file", "thin_file", "no_file"]:
        seg_df = df[df["segment"] == seg]
        res = compute_flips_metrics(seg_df, bootstrap_n=bootstrap_n, seed=seed)
        slices_data["segment"][seg] = res
        table_rows.append({
            "slice_type": "segment", "group": seg, "n": res["n"],
            "rescued": res["counts"]["rescued"], "let_in": res["counts"]["let_in"],
            "newly_declined_good": res["counts"]["newly_declined_good"], "newly_caught": res["counts"]["newly_caught"],
            "total_flips": res["counts"]["total_flips"], "net": res["net"],
            "net_ci_lower": res["net_ci_95"][0], "net_ci_upper": res["net_ci_95"][1],
            "rescued_share": res["shares_of_group"]["rescued"], "let_in_share": res["shares_of_group"]["let_in"],
            "newly_declined_good_share": res["shares_of_group"]["newly_declined_good"], "newly_caught_share": res["shares_of_group"]["newly_caught"],
            "unreliable": res["unreliable"]
        })

    # 3. Gender (exclude XNA)
    slices_data["gender"] = {}
    for g in ["F", "M"]:
        g_df = df[(df["gender"] == g) & (~df["gender_xna"])]
        res = compute_flips_metrics(g_df, bootstrap_n=bootstrap_n, seed=seed)
        slices_data["gender"][g] = res
        table_rows.append({
            "slice_type": "gender", "group": g, "n": res["n"],
            "rescued": res["counts"]["rescued"], "let_in": res["counts"]["let_in"],
            "newly_declined_good": res["counts"]["newly_declined_good"], "newly_caught": res["counts"]["newly_caught"],
            "total_flips": res["counts"]["total_flips"], "net": res["net"],
            "net_ci_lower": res["net_ci_95"][0], "net_ci_upper": res["net_ci_95"][1],
            "rescued_share": res["shares_of_group"]["rescued"], "let_in_share": res["shares_of_group"]["let_in"],
            "newly_declined_good_share": res["shares_of_group"]["newly_declined_good"], "newly_caught_share": res["shares_of_group"]["newly_caught"],
            "unreliable": res["unreliable"]
        })

    # 4. Binary age
    slices_data["binary_age"] = {}
    for ba in ["<median", ">=median"]:
        ba_df = df[df["binary_age"] == ba]
        res = compute_flips_metrics(ba_df, bootstrap_n=bootstrap_n, seed=seed)
        slices_data["binary_age"][ba] = res
        table_rows.append({
            "slice_type": "binary_age", "group": ba, "n": res["n"],
            "rescued": res["counts"]["rescued"], "let_in": res["counts"]["let_in"],
            "newly_declined_good": res["counts"]["newly_declined_good"], "newly_caught": res["counts"]["newly_caught"],
            "total_flips": res["counts"]["total_flips"], "net": res["net"],
            "net_ci_lower": res["net_ci_95"][0], "net_ci_upper": res["net_ci_95"][1],
            "rescued_share": res["shares_of_group"]["rescued"], "let_in_share": res["shares_of_group"]["let_in"],
            "newly_declined_good_share": res["shares_of_group"]["newly_declined_good"], "newly_caught_share": res["shares_of_group"]["newly_caught"],
            "unreliable": res["unreliable"]
        })

    # 5. Four age bins
    slices_data["four_age_bins"] = {}
    for ab in ["<30", "30-44", "45-59", "60+"]:
        ab_df = df[df["age_bin"] == ab]
        res = compute_flips_metrics(ab_df, bootstrap_n=bootstrap_n, seed=seed)
        slices_data["four_age_bins"][ab] = res
        table_rows.append({
            "slice_type": "four_age_bins", "group": ab, "n": res["n"],
            "rescued": res["counts"]["rescued"], "let_in": res["counts"]["let_in"],
            "newly_declined_good": res["counts"]["newly_declined_good"], "newly_caught": res["counts"]["newly_caught"],
            "total_flips": res["counts"]["total_flips"], "net": res["net"],
            "net_ci_lower": res["net_ci_95"][0], "net_ci_upper": res["net_ci_95"][1],
            "rescued_share": res["shares_of_group"]["rescued"], "let_in_share": res["shares_of_group"]["let_in"],
            "newly_declined_good_share": res["shares_of_group"]["newly_declined_good"], "newly_caught_share": res["shares_of_group"]["newly_caught"],
            "unreliable": res["unreliable"]
        })

    output_payload = {
        "metadata": {
            "policy": "15% validation decline threshold",
            "threshold_a": round(th_a, 6),
            "threshold_b": round(th_b, 6),
            "test_sample_size": len(df),
            "train_median_age": train_median_age,
            "bootstrap_n": bootstrap_n,
            "definitions": {
                "rescued": "declined by A, approved by B, TARGET == 0 (non-default)",
                "let_in": "declined by A, approved by B, TARGET == 1 (default)",
                "newly_declined_good": "approved by A, declined by B, TARGET == 0 (non-default)",
                "newly_caught": "approved by A, declined by B, TARGET == 1 (default)",
                "net": "(rescued + newly_caught) - (let_in + newly_declined_good)"
            }
        },
        "slices": slices_data
    }

    # Save outputs
    json_path = _PROJECT_ROOT / "reports" / "metrics" / "extras_flips.json"
    csv_path = _PROJECT_ROOT / "reports" / "tables" / "extras_flips.csv"
    fig_path = _PROJECT_ROOT / "reports" / "figures" / "extras_flips.png"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"[extras_flips] Saved {json_path}")

    csv_df = pd.DataFrame(table_rows)
    csv_df.to_csv(csv_path, index=False)
    print(f"[extras_flips] Saved {csv_path}")

    # Generate figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Plot 1: Flip categories by segment
    segs = ["full_file", "thin_file", "no_file"]
    bar_width = 0.2
    x = np.arange(len(segs))

    rescued_vals = [slices_data["segment"][s]["counts"]["rescued"] for s in segs]
    newly_caught_vals = [slices_data["segment"][s]["counts"]["newly_caught"] for s in segs]
    let_in_vals = [slices_data["segment"][s]["counts"]["let_in"] for s in segs]
    declined_good_vals = [slices_data["segment"][s]["counts"]["newly_declined_good"] for s in segs]

    axes[0].bar(x - 1.5 * bar_width, rescued_vals, bar_width, label="Rescued (Good Appr)", color="#2ca02c")
    axes[0].bar(x - 0.5 * bar_width, newly_caught_vals, bar_width, label="Newly Caught (Bad Decl)", color="#1f77b4")
    axes[0].bar(x + 0.5 * bar_width, let_in_vals, bar_width, label="Let-In (Bad Appr)", color="#d62728")
    axes[0].bar(x + 1.5 * bar_width, declined_good_vals, bar_width, label="Newly Decl Good (Good Decl)", color="#ff7f0e")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(segs, fontweight="bold")
    axes[0].set_ylabel("Applicant Count")
    axes[0].set_title("Decision Flips by Segment (Model A -> Model B)", fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Plot 2: Net gain with 95% CIs across groups
    plot_groups = ["overall", "full_file", "thin_file", "no_file", "F", "M", "<median", ">=median"]
    net_vals = []
    ci_err_lower = []
    ci_err_upper = []

    for pg in plot_groups:
        if pg == "overall":
            item = slices_data["overall"]
        elif pg in slices_data["segment"]:
            item = slices_data["segment"][pg]
        elif pg in slices_data["gender"]:
            item = slices_data["gender"][pg]
        else:
            item = slices_data["binary_age"][pg]

        n_val = item["net"]
        ci_l = item["net_ci_95"][0]
        ci_u = item["net_ci_95"][1]
        net_vals.append(n_val)
        ci_err_lower.append(n_val - ci_l)
        ci_err_upper.append(ci_u - n_val)

    y_pos = np.arange(len(plot_groups))
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in net_vals]
    axes[1].barh(y_pos, net_vals, xerr=[ci_err_lower, ci_err_upper], color=colors, alpha=0.85, capsize=4)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(plot_groups, fontweight="bold")
    axes[1].set_xlabel("Net Favorable Flips: (Rescued + Caught) - (Let-In + Decl Good)")
    axes[1].set_title("Net Beneficial Changes with 95% Bootstrap CI", fontweight="bold")
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)

    fig.suptitle("Winners and Losers Analysis (Model A to Model B Transition under 15% Policy)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"[extras_flips] Saved {fig_path}")

    return output_payload


if __name__ == "__main__":
    run_flips_analysis()
