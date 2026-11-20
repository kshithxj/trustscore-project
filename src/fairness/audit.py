"""Fairness audit and metric evaluation for TrustScore credit models.

DEFINITIONS:
- Positive class = default (TARGET = 1). Prediction = DECLINE (predicted 1 means decline).
- TPR = P(declined | default); FPR = P(declined | no default); FNR = P(approved | default), the costly error in credit.
- Approval (1 - decline) is the favorable outcome for approval rate, demographic parity difference and demographic parity ratio.
- equalized odds difference = the larger of the TPR gap and the FPR gap between groups (positive = default, prediction = decline).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config, set_seed
from src.fairness.policy import compute_threshold_from_validation, apply_decision_policy


def compute_group_metrics(
    y_true: np.ndarray,
    decline: np.ndarray
) -> Dict[str, float]:
    """Calculate approval rate, TPR, FPR, and FNR for a group."""
    n = len(y_true)
    if n == 0:
        return {"n": 0, "approval_rate": np.nan, "tpr": np.nan, "fpr": np.nan, "fnr": np.nan}

    approval = 1 - decline
    approval_rate = float(np.mean(approval))

    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)

    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    tpr = float(np.mean(decline[pos_mask])) if n_pos > 0 else np.nan
    fpr = float(np.mean(decline[neg_mask])) if n_neg > 0 else np.nan
    fnr = 1.0 - tpr if not np.isnan(tpr) else np.nan

    return {
        "n": n,
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "approval_rate": round(approval_rate, 6),
        "tpr": round(tpr, 6) if not np.isnan(tpr) else np.nan,
        "fpr": round(fpr, 6) if not np.isnan(fpr) else np.nan,
        "fnr": round(fnr, 6) if not np.isnan(fnr) else np.nan,
    }


def compute_fairness_gaps_hand(
    group_stats: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Calculate DP difference, DP ratio, and EO difference by hand."""
    approval_rates = [stats["approval_rate"] for stats in group_stats.values() if not np.isnan(stats["approval_rate"])]
    tprs = [stats["tpr"] for stats in group_stats.values() if not np.isnan(stats["tpr"])]
    fprs = [stats["fpr"] for stats in group_stats.values() if not np.isnan(stats["fpr"])]

    if len(approval_rates) < 2:
        return {"dp_difference": np.nan, "dp_ratio": np.nan, "eo_difference": np.nan}

    min_app = min(approval_rates)
    max_app = max(approval_rates)

    dp_diff = max_app - min_app
    dp_ratio = min_app / max_app if max_app > 0 else 0.0

    tpr_gap = max(tprs) - min(tprs) if len(tprs) >= 2 else 0.0
    fpr_gap = max(fprs) - min(fprs) if len(fprs) >= 2 else 0.0
    eo_diff = max(tpr_gap, fpr_gap)

    return {
        "dp_difference": round(dp_diff, 6),
        "dp_ratio": round(dp_ratio, 6),
        "eo_difference": round(eo_diff, 6),
    }


def run_stratified_bootstrap_gaps(
    df: pd.DataFrame,
    group_col: str,
    y_col: str,
    decline_a_col: str,
    decline_b_col: str,
    n_bootstraps: int = 200,
    seed: int = 42,
    alpha: float = 0.05
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]], int]:
    """Run group-stratified bootstrap for gaps for Model A, Model B, and paired delta (B - A)."""
    rng = np.random.RandomState(seed)
    unique_groups = df[group_col].unique()
    group_indices = {g: np.where(df[group_col].values == g)[0] for g in unique_groups}

    boot_gaps_a = {"dp_difference": [], "dp_ratio": [], "eo_difference": []}
    boot_gaps_b = {"dp_difference": [], "dp_ratio": [], "eo_difference": []}
    boot_diffs = {"dp_difference": [], "dp_ratio": [], "eo_difference": []}
    skipped = 0

    y_arr = df[y_col].values
    dec_a_arr = df[decline_a_col].values
    dec_b_arr = df[decline_b_col].values

    for _ in range(n_bootstraps):
        sampled_indices = []
        degenerate = False

        for g, idxs in group_indices.items():
            n_g = len(idxs)
            boot_idx = rng.choice(idxs, size=n_g, replace=True)
            # check degenerate
            g_y = y_arr[boot_idx]
            if len(np.unique(g_y)) < 2:
                degenerate = True
                break
            sampled_indices.append(boot_idx)

        if degenerate:
            skipped += 1
            continue

        sample_idx = np.concatenate(sampled_indices)
        sub_y = y_arr[sample_idx]
        sub_a = dec_a_arr[sample_idx]
        sub_b = dec_b_arr[sample_idx]
        sub_g = df[group_col].values[sample_idx]

        stats_a = {}
        stats_b = {}
        for g in unique_groups:
            mask_g = (sub_g == g)
            stats_a[g] = compute_group_metrics(sub_y[mask_g], sub_a[mask_g])
            stats_b[g] = compute_group_metrics(sub_y[mask_g], sub_b[mask_g])

        gap_a = compute_fairness_gaps_hand(stats_a)
        gap_b = compute_fairness_gaps_hand(stats_b)

        for metric in ["dp_difference", "dp_ratio", "eo_difference"]:
            val_a = gap_a[metric]
            val_b = gap_b[metric]
            boot_gaps_a[metric].append(val_a)
            boot_gaps_b[metric].append(val_b)
            boot_diffs[metric].append(val_b - val_a)

    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)

    ci_a = {}
    ci_b = {}
    ci_delta = {}

    for metric in ["dp_difference", "dp_ratio", "eo_difference"]:
        if len(boot_gaps_a[metric]) > 0:
            ci_a[metric] = (
                round(float(np.percentile(boot_gaps_a[metric], lower_pct)), 6),
                round(float(np.percentile(boot_gaps_a[metric], upper_pct)), 6),
            )
            ci_b[metric] = (
                round(float(np.percentile(boot_gaps_b[metric], lower_pct)), 6),
                round(float(np.percentile(boot_gaps_b[metric], upper_pct)), 6),
            )
            ci_delta[metric] = (
                round(float(np.percentile(boot_diffs[metric], lower_pct)), 6),
                round(float(np.percentile(boot_diffs[metric], upper_pct)), 6),
            )
        else:
            ci_a[metric] = (np.nan, np.nan)
            ci_b[metric] = (np.nan, np.nan)
            ci_delta[metric] = (np.nan, np.nan)

    return ci_a, ci_b, ci_delta, skipped


def run_fairness_audit(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Execute complete Phase 4 fairness audit."""
    if config is None:
        config = load_config()

    seed = set_seed(config=config)
    bootstrap_n = int(config.get("bootstrap_n", 200))
    decline_rate_cfg = float(config.get("decline_rate", 0.15))

    metrics_dir = _PROJECT_ROOT / config["paths"]["metrics"]
    tables_dir = _PROJECT_ROOT / "reports" / "tables"
    predictions_dir = _PROJECT_ROOT / config["paths"]["predictions"]
    processed_dir = _PROJECT_ROOT / config["paths"]["processed"]
    splits_dir = _PROJECT_ROOT / config["paths"]["splits"]

    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load validation & test predictions
    val_a = pd.read_parquet(predictions_dir / "model_a_val.parquet")
    test_a = pd.read_parquet(predictions_dir / "model_a_test.parquet")
    val_b = pd.read_parquet(predictions_dir / "model_b_val.parquet")
    test_b = pd.read_parquet(predictions_dir / "model_b_test.parquet")

    # 2. Decision policy: separate thresholds from validation
    thresh_a = compute_threshold_from_validation(val_a["score"].values, decline_rate=decline_rate_cfg)
    thresh_b = compute_threshold_from_validation(val_b["score"].values, decline_rate=decline_rate_cfg)

    decl_val_a, app_val_a = apply_decision_policy(val_a["score"].values, thresh_a)
    decl_val_b, app_val_b = apply_decision_policy(val_b["score"].values, thresh_b)

    decl_test_a, app_test_a = apply_decision_policy(test_a["score"].values, thresh_a)
    decl_test_b, app_test_b = apply_decision_policy(test_b["score"].values, thresh_b)

    realized_test_decl_a = round(float(np.mean(decl_test_a)), 6)
    realized_test_decl_b = round(float(np.mean(decl_test_b)), 6)

    print(f"[fairness] Model A: validation threshold = {thresh_a:.6f}, realized test decline = {realized_test_decl_a:.4f}")
    print(f"[fairness] Model B: validation threshold = {thresh_b:.6f}, realized test decline = {realized_test_decl_b:.4f}")

    # Check decline rates within 0.02 of 0.15
    if abs(realized_test_decl_a - decline_rate_cfg) > 0.02:
        raise ValueError(f"Model A realized decline {realized_test_decl_a} is not within 0.02 of {decline_rate_cfg}!")
    if abs(realized_test_decl_b - decline_rate_cfg) > 0.02:
        raise ValueError(f"Model B realized decline {realized_test_decl_b} is not within 0.02 of {decline_rate_cfg}!")

    # 3. Load sensitive attributes from eval_attrs.parquet
    eval_attrs = pd.read_parquet(processed_dir / "eval_attrs.parquet")
    train_ids = pd.read_csv(splits_dir / "train_ids.csv", usecols=["SK_ID_CURR"])
    train_eval = eval_attrs[eval_attrs["SK_ID_CURR"].isin(set(train_ids["SK_ID_CURR"]))]
    train_median_age = round(float(train_eval["age_years"].median()), 2)
    print(f"[fairness] Train-only median age: {train_median_age} years")

    # Prepare test audit dataset
    test_df = test_a[["SK_ID_CURR", "segment", "TARGET"]].copy()
    test_df["score_a"] = test_a["score"]
    test_df["score_b"] = test_b["score"]
    test_df["decline_a"] = decl_test_a
    test_df["decline_b"] = decl_test_b
    test_df["approval_a"] = app_test_a
    test_df["approval_b"] = app_test_b

    test_df = test_df.merge(eval_attrs, on="SK_ID_CURR", how="left")
    test_df["binary_age"] = np.where(test_df["age_years"] >= train_median_age, ">=median", "<median")

    xna_test_count = int((test_df["gender"] == "XNA").sum())
    print(f"[fairness] Test XNA gender count: {xna_test_count}")

    # 4. Cross-check hand calculation vs Fairlearn on gender
    test_clean_gender = test_df[test_df["gender"] != "XNA"].copy()
    fl_dp_diff_a = demographic_parity_difference(
        test_clean_gender["TARGET"], test_clean_gender["approval_a"], sensitive_features=test_clean_gender["gender"]
    )
    fl_dp_ratio_a = demographic_parity_ratio(
        test_clean_gender["TARGET"], test_clean_gender["approval_a"], sensitive_features=test_clean_gender["gender"]
    )
    fl_eo_diff_a = equalized_odds_difference(
        test_clean_gender["TARGET"], test_clean_gender["decline_a"], sensitive_features=test_clean_gender["gender"]
    )

    stats_gender_a_f = compute_group_metrics(
        test_clean_gender[test_clean_gender["gender"] == "F"]["TARGET"].values,
        test_clean_gender[test_clean_gender["gender"] == "F"]["decline_a"].values
    )
    stats_gender_a_m = compute_group_metrics(
        test_clean_gender[test_clean_gender["gender"] == "M"]["TARGET"].values,
        test_clean_gender[test_clean_gender["gender"] == "M"]["decline_a"].values
    )
    hand_gaps_a = compute_fairness_gaps_hand({"F": stats_gender_a_f, "M": stats_gender_a_m})

    diff_check = abs(hand_gaps_a["dp_difference"] - fl_dp_diff_a) < 1e-5
    ratio_check = abs(hand_gaps_a["dp_ratio"] - fl_dp_ratio_a) < 1e-5
    eo_check = abs(hand_gaps_a["eo_difference"] - fl_eo_diff_a) < 1e-5
    cross_check_passed = bool(diff_check and ratio_check and eo_check)
    print(f"[fairness] Fairlearn cross-check on gender passed: {cross_check_passed}")
    if not cross_check_passed:
        raise ValueError("Fairlearn cross-check failed!")

    # 5. Full Audit on Test Split
    # Attributes: Gender, Binary Age, Four Age Bins
    audit_results: Dict[str, Any] = {
        "policy": {
            "decline_rate_target": decline_rate_cfg,
            "threshold_a": round(thresh_a, 6),
            "threshold_b": round(thresh_b, 6),
            "realized_test_decline_a": realized_test_decl_a,
            "realized_test_decline_b": realized_test_decl_b,
            "train_median_age": train_median_age,
            "test_xna_count": xna_test_count,
        },
        "conventions": {
            "fairlearn_version": "0.15.0",
            "functions_used": [
                "demographic_parity_difference",
                "demographic_parity_ratio",
                "equalized_odds_difference",
            ],
            "multigroup_convention": "between_groups (worst_case difference/ratio across all groups)",
            "cross_check_passed": cross_check_passed,
        },
        "overall": {},
        "by_segment": {},
    }

    flat_gaps_rows = []

    # Attribute evaluations
    attrs_config = [
        ("gender", ["F", "M"], test_clean_gender),
        ("binary_age", ["<median", ">=median"], test_df),
        ("four_age_bins", ["<30", "30-44", "45-59", "60+"], test_df),
    ]

    for attr_name, groups, data_source in attrs_config:
        group_col = "gender" if attr_name == "gender" else ("binary_age" if attr_name == "binary_age" else "age_bin")
        
        # Calculate group metrics for A and B
        grp_metrics_a = {}
        grp_metrics_b = {}
        unreliable_groups = []

        for g in groups:
            sub = data_source[data_source[group_col] == g]
            y_sub = sub["TARGET"].values
            dec_a = sub["decline_a"].values
            dec_b = sub["decline_b"].values

            m_a = compute_group_metrics(y_sub, dec_a)
            m_b = compute_group_metrics(y_sub, dec_b)

            is_unreliable = bool(m_a["n"] < 100)
            if is_unreliable:
                unreliable_groups.append(g)

            m_a["unreliable"] = is_unreliable
            m_b["unreliable"] = is_unreliable

            grp_metrics_a[g] = m_a
            grp_metrics_b[g] = m_b

        # Point estimate gaps
        gaps_a = compute_fairness_gaps_hand(grp_metrics_a)
        gaps_b = compute_fairness_gaps_hand(grp_metrics_b)

        # Bootstrap CIs
        ci_a, ci_b, ci_delta, skipped_resamples = run_stratified_bootstrap_gaps(
            data_source, group_col, "TARGET", "decline_a", "decline_b",
            n_bootstraps=bootstrap_n, seed=seed
        )

        attr_entry = {
            "groups": groups,
            "unreliable_groups": unreliable_groups,
            "skipped_bootstrap_resamples": skipped_resamples,
            "model_a": {
                "group_metrics": grp_metrics_a,
                "gaps": gaps_a,
                "confidence_intervals_95": ci_a,
            },
            "model_b": {
                "group_metrics": grp_metrics_b,
                "gaps": gaps_b,
                "confidence_intervals_95": ci_b,
            },
            "paired_delta_b_minus_a": {
                "gaps_diff": {k: round(gaps_b[k] - gaps_a[k], 6) for k in gaps_a},
                "confidence_intervals_95": ci_delta,
            },
        }

        audit_results["overall"][attr_name] = attr_entry

        # Add to flat table
        for metric_key in ["dp_difference", "dp_ratio", "eo_difference"]:
            flat_gaps_rows.append({
                "scope": "overall",
                "segment": "all",
                "attribute": attr_name,
                "metric": metric_key,
                "model_a_gap": gaps_a[metric_key],
                "model_a_ci_lower": ci_a[metric_key][0],
                "model_a_ci_upper": ci_a[metric_key][1],
                "model_b_gap": gaps_b[metric_key],
                "model_b_ci_lower": ci_b[metric_key][0],
                "model_b_ci_upper": ci_b[metric_key][1],
                "delta_b_minus_a": round(gaps_b[metric_key] - gaps_a[metric_key], 6),
                "delta_ci_lower": ci_delta[metric_key][0],
                "delta_ci_upper": ci_delta[metric_key][1],
                "unreliable": bool(len(unreliable_groups) > 0),
            })

    # Segment evaluations
    segments = ["full_file", "thin_file", "no_file"]
    for seg in segments:
        seg_data = test_df[test_df["segment"] == seg].copy()
        seg_clean_gender = seg_data[seg_data["gender"] != "XNA"].copy()

        # Realized decline rate and TPR per segment
        seg_n = len(seg_data)
        seg_decl_a = float(np.mean(seg_data["decline_a"]))
        seg_decl_b = float(np.mean(seg_data["decline_b"]))
        seg_tpr_a = float(np.mean(seg_data[seg_data["TARGET"] == 1]["decline_a"]))
        seg_tpr_b = float(np.mean(seg_data[seg_data["TARGET"] == 1]["decline_b"]))

        seg_summary = {
            "n_applicants": seg_n,
            "model_a": {"decline_rate": round(seg_decl_a, 6), "tpr": round(seg_tpr_a, 6)},
            "model_b": {"decline_rate": round(seg_decl_b, 6), "tpr": round(seg_tpr_b, 6)},
            "attributes": {},
        }

        seg_attrs = [
            ("gender", ["F", "M"], seg_clean_gender),
            ("binary_age", ["<median", ">=median"], seg_data),
        ]

        for attr_name, groups, d_source in seg_attrs:
            group_col = "gender" if attr_name == "gender" else "binary_age"
            g_counts = d_source[group_col].value_counts().to_dict()
            has_under_100 = any(g_counts.get(g, 0) < 100 for g in groups)

            grp_metrics_a = {g: compute_group_metrics(d_source[d_source[group_col] == g]["TARGET"].values, d_source[d_source[group_col] == g]["decline_a"].values) for g in groups}
            grp_metrics_b = {g: compute_group_metrics(d_source[d_source[group_col] == g]["TARGET"].values, d_source[d_source[group_col] == g]["decline_b"].values) for g in groups}

            gaps_a = compute_fairness_gaps_hand(grp_metrics_a)
            gaps_b = compute_fairness_gaps_hand(grp_metrics_b)

            if not has_under_100:
                ci_a, ci_b, ci_delta, _ = run_stratified_bootstrap_gaps(
                    d_source, group_col, "TARGET", "decline_a", "decline_b",
                    n_bootstraps=bootstrap_n, seed=seed
                )
            else:
                ci_a = {k: (np.nan, np.nan) for k in gaps_a}
                ci_b = {k: (np.nan, np.nan) for k in gaps_b}
                ci_delta = {k: (np.nan, np.nan) for k in gaps_a}

            seg_summary["attributes"][attr_name] = {
                "group_sizes": g_counts,
                "unreliable": has_under_100,
                "model_a_gaps": gaps_a,
                "model_b_gaps": gaps_b,
                "delta_b_minus_a": {k: round(gaps_b[k] - gaps_a[k], 6) for k in gaps_a},
                "ci_delta_95": ci_delta,
            }

            for metric_key in ["dp_difference", "dp_ratio", "eo_difference"]:
                flat_gaps_rows.append({
                    "scope": "segment",
                    "segment": seg,
                    "attribute": attr_name,
                    "metric": metric_key,
                    "model_a_gap": gaps_a[metric_key],
                    "model_a_ci_lower": ci_a[metric_key][0],
                    "model_a_ci_upper": ci_a[metric_key][1],
                    "model_b_gap": gaps_b[metric_key],
                    "model_b_ci_lower": ci_b[metric_key][0],
                    "model_b_ci_upper": ci_b[metric_key][1],
                    "delta_b_minus_a": round(gaps_b[metric_key] - gaps_a[metric_key], 6),
                    "delta_ci_lower": ci_delta[metric_key][0],
                    "delta_ci_upper": ci_delta[metric_key][1],
                    "unreliable": has_under_100,
                })

        audit_results["by_segment"][seg] = seg_summary

    # Save reports/metrics/fairness.json
    fairness_json_path = metrics_dir / "fairness.json"
    with open(fairness_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"[fairness] Saved {fairness_json_path}")

    # Save flat table reports/tables/fairness_gaps.csv
    flat_gaps_df = pd.DataFrame(flat_gaps_rows)
    flat_csv_path = tables_dir / "fairness_gaps.csv"
    flat_gaps_df.to_csv(flat_csv_path, index=False)
    print(f"[fairness] Saved {flat_csv_path}")

    # 6. STEP 4 - Robustness sweep at decline rates 0.10, 0.15, 0.20
    robustness_rows = []
    for d_rate in [0.10, 0.15, 0.20]:
        t_a = compute_threshold_from_validation(val_a["score"].values, decline_rate=d_rate)
        t_b = compute_threshold_from_validation(val_b["score"].values, decline_rate=d_rate)

        d_test_a, _ = apply_decision_policy(test_a["score"].values, t_a)
        d_test_b, _ = apply_decision_policy(test_b["score"].values, t_b)

        temp_test = test_df[["TARGET", "gender", "binary_age"]].copy()
        temp_test["decline_a"] = d_test_a
        temp_test["decline_b"] = d_test_b

        for attr_col, attr_name, clean_df in [
            ("gender", "gender", temp_test[temp_test["gender"] != "XNA"]),
            ("binary_age", "binary_age", temp_test),
        ]:
            groups = clean_df[attr_col].unique()
            st_a = {g: compute_group_metrics(clean_df[clean_df[attr_col] == g]["TARGET"].values, clean_df[clean_df[attr_col] == g]["decline_a"].values) for g in groups}
            st_b = {g: compute_group_metrics(clean_df[clean_df[attr_col] == g]["TARGET"].values, clean_df[clean_df[attr_col] == g]["decline_b"].values) for g in groups}

            gp_a = compute_fairness_gaps_hand(st_a)
            gp_b = compute_fairness_gaps_hand(st_b)

            robustness_rows.append({
                "target_decline_rate": d_rate,
                "attribute": attr_name,
                "model_a_threshold": round(t_a, 6),
                "model_b_threshold": round(t_b, 6),
                "model_a_dp_diff": gp_a["dp_difference"],
                "model_a_dp_ratio": gp_a["dp_ratio"],
                "model_a_eo_diff": gp_a["eo_difference"],
                "model_b_dp_diff": gp_b["dp_difference"],
                "model_b_dp_ratio": gp_b["dp_ratio"],
                "model_b_eo_diff": gp_b["eo_difference"],
            })

    rob_df = pd.DataFrame(robustness_rows)
    rob_csv_path = tables_dir / "fairness_by_decline_rate.csv"
    rob_df.to_csv(rob_csv_path, index=False)
    print(f"[fairness] Saved robustness table to {rob_csv_path}")

    return audit_results


if __name__ == "__main__":
    run_fairness_audit()
