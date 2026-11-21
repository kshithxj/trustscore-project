"""Lift difference and multi-split stability analysis for TrustScore (Phase 7 Part D).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config, set_seed
from src.extras.common import (
    load_prepared_data,
    train_xgb,
)


def compute_paired_lift_diff(
    test_df: pd.DataFrame,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_bootstraps: int = 200,
    seed: int = 42,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Part D(i): Paired bootstrap of difference of lifts (Delta_thin - Delta_full)."""
    rng = np.random.RandomState(seed)
    n = len(test_df)
    
    y = test_df["TARGET"].values
    segments = test_df["segment"].values

    thin_mask = (segments == "thin_file")
    full_mask = (segments == "full_file")

    boot_diff_of_lifts = []
    
    # We can stratify resamples by segment
    thin_indices = np.where(thin_mask)[0]
    full_indices = np.where(full_mask)[0]

    for _ in range(n_bootstraps):
        thin_samp = rng.choice(thin_indices, size=len(thin_indices), replace=True)
        full_samp = rng.choice(full_indices, size=len(full_indices), replace=True)

        # Check binary classes
        if len(np.unique(y[thin_samp])) < 2 or len(np.unique(y[full_samp])) < 2:
            continue

        thin_auc_a = roc_auc_score(y[thin_samp], score_a[thin_samp])
        thin_auc_b = roc_auc_score(y[thin_samp], score_b[thin_samp])
        delta_thin = thin_auc_b - thin_auc_a

        full_auc_a = roc_auc_score(y[full_samp], score_a[full_samp])
        full_auc_b = roc_auc_score(y[full_samp], score_b[full_samp])
        delta_full = full_auc_b - full_auc_a

        boot_diff_of_lifts.append(delta_thin - delta_full)

    diff_arr = np.array(boot_diff_of_lifts)
    mean_val = float(np.mean(diff_arr))
    lower = float(np.percentile(diff_arr, 100 * (alpha / 2.0)))
    upper = float(np.percentile(diff_arr, 100 * (1.0 - alpha / 2.0)))
    share_gt_0 = float(np.mean(diff_arr > 0))

    return {
        "mean_diff_of_lifts": round(mean_val, 6),
        "ci_95_lower": round(lower, 6),
        "ci_95_upper": round(upper, 6),
        "share_gt_0": round(share_gt_0, 4),
        "bootstrap_samples_used": len(diff_arr),
    }


def generate_new_split(base: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Create 60/20/20 train/valid/test split stratified by TARGET x segment."""
    base_copy = base[["SK_ID_CURR", "TARGET", "segment"]].copy()
    base_copy["strat_target_segment"] = (
        base_copy["TARGET"].astype(str) + "_" + base_copy["segment"]
    )
    min_stratum = base_copy["strat_target_segment"].value_counts().min()

    if min_stratum < 10:
        strat_col = "segment"
        strat_used = "segment_only"
    else:
        strat_col = "strat_target_segment"
        strat_used = "target_x_segment"

    train_df, temp_df = train_test_split(
        base_copy,
        train_size=0.6,
        stratify=base_copy[strat_col],
        random_state=seed,
    )
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=0.5,
        stratify=temp_df[strat_col],
        random_state=seed,
    )

    keep_cols = ["SK_ID_CURR", "TARGET", "segment"]
    return train_df[keep_cols], valid_df[keep_cols], test_df[keep_cols], strat_used


def run_stability() -> None:
    start_time = time.time()
    config = load_config()
    main_seed = set_seed(config=config)
    bootstrap_n = int(config.get("bootstrap_n", 200))

    out_metrics_dir = _PROJECT_ROOT / "reports" / "metrics"
    out_tables_dir = _PROJECT_ROOT / "reports" / "tables"
    out_preds_dir = _PROJECT_ROOT / "reports" / "predictions" / "extras"
    processed_dir = _PROJECT_ROOT / config["paths"]["processed"]

    out_metrics_dir.mkdir(parents=True, exist_ok=True)
    out_tables_dir.mkdir(parents=True, exist_ok=True)
    out_preds_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Part D(i): Paired Bootstrap Difference of Lifts (main split)
    # -------------------------------------------------------------
    print("[stability] Running Part D(i): Difference of lifts on main test split...")
    preds_dir = _PROJECT_ROOT / "reports" / "predictions"
    main_test_pred_a = pd.read_parquet(preds_dir / "model_a_test.parquet")
    main_test_pred_b = pd.read_parquet(preds_dir / "model_b_test.parquet")

    eval_attrs = pd.read_parquet(processed_dir / "eval_attrs.parquet")
    main_test_merged = main_test_pred_a.merge(eval_attrs[["SK_ID_CURR", "gender"]], on="SK_ID_CURR", how="left")

    lift_diff_results = compute_paired_lift_diff(
        test_df=main_test_merged,
        score_a=main_test_pred_a["score"].values,
        score_b=main_test_pred_b["score"].values,
        n_bootstraps=bootstrap_n,
        seed=main_seed,
    )
    print(f"[stability] Difference of lifts: {lift_diff_results}")

    # -------------------------------------------------------------
    # Part D(ii): Multi-Split Stability across 5 new splits (seeds 1..5)
    # -------------------------------------------------------------
    print("\n[stability] Running Part D(ii): Stability across seeds 1 to 5...")
    base_df = pd.read_parquet(processed_dir / "applicant_base.parquet")

    stability_seeds = [1, 2, 3, 4, 5]
    split_records = []
    per_split_metrics = []

    for s in stability_seeds:
        split_t0 = time.time()
        print(f"\n[stability] Processing split seed={s}...")
        train_df, valid_df, test_df, strat_used = generate_new_split(base_df, seed=s)

        # Save split IDs
        split_all = pd.concat([
            train_df.assign(split="train"),
            valid_df.assign(split="valid"),
            test_df.assign(split="test"),
        ], ignore_index=True)
        split_ids_file = out_preds_dir / f"split_seed_{s}_ids.csv"
        split_all[["SK_ID_CURR", "split"]].to_csv(split_ids_file, index=False)
        print(f"[stability] Saved {split_ids_file}")

        # Prepare feature matrices for this split
        (
            X_tr_a, X_va_a, X_te_a,
            X_tr_b, X_va_b, X_te_b,
            y_tr, y_va, y_te,
            meta_tr, meta_va, meta_te,
            feat_a, feat_b
        ) = load_prepared_data(
            config=config,
            train_ids_df=train_df,
            valid_ids_df=valid_df,
            test_ids_df=test_df,
        )

        # Train Model A
        clf_a = train_xgb(X_tr_a, y_tr, X_va_a, y_va, seed=s)
        test_preds_a = clf_a.predict_proba(X_te_a)[:, 1]

        # Train Model B
        clf_b = train_xgb(X_tr_b, y_tr, X_va_b, y_va, seed=s)
        test_preds_b = clf_b.predict_proba(X_te_b)[:, 1]

        # Evaluate Overall AUC
        auc_a_overall = float(roc_auc_score(y_te, test_preds_a))
        auc_b_overall = float(roc_auc_score(y_te, test_preds_b))
        diff_overall = auc_b_overall - auc_a_overall

        split_res = {
            "seed": s,
            "runtime_sec": round(time.time() - split_t0, 2),
            "stratification": strat_used,
            "overall": {
                "auc_a": round(auc_a_overall, 6),
                "auc_b": round(auc_b_overall, 6),
                "diff": round(diff_overall, 6),
                "b_gt_a": bool(diff_overall > 0),
            },
            "by_segment": {},
        }

        row_common = {
            "seed": s,
            "auc_a_overall": round(auc_a_overall, 6),
            "auc_b_overall": round(auc_b_overall, 6),
            "diff_overall": round(diff_overall, 6),
        }

        # Evaluate Segment AUCs
        for seg in ["full_file", "thin_file", "no_file"]:
            mask = (meta_te["segment"] == seg).values
            y_sub = y_te.values[mask]
            
            if len(np.unique(y_sub)) >= 2:
                seg_auc_a = float(roc_auc_score(y_sub, test_preds_a[mask]))
                seg_auc_b = float(roc_auc_score(y_sub, test_preds_b[mask]))
                seg_diff = seg_auc_b - seg_auc_a
            else:
                seg_auc_a, seg_auc_b, seg_diff = float("nan"), float("nan"), float("nan")

            split_res["by_segment"][seg] = {
                "auc_a": round(seg_auc_a, 6) if not np.isnan(seg_auc_a) else None,
                "auc_b": round(seg_auc_b, 6) if not np.isnan(seg_auc_b) else None,
                "diff": round(seg_diff, 6) if not np.isnan(seg_diff) else None,
                "b_gt_a": bool(seg_diff > 0) if not np.isnan(seg_diff) else False,
            }
            row_common[f"auc_a_{seg}"] = round(seg_auc_a, 6) if not np.isnan(seg_auc_a) else None
            row_common[f"auc_b_{seg}"] = round(seg_auc_b, 6) if not np.isnan(seg_auc_b) else None
            row_common[f"diff_{seg}"] = round(seg_diff, 6) if not np.isnan(seg_diff) else None

        per_split_metrics.append(split_res)
        split_records.append(row_common)

    total_elapsed = round(time.time() - start_time, 2)
    print(f"\n[stability] Total stability runtime: {total_elapsed} seconds (guard <= 900s).")
    if total_elapsed > 900:
        raise RuntimeError(f"Stability analysis exceeded 15-minute runtime guard: {total_elapsed}s > 900s")

    # Aggregate statistics across 5 splits
    df_stability = pd.DataFrame(split_records)
    
    summary_stats = {}
    for metric_col in [
        "auc_a_overall", "auc_b_overall", "diff_overall",
        "auc_a_full_file", "auc_b_full_file", "diff_full_file",
        "auc_a_thin_file", "auc_b_thin_file", "diff_thin_file",
    ]:
        vals = df_stability[metric_col].dropna().values
        summary_stats[metric_col] = {
            "mean": round(float(np.mean(vals)), 6),
            "sd": round(float(np.std(vals, ddof=1)), 6) if len(vals) > 1 else 0.0,
            "min": round(float(np.min(vals)), 6),
            "max": round(float(np.max(vals)), 6),
        }

    b_gt_a_overall_count = int((df_stability["diff_overall"] > 0).sum())
    b_gt_a_thin_count = int((df_stability["diff_thin_file"] > 0).sum())

    summary_stats["b_gt_a_counts"] = {
        "overall": f"{b_gt_a_overall_count}/5",
        "thin_file": f"{b_gt_a_thin_count}/5",
    }

    # Save metrics JSON
    metrics_payload = {
        "lift_difference": lift_diff_results,
        "runtime_seconds": total_elapsed,
        "runtime_guard_passed": bool(total_elapsed <= 900),
        "splits": per_split_metrics,
        "summary": summary_stats,
    }
    with open(out_metrics_dir / "extras_stability.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[stability] Saved {out_metrics_dir / 'extras_stability.json'}")

    # Save table CSV
    df_stability.to_csv(out_tables_dir / "extras_stability.csv", index=False)
    print(f"[stability] Saved {out_tables_dir / 'extras_stability.csv'}")


if __name__ == "__main__":
    run_stability()
