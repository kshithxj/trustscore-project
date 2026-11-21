"""Phase 8 Stage 1: Precompute for TrustScore demo dashboard.

Precomputes:
1. Frontier table (5% to 30% decline rates) for Model A and Model B -> reports/tables/app_frontier.csv
2. SHAP contributions and raw feature values for sample applicants -> reports/predictions/extras/app_shap_sample.parquet
3. 1:1 join check between Model A and Model B test predictions.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import shap

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure MPLCONFIGDIR is writable in headless environments
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

from src.utils import load_config
from src.fairness.policy import compute_threshold_from_validation, apply_decision_policy
from src.fairness.audit import compute_group_metrics, compute_fairness_gaps_hand
from src.features.model_b_features import prepare_model_b_datasets


def run_frontier_precompute(
    config: Dict[str, Any] | None = None
) -> Tuple[pd.DataFrame, float, float, Dict[str, Any]]:
    """Stage 1a: Compute policy frontier table for Model A and Model B (5% to 30% decline rate).
    
    Returns:
        frontier_df: The 52-row DataFrame (26 per model)
        max_diff_robustness: Max absolute difference against fairness_by_decline_rate.csv
        max_diff_json: Max absolute difference against fairness.json
        meta: Runtime and file metadata
    """
    t0 = time.time()
    if config is None:
        config = load_config()

    predictions_dir = _PROJECT_ROOT / config["paths"]["predictions"]
    processed_dir = _PROJECT_ROOT / config["paths"]["processed"]
    splits_dir = _PROJECT_ROOT / config["paths"]["splits"]
    tables_dir = _PROJECT_ROOT / "reports" / "tables"
    metrics_dir = _PROJECT_ROOT / config["paths"]["metrics"]

    # 1. Load predictions & eval attributes
    val_a = pd.read_parquet(predictions_dir / "model_a_val.parquet")
    test_a = pd.read_parquet(predictions_dir / "model_a_test.parquet")
    val_b = pd.read_parquet(predictions_dir / "model_b_val.parquet")
    test_b = pd.read_parquet(predictions_dir / "model_b_test.parquet")

    eval_attrs = pd.read_parquet(processed_dir / "eval_attrs.parquet")
    train_ids = pd.read_csv(splits_dir / "train_ids.csv", usecols=["SK_ID_CURR"])
    train_eval = eval_attrs[eval_attrs["SK_ID_CURR"].isin(set(train_ids["SK_ID_CURR"]))]
    train_median_age = round(float(train_eval["age_years"].median()), 2)

    test_df = test_a[["SK_ID_CURR", "segment", "TARGET"]].copy()
    test_df["score_a"] = test_a["score"].values
    test_df["score_b"] = test_b["score"].values
    test_df = test_df.merge(eval_attrs[["SK_ID_CURR", "gender", "age_years"]], on="SK_ID_CURR", how="left")
    test_df["binary_age"] = np.where(test_df["age_years"] >= train_median_age, ">=median", "<median")

    # Clean gender split (exclude XNA)
    gender_clean_mask = test_df["gender"] != "XNA"

    # Decline rates 5% to 30% in steps of 1% (26 rates)
    decline_rates = [round(r / 100.0, 4) for r in range(5, 31)]

    y_test = test_df["TARGET"].values
    y_gender_clean = test_df.loc[gender_clean_mask, "TARGET"].values
    gender_clean_groups = test_df.loc[gender_clean_mask, "gender"].values
    binary_age_groups = test_df["binary_age"].values

    frontier_rows = []

    for d_rate in decline_rates:
        # Model A
        thresh_a = compute_threshold_from_validation(val_a["score"].values, decline_rate=d_rate)
        decl_a, _ = apply_decision_policy(test_a["score"].values, thresh_a)
        realized_decl_a = float(np.mean(decl_a))
        m_overall_a = compute_group_metrics(y_test, decl_a)

        decl_a_g = decl_a[gender_clean_mask]
        st_a_g = {
            g: compute_group_metrics(y_gender_clean[gender_clean_groups == g], decl_a_g[gender_clean_groups == g])
            for g in ["F", "M"]
        }
        gaps_a_g = compute_fairness_gaps_hand(st_a_g)

        st_a_age = {
            g: compute_group_metrics(y_test[binary_age_groups == g], decl_a[binary_age_groups == g])
            for g in ["<median", ">=median"]
        }
        gaps_a_age = compute_fairness_gaps_hand(st_a_age)

        frontier_rows.append({
            "model": "Model A",
            "model_id": "model_a",
            "decline_rate": d_rate,
            "target_decline_rate": d_rate,
            "threshold": round(thresh_a, 6),
            "validation_threshold": round(thresh_a, 6),
            "realized_decline_rate": round(realized_decl_a, 6),
            "default_capture": m_overall_a["tpr"],
            "overall_tpr": m_overall_a["tpr"],
            "overall_fpr": m_overall_a["fpr"],
            "gender_dp_diff": gaps_a_g["dp_difference"],
            "gender_dp_ratio": gaps_a_g["dp_ratio"],
            "gender_eo_diff": gaps_a_g["eo_difference"],
            "binary_age_dp_diff": gaps_a_age["dp_difference"],
            "binary_age_dp_ratio": gaps_a_age["dp_ratio"],
            "binary_age_eo_diff": gaps_a_age["eo_difference"],
        })

        # Model B
        thresh_b = compute_threshold_from_validation(val_b["score"].values, decline_rate=d_rate)
        decl_b, _ = apply_decision_policy(test_b["score"].values, thresh_b)
        realized_decl_b = float(np.mean(decl_b))
        m_overall_b = compute_group_metrics(y_test, decl_b)

        decl_b_g = decl_b[gender_clean_mask]
        st_b_g = {
            g: compute_group_metrics(y_gender_clean[gender_clean_groups == g], decl_b_g[gender_clean_groups == g])
            for g in ["F", "M"]
        }
        gaps_b_g = compute_fairness_gaps_hand(st_b_g)

        st_b_age = {
            g: compute_group_metrics(y_test[binary_age_groups == g], decl_b[binary_age_groups == g])
            for g in ["<median", ">=median"]
        }
        gaps_b_age = compute_fairness_gaps_hand(st_b_age)

        frontier_rows.append({
            "model": "Model B",
            "model_id": "model_b",
            "decline_rate": d_rate,
            "target_decline_rate": d_rate,
            "threshold": round(thresh_b, 6),
            "validation_threshold": round(thresh_b, 6),
            "realized_decline_rate": round(realized_decl_b, 6),
            "default_capture": m_overall_b["tpr"],
            "overall_tpr": m_overall_b["tpr"],
            "overall_fpr": m_overall_b["fpr"],
            "gender_dp_diff": gaps_b_g["dp_difference"],
            "gender_dp_ratio": gaps_b_g["dp_ratio"],
            "gender_eo_diff": gaps_b_g["eo_difference"],
            "binary_age_dp_diff": gaps_b_age["dp_difference"],
            "binary_age_dp_ratio": gaps_b_age["dp_ratio"],
            "binary_age_eo_diff": gaps_b_age["eo_difference"],
        })

    frontier_df = pd.DataFrame(frontier_rows)

    # Save to reports/tables/app_frontier.csv
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "app_frontier.csv"
    frontier_df.to_csv(out_csv, index=False)

    # 2. Consistency Checks
    # Compare against reports/tables/fairness_by_decline_rate.csv
    rob_csv_path = tables_dir / "fairness_by_decline_rate.csv"
    rob_df = pd.read_csv(rob_csv_path)

    diffs_rob: List[float] = []
    for d_rate in [0.10, 0.15, 0.20]:
        sub_a = frontier_df[(frontier_df["model_id"] == "model_a") & (np.isclose(frontier_df["target_decline_rate"], d_rate))].iloc[0]
        sub_b = frontier_df[(frontier_df["model_id"] == "model_b") & (np.isclose(frontier_df["target_decline_rate"], d_rate))].iloc[0]

        row_g = rob_df[(np.isclose(rob_df["target_decline_rate"], d_rate)) & (rob_df["attribute"] == "gender")].iloc[0]
        row_age = rob_df[(np.isclose(rob_df["target_decline_rate"], d_rate)) & (rob_df["attribute"] == "binary_age")].iloc[0]

        # Model A checks
        diffs_rob.append(abs(sub_a["threshold"] - row_g["model_a_threshold"]))
        diffs_rob.append(abs(sub_a["gender_dp_diff"] - row_g["model_a_dp_diff"]))
        diffs_rob.append(abs(sub_a["gender_dp_ratio"] - row_g["model_a_dp_ratio"]))
        diffs_rob.append(abs(sub_a["gender_eo_diff"] - row_g["model_a_eo_diff"]))
        diffs_rob.append(abs(sub_a["binary_age_dp_diff"] - row_age["model_a_dp_diff"]))
        diffs_rob.append(abs(sub_a["binary_age_dp_ratio"] - row_age["model_a_dp_ratio"]))
        diffs_rob.append(abs(sub_a["binary_age_eo_diff"] - row_age["model_a_eo_diff"]))

        # Model B checks
        diffs_rob.append(abs(sub_b["threshold"] - row_g["model_b_threshold"]))
        diffs_rob.append(abs(sub_b["gender_dp_diff"] - row_g["model_b_dp_diff"]))
        diffs_rob.append(abs(sub_b["gender_dp_ratio"] - row_g["model_b_dp_ratio"]))
        diffs_rob.append(abs(sub_b["gender_eo_diff"] - row_g["model_b_eo_diff"]))
        diffs_rob.append(abs(sub_b["binary_age_dp_diff"] - row_age["model_b_dp_diff"]))
        diffs_rob.append(abs(sub_b["binary_age_dp_ratio"] - row_age["model_b_dp_ratio"]))
        diffs_rob.append(abs(sub_b["binary_age_eo_diff"] - row_age["model_b_eo_diff"]))

    max_diff_rob = max(diffs_rob)

    # Compare 15% rows against reports/metrics/fairness.json
    with open(metrics_dir / "fairness.json", "r") as f:
        fair_json = json.load(f)

    sub_a_15 = frontier_df[(frontier_df["model_id"] == "model_a") & (np.isclose(frontier_df["target_decline_rate"], 0.15))].iloc[0]
    sub_b_15 = frontier_df[(frontier_df["model_id"] == "model_b") & (np.isclose(frontier_df["target_decline_rate"], 0.15))].iloc[0]

    diffs_json: List[float] = [
        abs(sub_a_15["threshold"] - fair_json["policy"]["threshold_a"]),
        abs(sub_b_15["threshold"] - fair_json["policy"]["threshold_b"]),
        abs(sub_a_15["realized_decline_rate"] - fair_json["policy"]["realized_test_decline_a"]),
        abs(sub_b_15["realized_decline_rate"] - fair_json["policy"]["realized_test_decline_b"]),
        abs(sub_a_15["gender_dp_diff"] - fair_json["overall"]["gender"]["model_a"]["gaps"]["dp_difference"]),
        abs(sub_a_15["gender_dp_ratio"] - fair_json["overall"]["gender"]["model_a"]["gaps"]["dp_ratio"]),
        abs(sub_a_15["gender_eo_diff"] - fair_json["overall"]["gender"]["model_a"]["gaps"]["eo_difference"]),
        abs(sub_b_15["gender_dp_diff"] - fair_json["overall"]["gender"]["model_b"]["gaps"]["dp_difference"]),
        abs(sub_b_15["gender_dp_ratio"] - fair_json["overall"]["gender"]["model_b"]["gaps"]["dp_ratio"]),
        abs(sub_b_15["gender_eo_diff"] - fair_json["overall"]["gender"]["model_b"]["gaps"]["eo_difference"]),
        abs(sub_a_15["binary_age_dp_diff"] - fair_json["overall"]["binary_age"]["model_a"]["gaps"]["dp_difference"]),
        abs(sub_a_15["binary_age_dp_ratio"] - fair_json["overall"]["binary_age"]["model_a"]["gaps"]["dp_ratio"]),
        abs(sub_a_15["binary_age_eo_diff"] - fair_json["overall"]["binary_age"]["model_a"]["gaps"]["eo_difference"]),
        abs(sub_b_15["binary_age_dp_diff"] - fair_json["overall"]["binary_age"]["model_b"]["gaps"]["dp_difference"]),
        abs(sub_b_15["binary_age_dp_ratio"] - fair_json["overall"]["binary_age"]["model_b"]["gaps"]["dp_ratio"]),
        abs(sub_b_15["binary_age_eo_diff"] - fair_json["overall"]["binary_age"]["model_b"]["gaps"]["eo_difference"]),
    ]
    max_diff_json = max(diffs_json)

    if max_diff_rob > 1e-6:
        raise ValueError(f"Frontier check failed: max difference against fairness_by_decline_rate.csv is {max_diff_rob:.2e} > 1e-6")
    if max_diff_json > 1e-6:
        raise ValueError(f"Frontier check failed: max difference against fairness.json is {max_diff_json:.2e} > 1e-6")

    meta = {
        "runtime_seconds": time.time() - t0,
        "rows": len(frontier_df),
        "file_path": str(out_csv),
        "file_size_bytes": out_csv.stat().st_size,
    }
    return frontier_df, max_diff_rob, max_diff_json, meta


def run_shap_precompute(
    config: Dict[str, Any] | None = None
) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
    """Stage 1b: Precompute Model B TreeSHAP log-odds contributions for applicant explorer.
    
    Returns:
        shap_sample_df: Parquet DataFrame with contributions and raw feature values
        max_diff_local: Maximum absolute local-accuracy difference vs model_b_test.parquet
        meta: Runtime and file metadata
    """
    t0 = time.time()
    if config is None:
        config = load_config()

    tables_dir = _PROJECT_ROOT / "reports" / "tables"
    predictions_dir = _PROJECT_ROOT / config["paths"]["predictions"]
    extras_dir = predictions_dir / "extras"
    extras_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load sample IDs
    shap_sample_ids_path = tables_dir / "shap_sample_ids.csv"
    sample_ids_df = pd.read_csv(shap_sample_ids_path)
    sample_ids_set = set(sample_ids_df["SK_ID_CURR"].astype(int))

    # 2. Prepare datasets reusing exact feature matrix code
    # Use temporary metrics dir so prepare_model_a_datasets does not touch reports/metrics/
    cfg_copy = copy.deepcopy(config)
    cfg_copy["paths"]["metrics"] = tempfile.gettempdir()
    _, _, _, _, X_test, _, _, _, meta_test, feature_names, _ = prepare_model_b_datasets(cfg_copy)

    # Filter to the sample applicants in exact index alignment with X_test
    meta_test_sample = meta_test[meta_test["SK_ID_CURR"].isin(sample_ids_set)].copy()
    sample_X = X_test.loc[meta_test_sample.index].copy()

    # 3. Load Model B and compute TreeSHAP (log-odds scale)
    model_path = _PROJECT_ROOT / "models" / "model_b.json"
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(sample_X)
    shap_values = shap_explanation.values  # shape: (N, 144)
    base_values = shap_explanation.base_values  # shape: (N,)

    # Load test predictions for local-accuracy check & scores
    test_b = pd.read_parquet(predictions_dir / "model_b_test.parquet")
    test_a = pd.read_parquet(predictions_dir / "model_a_test.parquet")
    b_score_map = dict(zip(test_b["SK_ID_CURR"].astype(int), test_b["score"].astype(float)))
    a_score_map = dict(zip(test_a["SK_ID_CURR"].astype(int), test_a["score"].astype(float)))
    target_map = dict(zip(test_b["SK_ID_CURR"].astype(int), test_b["TARGET"].astype(int)))

    # Local-accuracy check: sigmoid(bias + sum(contributions)) == score within 1e-5
    sum_contribs = np.sum(shap_values, axis=1)
    margin = base_values + sum_contribs
    computed_probs = 1.0 / (1.0 + np.exp(-margin))

    sk_ids = meta_test_sample["SK_ID_CURR"].astype(int).values
    actual_probs = np.array([b_score_map[sid] for sid in sk_ids])
    diffs = np.abs(computed_probs - actual_probs)
    max_diff_local = float(np.max(diffs))

    if max_diff_local > 1e-5:
        raise ValueError(f"Local-accuracy check failed: max difference is {max_diff_local:.2e} > 1e-5")

    # 4. Construct DataFrame for export
    # Required: contributions (one column per feature plus the bias term) and raw feature values
    out_dict: Dict[str, Any] = {
        "SK_ID_CURR": sk_ids,
        "segment": meta_test_sample["segment"].values,
        "TARGET": np.array([target_map[sid] for sid in sk_ids]),
        "score_a": np.array([a_score_map[sid] for sid in sk_ids]),
        "score_b": actual_probs,
        "bias": base_values,
    }

    # Add raw feature values
    for col in feature_names:
        out_dict[f"raw_{col}"] = sample_X[col].values

    # Add contributions (log-odds scale)
    for j, col in enumerate(feature_names):
        out_dict[f"shap_{col}"] = shap_values[:, j]

    shap_sample_df = pd.DataFrame(out_dict)
    out_parquet = extras_dir / "app_shap_sample.parquet"
    shap_sample_df.to_parquet(out_parquet, index=False)

    meta = {
        "runtime_seconds": time.time() - t0,
        "rows": len(shap_sample_df),
        "file_path": str(out_parquet),
        "file_size_bytes": out_parquet.stat().st_size,
    }
    return shap_sample_df, max_diff_local, meta


def run_join_check(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Stage 1c: Check that Model A and Model B test predictions join 1:1 on SK_ID_CURR with identical TARGET and segment."""
    t0 = time.time()
    if config is None:
        config = load_config()

    predictions_dir = _PROJECT_ROOT / config["paths"]["predictions"]
    test_a = pd.read_parquet(predictions_dir / "model_a_test.parquet")
    test_b = pd.read_parquet(predictions_dir / "model_b_test.parquet")

    if len(test_a) != len(test_b):
        raise ValueError(f"Length mismatch: Model A has {len(test_a)}, Model B has {len(test_b)}")

    merged = test_a.merge(test_b, on="SK_ID_CURR", suffixes=("_a", "_b"))
    if len(merged) != len(test_a):
        raise ValueError(f"1:1 join check failed: merged rows {len(merged)} != {len(test_a)}")

    target_mismatches = int((merged["TARGET_a"] != merged["TARGET_b"]).sum())
    segment_mismatches = int((merged["segment_a"] != merged["segment_b"]).sum())

    if target_mismatches > 0:
        raise ValueError(f"TARGET mismatch found on {target_mismatches} rows!")
    if segment_mismatches > 0:
        raise ValueError(f"Segment mismatch found on {segment_mismatches} rows!")

    return {
        "runtime_seconds": time.time() - t0,
        "total_test_rows": len(merged),
        "passed": True,
    }


def main():
    print("=" * 70)
    print("PHASE 8 STAGE 1: APP PRECOMPUTE")
    print("=" * 70)

    # 1a. Frontier table
    print("\n[Stage 1a] Computing policy frontier table...")
    frontier_df, diff_rob, diff_json, meta_1a = run_frontier_precompute()
    print(f"  Frontier rows: {meta_1a['rows']} (26 per model)")
    print(f"  Max diff vs fairness_by_decline_rate.csv: {diff_rob:.2e} (tolerance: 1e-6) -> PASS")
    print(f"  Max diff vs fairness.json: {diff_json:.2e} (tolerance: 1e-6) -> PASS")
    print(f"  Saved {meta_1a['file_path']} ({meta_1a['file_size_bytes']:,} bytes) in {meta_1a['runtime_seconds']:.2f}s")

    # 1b. SHAP sample precompute
    print("\n[Stage 1b] Computing Model B TreeSHAP log-odds contributions...")
    shap_df, diff_local, meta_1b = run_shap_precompute()
    print(f"  SHAP sample rows: {meta_1b['rows']}")
    print(f"  Local-accuracy max diff: {diff_local:.2e} (tolerance: 1e-5) -> PASS")
    print(f"  Saved {meta_1b['file_path']} ({meta_1b['file_size_bytes']:,} bytes) in {meta_1b['runtime_seconds']:.2f}s")

    # 1c. 1:1 join check
    print("\n[Stage 1c] Checking 1:1 join on SK_ID_CURR...")
    meta_1c = run_join_check()
    print(f"  Checked {meta_1c['total_test_rows']} test predictions: identical TARGET and segment -> PASS in {meta_1c['runtime_seconds']:.2f}s")

    print("\n" + "=" * 70)
    print("STAGE 1 COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
