"""Phase 5b: Matched-Decline-Rate Control for TrustScore Mitigation Comparison.

Compares Fairlearn ThresholdOptimizer mitigated policies against an unmitigated Model B control
calibrated to the exact same validation decline rate using a single decision threshold:
- Runs for:
  1. Gender - Equalized Odds
  2. Gender - Demographic Parity
  3. Binary Age - Equalized Odds
  4. Binary Age - Demographic Parity
- Verifies exact reproduction of test metrics from mitigation.json (~0 max abs diff).
- Calibrates a single-threshold unmitigated control on VALIDATION so that validation decline rate
  matches the optimizer's validation decline rate (r_val).
- Evaluates mitigated policy and single-threshold control on TEST side-by-side.
- Computes 95% stratified bootstrap CIs for all metrics and paired differences (mitigated minus control)
  using identical resampled rows across 200 iterations.
- Saves reports/metrics/mitigation_control.json and reports/tables/tradeoff_matched.csv.

Methodological Disclaimers:
(a) The control uses a single threshold and no sensitive attribute at decision time.
(b) Conclusions about fairness gains must be read from paired differences vs the matched control,
    not vs the arbitrary 15% decline policy.
(c) CIs reflect test-sample evaluation uncertainty only (thresholds are fixed post-validation).
(d) Using gender or age at decision time is legally and ethically contested and may be unlawful in
    underwriting; this is an experimental simulation.
(e) If mitigation provides little or no improvement over the matched-rate control, that is recorded
    faithfully as the result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import fairlearn.postprocessing as fpp
import xgboost as xgb

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config
from src.features.model_b_features import prepare_model_b_datasets
from src.fairness.mitigation import Float64XGBWrapper


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    preds: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    group_names: List[str],
) -> Dict[str, Any]:
    """Compute all evaluation metrics required for matched-rate policy comparison."""
    n_total = len(y_true)
    decline_rate = float(np.mean(preds == 1))

    # Defaults and errors
    n_defaults = int(np.sum(y_true == 1))
    n_non_defaults = int(np.sum(y_true == 0))

    default_capture = float(np.sum((y_true == 1) & (preds == 1)) / n_defaults) if n_defaults > 0 else 0.0
    overall_fpr = float(np.sum((y_true == 0) & (preds == 1)) / n_non_defaults) if n_non_defaults > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, preds))
    auc = float(roc_auc_score(y_true, scores))

    group_metrics: Dict[str, Dict[str, float]] = {}
    for g in group_names:
        g_mask = groups == g
        g_n = int(np.sum(g_mask))
        g_y = y_true[g_mask]
        g_pred = preds[g_mask]

        g_decline = float(np.mean(g_pred == 1)) if g_n > 0 else 0.0
        g_approval = float(1.0 - g_decline)

        n_def_g = int(np.sum(g_y == 1))
        n_non_def_g = int(np.sum(g_y == 0))

        tpr_g = float(np.sum((g_y == 1) & (g_pred == 1)) / n_def_g) if n_def_g > 0 else 0.0
        fpr_g = float(np.sum((g_y == 0) & (g_pred == 1)) / n_non_def_g) if n_non_def_g > 0 else 0.0

        group_metrics[g] = {
            "size": g_n,
            "approval_rate": g_approval,
            "tpr": tpr_g,
            "fpr": fpr_g,
        }

    g0 = group_metrics[group_names[0]]
    g1 = group_metrics[group_names[1]]

    dp_diff = float(abs(g0["approval_rate"] - g1["approval_rate"]))
    min_app = min(g0["approval_rate"], g1["approval_rate"])
    max_app = max(g0["approval_rate"], g1["approval_rate"])
    dp_ratio = float(min_app / max_app) if max_app > 0 else 0.0

    tpr_gap = abs(g0["tpr"] - g1["tpr"])
    fpr_gap = abs(g0["fpr"] - g1["fpr"])
    eo_diff = float(max(tpr_gap, fpr_gap))

    return {
        "decline_rate": decline_rate,
        "default_capture": default_capture,
        "overall_fpr": overall_fpr,
        "balanced_accuracy": bal_acc,
        "auc": auc,
        "group_metrics": group_metrics,
        "demographic_parity_difference": dp_diff,
        "demographic_parity_ratio": dp_ratio,
        "equalized_odds_difference": eo_diff,
    }


def run_matched_control_experiments(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if config is None:
        config = load_config()

    seed = config["seed"]
    bootstrap_n = config["bootstrap_n"]
    tables_dir = _PROJECT_ROOT / config["paths"]["reports"] / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = _PROJECT_ROOT / config["paths"]["metrics"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data and wrapped Model B
    print("[matched_control] Loading datasets and Model B...")
    X_train, y_train, X_val, y_val, X_test, y_test, meta_tr, meta_va, meta_te, feature_names, cat_maps = (
        prepare_model_b_datasets(config)
    )

    model_path = _PROJECT_ROOT / "models" / "model_b.json"
    raw_model = xgb.XGBClassifier()
    raw_model.load_model(str(model_path))
    wrapped_model = Float64XGBWrapper(raw_model)

    val_scores = wrapped_model.predict_proba(X_val)[:, 1]
    test_scores = wrapped_model.predict_proba(X_test)[:, 1]

    # Demographic attributes
    eval_attrs = pd.read_parquet(_PROJECT_ROOT / config["paths"]["processed"] / "eval_attrs.parquet")
    meta_va = meta_va.merge(eval_attrs[["SK_ID_CURR", "gender", "age_years"]], on="SK_ID_CURR", how="left")
    meta_te = meta_te.merge(eval_attrs[["SK_ID_CURR", "gender", "age_years"]], on="SK_ID_CURR", how="left")

    train_ids = pd.read_csv(_PROJECT_ROOT / config["paths"]["splits"] / "train_ids.csv")["SK_ID_CURR"]
    train_median_age = float(eval_attrs[eval_attrs["SK_ID_CURR"].isin(train_ids)]["age_years"].median())

    meta_va["binary_age"] = np.where(meta_va["age_years"] >= train_median_age, ">=median", "<median")
    meta_te["binary_age"] = np.where(meta_te["age_years"] >= train_median_age, ">=median", "<median")

    # Load original mitigation metrics for reproduction verification
    with open(metrics_dir / "mitigation.json", "r") as f:
        orig_mitigation = json.load(f)

    experiments = [
        ("gender", ["F", "M"], meta_va["gender"] != "XNA", meta_te["gender"] != "XNA"),
        ("binary_age", ["<median", ">=median"], np.ones(len(meta_va), dtype=bool), np.ones(len(meta_te), dtype=bool)),
    ]

    objective = "balanced_accuracy_score"
    random_state = seed
    constraints_list = ["equalized_odds", "demographic_parity"]

    reproduction_checks: Dict[str, Dict[str, float]] = {}
    matched_results: Dict[str, Any] = {
        "metadata": {
            "objective": objective,
            "random_state": random_state,
            "bootstrap_n": bootstrap_n,
            "train_median_age": train_median_age,
            "test_xna_count": int(np.sum(meta_te["gender"] == "XNA")),
            "val_xna_count": int(np.sum(meta_va["gender"] == "XNA")),
            "disclaimers": [
                "The control uses a single threshold and no sensitive attribute at decision time.",
                "Conclusions about fairness gains must be read from the paired differences vs the control, not vs the 15% row.",
                "CIs reflect test sampling only.",
                "Using gender or age at decision time is contested and may be unlawful in some jurisdictions, and this is an experiment only.",
                "If the mitigation gives little or no improvement over the control, report that as the result.",
            ],
        },
        "reproduction_verification": {},
        "runs": {},
    }

    tradeoff_matched_rows: List[Dict[str, Any]] = []
    rng = np.random.RandomState(seed)

    max_repro_decline_diff = 0.0
    max_repro_dp_diff = 0.0

    for attr_name, group_labels, va_mask, te_mask in experiments:
        va_X = X_val[va_mask]
        va_y = y_val[va_mask].values
        va_sens = meta_va.loc[va_mask, attr_name].values
        va_scores_sub = val_scores[va_mask]

        te_X = X_test[te_mask]
        te_y = y_test[te_mask].values
        te_scores_sub = test_scores[te_mask]
        te_sens = meta_te.loc[te_mask, attr_name].values

        g0_idx = np.where(te_sens == group_labels[0])[0]
        g1_idx = np.where(te_sens == group_labels[1])[0]

        for constraint in constraints_list:
            run_key = f"{attr_name}_{constraint}"
            print(f"\n[matched_control] Running run: {run_key}...")

            # 1. Fit ThresholdOptimizer on validation
            opt = fpp.ThresholdOptimizer(
                estimator=wrapped_model,
                constraints=constraint,
                objective=objective,
                prefit=True,
                predict_method="predict_proba",
            )
            opt.fit(va_X, va_y, sensitive_features=va_sens)

            # Predict on validation and test
            va_mit_pred = opt.predict(va_X, sensitive_features=va_sens, random_state=random_state).astype(int)
            te_mit_pred = opt.predict(te_X, sensitive_features=te_sens, random_state=random_state).astype(int)

            r_val = float(np.mean(va_mit_pred == 1))
            r_test_mit = float(np.mean(te_mit_pred == 1))

            # Reproduction check against mitigation.json
            orig_run = orig_mitigation["experiments"][attr_name][constraint]
            orig_decline = orig_run["decline_rate"]
            orig_dp = orig_run["demographic_parity_difference"]

            mit_metrics = compute_comprehensive_metrics(te_y, te_mit_pred, te_scores_sub, te_sens, group_labels)

            diff_decline = abs(mit_metrics["decline_rate"] - orig_decline)
            diff_dp = abs(mit_metrics["demographic_parity_difference"] - orig_dp)

            if diff_decline > max_repro_decline_diff:
                max_repro_decline_diff = diff_decline
            if diff_dp > max_repro_dp_diff:
                max_repro_dp_diff = diff_dp

            print(f"  Reproduction check: decline diff = {diff_decline:.8f}, DP diff = {diff_dp:.8f}")
            if diff_decline > 1e-5 or diff_dp > 1e-5:
                raise ValueError(f"Reproduction check failed for {run_key}! diff_decline={diff_decline}, diff_dp={diff_dp}")

            matched_results["reproduction_verification"][run_key] = {
                "mitigated_test_decline_reproduced": mit_metrics["decline_rate"],
                "mitigated_test_decline_orig": orig_decline,
                "abs_diff_decline": diff_decline,
                "mitigated_test_dp_reproduced": mit_metrics["demographic_parity_difference"],
                "mitigated_test_dp_orig": orig_dp,
                "abs_diff_dp": diff_dp,
            }

            # 2. Control single threshold chosen on VALIDATION to match r_val
            # Top r_val fraction declined -> score >= quantile(1 - r_val)
            control_threshold = float(np.quantile(va_scores_sub, 1.0 - r_val))
            va_ctrl_pred = (va_scores_sub >= control_threshold).astype(int)
            va_ctrl_decline = float(np.mean(va_ctrl_pred == 1))

            te_ctrl_pred = (te_scores_sub >= control_threshold).astype(int)
            te_ctrl_decline = float(np.mean(te_ctrl_pred == 1))

            print(f"  Validation decline rate: mitigated = {r_val:.6f}, control = {va_ctrl_decline:.6f} (threshold = {control_threshold:.6f})")
            print(f"  Test decline rate:       mitigated = {r_test_mit:.6f}, control = {te_ctrl_decline:.6f}")

            ctrl_metrics = compute_comprehensive_metrics(te_y, te_ctrl_pred, te_scores_sub, te_sens, group_labels)

            # 3. Stratified Paired Bootstrap (200 iterations)
            # Evaluates both policies on identical resampled test sets to compute paired differences
            paired_diffs: Dict[str, List[float]] = {
                "dp_difference": [],
                "eo_difference": [],
                "balanced_accuracy": [],
                "default_capture": [],
                "overall_fpr": [],
            }

            mit_boot_metrics: Dict[str, List[float]] = {
                "decline_rate": [], "default_capture": [], "overall_fpr": [], "balanced_accuracy": [],
                "dp_difference": [], "dp_ratio": [], "eo_difference": [],
                f"approval_rate_{group_labels[0]}": [], f"approval_rate_{group_labels[1]}": [],
                f"tpr_{group_labels[0]}": [], f"tpr_{group_labels[1]}": [],
                f"fpr_{group_labels[0]}": [], f"fpr_{group_labels[1]}": [],
            }
            ctrl_boot_metrics: Dict[str, List[float]] = {
                "decline_rate": [], "default_capture": [], "overall_fpr": [], "balanced_accuracy": [],
                "dp_difference": [], "dp_ratio": [], "eo_difference": [],
                f"approval_rate_{group_labels[0]}": [], f"approval_rate_{group_labels[1]}": [],
                f"tpr_{group_labels[0]}": [], f"tpr_{group_labels[1]}": [],
                f"fpr_{group_labels[0]}": [], f"fpr_{group_labels[1]}": [],
            }

            for _ in range(bootstrap_n):
                sample_g0 = rng.choice(g0_idx, size=len(g0_idx), replace=True)
                sample_g1 = rng.choice(g1_idx, size=len(g1_idx), replace=True)
                boot_idx = np.concatenate([sample_g0, sample_g1])

                b_y = te_y[boot_idx]
                b_sens = te_sens[boot_idx]
                b_scores = te_scores_sub[boot_idx]

                b_mit_p = te_mit_pred[boot_idx]
                b_ctrl_p = te_ctrl_pred[boot_idx]

                m_m = compute_comprehensive_metrics(b_y, b_mit_p, b_scores, b_sens, group_labels)
                c_m = compute_comprehensive_metrics(b_y, b_ctrl_p, b_scores, b_sens, group_labels)

                # Paired differences: mitigated minus control
                paired_diffs["dp_difference"].append(m_m["demographic_parity_difference"] - c_m["demographic_parity_difference"])
                paired_diffs["eo_difference"].append(m_m["equalized_odds_difference"] - c_m["equalized_odds_difference"])
                paired_diffs["balanced_accuracy"].append(m_m["balanced_accuracy"] - c_m["balanced_accuracy"])
                paired_diffs["default_capture"].append(m_m["default_capture"] - c_m["default_capture"])
                paired_diffs["overall_fpr"].append(m_m["overall_fpr"] - c_m["overall_fpr"])

                # Individual metrics
                for key_name in ["decline_rate", "default_capture", "overall_fpr", "balanced_accuracy"]:
                    mit_boot_metrics[key_name].append(m_m[key_name])
                    ctrl_boot_metrics[key_name].append(c_m[key_name])

                mit_boot_metrics["eo_difference"].append(m_m["equalized_odds_difference"])
                ctrl_boot_metrics["eo_difference"].append(c_m["equalized_odds_difference"])
                mit_boot_metrics["dp_difference"].append(m_m["demographic_parity_difference"])
                ctrl_boot_metrics["dp_difference"].append(c_m["demographic_parity_difference"])
                mit_boot_metrics["dp_ratio"].append(m_m["demographic_parity_ratio"])
                ctrl_boot_metrics["dp_ratio"].append(c_m["demographic_parity_ratio"])

                for gl in group_labels:
                    mit_boot_metrics[f"approval_rate_{gl}"].append(m_m["group_metrics"][gl]["approval_rate"])
                    ctrl_boot_metrics[f"approval_rate_{gl}"].append(c_m["group_metrics"][gl]["approval_rate"])
                    mit_boot_metrics[f"tpr_{gl}"].append(m_m["group_metrics"][gl]["tpr"])
                    ctrl_boot_metrics[f"tpr_{gl}"].append(c_m["group_metrics"][gl]["tpr"])
                    mit_boot_metrics[f"fpr_{gl}"].append(m_m["group_metrics"][gl]["fpr"])
                    ctrl_boot_metrics[f"fpr_{gl}"].append(c_m["group_metrics"][gl]["fpr"])

            mit_ci: Dict[str, List[float]] = {}
            for k, v in mit_boot_metrics.items():
                mit_ci[k] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

            ctrl_ci: Dict[str, List[float]] = {}
            for k, v in ctrl_boot_metrics.items():
                ctrl_ci[k] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

            paired_ci: Dict[str, List[float]] = {}
            paired_point: Dict[str, float] = {
                "dp_difference": mit_metrics["demographic_parity_difference"] - ctrl_metrics["demographic_parity_difference"],
                "eo_difference": mit_metrics["equalized_odds_difference"] - ctrl_metrics["equalized_odds_difference"],
                "balanced_accuracy": mit_metrics["balanced_accuracy"] - ctrl_metrics["balanced_accuracy"],
                "default_capture": mit_metrics["default_capture"] - ctrl_metrics["default_capture"],
                "overall_fpr": mit_metrics["overall_fpr"] - ctrl_metrics["overall_fpr"],
            }
            for k, v in paired_diffs.items():
                paired_ci[k] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

            mit_metrics["confidence_intervals_95"] = mit_ci
            ctrl_metrics["confidence_intervals_95"] = ctrl_ci
            ctrl_metrics["control_threshold"] = control_threshold
            ctrl_metrics["validation_decline_rate"] = va_ctrl_decline

            run_dict = {
                "sensitive_feature": attr_name,
                "constraint": constraint,
                "validation_target_decline_rate": r_val,
                "control_threshold": control_threshold,
                "mitigated": mit_metrics,
                "control": ctrl_metrics,
                "paired_difference_mitigated_minus_control": {
                    "point_estimates": paired_point,
                    "confidence_intervals_95": paired_ci,
                },
            }
            matched_results["runs"][run_key] = run_dict

            # Add rows to tradeoff_matched.csv
            for pol_label, m_dict, ci_dict in [("mitigated", mit_metrics, mit_ci), ("control", ctrl_metrics, ctrl_ci)]:
                row = {
                    "sensitive_feature": attr_name,
                    "constraint": constraint,
                    "policy_variant": pol_label,
                    "validation_decline_rate": r_val if pol_label == "mitigated" else va_ctrl_decline,
                    "control_threshold": None if pol_label == "mitigated" else control_threshold,
                    "decline_rate": m_dict["decline_rate"],
                    "decline_rate_ci_lower": ci_dict["decline_rate"][0],
                    "decline_rate_ci_upper": ci_dict["decline_rate"][1],
                    "default_capture": m_dict["default_capture"],
                    "default_capture_ci_lower": ci_dict["default_capture"][0],
                    "default_capture_ci_upper": ci_dict["default_capture"][1],
                    "overall_fpr": m_dict["overall_fpr"],
                    "overall_fpr_ci_lower": ci_dict["overall_fpr"][0],
                    "overall_fpr_ci_upper": ci_dict["overall_fpr"][1],
                    "balanced_accuracy": m_dict["balanced_accuracy"],
                    "balanced_accuracy_ci_lower": ci_dict["balanced_accuracy"][0],
                    "balanced_accuracy_ci_upper": ci_dict["balanced_accuracy"][1],
                    "auc": m_dict["auc"],
                    "dp_difference": m_dict["demographic_parity_difference"],
                    "dp_difference_ci_lower": ci_dict["dp_difference"][0],
                    "dp_difference_ci_upper": ci_dict["dp_difference"][1],
                    "dp_ratio": m_dict["demographic_parity_ratio"],
                    "dp_ratio_ci_lower": ci_dict["dp_ratio"][0],
                    "dp_ratio_ci_upper": ci_dict["dp_ratio"][1],
                    "eo_difference": m_dict["equalized_odds_difference"],
                    "eo_difference_ci_lower": ci_dict["eo_difference"][0],
                    "eo_difference_ci_upper": ci_dict["eo_difference"][1],
                    f"approval_rate_{group_labels[0]}": m_dict["group_metrics"][group_labels[0]]["approval_rate"],
                    f"approval_rate_{group_labels[1]}": m_dict["group_metrics"][group_labels[1]]["approval_rate"],
                    f"tpr_{group_labels[0]}": m_dict["group_metrics"][group_labels[0]]["tpr"],
                    f"tpr_{group_labels[1]}": m_dict["group_metrics"][group_labels[1]]["tpr"],
                    f"fpr_{group_labels[0]}": m_dict["group_metrics"][group_labels[0]]["fpr"],
                    f"fpr_{group_labels[1]}": m_dict["group_metrics"][group_labels[1]]["fpr"],
                }
                tradeoff_matched_rows.append(row)

    matched_results["metadata"]["max_reproduction_abs_diff"] = {
        "decline_rate": max_repro_decline_diff,
        "dp_difference": max_repro_dp_diff,
    }

    # 4. Save artifacts
    control_json_path = metrics_dir / "mitigation_control.json"
    with open(control_json_path, "w") as f:
        json.dump(matched_results, f, indent=2)
    print(f"\n[matched_control] Saved {control_json_path}")

    tradeoff_matched_df = pd.DataFrame(tradeoff_matched_rows)
    tradeoff_matched_path = tables_dir / "tradeoff_matched.csv"
    tradeoff_matched_df.to_csv(tradeoff_matched_path, index=False)
    print(f"[matched_control] Saved {tradeoff_matched_path} ({len(tradeoff_matched_df)} rows)")

    return matched_results


if __name__ == "__main__":
    run_matched_control_experiments()
