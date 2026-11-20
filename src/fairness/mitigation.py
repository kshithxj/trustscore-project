"""Part 2: Fairness Mitigation with Fairlearn ThresholdOptimizer for Model B.

Applies postprocessing threshold optimization using Fairlearn 0.15.0:
- Uses an already-fitted Model B wrapped in Float64XGBWrapper with prefit=True.
- Verifies score matching against saved predictions (reports max absolute difference ~0).
- Fixes objective: balanced_accuracy_score.
- Fixes predict_method: predict_proba.
- Fixes random_state: 42 (predictions can be randomized by interpolation).
- Evaluates constraints: equalized_odds and demographic_parity.
- Evaluates sensitive features: gender (excluding XNA) and binary_age (<median vs >=median).
- Fits on VALIDATION split, evaluates ONCE on TEST split.
- Generates tradeoff comparison with unmitigated Model B (Phase 4 15% policy threshold 0.144659).
- Computes 95% stratified bootstrap CIs (test-sample variation only, thresholds fixed after validation).
- Saves reports/metrics/mitigation.json and reports/tables/tradeoff.csv.

Disclaimers:
(1) Mitigation changes the decline rate, representing a fairness-accuracy trade-off, not a comparison at a fixed 15% decline rate.
(2) Underlying score AUC is identical across rows by construction because only decision thresholds move.
(3) CIs reflect test sampling variation only (thresholds are fixed post-validation).
(4) Using gender or age at decision time is legally and ethically contested; this is an experimental simulation.
(5) If mitigation yields little or no improvement, that is faithfully recorded as the result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import fairlearn.postprocessing as fpp
import xgboost as xgb

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config
from src.features.model_b_features import prepare_model_b_datasets


class Float64XGBWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper around XGBClassifier ensuring float64 predict_proba for Fairlearn/pandas compatibility."""

    def __init__(self, model: xgb.XGBClassifier):
        self.model = model
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X).astype(np.float64)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "Float64XGBWrapper":
        # Model is already fitted (prefit=True)
        return self


def compute_policy_metrics(
    y_true: np.ndarray,
    preds: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    group_names: List[str],
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for a decision policy."""
    n_total = len(y_true)
    decline_rate = float(np.mean(preds == 1))
    
    # Defaults and TPR
    n_defaults = int(np.sum(y_true == 1))
    default_capture = float(np.sum((y_true == 1) & (preds == 1)) / n_defaults) if n_defaults > 0 else 0.0
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

    # Gaps between the two groups (group_names[0] vs group_names[1])
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
        "balanced_accuracy": bal_acc,
        "auc": auc,
        "group_metrics": group_metrics,
        "demographic_parity_difference": dp_diff,
        "demographic_parity_ratio": dp_ratio,
        "equalized_odds_difference": eo_diff,
    }


def run_mitigation_experiments(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if config is None:
        config = load_config()

    seed = config["seed"]
    bootstrap_n = config["bootstrap_n"]
    tables_dir = _PROJECT_ROOT / config["paths"]["reports"] / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = _PROJECT_ROOT / config["paths"]["metrics"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data and features
    print("[mitigation] Preparing datasets and loading Model B...")
    X_train, y_train, X_val, y_val, X_test, y_test, meta_tr, meta_va, meta_te, feature_names, cat_maps = (
        prepare_model_b_datasets(config)
    )

    model_path = _PROJECT_ROOT / "models" / "model_b.json"
    raw_model = xgb.XGBClassifier()
    raw_model.load_model(str(model_path))
    wrapped_model = Float64XGBWrapper(raw_model)

    # 2. Score match verification
    val_preds_saved = pd.read_parquet(_PROJECT_ROOT / config["paths"]["predictions"] / "model_b_val.parquet")
    test_preds_saved = pd.read_parquet(_PROJECT_ROOT / config["paths"]["predictions"] / "model_b_test.parquet")

    val_scores = wrapped_model.predict_proba(X_val)[:, 1]
    test_scores = wrapped_model.predict_proba(X_test)[:, 1]

    val_merged = pd.merge(val_preds_saved, pd.DataFrame({"SK_ID_CURR": meta_va["SK_ID_CURR"].values, "score_w": val_scores}), on="SK_ID_CURR")
    test_merged = pd.merge(test_preds_saved, pd.DataFrame({"SK_ID_CURR": meta_te["SK_ID_CURR"].values, "score_w": test_scores}), on="SK_ID_CURR")

    val_max_diff = float(np.max(np.abs(val_merged["score"] - val_merged["score_w"])))
    test_max_diff = float(np.max(np.abs(test_merged["score"] - test_merged["score_w"])))

    print(f"[mitigation] Score match check: Val max diff = {val_max_diff:.8f}, Test max diff = {test_max_diff:.8f}")
    if val_max_diff > 1e-5 or test_max_diff > 1e-5:
        raise ValueError(f"Score match failed! Val max diff: {val_max_diff}, Test max diff: {test_max_diff}")

    # Load demographic attributes
    eval_attrs = pd.read_parquet(_PROJECT_ROOT / config["paths"]["processed"] / "eval_attrs.parquet")
    meta_va = meta_va.merge(eval_attrs[["SK_ID_CURR", "gender", "age_years"]], on="SK_ID_CURR", how="left")
    meta_te = meta_te.merge(eval_attrs[["SK_ID_CURR", "gender", "age_years"]], on="SK_ID_CURR", how="left")

    train_ids = pd.read_csv(_PROJECT_ROOT / config["paths"]["splits"] / "train_ids.csv")["SK_ID_CURR"]
    train_median_age = float(eval_attrs[eval_attrs["SK_ID_CURR"].isin(train_ids)]["age_years"].median())

    meta_va["binary_age"] = np.where(meta_va["age_years"] >= train_median_age, ">=median", "<median")
    meta_te["binary_age"] = np.where(meta_te["age_years"] >= train_median_age, ">=median", "<median")

    # Read unmitigated threshold from fairness.json
    with open(metrics_dir / "fairness.json", "r") as f:
        fairness_data = json.load(f)
    threshold_b = float(fairness_data["policy"]["threshold_b"])

    # Sensitive attributes setup
    experiments = [
        ("gender", ["F", "M"], meta_va["gender"] != "XNA", meta_te["gender"] != "XNA"),
        ("binary_age", ["<median", ">=median"], np.ones(len(meta_va), dtype=bool), np.ones(len(meta_te), dtype=bool)),
    ]

    objective = "balanced_accuracy_score"
    random_state = seed
    constraints_list = ["equalized_odds", "demographic_parity"]

    tradeoff_rows: List[Dict[str, Any]] = []
    mitigation_results: Dict[str, Any] = {
        "metadata": {
            "fairlearn_version": "0.15.0",
            "prefit_supported": True,
            "estimator_handling": "Wrapped Model B in Float64XGBWrapper with prefit=True; no refitting performed",
            "score_match_max_diff": {"validation": val_max_diff, "test": test_max_diff},
            "objective": objective,
            "random_state": random_state,
            "unmitigated_threshold": threshold_b,
            "train_median_age": train_median_age,
            "test_xna_count": int(np.sum(meta_te["gender"] == "XNA")),
            "val_xna_count": int(np.sum(meta_va["gender"] == "XNA")),
            "bootstrap_n": bootstrap_n,
            "disclaimers": [
                "Mitigation changes the decline rate, representing a fairness-accuracy trade-off, not a comparison at a fixed 15% decline rate.",
                "AUC of underlying score is identical across rows by construction because only decision thresholds move.",
                "CIs reflect test sampling variation only (thresholds are fixed post-validation).",
                "Using gender or age at decision time is legally and ethically contested; this is an experiment only.",
                "If mitigation yields little or no improvement, that is faithfully recorded as the result.",
            ],
        },
        "experiments": {},
    }

    rng = np.random.RandomState(seed)

    for attr_name, group_labels, va_mask, te_mask in experiments:
        print(f"\n[mitigation] Running experiments for sensitive attribute: {attr_name}...")
        va_X = X_val[va_mask]
        va_y = y_val[va_mask].values
        va_sens = meta_va.loc[va_mask, attr_name].values

        te_X = X_test[te_mask]
        te_y = y_test[te_mask].values
        te_scores = test_scores[te_mask]
        te_sens = meta_te.loc[te_mask, attr_name].values
        te_n = len(te_y)

        # 1. Unmitigated policy (15% validation threshold)
        unmit_preds = (te_scores >= threshold_b).astype(int)

        # 2. Mitigated: equalized_odds
        print(f"[mitigation] Fitting ThresholdOptimizer with equalized_odds on validation...")
        opt_eo = fpp.ThresholdOptimizer(
            estimator=wrapped_model,
            constraints="equalized_odds",
            objective=objective,
            prefit=True,
            predict_method="predict_proba",
        )
        opt_eo.fit(va_X, va_y, sensitive_features=va_sens)
        eo_preds = opt_eo.predict(te_X, sensitive_features=te_sens, random_state=random_state).astype(int)

        # 3. Mitigated: demographic_parity
        print(f"[mitigation] Fitting ThresholdOptimizer with demographic_parity on validation...")
        opt_dp = fpp.ThresholdOptimizer(
            estimator=wrapped_model,
            constraints="demographic_parity",
            objective=objective,
            prefit=True,
            predict_method="predict_proba",
        )
        opt_dp.fit(va_X, va_y, sensitive_features=va_sens)
        dp_preds = opt_dp.predict(te_X, sensitive_features=te_sens, random_state=random_state).astype(int)

        policies = [
            ("unmitigated", unmit_preds),
            ("equalized_odds", eo_preds),
            ("demographic_parity", dp_preds),
        ]

        attr_results: Dict[str, Any] = {}

        # Precompute group indices for stratified bootstrap
        g0_idx = np.where(te_sens == group_labels[0])[0]
        g1_idx = np.where(te_sens == group_labels[1])[0]

        for pol_name, pol_preds in policies:
            print(f"[mitigation] Evaluating {pol_name} for {attr_name}...")
            point_metrics = compute_policy_metrics(te_y, pol_preds, te_scores, te_sens, group_labels)

            # Stratified bootstrap for 95% CIs
            boot_metrics: Dict[str, List[float]] = {
                "decline_rate": [],
                "default_capture": [],
                "balanced_accuracy": [],
                "demographic_parity_difference": [],
                "demographic_parity_ratio": [],
                "equalized_odds_difference": [],
                f"approval_rate_{group_labels[0]}": [],
                f"approval_rate_{group_labels[1]}": [],
                f"tpr_{group_labels[0]}": [],
                f"tpr_{group_labels[1]}": [],
                f"fpr_{group_labels[0]}": [],
                f"fpr_{group_labels[1]}": [],
            }

            for _ in range(bootstrap_n):
                sample_g0 = rng.choice(g0_idx, size=len(g0_idx), replace=True)
                sample_g1 = rng.choice(g1_idx, size=len(g1_idx), replace=True)
                boot_idx = np.concatenate([sample_g0, sample_g1])

                b_y = te_y[boot_idx]
                b_pred = pol_preds[boot_idx]
                b_score = te_scores[boot_idx]
                b_sens = te_sens[boot_idx]

                b_m = compute_policy_metrics(b_y, b_pred, b_score, b_sens, group_labels)
                boot_metrics["decline_rate"].append(b_m["decline_rate"])
                boot_metrics["default_capture"].append(b_m["default_capture"])
                boot_metrics["balanced_accuracy"].append(b_m["balanced_accuracy"])
                boot_metrics["demographic_parity_difference"].append(b_m["demographic_parity_difference"])
                boot_metrics["demographic_parity_ratio"].append(b_m["demographic_parity_ratio"])
                boot_metrics["equalized_odds_difference"].append(b_m["equalized_odds_difference"])
                boot_metrics[f"approval_rate_{group_labels[0]}"].append(b_m["group_metrics"][group_labels[0]]["approval_rate"])
                boot_metrics[f"approval_rate_{group_labels[1]}"].append(b_m["group_metrics"][group_labels[1]]["approval_rate"])
                boot_metrics[f"tpr_{group_labels[0]}"].append(b_m["group_metrics"][group_labels[0]]["tpr"])
                boot_metrics[f"tpr_{group_labels[1]}"].append(b_m["group_metrics"][group_labels[1]]["tpr"])
                boot_metrics[f"fpr_{group_labels[0]}"].append(b_m["group_metrics"][group_labels[0]]["fpr"])
                boot_metrics[f"fpr_{group_labels[1]}"].append(b_m["group_metrics"][group_labels[1]]["fpr"])

            ci_dict: Dict[str, List[float]] = {}
            for k, v in boot_metrics.items():
                ci_dict[k] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

            point_metrics["confidence_intervals_95"] = ci_dict
            attr_results[pol_name] = point_metrics

            # Prepare row for tradeoff.csv
            row_dict = {
                "sensitive_feature": attr_name,
                "policy": pol_name,
                "decline_rate": point_metrics["decline_rate"],
                "decline_rate_ci_lower": ci_dict["decline_rate"][0],
                "decline_rate_ci_upper": ci_dict["decline_rate"][1],
                "default_capture": point_metrics["default_capture"],
                "default_capture_ci_lower": ci_dict["default_capture"][0],
                "default_capture_ci_upper": ci_dict["default_capture"][1],
                "balanced_accuracy": point_metrics["balanced_accuracy"],
                "balanced_accuracy_ci_lower": ci_dict["balanced_accuracy"][0],
                "balanced_accuracy_ci_upper": ci_dict["balanced_accuracy"][1],
                "auc": point_metrics["auc"],
                "dp_difference": point_metrics["demographic_parity_difference"],
                "dp_difference_ci_lower": ci_dict["demographic_parity_difference"][0],
                "dp_difference_ci_upper": ci_dict["demographic_parity_difference"][1],
                "dp_ratio": point_metrics["demographic_parity_ratio"],
                "dp_ratio_ci_lower": ci_dict["demographic_parity_ratio"][0],
                "dp_ratio_ci_upper": ci_dict["demographic_parity_ratio"][1],
                "eo_difference": point_metrics["equalized_odds_difference"],
                "eo_difference_ci_lower": ci_dict["equalized_odds_difference"][0],
                "eo_difference_ci_upper": ci_dict["equalized_odds_difference"][1],
                f"approval_rate_{group_labels[0]}": point_metrics["group_metrics"][group_labels[0]]["approval_rate"],
                f"approval_rate_{group_labels[1]}": point_metrics["group_metrics"][group_labels[1]]["approval_rate"],
                f"tpr_{group_labels[0]}": point_metrics["group_metrics"][group_labels[0]]["tpr"],
                f"tpr_{group_labels[1]}": point_metrics["group_metrics"][group_labels[1]]["tpr"],
                f"fpr_{group_labels[0]}": point_metrics["group_metrics"][group_labels[0]]["fpr"],
                f"fpr_{group_labels[1]}": point_metrics["group_metrics"][group_labels[1]]["fpr"],
            }
            tradeoff_rows.append(row_dict)

        mitigation_results["experiments"][attr_name] = attr_results

    # 4. Save mitigation.json and tradeoff.csv
    mitigation_json_path = metrics_dir / "mitigation.json"
    with open(mitigation_json_path, "w") as f:
        json.dump(mitigation_results, f, indent=2)
    print(f"[mitigation] Saved mitigation metrics to {mitigation_json_path}")

    tradeoff_df = pd.DataFrame(tradeoff_rows)
    tradeoff_csv_path = tables_dir / "tradeoff.csv"
    tradeoff_df.to_csv(tradeoff_csv_path, index=False)
    print(f"[mitigation] Saved tradeoff table to {tradeoff_csv_path}")

    return mitigation_results


if __name__ == "__main__":
    run_mitigation_experiments()
