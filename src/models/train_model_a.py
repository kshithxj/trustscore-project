"""Train and evaluate Model A (bureau-style baseline).

Usage:
    python -m src.models.train_model_a
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config, set_seed
from src.features.model_a_features import prepare_model_a_datasets


def compute_bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstraps: int = 200,
    seed: int = 42,
    alpha: float = 0.05
) -> Tuple[float, float, int]:
    """Compute percentile bootstrap confidence interval for ROC-AUC.
    
    Returns:
        ci_lower, ci_upper, skipped_count
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_aucs = []
    skipped = 0

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        sample_y = y_true[idx]
        # Skip if single class
        if len(np.unique(sample_y)) < 2:
            skipped += 1
            continue
        sample_score = y_score[idx]
        auc = roc_auc_score(sample_y, sample_score)
        boot_aucs.append(auc)

    if len(boot_aucs) == 0:
        return float("nan"), float("nan"), skipped

    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_aucs, lower_pct))
    ci_upper = float(np.percentile(boot_aucs, upper_pct))

    return round(ci_lower, 6), round(ci_upper, 6), skipped


def train_and_eval_model_a(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Execute training, evaluation, artifact saving and sanity guards for Model A."""
    if config is None:
        config = load_config()

    seed = set_seed(config=config)
    bootstrap_n = int(config.get("bootstrap_n", 200))

    # Load model hyperparameters
    params_file = _PROJECT_ROOT / "configs" / "model_params.yaml"
    with open(params_file, "r", encoding="utf-8") as f:
        all_params = yaml.safe_load(f)
    model_params = all_params["model_a"]

    print("[train_model_a] Preparing datasets...")
    (
        X_train, y_train,
        X_valid, y_valid,
        X_test, y_test,
        meta_train, meta_valid, meta_test,
        feature_names,
        audit_info
    ) = prepare_model_a_datasets(config=config)

    # Initialize XGBoost classifier
    xgb_params = dict(model_params)
    xgb_params["random_state"] = seed

    early_stopping_rounds = xgb_params.get("early_stopping_rounds", 50)
    early_stopping_location = "constructor"

    print(f"[train_model_a] Initializing XGBClassifier with params:\n{xgb_params}")
    clf = xgb.XGBClassifier(**xgb_params)

    # Train on TRAIN split, early-stop on VALIDATION split
    print(f"[train_model_a] Fitting model on {len(X_train):,} train samples, evaluating on {len(X_valid):,} valid samples...")
    t0 = time.time()
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100
    )
    training_time_sec = round(time.time() - t0, 2)
    print(f"[train_model_a] Training completed in {training_time_sec} seconds.")

    best_iteration = int(clf.best_iteration)
    print(f"[train_model_a] Best iteration: {best_iteration}")

    # Predictions
    print("[train_model_a] Generating probability predictions...")
    val_preds = clf.predict_proba(X_valid)[:, 1]
    test_preds = clf.predict_proba(X_test)[:, 1]

    # Sanity checks on predictions
    for split_name, preds, meta, expected_len in [
        ("valid", val_preds, meta_valid, len(X_valid)),
        ("test", test_preds, meta_test, len(X_test)),
    ]:
        if np.isnan(preds).any():
            raise ValueError(f"NaN found in {split_name} predictions!")
        if (preds < 0.0).any() or (preds > 1.0).any():
            raise ValueError(f"{split_name} predictions out of [0, 1] range: min={preds.min()}, max={preds.max()}")
        if len(preds) != expected_len:
            raise ValueError(f"Prediction count mismatch for {split_name}: {len(preds)} vs {expected_len}")
        if meta["SK_ID_CURR"].duplicated().any():
            raise ValueError(f"Duplicate IDs in {split_name} predictions!")

    # Check ID coverage
    splits_dir = _PROJECT_ROOT / config["paths"]["splits"]
    orig_valid_ids = pd.read_csv(splits_dir / "valid_ids.csv")["SK_ID_CURR"].tolist()
    orig_test_ids = pd.read_csv(splits_dir / "test_ids.csv")["SK_ID_CURR"].tolist()

    if set(meta_valid["SK_ID_CURR"]) != set(orig_valid_ids):
        raise ValueError("Validation prediction IDs do not exactly match valid_ids.csv!")
    if set(meta_test["SK_ID_CURR"]) != set(orig_test_ids):
        raise ValueError("Test prediction IDs do not exactly match test_ids.csv!")

    # Evaluation
    val_auc = round(float(roc_auc_score(y_valid, val_preds)), 6)
    test_auc = round(float(roc_auc_score(y_test, test_preds)), 6)
    print(f"[train_model_a] Validation overall ROC-AUC: {val_auc}")
    print(f"[train_model_a] Test overall ROC-AUC:       {test_auc}")

    # Guard on test AUC
    if test_auc > 0.85 or test_auc < 0.60:
        raise ValueError(
            f"STOP: Overall test AUC {test_auc} is outside the allowed range [0.60, 0.85]! "
            f"Possible data leakage or severe bug."
        )

    # Overall test bootstrap CI
    test_ci_lower, test_ci_upper, overall_skipped = compute_bootstrap_auc_ci(
        y_test.values, test_preds, n_bootstraps=bootstrap_n, seed=seed
    )

    # Segment evaluations on test split
    meta_test_df = meta_test.copy()
    meta_test_df["score"] = test_preds

    segments = ["full_file", "thin_file", "no_file"]
    segment_results = {}
    total_skipped_resamples = overall_skipped

    for seg in segments:
        seg_mask = meta_test_df["segment"] == seg
        seg_df = meta_test_df[seg_mask]
        n_applicants = len(seg_df)
        n_positives = int(seg_df["TARGET"].sum())
        low_precision = bool(n_positives < 50)

        seg_y = seg_df["TARGET"].values
        seg_score = seg_df["score"].values

        if len(np.unique(seg_y)) >= 2:
            seg_auc = round(float(roc_auc_score(seg_y, seg_score)), 6)
            seg_ci_lower, seg_ci_upper, seg_skipped = compute_bootstrap_auc_ci(
                seg_y, seg_score, n_bootstraps=bootstrap_n, seed=seed
            )
        else:
            seg_auc = float("nan")
            seg_ci_lower, seg_ci_upper, seg_skipped = float("nan"), float("nan"), 0

        total_skipped_resamples += seg_skipped

        segment_results[seg] = {
            "applicants": n_applicants,
            "positives": n_positives,
            "auc": seg_auc,
            "ci_95_lower": seg_ci_lower,
            "ci_95_upper": seg_ci_upper,
            "low_precision": low_precision,
            "skipped_resamples": seg_skipped,
        }
        print(f"  Segment '{seg}': n={n_applicants}, pos={n_positives}, AUC={seg_auc} (95% CI: [{seg_ci_lower}, {seg_ci_upper}]), low_precision={low_precision}")

    # Feature Importance (Gain)
    booster = clf.get_booster()
    score_dict = booster.get_score(importance_type="gain")
    gain_records = []
    for feat in feature_names:
        gain_val = float(score_dict.get(feat, 0.0))
        gain_records.append({"feature": feat, "gain": gain_val})

    gain_df = pd.DataFrame(gain_records).sort_values(by="gain", ascending=False).reset_index(drop=True)
    gain_csv_path = _PROJECT_ROOT / config["paths"]["metrics"] / "model_a_feature_gain.csv"
    gain_df.to_csv(gain_csv_path, index=False)
    print(f"[train_model_a] Saved feature gain rankings to {gain_csv_path}")

    # Save Native XGBoost Model
    models_dir = _PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    native_model_path = models_dir / "model_a.json"
    clf.save_model(str(native_model_path))
    print(f"[train_model_a] Saved native model to {native_model_path}")

    # Save Predictions Parquet files
    pred_dir = _PROJECT_ROOT / config["paths"]["predictions"]
    pred_dir.mkdir(parents=True, exist_ok=True)

    meta_valid_df = meta_valid.copy()
    meta_valid_df["score"] = val_preds
    val_pred_path = pred_dir / "model_a_val.parquet"
    meta_valid_df.to_parquet(val_pred_path, index=False, engine="pyarrow")

    test_pred_path = pred_dir / "model_a_test.parquet"
    meta_test_df.to_parquet(test_pred_path, index=False, engine="pyarrow")
    print(f"[train_model_a] Saved predictions to {val_pred_path} and {test_pred_path}")

    # Assemble and write reports/metrics/model_a.json
    results_summary = {
        "model_name": "Model A (Bureau Baseline)",
        "versions": {
            "xgboost": xgb.__version__,
            "pandas": pd.__version__,
        },
        "parameters": model_params,
        "early_stopping": {
            "rounds": early_stopping_rounds,
            "set_in": early_stopping_location,
            "best_iteration": best_iteration,
        },
        "training_time_seconds": training_time_sec,
        "feature_count": len(feature_names),
        "categoricals_handling": "pandas category dtype with categories learned on train split; XGBoost enable_categorical=True",
        "validation_auc_overall": val_auc,
        "test_results": {
            "overall": {
                "applicants": len(meta_test_df),
                "positives": int(meta_test_df["TARGET"].sum()),
                "auc": test_auc,
                "ci_95_lower": test_ci_lower,
                "ci_95_upper": test_ci_upper,
                "skipped_resamples": overall_skipped,
            },
            "by_segment": segment_results,
        },
        "bootstrap_settings": {
            "n_bootstraps": bootstrap_n,
            "seed": seed,
            "confidence_level": 0.95,
            "total_skipped_resamples_all": total_skipped_resamples,
        },
        "sanity_guards": {
            "score_range": [float(min(val_preds.min(), test_preds.min())), float(max(val_preds.max(), test_preds.max()))],
            "nan_count_val": int(np.isnan(val_preds).sum()),
            "nan_count_test": int(np.isnan(test_preds).sum()),
            "id_coverage_valid_exact": True,
            "id_coverage_test_exact": True,
            "auc_bug_guard_passed": True,
        },
        "top_15_features_by_gain": gain_df.head(15).to_dict(orient="records"),
    }

    metrics_json_path = _PROJECT_ROOT / config["paths"]["metrics"] / "model_a.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"[train_model_a] Saved metrics to {metrics_json_path}")

    return results_summary


if __name__ == "__main__":
    train_and_eval_model_a()
