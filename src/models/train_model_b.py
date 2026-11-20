"""Train and evaluate Model B (Model A + Behavioral features) and paired comparison with Model A.

Usage:
    python -m src.models.train_model_b
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config, set_seed
from src.features.model_b_features import prepare_model_b_datasets
from src.features.behavioral import ALL_BEHAVIORAL_COLS
from src.models.train_model_a import compute_bootstrap_auc_ci


def compute_paired_bootstrap_comparison(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_bootstraps: int = 200,
    seed: int = 42,
    alpha: float = 0.05
) -> Tuple[float, float, float, float, int]:
    """Compute paired bootstrap difference (AUC_B - AUC_A) across identical resamples.

    Returns:
        mean_diff, ci_lower, ci_upper, share_b_gt_a, skipped_count
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []
    skipped = 0

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        sample_y = y_true[idx]
        if len(np.unique(sample_y)) < 2:
            skipped += 1
            continue

        auc_a = roc_auc_score(sample_y, score_a[idx])
        auc_b = roc_auc_score(sample_y, score_b[idx])
        diffs.append(auc_b - auc_a)

    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), skipped

    diffs_arr = np.array(diffs)
    mean_diff = float(np.mean(diffs_arr))
    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(diffs_arr, lower_pct))
    ci_upper = float(np.percentile(diffs_arr, upper_pct))
    share_b_gt_a = float(np.mean(diffs_arr > 0))

    return round(mean_diff, 6), round(ci_lower, 6), round(ci_upper, 6), round(share_b_gt_a, 4), skipped


def train_and_eval_model_b(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Execute training, evaluation, paired comparison, and artifact saving for Model B."""
    if config is None:
        config = load_config()

    seed = set_seed(config=config)
    bootstrap_n = int(config.get("bootstrap_n", 200))

    # Load fixed hyperparameters
    params_file = _PROJECT_ROOT / "configs" / "model_params.yaml"
    with open(params_file, "r", encoding="utf-8") as f:
        all_params = yaml.safe_load(f)
    model_params = all_params["model_a"]  # reuses model_a params unchanged

    print("[train_model_b] Preparing Model B datasets...")
    (
        X_train, y_train,
        X_valid, y_valid,
        X_test, y_test,
        meta_train, meta_valid, meta_test,
        feature_names,
        audit_info
    ) = prepare_model_b_datasets(config=config)

    xgb_params = dict(model_params)
    xgb_params["random_state"] = seed

    print(f"[train_model_b] Initializing XGBClassifier with {len(feature_names)} features and params:\n{xgb_params}")
    clf = xgb.XGBClassifier(**xgb_params)

    # Train on TRAIN split, early-stop on VALIDATION split
    print(f"[train_model_b] Fitting model on {len(X_train):,} train samples, evaluating on {len(X_valid):,} valid samples...")
    t0 = time.time()
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100
    )
    training_time_sec = round(time.time() - t0, 2)
    print(f"[train_model_b] Training completed in {training_time_sec} seconds.")

    best_iteration = int(clf.best_iteration)
    print(f"[train_model_b] Best iteration: {best_iteration}")

    # Generate probability predictions
    print("[train_model_b] Generating probability predictions...")
    val_preds = clf.predict_proba(X_valid)[:, 1]
    test_preds = clf.predict_proba(X_test)[:, 1]

    # Sanity checks on predictions
    for split_name, preds, meta, expected_len in [
        ("valid", val_preds, meta_valid, len(X_valid)),
        ("test", test_preds, meta_test, len(X_test)),
    ]:
        if np.isnan(preds).any():
            raise ValueError(f"NaN found in Model B {split_name} predictions!")
        if (preds < 0.0).any() or (preds > 1.0).any():
            raise ValueError(f"Model B {split_name} predictions out of [0, 1] range: min={preds.min()}, max={preds.max()}")
        if len(preds) != expected_len:
            raise ValueError(f"Prediction count mismatch for {split_name}: {len(preds)} vs {expected_len}")
        if meta["SK_ID_CURR"].duplicated().any():
            raise ValueError(f"Duplicate IDs in Model B {split_name} predictions!")

    # Check ID coverage against original split files
    splits_dir = _PROJECT_ROOT / config["paths"]["splits"]
    orig_valid_ids = pd.read_csv(splits_dir / "valid_ids.csv")["SK_ID_CURR"].tolist()
    orig_test_ids = pd.read_csv(splits_dir / "test_ids.csv")["SK_ID_CURR"].tolist()

    if set(meta_valid["SK_ID_CURR"]) != set(orig_valid_ids):
        raise ValueError("Model B validation prediction IDs do not exactly match valid_ids.csv!")
    if set(meta_test["SK_ID_CURR"]) != set(orig_test_ids):
        raise ValueError("Model B test prediction IDs do not exactly match test_ids.csv!")

    # Evaluation
    val_auc = round(float(roc_auc_score(y_valid, val_preds)), 6)
    test_auc = round(float(roc_auc_score(y_test, test_preds)), 6)
    print(f"[train_model_b] Validation overall ROC-AUC: {val_auc}")
    print(f"[train_model_b] Test overall ROC-AUC:       {test_auc}")

    # Bug / Leakage guard on test AUC
    if test_auc > 0.85:
        raise ValueError(
            f"STOP: Model B overall test AUC {test_auc} is above 0.85! "
            f"Possible data leakage or severe bug."
        )

    # Load Model A predictions for paired comparison
    pred_dir = _PROJECT_ROOT / config["paths"]["predictions"]
    model_a_test_path = pred_dir / "model_a_test.parquet"
    if not model_a_test_path.exists():
        raise FileNotFoundError(f"Missing Model A test predictions: {model_a_test_path}")

    model_a_test_df = pd.read_parquet(model_a_test_path)
    # Check that SK_ID_CURR and TARGET match Model B's test rows exactly
    if not (model_a_test_df["SK_ID_CURR"].values == meta_test["SK_ID_CURR"].values).all():
        raise ValueError("Model A and Model B test SK_ID_CURR do not match in exact order!")
    if not (model_a_test_df["TARGET"].values == y_test.values).all():
        raise ValueError("Model A and Model B test TARGET values do not match!")

    score_a_test = model_a_test_df["score"].values
    score_b_test = test_preds

    # Overall test bootstrap CI for Model B
    test_ci_lower, test_ci_upper, overall_skipped = compute_bootstrap_auc_ci(
        y_test.values, score_b_test, n_bootstraps=bootstrap_n, seed=seed
    )

    # Segment evaluations on test split for Model B
    meta_test_df = meta_test.copy()
    meta_test_df["score"] = score_b_test

    segments = ["full_file", "thin_file", "no_file"]
    segment_results_b = {}
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

        segment_results_b[seg] = {
            "applicants": n_applicants,
            "positives": n_positives,
            "auc": seg_auc,
            "ci_95_lower": seg_ci_lower,
            "ci_95_upper": seg_ci_upper,
            "low_precision": low_precision,
            "skipped_resamples": seg_skipped,
        }
        print(f"  Model B '{seg}': n={n_applicants}, pos={n_positives}, AUC={seg_auc} (95% CI: [{seg_ci_lower}, {seg_ci_upper}]), low_precision={low_precision}")

    # Paired Bootstrap Comparison (AUC_B - AUC_A)
    print("[train_model_b] Computing paired bootstrap comparison (AUC_B - AUC_A)...")
    paired_overall_mean, paired_overall_lower, paired_overall_upper, paired_overall_share_b_gt_a, paired_overall_skipped = compute_paired_bootstrap_comparison(
        y_test.values, score_a_test, score_b_test, n_bootstraps=bootstrap_n, seed=seed
    )

    paired_segment_results = {}
    total_paired_skipped = paired_overall_skipped

    for seg in segments:
        seg_mask = meta_test_df["segment"] == seg
        seg_y = y_test.values[seg_mask]
        seg_score_a = score_a_test[seg_mask]
        seg_score_b = score_b_test[seg_mask]

        if len(np.unique(seg_y)) >= 2:
            seg_mean, seg_lower, seg_upper, seg_share_gt, seg_skip = compute_paired_bootstrap_comparison(
                seg_y, seg_score_a, seg_score_b, n_bootstraps=bootstrap_n, seed=seed
            )
        else:
            seg_mean, seg_lower, seg_upper, seg_share_gt, seg_skip = float("nan"), float("nan"), float("nan"), float("nan"), 0

        total_paired_skipped += seg_skip
        paired_segment_results[seg] = {
            "mean_difference": seg_mean,
            "ci_95_lower": seg_lower,
            "ci_95_upper": seg_upper,
            "share_b_gt_a": seg_share_gt,
            "skipped_resamples": seg_skip,
        }
        print(f"  Paired B - A '{seg}': mean_diff={seg_mean}, 95% CI: [{seg_lower}, {seg_upper}], P(B>A)={seg_share_gt}")

    # Feature Importance (Gain)
    booster = clf.get_booster()
    score_dict = booster.get_score(importance_type="gain")
    gain_records = []
    for feat in feature_names:
        gain_val = float(score_dict.get(feat, 0.0))
        is_beh = feat in ALL_BEHAVIORAL_COLS
        gain_records.append({"feature": feat, "gain": gain_val, "is_behavioral": is_beh})

    gain_df = pd.DataFrame(gain_records).sort_values(by="gain", ascending=False).reset_index(drop=True)
    gain_csv_path = _PROJECT_ROOT / config["paths"]["metrics"] / "model_b_feature_gain.csv"
    gain_df[["feature", "gain"]].to_csv(gain_csv_path, index=False)
    print(f"[train_model_b] Saved feature gain rankings to {gain_csv_path}")

    total_gain_all = float(gain_df["gain"].sum())
    total_gain_behavioral = float(gain_df[gain_df["is_behavioral"]]["gain"].sum())
    behavioral_gain_share = round(total_gain_behavioral / total_gain_all, 6) if total_gain_all > 0 else 0.0
    print(f"[train_model_b] Total gain: {total_gain_all:.2f}, Behavioral gain: {total_gain_behavioral:.2f} ({behavioral_gain_share*100:.2f}%)")

    # Native Model Save
    models_dir = _PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    native_model_path = models_dir / "model_b.json"
    clf.save_model(str(native_model_path))
    print(f"[train_model_b] Saved native model to {native_model_path}")

    # Save Prediction Parquet files
    meta_valid_df = meta_valid.copy()
    meta_valid_df["score"] = val_preds
    val_pred_path = pred_dir / "model_b_val.parquet"
    meta_valid_df.to_parquet(val_pred_path, index=False, engine="pyarrow")

    test_pred_path = pred_dir / "model_b_test.parquet"
    meta_test_df.to_parquet(test_pred_path, index=False, engine="pyarrow")
    print(f"[train_model_b] Saved predictions to {val_pred_path} and {test_pred_path}")

    # Load Model A json to populate comparative metrics
    model_a_json_path = _PROJECT_ROOT / config["paths"]["metrics"] / "model_a.json"
    with open(model_a_json_path, "r", encoding="utf-8") as f:
        model_a_json = json.load(f)

    # 1. reports/metrics/model_b.json
    model_b_summary = {
        "model_name": "Model B (Lender-History Behavioral)",
        "versions": {
            "xgboost": xgb.__version__,
            "pandas": pd.__version__,
        },
        "parameters": model_params,
        "parameters_identical_to_model_a": True,
        "early_stopping": {
            "rounds": model_params.get("early_stopping_rounds", 50),
            "set_in": "constructor",
            "best_iteration": best_iteration,
        },
        "training_time_seconds": training_time_sec,
        "feature_count": len(feature_names),
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
            "by_segment": segment_results_b,
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
        "importance": {
            "top_20_features_by_gain": gain_df.head(20)[["feature", "gain"]].to_dict(orient="records"),
            "total_gain_all": total_gain_all,
            "total_gain_behavioral": total_gain_behavioral,
            "behavioral_gain_share": behavioral_gain_share,
        },
    }

    metrics_b_path = _PROJECT_ROOT / config["paths"]["metrics"] / "model_b.json"
    with open(metrics_b_path, "w", encoding="utf-8") as f:
        json.dump(model_b_summary, f, indent=2)
    print(f"[train_model_b] Saved {metrics_b_path}")

    # 2. reports/metrics/a_vs_b.json
    a_vs_b_summary = {
        "comparison": "Model B vs Model A",
        "validation_auc": {
            "model_a": model_a_json["validation_auc_overall"],
            "model_b": val_auc,
            "diff_b_minus_a": round(val_auc - model_a_json["validation_auc_overall"], 6),
        },
        "test_auc": {
            "overall": {
                "model_a": model_a_json["test_results"]["overall"]["auc"],
                "model_b": test_auc,
                "diff_b_minus_a": round(test_auc - model_a_json["test_results"]["overall"]["auc"], 6),
                "paired_bootstrap": {
                    "mean_difference": paired_overall_mean,
                    "ci_95_lower": paired_overall_lower,
                    "ci_95_upper": paired_overall_upper,
                    "share_b_gt_a": paired_overall_share_b_gt_a,
                    "skipped_resamples": paired_overall_skipped,
                },
            },
            "by_segment": {
                seg: {
                    "model_a_auc": model_a_json["test_results"]["by_segment"][seg]["auc"],
                    "model_b_auc": segment_results_b[seg]["auc"],
                    "paired_bootstrap": paired_segment_results[seg],
                }
                for seg in segments
            },
        },
    }

    a_vs_b_path = _PROJECT_ROOT / config["paths"]["metrics"] / "a_vs_b.json"
    with open(a_vs_b_path, "w", encoding="utf-8") as f:
        json.dump(a_vs_b_summary, f, indent=2)
    print(f"[train_model_b] Saved {a_vs_b_path}")

    return model_b_summary


if __name__ == "__main__":
    train_and_eval_model_b()
