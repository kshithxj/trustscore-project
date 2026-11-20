"""Proxy-leakage test for TrustScore.

Evaluates whether credit-model inputs (application features S1, behavioral features S2,
or combined S3) can predict protected attributes (gender and binary age) via 5-fold CV.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config, set_seed
from src.features.model_a_features import validate_feature_list, FORBIDDEN_COLS
from src.features.behavioral import ALL_BEHAVIORAL_COLS

PROXY_MODEL_PARAMS = {
    "tree_method": "hist",
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "enable_categorical": True,
}


def prepare_proxy_feature_sets(
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    """Load and prepare feature matrices S1, S2, S3 alongside target attributes."""
    processed_dir = _PROJECT_ROOT / config["paths"]["processed"]
    metrics_dir = _PROJECT_ROOT / config["paths"]["metrics"]

    base_df = pd.read_parquet(processed_dir / "applicant_base.parquet")
    beh_df = pd.read_parquet(processed_dir / "behavioral_features.parquet")
    eval_attrs = pd.read_parquet(processed_dir / "eval_attrs.parquet")

    # Read Model A features and remove bureau_* to get S1
    features_a_path = metrics_dir / "model_a_features.txt"
    with open(features_a_path, "r", encoding="utf-8") as f:
        model_a_features = [line.strip() for line in f if line.strip()]

    s1_features = [f for f in model_a_features if not f.startswith("bureau_")]
    s2_features = list(ALL_BEHAVIORAL_COLS)
    s3_features = s1_features + s2_features

    validate_feature_list(s1_features)
    validate_feature_list(s2_features)
    validate_feature_list(s3_features)

    # Merge base and behavioral features on SK_ID_CURR
    full_df = base_df[["SK_ID_CURR"] + s1_features].merge(
        beh_df[["SK_ID_CURR"] + s2_features],
        on="SK_ID_CURR",
        how="left"
    )
    full_df = full_df.merge(
        eval_attrs[["SK_ID_CURR", "gender", "age_years"]],
        on="SK_ID_CURR",
        how="left"
    )

    return full_df, eval_attrs, s1_features, s2_features, s3_features


def encode_categoricals_fold(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert object columns to pandas category dtype using train categories only."""
    X_tr = X_train.copy()
    X_va = X_val.copy()

    cat_cols = X_tr.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cat_cols:
        cats = sorted([str(x) for x in X_tr[col].dropna().unique().tolist()])
        X_tr[col] = pd.Categorical(
            X_tr[col].where(X_tr[col].isin(cats), None),
            categories=cats
        )
        X_va[col] = pd.Categorical(
            X_va[col].where(X_va[col].isin(cats), None),
            categories=cats
        )
    return X_tr, X_va


def run_proxy_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int = 42
) -> Tuple[float, float, np.ndarray]:
    """Run 5-fold stratified CV and return mean AUC, std AUC, and OOF probabilities."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y), dtype=np.float32)
    fold_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr = X.iloc[train_idx]
        y_tr = y[train_idx]
        X_va = X.iloc[val_idx]
        y_va = y[val_idx]

        X_tr_enc, X_va_enc = encode_categoricals_fold(X_tr, X_va)

        params = dict(PROXY_MODEL_PARAMS)
        params["random_state"] = seed + fold_idx
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_tr_enc, y_tr, verbose=False)

        val_probs = clf.predict_proba(X_va_enc)[:, 1]
        oof_preds[val_idx] = val_probs
        auc = roc_auc_score(y_va, val_probs)
        fold_aucs.append(auc)

    mean_auc = round(float(np.mean(fold_aucs)), 6)
    std_auc = round(float(np.std(fold_aucs)), 6)
    return mean_auc, std_auc, oof_preds


def fit_full_and_get_top5_gain(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Fit once on full data for the best feature set to extract top 5 features by gain."""
    X_full = X.copy()
    cat_cols = X_full.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cat_cols:
        cats = sorted([str(x) for x in X_full[col].dropna().unique().tolist()])
        X_full[col] = pd.Categorical(
            X_full[col].where(X_full[col].isin(cats), None),
            categories=cats
        )

    params = dict(PROXY_MODEL_PARAMS)
    params["random_state"] = seed
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_full, y, verbose=False)

    booster = clf.get_booster()
    score_dict = booster.get_score(importance_type="gain")

    gain_records = []
    for feat in X.columns:
        gain_records.append({
            "feature": feat,
            "gain": round(float(score_dict.get(feat, 0.0)), 4)
        })

    gain_records.sort(key=lambda x: x["gain"], reverse=True)
    return gain_records[:5]


def run_proxy_leakage_tests(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Execute complete proxy-leakage evaluation."""
    if config is None:
        config = load_config()

    seed = set_seed(config=config)
    metrics_dir = _PROJECT_ROOT / config["paths"]["metrics"]
    splits_dir = _PROJECT_ROOT / config["paths"]["splits"]

    full_df, eval_attrs, s1_cols, s2_cols, s3_cols = prepare_proxy_feature_sets(config)

    # Compute train-only median age
    train_ids = pd.read_csv(splits_dir / "train_ids.csv", usecols=["SK_ID_CURR"])
    train_eval = eval_attrs[eval_attrs["SK_ID_CURR"].isin(set(train_ids["SK_ID_CURR"]))]
    train_median_age = float(train_eval["age_years"].median())

    # Targets:
    # 1. Gender: Female (1) vs Male (0), excluding XNA
    gender_mask = full_df["gender"].isin(["F", "M"])
    gender_df = full_df[gender_mask].copy()
    y_gender = (gender_df["gender"] == "F").astype(int).values

    # 2. Binary age: >= median (1) vs < median (0)
    y_age = (full_df["age_years"] >= train_median_age).astype(int).values

    targets = [
        ("gender", y_gender, gender_df, "Female (F)"),
        ("binary_age", y_age, full_df, f"Age >= {train_median_age:.2f}"),
    ]

    feature_sets = [
        ("S1", s1_cols, "Application features only (118 features)"),
        ("S2", s2_cols, "Behavioral features only (20 features)"),
        ("S3", s3_cols, "Combined application + behavioral (138 features)"),
    ]

    proxy_results = {
        "fixed_model_parameters": PROXY_MODEL_PARAMS,
        "train_median_age": train_median_age,
        "interpretation_notes": (
            "An AUC near 0.5 means little leakage; a clearly higher AUC for S3 than S1 means "
            "the behavioral features add proxy information; S1 may already predict gender or "
            "age through legitimate fields such as occupation and employment."
        ),
        "targets": {},
    }

    print("\n[proxy] Starting 5-fold CV proxy-leakage experiments...")

    for target_name, y_target, df_target, pos_label in targets:
        print(f"\n--- Target: {target_name} ({pos_label}, n={len(y_target):,}) ---")
        prev = round(float(np.mean(y_target)), 6)
        majority_acc = round(float(max(prev, 1.0 - prev)), 6)
        print(f"Prevalence: {prev:.4f}, Majority-class baseline accuracy: {majority_acc:.4f}")

        target_summary: Dict[str, Any] = {
            "positive_label": pos_label,
            "n_samples": len(y_target),
            "prevalence": prev,
            "majority_class_accuracy": majority_acc,
            "auc_baseline": 0.5,
            "feature_sets": {},
        }

        best_set = None
        best_auc = -1.0

        for fset_name, fset_cols, fset_desc in feature_sets:
            t0 = time.time()
            X_sub = df_target[fset_cols]
            mean_auc, std_auc, _ = run_proxy_cv(X_sub, y_target, seed=seed)
            elapsed = round(time.time() - t0, 1)

            print(f"  [{fset_name}] {fset_desc}: Mean AUC = {mean_auc:.4f} +/- {std_auc:.4f} ({elapsed}s)")

            target_summary["feature_sets"][fset_name] = {
                "description": fset_desc,
                "n_features": len(fset_cols),
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "runtime_seconds": elapsed,
            }

            if mean_auc > best_auc:
                best_auc = mean_auc
                best_set = fset_name

        # Top 5 features by gain for the highest-AUC feature set
        print(f"  Highest AUC set for {target_name} is {best_set} (AUC={best_auc:.4f}). Fitting once on full data for gain ranking...")
        best_cols = s1_cols if best_set == "S1" else (s2_cols if best_set == "S2" else s3_cols)
        top5_gain = fit_full_and_get_top5_gain(df_target[best_cols], y_target, seed=seed)

        target_summary["best_feature_set"] = best_set
        target_summary["top_5_features_by_gain_for_best_set"] = {
            "feature_set": best_set,
            "fit_method": "fit once on full data for ranking only",
            "top_5": top5_gain,
        }

        print("  Top 5 features by gain:")
        for r in top5_gain:
            print(f"    - {r['feature']}: {r['gain']:.2f}")

        proxy_results["targets"][target_name] = target_summary

    # Save reports/metrics/proxy.json
    proxy_out_path = metrics_dir / "proxy.json"
    with open(proxy_out_path, "w", encoding="utf-8") as f:
        json.dump(proxy_results, f, indent=2)
    print(f"\n[proxy] Saved {proxy_out_path}")

    return proxy_results


if __name__ == "__main__":
    run_proxy_leakage_tests()
