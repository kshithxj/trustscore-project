"""Common helper functions for Phase 7 extra analyses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
import xgboost as xgb
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.utils import load_config, set_seed
from src.features.model_a_features import (
    FORBIDDEN_COLS,
    validate_feature_list,
)
from src.features.bureau import build_bureau_features, BUREAU_FEATURE_COLS
from src.features.behavioral import ALL_BEHAVIORAL_COLS, BEHAVIORAL_COLS_BY_TABLE


def load_model_params() -> Dict[str, Any]:
    params_file = _PROJECT_ROOT / "configs" / "model_params.yaml"
    with open(params_file, "r", encoding="utf-8") as f:
        all_params = yaml.safe_load(f)
    return all_params["model_a"]


def load_prepared_data(
    config: Optional[Dict[str, Any]] = None,
    train_ids_df: Optional[pd.DataFrame] = None,
    valid_ids_df: Optional[pd.DataFrame] = None,
    test_ids_df: Optional[pd.DataFrame] = None,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,  # X_train_a, X_valid_a, X_test_a
    pd.DataFrame, pd.DataFrame, pd.DataFrame,  # X_train_b, X_valid_b, X_test_b
    pd.Series, pd.Series, pd.Series,            # y_train, y_valid, y_test
    pd.DataFrame, pd.DataFrame, pd.DataFrame,  # meta_train, meta_valid, meta_test
    List[str], List[str]                       # feat_a, feat_b
]:
    """Prepare Model A and Model B feature matrices without modifying reports/metrics."""
    if config is None:
        config = load_config()

    splits_dir = _PROJECT_ROOT / config["paths"]["splits"]
    processed_dir = _PROJECT_ROOT / config["paths"]["processed"]

    # 1. Base + bureau features
    base_df = pd.read_parquet(processed_dir / "applicant_base.parquet")
    joined_df, _ = build_bureau_features(base_df=base_df, config=config)

    # 2. Split IDs
    if train_ids_df is None:
        train_ids_df = pd.read_csv(splits_dir / "train_ids.csv")
    if valid_ids_df is None:
        valid_ids_df = pd.read_csv(splits_dir / "valid_ids.csv")
    if test_ids_df is None:
        test_ids_df = pd.read_csv(splits_dir / "test_ids.csv")

    train_mask = joined_df["SK_ID_CURR"].isin(set(train_ids_df["SK_ID_CURR"]))
    valid_mask = joined_df["SK_ID_CURR"].isin(set(valid_ids_df["SK_ID_CURR"]))
    test_mask = joined_df["SK_ID_CURR"].isin(set(test_ids_df["SK_ID_CURR"]))

    df_train = joined_df[train_mask].set_index("SK_ID_CURR").loc[train_ids_df["SK_ID_CURR"]].reset_index()
    df_valid = joined_df[valid_mask].set_index("SK_ID_CURR").loc[valid_ids_df["SK_ID_CURR"]].reset_index()
    df_test = joined_df[test_mask].set_index("SK_ID_CURR").loc[test_ids_df["SK_ID_CURR"]].reset_index()

    meta_cols = ["SK_ID_CURR", "segment", "TARGET"]
    meta_train = df_train[meta_cols].copy()
    meta_valid = df_valid[meta_cols].copy()
    meta_test = df_test[meta_cols].copy()

    y_train = df_train["TARGET"].copy()
    y_valid = df_valid["TARGET"].copy()
    y_test = df_test["TARGET"].copy()

    # Features for Model A
    candidate_cols = [c for c in joined_df.columns if c not in FORBIDDEN_COLS]
    validate_feature_list(candidate_cols)

    X_train_a = df_train[candidate_cols].copy()
    X_valid_a = df_valid[candidate_cols].copy()
    X_test_a = df_test[candidate_cols].copy()

    cat_cols = X_train_a.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cat_cols:
        cats = sorted([str(x) for x in X_train_a[col].dropna().unique().tolist()])
        X_train_a[col] = pd.Categorical(
            X_train_a[col].where(X_train_a[col].isin(cats), None),
            categories=cats
        )
        X_valid_a[col] = pd.Categorical(
            X_valid_a[col].where(X_valid_a[col].isin(cats), None),
            categories=cats
        )
        X_test_a[col] = pd.Categorical(
            X_test_a[col].where(X_test_a[col].isin(cats), None),
            categories=cats
        )

    # Features for Model B (Model A + Behavioral)
    beh_df = pd.read_parquet(processed_dir / "behavioral_features.parquet")
    beh_indexed = beh_df.set_index("SK_ID_CURR")

    train_beh = beh_indexed.loc[meta_train["SK_ID_CURR"], ALL_BEHAVIORAL_COLS].reset_index(drop=True)
    valid_beh = beh_indexed.loc[meta_valid["SK_ID_CURR"], ALL_BEHAVIORAL_COLS].reset_index(drop=True)
    test_beh = beh_indexed.loc[meta_test["SK_ID_CURR"], ALL_BEHAVIORAL_COLS].reset_index(drop=True)

    X_train_b = pd.concat([X_train_a.reset_index(drop=True), train_beh], axis=1)
    X_valid_b = pd.concat([X_valid_a.reset_index(drop=True), valid_beh], axis=1)
    X_test_b = pd.concat([X_test_a.reset_index(drop=True), test_beh], axis=1)

    feat_a = list(candidate_cols)
    feat_b = list(candidate_cols) + list(ALL_BEHAVIORAL_COLS)

    X_train_b = X_train_b[feat_b]
    X_valid_b = X_valid_b[feat_b]
    X_test_b = X_test_b[feat_b]

    return (
        X_train_a, X_valid_a, X_test_a,
        X_train_b, X_valid_b, X_test_b,
        y_train, y_valid, y_test,
        meta_train, meta_valid, meta_test,
        feat_a, feat_b
    )


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    seed: int = 42,
    verbose: bool = False
) -> xgb.XGBClassifier:
    """Train XGBClassifier with fixed model_params and early stopping on validation."""
    params = load_model_params()
    xgb_params = dict(params)
    xgb_params["random_state"] = seed

    clf = xgb.XGBClassifier(**xgb_params)
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100 if verbose else False
    )
    return clf


def compute_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstraps: int = 200,
    seed: int = 42,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Compute AUC and 95% bootstrap CI."""
    auc_val = float(roc_auc_score(y_true, y_score))
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_aucs = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        sample_y = y_true[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        boot_aucs.append(roc_auc_score(sample_y, y_score[idx]))

    if len(boot_aucs) == 0:
        return auc_val, float("nan"), float("nan")

    lower = float(np.percentile(boot_aucs, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_aucs, 100 * (1.0 - alpha / 2.0)))
    return round(auc_val, 6), round(lower, 6), round(upper, 6)


def compute_paired_auc_diff_ci(
    y_true: np.ndarray,
    score_model: np.ndarray,
    score_ref: np.ndarray,
    n_bootstraps: int = 200,
    seed: int = 42,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Compute paired AUC difference (score_model - score_ref) with 95% CI."""
    diff_val = float(roc_auc_score(y_true, score_model) - roc_auc_score(y_true, score_ref))
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_diffs = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        sample_y = y_true[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        auc_m = roc_auc_score(sample_y, score_model[idx])
        auc_r = roc_auc_score(sample_y, score_ref[idx])
        boot_diffs.append(auc_m - auc_r)

    if len(boot_diffs) == 0:
        return diff_val, float("nan"), float("nan")

    lower = float(np.percentile(boot_diffs, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_diffs, 100 * (1.0 - alpha / 2.0)))
    return round(diff_val, 6), round(lower, 6), round(upper, 6)
