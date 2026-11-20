"""Feature matrix preparation for Model A.

Usage:
    python -m src.features.model_a_features
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config
from src.features.bureau import build_bureau_features, BUREAU_FEATURE_COLS

FORBIDDEN_COLS = {
    "TARGET",
    "SK_ID_CURR",
    "CODE_GENDER",
    "DAYS_BIRTH",
    "segment",
    "has_bureau",
    "has_lender_history",
}

# Allowed columns that might contain substring 'age' or 'gender'
ALLOWED_SUBSTRING_EXCEPTIONS = {"OWN_CAR_AGE"}


def validate_feature_list(features: List[str]) -> None:
    """Enforce strict exclusion of protected and non-feature columns."""
    for col in features:
        if col in FORBIDDEN_COLS:
            raise ValueError(f"Forbidden column found in feature list: '{col}'")

        col_upper = col.upper()
        if "GENDER" in col_upper or "AGE" in col_upper:
            if col not in ALLOWED_SUBSTRING_EXCEPTIONS:
                raise ValueError(
                    f"Column '{col}' matched GENDER/AGE check and is not in allowed exceptions."
                )


def compute_bureau_missingness(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute share of missing values for each bureau column per segment."""
    segments = ["full_file", "thin_file", "no_file"]
    result: Dict[str, Dict[str, float]] = {}

    for seg in segments:
        seg_mask = df["segment"] == seg
        seg_len = int(seg_mask.sum())
        result[seg] = {}
        for col in BUREAU_FEATURE_COLS:
            if seg_len > 0:
                missing_count = int(df.loc[seg_mask, col].isna().sum())
                share_missing = round(missing_count / seg_len, 6)
            else:
                share_missing = 0.0
            result[seg][col] = share_missing

    return result


def prepare_model_a_datasets(
    config: Dict[str, Any] | None = None
) -> Tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    List[str],
    Dict[str, Any]
]:
    """Prepare train, valid, and test feature matrices and targets for Model A.

    Returns:
        X_train, y_train,
        X_valid, y_valid,
        X_test, y_test,
        meta_train, meta_valid, meta_test,
        feature_names,
        audit_info
    """
    if config is None:
        config = load_config()

    metrics_dir = _PROJECT_ROOT / config["paths"]["metrics"]
    splits_dir = _PROJECT_ROOT / config["paths"]["splits"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build base + bureau joined table
    joined_df, join_stats = build_bureau_features(config=config)

    # 2. Compute and persist bureau missingness by segment
    missingness = compute_bureau_missingness(joined_df)
    missingness_path = metrics_dir / "model_a_bureau_missingness.json"
    with open(missingness_path, "w", encoding="utf-8") as f:
        json.dump(missingness, f, indent=2)
    print(f"[model_a_features] Saved {missingness_path}")

    # 3. Load splits
    train_ids = pd.read_csv(splits_dir / "train_ids.csv")
    valid_ids = pd.read_csv(splits_dir / "valid_ids.csv")
    test_ids = pd.read_csv(splits_dir / "test_ids.csv")

    train_mask = joined_df["SK_ID_CURR"].isin(set(train_ids["SK_ID_CURR"]))
    valid_mask = joined_df["SK_ID_CURR"].isin(set(valid_ids["SK_ID_CURR"]))
    test_mask = joined_df["SK_ID_CURR"].isin(set(test_ids["SK_ID_CURR"]))

    # Sort each split to match split file ordering exactly
    df_train = joined_df[train_mask].set_index("SK_ID_CURR").loc[train_ids["SK_ID_CURR"]].reset_index()
    df_valid = joined_df[valid_mask].set_index("SK_ID_CURR").loc[valid_ids["SK_ID_CURR"]].reset_index()
    df_test = joined_df[test_mask].set_index("SK_ID_CURR").loc[test_ids["SK_ID_CURR"]].reset_index()

    # Meta columns for predictions and evaluation
    meta_cols = ["SK_ID_CURR", "segment", "TARGET"]
    meta_train = df_train[meta_cols].copy()
    meta_valid = df_valid[meta_cols].copy()
    meta_test = df_test[meta_cols].copy()

    y_train = df_train["TARGET"].copy()
    y_valid = df_valid["TARGET"].copy()
    y_test = df_test["TARGET"].copy()

    # 4. Determine feature columns
    candidate_cols = [c for c in joined_df.columns if c not in FORBIDDEN_COLS]
    validate_feature_list(candidate_cols)

    # Save feature names list to report file
    features_txt_path = metrics_dir / "model_a_features.txt"
    with open(features_txt_path, "w", encoding="utf-8") as f:
        for feat in candidate_cols:
            f.write(f"{feat}\n")
    print(f"[model_a_features] Saved {features_txt_path} ({len(candidate_cols)} features)")

    # 5. Categorical encodings (learned on TRAIN split only)
    X_train = df_train[candidate_cols].copy()
    X_valid = df_valid[candidate_cols].copy()
    X_test = df_test[candidate_cols].copy()

    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"[model_a_features] Converting {len(cat_cols)} object columns to pandas category dtype...")

    for col in cat_cols:
        cats = sorted([str(x) for x in X_train[col].dropna().unique().tolist()])
        # Values unseen in train become NaN
        X_train[col] = pd.Categorical(
            X_train[col].where(X_train[col].isin(cats), None),
            categories=cats
        )
        X_valid[col] = pd.Categorical(
            X_valid[col].where(X_valid[col].isin(cats), None),
            categories=cats
        )
        X_test[col] = pd.Categorical(
            X_test[col].where(X_test[col].isin(cats), None),
            categories=cats
        )

    matched_age_gender = [
        c for c in candidate_cols
        if ("GENDER" in c.upper() or "AGE" in c.upper())
    ]

    audit_info = {
        "n_features": len(candidate_cols),
        "feature_list": candidate_cols,
        "n_bureau_features": len(BUREAU_FEATURE_COLS),
        "matched_age_gender_cols": matched_age_gender,
        "missingness": missingness,
        "join_stats": join_stats,
    }

    return (
        X_train, y_train,
        X_valid, y_valid,
        X_test, y_test,
        meta_train, meta_valid, meta_test,
        candidate_cols,
        audit_info
    )


if __name__ == "__main__":
    prepare_model_a_datasets()
