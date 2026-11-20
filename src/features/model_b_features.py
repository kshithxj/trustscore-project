"""Feature matrix preparation for Model B (Model A features + Behavioral features).

Total features: 124 (Model A) + 20 (Behavioral) = 144 features.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config
from src.features.model_a_features import (
    prepare_model_a_datasets,
    validate_feature_list,
    FORBIDDEN_COLS,
    ALLOWED_SUBSTRING_EXCEPTIONS,
)
from src.features.behavioral import (
    build_and_join_behavioral_features,
    ALL_BEHAVIORAL_COLS,
)


def prepare_model_b_datasets(
    config: Dict[str, Any] | None = None
) -> Tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    List[str],
    Dict[str, Any]
]:
    """Prepare train, valid, and test feature matrices for Model B.
    
    Reuses the exact Model A feature preparation (and train-only categoricals)
    and joins the 20 behavioral features.
    """
    if config is None:
        config = load_config()

    metrics_dir = _PROJECT_ROOT / config["paths"]["metrics"]
    splits_dir = _PROJECT_ROOT / config["paths"]["splits"]
    processed_dir = _PROJECT_ROOT / config["paths"]["processed"]

    # 1. Get Model A datasets (ensuring identical categorical mappings and feature ordering)
    (
        X_train_a, y_train,
        X_valid_a, y_valid,
        X_test_a, y_test,
        meta_train, meta_valid, meta_test,
        model_a_feature_names,
        audit_info_a
    ) = prepare_model_a_datasets(config=config)

    # 2. Load or build behavioral features
    behavioral_parquet = processed_dir / "behavioral_features.parquet"
    if behavioral_parquet.exists():
        print(f"[model_b_features] Loading precomputed behavioral features from {behavioral_parquet}...")
        beh_df = pd.read_parquet(behavioral_parquet)
    else:
        print("[model_b_features] Building behavioral features...")
        _, _ = build_and_join_behavioral_features(config=config)
        beh_df = pd.read_parquet(behavioral_parquet)

    # 3. Construct Model B feature list
    model_b_feature_names = list(model_a_feature_names) + list(ALL_BEHAVIORAL_COLS)

    # Validate feature list against forbidden columns and GENDER/AGE checks
    validate_feature_list(model_b_feature_names)

    # Persist feature list
    features_txt_path = metrics_dir / "model_b_features.txt"
    with open(features_txt_path, "w", encoding="utf-8") as f:
        for feat in model_b_feature_names:
            f.write(f"{feat}\n")
    print(f"[model_b_features] Saved {features_txt_path} ({len(model_b_feature_names)} features)")

    # 4. Align behavioral features by SK_ID_CURR for each split
    beh_indexed = beh_df.set_index("SK_ID_CURR")

    train_beh = beh_indexed.loc[meta_train["SK_ID_CURR"], ALL_BEHAVIORAL_COLS].reset_index(drop=True)
    valid_beh = beh_indexed.loc[meta_valid["SK_ID_CURR"], ALL_BEHAVIORAL_COLS].reset_index(drop=True)
    test_beh = beh_indexed.loc[meta_test["SK_ID_CURR"], ALL_BEHAVIORAL_COLS].reset_index(drop=True)

    # Concatenate Model A features with behavioral features
    X_train_b = pd.concat([X_train_a.reset_index(drop=True), train_beh], axis=1)
    X_valid_b = pd.concat([X_valid_a.reset_index(drop=True), valid_beh], axis=1)
    X_test_b = pd.concat([X_test_a.reset_index(drop=True), test_beh], axis=1)

    # Ensure columns match exact order of model_b_feature_names
    X_train_b = X_train_b[model_b_feature_names]
    X_valid_b = X_valid_b[model_b_feature_names]
    X_test_b = X_test_b[model_b_feature_names]

    audit_info = {
        "n_model_a_features": len(model_a_feature_names),
        "n_behavioral_features": len(ALL_BEHAVIORAL_COLS),
        "total_features": len(model_b_feature_names),
        "feature_list": model_b_feature_names,
    }

    return (
        X_train_b, y_train,
        X_valid_b, y_valid,
        X_test_b, y_test,
        meta_train, meta_valid, meta_test,
        model_b_feature_names,
        audit_info
    )


if __name__ == "__main__":
    prepare_model_b_datasets()
