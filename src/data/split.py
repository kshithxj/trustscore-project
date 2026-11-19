"""Create the stratified 60/20/20 train-valid-test split.

Usage:
    python -m src.data.split
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config, set_seed


def create_split(config: dict | None = None) -> dict:
    """Create and persist a stratified train / valid / test split.

    Returns
    -------
    dict
        ``{"stratification": ..., "train": DataFrame, "valid": ..., "test": ...}``
    """
    if config is None:
        config = load_config()

    seed = set_seed(config=config)
    fracs = config["split"]
    processed = _PROJECT_ROOT / config["paths"]["processed"]
    splits_dir = _PROJECT_ROOT / config["paths"]["splits"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("[split] Loading applicant_base.parquet …")
    base = pd.read_parquet(
        processed / "applicant_base.parquet",
        columns=["SK_ID_CURR", "TARGET", "segment"],
    )

    # --- Decide stratification column ----------------------------------------
    base["strat_target_segment"] = (
        base["TARGET"].astype(str) + "_" + base["segment"]
    )
    min_stratum = base["strat_target_segment"].value_counts().min()

    if min_stratum < 10:
        strat_col = "segment"
        strat_used = "segment_only"
        print(
            f"[split] WARNING: smallest TARGET×segment stratum has {min_stratum} rows "
            f"(< 10). Falling back to stratify by segment only."
        )
    else:
        strat_col = "strat_target_segment"
        strat_used = "target_x_segment"
        print(
            f"[split] Stratifying by TARGET × segment "
            f"(smallest stratum: {min_stratum})."
        )

    # --- First split: train vs temp (valid + test) ---------------------------
    train_frac = fracs["train"]
    temp_frac = fracs["valid"] + fracs["test"]
    valid_of_temp = fracs["valid"] / temp_frac  # 0.5 when 0.2/0.4

    train_df, temp_df = train_test_split(
        base,
        train_size=train_frac,
        stratify=base[strat_col],
        random_state=seed,
    )

    # --- Second split: valid vs test -----------------------------------------
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=valid_of_temp,
        stratify=temp_df[strat_col],
        random_state=seed,
    )

    # --- Save id files -------------------------------------------------------
    keep_cols = ["SK_ID_CURR", "TARGET", "segment"]
    for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        out = splits_dir / f"{name}_ids.csv"
        df[keep_cols].to_csv(out, index=False)
        print(f"[split] {name}: {len(df):,} rows → {out}")

    print(f"[split] Stratification used: {strat_used}")

    return {
        "stratification": strat_used,
        "train": train_df[keep_cols],
        "valid": valid_df[keep_cols],
        "test": test_df[keep_cols],
    }


if __name__ == "__main__":
    create_split()
