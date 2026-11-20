"""Bureau feature aggregation and join logic.

Usage:
    python -m src.features.bureau
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config

BUREAU_FEATURE_COLS = [
    "bureau_n_credits",
    "bureau_n_active",
    "bureau_days_since_first_credit",
    "bureau_days_since_last_credit",
    "bureau_total_overdue",
    "bureau_total_debt",
]


def aggregate_bureau_data(bureau_path: Path) -> pd.DataFrame:
    """Read bureau.csv and aggregate metrics per SK_ID_CURR before joining.
    
    Columns produced:
    - bureau_n_credits
    - bureau_n_active
    - bureau_days_since_first_credit (-min(DAYS_CREDIT))
    - bureau_days_since_last_credit (-max(DAYS_CREDIT))
    - bureau_total_overdue (sum of AMT_CREDIT_SUM_OVERDUE, NA -> 0)
    - bureau_total_debt (sum of AMT_CREDIT_SUM_DEBT, NA -> 0)
    """
    usecols = [
        "SK_ID_CURR",
        "CREDIT_ACTIVE",
        "DAYS_CREDIT",
        "AMT_CREDIT_SUM_OVERDUE",
        "AMT_CREDIT_SUM_DEBT",
    ]
    print(f"[bureau] Reading {bureau_path} with {len(usecols)} columns...")
    df = pd.read_csv(bureau_path, usecols=usecols)

    # Fill NA with 0 for monetary overdue and debt before summing
    df["AMT_CREDIT_SUM_OVERDUE"] = df["AMT_CREDIT_SUM_OVERDUE"].fillna(0.0)
    df["AMT_CREDIT_SUM_DEBT"] = df["AMT_CREDIT_SUM_DEBT"].fillna(0.0)
    df["is_active"] = (df["CREDIT_ACTIVE"] == "Active").astype(int)

    print("[bureau] Aggregating by SK_ID_CURR...")
    agg = df.groupby("SK_ID_CURR").agg(
        bureau_n_credits=("DAYS_CREDIT", "count"),
        bureau_n_active=("is_active", "sum"),
        min_days_credit=("DAYS_CREDIT", "min"),
        max_days_credit=("DAYS_CREDIT", "max"),
        bureau_total_overdue=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        bureau_total_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
    ).reset_index()

    # DAYS_CREDIT are negative days relative to current application
    # -min gives the days since the earliest/first credit (larger positive number)
    # -max gives the days since the latest/last credit (smaller positive number)
    agg["bureau_days_since_first_credit"] = -agg["min_days_credit"]
    agg["bureau_days_since_last_credit"] = -agg["max_days_credit"]
    agg = agg.drop(columns=["min_days_credit", "max_days_credit"])

    # Enforce column order
    agg = agg[["SK_ID_CURR"] + BUREAU_FEATURE_COLS]

    # Validate non-negativity of days_since columns
    min_first = agg["bureau_days_since_first_credit"].min()
    min_last = agg["bureau_days_since_last_credit"].min()
    if min_first < 0 or min_last < 0:
        raise ValueError(
            f"Negative days_since detected: min_first={min_first}, min_last={min_last}"
        )

    print("[bureau] Summary of aggregated bureau columns:")
    for col in BUREAU_FEATURE_COLS:
        print(f"  {col}: min = {agg[col].min()}, max = {agg[col].max()}")

    return agg


def build_bureau_features(
    base_df: pd.DataFrame | None = None,
    config: Dict[str, Any] | None = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Left-join bureau aggregate features onto the applicant base table.
    
    Returns:
        joined_df: DataFrame with the 6 bureau features attached (NaN for unlinked)
        stats: Dictionary containing join check statistics and min/max metrics
    """
    if config is None:
        config = load_config()

    raw_path = _PROJECT_ROOT / config["paths"]["raw"] / "bureau.csv"
    processed_dir = _PROJECT_ROOT / config["paths"]["processed"]

    if base_df is None:
        base_path = processed_dir / "applicant_base.parquet"
        print(f"[bureau] Loading applicant base from {base_path}...")
        base_df = pd.read_parquet(base_path)

    rows_before = len(base_df)
    bureau_agg = aggregate_bureau_data(raw_path)

    print(f"[bureau] Joining with applicant base ({rows_before:,} rows)...")
    joined = base_df.merge(bureau_agg, on="SK_ID_CURR", how="left")
    rows_after = len(joined)
    dup_ids_after = int(joined["SK_ID_CURR"].duplicated().sum())

    print(f"[bureau] Applicant rows before: {rows_before:,}")
    print(f"[bureau] Applicant rows after:  {rows_after:,}")
    print(f"[bureau] Duplicated SK_ID_CURR after: {dup_ids_after}")

    if rows_after != rows_before:
        raise ValueError(
            f"Row count mismatch after bureau join: before={rows_before}, after={rows_after}"
        )
    if dup_ids_after > 0:
        raise ValueError(
            f"Duplicate SK_ID_CURR created after bureau join: {dup_ids_after}"
        )

    # Collect stats
    col_min_max = {}
    for col in BUREAU_FEATURE_COLS:
        col_min_max[col] = {
            "min": float(joined[col].min()),
            "max": float(joined[col].max()),
        }

    stats = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicated_sk_id_curr": dup_ids_after,
        "col_min_max": col_min_max,
    }

    return joined, stats


if __name__ == "__main__":
    build_bureau_features()
