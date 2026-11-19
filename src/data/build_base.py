"""Build the applicant base table from application_train + side-table presence flags.

Usage:
    python -m src.data.build_base
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config

# Side tables that contribute to the "has_lender_history" flag
_LENDER_TABLES = [
    "previous_application.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
]


def _unique_ids_from_csv(path: Path) -> set:
    """Read only SK_ID_CURR from a CSV and return the unique set."""
    return set(
        pd.read_csv(path, usecols=["SK_ID_CURR"], dtype={"SK_ID_CURR": np.int64})[
            "SK_ID_CURR"
        ].unique()
    )


def _validate_train(app: pd.DataFrame) -> None:
    """Hard guards on application_train — raises on any violation."""
    errors = []

    # Required columns
    for col in ("SK_ID_CURR", "TARGET"):
        if col not in app.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        raise ValueError(
            "application_train failed column checks:\n  " + "\n  ".join(errors)
        )

    # Row count
    if len(app) < 300_000:
        raise ValueError(
            f"application_train has only {len(app):,} rows (expected > 300,000). "
            f"This may be a synthetic or truncated file."
        )

    # Duplicate IDs
    dup_count = int(app["SK_ID_CURR"].duplicated().sum())
    if dup_count > 0:
        raise ValueError(
            f"application_train has {dup_count:,} duplicate SK_ID_CURR values."
        )

    # TARGET mean sanity
    target_mean = float(app["TARGET"].mean())
    if not (0.05 <= target_mean <= 0.12):
        raise ValueError(
            f"application_train TARGET mean is {target_mean:.6f} "
            f"(expected 0.05–0.12). Data may be synthetic or corrupted."
        )

    print(
        f"[build_base] Guards passed: {len(app):,} rows, "
        f"TARGET mean {target_mean:.6f}, 0 duplicate IDs."
    )


def build_base(config: dict | None = None) -> pd.DataFrame:
    """Build and return the applicant base DataFrame.

    Steps
    -----
    1. Load application_train.csv.
    2. Fix DAYS_EMPLOYED sentinel (365243 → NaN).
    3. Add has_bureau, has_lender_history, segment.
    4. Save to data/processed/applicant_base.parquet.
    """
    if config is None:
        config = load_config()

    raw = _PROJECT_ROOT / config["paths"]["raw"]
    processed = _PROJECT_ROOT / config["paths"]["processed"]
    processed.mkdir(parents=True, exist_ok=True)

    # --- 1. Load application_train ------------------------------------------
    print("[build_base] Loading application_train.csv …")
    app = pd.read_csv(raw / "application_train.csv")
    n_total = len(app)
    print(f"[build_base] Loaded {n_total:,} applicants, {app.shape[1]} columns.")

    # --- 1b. Hard guards -----------------------------------------------------
    _validate_train(app)

    # --- 2. Fix DAYS_EMPLOYED sentinel ---------------------------------------
    sentinel_mask = app["DAYS_EMPLOYED"] == 365243
    n_sentinel = sentinel_mask.sum()
    app.loc[sentinel_mask, "DAYS_EMPLOYED"] = np.nan
    print(f"[build_base] Replaced {n_sentinel:,} DAYS_EMPLOYED sentinels with NaN.")

    # --- 3a. has_bureau ------------------------------------------------------
    print("[build_base] Scanning bureau.csv for SK_ID_CURR …")
    bureau_ids = _unique_ids_from_csv(raw / "bureau.csv")
    app["has_bureau"] = app["SK_ID_CURR"].isin(bureau_ids)
    print(f"[build_base] {app['has_bureau'].sum():,} applicants have bureau records.")

    # --- 3b. has_lender_history (any of 4 lender tables) ---------------------
    lender_ids: set = set()
    for fname in _LENDER_TABLES:
        fpath = raw / fname
        if fpath.exists():
            print(f"[build_base] Scanning {fname} …")
            lender_ids |= _unique_ids_from_csv(fpath)
        else:
            print(f"[build_base] WARNING: {fname} not found, skipping.")
    app["has_lender_history"] = app["SK_ID_CURR"].isin(lender_ids)
    print(
        f"[build_base] {app['has_lender_history'].sum():,} applicants have lender history."
    )

    # --- 3c. Segment ---------------------------------------------------------
    conditions = [
        app["has_bureau"],
        ~app["has_bureau"] & app["has_lender_history"],
    ]
    choices = ["full_file", "thin_file"]
    app["segment"] = np.select(conditions, choices, default="no_file")
    print("[build_base] Segment distribution:")
    print(app["segment"].value_counts().to_string())

    # --- 4. Save -------------------------------------------------------------
    out_path = processed / "applicant_base.parquet"
    app.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"[build_base] Saved {out_path}  ({n_total:,} rows)")

    return app


if __name__ == "__main__":
    build_base()
