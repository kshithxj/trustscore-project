"""Behavioral lender-history feature engineering and validation.

Tables aggregated:
- previous_application (6 features)
- installments_payments (6 features)
- POS_CASH_balance (4 features)
- credit_card_balance (4 features)

Total: 20 behavioral features.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config

BEHAVIORAL_COLS_BY_TABLE = {
    "previous_application": [
        "prev_n_apps",
        "prev_approved_rate",
        "prev_refused_rate",
        "prev_mean_amt_credit",
        "prev_mean_amt_annuity",
        "prev_n_apps_last_365d",
    ],
    "installments_payments": [
        "inst_days_late_mean",
        "inst_days_late_max",
        "inst_days_late_std",
        "inst_frac_late",
        "inst_frac_early",
        "inst_payment_ratio_mean",
    ],
    "POS_CASH_balance": [
        "pos_n_records",
        "pos_dpd_mean",
        "pos_dpd_max",
        "pos_months_with_dpd",
    ],
    "credit_card_balance": [
        "cc_n_records",
        "cc_util_mean",
        "cc_util_max",
        "cc_months_with_dpd",
    ],
}

ALL_BEHAVIORAL_COLS = [
    col for cols in BEHAVIORAL_COLS_BY_TABLE.values() for col in cols
]


def aggregate_previous_application(raw_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Aggregate previous_application.csv per SK_ID_CURR."""
    path = raw_dir / "previous_application.csv"
    print(f"[behavioral] Reading {path.name}...")
    usecols = ["SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_DECISION"]
    dtypes = {
        "SK_ID_CURR": np.int32,
        "AMT_CREDIT": np.float32,
        "AMT_ANNUITY": np.float32,
        "DAYS_DECISION": np.float32,
    }
    df = pd.read_csv(path, usecols=usecols, dtype=dtypes)

    time_gt_0 = int((df["DAYS_DECISION"] > 0).sum())
    if time_gt_0 > 0:
        raise ValueError(f"CRITICAL: DAYS_DECISION has {time_gt_0} values > 0 (future leakage)!")

    df["is_approved"] = (df["NAME_CONTRACT_STATUS"] == "Approved").astype(np.float32)
    df["is_refused"] = (df["NAME_CONTRACT_STATUS"] == "Refused").astype(np.float32)
    df["is_last_365d"] = (df["DAYS_DECISION"] >= -365.0).astype(np.float32)

    agg = df.groupby("SK_ID_CURR").agg(
        prev_n_apps=("DAYS_DECISION", "count"),
        prev_approved_rate=("is_approved", "mean"),
        prev_refused_rate=("is_refused", "mean"),
        prev_mean_amt_credit=("AMT_CREDIT", "mean"),
        prev_mean_amt_annuity=("AMT_ANNUITY", "mean"),
        prev_n_apps_last_365d=("is_last_365d", "sum"),
    ).reset_index()

    stats = {
        "time_gt_0": {"DAYS_DECISION": time_gt_0},
        "rows": len(agg),
    }
    return agg, stats


def aggregate_installments_payments(raw_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Aggregate installments_payments.csv per SK_ID_CURR."""
    path = raw_dir / "installments_payments.csv"
    print(f"[behavioral] Reading {path.name}...")
    usecols = ["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]
    dtypes = {
        "SK_ID_CURR": np.int32,
        "DAYS_INSTALMENT": np.float32,
        "DAYS_ENTRY_PAYMENT": np.float32,
        "AMT_INSTALMENT": np.float32,
        "AMT_PAYMENT": np.float32,
    }
    df = pd.read_csv(path, usecols=usecols, dtype=dtypes)

    days_inst_gt_0 = int((df["DAYS_INSTALMENT"] > 0).sum())
    days_entry_gt_0 = int((df["DAYS_ENTRY_PAYMENT"] > 0).sum())
    if days_inst_gt_0 > 0 or days_entry_gt_0 > 0:
        raise ValueError(
            f"CRITICAL: installments_payments has time values > 0: "
            f"DAYS_INSTALMENT > 0 ({days_inst_gt_0}), DAYS_ENTRY_PAYMENT > 0 ({days_entry_gt_0})!"
        )

    # days_late = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT (positive = late)
    df["days_late"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    # only where DAYS_ENTRY_PAYMENT is not NaN
    valid_late = df["days_late"].notna()

    df["is_late"] = np.nan
    df.loc[valid_late, "is_late"] = (df.loc[valid_late, "days_late"] > 0).astype(np.float32)

    df["is_early"] = np.nan
    df.loc[valid_late, "is_early"] = (df.loc[valid_late, "days_late"] < 0).astype(np.float32)

    mask_amt = df["AMT_INSTALMENT"] > 0
    df["payment_ratio"] = np.nan
    df.loc[mask_amt, "payment_ratio"] = df.loc[mask_amt, "AMT_PAYMENT"] / df.loc[mask_amt, "AMT_INSTALMENT"]

    agg = df.groupby("SK_ID_CURR").agg(
        inst_days_late_mean=("days_late", "mean"),
        inst_days_late_max=("days_late", "max"),
        inst_days_late_std=("days_late", "std"),
        inst_frac_late=("is_late", "mean"),
        inst_frac_early=("is_early", "mean"),
        inst_payment_ratio_mean=("payment_ratio", "mean"),
    ).reset_index()

    stats = {
        "time_gt_0": {
            "DAYS_INSTALMENT": days_inst_gt_0,
            "DAYS_ENTRY_PAYMENT": days_entry_gt_0,
        },
        "rows": len(agg),
    }
    return agg, stats


def aggregate_pos_cash(raw_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Aggregate POS_CASH_balance.csv per SK_ID_CURR."""
    path = raw_dir / "POS_CASH_balance.csv"
    print(f"[behavioral] Reading {path.name}...")
    usecols = ["SK_ID_CURR", "SK_DPD", "MONTHS_BALANCE"]
    dtypes = {
        "SK_ID_CURR": np.int32,
        "SK_DPD": np.int32,
        "MONTHS_BALANCE": np.int32,
    }
    df = pd.read_csv(path, usecols=usecols, dtype=dtypes)

    months_gt_0 = int((df["MONTHS_BALANCE"] > 0).sum())
    if months_gt_0 > 0:
        raise ValueError(f"CRITICAL: POS_CASH_balance MONTHS_BALANCE has {months_gt_0} values > 0!")

    df["has_dpd"] = (df["SK_DPD"] > 0).astype(np.float32)

    agg = df.groupby("SK_ID_CURR").agg(
        pos_n_records=("MONTHS_BALANCE", "count"),
        pos_dpd_mean=("SK_DPD", "mean"),
        pos_dpd_max=("SK_DPD", "max"),
        pos_months_with_dpd=("has_dpd", "sum"),
    ).reset_index()

    stats = {
        "time_gt_0": {"MONTHS_BALANCE": months_gt_0},
        "rows": len(agg),
    }
    return agg, stats


def aggregate_credit_card(raw_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Aggregate credit_card_balance.csv per SK_ID_CURR."""
    path = raw_dir / "credit_card_balance.csv"
    print(f"[behavioral] Reading {path.name}...")
    usecols = ["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD", "MONTHS_BALANCE"]
    dtypes = {
        "SK_ID_CURR": np.int32,
        "AMT_BALANCE": np.float32,
        "AMT_CREDIT_LIMIT_ACTUAL": np.float32,
        "SK_DPD": np.int32,
        "MONTHS_BALANCE": np.int32,
    }
    df = pd.read_csv(path, usecols=usecols, dtype=dtypes)

    months_gt_0 = int((df["MONTHS_BALANCE"] > 0).sum())
    if months_gt_0 > 0:
        raise ValueError(f"CRITICAL: credit_card_balance MONTHS_BALANCE has {months_gt_0} values > 0!")

    mask_limit = df["AMT_CREDIT_LIMIT_ACTUAL"] > 0
    df["utilization"] = np.nan
    df.loc[mask_limit, "utilization"] = (
        df.loc[mask_limit, "AMT_BALANCE"] / df.loc[mask_limit, "AMT_CREDIT_LIMIT_ACTUAL"]
    )
    df["has_dpd"] = (df["SK_DPD"] > 0).astype(np.float32)

    util_min = float(df["utilization"].min())
    util_max = float(df["utilization"].max())

    agg = df.groupby("SK_ID_CURR").agg(
        cc_n_records=("MONTHS_BALANCE", "count"),
        cc_util_mean=("utilization", "mean"),
        cc_util_max=("utilization", "max"),
        cc_months_with_dpd=("has_dpd", "sum"),
    ).reset_index()

    stats = {
        "time_gt_0": {"MONTHS_BALANCE": months_gt_0},
        "utilization_min": util_min,
        "utilization_max": util_max,
        "rows": len(agg),
    }
    return agg, stats


def build_and_join_behavioral_features(
    config: Dict[str, Any] | None = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build all 4 behavioral tables, perform left-joins to applicant base, and validate."""
    if config is None:
        config = load_config()

    raw_dir = _PROJECT_ROOT / config["paths"]["raw"]
    processed_dir = _PROJECT_ROOT / config["paths"]["processed"]
    metrics_dir = _PROJECT_ROOT / config["paths"]["metrics"]

    # Load base table
    base_path = processed_dir / "applicant_base.parquet"
    print(f"[behavioral] Loading applicant_base from {base_path}...")
    base = pd.read_parquet(base_path)
    initial_rows = len(base)

    # 1. Aggregate each table
    prev_agg, prev_stats = aggregate_previous_application(raw_dir)
    inst_agg, inst_stats = aggregate_installments_payments(raw_dir)
    pos_agg, pos_stats = aggregate_pos_cash(raw_dir)
    cc_agg, cc_stats = aggregate_credit_card(raw_dir)

    # 2. Sequential LEFT joins with strict invariants
    current_df = base
    join_checks = {}

    tables = [
        ("previous_application", prev_agg),
        ("installments_payments", inst_agg),
        ("POS_CASH_balance", pos_agg),
        ("credit_card_balance", cc_agg),
    ]

    for name, agg_table in tables:
        rows_before = len(current_df)
        current_df = current_df.merge(agg_table, on="SK_ID_CURR", how="left")
        rows_after = len(current_df)
        dup_ids = int(current_df["SK_ID_CURR"].duplicated().sum())

        print(f"[behavioral] Joined {name}: rows before={rows_before:,}, rows after={rows_after:,}, duplicated={dup_ids}")
        if rows_after != rows_before:
            raise ValueError(f"Row count changed after joining {name}: before={rows_before}, after={rows_after}")
        if dup_ids > 0:
            raise ValueError(f"Duplicates introduced after joining {name}: {dup_ids}")

        join_checks[name] = {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "duplicated_sk_id_curr": dup_ids,
        }

    # 3. Save applicant-level behavioral table
    behavioral_table = current_df[["SK_ID_CURR"] + ALL_BEHAVIORAL_COLS].copy()
    out_parquet = processed_dir / "behavioral_features.parquet"
    behavioral_table.to_parquet(out_parquet, index=False, engine="pyarrow")
    print(f"[behavioral] Saved {out_parquet} ({len(behavioral_table):,} rows, {len(ALL_BEHAVIORAL_COLS)} features)")

    # 4. Leakage and Consistency Checks (STEP 2)
    time_checks = {
        "previous_application_DAYS_DECISION": prev_stats["time_gt_0"]["DAYS_DECISION"],
        "installments_payments_DAYS_INSTALMENT": inst_stats["time_gt_0"]["DAYS_INSTALMENT"],
        "installments_payments_DAYS_ENTRY_PAYMENT": inst_stats["time_gt_0"]["DAYS_ENTRY_PAYMENT"],
        "POS_CASH_balance_MONTHS_BALANCE": pos_stats["time_gt_0"]["MONTHS_BALANCE"],
        "credit_card_balance_MONTHS_BALANCE": cc_stats["time_gt_0"]["MONTHS_BALANCE"],
    }
    for col_name, cnt in time_checks.items():
        if cnt > 0:
            raise ValueError(f"STOP: Found {cnt} values > 0 in {col_name}!")

    # Check segment consistency:
    # no_file: ALL behavioral columns must be NaN
    # thin_file: at least one behavioral table must be non-NaN
    no_file_mask = current_df["segment"] == "no_file"
    no_file_df = current_df.loc[no_file_mask, ALL_BEHAVIORAL_COLS]
    # Violations: any row where not all behavioral columns are NaN
    no_file_violations = int((~no_file_df.isna().all(axis=1)).sum())

    thin_file_mask = current_df["segment"] == "thin_file"
    thin_file_df = current_df.loc[thin_file_mask, ALL_BEHAVIORAL_COLS]
    # Violations: any row where all behavioral columns are NaN
    thin_file_violations = int((thin_file_df.isna().all(axis=1)).sum())

    print(f"[behavioral] Segment consistency: no_file violations = {no_file_violations}, thin_file violations = {thin_file_violations}")
    if no_file_violations > 0 or thin_file_violations > 0:
        raise ValueError(
            f"Segment consistency check failed! no_file violations: {no_file_violations}, "
            f"thin_file violations: {thin_file_violations}"
        )

    # Share of missing values for each table's columns by segment
    segments = ["full_file", "thin_file", "no_file"]
    missingness_by_segment: Dict[str, Dict[str, Dict[str, float]]] = {}

    for seg in segments:
        seg_mask = current_df["segment"] == seg
        seg_count = int(seg_mask.sum())
        missingness_by_segment[seg] = {}
        for table_name, cols in BEHAVIORAL_COLS_BY_TABLE.items():
            missingness_by_segment[seg][table_name] = {}
            for col in cols:
                miss_cnt = int(current_df.loc[seg_mask, col].isna().sum())
                share_miss = round(miss_cnt / seg_count, 6) if seg_count > 0 else 0.0
                missingness_by_segment[seg][table_name][col] = share_miss

    checks_json = {
        "time_sign_check": time_checks,
        "segment_consistency": {
            "no_file_violations": no_file_violations,
            "thin_file_violations": thin_file_violations,
        },
        "cc_utilization": {
            "min": cc_stats["utilization_min"],
            "max": cc_stats["utilization_max"],
        },
        "missingness_by_segment": missingness_by_segment,
        "join_checks": join_checks,
    }

    checks_out_path = metrics_dir / "model_b_checks.json"
    with open(checks_out_path, "w", encoding="utf-8") as f:
        json.dump(checks_json, f, indent=2)
    print(f"[behavioral] Saved {checks_out_path}")

    return current_df, checks_json


if __name__ == "__main__":
    build_and_join_behavioral_features()
