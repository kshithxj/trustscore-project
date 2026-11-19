"""Generate the Phase-1 audit report (JSON + console).

Usage:
    python -m src.data.audit
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config

# Columns to check for missingness
_KEY_COLS = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_GOODS_PRICE",
    "AMT_ANNUITY",
    "OCCUPATION_TYPE",
    "OWN_CAR_AGE",
    "DAYS_EMPLOYED",
    "APARTMENTS_AVG",
    "BASEMENTAREA_AVG",
    "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG",
    "COMMONAREA_AVG",
    "ELEVATORS_AVG",
    "ENTRANCES_AVG",
    "FLOORSMAX_AVG",
    "FLOORSMIN_AVG",
    "LANDAREA_AVG",
    "LIVINGAPARTMENTS_AVG",
    "LIVINGAREA_AVG",
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAREA_AVG",
]


def _segment_counts(df: pd.DataFrame, col: str = "segment") -> dict:
    return df[col].value_counts().to_dict()


def run_audit(config: dict | None = None) -> dict:
    """Run the Phase-1 audit and save the report JSON."""
    if config is None:
        config = load_config()

    processed = _PROJECT_ROOT / config["paths"]["processed"]
    splits_dir = _PROJECT_ROOT / config["paths"]["splits"]
    metrics_dir = _PROJECT_ROOT / config["paths"]["metrics"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # --- Load artefacts ------------------------------------------------------
    base = pd.read_parquet(processed / "applicant_base.parquet")
    eval_attrs = pd.read_parquet(processed / "eval_attrs.parquet")

    # --- Hard guards on base -------------------------------------------------
    n_base = len(base)
    if n_base < 300_000:
        raise ValueError(
            f"applicant_base.parquet has only {n_base:,} rows (expected > 300,000). "
            f"It may have been built from a synthetic or truncated file."
        )
    if "TARGET" not in base.columns:
        raise ValueError("applicant_base.parquet is missing the TARGET column.")
    if "SK_ID_CURR" not in base.columns:
        raise ValueError("applicant_base.parquet is missing the SK_ID_CURR column.")
    dup_count = int(base["SK_ID_CURR"].duplicated().sum())
    if dup_count > 0:
        raise ValueError(
            f"applicant_base.parquet has {dup_count:,} duplicate SK_ID_CURR values."
        )
    target_mean = float(base["TARGET"].mean())
    if not (0.05 <= target_mean <= 0.12):
        raise ValueError(
            f"applicant_base TARGET mean is {target_mean:.6f} "
            f"(expected 0.05–0.12). Data may be synthetic or corrupted."
        )
    print(
        f"[audit] Guards passed: {n_base:,} rows, "
        f"TARGET mean {target_mean:.6f}, 0 duplicate IDs."
    )

    split_dfs = {}
    for name in ("train", "valid", "test"):
        split_dfs[name] = pd.read_csv(splits_dir / f"{name}_ids.csv")

    # --- Determine which stratification was used -----------------------------
    # Read from the split.py output by checking if a metadata hint was saved,
    # otherwise infer by re-computing.
    base_tmp = base[["SK_ID_CURR", "TARGET", "segment"]].copy()
    base_tmp["strat_target_segment"] = (
        base_tmp["TARGET"].astype(str) + "_" + base_tmp["segment"]
    )
    min_stratum = base_tmp["strat_target_segment"].value_counts().min()
    strat_used = "segment_only" if min_stratum < 10 else "target_x_segment"

    # --- Segment counts (overall + per split) --------------------------------
    segment_counts: dict = {
        "overall": _segment_counts(base),
    }
    for name, sdf in split_dfs.items():
        segment_counts[name] = _segment_counts(sdf)

    # --- Default rate per segment (overall) -----------------------------------
    default_rate = (
        base.groupby("segment")["TARGET"]
        .mean()
        .round(6)
        .to_dict()
    )

    # --- Default rate per split, overall and per segment ---------------------
    default_rate_by_split: dict = {}
    for name, sdf in split_dfs.items():
        split_base = base[base["SK_ID_CURR"].isin(sdf["SK_ID_CURR"])]
        overall_rate = round(float(split_base["TARGET"].mean()), 6)
        per_seg = split_base.groupby("segment")["TARGET"].mean().round(6).to_dict()
        default_rate_by_split[name] = {"overall": overall_rate, "by_segment": per_seg}

    # --- Gender & age_bin counts ---------------------------------------------
    gender_counts = eval_attrs["gender"].value_counts().to_dict()
    age_bin_counts = (
        eval_attrs["age_bin"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # --- Missingness of key columns ------------------------------------------
    present_key_cols = [c for c in _KEY_COLS if c in base.columns]
    missingness: dict = {}
    for col in present_key_cols:
        n_miss = int(base[col].isna().sum())
        missingness[col] = {
            "n_missing": n_miss,
            "pct_missing": round(n_miss / len(base) * 100, 2),
        }

    # --- Warnings ------------------------------------------------------------
    warnings: list[str] = []
    for name in ("train", "valid", "test"):
        seg_c = segment_counts[name]
        for seg, cnt in seg_c.items():
            if name == "test" and cnt < 100:
                msg = f"Segment '{seg}' has only {cnt} applicants in the test split."
                warnings.append(msg)

    # --- Assemble report -----------------------------------------------------
    report = {
        "n_applicants": len(base),
        "n_columns_base": base.shape[1],
        "train_row_count": len(base),
        "target_mean": round(target_mean, 6),
        "duplicate_sk_id_curr": dup_count,
        "stratification_used": strat_used,
        "smallest_stratum": int(min_stratum),
        "segment_counts": segment_counts,
        "default_rate_by_segment": default_rate,
        "default_rate_by_split": default_rate_by_split,
        "gender_counts": gender_counts,
        "age_bin_counts": age_bin_counts,
        "missingness": missingness,
        "warnings": warnings,
    }

    # --- Save JSON -----------------------------------------------------------
    out_path = metrics_dir / "phase1_audit.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"[audit] Saved {out_path}")

    # --- Console summary -----------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 1 AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total applicants : {report['n_applicants']:,}")
    print(f"Stratification   : {report['stratification_used']}")
    print(f"\nSegment counts (overall):")
    for seg, cnt in report["segment_counts"]["overall"].items():
        print(f"  {seg:12s}: {cnt:>8,}")
    print(f"\nDefault rate by segment:")
    for seg, rate in report["default_rate_by_segment"].items():
        print(f"  {seg:12s}: {rate:.4f}")
    print(f"\nSplit sizes:")
    for name in ("train", "valid", "test"):
        total = sum(report["segment_counts"][name].values())
        print(f"  {name:6s}: {total:>8,}")
    print(f"\nGender: {report['gender_counts']}")
    print(f"Age bins: {report['age_bin_counts']}")
    print(f"\nTop-5 missing columns:")
    sorted_miss = sorted(
        report["missingness"].items(), key=lambda x: x[1]["pct_missing"], reverse=True
    )
    for col, info in sorted_miss[:5]:
        print(f"  {col:35s}: {info['pct_missing']:6.2f}% ({info['n_missing']:,})")
    if warnings:
        print(f"\n⚠ WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print(f"\n✓ No warnings.")
    print("=" * 60)

    return report


if __name__ == "__main__":
    run_audit()
