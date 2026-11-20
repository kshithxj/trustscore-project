"""Part 0: Descriptive Base Rates on the TEST split for TrustScore.

Computes group size, defaults count, and default rate across:
- gender (F, M; excludes XNA)
- binary age (<median, >=median based on TRAIN median age)
- four age bins (<30, 30-44, 45-59, 60+)
- segment (full_file, thin_file, no_file)
- segment x gender (where group size >= 100)
- segment x binary age (where group size >= 100)

Saves results to reports/tables/base_rates.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config


def compute_base_rates(config: Dict[str, Any] | None = None) -> pd.DataFrame:
    """Compute base rates for all required demographic groups on the TEST split."""
    if config is None:
        config = load_config()

    test_ids_path = _PROJECT_ROOT / config["paths"]["splits"] / "test_ids.csv"
    train_ids_path = _PROJECT_ROOT / config["paths"]["splits"] / "train_ids.csv"
    app_base_path = _PROJECT_ROOT / config["paths"]["processed"] / "applicant_base.parquet"
    eval_attrs_path = _PROJECT_ROOT / config["paths"]["processed"] / "eval_attrs.parquet"
    tables_dir = _PROJECT_ROOT / config["paths"]["reports"] / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    test_ids = pd.read_csv(test_ids_path)["SK_ID_CURR"]
    train_ids = pd.read_csv(train_ids_path)["SK_ID_CURR"]

    app_base = pd.read_parquet(app_base_path, columns=["SK_ID_CURR", "TARGET", "segment"])
    eval_attrs = pd.read_parquet(eval_attrs_path, columns=["SK_ID_CURR", "gender", "age_years", "age_bin"])

    df_all = app_base.merge(eval_attrs, on="SK_ID_CURR", how="inner")

    # Determine train median age
    train_df = df_all[df_all["SK_ID_CURR"].isin(train_ids)]
    train_median_age = float(train_df["age_years"].median())

    test_df = df_all[df_all["SK_ID_CURR"].isin(test_ids)].copy()
    test_df["binary_age"] = np.where(test_df["age_years"] >= train_median_age, ">=median", "<median")

    rows: List[Dict[str, Any]] = []

    def _add_groups(dimension: str, group_col: str, data: pd.DataFrame, min_size: int = 0):
        for grp, sub in data.groupby(group_col, observed=True):
            n = len(sub)
            if n < min_size:
                continue
            defs = int(sub["TARGET"].sum())
            dr = float(defs / n) if n > 0 else 0.0
            rows.append({
                "dimension": dimension,
                "group": str(grp),
                "size": n,
                "defaults": defs,
                "default_rate": dr,
            })

    # 1. Gender (exclude XNA)
    test_gender = test_df[test_df["gender"] != "XNA"]
    _add_groups("gender", "gender", test_gender)

    # 2. Binary age
    _add_groups("binary_age", "binary_age", test_df)

    # 3. Four age bins
    _add_groups("age_bin", "age_bin", test_df)

    # 4. Segment
    _add_groups("segment", "segment", test_df)

    # 5. Segment x Gender (>= 100)
    test_gender_non_xna = test_df[test_df["gender"] != "XNA"].copy()
    test_gender_non_xna["segment_x_gender"] = (
        test_gender_non_xna["segment"].astype(str) + " x " + test_gender_non_xna["gender"].astype(str)
    )
    _add_groups("segment_x_gender", "segment_x_gender", test_gender_non_xna, min_size=100)

    # 6. Segment x Binary Age (>= 100)
    test_df_seg_age = test_df.copy()
    test_df_seg_age["segment_x_binary_age"] = (
        test_df_seg_age["segment"].astype(str) + " x " + test_df_seg_age["binary_age"].astype(str)
    )
    _add_groups("segment_x_binary_age", "segment_x_binary_age", test_df_seg_age, min_size=100)

    out_df = pd.DataFrame(rows)
    out_path = tables_dir / "base_rates.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[base_rates] Saved {len(out_df)} rows to {out_path}")
    return out_df


if __name__ == "__main__":
    df = compute_base_rates()
    print(df.to_string())
