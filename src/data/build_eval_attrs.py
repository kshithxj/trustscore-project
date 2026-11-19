"""Build eval_attrs.parquet — protected-attribute columns for fairness evaluation.

Usage:
    python -m src.data.build_eval_attrs
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config

_AGE_BINS = [0, 30, 45, 60, np.inf]
_AGE_LABELS = ["<30", "30-44", "45-59", "60+"]


def build_eval_attrs(config: dict | None = None) -> pd.DataFrame:
    """Create the evaluation-only attributes table.

    Columns produced
    ----------------
    SK_ID_CURR, gender, gender_xna, age_years, age_bin
    """
    if config is None:
        config = load_config()

    processed = _PROJECT_ROOT / config["paths"]["processed"]
    base_path = processed / "applicant_base.parquet"

    print("[build_eval_attrs] Loading applicant_base.parquet …")
    base = pd.read_parquet(base_path, columns=["SK_ID_CURR", "CODE_GENDER", "DAYS_BIRTH"])

    df = pd.DataFrame({"SK_ID_CURR": base["SK_ID_CURR"]})
    df["gender"] = base["CODE_GENDER"]
    df["gender_xna"] = base["CODE_GENDER"] == "XNA"

    # DAYS_BIRTH is negative (days before application)
    df["age_years"] = (base["DAYS_BIRTH"].abs() / 365.25).round(2)
    df["age_bin"] = pd.cut(
        df["age_years"], bins=_AGE_BINS, labels=_AGE_LABELS, right=False
    )

    out_path = processed / "eval_attrs.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"[build_eval_attrs] Saved {out_path}  ({len(df):,} rows)")
    print(f"  gender counts : {df['gender'].value_counts().to_dict()}")
    print(f"  gender_xna    : {df['gender_xna'].sum()}")
    print(f"  age_bin counts: {df['age_bin'].value_counts().sort_index().to_dict()}")

    return df


if __name__ == "__main__":
    build_eval_attrs()
