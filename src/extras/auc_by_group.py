"""Evaluate Model A and Model B AUC by group on the TEST split (Phase 7b).

Groups evaluated:
- Overall (for reproduction check)
- Four age bins (<30, 30-44, 45-59, 60+)
- Segment x Gender (where N >= 100, excluding gender XNA)
- Segment x Binary Age (where N >= 100)

Flags groups with < 50 defaulters as low precision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config, set_seed


def compute_group_auc_bootstrap(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_bootstraps: int = 200,
    seed: int = 42,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Compute within-group AUC and 95% bootstrap CI for Model A, Model B, and paired (B - A)."""
    n = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos == 0 or n_neg == 0:
        return {
            "auc_a": None, "auc_a_ci": [None, None],
            "auc_b": None, "auc_b_ci": [None, None],
            "diff": None, "diff_ci": [None, None],
            "skipped_resamples": 0,
        }

    auc_a = float(roc_auc_score(y_true, score_a))
    auc_b = float(roc_auc_score(y_true, score_b))
    diff = auc_b - auc_a

    rng = np.random.RandomState(seed)
    boot_a, boot_b, boot_diff = [], [], []
    skipped = 0

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        sample_y = y_true[idx]
        if len(np.unique(sample_y)) < 2:
            skipped += 1
            continue
        sa = roc_auc_score(sample_y, score_a[idx])
        sb = roc_auc_score(sample_y, score_b[idx])
        boot_a.append(sa)
        boot_b.append(sb)
        boot_diff.append(sb - sa)

    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)

    res = {
        "auc_a": round(auc_a, 6),
        "auc_a_ci": [
            round(float(np.percentile(boot_a, lower_pct)), 6) if boot_a else None,
            round(float(np.percentile(boot_a, upper_pct)), 6) if boot_a else None,
        ],
        "auc_b": round(auc_b, 6),
        "auc_b_ci": [
            round(float(np.percentile(boot_b, lower_pct)), 6) if boot_b else None,
            round(float(np.percentile(boot_b, upper_pct)), 6) if boot_b else None,
        ],
        "diff": round(diff, 6),
        "diff_ci": [
            round(float(np.percentile(boot_diff, lower_pct)), 6) if boot_diff else None,
            round(float(np.percentile(boot_diff, upper_pct)), 6) if boot_diff else None,
        ],
        "skipped_resamples": skipped,
    }
    return res


def run_auc_by_group() -> None:
    config = load_config()
    seed = set_seed(config=config)
    bootstrap_n = int(config.get("bootstrap_n", 200))

    metrics_dir = _PROJECT_ROOT / "reports" / "metrics"
    tables_dir = _PROJECT_ROOT / "reports" / "tables"
    preds_dir = _PROJECT_ROOT / "reports" / "predictions"
    processed_dir = _PROJECT_ROOT / config["paths"]["processed"]

    # Load test predictions
    test_a = pd.read_parquet(preds_dir / "model_a_test.parquet")
    test_b = pd.read_parquet(preds_dir / "model_b_test.parquet")
    eval_attrs = pd.read_parquet(processed_dir / "eval_attrs.parquet")

    # Merge on SK_ID_CURR
    df = test_a[["SK_ID_CURR", "TARGET", "segment", "score"]].rename(columns={"score": "score_a"})
    df = df.merge(test_b[["SK_ID_CURR", "score"]].rename(columns={"score": "score_b"}), on="SK_ID_CURR", how="inner")
    df = df.merge(eval_attrs[["SK_ID_CURR", "gender", "gender_xna", "age_years", "age_bin"]], on="SK_ID_CURR", how="left")

    train_median_age = 43.12
    df["binary_age"] = np.where(df["age_years"] < train_median_age, "<median", ">=median")

    # Verify overall reproduction vs model_a.json and model_b.json
    saved_a = json.loads((metrics_dir / "model_a.json").read_text(encoding="utf-8"))
    saved_b = json.loads((metrics_dir / "model_b.json").read_text(encoding="utf-8"))
    saved_auc_a = float(saved_a["test_results"]["overall"]["auc"])
    saved_auc_b = float(saved_b["test_results"]["overall"]["auc"])

    overall_metrics = compute_group_auc_bootstrap(
        df["TARGET"].values, df["score_a"].values, df["score_b"].values,
        n_bootstraps=bootstrap_n, seed=seed
    )
    diff_check_a = abs(overall_metrics["auc_a"] - saved_auc_a)
    diff_check_b = abs(overall_metrics["auc_b"] - saved_auc_b)
    print(f"[auc_by_group] Overall AUC Model A: {overall_metrics['auc_a']} vs saved {saved_auc_a} (diff={diff_check_a:.2e})")
    print(f"[auc_by_group] Overall AUC Model B: {overall_metrics['auc_b']} vs saved {saved_auc_b} (diff={diff_check_b:.2e})")

    slices: List[Tuple[str, str, np.ndarray]] = []

    # 1. Overall
    slices.append(("overall", "all", np.ones(len(df), dtype=bool)))

    # 2. Four age bins
    for ab in ["<30", "30-44", "45-59", "60+"]:
        slices.append(("four_age_bins", ab, (df["age_bin"] == ab).values))

    # 3. Segment x Gender (excluding gender_xna)
    valid_gender_mask = (df["gender_xna"] == False).values
    for seg in ["full_file", "thin_file", "no_file"]:
        for g in ["F", "M"]:
            m = (df["segment"] == seg).values & (df["gender"] == g).values & valid_gender_mask
            if np.sum(m) >= 100:
                slices.append(("segment_x_gender", f"{seg}_x_{g}", m))

    # 4. Segment x Binary Age
    for seg in ["full_file", "thin_file", "no_file"]:
        for ba in ["<median", ">=median"]:
            m = (df["segment"] == seg).values & (df["binary_age"] == ba).values
            if np.sum(m) >= 100:
                slices.append(("segment_x_binary_age", f"{seg}_x_{ba}", m))

    results_dict: Dict[str, Any] = {
        "reproduction_check": {
            "model_a_saved_auc": saved_auc_a,
            "model_a_evaluated_auc": overall_metrics["auc_a"],
            "diff_a": diff_check_a,
            "model_b_saved_auc": saved_auc_b,
            "model_b_evaluated_auc": overall_metrics["auc_b"],
            "diff_b": diff_check_b,
            "passed": bool(diff_check_a <= 1e-6 and diff_check_b <= 1e-6),
        },
        "groups": {},
    }

    table_rows: List[Dict[str, Any]] = []

    for slice_cat, grp_name, mask in slices:
        sub_df = df[mask]
        n_tot = len(sub_df)
        n_pos = int(sub_df["TARGET"].sum())
        n_neg = n_tot - n_pos
        def_rate = round(n_pos / n_tot, 6) if n_tot > 0 else 0.0
        low_precision = bool(n_pos < 50)

        m_res = compute_group_auc_bootstrap(
            sub_df["TARGET"].values, sub_df["score_a"].values, sub_df["score_b"].values,
            n_bootstraps=bootstrap_n, seed=seed
        )

        grp_record = {
            "slice_category": slice_cat,
            "group": grp_name,
            "n": n_tot,
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "default_rate": def_rate,
            "low_precision": low_precision,
            "auc_a": m_res["auc_a"],
            "auc_a_ci": m_res["auc_a_ci"],
            "auc_b": m_res["auc_b"],
            "auc_b_ci": m_res["auc_b_ci"],
            "diff_b_minus_a": m_res["diff"],
            "diff_ci": m_res["diff_ci"],
            "skipped_resamples": m_res["skipped_resamples"],
        }
        results_dict["groups"][f"{slice_cat}:{grp_name}"] = grp_record

        table_rows.append({
            "slice_category": slice_cat,
            "group": grp_name,
            "n": n_tot,
            "n_positives": n_pos,
            "default_rate": def_rate,
            "low_precision": low_precision,
            "auc_a": m_res["auc_a"],
            "auc_a_ci_lower": m_res["auc_a_ci"][0],
            "auc_a_ci_upper": m_res["auc_a_ci"][1],
            "auc_b": m_res["auc_b"],
            "auc_b_ci_lower": m_res["auc_b_ci"][0],
            "auc_b_ci_upper": m_res["auc_b_ci"][1],
            "diff_b_minus_a": m_res["diff"],
            "diff_ci_lower": m_res["diff_ci"][0],
            "diff_ci_upper": m_res["diff_ci"][1],
        })

    # Save JSON and CSV
    out_json = metrics_dir / "extras_auc_by_group.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"[auc_by_group] Saved {out_json}")

    out_csv = tables_dir / "extras_auc_by_group.csv"
    pd.DataFrame(table_rows).to_csv(out_csv, index=False)
    print(f"[auc_by_group] Saved {out_csv}")


if __name__ == "__main__":
    run_auc_by_group()
