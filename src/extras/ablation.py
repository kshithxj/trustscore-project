"""Feature-source ablation analysis for TrustScore (Phase 7 Part B).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config, set_seed
from src.extras.common import (
    load_prepared_data,
    train_xgb,
    compute_auc_ci,
    compute_paired_auc_diff_ci,
    load_model_params,
)
from src.fairness.policy import compute_threshold_from_validation
from src.features.behavioral import BEHAVIORAL_COLS_BY_TABLE, ALL_BEHAVIORAL_COLS


def compute_gaps(y_true: np.ndarray, decline: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    """Compute DP difference and EO difference."""
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return {"dp_diff": np.nan, "eo_diff": np.nan}

    app_rates = []
    tprs = []
    fprs = []

    for g in unique_groups:
        mask = (groups == g)
        n_g = int(mask.sum())
        if n_g == 0:
            continue
        app_rates.append(float(np.mean(1 - decline[mask])))
        
        pos_mask = mask & (y_true == 1)
        neg_mask = mask & (y_true == 0)
        tpr = float(np.mean(decline[pos_mask])) if pos_mask.sum() > 0 else np.nan
        fpr = float(np.mean(decline[neg_mask])) if neg_mask.sum() > 0 else np.nan
        if not np.isnan(tpr):
            tprs.append(tpr)
        if not np.isnan(fpr):
            fprs.append(fpr)

    dp_diff = max(app_rates) - min(app_rates) if len(app_rates) >= 2 else np.nan
    tpr_gap = max(tprs) - min(tprs) if len(tprs) >= 2 else 0.0
    fpr_gap = max(fprs) - min(fprs) if len(fprs) >= 2 else 0.0
    eo_diff = max(tpr_gap, fpr_gap)

    return {
        "dp_diff": round(float(dp_diff), 6) if not np.isnan(dp_diff) else np.nan,
        "eo_diff": round(float(eo_diff), 6) if not np.isnan(eo_diff) else np.nan,
    }


def bootstrap_fairness_gaps(
    y_true: np.ndarray,
    decline: np.ndarray,
    groups: np.ndarray,
    n_bootstraps: int = 200,
    seed: int = 42,
    alpha: float = 0.05
) -> Dict[str, Tuple[float, float]]:
    """Group-stratified bootstrap for DP and EO differences."""
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    group_indices = {g: np.where(groups == g)[0] for g in unique_groups}

    boot_dp = []
    boot_eo = []

    for _ in range(n_bootstraps):
        sampled_idx = []
        for g, idxs in group_indices.items():
            sampled_idx.append(rng.choice(idxs, size=len(idxs), replace=True))
        all_idx = np.concatenate(sampled_idx)

        gaps = compute_gaps(y_true[all_idx], decline[all_idx], groups[all_idx])
        if not np.isnan(gaps["dp_diff"]):
            boot_dp.append(gaps["dp_diff"])
        if not np.isnan(gaps["eo_diff"]):
            boot_eo.append(gaps["eo_diff"])

    res = {}
    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)

    if len(boot_dp) > 0:
        res["dp_diff_ci"] = (
            round(float(np.percentile(boot_dp, lower_pct)), 6),
            round(float(np.percentile(boot_dp, upper_pct)), 6),
        )
    else:
        res["dp_diff_ci"] = (float("nan"), float("nan"))

    if len(boot_eo) > 0:
        res["eo_diff_ci"] = (
            round(float(np.percentile(boot_eo, lower_pct)), 6),
            round(float(np.percentile(boot_eo, upper_pct)), 6),
        )
    else:
        res["eo_diff_ci"] = (float("nan"), float("nan"))

    return res


def run_ablation() -> None:
    config = load_config()
    seed = set_seed(config=config)
    bootstrap_n = int(config.get("bootstrap_n", 200))
    decline_rate = float(config.get("decline_rate", 0.15))

    out_models_dir = _PROJECT_ROOT / "models" / "extras"
    out_preds_dir = _PROJECT_ROOT / "reports" / "predictions" / "extras"
    out_metrics_dir = _PROJECT_ROOT / "reports" / "metrics"
    out_tables_dir = _PROJECT_ROOT / "reports" / "tables"
    out_figures_dir = _PROJECT_ROOT / "reports" / "figures"

    out_models_dir.mkdir(parents=True, exist_ok=True)
    out_preds_dir.mkdir(parents=True, exist_ok=True)
    out_metrics_dir.mkdir(parents=True, exist_ok=True)
    out_tables_dir.mkdir(parents=True, exist_ok=True)
    out_figures_dir.mkdir(parents=True, exist_ok=True)

    print("[ablation] Loading datasets...")
    (
        X_train_a, X_valid_a, X_test_a,
        X_train_b, X_valid_b, X_test_b,
        y_train, y_valid, y_test,
        meta_train, meta_valid, meta_test,
        feat_a, feat_b
    ) = load_prepared_data(config=config)

    # Load eval_attrs for protected attributes
    eval_attrs = pd.read_parquet(_PROJECT_ROOT / config["paths"]["processed"] / "eval_attrs.parquet")
    test_eval = meta_test.merge(eval_attrs, on="SK_ID_CURR", how="left")
    
    # Train median age from split
    train_eval = meta_train.merge(eval_attrs, on="SK_ID_CURR", how="left")
    median_age = float(train_eval["age_years"].median())
    test_eval["binary_age"] = np.where(test_eval["age_years"] < median_age, "below_median", "above_median")
    test_eval_gender = test_eval[test_eval["gender_xna"] == False].copy()

    # SANITY GUARD: Refit Model A and Model B
    print("[ablation] Sanity Guard: refitting Model A and Model B...")
    saved_pred_a = pd.read_parquet(_PROJECT_ROOT / "reports" / "predictions" / "model_a_test.parquet")
    saved_pred_b = pd.read_parquet(_PROJECT_ROOT / "reports" / "predictions" / "model_b_test.parquet")
    saved_auc_a = float(roc_auc_score(y_test, saved_pred_a["score"]))
    saved_auc_b = float(roc_auc_score(y_test, saved_pred_b["score"]))

    # Refit Model A
    clf_a = train_xgb(X_train_a, y_train, X_valid_a, y_valid, seed=seed)
    val_preds_a = clf_a.predict_proba(X_valid_a)[:, 1]
    test_preds_a = clf_a.predict_proba(X_test_a)[:, 1]
    refit_auc_a = float(roc_auc_score(y_test, test_preds_a))
    diff_a = abs(refit_auc_a - saved_auc_a)

    # Refit Model B
    clf_b = train_xgb(X_train_b, y_train, X_valid_b, y_valid, seed=seed)
    val_preds_b = clf_b.predict_proba(X_valid_b)[:, 1]
    test_preds_b = clf_b.predict_proba(X_test_b)[:, 1]
    refit_auc_b = float(roc_auc_score(y_test, test_preds_b))
    diff_b = abs(refit_auc_b - saved_auc_b)

    print(f"[ablation] Model A saved AUC={saved_auc_a:.6f}, refit={refit_auc_a:.6f}, diff={diff_a:.2e}")
    print(f"[ablation] Model B saved AUC={saved_auc_b:.6f}, refit={refit_auc_b:.6f}, diff={diff_b:.2e}")

    if diff_a > 1e-9 or diff_b > 1e-9:
        print(f"[WARNING] Refit difference: Model A diff={diff_a}, Model B diff={diff_b}")
    else:
        print("[ablation] Sanity check PASSED: differences <= 1e-9.")

    # Save refit models and predictions
    clf_a.save_model(str(out_models_dir / "model_a_refit.json"))
    clf_b.save_model(str(out_models_dir / "model_b_refit.json"))
    pd.DataFrame({"SK_ID_CURR": meta_test["SK_ID_CURR"], "pred": test_preds_a}).to_parquet(out_preds_dir / "model_a_refit_test.parquet", index=False)
    pd.DataFrame({"SK_ID_CURR": meta_test["SK_ID_CURR"], "pred": test_preds_b}).to_parquet(out_preds_dir / "model_b_refit_test.parquet", index=False)

    # Define ablation setups
    sources = ["previous_application", "installments_payments", "POS_CASH_balance", "credit_card_balance"]
    
    models_to_run = {}
    
    # Model A baseline
    models_to_run["model_a"] = {
        "type": "baseline",
        "features": feat_a,
        "ref": "model_a",
        "compare_to": None,
        "val_preds": val_preds_a,
        "test_preds": test_preds_a,
        "clf": clf_a,
    }
    # Model B full
    models_to_run["model_b"] = {
        "type": "full",
        "features": feat_b,
        "ref": "model_b",
        "compare_to": "model_a",
        "val_preds": val_preds_b,
        "test_preds": test_preds_b,
        "clf": clf_b,
    }

    # B minus source
    for src in sources:
        remove_cols = set(BEHAVIORAL_COLS_BY_TABLE[src])
        feats = [c for c in feat_b if c not in remove_cols]
        models_to_run[f"model_b_minus_{src}"] = {
            "type": "b_minus",
            "source": src,
            "features": feats,
            "compare_to": "model_b",
        }

    # A plus source
    for src in sources:
        add_cols = BEHAVIORAL_COLS_BY_TABLE[src]
        feats = list(feat_a) + list(add_cols)
        models_to_run[f"model_a_plus_{src}"] = {
            "type": "a_plus",
            "source": src,
            "features": feats,
            "compare_to": "model_a",
        }

    # Train and evaluate each ablation setup
    results = {}
    table_rows = []

    # Cache thin file mask
    thin_mask = (meta_test["segment"] == "thin_file").values
    y_test_thin = y_test.values[thin_mask]

    for model_name, info in models_to_run.items():
        print(f"\n[ablation] Processing {model_name} ({len(info['features'])} features)...")
        feats = info["features"]
        
        if "clf" in info:
            clf = info["clf"]
            val_preds = info["val_preds"]
            test_preds = info["test_preds"]
        else:
            t0 = time.time()
            clf = train_xgb(X_train_b[feats], y_train, X_valid_b[feats], y_valid, seed=seed)
            fit_time = round(time.time() - t0, 2)
            print(f"[ablation] Fitted {model_name} in {fit_time}s")
            val_preds = clf.predict_proba(X_valid_b[feats])[:, 1]
            test_preds = clf.predict_proba(X_test_b[feats])[:, 1]
            # Save predictions and model
            clf.save_model(str(out_models_dir / f"{model_name}.json"))
            pd.DataFrame({"SK_ID_CURR": meta_test["SK_ID_CURR"], "pred": test_preds}).to_parquet(
                out_preds_dir / f"{model_name}_test.parquet", index=False
            )
            info["clf"] = clf
            info["val_preds"] = val_preds
            info["test_preds"] = test_preds

        # Overall AUC & CI
        auc_val, ci_lower, ci_upper = compute_auc_ci(
            y_test.values, test_preds, n_bootstraps=bootstrap_n, seed=seed
        )

        # Thin-file AUC
        test_preds_thin = test_preds[thin_mask]
        thin_auc, thin_lower, thin_upper = compute_auc_ci(
            y_test_thin, test_preds_thin, n_bootstraps=bootstrap_n, seed=seed
        )

        # Paired delta vs compare_to
        comp_name = info["compare_to"]
        if comp_name is not None:
            comp_preds = models_to_run[comp_name]["test_preds"]
            diff_val, diff_lower, diff_upper = compute_paired_auc_diff_ci(
                y_test.values, test_preds, comp_preds, n_bootstraps=bootstrap_n, seed=seed
            )
        else:
            diff_val, diff_lower, diff_upper = 0.0, 0.0, 0.0

        # Decision threshold from validation (15% decline rate)
        thresh = compute_threshold_from_validation(val_preds, decline_rate=decline_rate)
        declines = (test_preds >= thresh).astype(int)

        # Fairness metrics: Gender (excluding XNA)
        gender_mask = (test_eval["gender_xna"] == False).values
        gender_gaps = compute_gaps(
            y_test.values[gender_mask],
            declines[gender_mask],
            test_eval["gender"].values[gender_mask]
        )
        gender_boot = bootstrap_fairness_gaps(
            y_test.values[gender_mask],
            declines[gender_mask],
            test_eval["gender"].values[gender_mask],
            n_bootstraps=bootstrap_n,
            seed=seed
        )

        # Fairness metrics: Binary Age
        age_gaps = compute_gaps(
            y_test.values,
            declines,
            test_eval["binary_age"].values
        )
        age_boot = bootstrap_fairness_gaps(
            y_test.values,
            declines,
            test_eval["binary_age"].values,
            n_bootstraps=bootstrap_n,
            seed=seed
        )

        model_res = {
            "model_name": model_name,
            "type": info["type"],
            "source": info.get("source", None),
            "n_features": len(feats),
            "compare_to": comp_name,
            "threshold": round(thresh, 6),
            "auc": auc_val,
            "auc_ci_lower": ci_lower,
            "auc_ci_upper": ci_upper,
            "thin_auc": thin_auc,
            "thin_auc_ci_lower": thin_lower,
            "thin_auc_ci_upper": thin_upper,
            "delta_auc_vs_ref": diff_val,
            "delta_auc_ci_lower": diff_lower,
            "delta_auc_ci_upper": diff_upper,
            "gender_dp_diff": gender_gaps["dp_diff"],
            "gender_dp_ci": gender_boot["dp_diff_ci"],
            "gender_eo_diff": gender_gaps["eo_diff"],
            "gender_eo_ci": gender_boot["eo_diff_ci"],
            "age_dp_diff": age_gaps["dp_diff"],
            "age_dp_ci": age_boot["dp_diff_ci"],
            "age_eo_diff": age_gaps["eo_diff"],
            "age_eo_ci": age_boot["eo_diff_ci"],
        }
        results[model_name] = model_res

    # Compute changes vs Model A for fairness and thin file lift
    base_a = results["model_a"]
    for m_name, m_res in results.items():
        m_res["change_gender_dp_vs_a"] = round(m_res["gender_dp_diff"] - base_a["gender_dp_diff"], 6)
        m_res["change_gender_eo_vs_a"] = round(m_res["gender_eo_diff"] - base_a["gender_eo_diff"], 6)
        m_res["change_age_dp_vs_a"] = round(m_res["age_dp_diff"] - base_a["age_dp_diff"], 6)
        m_res["change_age_eo_vs_a"] = round(m_res["age_eo_diff"] - base_a["age_eo_diff"], 6)
        m_res["thin_file_lift_vs_a"] = round(m_res["thin_auc"] - base_a["thin_auc"], 6)

        table_rows.append({
            "model": m_name,
            "type": m_res["type"],
            "source": m_res["source"] if m_res["source"] else "none",
            "n_features": m_res["n_features"],
            "test_auc": m_res["auc"],
            "auc_ci_lower": m_res["auc_ci_lower"],
            "auc_ci_upper": m_res["auc_ci_upper"],
            "thin_auc": m_res["thin_auc"],
            "thin_lift_vs_a": m_res["thin_file_lift_vs_a"],
            "compare_ref": m_res["compare_to"] if m_res["compare_to"] else "none",
            "delta_auc_vs_ref": m_res["delta_auc_vs_ref"],
            "gender_dp_diff": m_res["gender_dp_diff"],
            "change_gender_dp_vs_a": m_res["change_gender_dp_vs_a"],
            "gender_eo_diff": m_res["gender_eo_diff"],
            "change_gender_eo_vs_a": m_res["change_gender_eo_vs_a"],
            "age_dp_diff": m_res["age_dp_diff"],
            "change_age_dp_vs_a": m_res["change_age_dp_vs_a"],
            "age_eo_diff": m_res["age_eo_diff"],
            "change_age_eo_vs_a": m_res["change_age_eo_vs_a"],
        })

    # Save metrics JSON
    metrics_payload = {
        "sanity_guard": {
            "model_a_saved_auc": saved_auc_a,
            "model_a_refit_auc": refit_auc_a,
            "model_a_diff": diff_a,
            "model_b_saved_auc": saved_auc_b,
            "model_b_refit_auc": refit_auc_b,
            "model_b_diff": diff_b,
            "passed": bool(diff_a <= 1e-9 and diff_b <= 1e-9),
        },
        "models": results,
    }
    with open(out_metrics_dir / "extras_ablation.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[ablation] Saved {out_metrics_dir / 'extras_ablation.json'}")

    # Save table CSV
    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(out_tables_dir / "extras_ablation.csv", index=False)
    print(f"[ablation] Saved {out_tables_dir / 'extras_ablation.csv'}")

    # Generate Figure: lift on thin_file vs change in age EO gap
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    for row in table_rows:
        m = row["model"]
        x = row["thin_lift_vs_a"]
        y = row["change_age_eo_vs_a"]
        
        if "model_a_plus" in m:
            color = "#1f77b4"
            marker = "o"
        elif "model_b_minus" in m:
            color = "#ff7f0e"
            marker = "s"
        elif m == "model_a":
            color = "#7f7f7f"
            marker = "D"
        else:  # model_b
            color = "#2ca02c"
            marker = "^"
            
        ax.scatter(x, y, color=color, marker=marker, s=80, alpha=0.9, zorder=4)
        # Label offset
        ax.annotate(
            m.replace("model_", "").replace("_", " "),
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            alpha=0.85
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Thin-File AUC Lift vs Model A (AUC_thin - AUC_thin,A)", fontsize=11)
    ax.set_ylabel("Change in Binary Age Equalized Odds Gap vs Model A", fontsize=11)
    ax.set_title("TrustScore Ablation: Thin-File Lift vs Age Fairness Gap Change", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_figures_dir / "extras_ablation.png")
    plt.close(fig)
    print(f"[ablation] Saved {out_figures_dir / 'extras_ablation.png'}")

    # Generate reports/extras_notes.md (4 plain-English sentences with code-inserted numbers)
    # Find key stats
    a_plus_rows = [r for r in table_rows if r["type"] == "a_plus"]
    best_single_lift = max(a_plus_rows, key=lambda r: r["delta_auc_vs_ref"])
    worst_age_gap_source = max(a_plus_rows, key=lambda r: r["change_age_eo_vs_a"])
    b_minus_rows = [r for r in table_rows if r["type"] == "b_minus"]
    most_critical_source = min(b_minus_rows, key=lambda r: r["delta_auc_vs_ref"])
    
    notes_content = (
        f"Among individual lender-history sources added to Model A, {best_single_lift['source']} provides the largest predictive lift with a paired ΔAUC of +{best_single_lift['delta_auc_vs_ref']:.6f} over the bureau-only baseline.\n"
        f"Removing {most_critical_source['source']} from Model B causes the steepest performance drop of {most_critical_source['delta_auc_vs_ref']:.6f} AUC, confirming it as the most critical behavioral component.\n"
        f"On thin-file applicants, Model B delivers a total AUC lift of +{results['model_b']['thin_file_lift_vs_a']:.6f} compared to Model A, while the single addition of {best_single_lift['source']} achieves +{best_single_lift['thin_lift_vs_a']:.6f}.\n"
        f"Regarding demographic fairness, adding {worst_age_gap_source['source']} shifts the binary age equalized odds gap by {worst_age_gap_source['change_age_eo_vs_a']:+.6f}, indicating that behavioral sources modestly impact age disparity alongside accuracy gains.\n"
    )
    with open(_PROJECT_ROOT / "reports" / "extras_notes.md", "w", encoding="utf-8") as f:
        f.write(notes_content)
    print(f"[ablation] Saved {_PROJECT_ROOT / 'reports' / 'extras_notes.md'}")


if __name__ == "__main__":
    run_ablation()
