"""One-command full pipeline rerun and reproducibility verification for TrustScore.

Reruns Phases 1 through 5b from data/raw/home_credit/*.csv:
- Phase 1: build_base, build_eval_attrs, split, audit
- Phase 2: Model A feature matrix, training, evaluation
- Phase 3: behavioral features, Model B feature matrix, training, evaluation
- Phase 4: fairness audit (15% decline policy, demographic gaps), proxy leakage test (S1, S2, S3)
- Phase 5: base rates, SHAP explanations (unless --skip-shap is passed), ThresholdOptimizer mitigation
- Phase 5b: matched-decline-rate control experiments

Backs up existing reports to scratch_compare/ before running, then verifies that every numeric
value in reports/metrics and reports/tables matches the backup within 1e-9 tolerance.
Saves comparison results to reports/metrics/reproducibility_check.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config
import src.data.build_base as bb
import src.data.build_eval_attrs as bea
import src.data.split as sp
import src.data.audit as au
import src.features.model_a_features as maf
import src.models.train_model_a as tma
import src.features.behavioral as bh
import src.features.model_b_features as mbf
import src.models.train_model_b as tmb
import src.fairness.audit as fa
import src.fairness.proxy as fp
import src.fairness.base_rates as fbr
import src.explainability.shap_analysis as sa
import src.fairness.mitigation as fm
import src.fairness.mitigation_control as fmc


def compare_json(obj1: Any, obj2: Any) -> float:
    """Recursively find maximum absolute difference between numeric values in two JSON structures."""
    max_d = 0.0
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        for k in obj1:
            if k in obj2:
                max_d = max(max_d, compare_json(obj1[k], obj2[k]))
    elif isinstance(obj1, list) and isinstance(obj2, list):
        for v1, v2 in zip(obj1, obj2):
            max_d = max(max_d, compare_json(v1, v2))
    elif isinstance(obj1, (int, float, np.number)) and isinstance(obj2, (int, float, np.number)):
        if not (np.isnan(obj1) and np.isnan(obj2)):
            max_d = max(max_d, float(abs(obj1 - obj2)))
    return max_d


def compare_csv(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Find maximum absolute difference between numeric columns in two CSV DataFrames."""
    max_d = 0.0
    numeric_cols = [c for c in df1.columns if c in df2.columns and pd.api.types.is_numeric_dtype(df1[c])]
    for col in numeric_cols:
        s1 = df1[col].values
        s2 = df2[col].values
        mask = ~(np.isnan(s1) & np.isnan(s2))
        if np.any(mask):
            diff = float(np.max(np.abs(s1[mask] - s2[mask])))
            max_d = max(max_d, diff)
    return max_d


def run_pipeline(skip_shap: bool = False) -> Dict[str, Any]:
    config = load_config()
    total_start = time.time()
    step_runtimes: Dict[str, float] = {}

    reports_dir = _PROJECT_ROOT / "reports"
    backup_dir = _PROJECT_ROOT / "scratch_compare" / "reports_backup"

    # 1. Back up reports/
    print(f"[run_all] Backing up {reports_dir} to {backup_dir}...")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(reports_dir, backup_dir)

    try:
        # Phase 1: Data Build & Audit
        t0 = time.time()
        print("\n--- Running Phase 1: Data Build & Audit ---")
        bb.build_applicant_base(config)
        bea.build_eval_attrs(config)
        sp.create_stratified_split(config)
        au.audit_phase1(config)
        step_runtimes["phase1_data"] = time.time() - t0
        print(f"[Phase 1] Done in {step_runtimes['phase1_data']:.2f}s")

        # Phase 2: Model A (Bureau Baseline)
        t0 = time.time()
        print("\n--- Running Phase 2: Model A ---")
        maf.prepare_model_a_datasets(config)
        tma.train_model_a(config)
        step_runtimes["phase2_model_a"] = time.time() - t0
        print(f"[Phase 2] Done in {step_runtimes['phase2_model_a']:.2f}s")

        # Phase 3: Behavioral & Model B
        t0 = time.time()
        print("\n--- Running Phase 3: Behavioral Features & Model B ---")
        bh.build_and_join_behavioral_features(config)
        mbf.prepare_model_b_datasets(config)
        tmb.train_model_b(config)
        step_runtimes["phase3_model_b"] = time.time() - t0
        print(f"[Phase 3] Done in {step_runtimes['phase3_model_b']:.2f}s")

        # Phase 4: Fairness Audit & Proxy Tests
        t0 = time.time()
        print("\n--- Running Phase 4: Fairness Audit & Proxy Tests ---")
        fa.run_fairness_audit(config)
        fp.run_proxy_leakage_tests(config)
        step_runtimes["phase4_fairness_proxy"] = time.time() - t0
        print(f"[Phase 4] Done in {step_runtimes['phase4_fairness_proxy']:.2f}s")

        # Phase 5: Base Rates, SHAP & Mitigation
        t0 = time.time()
        print("\n--- Running Phase 5: Base Rates, SHAP & Mitigation ---")
        fbr.compute_base_rates(config)
        if not skip_shap:
            sa.run_shap_analysis(config)
        else:
            print("[Phase 5] Skipping SHAP analysis as requested (--skip-shap)")
        fm.run_mitigation_experiments(config)
        step_runtimes["phase5_shap_mitigation"] = time.time() - t0
        print(f"[Phase 5] Done in {step_runtimes['phase5_shap_mitigation']:.2f}s")

        # Phase 5b: Matched-Rate Control Experiments
        t0 = time.time()
        print("\n--- Running Phase 5b: Matched Control Experiments ---")
        fmc.run_matched_control_experiments(config)
        step_runtimes["phase5b_matched_control"] = time.time() - t0
        print(f"[Phase 5b] Done in {step_runtimes['phase5b_matched_control']:.2f}s")

        total_runtime = time.time() - total_start
        print(f"\n[run_all] All pipeline steps completed in {total_runtime:.2f}s total.")

        # 2. Compare regenerated outputs against backup
        print("\n--- Checking Reproducibility Against Backup ---")
        comparison_results: Dict[str, float] = {}
        discrepancies: List[str] = []

        # Check all JSON and CSV in reports/metrics and reports/tables
        for subfolder in ["metrics", "tables"]:
            curr_sub = reports_dir / subfolder
            back_sub = backup_dir / subfolder
            for curr_file in curr_sub.glob("*.*"):
                if curr_file.name == "reproducibility_check.json":
                    continue
                if curr_file.suffix not in [".json", ".csv"]:
                    continue

                back_file = back_sub / curr_file.name
                rel_path = f"reports/{subfolder}/{curr_file.name}"

                if not back_file.exists():
                    discrepancies.append(f"Missing in backup: {rel_path}")
                    continue

                if curr_file.suffix == ".json":
                    with open(curr_file, "r") as f1, open(back_file, "r") as f2:
                        d1 = json.load(f1)
                        d2 = json.load(f2)
                    max_diff = compare_json(d1, d2)
                elif curr_file.suffix == ".csv":
                    df1 = pd.read_csv(curr_file)
                    df2 = pd.read_csv(back_file)
                    max_diff = compare_csv(df1, df2)

                comparison_results[rel_path] = max_diff
                print(f"  {rel_path}: max numeric diff = {max_diff:.10e}")
                if max_diff > 1e-9:
                    discrepancies.append(f"{rel_path}: diff = {max_diff:.10e} > 1e-9")

        overall_max_diff = max(comparison_results.values()) if comparison_results else 0.0

        if discrepancies:
            print("\n[ERROR] Reproducibility check failed! Restoring backup...")
            shutil.rmtree(reports_dir)
            shutil.copytree(backup_dir, reports_dir)
            raise ValueError(f"Reproducibility failure in files: {discrepancies}")

        # 3. Save reproducibility_check.json
        import platform
        import sklearn
        import xgboost
        import fairlearn
        import shap
        import matplotlib

        repro_check = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_runtime_seconds": total_runtime,
                "step_runtimes_seconds": step_runtimes,
                "overall_max_numeric_difference": overall_max_diff,
                "tolerance": 1e-9,
                "reproducibility_status": "PASS" if overall_max_diff <= 1e-9 else "FAIL",
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "packages": {
                        "numpy": np.__version__,
                        "pandas": pd.__version__,
                        "scikit-learn": sklearn.__version__,
                        "xgboost": xgboost.__version__,
                        "fairlearn": fairlearn.__version__,
                        "shap": shap.__version__,
                        "matplotlib": matplotlib.__version__,
                    },
                },
            },
            "per_file_max_absolute_difference": comparison_results,
        }

        repro_path = reports_dir / "metrics" / "reproducibility_check.json"
        with open(repro_path, "w") as f:
            json.dump(repro_check, f, indent=2)
        print(f"\n[run_all] Successfully wrote {repro_path}")

        return repro_check

    except Exception as e:
        print(f"\n[run_all] Encountered error: {e}")
        print("[run_all] Restoring backup...")
        shutil.rmtree(reports_dir)
        shutil.copytree(backup_dir, reports_dir)
        raise


def main():
    parser = argparse.ArgumentParser(description="TrustScore full pipeline runner and reproducibility check")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis for faster execution")
    args = parser.parse_args()
    run_pipeline(skip_shap=args.skip_shap)


if __name__ == "__main__":
    main()

