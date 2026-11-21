"""Test suite for Phase 7 extra analyses.
"""

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_extras_files_exist():
    expected_files = [
        "reports/metrics/extras_flips.json",
        "reports/tables/extras_flips.csv",
        "reports/figures/extras_flips.png",
        "reports/metrics/extras_ablation.json",
        "reports/tables/extras_ablation.csv",
        "reports/figures/extras_ablation.png",
        "reports/extras_notes.md",
        "reports/metrics/extras_calibration.json",
        "reports/tables/extras_calibration_bins.csv",
        "reports/figures/extras_calibration.png",
        "reports/metrics/extras_stability.json",
        "reports/tables/extras_stability.csv",
    ]
    for rel_path in expected_files:
        p = _PROJECT_ROOT / rel_path
        assert p.exists(), f"Expected extra analysis file does not exist: {rel_path}"
        assert p.stat().st_size > 0, f"Extra analysis file is empty: {rel_path}"


def test_no_existing_result_file_changed():
    """Verify git status shows no changes to previously tracked files under reports/."""
    cmd = ["git", "status", "--porcelain", "reports/"]
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), capture_output=True, text=True, check=True)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    
    modified_tracked = []
    for line in lines:
        status, path = line[:2], line[3:]
        # Only untracked new files (??) or files starting with extras_ are allowed
        if not path.startswith("reports/metrics/extras_") and \
           not path.startswith("reports/tables/extras_") and \
           not path.startswith("reports/figures/extras_") and \
           not path.startswith("reports/predictions/extras/") and \
           not path == "reports/extras_notes.md":
            modified_tracked.append(line)

    assert len(modified_tracked) == 0, f"Tracked reports files were modified: {modified_tracked}"


def test_refit_sanity_guard():
    metrics_path = _PROJECT_ROOT / "reports" / "metrics" / "extras_ablation.json"
    assert metrics_path.exists(), "extras_ablation.json not found"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    
    sanity = data.get("sanity_guard", {})
    assert "model_a_diff" in sanity
    assert "model_b_diff" in sanity
    assert sanity["model_a_diff"] <= 1e-9, f"Model A refit diff {sanity['model_a_diff']} > 1e-9"
    assert sanity["model_b_diff"] <= 1e-9, f"Model B refit diff {sanity['model_b_diff']} > 1e-9"
    assert sanity["passed"] is True


def test_flip_counts_sum():
    flips_json = _PROJECT_ROOT / "reports" / "metrics" / "extras_flips.json"
    assert flips_json.exists(), "extras_flips.json not found"
    data = json.loads(flips_json.read_text(encoding="utf-8"))
    
    overall = data["slices"]["overall"]
    counts = overall["counts"]
    cell_keys = [
        "rescued", "let_in", "newly_declined_good", "newly_caught",
        "both_declined_default", "both_declined_good",
        "both_approved_good", "both_approved_default"
    ]
    total = sum(counts[k] for k in cell_keys)
    assert total == 61503, f"Overall flip cells sum to {total}, expected 61,503"
    assert overall["n"] == 61503


def test_calibration_bins_cover_test_rows():
    bins_csv = _PROJECT_ROOT / "reports" / "tables" / "extras_calibration_bins.csv"
    assert bins_csv.exists(), "extras_calibration_bins.csv not found"
    df = pd.read_csv(bins_csv)
    
    # Check overall Model B bins
    sub = df[(df["model"] == "model_b") & (df["group_type"] == "overall")]
    assert len(sub) == 10, f"Expected 10 bins, got {len(sub)}"
    total_count = sub["count"].sum()
    assert total_count == 61503, f"Calibration bins cover {total_count} rows, expected 61,503"


def test_stability_splits_distinct():
    preds_dir = _PROJECT_ROOT / "reports" / "predictions" / "extras"
    main_train = pd.read_csv(_PROJECT_ROOT / "data" / "splits" / "train_ids.csv")
    main_train_ids = set(main_train["SK_ID_CURR"])

    split_train_sets = []
    for s in [1, 2, 3, 4, 5]:
        split_file = preds_dir / f"split_seed_{s}_ids.csv"
        assert split_file.exists(), f"Split file {split_file} missing"
        df = pd.read_csv(split_file)
        assert len(df) == 307511, f"Expected 307,511 rows, got {len(df)}"
        train_ids = set(df[df["split"] == "train"]["SK_ID_CURR"])
        
        # Distinct from main split
        assert train_ids != main_train_ids, f"Split seed {s} identical to main split"
        split_train_sets.append(train_ids)

    # Distinct from each other
    for i in range(len(split_train_sets)):
        for j in range(i + 1, len(split_train_sets)):
            assert split_train_sets[i] != split_train_sets[j], f"Split {i+1} identical to split {j+1}"
