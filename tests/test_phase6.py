"""Phase 6 Verification Tests for TrustScore."""

import json
from pathlib import Path
import subprocess

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_reproducibility_check_exists_and_passes():
    repro_path = _PROJECT_ROOT / "reports" / "metrics" / "reproducibility_check.json"
    assert repro_path.exists(), f"Missing {repro_path}"
    with open(repro_path, "r", encoding="utf-8") as f:
        repro = json.load(f)
    max_diff = repro["metadata"]["overall_max_numeric_difference"]
    assert max_diff <= 1e-9, f"Max numeric difference {max_diff} exceeds 1e-9 tolerance"
    assert repro["metadata"]["reproducibility_status"] == "PASS"


def test_results_summary_headings():
    summary_path = _PROJECT_ROOT / "reports" / "results_summary.md"
    assert summary_path.exists(), f"Missing {summary_path}"
    content = summary_path.read_text(encoding="utf-8")
    required_headings = [
        "1. Data Overview",
        "2. Model Performance",
        "3. Fairness Audit",
        "4. Policy Robustness",
        "5. Proxy Leakage",
        "6. SHAP Explanations",
        "7. Fairness Mitigation",
    ]
    for heading in required_headings:
        assert heading in content, f"Heading '{heading}' not found in {summary_path}"


def test_results_note_todo_lines():
    note_path = _PROJECT_ROOT / "reports" / "results_note.md"
    assert note_path.exists(), f"Missing {note_path}"
    content = note_path.read_text(encoding="utf-8")
    todo_lines = [line.strip() for line in content.splitlines() if line.strip() == "TODO (my interpretation):"]
    assert len(todo_lines) == 3, f"Expected exactly 3 'TODO (my interpretation):' lines, found {len(todo_lines)}"


def test_publication_figures_exist():
    fig_dir = _PROJECT_ROOT / "reports" / "figures"
    expected_figures = [
        "auc_by_segment.png",
        "fairness_gaps.png",
        "mitigation_matched.png",
    ]
    for fig_name in expected_figures:
        fig_path = fig_dir / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_path}"
        assert fig_path.stat().st_size > 1000, f"Figure {fig_name} appears empty"


def test_readme_mentions_run_all():
    readme_path = _PROJECT_ROOT / "README.md"
    assert readme_path.exists(), f"Missing {readme_path}"
    content = readme_path.read_text(encoding="utf-8")
    assert "python -m src.run_all" in content, "README.md must mention 'python -m src.run_all'"


def test_no_data_or_model_files_tracked():
    res = subprocess.run(["git", "ls-files"], cwd=str(_PROJECT_ROOT), capture_output=True, text=True, check=True)
    tracked_files = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    disallowed = []
    for f in tracked_files:
        if f.endswith(".parquet"):
            disallowed.append(f)
        elif f.endswith(".csv"):
            disallowed.append(f)
        elif f.startswith("models/") and not f.endswith(".gitkeep"):
            disallowed.append(f)
        elif f.startswith("data/") and not f.endswith(".gitkeep"):
            disallowed.append(f)
    assert not disallowed, f"Disallowed files tracked by git: {disallowed}"
