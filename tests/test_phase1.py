"""Phase 1 tests — data artefact integrity checks."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config

_CFG = load_config()
_PROCESSED = _PROJECT_ROOT / _CFG["paths"]["processed"]
_SPLITS = _PROJECT_ROOT / _CFG["paths"]["splits"]
_METRICS = _PROJECT_ROOT / _CFG["paths"]["metrics"]


# ── applicant_base.parquet ──────────────────────────────────────────────────


class TestApplicantBase:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.base = pd.read_parquet(_PROCESSED / "applicant_base.parquet")

    def test_not_empty(self):
        assert len(self.base) > 0

    def test_required_columns(self):
        required = {
            "SK_ID_CURR", "TARGET", "has_bureau",
            "has_lender_history", "segment",
        }
        assert required.issubset(set(self.base.columns))

    def test_no_duplicate_ids(self):
        assert self.base["SK_ID_CURR"].is_unique

    def test_days_employed_no_sentinel(self):
        if "DAYS_EMPLOYED" in self.base.columns:
            assert (self.base["DAYS_EMPLOYED"] == 365243).sum() == 0

    def test_segment_values(self):
        valid = {"full_file", "thin_file", "no_file"}
        actual = set(self.base["segment"].unique())
        assert actual.issubset(valid), f"Unexpected segments: {actual - valid}"

    def test_target_binary(self):
        assert set(self.base["TARGET"].unique()).issubset({0, 1})


# ── eval_attrs.parquet ──────────────────────────────────────────────────────


class TestEvalAttrs:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.base = pd.read_parquet(_PROCESSED / "applicant_base.parquet")
        self.eval_a = pd.read_parquet(_PROCESSED / "eval_attrs.parquet")

    def test_same_ids(self):
        base_ids = set(self.base["SK_ID_CURR"])
        eval_ids = set(self.eval_a["SK_ID_CURR"])
        assert base_ids == eval_ids

    def test_required_columns(self):
        required = {"SK_ID_CURR", "gender", "gender_xna", "age_years", "age_bin"}
        assert required.issubset(set(self.eval_a.columns))

    def test_age_positive(self):
        assert (self.eval_a["age_years"] > 0).all()

    def test_age_bins_valid(self):
        valid = {"<30", "30-44", "45-59", "60+"}
        actual = set(self.eval_a["age_bin"].dropna().unique())
        assert actual.issubset(valid)


# ── Splits ──────────────────────────────────────────────────────────────────


class TestSplits:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.base = pd.read_parquet(
            _PROCESSED / "applicant_base.parquet", columns=["SK_ID_CURR"]
        )
        self.splits = {}
        for name in ("train", "valid", "test"):
            self.splits[name] = pd.read_csv(_SPLITS / f"{name}_ids.csv")

    def test_files_exist(self):
        for name in ("train", "valid", "test"):
            assert (_SPLITS / f"{name}_ids.csv").exists()

    def test_splits_disjoint(self):
        ids = {n: set(df["SK_ID_CURR"]) for n, df in self.splits.items()}
        assert ids["train"].isdisjoint(ids["valid"])
        assert ids["train"].isdisjoint(ids["test"])
        assert ids["valid"].isdisjoint(ids["test"])

    def test_union_equals_full(self):
        full = set(self.base["SK_ID_CURR"])
        union = set()
        for df in self.splits.values():
            union |= set(df["SK_ID_CURR"])
        assert union == full

    def test_fractions_within_tolerance(self):
        total = len(self.base)
        cfg_split = _CFG["split"]
        for name, df in self.splits.items():
            expected = cfg_split[name]
            actual = len(df) / total
            assert abs(actual - expected) < 0.01, (
                f"{name}: expected ~{expected:.2f}, got {actual:.4f}"
            )


# ── Audit JSON ──────────────────────────────────────────────────────────────


class TestAuditJson:
    @pytest.fixture(autouse=True)
    def _load(self):
        path = _METRICS / "phase1_audit.json"
        assert path.exists(), "phase1_audit.json not found"
        with open(path, encoding="utf-8") as f:
            self.audit = json.load(f)

    def test_has_required_keys(self):
        required = {
            "n_applicants",
            "train_row_count",
            "target_mean",
            "duplicate_sk_id_curr",
            "stratification_used",
            "smallest_stratum",
            "segment_counts",
            "default_rate_by_segment",
            "default_rate_by_split",
            "gender_counts",
            "age_bin_counts",
            "missingness",
            "warnings",
        }
        assert required.issubset(set(self.audit.keys()))

    def test_stratification_valid(self):
        valid = {"target_x_segment", "segment_only"}
        assert self.audit["stratification_used"] in valid

    def test_segment_counts_overall(self):
        overall = self.audit["segment_counts"]["overall"]
        assert sum(overall.values()) == self.audit["n_applicants"]

    def test_default_rates_in_range(self):
        for seg, rate in self.audit["default_rate_by_segment"].items():
            assert 0 <= rate <= 1, f"{seg} default rate {rate} out of [0,1]"
