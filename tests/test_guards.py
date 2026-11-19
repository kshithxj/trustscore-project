"""Tests for the hard data-quality guards in build_base and audit."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.build_base import _validate_train


def _make_valid_df(n: int = 307_511) -> pd.DataFrame:
    """Create a minimal valid application_train-like DataFrame."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "SK_ID_CURR": np.arange(100_000, 100_000 + n),
        "TARGET": rng.choice([0, 1], size=n, p=[0.92, 0.08]),
    })


class TestValidateTrainGuards:
    """Each test verifies that _validate_train raises on exactly one violation."""

    def test_missing_target_column(self):
        df = _make_valid_df()
        df = df.drop(columns=["TARGET"])
        with pytest.raises(ValueError, match="Missing required column.*TARGET"):
            _validate_train(df)

    def test_missing_sk_id_curr_column(self):
        df = _make_valid_df()
        df = df.drop(columns=["SK_ID_CURR"])
        with pytest.raises(ValueError, match="Missing required column.*SK_ID_CURR"):
            _validate_train(df)

    def test_too_few_rows(self):
        df = _make_valid_df(n=1_000)
        with pytest.raises(ValueError, match="only 1,000 rows"):
            _validate_train(df)

    def test_duplicate_ids(self):
        df = _make_valid_df()
        df.loc[0, "SK_ID_CURR"] = df.loc[1, "SK_ID_CURR"]
        with pytest.raises(ValueError, match="duplicate SK_ID_CURR"):
            _validate_train(df)

    def test_target_mean_too_low(self):
        df = _make_valid_df()
        df["TARGET"] = 0  # mean = 0.0
        with pytest.raises(ValueError, match="TARGET mean is 0.000000"):
            _validate_train(df)

    def test_target_mean_too_high(self):
        df = _make_valid_df()
        df["TARGET"] = 1  # mean = 1.0
        with pytest.raises(ValueError, match="TARGET mean is 1.000000"):
            _validate_train(df)

    def test_valid_df_passes(self):
        df = _make_valid_df()
        # Should not raise
        _validate_train(df)
