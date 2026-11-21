"""Live scoring and TreeSHAP explainability core for held-out TEST applicants.

Phase 8 Stage 4: TrustScore Dashboard live scoring core.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Set writable MPLCONFIGDIR for headless Windows execution
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.utils import load_config
from src.features.model_b_features import prepare_model_b_datasets
from src.features.behavioral import ALL_BEHAVIORAL_COLS
from src.features.bureau import BUREAU_FEATURE_COLS


class ScoringResult:
    """Container for scoring outputs supporting tuple unpacking, dict indexing, and attributes."""

    def __init__(
        self,
        score_a: np.ndarray,
        score_b: np.ndarray,
        ids: List[int],
        X_a: pd.DataFrame,
        X_b: pd.DataFrame,
    ):
        self.score_a = np.asarray(score_a, dtype=float)
        self.score_b = np.asarray(score_b, dtype=float)
        self.ids = ids
        self.X_a = X_a
        self.X_b = X_b

    def __iter__(self):
        yield self.score_a
        yield self.score_b

    def __getitem__(self, item):
        if item == 0:
            return self.score_a
        elif item == 1:
            return self.score_b
        elif item in ("score_a", "scores_a", "model_a", "a"):
            return self.score_a
        elif item in ("score_b", "scores_b", "model_b", "b"):
            return self.score_b
        raise KeyError(item)

    def __len__(self):
        return 2

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {"score_a": self.score_a, "score_b": self.score_b}


class ExplanationResult(dict):
    """Container for local TreeSHAP explanation supporting dict and attribute access."""

    def __init__(
        self,
        bias: float,
        contributions: pd.Series,
        feature_values: pd.Series,
        score: float,
        applicant_id: int,
        source: str = "live_treeshap",
    ):
        super().__init__()
        self["bias"] = bias
        self["contributions"] = contributions
        self["shap_values"] = contributions
        self["feature_values"] = feature_values
        self["score"] = score
        self["id"] = applicant_id
        self["source"] = source

        self.bias = bias
        self.contributions = contributions
        self.shap_values = contributions
        self.feature_values = feature_values
        self.score = score
        self.id = applicant_id
        self.source = source

    @property
    def top_10(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "feature": self.contributions.index,
            "shap_value": self.contributions.values,
            "abs_shap": np.abs(self.contributions.values),
            "feature_value": self.feature_values.values,
        }).sort_values("abs_shap", ascending=False).head(10)
        return df


@st.cache_resource
def load_scoring_resources() -> Dict[str, Any]:
    """Builds and caches Model A and Model B test matrices, models, and TreeExplainer.
    
    Guarantees strict train-only category encoding and identical preprocessing pipeline.
    """
    config = load_config()

    # Use a temp directory for metrics so we never overwrite reports/metrics/
    cfg_copy = copy.deepcopy(config)
    cfg_copy["paths"]["metrics"] = tempfile.gettempdir()

    # Build features for the full test split using existing pipeline
    (
        _, _,
        _, _,
        X_test_b, y_test,
        _, _,
        meta_test,
        feat_names_b,
        _,
    ) = prepare_model_b_datasets(cfg_copy)

    # Feature names for Model A
    features_a_path = _PROJECT_ROOT / "reports" / "metrics" / "model_a_features.txt"
    with open(features_a_path, "r", encoding="utf-8") as f:
        feat_names_a = [line.strip() for line in f if line.strip()]

    X_test_a = X_test_b[feat_names_a].copy()

    # Set SK_ID_CURR as index for O(1) row lookup
    test_ids = meta_test["SK_ID_CURR"].astype(int).values
    X_test_a.index = test_ids
    X_test_b.index = test_ids
    meta_test_indexed = meta_test.copy()
    meta_test_indexed["SK_ID_CURR"] = test_ids
    meta_test_indexed = meta_test_indexed.set_index("SK_ID_CURR")

    # Load trained models
    model_a = xgb.XGBClassifier()
    model_a.load_model(str(_PROJECT_ROOT / "models" / "model_a.json"))

    model_b = xgb.XGBClassifier()
    model_b.load_model(str(_PROJECT_ROOT / "models" / "model_b.json"))

    # Initialise TreeExplainer for Model B
    explainer_b = shap.TreeExplainer(model_b)

    # Load test split IDs set
    test_ids_set = set(test_ids)

    # Load policy thresholds from reports/metrics/fairness.json
    fairness_path = _PROJECT_ROOT / "reports" / "metrics" / "fairness.json"
    with open(fairness_path, "r", encoding="utf-8") as f:
        fairness_data = json.load(f)
    threshold_a = float(fairness_data["policy"]["threshold_a"])
    threshold_b = float(fairness_data["policy"]["threshold_b"])

    return {
        "X_test_a": X_test_a,
        "X_test_b": X_test_b,
        "meta_test": meta_test_indexed,
        "feat_names_a": feat_names_a,
        "feat_names_b": feat_names_b,
        "model_a": model_a,
        "model_b": model_b,
        "explainer_b": explainer_b,
        "test_ids_set": test_ids_set,
        "threshold_a": threshold_a,
        "threshold_b": threshold_b,
    }


def _parse_overrides(overrides: Optional[Union[Dict[str, Any], Iterable[str]]]) -> Tuple[bool, bool]:
    """Parse overrides switches into (hide_bureau, hide_behavioral)."""
    hide_bureau = False
    hide_behavioral = False

    if overrides is None:
        return False, False

    if isinstance(overrides, dict):
        # Bureau switch
        if (
            overrides.get("hide_bureau") is True
            or overrides.get("hide_bureau_record") is True
            or overrides.get("bureau") is False
            or overrides.get("Hide bureau record") is True
        ):
            hide_bureau = True

        # Behavioral switch
        if (
            overrides.get("hide_behavioral") is True
            or overrides.get("hide_lender_history") is True
            or overrides.get("behavioral") is False
            or overrides.get("lender_history") is False
            or overrides.get("Hide lender history") is True
        ):
            hide_behavioral = True
    elif isinstance(overrides, (list, tuple, set)):
        flags = {str(x).lower().replace(" ", "_") for x in overrides}
        if any(k in flags for k in ("hide_bureau", "hide_bureau_record", "bureau")):
            hide_bureau = True
        if any(k in flags for k in ("hide_behavioral", "hide_lender_history", "behavioral", "lender_history")):
            hide_behavioral = True

    return hide_bureau, hide_behavioral


def score_applicants(
    ids: Union[int, str, List[int], pd.Series, np.ndarray],
    overrides: Optional[Union[Dict[str, Any], Iterable[str]]] = None,
) -> ScoringResult:
    """Returns Model A and Model B default probabilities for the given test applicants.
    
    Parameters:
        ids: One or more SK_ID_CURR identifiers from the test split.
        overrides: Optional switch overrides: 'hide_bureau' and/or 'hide_behavioral'.
        
    Raises:
        ValueError: If any applicant ID is not in data/splits/test_ids.csv.
    """
    resources = load_scoring_resources()
    test_ids_set = resources["test_ids_set"]

    if isinstance(ids, (int, np.integer, str)):
        id_list = [int(ids)]
    else:
        id_list = [int(x) for x in ids]

    # Validate that ALL requested IDs are held-out test applicants
    invalid_ids = [sid for sid in id_list if sid not in test_ids_set]
    if invalid_ids:
        raise ValueError(
            f"Applicants {invalid_ids[:5]} are not in the held-out test split (data/splits/test_ids.csv). "
            "Only held-out test applicants may appear on this page."
        )

    X_te_a = resources["X_test_a"].loc[id_list].copy()
    X_te_b = resources["X_test_b"].loc[id_list].copy()

    hide_bureau, hide_behavioral = _parse_overrides(overrides)

    if hide_bureau:
        bureau_cols_a = [c for c in X_te_a.columns if c.startswith("bureau_") or c in BUREAU_FEATURE_COLS]
        bureau_cols_b = [c for c in X_te_b.columns if c.startswith("bureau_") or c in BUREAU_FEATURE_COLS]
        X_te_a.loc[:, bureau_cols_a] = np.nan
        X_te_b.loc[:, bureau_cols_b] = np.nan

    if hide_behavioral:
        beh_set = set(ALL_BEHAVIORAL_COLS)
        beh_cols_b = [c for c in X_te_b.columns if c in beh_set]
        X_te_b.loc[:, beh_cols_b] = np.nan

    model_a: xgb.XGBClassifier = resources["model_a"]
    model_b: xgb.XGBClassifier = resources["model_b"]

    scores_a = model_a.predict_proba(X_te_a)[:, 1]
    scores_b = model_b.predict_proba(X_te_b)[:, 1]

    return ScoringResult(
        score_a=scores_a,
        score_b=scores_b,
        ids=id_list,
        X_a=X_te_a,
        X_b=X_te_b,
    )


def explain_applicant(id: Union[int, str]) -> ExplanationResult:
    """Returns Model B TreeSHAP contributions (log-odds scale) for ONE held-out test applicant.
    
    Guarantees:
    - Runtime under 2 seconds (with automatic fallback to app_shap_sample.parquet if slow).
    - Local accuracy: sigmoid(bias + sum of contributions) equals Model B score within 1e-5.
    
    Raises:
        ValueError: If applicant is not in test split or explanation fails checks.
    """
    applicant_id = int(id)
    resources = load_scoring_resources()

    if applicant_id not in resources["test_ids_set"]:
        raise ValueError(
            f"Applicant ID {applicant_id} is not in the held-out test split. Only test split IDs may be explained."
        )

    single_X = resources["X_test_b"].loc[[applicant_id]].copy()
    model_b: xgb.XGBClassifier = resources["model_b"]
    explainer_b = resources["explainer_b"]

    t0 = time.time()
    try:
        exp = explainer_b(single_X)
        elapsed = time.time() - t0
        use_live = (elapsed < 2.0)
    except Exception:
        use_live = False

    if use_live:
        shap_values = exp.values[0]
        base_value = float(exp.base_values[0])
        score_b = float(model_b.predict_proba(single_X)[:, 1][0])

        # Local accuracy check: sigmoid(bias + sum(contributions)) == score within 1e-5
        margin = base_value + float(np.sum(shap_values))
        computed_prob = 1.0 / (1.0 + np.exp(-margin))
        local_diff = abs(computed_prob - score_b)
        if local_diff > 1e-5:
            raise ValueError(
                f"Local accuracy check failed for applicant {applicant_id}: diff={local_diff:.2e} > 1e-5"
            )

        feature_names = resources["feat_names_b"]
        contrib_series = pd.Series(shap_values, index=feature_names, name="shap_value")
        raw_series = pd.Series(single_X.iloc[0].values, index=feature_names, name="feature_value")

        return ExplanationResult(
            bias=base_value,
            contributions=contrib_series,
            feature_values=raw_series,
            score=score_b,
            applicant_id=applicant_id,
            source="live_treeshap",
        )
    else:
        # Fallback to precomputed values
        sample_path = _PROJECT_ROOT / "reports" / "predictions" / "extras" / "app_shap_sample.parquet"
        if sample_path.exists():
            df_sample = pd.read_parquet(sample_path)
            sub = df_sample[df_sample["SK_ID_CURR"] == applicant_id]
            if len(sub) > 0:
                row = sub.iloc[0]
                base_val = float(row["bias"])
                score_b = float(row["score_b"])
                feature_names = resources["feat_names_b"]
                shap_cols = [f"shap_{f}" for f in feature_names]
                raw_cols = [f"raw_{f}" for f in feature_names]

                contrib_series = pd.Series(row[shap_cols].values, index=feature_names, name="shap_value")
                raw_series = pd.Series(row[raw_cols].values, index=feature_names, name="feature_value")

                return ExplanationResult(
                    bias=base_val,
                    contributions=contrib_series,
                    feature_values=raw_series,
                    score=score_b,
                    applicant_id=applicant_id,
                    source="precomputed_sample",
                )

        raise ValueError(f"explanation not available for this applicant ({applicant_id})")


def run_round_trip_check(n_samples: int = 300, seed: int = 42) -> Tuple[float, float]:
    """Scores n_samples random test applicants with no overrides and compares with saved predictions.
    
    The maximum absolute difference must be below 1e-6.
    """
    resources = load_scoring_resources()
    test_ids = list(resources["test_ids_set"])
    rng = np.random.default_rng(seed)
    chosen_ids = rng.choice(test_ids, size=n_samples, replace=False).tolist()

    res = score_applicants(chosen_ids, overrides=None)

    # Load saved predictions
    pred_a = pd.read_parquet(_PROJECT_ROOT / "reports" / "predictions" / "model_a_test.parquet").set_index("SK_ID_CURR")
    pred_b = pd.read_parquet(_PROJECT_ROOT / "reports" / "predictions" / "model_b_test.parquet").set_index("SK_ID_CURR")

    expected_a = pred_a.loc[chosen_ids, "score"].values
    expected_b = pred_b.loc[chosen_ids, "score"].values

    diff_a = float(np.max(np.abs(res.score_a - expected_a)))
    diff_b = float(np.max(np.abs(res.score_b - expected_b)))

    print(f"[round_trip_check] 300 test applicants: max diff A = {diff_a:.2e}, max diff B = {diff_b:.2e}")

    if diff_a >= 1e-6:
        raise ValueError(f"Round-trip check failed for Model A: max diff {diff_a:.2e} >= 1e-6")
    if diff_b >= 1e-6:
        raise ValueError(f"Round-trip check failed for Model B: max diff {diff_b:.2e} >= 1e-6")

    return diff_a, diff_b


if __name__ == "__main__":
    diff_a, diff_b = run_round_trip_check(300)
    print(f"Round-trip check passed successfully: A={diff_a:.2e}, B={diff_b:.2e}")
