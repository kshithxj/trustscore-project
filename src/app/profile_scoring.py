"""Profile scoring and information ablation helper for applicant evaluation.

Phase 8 Stage 6 / Stage 5: Profile scoring across varying information levels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.app.scoring import (
    load_scoring_resources,
    score_applicants,
    explain_applicant,
    ScoringResult,
    ExplanationResult,
)

INFORMATION_LEVEL_MAP = {
    "No history": {"hide_bureau": True, "hide_behavioral": True},
    "Lender history only": {"hide_bureau": True, "hide_behavioral": False},
    "Bureau only": {"hide_bureau": False, "hide_behavioral": True},
    "Full information": {"hide_bureau": False, "hide_behavioral": False},
}


def score_with_information_level(
    applicant_id: Union[int, str],
    information_level: str = "Full information",
) -> ScoringResult:
    """Scores an applicant at a designated information ablation level.
    
    Levels:
    - 'No history': suppresses bureau and behavioral features.
    - 'Lender history only': suppresses bureau features, retains behavioral features.
    - 'Bureau only': retains bureau features, suppresses behavioral features.
    - 'Full information': retains both bureau and behavioral features.
    """
    overrides = INFORMATION_LEVEL_MAP.get(
        information_level,
        {"hide_bureau": False, "hide_behavioral": False},
    )
    return score_applicants(applicant_id, overrides=overrides)


def get_applicant_meta(applicant_id: Union[int, str]) -> Dict[str, Any]:
    """Retrieves metadata for an applicant from the held-out test split."""
    resources = load_scoring_resources()
    sid = int(applicant_id)
    if sid not in resources["test_ids_set"]:
        raise ValueError(f"Applicant {sid} is not in the held-out test split.")

    row = resources["meta_test"].loc[sid]
    return {
        "SK_ID_CURR": sid,
        "segment": str(row["segment"]),
        "TARGET": int(row["TARGET"]),
    }
