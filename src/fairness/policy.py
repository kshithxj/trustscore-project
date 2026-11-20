"""Decision policy for credit risk models.

Computes threshold on validation scores targeting a specific decline rate,
and applies it to scores to generate binary decisions (decline = 1, approve = 0).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def compute_threshold_from_validation(
    val_scores: np.ndarray,
    decline_rate: float = 0.15
) -> float:
    """Choose threshold on validation scores so top `decline_rate` fraction is declined.
    
    Applicants with score >= threshold are declined.
    Threshold corresponds to the (1 - decline_rate) quantile.
    """
    pct = 100.0 * (1.0 - decline_rate)
    threshold = float(np.percentile(val_scores, pct))
    return threshold


def apply_decision_policy(
    scores: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply threshold to scores.
    
    Returns:
        declines: array of 0 (approve) or 1 (decline)
        approvals: array of 1 (approve) or 0 (decline)
    """
    declines = (scores >= threshold).astype(int)
    approvals = 1 - declines
    return declines, approvals
