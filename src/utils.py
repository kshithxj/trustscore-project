"""Shared utilities: configuration loading and reproducibility."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load the YAML configuration file and return it as a dict.

    Parameters
    ----------
    path : str | Path | None
        Path to config file.  Defaults to ``configs/config.yaml``
        relative to the project root.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG
    with open(config_path, "r", encoding="utf-8") as fh:
        config: Dict[str, Any] = yaml.safe_load(fh)
    return config


def set_seed(seed: int | None = None, config: Dict[str, Any] | None = None) -> int:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int | None
        Explicit seed value.  If *None*, falls back to ``config["seed"]``.
    config : dict | None
        Configuration dict (used only when *seed* is None).

    Returns
    -------
    int
        The seed that was set.
    """
    if seed is None:
        if config is None:
            config = load_config()
        seed = int(config["seed"])

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    return seed
