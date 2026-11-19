"""Smoke tests – verify project scaffolding is healthy."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src` is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config, set_seed


def test_config_loads():
    """Config file loads without error."""
    cfg = load_config()
    assert isinstance(cfg, dict), "load_config() should return a dict"


def test_config_has_seed_42():
    """Config contains seed == 42."""
    cfg = load_config()
    assert cfg["seed"] == 42, f"Expected seed 42, got {cfg['seed']}"


def test_config_has_target_column():
    """Config specifies the target column."""
    cfg = load_config()
    assert cfg["target_column"] == "TARGET"


def test_config_has_split_fractions():
    """Split fractions sum to 1.0."""
    cfg = load_config()
    total = cfg["split"]["train"] + cfg["split"]["valid"] + cfg["split"]["test"]
    assert abs(total - 1.0) < 1e-9, f"Split fractions sum to {total}, expected 1.0"


def test_set_seed_returns_42():
    """set_seed() returns the seed it applied."""
    seed = set_seed(42)
    assert seed == 42


def test_set_seed_from_config():
    """set_seed() reads from config when no explicit seed given."""
    seed = set_seed()
    assert seed == 42
