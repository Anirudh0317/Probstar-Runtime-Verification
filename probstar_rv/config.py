"""Configuration loading and a typed default fallback.

The project reads parameters from ``config.yaml`` at the repository root.
``load_config`` returns a plain nested ``dict``.  If PyYAML or the file is
unavailable, a built-in default (identical to the shipped ``config.yaml``)
is returned so the library still works out of the box.
"""

from __future__ import annotations

import os
from typing import Any, Dict

# A copy of config.yaml's contents, used as a fallback if the file / PyYAML
# cannot be read.  Keep this in sync with config.yaml.
_DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 12345,
    "vehicle": {
        "wheelbase_L": 0.33,
        "length_half": 0.275,
        "width_half": 0.15,
    },
    "reachability": {
        "dt_r": 0.25,
        "dt_m": 0.025,
        "k_max": 100,
        "delta": 6.0,
    },
    "probability": {
        "mvn_backend": "auto",
        "genz_samples": 20000,
        "scipy_maxdim": 6,
    },
    "monte_carlo": {
        "n_samples": 100000,
    },
}


def _repo_root() -> str:
    """Return the repository root (the directory containing config.yaml)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def default_config() -> Dict[str, Any]:
    """Return a deep-ish copy of the built-in default configuration."""
    import copy

    return copy.deepcopy(_DEFAULT_CONFIG)


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        path: Explicit path to a YAML config.  If ``None``, looks for
            ``config.yaml`` next to the repository root.

    Returns:
        A nested dict of configuration values.  Falls back to the built-in
        default if the file or PyYAML is unavailable.  Any keys present in the
        file override the defaults (shallow-merged at the top level, deep at
        the section level).
    """
    cfg = default_config()

    if path is None:
        path = os.path.join(_repo_root(), "config.yaml")

    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
    except Exception:
        # No file or no PyYAML -> use defaults.  This keeps the library usable
        # even in a minimal environment; experiments still print the values.
        return cfg

    # Deep-merge section dicts so a partial config file is acceptable.
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg
