"""Shared helpers for experiment scripts: output paths, JSON/CSV writers, and a
consistent matplotlib style.  Importing this also puts the repo root on sys.path
so the experiments run both as `python experiments/exp1_...py` and as modules.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any, Dict, Iterable

# Make the package importable when run as a plain script.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


RESULTS_DIR = os.path.join(_REPO, "results")
DATA_DIR = os.path.join(RESULTS_DIR, "data")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def save_json(name: str, obj: Dict[str, Any]) -> str:
    ensure_dirs()
    path = os.path.join(DATA_DIR, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    return path


def save_csv(name: str, rows: Iterable[Dict[str, Any]], fieldnames=None) -> str:
    ensure_dirs()
    rows = list(rows)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    path = os.path.join(DATA_DIR, name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames or [])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def save_fig(fig, name: str) -> str:
    ensure_dirs()
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    })
