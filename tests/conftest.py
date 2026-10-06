import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "prop"))

from qenivo.model import ModelBuilder  # noqa: E402
from qenivo.provenance import install_tripwires  # noqa: E402

install_tripwires()
DATA = ROOT / "tests" / "data"


@pytest.fixture
def data_dir():
    return DATA


def lp_all_bound_types():
    """Min LP exercising every row and column bound pattern; optimum from the simplex engine."""
    b = ModelBuilder("mixed")
    b.var("x1", 0, 4, obj=-1.0)          # boxed
    b.var("x2", -np.inf, np.inf, obj=2.0)  # free
    b.var("x3", -np.inf, 3, obj=-1.0)     # upper only
    b.var("x4", 1, np.inf, obj=1.0)       # lower only
    b.var("x5", 2, 2, obj=0.5)            # fixed
    b.row("eq", {"x1": 1, "x2": 1, "x3": 1}, 5, 5)
    b.row("le", {"x1": 2, "x3": -1, "x4": 1}, hi=6)
    b.row("ge", {"x2": 1, "x4": 1, "x5": 1}, lo=1)
    b.row("rng", {"x1": 1, "x3": 1, "x4": -1}, lo=-2, hi=4)
    b.row("free_row", {"x1": 1, "x2": 1})
    return b.build()


def infeasible_lp():
    b = ModelBuilder("infeasible")
    b.var("x", 0, 10); b.var("y", 0, 10)
    b.row("need", {"x": 1, "y": 1}, lo=25)
    b.row("cap_x", {"x": 1}, hi=3)
    return b.build()


def unbounded_lp():
    b = ModelBuilder("unbounded")
    b.var("x", 0, obj=-1.0); b.var("y", 0, obj=-1.0)
    b.row("r1", {"x": 1, "y": -1}, hi=2)
    b.row("r2", {"x": 1}, lo=1)
    return b.build()
