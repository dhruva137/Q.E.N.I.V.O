"""Smoke benchmark for the exact rational LP engine.

Writes bench/results/exact_lp.jsonl from this run. Exit 0 only when the
textbook LP is proved at x = (4/3, 4/3), so x1 + x2 = 8/3 exactly.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qenivo.engines.exact_lp import library, solve_exact_lp, status  # noqa: E402
from qenivo.model import ModelBuilder  # noqa: E402


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _textbook():
    builder = ModelBuilder("tiny")
    builder.var("x1", 0, np.inf, obj=-1)
    builder.var("x2", 0, np.inf, obj=-1)
    builder.row("r1", {"x1": 1, "x2": 2}, hi=4)
    builder.row("r2", {"x1": 2, "x2": 1}, hi=4)
    return builder.build()


def _infeasible():
    builder = ModelBuilder("infeas")
    builder.var("x", 0, np.inf, obj=0)
    builder.row("r", {"x": 1}, hi=-1)
    return builder.build()


def _unbounded():
    builder = ModelBuilder("unbounded")
    builder.var("x1", 0, np.inf, obj=-1)
    builder.var("x2", 0, np.inf, obj=0)
    builder.row("r", {"x1": -1, "x2": 1}, hi=3)
    return builder.build()


def _row(name, arithmetic, sol, seconds):
    proof = sol.extra["rational"]
    x_sum = None
    if proof.get("x") and len(proof["x"]) >= 2:
        x_sum = str(Fraction(proof["x"][0]) + Fraction(proof["x"][1]))
    return {
        "suite": "bench_exact_lp",
        "commit": _commit(),
        "machine": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "native": status(),
        "problem": name,
        "arithmetic": arithmetic,
        "verdict": sol.verdict,
        "objective": proof.get("objective"),
        "x": proof.get("x"),
        "y": proof.get("y"),
        "x1_plus_x2": x_sum,
        "farkas": proof.get("farkas"),
        "phi": proof.get("phi"),
        "ray": proof.get("ray"),
        "descent": proof.get("descent"),
        "linear_solver": proof.get("linear_solver"),
        "basis_source": proof.get("basis_source"),
        "seconds": round(seconds, 6),
        "proven_exactly": bool(proof.get("proven")),
    }


def main() -> int:
    cases = [("textbook", _textbook()), ("infeasible", _infeasible()), ("unbounded", _unbounded())]
    modes = ["python"]
    if library() is not None:
        modes.append("native")
    rows = []
    for name, prob in cases:
        for mode in modes:
            t0 = time.perf_counter()
            sol = solve_exact_lp(prob, arithmetic=mode, candidate=False)
            rows.append(_row(name, mode, sol, time.perf_counter() - t0))
    out = ROOT / "bench" / "results" / "exact_lp.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    textbook = [row for row in rows if row["problem"] == "textbook"]
    proved = all(
        row["proven_exactly"] and row["x"] == ["4/3", "4/3"] and row["x1_plus_x2"] == "8/3"
        and row["objective"] == "-8/3"
        for row in textbook
    )
    print(f"wrote {out} ({len(rows)} rows, 8/3 proved={proved})")
    return 0 if proved else 1


if __name__ == "__main__":
    raise SystemExit(main())
