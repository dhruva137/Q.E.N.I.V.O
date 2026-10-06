"""Solve each tiny E5 model and write bench/results/models.jsonl.

Exits 1 if any proven objective disagrees with the hand-checked EXPECTED value.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from qenivo import solve
from qenivo.models.industry import LIBRARY
from qenivo.models.industry_ext import EXPECTED as EXT_EXPECTED
from qenivo.models.refinery_full.planner import EXPECTED as REF_EXPECTED

EXPECTED = {**REF_EXPECTED, **EXT_EXPECTED}
NAMES = (
    "refinery_pims",
    "refinery_delta",
    "refinery_pool_relax",
    "petrochemical",
    "sced",
    "hydro_thermal",
    "vrptw",
    "multi_echelon",
    "rcpsp",
)
MILP = {"vrptw", "rcpsp"}
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results" / "models.jsonl"


def agrees(got, exp) -> bool:
    if got is None:
        return False
    return abs(got - exp) <= 1e-6 * max(1.0, abs(exp))


def main() -> int:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    machine = platform.node() or platform.machine()
    rows = []
    failed = False
    for name in NAMES:
        prob = LIBRARY[name]()
        started = time.perf_counter()
        if name in MILP:
            sol = solve(prob, engine="milp", time_limit=15)
        else:
            sol = solve(prob, engine="simplex", polish=False, time_limit=15)
        seconds = time.perf_counter() - started
        row = {
            "commit": commit,
            "machine": machine,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "name": name,
            "verdict": sol.verdict,
            "objective": sol.objective,
            "expected": EXPECTED[name],
            "seconds": seconds,
        }
        rows.append(row)
        ok = sol.verdict == "optimal" and agrees(sol.objective, EXPECTED[name])
        if not ok:
            failed = True
            print(f"DISAGREE {name}: {sol.summary()}", file=sys.stderr)
        else:
            print(f"{name}: {sol.objective} in {seconds:.3f}s")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"wrote {OUT} ({len(rows)} rows)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
