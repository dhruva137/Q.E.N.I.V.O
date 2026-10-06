"""Fixed-seed E6 smoke. Writes bench/results/robustness.jsonl from this run only.

Exit 0 when every proven optimum matches the planted value. Exit 1 when a
proven answer is wrong. ``not_proven`` is recorded and is not a failure.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "prop"))

from support import infeasible_box, planted_lp, planted_qp, solve_checked, unbounded_ray  # noqa: E402


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _row(kind, seed, engine, verdict, planted, got, seconds, ok):
    return {
        "suite": "bench_robustness",
        "commit": _commit(),
        "machine": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "kind": kind,
        "seed": seed,
        "engine": engine,
        "verdict": verdict,
        "planted": planted,
        "objective": got,
        "seconds": round(seconds, 6),
        "ok": ok,
    }


def main() -> int:
    rows = []
    failed = False
    cases = []
    for seed, m, n in ((1, 2, 4), (2, 3, 5), (3, 2, 6)):
        prob, value = planted_lp(np.random.default_rng(seed), m, n)
        cases.append(("planted_lp", seed, prob, "optimal", value, ("simplex", "ipm", "pdhg")))
    prob, value = planted_qp(np.random.default_rng(4), 3)
    cases.append(("planted_qp", 4, prob, "optimal", value, ("pdqp",)))
    cases.append(("infeasible", 5, infeasible_box(np.random.default_rng(5), 3), "infeasible", None, ("simplex",)))
    cases.append(("unbounded", 6, unbounded_ray(np.random.default_rng(6)), "unbounded", None, ("simplex",)))

    for kind, seed, prob, expect, planted, engines in cases:
        for engine in engines:
            t0 = time.perf_counter()
            try:
                sol = solve_checked(prob, engine, expect, planted, time_limit=5.0)
                ok = True
                verdict, got = sol.verdict, sol.objective
            except AssertionError as exc:
                ok = False
                failed = True
                verdict, got = "FAIL", str(exc).splitlines()[0][:240]
            rows.append(_row(kind, seed, engine, verdict, planted, got, time.perf_counter() - t0, ok))

    out = ROOT / "bench" / "results" / "robustness.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    proven = sum(1 for row in rows if row["verdict"] == "optimal")
    print(f"wrote {out} ({len(rows)} rows, {proven} optimal, failed={failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
