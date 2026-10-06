"""One measured run of the C ABI: the bound LP and the two-binary MILP.

The simplex row is the native core. The MILP row goes through the embedded Python
bridge. Seconds are the wall time of the solve call after the library is loaded.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "capi"))
sys.path.insert(0, str(ROOT / "src"))

import capi_loader  # noqa: E402


def commit() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True, text=True)
    return (proc.stdout or "unknown").strip() or "unknown"


def compiler_line() -> str:
    cxx = capi_loader.compiler_used()
    proc = subprocess.run([cxx, "--version"], capture_output=True, text=True)
    line = (proc.stdout or proc.stderr or cxx).splitlines()
    return line[0] if line else cxx


def solve_bound():
    return capi_loader.solve_csr(
        np.array([-1.0]), np.array([0], dtype=np.int32), None, None,
        np.zeros(0), np.zeros(0), np.array([0.0]), np.array([1.0]), engine="simplex", name="bound")


def solve_milp():
    return capi_loader.solve_csr(
        np.array([-1.0, -2.0]), np.array([0, 2], dtype=np.int32), np.array([0, 1], dtype=np.int32),
        np.array([1.0, 1.0]), np.array([-np.inf]), np.array([1.0]), np.zeros(2), np.ones(2),
        integrality=np.array([1, 1], dtype=np.int32), engine="milp", name="two_binary")


def row(kind, engine, sol, seconds, meta):
    return {
        "suite": "bench_capi",
        "commit": meta["commit"],
        "machine": meta["machine"],
        "python": meta["python"],
        "numpy": meta["numpy"],
        "compiler": meta["compiler"],
        "kind": kind,
        "engine": engine,
        "status": sol.status_name,
        "optimum": sol.objective,
        "x": [float(v) for v in sol.x],
        "seconds": seconds,
        "ok": abs(sol.objective - meta["expect"][kind]) < 1e-6,
    }


def main():
    capi_loader.library()
    meta = {
        "commit": commit(),
        "machine": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "compiler": compiler_line(),
        "expect": {"bound_lp": -1.0, "two_binary": -2.0},
    }
    t0 = time.perf_counter()
    bound = solve_bound()
    bound_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    milp = solve_milp()
    milp_s = time.perf_counter() - t1
    rows = [
        row("bound_lp", "simplex", bound, bound_s, meta),
        row("two_binary", "milp", milp, milp_s, meta),
    ]
    if not all(item["ok"] for item in rows):
        raise SystemExit("C ABI benchmark did not reach the known optima: " + json.dumps(rows))
    out = HERE / "results" / "capi.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
    print(out)
    for item in rows:
        print(f"{item['kind']} optimum {item['optimum']} seconds {item['seconds']:.6f}")


if __name__ == "__main__":
    main()
