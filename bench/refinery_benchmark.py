#!/usr/bin/env python3
"""Open refinery-petrochemical benchmark (Du et al. 2025, arXiv 2503.22057).

Runs cases 1-3 with QENIVO's warm-started homotopy + penalty SLP, checks every plan on the
original bilinear model, and records the gap to BARON's published bound (reference only).

Usage:
  python -m bench.refinery_benchmark --cases 1 --time-limit 120 --out bench/results/refinery_benchmark.jsonl
  python bench/refinery_benchmark.py --cases 1 2 3 --time-limit 1800
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

# Published BARON numbers from Du et al. 2025 — reference only, not our solve.
BARON = {
    1: {"best_found": 34168000.0, "best_possible": 37910846.9074, "time_s": 18015},
    2: {"best_found": 67303200.0, "best_possible": 76374974.0096, "time_s": 18020},
    3: {"best_found": 125250000.0, "best_possible": 135294529.942, "time_s": 18026},
}
GAMS_URL = "https://raw.githubusercontent.com/EMRPS/refinery-planning-benchmark/main/case{n}/case{n}.gms"


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def ensure_case(n: int, inst_dir: Path) -> Path:
    inst_dir.mkdir(parents=True, exist_ok=True)
    path = inst_dir / f"case{n}.gms"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = GAMS_URL.format(n=n)
    print(f"downloading case{n} from {url}")
    urllib.request.urlretrieve(url, path)
    return path


def run_case(n: int, time_limit: float, engine: str, bound_time: float, verbose: bool) -> dict:
    from qenivo.io.gams import read_gams
    from qenivo.workload.bilinear_bound import mccormick_bound
    from qenivo.workload.slp import run_slp

    path = ensure_case(n, HERE / "instances" / "refinery_benchmark")
    M = read_gams(path)
    ref = BARON[n]
    t0 = time.perf_counter()
    r = run_slp(M, engine=engine, max_iter=200, time_limit=time_limit, lp_tol=1e-8,
                homotopy_first=True, homotopy_iters=60, verbose=verbose)
    note = "continuous"
    if M.linear.integer is not None and M.linear.integer.any():
        x = r.x.copy()
        x[M.linear.integer] = np.round(x[M.linear.integer])
        M.linear.lx[M.linear.integer] = M.linear.ux[M.linear.integer] = x[M.linear.integer]
        remain = max(1.0, time_limit - (time.perf_counter() - t0))
        r = run_slp(M, x0=x, engine=engine, max_iter=120, time_limit=remain, lp_tol=1e-8,
                    homotopy_first=True, homotopy_iters=40, verbose=verbose)
        note = "binaries fixed by rounding the relaxed SLP plan"

    # Re-check on the model as loaded (after any binary fix the linear bounds are those used).
    abs_v = float(M.violation(r.x).max(initial=0.0))
    act = np.abs(M.evaluate(r.x))
    rel_v = float(np.max(M.violation(r.x) / (1.0 + act))) if M.violation(r.x).size else 0.0
    feasible = bool(rel_v <= 1e-6)

    bound_info = None
    certified_bound = None
    if bound_time > 0:
        remain_b = min(bound_time, max(1.0, time_limit - (time.perf_counter() - t0)))
        try:
            bound_info = mccormick_bound(M, engine=engine, time_limit=remain_b,
                                         use_obbt=True, obbt_vars=32, obbt_time=min(60.0, remain_b),
                                         verbose=verbose)
            certified_bound = bound_info.get("bound")
        except Exception as e:
            bound_info = {"error": str(e)}

    wall = time.perf_counter() - t0
    gap_baron = None
    if feasible and r.objective is not None:
        gap_baron = (ref["best_possible"] - r.objective) / abs(ref["best_possible"])
    vs_best = None
    if feasible and r.objective is not None and abs(ref["best_found"]) > 0:
        vs_best = r.objective / ref["best_found"] - 1.0

    row = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "commit": _commit(),
        "machine": platform.node(),
        "python": platform.python_version(),
        "case": n,
        "size": M.size_str(),
        "engine": engine,
        "objective": float(r.objective),
        "max_violation": abs_v,
        "rel_violation": rel_v,
        "feasible": feasible,
        "iterations": r.iterations,
        "time": wall,
        "status": r.status,
        "phase": r.phase,
        "note": note,
        "baron_reference": {
            "best_found": ref["best_found"],
            "best_possible": ref["best_possible"],
            "time_s": ref["time_s"],
            "label": "published BARON numbers (Du et al. 2025); not a QENIVO solve",
        },
        "vs_baron_best_found": vs_best,
        "gap_to_baron_bound": gap_baron,
        "certified_mccormick_bound": certified_bound,
        "bound_detail": _bound_detail(bound_info),
        "history_tail": r.history[-8:],
    }
    return row


def _bound_detail(bound_info):
    """Compact JSON-safe bound report (truncate huge unbounded-factor lists)."""
    if bound_info is None:
        return None
    if not isinstance(bound_info, dict):
        return bound_info
    out = {}
    for k, v in bound_info.items():
        if k == "solution":
            out[k] = {"verdict": getattr(v, "verdict", None),
                      "objective": getattr(v, "objective", None)}
        elif k == "unbounded_factors" and isinstance(v, list):
            out[k] = {"count": len(v), "sample": v[:16]}
        else:
            out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, nargs="+", default=[1], choices=[1, 2, 3])
    ap.add_argument("--time-limit", type=float, default=120.0,
                    help="wall seconds per case (default 120 short smoke; use 1800 for acceptance)")
    ap.add_argument("--engine", default="simplex",
                    help="LP engine inside SLP (default simplex with basis warm starts)")
    ap.add_argument("--bound-time", type=float, default=60.0,
                    help="seconds for McCormick+FBBT/OBBT bound (0 to skip)")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "refinery_benchmark.jsonl")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for n in args.cases:
        print(f"=== case {n}  time_limit={args.time_limit}s  engine={args.engine} ===")
        row = run_case(n, args.time_limit, args.engine, args.bound_time, args.verbose)
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        feas = "FEASIBLE" if row["feasible"] else "INFEASIBLE"
        print(f"case{n}: {feas}  obj={row['objective']:,.4f}  max_viol={row['max_violation']:.3e}  "
              f"rel={row['rel_violation']:.3e}  iters={row['iterations']}  time={row['time']:.1f}s  "
              f"status={row['status']}")
        if row["certified_mccormick_bound"] is not None:
            print(f"  certified McCormick bound {row['certified_mccormick_bound']:,.4f}")
        br = row["baron_reference"]
        print(f"  BARON reference (published): best {br['best_found']:,.0f}  "
              f"bound {br['best_possible']:,.4f}  time {br['time_s']}s")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
