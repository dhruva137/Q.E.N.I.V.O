"""Bench small crude-unloading MILP: QENIVO milp vs HiGHS (1 thread, 60 s).

    python bench/crude_unloading_bench.py

Writes ``bench/results/crude_unloading_<YYYYMMDD>.jsonl``. Skips HiGHS if
``highspy`` is missing. Does not run medium/large when free RAM < 2.5 GB.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from qenivo import solve  # noqa: E402
from qenivo.io.mps import write_mps  # noqa: E402
from qenivo.models.industry_ext.crude_unloading import (  # noqa: E402
    _free_ram_gb,
    crude_unloading,
    model_sizes,
    size_ok_to_solve,
)

OUT = HERE / "results" / f"crude_unloading_{date.today().strftime('%Y%m%d')}.jsonl"
TIME_LIMIT = 60.0
THREADS = 1


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return ""


def _highs(path: Path, time_limit: float, threads: int = 1) -> dict:
    try:
        import highspy
    except ImportError:
        return {"status": "skipped", "reason": "highspy not installed"}
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("threads", int(threads))
    h.readModel(str(path))
    t0 = time.perf_counter()
    h.run()
    dt = time.perf_counter() - t0
    st = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    return {
        "status": st,
        "objective": float(info.objective_function_value) if st in ("Optimal", "Feasible") else None,
        "bound": float(getattr(info, "mip_dual_bound", float("nan"))),
        "nodes": int(getattr(info, "mip_node_count", 0)),
        "gap": float(getattr(info, "mip_gap", float("nan"))),
        "time": dt,
        "threads": threads,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    free = _free_ram_gb()
    sizes = model_sizes(seed=0)
    print("model sizes:", json.dumps(sizes, indent=2))
    print(f"free_ram_gb={free:.2f}")

    # Agent A: small only vs HiGHS under the RAM rule.
    size = "small"
    if size != "small" and not size_ok_to_solve(size):
        print(f"skip solve {size}: free RAM {free:.2f} GB < 2.5")
        return

    prob = crude_unloading(size, seed=0)
    mps = HERE / "results" / f"_tmp_crude_unloading_{size}.mps"
    write_mps(prob, mps, name=f"crude_unloading_{size}")

    t0 = time.perf_counter()
    sol = solve(prob, engine="milp", time_limit=TIME_LIMIT)
    ours_time = time.perf_counter() - t0
    ours = {
        "verdict": sol.verdict,
        "status": sol.status,
        "objective": sol.objective,
        "bound": sol.engine.get("bound"),
        "gap": sol.engine.get("gap"),
        "nodes": sol.engine.get("nodes"),
        "time": sol.engine.get("time", ours_time),
        "threads": THREADS,
        "cuts": sol.engine.get("cuts"),
        "presolve": sol.engine.get("presolve"),
    }
    highs = _highs(mps, TIME_LIMIT, THREADS)
    try:
        mps.unlink()
    except OSError:
        pass

    row = {
        "commit": _commit(),
        "machine": platform.node(),
        "python": platform.python_version(),
        "instance": f"crude_unloading_{size}",
        "size": size,
        "seed": 0,
        "rows": prob.m,
        "cols": prob.n,
        "nnz": prob.nnz,
        "integers": int(prob.integer.sum()) if prob.integer is not None else 0,
        "time_limit": TIME_LIMIT,
        "threads": THREADS,
        "free_ram_gb": free,
        "ours": ours,
        "highs": highs,
        "sizes": sizes,
        "citation": prob.meta.get("citation"),
    }
    if (
        ours.get("objective") is not None
        and highs.get("objective") is not None
        and highs.get("status") == "Optimal"
        and ours.get("verdict") == "optimal"
    ):
        denom = max(1.0, abs(highs["objective"]))
        row["obj_rel_err_vs_highs"] = abs(ours["objective"] - highs["objective"]) / denom

    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
