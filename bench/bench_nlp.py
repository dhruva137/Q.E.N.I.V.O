"""E3 timings. Writes bench/results/nlp.jsonl from this process only."""
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

from qenivo.nlp.ad import gradient, hvp  # noqa: E402
from qenivo.nlp.dense_kkt import factor  # noqa: E402
from qenivo.nlp.expr import Graph, cos, evaluate, exp, log, sin, sqrt  # noqa: E402
from qenivo.nlp.ipm import solve_ipm  # noqa: E402
from qenivo.nlp.oa import convex_demo, outer_approximation  # noqa: E402
from qenivo.nlp.pooling import haverly_nlp, solve_haverly  # noqa: E402
from qenivo.nlp.problem import NLPModel  # noqa: E402


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _base():
    return {
        "suite": "bench_nlp",
        "commit": _commit(),
        "machine": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def _derivative():
    g = Graph()
    x = g.var("x", 0.2, 5.0)
    y = g.var("y", 0.2, 5.0)
    f = sin(x) * exp(y) + (x ** 2) * y + log(x) + sqrt(y) + x / y + cos(x) - y ** 3
    p = np.array([1.3, 0.8])
    got = gradient(f, p)
    eps = 1e-6
    fd = np.zeros(2)
    for i in range(2):
        step = np.zeros(2)
        step[i] = eps
        fd[i] = (evaluate(f, p + step) - evaluate(f, p - step)) / (2.0 * eps)
    err = float(np.max(np.abs(got - fd)))
    hv = hvp(f, p, np.array([0.4, -0.2]))
    fd_h = (gradient(f, p + eps * np.array([0.4, -0.2])) - gradient(f, p - eps * np.array([0.4, -0.2]))) / (2.0 * eps)
    herr = float(np.max(np.abs(hv - fd_h)))
    return {"case": "derivatives", "gradient_error": err, "hvp_error": herr, "ok": err < 1e-5 and herr < 1e-4}


def _ldl():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(5, 5))
    spd = base.T @ base + np.eye(5)
    rhs = rng.normal(size=5)
    fac = factor(spd)
    sol = fac.solve(rhs)
    err = float(np.linalg.norm(spd @ sol - rhs))
    return {"case": "ldl", "residual": err, "inertia": list(fac.inertia), "ok": err < 1e-8 and fac.inertia == (5, 0, 0)}


def _ipm():
    g = Graph()
    x, y = g.var("x"), g.var("y")
    model = NLPModel(g, (x - 1.0) ** 2 + (y - 1.0) ** 2, equalities=[x + y - 1.0])
    sol = solve_ipm(model, tol=1e-8, time_limit=10.0)
    return {
        "case": "ipm_equality",
        "status": sol.status,
        "objective": sol.objective,
        "max_res": sol.residuals.get("max_res"),
        "ok": sol.status == "optimal" and abs(sol.objective - 0.5) < 1e-6,
    }


def _haverly(instance, published):
    result = solve_haverly(instance, tol=1e-4, time_limit=30.0)
    return {
        "case": f"haverly_{instance}",
        "status": result.status,
        "profit": result.profit,
        "objective": result.objective,
        "lower_bound": result.lower_bound,
        "root_relaxation": result.root_relaxation,
        "nodes": result.extra.get("nodes"),
        "published_profit": published,
        "ok": result.status == "optimal" and result.profit is not None and abs(result.profit - published) < 1e-3,
    }


def _oa():
    result = outer_approximation(convex_demo(), tol=1e-4, time_limit=30.0)
    return {
        "case": "outer_approximation",
        "status": result.status,
        "objective": result.objective,
        "lower_bound": result.lower_bound,
        "nlp_solves": result.nlp_solves,
        "master_solves": result.master_solves,
        "notes": result.notes,
        "ok": result.status == "optimal" and result.objective is not None and abs(result.objective) < 1e-3,
    }


def _local_pool():
    model = haverly_nlp(1)
    rows = []
    starts = {
        "global_reference": np.array([0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 1.0]),
        "local_basin": np.array([50.0, 0.0, 50.0, 0.0, 50.0, 0.0, 2.5]),
    }
    for name, x0 in starts.items():
        sol = solve_ipm(model, tol=1e-6, time_limit=8.0, x0=x0, max_iter=40)
        profit = None if sol.objective is None else -sol.objective
        rows.append({
            "case": f"local_ipm_haverly1_{name}",
            "status": sol.status,
            "objective": sol.objective,
            "profit": profit,
            "max_res": None if not sol.residuals else sol.residuals.get("max_res"),
            "notes": sol.notes,
            "ok": sol.status == "optimal",
        })
    return rows


def main() -> int:
    jobs = [_derivative, _ldl, _ipm, lambda: _haverly(1, 400.0), lambda: _haverly(2, 600.0),
            lambda: _haverly(3, 750.0), _oa]
    rows = []
    for job in jobs:
        t0 = time.perf_counter()
        row = job()
        row["seconds"] = round(time.perf_counter() - t0, 6)
        row.update(_base())
        rows.append(row)
    for row in _local_pool():
        row.update(_base())
        rows.append(row)
    out = ROOT / "bench" / "results" / "nlp.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    failed = [row["case"] for row in rows
              if not row["ok"] and (row["case"] == "derivatives" or str(row["case"]).startswith("haverly"))]
    print(f"wrote {out} ({len(rows)} rows)")
    for row in rows:
        print(row["case"], row.get("status", ""), "ok" if row["ok"] else "FAIL", row.get("seconds"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
