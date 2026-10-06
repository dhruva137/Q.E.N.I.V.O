"""Certificates: every QENIVO answer as a self-describing, re-checkable JSON record.

Schema "qenivo.certificate/1":
    model        name, SHA-256 of the numerical data, size, source path
    verdict      optimal | infeasible | unbounded | not_proven
    evidence     kkt  (primal x and row duals y, residuals at the stated tolerance)
                 farkas (a dual ray y that proves no feasible point exists)
                 ray  (a primal direction that proves the objective is unbounded)
    engine       which engine ran, on what backend, why it was chosen, iterations, time
    environment  QENIVO version, Python, NumPy, OS, CPU, GPU
A certificate is re-checked by `qenivo verify model.mps cert.json`, which parses the model with
its own reader and recomputes every residual with compensated summation (and exact rational
arithmetic on request), sharing no code with the engines.
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import sys
from pathlib import Path

import numpy as np

SCHEMA = "qenivo.certificate/1"


def environment() -> dict:
    from .. import __version__
    from ..kernels.backend import gpu_info
    env = {"qenivo": __version__, "python": sys.version.split()[0], "numpy": np.__version__,
           "platform": platform.platform(), "machine": platform.machine(),
           "processor": platform.processor() or platform.machine()}
    try:
        import scipy
        env["scipy"] = scipy.__version__
    except Exception:
        pass
    env.update(gpu_info())
    return env


def build(prob, *, verdict: str, status: str, tol: float, residuals: dict | None,
          evidence_kind: str, x=None, y=None, ray=None, ray_check=None, engine: dict,
          source: str | None = None, include_solution: bool = True, extra: dict | None = None) -> dict:
    rn, cn = prob.names()
    cert = {
        "schema": SCHEMA,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "model": {"name": prob.name, "sha256": prob.fingerprint(), "rows": prob.m, "cols": prob.n,
                  "nnz": prob.nnz, "sense": "maximize" if prob.obj_sign < 0 else "minimize",
                  "source": source},
        "verdict": verdict,
        "status": status,
        "tolerance": tol,
        "objective": None,
        "residuals": residuals or {},
        "evidence": {"kind": evidence_kind},
        "engine": engine,
        "environment": environment(),
    }
    if residuals and "objective" in residuals:
        cert["objective"] = residuals["objective"]
        cert["dual_objective"] = residuals.get("dual_objective")
    if evidence_kind in ("farkas", "ray") and ray is not None:
        cert["evidence"]["check"] = ray_check
        cert["evidence"]["vector"] = _named(rn if evidence_kind == "farkas" else cn, ray)
    if include_solution and x is not None:
        # duals are reported in the user's sense (a max problem's duals flip sign)
        cert["solution"] = {"x": _named(cn, x),
                            "y": _named(rn, prob.obj_sign * np.asarray(y)) if y is not None else None}
    if extra:
        cert.update(extra)
    return cert


def _named(names, v):
    v = np.asarray(v, dtype=float).reshape(-1)
    return {str(k): (float(a) if np.isfinite(a) else None) for k, a in zip(names, v)}


def save(cert: dict, path) -> None:
    Path(path).write_text(json.dumps(cert, indent=1, allow_nan=False, default=_default) + "\n",
                          encoding="utf-8")


def load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return [_default(a) for a in o.tolist()]
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
