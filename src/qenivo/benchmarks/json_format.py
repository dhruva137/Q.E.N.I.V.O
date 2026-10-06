"""JSON sparse-triplet model format for web / API clients.

Owned by the packaging/CLI task; does not modify ``qenivo.io``.

Schema (version 1)::

    {
      "format": "qenivo-json",
      "version": 1,
      "name": "...",
      "sense": "min" | "max",
      "objective_offset": 0.0,
      "columns": [{"name": "x1", "obj": 1.0, "lb": 0.0, "ub": null, "integer": false}, ...],
      "rows": [{"name": "r1", "lb": null, "ub": 10.0}, ...],
      "A": {"rows": [i, ...], "cols": [j, ...], "values": [a, ...]},
      "Q": null | {"rows": [...], "cols": [...], "values": [...]}
    }

Infinite bounds are encoded as ``null``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..model import Problem

FORMAT = "qenivo-json"
VERSION = 1


def _finite_or_none(v: float):
    v = float(v)
    return None if not np.isfinite(v) else v


def problem_to_json(prob: Problem) -> dict:
    """Serialize a ``Problem`` to the qenivo-json dict (sparse triplets)."""
    rn, cn = prob.names()
    A = prob.A.tocoo()
    columns = []
    for j in range(prob.n):
        columns.append({
            "name": cn[j],
            "obj": float(prob.obj_sign * prob.c[j]),
            "lb": _finite_or_none(prob.lx[j]),
            "ub": _finite_or_none(prob.ux[j]),
            "integer": bool(prob.integer[j]) if prob.integer is not None else False,
        })
    rows = []
    for i in range(prob.m):
        rows.append({
            "name": rn[i],
            "lb": _finite_or_none(prob.lc[i]),
            "ub": _finite_or_none(prob.uc[i]),
        })
    payload = {
        "format": FORMAT,
        "version": VERSION,
        "name": prob.name or "",
        "sense": "max" if prob.obj_sign < 0 else "min",
        "objective_offset": float(prob.obj_sign * prob.c0),
        "columns": columns,
        "rows": rows,
        "A": {
            "rows": [int(i) for i in A.row.tolist()],
            "cols": [int(j) for j in A.col.tolist()],
            "values": [float(v) for v in A.data.tolist()],
        },
        "Q": None,
    }
    if prob.Q is not None and prob.Q.nnz:
        Q = prob.Q.tocoo()
        payload["Q"] = {
            "rows": [int(i) for i in Q.row.tolist()],
            "cols": [int(j) for j in Q.col.tolist()],
            "values": [float(v) for v in Q.data.tolist()],
        }
    return payload


def problem_from_json(data: dict) -> Problem:
    """Build a ``Problem`` from a qenivo-json dict."""
    if data.get("format") not in (FORMAT, "qenivo", None) and "columns" not in data:
        raise ValueError(f"unsupported JSON model format: {data.get('format')!r}")
    cols = data["columns"]
    rows = data["rows"]
    n, m = len(cols), len(rows)
    sense = str(data.get("sense", "min")).lower()
    obj_sign = -1.0 if sense.startswith("max") else 1.0
    c = np.array([float(c.get("obj", 0.0)) for c in cols], dtype=np.float64) * obj_sign
    lx = np.array([
        -np.inf if c.get("lb") is None else float(c["lb"]) for c in cols
    ], dtype=np.float64)
    ux = np.array([
        np.inf if c.get("ub") is None else float(c["ub"]) for c in cols
    ], dtype=np.float64)
    lc = np.array([
        -np.inf if r.get("lb") is None else float(r["lb"]) for r in rows
    ], dtype=np.float64)
    uc = np.array([
        np.inf if r.get("ub") is None else float(r["ub"]) for r in rows
    ], dtype=np.float64)
    integer = np.array([bool(c.get("integer", False)) for c in cols], dtype=bool)
    Ablock = data.get("A") or {}
    Ai = np.asarray(Ablock.get("rows", []), dtype=np.int32)
    Aj = np.asarray(Ablock.get("cols", []), dtype=np.int32)
    Av = np.asarray(Ablock.get("values", []), dtype=np.float64)
    A = sp.csr_matrix((Av, (Ai, Aj)), shape=(m, n), dtype=np.float64)
    Q = None
    if data.get("Q"):
        Qb = data["Q"]
        Qi = np.asarray(Qb.get("rows", []), dtype=np.int32)
        Qj = np.asarray(Qb.get("cols", []), dtype=np.int32)
        Qv = np.asarray(Qb.get("values", []), dtype=np.float64)
        Q = sp.csr_matrix((Qv, (Qi, Qj)), shape=(n, n), dtype=np.float64)
    c0 = float(data.get("objective_offset", 0.0)) * obj_sign
    return Problem(
        c=c, A=A, lc=lc, uc=uc, lx=lx, ux=ux, c0=c0, obj_sign=obj_sign,
        name=str(data.get("name", "")),
        row_names=[str(r.get("name", f"R{i}")) for i, r in enumerate(rows)],
        col_names=[str(c.get("name", f"C{j}")) for j, c in enumerate(cols)],
        integer=integer if integer.any() else None,
        Q=Q,
    )


def write_json(prob: Problem, path) -> None:
    path = Path(path)
    path.write_text(json.dumps(problem_to_json(prob), indent=1), encoding="utf-8")


def read_json(path) -> Problem:
    path = Path(path)
    return problem_from_json(json.loads(path.read_text(encoding="utf-8")))
