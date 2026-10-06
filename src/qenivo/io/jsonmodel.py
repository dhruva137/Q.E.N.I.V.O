"""JSON sparse-triplet model reader and writer for web / API clients."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..model import Problem as LPProblem

INF = math.inf


def _finite_or_str(v: float):
    if np.isinf(v):
        return "inf" if v > 0 else "-inf"
    return float(v)


def _from_json_num(v) -> float:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("inf", "+inf", "infinity"):
            return INF
        if s in ("-inf", "-infinity"):
            return -INF
        return float(v)
    return float(v)


def read_json(path, name: str | None = None) -> LPProblem:
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    sense = -1.0 if str(doc.get("sense", "min")).lower().startswith("max") else 1.0
    obj = doc.get("objective", {})
    c = np.asarray(obj.get("c", []), dtype=np.float64)
    c0 = float(obj.get("c0", 0.0))
    Ainfo = doc["A"]
    rows = np.asarray(Ainfo["rows"], dtype=np.int32)
    cols = np.asarray(Ainfo["cols"], dtype=np.int32)
    data = np.asarray(Ainfo["data"], dtype=np.float64)
    cons = doc.get("constraints", {})
    lc = np.asarray([_from_json_num(v) for v in cons.get("lc", [])], dtype=np.float64)
    uc = np.asarray([_from_json_num(v) for v in cons.get("uc", [])], dtype=np.float64)
    row_names = cons.get("names")
    vars_ = doc.get("variables", {})
    lx = np.asarray([_from_json_num(v) for v in vars_.get("lx", [])], dtype=np.float64)
    ux = np.asarray([_from_json_num(v) for v in vars_.get("ux", [])], dtype=np.float64)
    col_names = vars_.get("names")
    integer = vars_.get("integer")
    if integer is not None:
        integer = np.asarray(integer, dtype=bool)
        if not integer.any():
            integer = None
    n = int(doc.get("n", len(c)))
    m = int(doc.get("m", len(lc)))
    if n == 0:
        n = max(int(cols.max()) + 1 if cols.size else 0, len(c), len(lx))
    if m == 0:
        m = max(int(rows.max()) + 1 if rows.size else 0, len(lc))
    A = sp.csr_matrix((data, (rows, cols)), shape=(m, n))
    A.eliminate_zeros()
    Q = None
    if "Q" in doc and doc["Q"] is not None:
        q = doc["Q"]
        Q = sp.csr_matrix(
            (np.asarray(q["data"], dtype=np.float64),
             (np.asarray(q["rows"], dtype=np.int32), np.asarray(q["cols"], dtype=np.int32))),
            shape=(n, n),
        )
    if sense < 0:
        c, c0 = -c, -c0
        if Q is not None:
            Q = -Q
    return LPProblem(
        c=c, A=A, lc=lc, uc=uc, lx=lx, ux=ux, c0=c0, obj_sign=sense,
        name=name or doc.get("name") or Path(str(path)).stem,
        row_names=row_names, col_names=col_names, integer=integer, Q=Q,
    )


def write_json(prob: LPProblem, path, name: str | None = None) -> None:
    sgn = prob.obj_sign
    cost = sgn * prob.c
    c0 = sgn * prob.c0
    A = sp.coo_matrix(prob.A)
    rn, cn = prob.names()
    doc = {
        "name": name or prob.name or "LP",
        "sense": "max" if sgn < 0 else "min",
        "m": prob.m,
        "n": prob.n,
        "objective": {"c": cost.tolist(), "c0": float(c0)},
        "A": {"rows": A.row.tolist(), "cols": A.col.tolist(), "data": A.data.tolist()},
        "constraints": {
            "lc": [_finite_or_str(v) for v in prob.lc],
            "uc": [_finite_or_str(v) for v in prob.uc],
            "names": list(rn),
        },
        "variables": {
            "lx": [_finite_or_str(v) for v in prob.lx],
            "ux": [_finite_or_str(v) for v in prob.ux],
            "names": list(cn),
        },
    }
    if prob.integer is not None:
        doc["variables"]["integer"] = [bool(v) for v in prob.integer]
    if prob.Q is not None and prob.Q.nnz:
        Q = sp.coo_matrix(sgn * prob.Q)
        doc["Q"] = {"rows": Q.row.tolist(), "cols": Q.col.tolist(), "data": Q.data.tolist()}
    Path(path).write_text(json.dumps(doc, indent=2) + "\n")
