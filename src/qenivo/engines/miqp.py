"""MIQP by branch and bound over convex QP relaxations (engine "miqp"), written as a plugin.

    min 0.5 x'Qx + c'x + c0   s.t.  lc <= Ax <= uc,  lx <= x <= ux,  x_j integer (j in I),  Q PSD

This module is the worked example of the extension point in engines/registry.py: it uses only
public pieces of the core (the PDQP engine for relaxations, the model, the result type) and
registers itself; api.solve, the CLI and the server then accept engine="miqp".

  * relaxations  convex QP by PDQP, warm-started from the parent (x, y), tolerance 1e-8
  * bound        the relaxation objective; with a first-order relaxation this bound is exact
                 only to the relaxation tolerance, which is stated in the result
  * selection    depth-first until the first incumbent, then best-bound
  * branching    most fractional
  * heuristic    round and fix the integers, re-solve the continuous QP
  * incumbent    accepted only after the float64 check on the ORIGINAL model (rows, bounds,
                 integrality 1e-6); the objective is recomputed exactly from x
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np

from ..model import Problem
from .registry import register_engine

INT_TOL = 1e-6
RELAX_TOL = 1e-8


def _objective(prob, x):
    return float(0.5 * x @ (prob.Q @ x) + prob.c @ x + prob.c0)


def _feasible(prob, x, tol=1e-6):
    if np.any(x < prob.lx - tol) or np.any(x > prob.ux + tol):
        return False
    ax = prob.A @ x
    s = 1 + np.abs(ax)
    return bool(np.all(ax >= prob.lc - tol * s) and np.all(ax <= prob.uc + tol * s))


def _relax(prob, lx, ux, time_limit, warm=None):
    from .pdqp import solve_qp
    sub = Problem(c=prob.c, A=prob.A, lc=prob.lc, uc=prob.uc, lx=lx, ux=ux, c0=prob.c0, Q=prob.Q, name=prob.name)
    out = solve_qp(sub, tol=RELAX_TOL, time_limit=time_limit,
                   warm_x=None if warm is None else np.clip(warm[0], lx, ux),
                   warm_y=None if warm is None else warm[1])
    return out


def solve_miqp(prob, tol=1e-6, time_limit=600.0, backend="numpy", verbose=False, gap_tol=1e-6,
               node_limit=50_000, **_):
    from ..api import Solution, _finish
    t0 = time.perf_counter()
    integer = prob.integer if prob.integer is not None else np.zeros(prob.n, bool)
    lx0, ux0 = prob.lx.copy(), prob.ux.copy()
    lx0[integer], ux0[integer] = np.ceil(lx0[integer] - INT_TOL), np.floor(ux0[integer] + INT_TOL)
    meta = {"engine": "miqp", "backend": "numpy", "relaxation": f"PDQP at {RELAX_TOL:g}",
            "reason": "integer variables and a quadratic objective"}
    root = _relax(prob, lx0, ux0, time_limit)
    if root["status"] != "optimal":
        return _finish(prob, "not_proven", None, None, tol, dict(meta, nodes=0, iterations=0,
                       time=time.perf_counter() - t0, note=f"root relaxation {root['status']}"))
    heap = [(_objective(prob, root["x"]), 0, 0, lx0, ux0, root)]
    counter, nodes = 0, 0
    inc_x, inc = None, math.inf
    status = "optimal"

    def try_round(x, lx, ux):
        xi = np.clip(np.round(x[integer]), lx[integer], ux[integer])
        flx, fux = lx.copy(), ux.copy()
        flx[integer], fux[integer] = xi, xi
        out = _relax(prob, flx, fux, min(10.0, time_limit))
        if out["status"] != "optimal":
            return None
        z = out["x"].copy()
        z[integer] = xi
        return z if _feasible(prob, z) else None

    z = try_round(root["x"], lx0, ux0)
    if z is not None:
        inc_x, inc = z, _objective(prob, z)
    while heap:
        if time.perf_counter() - t0 > time_limit:
            status = "time_limit"
            break
        if nodes >= node_limit:
            status = "node_limit"
            break
        if inc_x is None:
            k = max(range(len(heap)), key=lambda i: heap[i][2])
            bound, _, depth, lx, ux, rel = heap.pop(k)
            heapq.heapify(heap)
        else:
            bound, _, depth, lx, ux, rel = heapq.heappop(heap)
        if math.isfinite(inc) and bound >= inc - gap_tol * max(1.0, abs(inc)):
            continue
        nodes += 1
        x = rel["x"]
        frac = np.abs(x - np.round(x))
        cand = np.flatnonzero(integer & (frac > INT_TOL))
        if cand.size == 0:
            xr = x.copy()
            xr[integer] = np.round(xr[integer])
            if _feasible(prob, xr) and _objective(prob, xr) < inc:
                inc_x, inc = xr, _objective(prob, xr)
            continue
        if nodes % 20 == 0:
            z = try_round(x, lx, ux)
            if z is not None and _objective(prob, z) < inc:
                inc_x, inc = z, _objective(prob, z)
        j = int(cand[np.argmax(np.minimum(frac[cand], 1 - frac[cand]))])
        for side in ("down", "up"):
            clx, cux = lx.copy(), ux.copy()
            if side == "down":
                cux[j] = math.floor(x[j])
            else:
                clx[j] = math.ceil(x[j])
            if clx[j] > cux[j]:
                continue
            ch = _relax(prob, clx, cux, max(1.0, time_limit - (time.perf_counter() - t0)), warm=(x, rel["y"]))
            if ch["status"] != "optimal":
                continue
            cb = _objective(prob, ch["x"])
            if not math.isfinite(inc) or cb < inc - gap_tol * max(1.0, abs(inc)):
                counter += 1
                heapq.heappush(heap, (cb, counter, depth + 1, clx, cux, ch))
        if verbose and nodes % 50 == 0:
            print(f"miqp nodes {nodes} open {len(heap)} incumbent {inc:.8g}")
    lb = min([h[0] for h in heap], default=inc)
    lb = min(lb, inc) if inc_x is not None else lb
    gap = (inc - lb) / max(1.0, abs(inc)) if inc_x is not None else math.inf
    meta.update(nodes=nodes, iterations=nodes, time=time.perf_counter() - t0, gap=gap, bound=prob.obj_sign * lb)
    if inc_x is None:
        return _finish(prob, "not_proven", None, None, tol, dict(meta, note="no integer point found"))
    closed = (not heap or gap <= gap_tol) and status == "optimal"
    ax = prob.A @ inc_x
    res = {"objective": prob.obj_sign * inc, "max_row_violation": float(max(np.max(np.maximum(prob.lc - ax, 0), initial=0),
                                                                        np.max(np.maximum(ax - prob.uc, 0), initial=0))),
           "max_integrality": float(np.max(np.abs(inc_x[integer] - np.round(inc_x[integer])), initial=0)),
           "gap": gap, "bound": prob.obj_sign * lb}
    return Solution(status="optimal" if closed else status,
                    verdict="optimal" if closed and _feasible(prob, inc_x) else "not_proven",
                    objective=prob.obj_sign * inc, x=inc_x, y=None, residuals=res, engine=meta, problem=prob,
                    tol=tol, extra={"mip": {"nodes": nodes, "gap": gap, "bound": prob.obj_sign * lb,
                                            "incumbent_feasible": _feasible(prob, inc_x)}})


register_engine("miqp", solve_miqp, classes=("MIQP", "QP", "MILP"),
                description="branch and bound over convex QP relaxations (PDQP); plugin example")
