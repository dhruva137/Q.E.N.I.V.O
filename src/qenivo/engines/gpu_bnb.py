"""GPU-batched branch and bound (engine "milp-gpu"), written as a plugin.

Idea (as in B^3-PWL, arXiv 2608.28988, 2026, which batches LP relaxations of a branch-and-bound
tree on the GPU): the open nodes of a tree differ only in their variable bounds, and the batched
PDHG engine already accepts per-scenario bounds. So up to `batch` nodes are solved in ONE GPU
batch - the matrix is read once per iteration for all of them.

  * selection   best-bound, with the deepest node of each batch replaced by a dive while no
                incumbent exists
  * node LP     batched PDHG (float64, 1e-6) with in-loop infeasibility detection (Farkas rays)
  * bound       the node's certified dual objective (a valid lower bound up to the relaxation
                tolerance, stated in the result)
  * branching   most fractional
  * incumbents  rounding and fix-and-resolve (simplex on the continuous part), Feasibility Jump at
                the root; every incumbent is checked on the ORIGINAL model
The CPU branch and cut (engine "milp") stays the exact default; this engine is for large trees
on a GPU, where node throughput decides the time.
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np

from .registry import register_engine

INT_TOL = 1e-6


def solve_milp_gpu(prob, tol=1e-6, time_limit=600.0, backend="cupy", verbose=False, gap_tol=1e-4,
                   batch=64, relax_tol=1e-6, **_):
    from ..api import Solution, _finish
    from ..kernels.backend import gpu_available
    from . import pdhg
    from .fj import feasibility_jump
    from .milp import _feasible, _heuristic
    t0 = time.perf_counter()
    dev = "cupy" if (backend in ("auto", "cupy", "gpu") and gpu_available()) else "numpy"
    integer = prob.integer if prob.integer is not None else np.zeros(prob.n, bool)
    lx0, ux0 = prob.lx.copy(), prob.ux.copy()
    lx0[integer], ux0[integer] = np.ceil(lx0[integer] - INT_TOL), np.floor(ux0[integer] + INT_TOL)
    meta = {"engine": "milp-gpu", "backend": dev, "relaxation": f"batched PDHG at {relax_tol:g}",
            "reason": "GPU-batched branch and bound (plugin)"}
    obj = lambda x: float(prob.c @ x + prob.c0)  # noqa: E731
    inc_x, inc = None, math.inf
    heap = [(-math.inf, 0, 0, lx0, ux0)]
    counter, nodes, batches = 0, 0, 0
    root_done = False
    status = "optimal"

    def prune_at():
        return inc - gap_tol * max(1.0, abs(inc)) if math.isfinite(inc) else math.inf

    while heap:
        if time.perf_counter() - t0 > time_limit:
            status = "time_limit"
            break
        take = []
        while heap and len(take) < batch:
            b, _, d, lx, ux = heapq.heappop(heap)
            if b < prune_at():
                take.append((b, d, lx, ux))
        if not take:
            break
        LX = np.stack([t[2] for t in take], axis=1)
        UX = np.stack([t[3] for t in take], axis=1)
        o = pdhg.PDHGOptions(tol=relax_tol, time_limit=max(1.0, time_limit - (time.perf_counter() - t0)))
        br = pdhg.solve_batch(prob.A, prob.c, prob.lc, prob.uc, LX, UX, prob.c0, o, dev)
        batches += 1
        nodes += len(take)
        for k, (b, d, lx, ux) in enumerate(take):
            st = br.status[k]
            if st == "infeasible":
                continue                                         # proven by a checked Farkas ray
            if st != "optimal":
                counter += 1                                     # unfinished: requeue at its parent bound
                heapq.heappush(heap, (b, counter, d, lx, ux))
                continue
            x = br.x[:, k]
            bound = float(br.kkt["dobj"][k]) if "dobj" in br.kkt else obj(x)
            if bound >= prune_at():
                continue
            frac = np.abs(x - np.round(x))
            cand = np.flatnonzero(integer & (frac > INT_TOL))
            if cand.size == 0 or k < 4:                         # integral, or one of the best nodes
                z = _heuristic(prob, x, integer, lx, ux, min(10.0, time_limit))
                if z is not None and obj(z) < inc:
                    inc_x, inc = z, obj(z)
            if not root_done:
                root_done = True
                if inc_x is None:
                    fx = feasibility_jump(prob, integer, lx0, ux0, x0=x, time_limit=min(10.0, 0.05 * time_limit))
                    z = None if fx is None else _heuristic(prob, fx, integer, lx0, ux0, 10.0)
                    z = z if z is not None else (fx if fx is not None and _feasible(prob, fx) else None)
                    if z is not None and obj(z) < inc:
                        inc_x, inc = z, obj(z)
            if cand.size == 0:
                continue
            j = int(cand[np.argmax(np.minimum(frac[cand], 1 - frac[cand]))])
            for side in ("down", "up"):
                clx, cux = lx.copy(), ux.copy()
                if side == "down":
                    cux[j] = math.floor(x[j])
                else:
                    clx[j] = math.ceil(x[j])
                if clx[j] <= cux[j]:
                    counter += 1
                    key = bound if inc_x is not None else bound - 1e-9 * (d + 1)   # dive while no incumbent
                    heapq.heappush(heap, (key, counter, d + 1, clx, cux))
        if verbose:
            lb = min([h[0] for h in heap], default=inc)
            print(f"milp-gpu batch {batches}: nodes {nodes} open {len(heap)} incumbent {inc:.8g} bound {lb:.8g}")
    lb = min([h[0] for h in heap], default=inc)
    lb = min(lb, inc) if inc_x is not None else lb
    gap = (inc - lb) / max(1.0, abs(inc)) if inc_x is not None else math.inf
    meta.update(nodes=nodes, batches=batches, iterations=nodes, time=time.perf_counter() - t0, gap=gap,
                bound=prob.obj_sign * lb)
    if inc_x is None:
        return _finish(prob, "not_proven", None, None, tol, dict(meta, note="no integer point found"))
    closed = gap <= gap_tol and status == "optimal"
    ax = prob.A @ inc_x
    res = {"objective": prob.obj_sign * inc, "gap": gap, "bound": prob.obj_sign * lb,
           "max_row_violation": float(max(np.max(np.maximum(prob.lc - ax, 0), initial=0),
                                          np.max(np.maximum(ax - prob.uc, 0), initial=0))),
           "max_integrality": float(np.max(np.abs(inc_x[integer] - np.round(inc_x[integer])), initial=0))}
    return Solution(status="optimal" if closed else status,
                    verdict="optimal" if closed and _feasible(prob, inc_x) else "not_proven",
                    objective=prob.obj_sign * inc, x=inc_x, y=None, residuals=res, engine=meta, problem=prob,
                    tol=tol, extra={"mip": {"nodes": nodes, "batches": batches, "gap": gap,
                                            "bound": prob.obj_sign * lb, "incumbent_feasible": _feasible(prob, inc_x)}})


register_engine("milp-gpu", solve_milp_gpu, classes=("MILP",),
                description="GPU-batched branch and bound: open nodes solved together by batched PDHG")
