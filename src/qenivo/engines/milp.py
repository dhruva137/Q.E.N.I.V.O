"""MILP by LP-based branch and cut (engine "milp").

  * presolve          fixed/empty columns, singleton rows, activity-based integer bound
                      tightening, redundant rows (engines/presolve.py); postsolve before the check
  * root cuts         rounds of Gomory mixed-integer cuts from the optimal simplex tableau and
                      knapsack cover cuts (engines/cuts.py), until the bound stops moving
  * node LPs          simplex warm-started from the parent's basis (dual simplex after a bound
                      change), interior point for larger nodes
  * node selection    depth-first until the first incumbent, then best-bound with periodic
                      dives (a best-estimate/best-bound hybrid, Achterberg 2007 ch. 6)
  * branching         pseudocost branching, initialised by most-fractional; pseudocosts are
                      learnt from the objective change of every solved child
  * primal heuristics rounding + fix-and-resolve, and fractional diving, at the root and
                      periodically in the tree; Feasibility Jump (engines/fj.py) when the root
                      has no incumbent
  * incumbents        accepted only after the integer point passes the feasibility check on the
                      ORIGINAL model (bounds, rows, integrality 1e-6)
  * bound             the global lower bound is the minimum LP bound over open nodes; the
                      reported gap is (incumbent - bound) / max(1, |incumbent|)
Scope: a certified branch and cut for small and medium models (node LPs use the dense-LU
simplex); parallel tree search is the next milestone (docs/ROADMAP.md).
"""
from __future__ import annotations

import copy
import heapq
import math
import time

import numpy as np
import scipy.sparse as sp

from ..model import Problem

INT_TOL = 1e-6


def _lp(prob: Problem, lx, ux, time_limit, start=None):
    sub = copy.copy(prob)            # shares A (no re-validation, no copy): only the bounds differ per node
    sub.lx, sub.ux, sub.integer = np.asarray(lx, dtype=float), np.asarray(ux, dtype=float), None
    if sub.m <= 1500:
        from .simplex import solve_simplex
        ok = bool(start) and "col_statuses" in start and len(start["row_statuses"]) == sub.m
        out = solve_simplex(sub, tol=1e-9, time_limit=time_limit, start=start if ok else None)
        if out["status"] in ("optimal", "infeasible", "unbounded"):
            return out
    from .ipm import solve_ipm
    return solve_ipm(sub, tol=1e-9, time_limit=time_limit)


RELIABLE = 4          # pseudocost observations per direction before a column is trusted
STRONG_CANDS = 8      # unreliable candidates tested by strong branching at a node
STRONG_NODES = 10**9  # strong branching (on unreliable candidates only) at every node


def _child(prob, lx, ux, j, side, v, lp, time_limit):
    clx, cux = lx.copy(), ux.copy()
    if side == "down":
        cux[j] = math.floor(v)
    else:
        clx[j] = math.ceil(v)
    if clx[j] > cux[j]:
        return clx, cux, None
    return clx, cux, _lp(prob, clx, cux, time_limit, start=lp)


def _learn(j, side, v, gain, pc_dn, pc_up, n_dn, n_up):
    if side == "down":
        n_dn[j] += 1
        pc_dn[j] += (gain / max(v - math.floor(v), 1e-9) - pc_dn[j]) / n_dn[j]
    else:
        n_up[j] += 1
        pc_up[j] += (gain / max(math.ceil(v) - v, 1e-9) - pc_up[j]) / n_up[j]


def _branch(prob, x, lp, bound, cand, lx, ux, pc_dn, pc_up, n_dn, n_up, time_limit, strong=True):
    """Reliability branching (Achterberg, Koch & Martin 2005): pseudocost scores, with strong
    branching on the best-scored candidates whose pseudocosts are not yet reliable. Returns the
    children [(bound, lx, ux, lp)] of the chosen column; an infeasible side yields no child."""
    t0 = time.perf_counter()
    f = x[cand] - np.floor(x[cand])
    ps = np.maximum(f * pc_dn[cand], 1e-6) * np.maximum((1 - f) * pc_up[cand], 1e-6)
    order = np.argsort(-ps)
    best, best_score, tested = None, -1.0, {}
    if strong:
        unreliable = [k for k in order if n_dn[cand[k]] < RELIABLE or n_up[cand[k]] < RELIABLE][:STRONG_CANDS]
        for k in unreliable:
            j, v = int(cand[k]), float(x[cand[k]])
            sides, gains = [], []
            for side in ("down", "up"):
                clx, cux, ch = _child(prob, lx, ux, j, side, v, lp, max(1.0, time_limit - (time.perf_counter() - t0)))
                if ch is not None and ch["status"] == "optimal":
                    cobj = float(prob.c @ ch["x"] + prob.c0)
                    gain = max(cobj - bound, 0.0)
                    _learn(j, side, v, gain, pc_dn, pc_up, n_dn, n_up)
                    sides.append((cobj, clx, cux, ch))
                    gains.append(gain)
                else:
                    gains.append(math.inf)                 # infeasible side: the column is decisive
            tested[j] = sides
            g_dn, g_up = gains
            sc = max(min(g_dn, 1e12), 1e-6) * max(min(g_up, 1e12), 1e-6)
            if not sides:
                return []                                  # both sides infeasible: prune the node
            if sc > best_score:
                best, best_score = j, sc
    if best is None:
        best = int(cand[order[0]])
    if best in tested:
        return tested[best]
    v = float(x[best])
    out = []
    for side in ("down", "up"):
        clx, cux, ch = _child(prob, lx, ux, best, side, v, lp, time_limit)
        if ch is None or ch["status"] != "optimal":
            continue
        cobj = float(prob.c @ ch["x"] + prob.c0)
        _learn(best, side, v, max(cobj - bound, 0.0), pc_dn, pc_up, n_dn, n_up)
        out.append((cobj, clx, cux, ch))
    return out


def _redcost_fix(prob, lp, bound, lx, ux, integer, cutoff):
    """Reduced cost fixing: an integer column nonbasic at its lower bound with reduced cost d > 0 can
    rise by at most (cutoff - bound) / d before the node's LP bound reaches the cutoff (mirror at the
    upper bound). Returns tightened copies (lx, ux) and the number of changed bounds."""
    if not math.isfinite(cutoff) or "y" not in lp or "col_statuses" not in lp:
        return lx, ux, 0
    d = prob.c - prob.A.T @ lp["y"]
    st = np.asarray(lp["col_statuses"])
    room = cutoff - bound
    if room <= 0:
        return lx, ux, 0
    lo = integer & (st == "at_lower") & (d > 1e-7) & np.isfinite(lx)
    up = integer & (st == "at_upper") & (d < -1e-7) & np.isfinite(ux)
    nlx, nux = lx, ux
    cnt = 0
    if lo.any():
        cap = lx[lo] + np.floor(room / d[lo] + 1e-6)
        tight = cap < ux[lo]
        if tight.any():
            nux = ux.copy()
            idx = np.flatnonzero(lo)[tight]
            nux[idx] = cap[tight]
            cnt += int(tight.sum())
    if up.any():
        cap = ux[up] - np.floor(room / -d[up] + 1e-6)
        tight = cap > lx[up]
        if tight.any():
            nlx = lx.copy()
            idx = np.flatnonzero(up)[tight]
            nlx[idx] = cap[tight]
            cnt += int(tight.sum())
    return nlx, nux, cnt


def _prune_at(inc_obj, gap_tol):
    """Nodes whose bound reaches this value cannot improve the incumbent (inf without one)."""
    return inc_obj - gap_tol * max(1.0, abs(inc_obj)) if math.isfinite(inc_obj) else math.inf


def _feasible(prob, x, tol=1e-6) -> bool:
    if np.any(x < prob.lx - tol) or np.any(x > prob.ux + tol):
        return False
    ax = prob.A @ x
    scale = 1 + np.abs(ax)
    return bool(np.all(ax >= prob.lc - tol * scale) and np.all(ax <= prob.uc + tol * scale))


def _heuristic(prob, x, integer, lx, ux, time_limit, start=None):
    """Round the integers of an LP point, fix them, re-solve the continuous rest (warm-started
    from the node's basis, so an infeasible rounding is detected in a few dual pivots)."""
    time_limit = min(time_limit, 5.0)
    xi = np.clip(np.round(x[integer]), lx[integer], ux[integer])
    flx, fux = lx.copy(), ux.copy()
    flx[integer], fux[integer] = xi, xi
    out = _lp(prob, flx, fux, time_limit, start=start)
    if out["status"] != "optimal":
        return None
    z = out["x"].copy()
    z[integer] = xi
    return z if _feasible(prob, z) else None


def _dive(prob, lp, integer, lx, ux, time_limit, max_depth=40):
    """Fractional diving: fix the least fractional integer to its rounding, re-solve, repeat."""
    t0 = time.perf_counter()
    lx, ux, cur = lx.copy(), ux.copy(), lp
    for _ in range(max_depth):
        x = cur["x"]
        frac = np.abs(x - np.round(x))
        cand = np.flatnonzero(integer & (frac > INT_TOL))
        if cand.size == 0:
            return x.copy() if _feasible(prob, x) else None
        j = int(cand[np.argmin(frac[cand])])
        v = min(max(float(np.round(x[j])), lx[j]), ux[j])
        lx[j] = ux[j] = v
        cur = _lp(prob, lx, ux, max(1.0, time_limit - (time.perf_counter() - t0)), start=cur)
        if cur["status"] != "optimal" or time.perf_counter() - t0 > time_limit:
            return None
    return None


CUT_MIN_EFFICACY = 1e-5    # Euclidean distance of the LP point from the cut
CUT_MAX_PARALLEL = 0.98    # |cos| above which a new cut duplicates an accepted one
CUT_MAX_AGE = 3            # rounds a cut may stay non-binding before it leaves the root LP
CUTS_PER_ROUND = 100
DROP_LOOSE_CUTS = True     # the tree starts from the root LP without its non-binding cuts


def _select_cuts(blocks, x, accepted=None, max_cuts=CUTS_PER_ROUND):
    """Cut management: efficacy threshold and parallelism filter. `blocks` are (P, rhs) for P x >= rhs;
    `accepted` the unit-normalised rows of cuts already in the LP. Returns (P, rhs) or None."""
    if not blocks:
        return None
    P = sp.vstack([b[0] for b in blocks], format="csr")
    rhs = np.concatenate([b[1] for b in blocks])
    nrm = np.sqrt(np.asarray(P.multiply(P).sum(axis=1)).ravel())
    eff = (rhs - P @ x) / np.maximum(nrm, 1e-12)
    order = [k for k in np.argsort(-eff) if eff[k] > CUT_MIN_EFFICACY]
    D = P.toarray() / np.maximum(nrm, 1e-12)[:, None]
    keep, K = [], []
    ref = accepted if accepted is not None and accepted.shape[0] else None
    for k in order:
        d = D[k]
        if K and np.max(np.abs(np.array(K) @ d)) > CUT_MAX_PARALLEL:
            continue
        if ref is not None and np.max(np.abs(ref @ d)) > CUT_MAX_PARALLEL:
            continue
        keep.append(k)
        K.append(d)
        if len(keep) >= max_cuts:
            break
    if not keep:
        return None
    keep = np.array(keep)
    return P[keep], rhs[keep]


def _with_rows(cur, keep_rows=None, add=None):
    """The problem with only `keep_rows` of its rows and the cut block `add` (P x >= rhs) appended."""
    A, lc, uc = cur.A, cur.lc, cur.uc
    if keep_rows is not None:
        A, lc, uc = A[keep_rows], lc[keep_rows], uc[keep_rows]
    if add is not None:
        A = sp.vstack([A, add[0]], format="csr")
        lc = np.concatenate([lc, add[1]])
        uc = np.concatenate([uc, np.full(len(add[1]), np.inf)])
    return Problem(c=cur.c, A=A, lc=lc, uc=uc, lx=cur.lx, ux=cur.ux, c0=cur.c0, obj_sign=cur.obj_sign,
                   name=cur.name, integer=cur.integer)


def _root_cuts(prob, integer, lx, ux, root, time_limit, rounds=20, verbose=False):
    """Rounds of GMI, knapsack cover and c-MIR (with variable bound substitution: flow covers) cuts at
    the root, with cut management: an efficacy threshold, a parallelism filter, and aging (a cut that
    stays non-binding for CUT_MAX_AGE rounds leaves the LP; non-binding cuts are dropped at the end).
    Returns (prob with cuts, root LP, info)."""
    from .cuts import cover_cuts, gmi_cuts, mir_cuts
    t0 = time.perf_counter()
    info = {"rounds": 0, "gmi": 0, "cover": 0, "mir": 0, "bound_before": float(prob.c @ root["x"] + prob.c0)}
    m0 = prob.m
    cur, lp = prob, root
    age = np.zeros(0)
    last = info["bound_before"]
    stalled = 0
    for _ in range(rounds):
        if time.perf_counter() - t0 > time_limit or cur.m > 1500:
            break
        blocks = []
        for key, sep in (("gmi", lambda: gmi_cuts(cur, lp, integer, lx, ux)),
                         ("cover", lambda: cover_cuts(cur, lp["x"], integer, lx, ux)),
                         ("mir", lambda: mir_cuts(cur, lp["x"], integer, lx, ux))):
            r = sep()
            if r is not None:
                blocks.append(r)
                info[key] += r[0].shape[0]
        if cur.m > m0:
            C = cur.A[m0:]
            cn = np.sqrt(np.asarray(C.multiply(C).sum(axis=1)).ravel())
            acc = C.toarray() / np.maximum(cn, 1e-12)[:, None]
        else:
            acc = None
        sel = _select_cuts(blocks, lp["x"], acc)
        if sel is None:
            break
        nxt = _with_rows(cur, add=sel)
        # warm start: the old basis plus the new cut slacks (basic) is dual feasible
        start = {"col_statuses": lp["col_statuses"], "row_statuses": list(lp["row_statuses"]) + ["basic"] * len(sel[1])}             if "col_statuses" in lp else None
        out = _lp(nxt, lx, ux, max(1.0, time_limit - (time.perf_counter() - t0)), start=start)
        if out["status"] != "optimal":
            break                                   # a cut round that breaks the LP is discarded
        bound = float(nxt.c @ out["x"] + nxt.c0)
        if bound < last - 1e-7 * max(1.0, abs(last)):
            break                                   # cuts may only raise the bound: numerical trouble
        info["rounds"] += 1
        info["added"] = info.get("added", 0) + len(sel[1])
        cur, lp = nxt, out
        # aging: cuts whose slack is basic and strictly inside stay older; old ones leave the LP
        age = np.concatenate([age, np.zeros(len(sel[1]))])
        slack = cur.A[m0:] @ lp["x"] - cur.lc[m0:]
        rs = np.array(lp.get("row_statuses", ["basic"] * cur.m))[m0:]
        loose = (rs == "basic") & (slack > 1e-6 * (1 + np.abs(cur.lc[m0:])))
        age = np.where(loose, age + 1, 0)
        old = loose & (age >= CUT_MAX_AGE)
        if old.any():
            cur, lp = _drop_cuts(cur, lp, m0, ~old, lx, ux, time_limit - (time.perf_counter() - t0))
            age = age[~old]
            info["aged_out"] = info.get("aged_out", 0) + int(old.sum())
        if verbose:
            print(f"cut round {info['rounds']}: +{len(sel[1])} cuts ({cur.m - m0} in LP)  bound {bound:.10g}")
        if bound - last <= 1e-4 * max(1.0, abs(bound)):   # stop after two rounds without real progress
            stalled += 1
            if stalled >= 2:
                break
        else:
            stalled = 0
        last = bound
    if DROP_LOOSE_CUTS and cur.m > m0:              # the tree keeps only the cuts binding at the root
        slack = cur.A[m0:] @ lp["x"] - cur.lc[m0:]
        rs = np.array(lp.get("row_statuses", ["basic"] * cur.m))[m0:]
        loose = (rs == "basic") & (slack > 1e-6 * (1 + np.abs(cur.lc[m0:])))
        if loose.any():
            cur, lp = _drop_cuts(cur, lp, m0, ~loose, lx, ux, max(1.0, time_limit - (time.perf_counter() - t0)))
    info["in_lp"] = cur.m - m0
    info["bound_after"] = float(cur.c @ lp["x"] + cur.c0)
    return cur, lp, info


def _drop_cuts(cur, lp, m0, keep_cut, lx, ux, time_limit):
    """Remove the cut rows m0 + k with keep_cut[k] False (their slacks are basic, so the basis of the
    remaining rows stays valid) and re-solve from that basis (no pivots)."""
    keep = np.concatenate([np.arange(m0), m0 + np.flatnonzero(keep_cut)])
    nxt = _with_rows(cur, keep_rows=keep)
    start = None
    if "col_statuses" in lp:
        start = {"col_statuses": lp["col_statuses"], "row_statuses": [lp["row_statuses"][i] for i in keep]}
    out = _lp(nxt, lx, ux, max(1.0, time_limit), start=start)
    if out["status"] != "optimal":
        return cur, lp
    return nxt, out


def solve_milp(prob: Problem, tol: float = 1e-6, time_limit: float = 600.0, gap_tol: float = 1e-6,
               node_limit: int = 200_000, verbose: bool = False, meta: dict | None = None,
               source: str | None = None, presolve: bool = True, cuts: bool = True):
    """Branch and cut. The answer is always checked on the ORIGINAL model."""
    t0 = time.perf_counter()
    orig, pre = prob, None
    if presolve and prob.integer is not None:
        from .presolve import presolve as _presolve
        pre = _presolve(prob)
        if pre.status == "reduced" and pre.prob.n > 0 and pre.prob.m > 0:
            prob = pre.prob
        else:
            pre = None
    return _solve_tree(prob, orig, pre, tol, time_limit, gap_tol, node_limit, verbose, meta, source, cuts, t0)


def _solve_tree(prob, orig, pre, tol, time_limit, gap_tol, node_limit, verbose, meta, source, cuts, t0):
    from ..api import _finish
    integer = prob.integer if prob.integer is not None else np.zeros(prob.n, bool)
    lx0, ux0 = prob.lx.copy(), prob.ux.copy()
    lx0[integer], ux0[integer] = np.ceil(lx0[integer] - INT_TOL), np.floor(ux0[integer] + INT_TOL)
    meta = dict(meta or {}, engine="milp", backend="numpy")

    inc_x, inc_obj = None, math.inf
    pc_up = np.ones(prob.n)
    pc_dn = np.ones(prob.n)
    n_up = np.zeros(prob.n)
    n_dn = np.zeros(prob.n)
    heap = []
    counter = 0
    nodes = 0
    root = _lp(prob, lx0, ux0, time_limit)
    if root["status"] == "infeasible":
        from ..api import _certify_infeasible
        sol = _certify_infeasible(orig, tol, time_limit, dict(meta, nodes=0, reason="LP relaxation infeasible"), source)
        if sol is not None:
            return sol
    if root["status"] != "optimal":
        return _finish(orig, "not_proven", None, None, tol, dict(meta, nodes=0, iterations=0,
                       time=time.perf_counter() - t0, note=f"root LP {root['status']}"), source=source)
    cut_info = None
    if cuts and integer.any():
        prob, root, cut_info = _root_cuts(prob, integer, lx0, ux0, root, 0.2 * time_limit, verbose=verbose)
    root_obj = float(prob.c @ root["x"] + prob.c0)
    heapq.heappush(heap, (root_obj, (0, counter), 0, lx0, ux0, root))
    for z in (_heuristic(prob, root["x"], integer, lx0, ux0, time_limit, start=root),
              _dive(prob, root, integer, lx0, ux0, 0.1 * time_limit)):
        if z is not None and float(prob.c @ z + prob.c0) < inc_obj:
            inc_x, inc_obj = z, float(prob.c @ z + prob.c0)
    if inc_x is None:                     # no incumbent yet: Feasibility Jump, then fix and solve the LP
        from .fj import feasibility_jump
        fx = feasibility_jump(prob, integer, lx0, ux0, x0=root["x"], time_limit=min(10.0, 0.05 * time_limit))
        if fx is not None:
            z = _heuristic(prob, fx, integer, lx0, ux0, time_limit, start=root)
            z = z if z is not None else (fx if _feasible(prob, fx) else None)
            if z is not None:
                inc_x, inc_obj = z, float(prob.c @ z + prob.c0)
    dive = inc_x is None
    global_lb = root_obj
    glx = gux = None
    root_fix_at = math.inf
    redcost_fixed = 0
    status = "optimal"
    while heap:
        if time.perf_counter() - t0 > time_limit:
            status = "time_limit"
            break
        if nodes >= node_limit:
            status = "node_limit"
            break
        if dive:
            idx = max(range(len(heap)), key=lambda k: heap[k][2])     # deepest node
            bound, _, depth, lx, ux, lp = heap.pop(idx)
            heapq.heapify(heap)
        else:
            bound, _, depth, lx, ux, lp = heapq.heappop(heap)
        global_lb = min([bound] + [h[0] for h in heap])
        if bound >= _prune_at(inc_obj, gap_tol):
            continue
        nodes += 1
        x = lp["x"]
        frac = np.abs(x - np.round(x))
        cand = np.flatnonzero(integer & (frac > INT_TOL))
        if cand.size == 0:
            if _feasible(prob, x) and bound < inc_obj:
                inc_x, inc_obj, dive = x.copy(), bound, False
            continue
        if math.isfinite(inc_obj):
            if glx is not None:                        # root reduced cost fixing, applied to the node
                nlx, nux = np.maximum(lx, glx), np.minimum(ux, gux)
                if np.any(nlx > nux):
                    continue
                if np.any(x < nlx - 1e-9) or np.any(x > nux + 1e-9):
                    rl = _lp(prob, nlx, nux, max(1.0, time_limit - (time.perf_counter() - t0)), start=lp)
                    if rl["status"] != "optimal":
                        continue
                    lp, x, lx, ux = rl, rl["x"], nlx, nux
                    bound = max(bound, float(prob.c @ x + prob.c0))
                    if bound >= _prune_at(inc_obj, gap_tol):
                        continue
                    frac = np.abs(x - np.round(x))
                    cand = np.flatnonzero(integer & (frac > INT_TOL))
                    if cand.size == 0:
                        if _feasible(prob, x) and bound < inc_obj:
                            inc_x, inc_obj, dive = x.copy(), bound, False
                        continue
                else:
                    lx, ux = nlx, nux
            lx, ux, nfix = _redcost_fix(prob, lp, bound, lx, ux, integer, _prune_at(inc_obj, gap_tol))
            redcost_fixed += nfix
            if inc_obj < root_fix_at:                  # a better incumbent: redo the root fixing
                root_fix_at = inc_obj
                glx, gux, _ = _redcost_fix(prob, root, root_obj, lx0, ux0, integer, _prune_at(inc_obj, gap_tol))
        if nodes % 25 == 0:
            zs = [_heuristic(prob, x, integer, lx, ux, time_limit, start=lp)]
            if nodes % 100 == 0:
                zs.append(_dive(prob, lp, integer, lx, ux, 0.02 * time_limit))
            for z in zs:
                if z is not None and float(prob.c @ z + prob.c0) < inc_obj:
                    inc_x, inc_obj, dive = z, float(prob.c @ z + prob.c0), False
        remaining = max(1.0, time_limit - (time.perf_counter() - t0))
        children = _branch(prob, x, lp, bound, cand, lx, ux, pc_dn, pc_up, n_dn, n_up, remaining,
                           strong=nodes <= STRONG_NODES)
        for cobj, clx, cux, child in children:
            if cobj < _prune_at(inc_obj, gap_tol):
                counter += 1
                heapq.heappush(heap, (cobj, (-(depth + 1), counter), depth + 1, clx, cux, child))
        if verbose and nodes % 100 == 0:
            print(f"nodes {nodes:7d}  open {len(heap):6d}  lb {global_lb:.8g}  inc {inc_obj:.8g}")
        if not dive and nodes % 50 == 0:
            dive = True
        elif dive and inc_x is not None and nodes % 50 == 10:
            dive = False
    lb = min([h[0] for h in heap], default=inc_obj if inc_x is not None else global_lb)
    lb = min(lb, inc_obj) if inc_x is not None else lb
    gap = (inc_obj - lb) / max(1.0, abs(inc_obj)) if inc_x is not None else math.inf
    meta.update(nodes=nodes, iterations=nodes, time=time.perf_counter() - t0,
                bound=prob.obj_sign * lb, gap=gap, redcost_fixed=redcost_fixed)
    if pre is not None:
        meta["presolve"] = pre.log
    if cut_info is not None:
        meta["cuts"] = cut_info
    if inc_x is not None:
        inc_x = inc_x[:prob.n]
        if pre is not None:
            inc_x = pre.postsolve(inc_x, orig.n)
    prob = orig
    integer = prob.integer if prob.integer is not None else np.zeros(prob.n, bool)
    if inc_x is None:
        if not heap and status == "optimal":
            from ..api import _certify_infeasible
            sol = _certify_infeasible(prob, tol, time_limit, dict(meta, reason="tree exhausted, no integer point"), source)
            if sol is not None:
                sol.extra["note"] = "LP-relaxation infeasibility certificate"
                return sol
            meta["note"] = "tree exhausted without an integer point (integer infeasible)"
        return _finish(prob, "not_proven", None, None, tol, meta, source=source)
    x = inc_x
    closed = not heap or gap <= gap_tol
    res = {"objective": prob.objective(x), "rel_primal": 0.0, "max_rel": 0.0}
    from ..api import Solution
    ax = prob.A @ x
    viol = max(float(np.max(np.maximum(prob.lc - ax, 0), initial=0)), float(np.max(np.maximum(ax - prob.uc, 0), initial=0)))
    res.update(max_row_violation=viol, max_integrality=float(np.max(np.abs(x[integer] - np.round(x[integer])), initial=0)),
               gap=gap, bound=prob.obj_sign * lb)
    verdict = "optimal" if (closed and status == "optimal" and _feasible(prob, x)) else "not_proven"
    return Solution(status="optimal" if closed else status, verdict=verdict, objective=prob.objective(x),
                    x=x, y=None, residuals=res, engine=meta, problem=prob, tol=tol, source=source,
                    extra={"mip": {"nodes": nodes, "gap": gap, "bound": prob.obj_sign * lb,
                                   "incumbent_feasible": _feasible(prob, x)}})
