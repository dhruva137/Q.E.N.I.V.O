"""Crossover: from a first-order (PDHG) point to an exact optimal vertex with the native simplex.

Why: a first-order method reaches 1e-3..1e-4 quickly but its iteration count explodes near 1e-6
(measured: Linf_520c needs 473k PDHG iterations at 1e-6). Industrial solvers stop the first-order
method early and finish with simplex crossover (Gurobi 13's concurrent crossover, arXiv 2510.24429;
"Backing PDHG into a corner", arXiv 2511.13894; NVIDIA cuOpt; COPT). Here the finish is the native
C++ simplex and the result is an exact vertex: primal, duals and basis, so certificates, shadow
prices and ranging come out exact.

Method ("reduce, solve, verify, repair"), on the standard form z = (x, s), s = A x, with the PDHG
reduced costs d = (c - A'y, y):
  1. Dual evidence fixes variables: a column (or row slack) near a bound whose reduced cost has
     the sign that holds it there, by a clear margin, is fixed at that bound. Primal evidence drops
     rows: an inequality row with a negligible dual whose activity is well inside its bounds is
     left out. Fixing restricts, dropping relaxes; the reduced LP is usually a fraction of the size.
  2. The reduced LP is solved by the native simplex, warm-started from a basis guessed from the
     PDHG point (see `basis_from_point`) or, if that basis is singular, from the slack basis.
  3. The reduced vertex is lifted (fixed columns at their bounds, dropped rows' duals zero, their
     slacks basic) and checked on the full model: every dropped row must hold and every fixed
     variable must price out with the right sign. Then the lifted basis is optimal for the full LP.
  4. Otherwise the offending rows are restored and the offending variables freed, and the reduced
     LP is re-solved warm from the lifted basis (a few simplex pivots). After `rounds` repairs the
     full LP is solved warm from the last basis, so the answer is always an exact vertex.
Every answer is scored by the float64 KKT test on the original model (certify.kkt).
"""
from __future__ import annotations

import time

import numpy as np

from ..certify.kkt import kkt_residuals
from ..model import Problem


# ------------------------------------------------------------------ helpers
def _simplex(prob, tol, time_limit, start=None):
    """Native simplex when available (no silent drop to the slow Python engine), else Python."""
    from .native_simplex import library, solve_simplex_native
    if library() is not None:
        return solve_simplex_native(prob, tol=tol, time_limit=time_limit, start=start)
    from .simplex import solve_simplex_python
    return solve_simplex_python(prob, tol=tol, time_limit=time_limit, start=start)


def _std_bounds(prob):
    return np.concatenate([prob.lx, prob.lc]), np.concatenate([prob.ux, prob.uc])


def _nonbasic_side(lo, hi, near_lo):
    st = np.where(near_lo, "at_lower", "at_upper").astype(object)
    st[(st == "at_lower") & ~np.isfinite(lo)] = "at_upper"
    st[(st == "at_upper") & ~np.isfinite(hi)] = "at_lower"
    st[~np.isfinite(lo) & ~np.isfinite(hi)] = "free"
    return st


try:                                                     # optional: compiles the crash loop
    from numba import njit as _njit
except Exception:                                        # pragma: no cover - plain Python fallback
    def _njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]


@_njit(cache=True)
def _crash_lu(m, n, Ap, Ai, Ax, order, rowcount, piv_rel, max_lnz):
    """Rank-revealing crash: accept columns of [A -I] in priority `order`, keeping the accepted set
    independent. Left-looking sparse LU (Gilbert-Peierls reach, threshold pivoting 0.1 with the
    sparsest row); a column whose largest remaining entry is below piv_rel * its max is dependent
    and skipped. Returns (accepted mask over n+m, step_of_row) - rows left unpivoted take slacks."""
    N = n + m
    acc = np.zeros(N, dtype=np.bool_)
    step_of_row = -np.ones(m, dtype=np.int64)
    prow = np.empty(m, dtype=np.int64)
    Lp = np.zeros(m + 1, dtype=np.int64)
    cap = 1 << 16
    Li = np.empty(cap, dtype=np.int64)
    Lx = np.empty(cap, dtype=np.float64)
    w = np.zeros(m)
    mark = np.zeros(m, dtype=np.int64)
    stamp = 0
    pat = np.empty(m, dtype=np.int64)
    stack = np.empty(m, dtype=np.int64)
    smark = np.zeros(m, dtype=np.int64)
    steps = np.empty(m, dtype=np.int64)
    k = 0
    lnz = 0
    for t in range(order.shape[0]):
        if k >= m:
            break
        j = order[t]
        stamp += 1
        np_ = 0
        cmax = 0.0
        if j < n:
            for q in range(Ap[j], Ap[j + 1]):
                r = Ai[q]
                w[r] += Ax[q]
                if mark[r] != stamp:
                    mark[r] = stamp
                    pat[np_] = r
                    np_ += 1
                cmax = max(cmax, abs(Ax[q]))
        else:
            r = j - n
            w[r] -= 1.0
            mark[r] = stamp
            pat[0] = r
            np_ = 1
            cmax = 1.0
        if cmax == 0.0:
            for a in range(np_):
                w[pat[a]] = 0.0
            continue
        # reach: pivoted steps touched (transitively through L columns); dependencies always
        # run from an earlier step to a later one, so ascending step order is a valid order
        ns = 0
        sp_ = 0
        for a in range(np_):
            s0 = step_of_row[pat[a]]
            if s0 >= 0 and smark[s0] != stamp:
                smark[s0] = stamp
                stack[sp_] = s0
                sp_ += 1
        while sp_ > 0:
            sp_ -= 1
            s0 = stack[sp_]
            steps[ns] = s0
            ns += 1
            for q in range(Lp[s0], Lp[s0 + 1]):
                r = Li[q]
                s1 = step_of_row[r]
                if s1 >= 0 and smark[s1] != stamp:
                    smark[s1] = stamp
                    stack[sp_] = s1
                    sp_ += 1
        st = np.sort(steps[:ns])
        for a in range(ns):                              # numeric elimination in pivot order
            p = st[a]
            coef = w[prow[p]]
            if coef == 0.0:
                continue
            for q in range(Lp[p], Lp[p + 1]):
                r = Li[q]
                w[r] -= Lx[q] * coef
                if mark[r] != stamp:
                    mark[r] = stamp
                    pat[np_] = r
                    np_ += 1
        big = 0.0
        for a in range(np_):
            r = pat[a]
            if step_of_row[r] < 0:
                big = max(big, abs(w[r]))
        if big <= piv_rel * cmax or lnz + np_ > max_lnz:
            for a in range(np_):
                w[pat[a]] = 0.0
            continue
        piv = -1
        best = 1 << 62
        for a in range(np_):
            r = pat[a]
            if step_of_row[r] >= 0 or abs(w[r]) < 0.1 * big:
                continue
            if rowcount[r] < best or (rowcount[r] == best and abs(w[r]) > abs(w[piv])):
                best = rowcount[r]
                piv = r
        pv = w[piv]
        if lnz + np_ > cap:
            while lnz + np_ > cap:
                cap *= 2
            Li2 = np.empty(cap, dtype=np.int64)
            Lx2 = np.empty(cap, dtype=np.float64)
            Li2[:lnz] = Li[:lnz]
            Lx2[:lnz] = Lx[:lnz]
            Li, Lx = Li2, Lx2
        for a in range(np_):
            r = pat[a]
            if r != piv and step_of_row[r] < 0 and w[r] != 0.0:
                Li[lnz] = r
                Lx[lnz] = w[r] / pv
                lnz += 1
            w[r] = 0.0
        prow[k] = piv
        step_of_row[piv] = k
        Lp[k + 1] = lnz
        acc[j] = True
        k += 1
    return acc, step_of_row


def crash_basis(prob, x, y, bound_tol=1e-6, piv_rel=1e-4, max_lnz=50_000_000):
    """Nonsingular basis (statuses, exactly m basic) from a near-optimal primal-dual point.

    Priority: free variables and variables away from their bounds (farthest first), then
    variables at a bound with the smallest |reduced cost| (dual-degenerate, likely basic at a
    bound), fixed variables never; `_crash_lu` accepts them in that order while they stay
    independent, and each row left without a pivot gets its own slack. The basis is therefore
    nonsingular (conditioned up to piv_rel) by construction.
    """
    m, n = prob.m, prob.n
    x = np.clip(np.asarray(x, dtype=float), prob.lx, prob.ux)
    y = np.asarray(y, dtype=float)
    z = np.concatenate([x, prob.A @ x])
    lo, hi = _std_bounds(prob)
    with np.errstate(invalid="ignore"):
        dlo = np.where(np.isfinite(lo), (z - lo) / (1.0 + np.abs(lo)), np.inf)
        dhi = np.where(np.isfinite(hi), (hi - z) / (1.0 + np.abs(hi)), np.inf)
    dist = np.minimum(dlo, dhi)
    d = np.abs(np.concatenate([prob.c - prob.A.T @ y, y]))
    cs = 1.0 + (np.abs(prob.c).max() if n else 0.0)
    free = ~np.isfinite(lo) & ~np.isfinite(hi)
    fixed = np.isfinite(lo) & (lo == hi)
    away = ((dist > bound_tol) | free) & ~fixed
    # tier key: away -> (-1, 0] by -dist ; at a bound -> [2, 3) by |d|
    key = np.where(away, -np.minimum(dist, 1e6) / 1e6 * 0.999, 2.0 + np.minimum(d / cs, 0.999))
    order = np.argsort(key, kind="stable").astype(np.int64)
    order = order[~fixed[order]]
    C = prob.A.tocsc()
    C.sort_indices()
    rowcount = (np.diff(prob.A.indptr) + 1).astype(np.int64)
    acc, step_of_row = _crash_lu(m, n, C.indptr.astype(np.int64), C.indices.astype(np.int64),
                                 C.data.astype(np.float64), order, rowcount, float(piv_rel), int(max_lnz))
    acc[n + np.flatnonzero(step_of_row < 0)] = True       # slacks fill the unpivoted rows
    status = _nonbasic_side(lo, hi, dlo <= dhi)
    status[acc] = "basic"
    st = [str(v) for v in status]
    return {"col_statuses": st[:n], "row_statuses": st[n:]}


def basis_from_point(prob, x, y, bound_tol=1e-6):
    """Column and row statuses (exactly m basic) guessed from a near-optimal primal-dual point.

    A variable within `bound_tol` (relative) of a bound is nonbasic there; the others are ranked by
    distance from their nearest bound (farthest first, small |reduced cost| breaking ties) and the
    first m are basic; if fewer than m qualify, the row slacks farthest from their bounds fill in.
    """
    m, n = prob.m, prob.n
    x = np.clip(np.asarray(x, dtype=float), prob.lx, prob.ux)
    y = np.asarray(y, dtype=float)
    z = np.concatenate([x, prob.A @ x])
    lo, hi = _std_bounds(prob)
    scale = 1.0 + np.abs(z)
    with np.errstate(invalid="ignore"):
        dlo = np.where(np.isfinite(lo), (z - lo) / scale, np.inf)
        dhi = np.where(np.isfinite(hi), (hi - z) / scale, np.inf)
    dist = np.minimum(dlo, dhi)
    lam = np.concatenate([prob.c - prob.A.T @ y, y])
    free = ~np.isfinite(lo) & ~np.isfinite(hi)
    status = _nonbasic_side(lo, hi, dlo <= dhi)
    cand = (dist > bound_tol) | free
    key = np.where(cand, np.minimum(dist, 1e6), -1.0) - 1e-12 * np.abs(lam)
    order = np.argsort(-key, kind="stable")
    basic = order[cand[order]][:m]
    if len(basic) < m:                                   # promote row slacks farthest from their bounds
        chosen = np.zeros(n + m, dtype=bool)
        chosen[basic] = True
        sl = n + np.argsort(-dist[n:], kind="stable")
        sl = sl[~chosen[sl]]
        basic = np.concatenate([basic, sl[:m - len(basic)]])
    status[basic.astype(int)] = "basic"
    st = [str(v) for v in status]
    return {"col_statuses": st[:n], "row_statuses": st[n:]}


def classify(prob, x, y, fix_tol=1e-3, near_tol=1e-3, drop_tol=1e-3):
    """Fixings and row drops from dual / primal evidence at a PDHG point.

    Returns (fix, drop): fix is an int8 array over z = (x, s) with -1 (fix at lower), +1 (fix at
    upper), 0 (keep); drop is a bool array over rows (row left out of the reduced LP).
    Tolerances are relative: reduced costs to 1 + max|c|, distances to 1 + |bound|.
    """
    m, n = prob.m, prob.n
    x = np.clip(np.asarray(x, dtype=float), prob.lx, prob.ux)
    y = np.asarray(y, dtype=float)
    s = prob.A @ x
    z = np.concatenate([x, s])
    lo, hi = _std_bounds(prob)
    d = np.concatenate([prob.c - prob.A.T @ y, y])
    cs = 1.0 + (np.abs(prob.c).max() if n else 0.0)
    with np.errstate(invalid="ignore"):
        near_lo = np.isfinite(lo) & (z - lo <= near_tol * (1.0 + np.abs(lo)))
        near_hi = np.isfinite(hi) & (hi - z <= near_tol * (1.0 + np.abs(hi)))
    fix = np.zeros(n + m, dtype=np.int8)
    fix[near_lo & (d > fix_tol * cs)] = -1
    fix[near_hi & (d < -fix_tol * cs)] = 1
    fix[np.isfinite(lo) & (lo == hi)] = -1               # fixed variables (and equality slacks) stay fixed
    with np.errstate(invalid="ignore"):
        inside = (~np.isfinite(prob.lc) | (s - prob.lc > drop_tol * (1.0 + np.abs(prob.lc)))) & \
                 (~np.isfinite(prob.uc) | (prob.uc - s > drop_tol * (1.0 + np.abs(prob.uc))))
    drop = (np.abs(y) <= drop_tol * 1e-3 * cs) & inside & (prob.lc < prob.uc)
    fix[n:][drop] = 0
    return fix, drop


def _hard_fix(prob):
    lo, hi = _std_bounds(prob)
    return np.where(np.isfinite(lo) & (lo == hi), -1, 0).astype(np.int8)


def _reduce(prob, fix, drop):
    """The reduced LP: fixed columns substituted, dropped rows removed, fixed slacks turned into
    equalities. Returns (red, keep_cols, keep_rows, xfix) with xfix the full-length fixed values."""
    n = prob.n
    lo, hi = _std_bounds(prob)
    val = np.where(fix < 0, lo, np.where(fix > 0, hi, 0.0))
    cfix = fix[:n] != 0
    keep_cols = np.flatnonzero(~cfix)
    keep_rows = np.flatnonzero(~drop)
    xfix = np.where(cfix, val[:n], 0.0)
    Ar = prob.A[keep_rows]
    shift = Ar @ xfix
    rf = fix[n:][keep_rows]
    rv = val[n:][keep_rows] - shift
    lc = np.where(rf != 0, rv, prob.lc[keep_rows] - shift)
    uc = np.where(rf != 0, rv, prob.uc[keep_rows] - shift)
    red = Problem(c=prob.c[keep_cols], A=Ar[:, keep_cols], lc=lc, uc=uc, lx=prob.lx[keep_cols],
                  ux=prob.ux[keep_cols], c0=prob.c0 + float(prob.c @ xfix), obj_sign=prob.obj_sign,
                  name=(prob.name or "lp") + "/reduced")
    return red, keep_cols, keep_rows, xfix


def _lift(prob, fix, keep_cols, keep_rows, xfix, r):
    """Full-model x, y and basis statuses from a reduced-LP result."""
    m, n = prob.m, prob.n
    x = xfix.copy()
    x[keep_cols] = r["x"]
    y = np.zeros(m)
    y[keep_rows] = r["y"]
    cst = np.where(fix[:n] > 0, "at_upper", "at_lower").astype(object)
    cst[keep_cols] = np.asarray(r["col_statuses"], dtype=object)
    rst = np.full(m, "basic", dtype=object)
    rr = np.asarray(r["row_statuses"], dtype=object).copy()
    rf = fix[n:][keep_rows]
    # a row turned into an equality at one of its bounds: its nonbasic slack sits at that side
    ineq = prob.lc[keep_rows] < prob.uc[keep_rows]
    nb = (rf != 0) & ineq & (rr != "basic")
    rr[nb & (rf < 0)] = "at_lower"
    rr[nb & (rf > 0)] = "at_upper"
    rst[keep_rows] = rr
    return x, y, {"col_statuses": [str(v) for v in cst], "row_statuses": [str(v) for v in rst]}


def _violations(prob, x, y, fix, drop, ptol, dtol):
    """Which fixings / drops the lifted vertex contradicts on the full model."""
    n = prob.n
    s = prob.A @ x
    with np.errstate(invalid="ignore"):
        viol_rows = drop & ((s < prob.lc - ptol * (1.0 + np.abs(prob.lc))) |
                            (s > prob.uc + ptol * (1.0 + np.abs(prob.uc))))
    d = np.concatenate([prob.c - prob.A.T @ y, y])
    lo, hi = _std_bounds(prob)
    real = ~(np.isfinite(lo) & (lo == hi))
    viol_fix = real & (((fix < 0) & (d < -dtol)) | ((fix > 0) & (d > dtol)))
    viol_fix[n:] &= ~drop
    return viol_rows, viol_fix


def _reduced_start(prob, fix, keep_cols, keep_rows, full_start):
    """Statuses of the reduced LP from full-model statuses."""
    n = prob.n
    cs = np.asarray(full_start["col_statuses"], dtype=object)[keep_cols]
    rs = np.asarray(full_start["row_statuses"], dtype=object)[keep_rows].copy()
    eq = fix[n:][keep_rows] != 0                         # rows made equalities: slack is fixed
    rs[eq & (rs != "basic")] = "at_lower"
    return {"col_statuses": [str(v) for v in cs], "row_statuses": [str(v) for v in rs]}


def _count_basic(start):
    return sum(v == "basic" for v in start["col_statuses"]) + sum(v == "basic" for v in start["row_statuses"])


# ------------------------------------------------------------------ driver
def crossover(prob, x, y, time_limit=3600.0, tol=1e-9, reduce=True, fix_tol=1e-3, near_tol=1e-3,
              drop_tol=1e-3, rounds=4, start=None, verbose=False) -> dict:
    """Exact vertex from a PDHG point (x, y).

    reduce=False runs the simplex on the full LP from the guessed basis (no fixing / dropping).
    start: a full-model basis to use instead of the guess (e.g. a neighbouring scenario's basis).
    The result has the simplex keys (status, x, y, col/row statuses, iterations) plus
    crossover_* timings and log, the reduced size, and the KKT residuals ("kkt").
    """
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    left = lambda: max(deadline - time.perf_counter(), 1e-3)  # noqa: E731
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    log = []
    iters = 0
    if reduce:
        fix, drop = classify(prob, x, y, fix_tol, near_tol, drop_tol)
    else:
        fix, drop = _hard_fix(prob), np.zeros(prob.m, dtype=bool)
    full_start = start
    t_classify = time.perf_counter() - t0
    r, rr, shape = None, None, None
    t_crash = 0.0
    for rnd in range(rounds + 1):
        red, kc, kr, xfix = _reduce(prob, fix, drop)
        shape = shape or (red.m, red.n)
        rstart = None if full_start is None else _reduced_start(prob, fix, kc, kr, full_start)
        if rstart is None or _count_basic(rstart) != red.m:
            tc = time.perf_counter()
            rstart = crash_basis(red, np.clip(x, prob.lx, prob.ux)[kc], y[kr])
            t_crash += time.perf_counter() - tc
        rr = _simplex(red, tol, left(), start=rstart)
        iters += rr["iterations"]
        if rr["status"] == "numerical_error":            # singular / ill-conditioned guess: slack basis
            rr = _simplex(red, tol, left())
            iters += rr["iterations"]
        log.append({"round": rnd, "m": red.m, "n": red.n, "status": rr["status"],
                    "iterations": rr["iterations"], "time": round(rr.get("time", 0.0), 4)})
        if verbose:
            print("crossover", log[-1], flush=True)
        if rr["status"] == "optimal":
            xl, yl, full_start = _lift(prob, fix, kc, kr, xfix, rr)
            vr, vf = _violations(prob, xl, yl, fix, drop, 1e-9, tol)
            log[-1].update(viol_rows=int(vr.sum()), viol_fix=int(vf.sum()))
            if not vr.any() and not vf.any():
                r = dict(rr, x=xl, y=yl, **full_start)
                break
            drop = drop & ~vr
            fix = np.where(vf, 0, fix).astype(np.int8)
            if rnd >= rounds - 1:                          # last round: the full LP from this basis
                fix, drop = _hard_fix(prob), np.zeros(prob.m, dtype=bool)
        elif rr["status"] in ("infeasible", "unbounded"):
            full_lp = not drop.any() and np.array_equal(fix, _hard_fix(prob))
            if full_lp:                                    # the model itself is infeasible / unbounded
                r = dict(rr)
                break
            fix, drop = _hard_fix(prob), np.zeros(prob.m, dtype=bool)   # restriction too tight: full LP
        else:                                              # time / iteration limit or numerical trouble
            break
    if r is None:
        r = dict(rr or {})
        r.setdefault("status", "numerical_error")
        if r["status"] == "optimal":
            r["status"] = "iteration_limit"                # repairs ran out (should not happen)
    if "x" not in r or len(r["x"]) != prob.n:
        r["x"], r["y"] = x, y
    r["iterations"] = iters
    r["kkt"] = kkt_residuals(prob, r["x"], r["y"])
    if r["status"] == "optimal" and r["kkt"]["max_rel"] > 1e-6:
        r["status"] = "numerical_error"
    r.update(crossover_time=time.perf_counter() - t0, crossover_classify_time=t_classify, crossover_crash_time=t_crash,
             crossover_rounds=len(log), crossover_log=log, reduced_shape=shape)
    return r
