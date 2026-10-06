"""Certified global bounds for bilinear models: bound tightening + McCormick relaxation.

1. FBBT (feasibility-based bound tightening). For every row  lc <= sum a_k x_k + sum c x_i x_j <= uc
   and every linear variable k in it:
        a_k x_k in [lc - max(rest), uc - min(rest)]
   with the rest evaluated in interval arithmetic (products by the four-corner rule). Repeated
   until the bounds stop moving. Every step is exact interval reasoning, so no feasible point
   is ever cut off.
2. OBBT (optimisation-based bound tightening). On the McCormick LP relaxation, minimise and
   maximise each product-factor variable (warm-started simplex). New bounds that the
   relaxation proves still cut no feasible point of the original bilinear model.
3. McCormick relaxation. Each product x_i x_j is replaced by a new variable w_ij with the
   four envelopes of Al-Khayyal & Falk (1983); a square x_i^2 by two tangents and a secant.
   Integer variables are relaxed. The relaxation is an LP whose optimum bounds every feasible
   plan: an upper bound when maximising, a lower bound when minimising.
The LP is solved by a QENIVO engine and certified; the reported bound is the LP's objective.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from ..model import Problem


def _imul(a, b, c, d):
    """[a,b] * [c,d]."""
    with np.errstate(invalid="ignore"):
        v = np.array([a * c, a * d, b * c, b * d], dtype=float)
    v = np.where(np.isnan(v), 0.0, v)          # 0 * inf -> 0
    return v.min(), v.max()


def _sum_except(parts, k):
    """Sum of all interval end points except entry k (inf-aware)."""
    tot = 0.0
    for idx, v in enumerate(parts):
        if idx == k:
            continue
        if not np.isfinite(v):
            return v
        tot += v
    return tot


def _tighten_product(L, U, i, j, Tlo, Thi, eps) -> int:
    """Given x_i x_j in [Tlo, Thi], tighten x_i and x_j where the other factor keeps one sign."""
    changed = 0
    if i == j:
        if np.isfinite(Thi) and Thi >= 0:
            r_ = float(np.sqrt(Thi))
            if r_ < U[i] - eps * (1 + abs(U[i])):
                U[i] = r_; changed += 1
            if -r_ > L[i] + eps * (1 + abs(L[i])):
                L[i] = -r_; changed += 1
        return changed
    for a_, b_ in ((i, j), (j, i)):
        lb, ub = L[b_], U[b_]
        if not (lb > 0 or ub < 0) or not (np.isfinite(Tlo) or np.isfinite(Thi)):
            continue
        # x_a = (x_a x_b) / x_b with x_b in [lb, ub] of one strict sign
        cand = []
        for T in (Tlo, Thi):
            for d in (lb, ub):
                if np.isinf(d):
                    cand.append(0.0 if np.isfinite(T) else np.sign(T) * np.sign(d) * np.inf)
                else:
                    cand.append(T / d)
        nl, nu = min(cand), max(cand)
        if np.isfinite(nl) and nl > L[a_] + eps * (1 + abs(L[a_])):
            L[a_] = nl; changed += 1
        if np.isfinite(nu) and nu < U[a_] - eps * (1 + abs(U[a_])):
            U[a_] = nu; changed += 1
    return changed


def fbbt(M, passes: int = 30, eps: float = 1e-9, time_limit: float = 120.0):
    p = M.linear
    L, U = p.lx.copy(), p.ux.copy()
    A = p.A.tocsr()
    rows_terms = {}
    for r, i, j, c in M.terms:
        rows_terms.setdefault(r, []).append((i, j, c))
    t0 = time.perf_counter()
    for _ in range(passes):
        changed = 0
        for r in range(p.m):
            s, e = A.indptr[r], A.indptr[r + 1]
            cols, vals = A.indices[s:e], A.data[s:e]
            lo_t = np.where(vals > 0, vals * L[cols], vals * U[cols])
            hi_t = np.where(vals > 0, vals * U[cols], vals * L[cols])
            lo_t = np.where(np.isnan(lo_t), -np.inf, lo_t)
            hi_t = np.where(np.isnan(hi_t), np.inf, hi_t)
            blo = bhi = 0.0
            for i, j, c in rows_terms.get(r, ()):
                a, b = _imul(L[i], U[i], L[j], U[j])
                a, b = (c * a, c * b) if c > 0 else (c * b, c * a)
                blo += a; bhi += b
            n_inf_lo = int(np.isinf(lo_t).sum()) + int(np.isinf(blo))
            n_inf_hi = int(np.isinf(hi_t).sum()) + int(np.isinf(bhi))
            fin_lo = float(np.sum(lo_t[np.isfinite(lo_t)])) + (blo if np.isfinite(blo) else 0.0)
            fin_hi = float(np.sum(hi_t[np.isfinite(hi_t)])) + (bhi if np.isfinite(bhi) else 0.0)
            # backward propagation through each product: c x_i x_j in [T_lo, T_hi] = row limits minus
            # the rest of the row; then x_i in T/c / [L_j, U_j] whenever x_j keeps one strict sign
            tl_ = rows_terms.get(r, ())
            if tl_:
                ivs = []
                for i, j, c in tl_:
                    a, b = _imul(L[i], U[i], L[j], U[j])
                    ivs.append((c * a, c * b) if c > 0 else (c * b, c * a))
                parts_lo = list(lo_t) + [v[0] for v in ivs]
                parts_hi = list(hi_t) + [v[1] for v in ivs]
                off = len(lo_t)
                for t_idx, (i, j, c) in enumerate(tl_):
                    rest_lo = _sum_except(parts_lo, off + t_idx)
                    rest_hi = _sum_except(parts_hi, off + t_idx)
                    Tlo, Thi = (p.lc[r] - rest_hi) / c, (p.uc[r] - rest_lo) / c    # bounds on x_i x_j
                    if c < 0:
                        Tlo, Thi = Thi, Tlo
                    changed += _tighten_product(L, U, i, j, Tlo, Thi, eps)
            for k, (col, a) in enumerate(zip(cols, vals)):
                # min / max of the rest of the row, excluding term k
                if np.isinf(lo_t[k]):
                    rest_lo = fin_lo if n_inf_lo == 1 else -np.inf
                else:
                    rest_lo = fin_lo - lo_t[k] if n_inf_lo == 0 else -np.inf
                if np.isinf(hi_t[k]):
                    rest_hi = fin_hi if n_inf_hi == 1 else np.inf
                else:
                    rest_hi = fin_hi - hi_t[k] if n_inf_hi == 0 else np.inf
                t_lo = p.lc[r] - rest_hi           # a x in [t_lo, t_hi]
                t_hi = p.uc[r] - rest_lo
                if a > 0:
                    nl, nu = t_lo / a, t_hi / a
                else:
                    nl, nu = t_hi / a, t_lo / a
                if np.isfinite(nl) and np.isfinite(L[col]) and nl > L[col] + eps * (1 + abs(L[col])):
                    L[col] = nl; changed += 1
                if np.isfinite(nu) and np.isfinite(U[col]) and nu < U[col] - eps * (1 + abs(U[col])):
                    U[col] = nu; changed += 1
        if p.integer is not None:
            L[p.integer], U[p.integer] = np.ceil(L[p.integer] - 1e-9), np.floor(U[p.integer] + 1e-9)
        if changed == 0 or time.perf_counter() - t0 > time_limit:
            break
    return L, U


def _mccormick_lp(M, L, U):
    """Build the McCormick relaxation LP at the given factor bounds. Returns (relax, keys) or a
    reason string if some product factor is still unbounded."""
    p = M.linear
    pairs = {}
    for r, i, j, c in M.terms:
        key = (min(i, j), max(i, j))
        pairs.setdefault(key, []).append((r, c))
    bad = sorted({k for key in pairs for k in key if not (np.isfinite(L[k]) and np.isfinite(U[k]))})
    if bad:
        return None, bad
    n, m = p.n, p.m
    W = len(pairs)
    keys = list(pairs)
    ri, ci, vv = [], [], []
    for w, key in enumerate(keys):
        for r, c in pairs[key]:
            ri.append(r); ci.append(n + w); vv.append(c)
    Aw = sp.csr_matrix((vv, (ri, ci)), shape=(m, n + W))
    A0 = sp.hstack([p.A, sp.csr_matrix((m, W))], format="csr") + Aw
    er, ec, ev, elo, ehi = [], [], [], [], []
    rows = 0

    def env(coefs, lo, hi):
        nonlocal rows
        for col, v in coefs:
            er.append(rows); ec.append(col); ev.append(v)
        elo.append(lo); ehi.append(hi); rows += 1

    wl, wu = np.zeros(W), np.zeros(W)
    for w, (i, j) in enumerate(keys):
        col = n + w
        Li, Ui, Lj, Uj = L[i], U[i], L[j], U[j]
        if i == j:
            env([(col, 1.0), (i, -2 * Li)], -Li * Li, np.inf)          # tangent at L
            env([(col, 1.0), (i, -2 * Ui)], -Ui * Ui, np.inf)          # tangent at U
            env([(col, 1.0), (i, -(Li + Ui))], -np.inf, -Li * Ui)      # secant
            wl[w] = 0.0 if Li <= 0 <= Ui else min(Li * Li, Ui * Ui)
            wu[w] = max(Li * Li, Ui * Ui)
        else:
            env([(col, 1.0), (j, -Li), (i, -Lj)], -Li * Lj, np.inf)
            env([(col, 1.0), (j, -Ui), (i, -Uj)], -Ui * Uj, np.inf)
            env([(col, 1.0), (j, -Ui), (i, -Lj)], -np.inf, -Ui * Lj)
            env([(col, 1.0), (j, -Li), (i, -Uj)], -np.inf, -Li * Uj)
            wl[w], wu[w] = _imul(Li, Ui, Lj, Uj)
    E = sp.csr_matrix((ev, (er, ec)), shape=(rows, n + W))
    relax = Problem(c=np.concatenate([p.c, np.zeros(W)]), A=sp.vstack([A0, E], format="csr"),
                    lc=np.concatenate([p.lc, elo]), uc=np.concatenate([p.uc, ehi]),
                    lx=np.concatenate([L, wl]), ux=np.concatenate([U, wu]), obj_sign=p.obj_sign,
                    name=f"{p.name}_mccormick")
    return relax, keys


def _solve_simplex_lp(lp, tol, time_limit, warm_basis=None):
    from ..api import _finish
    from ..engines.simplex import solve_simplex
    start = None
    if warm_basis is not None:
        cs, rs = warm_basis.get("col_statuses"), warm_basis.get("row_statuses")
        if cs is not None and rs is not None and len(cs) == lp.n and len(rs) == lp.m:
            start = warm_basis
    out = solve_simplex(lp, tol=min(tol, 1e-9), time_limit=time_limit, start=start)
    sol = _finish(lp, out["status"], out["x"], out["y"], tol,
                  {"engine": "simplex", "backend": "cpu", "iterations": out["iterations"],
                   "time": out["time"], "reason": "mccormick/obbt",
                   "engine_impl": out.get("engine_impl", "unknown")})
    if "col_statuses" in out:
        sol.extra["basis"] = {"col_statuses": out["col_statuses"],
                              "row_statuses": out["row_statuses"]}
    return sol


def obbt(M, L=None, U=None, max_vars: int | None = 64, time_limit: float = 120.0,
         tol: float = 1e-8, engine: str = "simplex", eps: float = 1e-9):
    """Optimisation-based bound tightening on product factors via the McCormick LP.

    For each selected factor, solve min x_j and max x_j over the relaxation (warm-started
    simplex). Returns tightened (L, U) and a small report. Never cuts a feasible bilinear point.
    """
    p = M.linear
    if L is None or U is None:
        L, U = fbbt(M)
    else:
        L, U = L.copy(), U.copy()
    factors = sorted({i for _, i, _, _ in M.terms} | {j for _, _, j, _ in M.terms})
    # Prefer variables with the widest remaining boxes (most room to tighten).
    width = [(U[j] - L[j] if np.isfinite(U[j] - L[j]) else np.inf, j) for j in factors]
    width.sort(reverse=True)
    if max_vars is not None:
        width = width[:max_vars]
    t0 = time.perf_counter()
    tightened = 0
    solves = 0
    warm = None
    for _, j in width:
        if time.perf_counter() - t0 > time_limit:
            break
        if not (np.isfinite(L[j]) and np.isfinite(U[j])):
            continue
        if U[j] - L[j] <= eps * (1 + abs(L[j]) + abs(U[j])):
            continue
        built, keys = _mccormick_lp(M, L, U)
        if built is None:
            break
        for sense, attr in ((1.0, "L"), (-1.0, "U")):   # minimise / maximise x_j
            remain = time_limit - (time.perf_counter() - t0)
            if remain <= 0:
                break
            lp = built.copy()
            lp.c = np.zeros(lp.n)
            lp.c[j] = sense
            lp.obj_sign = 1.0
            lp.name = f"{p.name}_obbt_{attr}{j}"
            if engine == "simplex":
                sol = _solve_simplex_lp(lp, tol, remain, warm_basis=warm)
            else:
                from ..api import solve
                sol = solve(lp, engine=engine, tol=tol, time_limit=remain, polish=True)
            solves += 1
            if sol.verdict != "optimal" or sol.x is None:
                warm = None
                continue
            warm = sol.extra.get("basis")
            val = float(sol.x[j])
            if attr == "L" and val > L[j] + eps * (1 + abs(L[j])):
                L[j] = val; tightened += 1
            elif attr == "U" and val < U[j] - eps * (1 + abs(U[j])):
                U[j] = val; tightened += 1
        if L[j] > U[j]:
            mid = 0.5 * (L[j] + U[j])
            L[j] = U[j] = mid
    return L, U, {"tightened": tightened, "solves": solves, "time": time.perf_counter() - t0,
                  "factors_tried": len(width)}


def mccormick_bound(M, tol: float = 1e-7, engine: str = "auto", time_limit: float = 3600.0,
                    verbose=False, use_obbt: bool = True, obbt_vars: int | None = 64,
                    obbt_time: float = 120.0):
    """Return {'bound', 'solution', 'unbounded_factors', 'fbbt_time', ...}."""
    from ..api import solve
    p = M.linear
    t0 = time.perf_counter()
    L, U = fbbt(M)
    t_fbbt = time.perf_counter() - t0
    obbt_info = None
    if use_obbt:
        remain = max(1.0, min(obbt_time, time_limit - (time.perf_counter() - t0)))
        eng = "simplex" if engine in ("auto", "simplex") else engine
        L, U, obbt_info = obbt(M, L, U, max_vars=obbt_vars, time_limit=remain, tol=tol, engine=eng)
    built, keys_or_bad = _mccormick_lp(M, L, U)
    if built is None:
        return {"bound": None, "reason": f"{len(keys_or_bad)} product factors still unbounded after FBBT",
                "unbounded_factors": keys_or_bad, "fbbt_time": t_fbbt, "obbt": obbt_info}
    relax = built
    if verbose:
        print(f"  McCormick relaxation: {relax.size_str()} (FBBT {t_fbbt:.1f}s"
              + (f", OBBT {obbt_info['tightened']} tightens in {obbt_info['time']:.1f}s"
                 if obbt_info else "") + ")")
    remain = max(1.0, time_limit - (time.perf_counter() - t0))
    if engine in ("auto", "simplex"):
        sol = _solve_simplex_lp(relax, tol, remain)
    else:
        sol = solve(relax, engine=engine, tol=tol, time_limit=remain)
    return {"bound": sol.objective if sol.verdict == "optimal" else None, "solution": sol,
            "fbbt_time": t_fbbt, "obbt": obbt_info, "relaxation_size": relax.size_str(),
            "products": len(keys_or_bad),
            "tightened_bounds": int(np.sum(np.isfinite(L) & ~np.isfinite(p.lx))
                                    + np.sum(np.isfinite(U) & ~np.isfinite(p.ux)))}
