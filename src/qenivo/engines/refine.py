"""Precision-refined PDHG (engine "pdhg-rf"): float32 iterations, float64-certified answers.

Why: a first-order LP method spends almost all its time streaming the matrix through two
sparse products. Consumer and inference GPUs (T4, L4, RTX) run float32 about 32x faster than
float64 and float32 moves half the bytes, yet float32 alone stalls near 1e-5 relative accuracy.
Iterative refinement removes that ceiling: every round solves a *correction* LP whose data are
the current residuals magnified to order one, so each float32 solve buys a fixed factor of
accuracy and the float64 work is two host products per round (Gleixner, Steffy & Wolter,
"Iterative refinement for linear programming", INFORMS J. Computing 2016; here on PDHG).

Slack form. Rows are written Ax - s = 0 with lc <= s <= uc, so every row dual is free and a
correction may move a dual in either direction (the compact form would let a one-sided row's
dual only grow). For the current (x, s, y) with residual r = Ax - s:

    min   dD * [c - A'y ; y]' (dx, ds)
    s.t.  A dx - ds = -dP * r
          dP (lx - x) <= dx <= dP (ux - x),   dP (lc - s) <= ds <= dP (uc - s)

then x += dx/dP, s += ds/dP, y += dy/dD, where dy is the correction's row dual. dP and dD are
the inverse primal and dual residuals, shrunk together when the duality gap would otherwise
dominate. The correction is solved to an *absolute* accuracy (norms passed to the engine), so
every round multiplies the error by about the inner tolerance.

The verdict is always the float64 KKT check of (x, y) on the original problem. A scenario that
stops improving is finished by the float64 engine, warm-started from its best point.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sps

from ..certify.kkt import is_optimal, kkt_from_products, rhs_norm
from . import pdhg

FAR = 1e4           # in correction units (the residual is of order one there)
GIVE_UP = 1e-3      # a first float32 answer worse than this goes straight to the float64 engine


def _as2d(v, rows, S):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        v = v[:, None]
    return np.ascontiguousarray(np.broadcast_to(v, (rows, S)))


def solve_refined(A, c, lc, uc, lx, ux, c0=0.0, tol=1e-6, backend="numpy",
                  inner_dtype="float32", inner_tol=1e-4, max_rounds=10, time_limit=3600.0,
                  polish=True, inner_max_iter=50_000, stall_checks=40, opts: pdhg.PDHGOptions | None = None,
                  verbose=False) -> pdhg.BatchResult:
    """Solve S LPs sharing A to `tol` (float64 KKT) with float32 inner solves. Shapes as in
    pdhg.solve_batch. Returns a pdhg.BatchResult; `history` holds one entry per round."""
    t_start = time.perf_counter()
    A = sps.csr_matrix(A, dtype=np.float64)
    m, n = A.shape
    S = max(np.ndim(c) == 2 and np.shape(c)[1] or 1,
            *[np.ndim(v) == 2 and np.shape(v)[1] or 1 for v in (lc, uc, lx, ux)])
    C, LX, UX = (_as2d(v, n, S) for v in (c, lx, ux))
    LC, UC = (_as2d(v, m, S) for v in (lc, uc))
    c0v = np.broadcast_to(np.asarray(c0, dtype=np.float64), (S,)).copy()
    AT = A.T.tocsr()
    bnorm, cnorm = rhs_norm(LC, UC), np.linalg.norm(C, axis=0)

    # slack form: z = (x, s), rows A x - s = 0
    Ae = sps.hstack([A, -sps.identity(m, format="csr")], format="csr")
    AeT = Ae.T.tocsr()
    LZ, UZ = np.vstack([LX, LC]), np.vstack([UX, UC])
    CZ = np.vstack([C, np.zeros((m, S))])
    base = opts or pdhg.PDHGOptions()
    inner = pdhg.PDHGOptions(**{**base.__dict__, "dtype": inner_dtype, "tol": inner_tol, "bound_obj": False,
                                "detect_infeasibility": False, "verbose": False, "stall_checks": stall_checks,
                                "max_iter": min(base.max_iter, inner_max_iter)})

    # round 0: the original problem in float32, standard scaling, to the inner tolerance
    ti = time.perf_counter()
    first = pdhg.PDHGOptions(**{**base.__dict__, "dtype": inner_dtype, "tol": inner_tol, "verbose": False,
                                "detect_infeasibility": False, "time_limit": 0.15 * time_limit,
                                "stall_checks": stall_checks, "max_iter": min(base.max_iter, 4 * inner_max_iter)})
    b0 = pdhg.solve_batch(A, C, LC, UC, LX, UX, c0v, first, backend)
    t_inner = time.perf_counter() - ti
    x0 = np.clip(np.nan_to_num(b0.x), LX, UX)
    z = np.vstack([x0, np.clip(A @ x0, LC, UC)])
    y = np.nan_to_num(b0.y)
    iters = np.asarray(b0.iterations, dtype=np.int64).copy()
    round_iters = [iters.tolist()]

    cache: dict = {}
    status = ["iteration_limit"] * S
    done = np.zeros(S, dtype=bool)
    rounds = np.zeros(S, dtype=np.int64)
    best = np.full(S, np.inf)
    best_z, best_y = z.copy(), y.copy()
    stalls = np.zeros(S, dtype=np.int64)
    history = []
    t_host = 0.0
    floor_p = 1e-3 * tol * (1.0 + bnorm)
    floor_d = 1e-3 * tol * (1.0 + cnorm)

    def original_kkt(cols):
        X, Y = z[:n, cols], y[:, cols]
        return kkt_from_products(X, Y, A @ X, AT @ Y, C[:, cols], c0v[cols], LC[:, cols], UC[:, cols],
                                 LX[:, cols], UX[:, cols], bnorm[cols], cnorm[cols])

    for rnd in range(max_rounds + 1):
        th = time.perf_counter()
        act = np.flatnonzero(~done)
        k = original_kkt(act)
        err = np.maximum(np.maximum(k["rel_primal"], k["rel_dual"]), k["rel_gap"])
        ok = is_optimal(k, tol)
        improved = err < best[act]
        for a_, j in enumerate(act):
            if improved[a_]:
                stalls[j] = 0 if err[a_] < 0.5 * best[j] else stalls[j] + 1
                best[j] = err[a_]
                best_z[:, j], best_y[:, j] = z[:, j], y[:, j]
            else:
                stalls[j] += 1
            if ok[a_]:
                done[j] = True
                status[j] = "optimal"
        history.append({"round": rnd, "time": time.perf_counter() - t_start, "cols": act.tolist(),
                        "max_rel": err.tolist()})
        if verbose:
            print(f"round {rnd}: active {len(act)}  worst {err.max():.2e}  best {np.min(err):.2e}  "
                  f"t {time.perf_counter() - t_start:.2f}s  iters {round_iters[-1]}")
        elapsed = time.perf_counter() - t_start
        act = np.flatnonzero(~done & (stalls < 2))
        if rnd == 0:
            # guard: refinement needs a roughly right first answer. Where float32 could not reach
            # GIVE_UP, the remaining time goes to the float64 engine (the polish step below).
            act = act[best[act] <= GIVE_UP]
        t_host += time.perf_counter() - th
        if len(act) == 0 or rnd == max_rounds or elapsed > time_limit:
            break

        # ---- correction problem for the active scenarios (float64 on the host)
        th = time.perf_counter()
        Z, Y = z[:, act], y[:, act]
        r = Ae @ Z                                              # = A x - s
        lam = CZ[:, act] - AeT @ Y                              # reduced costs of (x, s)
        rd = (np.where(np.isfinite(LZ[:, act]), 0.0, np.maximum(lam, 0.0))
              - np.where(np.isfinite(UZ[:, act]), 0.0, np.maximum(-lam, 0.0)))
        gap = np.abs(np.sum(CZ[:, act] * Z, axis=0) - _dual_obj(lam, Y, LZ[:, act], UZ[:, act]))
        # error in primal units and in dual units; the gap counts on both sides, converted
        # through the problem's own scale, so the two magnifications stay balanced
        eps_p = np.maximum.reduce([np.linalg.norm(r, axis=0), gap / (1.0 + cnorm[act]), floor_p[act]])
        eps_d = np.maximum.reduce([np.linalg.norm(rd, axis=0), gap / (1.0 + bnorm[act]), floor_d[act]])
        dP, dD = 1.0 / eps_p, 1.0 / eps_d
        shrink = np.sqrt(np.maximum(dP * dD * gap, 1.0))
        dP, dD = dP / shrink, dD / shrink
        rhs = -dP * r
        lz_c = dP * (LZ[:, act] - Z)
        uz_c = dP * (UZ[:, act] - Z)
        cz_c = dD * lam
        # far-away bounds cannot bind in an order-one correction: drop them. A variable sitting
        # at a bound whose reduced cost pushes into it by far more than the correction scale is
        # held there this round; its cost is then irrelevant, so float32 never has to carry it.
        lz_c[lz_c < -FAR] = -np.inf
        uz_c[uz_c > FAR] = np.inf
        hold = (((lz_c == 0.0) & (cz_c > FAR)) | ((uz_c == 0.0) & (cz_c < -FAR)))
        lz_c[hold], uz_c[hold], cz_c[hold] = 0.0, 0.0, 0.0
        t_host += time.perf_counter() - th

        ti = time.perf_counter()
        inner.time_limit = max(1.0, time_limit - (time.perf_counter() - t_start))
        br = pdhg.solve_batch(Ae, cz_c, rhs, rhs, lz_c, uz_c, 0.0, inner, backend,
                              norms=(dP * eps_p, dD * eps_d), cache=cache)
        t_inner += time.perf_counter() - ti
        iters[act] += br.iterations
        round_iters.append({int(j): int(i) for j, i in zip(act, br.iterations)})
        rounds[act] += 1
        dz, dy = np.nan_to_num(br.x), np.nan_to_num(br.y)
        z[:, act] = np.clip(Z + dz / dP, LZ[:, act], UZ[:, act])
        y[:, act] = Y + dy / dD

    # every scenario reports its best point; stalled ones are finished in float64
    z[:, ~done], y[:, ~done] = best_z[:, ~done], best_y[:, ~done]
    t_polish = 0.0
    left = np.flatnonzero(~done)
    if polish and len(left) and time.perf_counter() - t_start < time_limit:
        tp = time.perf_counter()
        fin = pdhg.PDHGOptions(**{**base.__dict__, "dtype": "float64", "tol": tol, "verbose": False,
                                  "time_limit": max(1.0, time_limit - (time.perf_counter() - t_start))})
        good = best[left] <= 100 * tol             # a rough float32 point is a worse start than none
        wx = np.where(good[None, :], z[:n, left], np.clip(0.0, LX[:, left], UX[:, left]))
        wy = np.where(good[None, :], y[:, left], 0.0)
        br = pdhg.solve_batch(A, C[:, left], LC[:, left], UC[:, left], LX[:, left], UX[:, left], c0v[left],
                              fin, backend, warm_x=wx, warm_y=wy)
        kp = kkt_from_products(br.x, br.y, A @ br.x, AT @ br.y, C[:, left], c0v[left], LC[:, left], UC[:, left],
                               LX[:, left], UX[:, left], bnorm[left], cnorm[left])
        ep = np.maximum(np.maximum(kp["rel_primal"], kp["rel_dual"]), kp["rel_gap"])
        take = ep <= best[left]                    # never return a worse answer than one already seen
        cols = left[take]
        z[:n, cols], y[:, cols] = br.x[:, take], br.y[:, take]
        iters[left] += br.iterations
        for a_, j in enumerate(left):
            status[j] = br.status[a_]
        t_polish = time.perf_counter() - tp
    polished = np.zeros(S, dtype=bool)
    polished[left] = polish

    kf = original_kkt(np.arange(S))
    okf = is_optimal(kf, tol)
    for j in range(S):
        if okf[j]:
            status[j] = "optimal"
        elif status[j] == "optimal":
            status[j] = "iteration_limit"
    total = time.perf_counter() - t_start
    res = pdhg.BatchResult(x=z[:n].copy(), y=y.copy(), status=status, iterations=iters, restarts=rounds,
                           kkt={k_: np.asarray(v, dtype=np.float64) for k_, v in kf.items()},
                           solve_time=total, setup_time=0.0,
                           backend=f"{backend}/{inner_dtype}+refine", eta=float("nan"),
                           primal_weight=np.full(S, np.nan), history=history)
    res.rays = {"_timing": {"inner_s": t_inner, "host_s": t_host, "polish_s": t_polish, "total_s": total,
                            "rounds": rounds.tolist(), "polished": polished.tolist(),
                            "round_iters": round_iters}}
    return res


def _dual_obj(lam, y, lz, uz):
    """Dual objective of the slack form (rows are equalities with zero right-hand side)."""
    lp, ln = np.maximum(lam, 0.0), np.maximum(-lam, 0.0)
    return np.sum(lp * np.where(np.isfinite(lz), lz, 0.0) - ln * np.where(np.isfinite(uz), uz, 0.0), axis=0)


def solve(prob, tol=1e-6, backend="numpy", **kw) -> pdhg.BatchResult:
    """Solve one Problem (S = 1) with float32 iterations and float64 refinement."""
    if prob.is_qp:
        raise ValueError("pdhg-rf is an LP engine")
    return solve_refined(prob.A, prob.c, prob.lc, prob.uc, prob.lx, prob.ux, prob.c0, tol=tol,
                         backend=backend, **kw)
