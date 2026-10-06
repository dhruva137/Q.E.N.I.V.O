"""Successive linear programming for bilinear refinery models (generalised distributive recursion).

Model (from `qenivo.io.gams.BilinearModel`): linear rows plus rows with terms c * x_i * x_j
(pool qualities times flows, yields times feeds). Two phases:

1. Feasibility homotopy (fixed-factor successive LP). Treat one factor of each product as a
   fixed "quality", replace c x_i x_j by c q_i x_j (or c x_i q_j), solve the LP, update the
   fixed factors from the new plan. This is the industrial distributive-recursion pattern for
   general bilinears; every LP is warm-started from the previous basis.
2. Penalty SLP with a trust region. At the current plan x^k each product is replaced by its
   first-order expansion
        c x_i x_j  ~  c (x_i^k x_j + x_j^k x_i - x_i^k x_j^k)
   which is exact at x^k, with elastic l1 slacks on nonlinear rows and a box trust region on
   product variables. A step is accepted when the true nonlinear merit improves by a fair share
   of what the LP predicted (Fletcher & Sainz de la Maza 1989; Zhang et al. arXiv 2411.09554).

Every LP uses QENIVO's native simplex (or IPM / PDHG if requested) with a warm start from the
previous iterate — never a cold first-order solve by default. The final plan is re-evaluated on
the original nonlinear rows; feasibility is the measured max violation, never assumed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..model import Problem


@dataclass
class SLPResult:
    x: np.ndarray
    objective: float
    max_violation: float            # absolute, on the original nonlinear rows
    rel_violation: float            # relative to 1 + |row activity|
    feasible: bool
    iterations: int
    time: float
    history: list = field(default_factory=list)
    status: str = ""
    phase: str = ""                 # homotopy | slp | combined

    def summary(self) -> str:
        return (f"SLP {self.status}: objective {self.objective:,.6f}  max violation {self.max_violation:.2e} "
                f"(rel {self.rel_violation:.1e})  iterations {self.iterations}  time {self.time:.1f}s")


def _product_vars(M):
    return np.array(sorted({i for _, i, _, _ in M.terms} | {j for _, _, j, _ in M.terms}), dtype=int)


def _quality_side(M):
    """For each term, True if the first factor is treated as the fixed quality.

    Prefer the factor that appears in more bilinear terms (quality-like), then the one with
    the tighter finite bound width.
    """
    p = M.linear
    count = np.zeros(p.n, dtype=int)
    for _, i, j, _ in M.terms:
        count[i] += 1
        if j != i:
            count[j] += 1
    width = np.where(np.isfinite(p.ux - p.lx), p.ux - p.lx, np.inf)
    out = []
    for _, i, j, _ in M.terms:
        if i == j:
            out.append(True)
            continue
        if count[i] != count[j]:
            out.append(count[i] >= count[j])
        elif width[i] != width[j]:
            out.append(width[i] <= width[j])
        else:
            out.append(i <= j)
    return out


def _fixed_factor_lp(M, q, fix_first, elastic_mu: float | None = None, pin_qualities: bool = True):
    """LP with each product c*x_i*x_j replaced by c*q_i*x_j or c*x_i*q_j (linear).

    When pin_qualities is True (distributive-recursion style), the factors treated as
    qualities are fixed at their current values so the LP cannot move them while still
    using the stale coefficient.
    """
    p = M.linear
    m, n = p.m, p.n
    r_i, c_i, v_i = [], [], []
    pinned = set()
    for (r, i, j, c), use_i in zip(M.terms, fix_first):
        if use_i:
            r_i.append(r); c_i.append(j); v_i.append(c * q[i])
            if pin_qualities:
                pinned.add(i)
        else:
            r_i.append(r); c_i.append(i); v_i.append(c * q[j])
            if pin_qualities:
                pinned.add(j)
    J = sp.csr_matrix((v_i, (r_i, c_i)), shape=(m, n)) if v_i else sp.csr_matrix((m, n))
    A = p.A + J
    lx, ux = p.lx.copy(), p.ux.copy()
    if pin_qualities:
        for i in pinned:
            lx[i] = ux[i] = float(q[i])
        lx, ux = np.minimum(lx, ux), np.maximum(lx, ux)
    if elastic_mu is None:
        return Problem(c=p.c.copy(), A=A, lc=p.lc.copy(), uc=p.uc.copy(),
                       lx=lx, ux=ux, obj_sign=p.obj_sign,
                       name=f"{p.name}_homotopy"), 0
    rows_nl = np.array(M.nonlinear_rows, dtype=int)
    k = len(rows_nl)
    E = sp.csr_matrix((np.ones(k), (rows_nl, np.arange(k))), shape=(m, k))
    A = sp.hstack([A, E, -E], format="csr")
    c = np.concatenate([p.c, np.full(2 * k, elastic_mu)])
    return Problem(c=c, A=A, lc=p.lc.copy(), uc=p.uc.copy(),
                   lx=np.concatenate([lx, np.zeros(2 * k)]),
                   ux=np.concatenate([ux, np.full(2 * k, np.inf)]),
                   obj_sign=p.obj_sign, name=f"{p.name}_homotopy_e"), k


def _linearised(M, xk, delta, mu, prod_vars):
    """LP at x^k: linearised products, elastic slacks on nonlinear rows, trust region."""
    p = M.linear
    m, n = p.m, p.n
    rows_nl = np.array(M.nonlinear_rows, dtype=int)
    k = len(rows_nl)
    r_i, c_i, v_i = [], [], []
    const = np.zeros(m)
    for r, i, j, c in M.terms:
        r_i += [r, r]; c_i += [i, j]; v_i += [c * xk[j], c * xk[i]]
        const[r] -= c * xk[i] * xk[j]
    J = sp.csr_matrix((v_i, (r_i, c_i)), shape=(m, n))
    E = sp.csr_matrix((np.ones(k), (rows_nl, np.arange(k))), shape=(m, k))
    A = sp.hstack([p.A + J, E, -E], format="csr")
    lc = p.lc - const
    uc = p.uc - const
    lx, ux = p.lx.copy(), p.ux.copy()
    lx[prod_vars] = np.maximum(lx[prod_vars], xk[prod_vars] - delta)
    ux[prod_vars] = np.minimum(ux[prod_vars], xk[prod_vars] + delta)
    lx, ux = np.minimum(lx, ux), np.maximum(lx, ux)
    c = np.concatenate([p.c, np.full(2 * k, mu)])
    lp = Problem(c=c, A=A, lc=lc, uc=uc, lx=np.concatenate([lx, np.zeros(2 * k)]),
                 ux=np.concatenate([ux, np.full(2 * k, np.inf)]), name=f"{p.name}_slp")
    return lp, k


def _merit(M, x, mu):
    return float(M.linear.c @ x) + mu * float(M.violation(x).sum())


def _finish_simplex(lp, out, tol, reason="slp warm simplex"):
    from ..api import _finish
    meta = {"engine": "simplex", "backend": "cpu", "iterations": out["iterations"],
            "time": out["time"], "reason": reason,
            "engine_impl": out.get("engine_impl", "unknown")}
    sol = _finish(lp, out["status"], out["x"], out["y"], tol, meta)
    if "col_statuses" in out:
        sol.extra["basis"] = {"col_statuses": out["col_statuses"],
                              "row_statuses": out["row_statuses"]}
    return sol


def _solve_lp(lp, engine: str, tol: float, time_limit: float, warm_basis=None, warm_sol=None,
              backend: str = "auto", polish: bool = True):
    """Solve one LP. Default path: native/Python simplex with an optional basis warm start."""
    eng = "simplex" if engine in ("auto", "simplex") else engine
    if eng == "simplex":
        from ..engines.simplex import solve_simplex
        start = None
        if warm_basis is not None:
            cs, rs = warm_basis.get("col_statuses"), warm_basis.get("row_statuses")
            if cs is not None and rs is not None and len(cs) == lp.n and len(rs) == lp.m:
                start = warm_basis
        out = solve_simplex(lp, tol=min(tol, 1e-9), time_limit=time_limit, start=start)
        return _finish_simplex(lp, out, tol)
    from ..api import solve
    if eng == "ipm":
        # IPM has no public warm-start hook; still prefer it over cold PDHG when requested.
        return solve(lp, engine="ipm", tol=tol, time_limit=time_limit, polish=polish)
    return solve(lp, engine=eng, tol=tol, backend=backend, time_limit=time_limit,
                 warm=warm_sol, polish=polish)


def _violation_report(M, x):
    v = M.violation(x)
    act = np.abs(M.evaluate(x))
    abs_v = float(v.max(initial=0.0)) if v.size else 0.0
    rel = float(np.max(v / (1.0 + act))) if v.size else 0.0
    return abs_v, rel


def run_homotopy(M, x0=None, max_iter: int = 80, tol_viol: float = 1e-6, tol_step: float = 1e-9,
                 engine: str = "simplex", backend: str = "auto", lp_tol: float = 1e-8,
                 time_limit: float = 3600.0, damping: float = 0.5, elastic_mu: float | None = None,
                 start_elastic: bool = True, verbose: bool = False) -> SLPResult:
    """Fixed-factor successive LP (distributive-recursion style) for feasibility first.

    Starts with elastic slacks by default so the LP stays feasible when the fixed-factor
    linearisation cuts off the true nonlinear set. Accepts any primal that improves the
    measured nonlinear violation, even when the engine verdict is not yet certified optimal
    (e.g. Phase-I / time-limited points).
    """
    t0 = time.perf_counter()
    p = M.linear
    x = np.clip(M.start if x0 is None else np.asarray(x0, float), p.lx, p.ux)
    fix_first = _quality_side(M)
    if elastic_mu is None:
        elastic_mu = 10.0 * max(1.0, float(np.abs(p.c).max()))
    hist = []
    status = "iteration_limit"
    warm_basis = None
    warm_sol = None
    use_elastic = bool(start_elastic)
    stall = 0
    it = 0
    best_viol = float(M.violation(x).max(initial=0.0))
    for it in range(1, max_iter + 1):
        remain = time_limit - (time.perf_counter() - t0)
        if remain <= 0:
            status = "time_limit"
            break
        lp, k = _fixed_factor_lp(M, x, fix_first, elastic_mu if use_elastic else None)
        # Per-LP budget: leave room for later SLP, but allow a solid first simplex resolve.
        lp_budget = min(remain, max(15.0, remain / max(2, max_iter - it + 1)))
        sol = _solve_lp(lp, engine, lp_tol, lp_budget, warm_basis=warm_basis, warm_sol=warm_sol,
                        backend=backend)
        xn = None
        if sol.x is not None and len(sol.x) >= p.n:
            xn = np.clip(x + damping * (np.asarray(sol.x[:p.n], float) - x), p.lx, p.ux)
        eng_status = sol.status
        if sol.verdict == "optimal":
            warm_basis = sol.extra.get("basis")
            warm_sol = sol
        elif eng_status == "infeasible" and not use_elastic:
            use_elastic = True
            warm_basis = warm_sol = None
            hist.append({"it": it, "event": "LP infeasible; enabling elastic", "phase": "homotopy"})
            if xn is None:
                continue
        elif sol.verdict != "optimal":
            # Keep a useful basis only after a certified optimal; otherwise cold next LP.
            warm_basis = warm_sol = None
            if not use_elastic:
                use_elastic = True
                hist.append({"it": it, "event": f"LP {sol.verdict}/{eng_status}; enabling elastic",
                             "phase": "homotopy"})
                if xn is None:
                    continue
            elif xn is None:
                stall += 1
                hist.append({"it": it, "event": f"LP {sol.verdict}/{eng_status}", "phase": "homotopy"})
                if stall >= 3:
                    status = "lp_failed"
                    break
                continue

        step = float(np.max(np.abs(xn - x))) if xn is not None and p.n else 0.0
        abs_v, rel_v = _violation_report(M, xn)
        hist.append({"it": it, "objective": M.objective(xn), "viol": abs_v, "rel_viol": rel_v,
                     "step": step, "phase": "homotopy", "lp_engine": sol.engine.get("engine"),
                     "lp_time": sol.engine.get("time"), "elastic": use_elastic,
                     "lp_verdict": sol.verdict, "lp_status": eng_status})
        if verbose:
            print(f"  homotopy {it:3d} obj {M.objective(xn):+.8e} viol {abs_v:.2e} "
                  f"step {step:.2e} [{sol.engine.get('engine')} {sol.engine.get('time', 0):.2f}s "
                  f"{sol.verdict}/{eng_status}]")
        if abs_v <= best_viol * (1.0 + 1e-6) + 1e-9:
            x = xn
            if abs_v < best_viol - 1e-12:
                best_viol = abs_v
                stall = 0
            else:
                stall += 1
        else:
            stall += 1
            # Reject worsening step; raise the penalty so the next LP prices violation harder.
            elastic_mu *= 2.0
            warm_basis = warm_sol = None
        if abs_v <= tol_viol or rel_v <= tol_viol:
            status = "converged"
            break
        if stall >= 3:
            # One undamped fixed-factor resolve before declaring stalled.
            lp2, _ = _fixed_factor_lp(M, x, fix_first, elastic_mu if use_elastic else None)
            remain = time_limit - (time.perf_counter() - t0)
            if remain > 0:
                sol2 = _solve_lp(lp2, engine, lp_tol, remain, warm_basis=warm_basis,
                                 warm_sol=warm_sol, backend=backend)
                if sol2.x is not None and len(sol2.x) >= p.n:
                    xn2 = np.clip(np.asarray(sol2.x[:p.n], float), p.lx, p.ux)
                    av2, rv2 = _violation_report(M, xn2)
                    hist.append({"it": it, "event": "undamped_retry", "viol": av2, "phase": "homotopy"})
                    if av2 < best_viol:
                        x, best_viol, stall = xn2, av2, 0
                        if sol2.verdict == "optimal":
                            warm_basis, warm_sol = sol2.extra.get("basis"), sol2
                        if av2 <= tol_viol or rv2 <= tol_viol:
                            status = "converged"
                            break
                        continue
            status = "stalled"
            break
        if stall >= 8:
            status = "stalled"
            break
    abs_v, rel_v = _violation_report(M, x)
    return SLPResult(x=x, objective=M.objective(x), max_violation=abs_v, rel_violation=rel_v,
                     feasible=bool(rel_v <= 1e-6), iterations=it, time=time.perf_counter() - t0,
                     history=hist, status=status, phase="homotopy")


def run_slp(M, x0=None, mu: float | None = None, max_iter: int = 200, tol_viol: float = 1e-6,
            tol_step: float = 1e-9, engine: str = "simplex", backend: str = "auto",
            lp_tol: float = 1e-8, time_limit: float = 3600.0, delta0: float = 0.25,
            verbose: bool = False, homotopy_first: bool = True, homotopy_iters: int = 40,
            homotopy_damping: float = 0.5) -> SLPResult:
    """Homotopy (optional) then penalty trust-region SLP with warm-started simplex/IPM LPs."""
    t0 = time.perf_counter()
    p = M.linear
    hist: list = []
    x = np.clip(M.start if x0 is None else np.asarray(x0, float), p.lx, p.ux)
    homo_iters = 0
    if homotopy_first:
        # Cap homotopy wall time so SLP still has budget to polish a feasible plan.
        remain = time_limit - (time.perf_counter() - t0)
        homo_budget = min(remain, max(5.0, 0.6 * remain)) if remain > 0 else 0.0
        if homo_budget > 0:
            hr = run_homotopy(M, x0=x, max_iter=homotopy_iters, tol_viol=tol_viol,
                              engine=engine, backend=backend, lp_tol=lp_tol,
                              time_limit=homo_budget, damping=homotopy_damping, verbose=verbose)
            hist.extend(hr.history)
            x = hr.x
            homo_iters = hr.iterations
            # Feasibility-first: keep the homotopy plan, then polish with trust-region SLP
            # (do not return early — a fixed-quality feasible point is often suboptimal).

    prod_vars = _product_vars(M)
    width = np.where(np.isfinite(p.ux[prod_vars] - p.lx[prod_vars]),
                     p.ux[prod_vars] - p.lx[prod_vars], np.inf)
    delta = np.maximum(delta0 * np.maximum(np.abs(x[prod_vars]), 1.0),
                       np.minimum(delta0 * width, 1e6))
    delta = np.where(np.isfinite(delta), delta, 1e6)
    if mu is None:
        mu = 10.0 * max(1.0, float(np.abs(p.c).max()))
    status = "iteration_limit"
    warm_basis = None
    warm_sol = None
    it = 0
    for it in range(1, max_iter + 1):
        remain = time_limit - (time.perf_counter() - t0)
        if remain <= 0:
            status = "time_limit"
            break
        lp, k = _linearised(M, x, delta, mu, prod_vars)
        used_warm = warm_basis is not None
        sol = _solve_lp(lp, engine, lp_tol, remain, warm_basis=warm_basis, warm_sol=warm_sol,
                        backend=backend)
        if sol.verdict != "optimal":
            hist.append({"it": it, "event": f"LP {sol.verdict}/{sol.status}", "delta": float(delta.max()),
                         "phase": "slp", "warm": used_warm})
            if sol.x is not None and len(sol.x) >= p.n and sol.status != "infeasible":
                xn = sol.x[:p.n]
                if _merit(M, xn, mu) < _merit(M, x, mu):
                    x = xn
                    warm_basis = sol.extra.get("basis")
                    warm_sol = sol
                    continue
            delta *= 0.5
            warm_basis = warm_sol = None
            if delta.max() < tol_step:
                status = "lp_failed"
                break
            continue
        warm_basis = sol.extra.get("basis")
        warm_sol = sol
        xn = sol.x[:p.n]
        slack = float(sol.x[p.n:].sum())
        phi, phi_n = _merit(M, x, mu), _merit(M, xn, mu)
        pred = phi - (float(p.c @ xn) + mu * slack)
        act = phi - phi_n
        rho = act / pred if pred > 1e-12 * (1 + abs(phi)) else (1.0 if act >= 0 else -1.0)
        step = float(np.max(np.abs(xn - x)[prod_vars])) if len(prod_vars) else 0.0
        viol = M.violation(xn)
        hist.append({"it": it, "objective": M.objective(xn), "viol": float(viol.max()), "rho": rho,
                     "delta": float(delta.max()), "mu": mu, "lp_engine": sol.engine.get("engine"),
                     "lp_time": sol.engine.get("time"), "phase": "slp",
                     "warm": used_warm})
        if verbose:
            print(f"  slp {it:3d} obj {M.objective(xn):+.8e} viol {viol.max():.2e} rho {rho:+.2f} "
                  f"delta {delta.max():.2e} mu {mu:.1e} step {step:.2e} [{sol.engine.get('engine')} "
                  f"{sol.engine.get('time', 0):.1f}s]")
        if rho > 1e-4:
            x = xn
            if rho > 0.75 and step >= 0.9 * float(np.min(delta)):
                delta = np.minimum(2.0 * delta, 1e8)
        else:
            delta *= 0.25
        cur_viol = float(M.violation(x).max())
        if cur_viol > tol_viol and it % 5 == 0 and slack > 0:
            mu *= 4.0
        if cur_viol <= tol_viol and (step <= tol_step * (1 + np.abs(x[prod_vars]).max()) or
                                     abs(act) <= 1e-10 * (1 + abs(phi))):
            status = "converged"
            break
        if delta.max() < tol_step:
            status = "trust_region_collapsed"
            break
    abs_v, rel_v = _violation_report(M, x)
    return SLPResult(x=x, objective=M.objective(x), max_violation=abs_v, rel_violation=rel_v,
                     feasible=bool(rel_v <= 1e-6), iterations=homo_iters + it,
                     time=time.perf_counter() - t0, history=hist, status=status, phase="combined")
