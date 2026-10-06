"""Primal-dual interior-point method with a filter line search.

A. Waechter and L. T. Biegler, On the implementation of an interior-point
filter line-search algorithm for large-scale nonlinear programming,
Mathematical Programming 106 (2006) 25-57.

The barrier KKT system is factored by :mod:`qenivo.nlp.dense_kkt`. When the
inertia is not (n free variables, m equalities, 0), a multiple of the identity
is added to the Hessian (δ_w) and, if the constraint block is singular, a
negative multiple (δ_c) is added there. The line search accepts a step that
either decreases the barrier objective enough (an f-type step) or is acceptable
to the filter of constraint violation and barrier objective (an h-type step).
One second-order correction is tried when the full step is rejected. If the
step length collapses, a restoration phase reduces the constraint violation by
Gauss-Newton; that is a shorter restoration than the second filter problem in
the paper, and it is recorded on the result when it runs.

IPOPT is not used.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .ad import gradient, lagrangian_hessian
from .dense_kkt import factor
from .expr import evaluate
from .kkt import kkt_residual
from .problem import NLPModel

# Waechter and Biegler, Section 3, the constants used by the reference filter.
_GAMMA_TH = 1e-5
_GAMMA_PH = 1e-5
_DELTA = 1.0
_S_TH = 1.1
_S_PH = 2.3
_ETA = 1e-4
_KAPPA_EPS = 10.0
_KAPPA_MU = 0.2
_THETA_MU = 1.5
_TAU_MIN = 0.99


@dataclass
class NLPResult:
    """Local NLP solve. ``residuals`` is the KKT certificate from :func:`kkt_residual`."""

    status: str
    objective: float | None
    x: np.ndarray | None
    lam: np.ndarray | None
    residuals: dict
    engine: dict
    extra: dict = field(default_factory=dict)
    source: str | None = None
    iterations: int = 0
    notes: list = field(default_factory=list)
    z_lower: np.ndarray | None = None
    z_upper: np.ndarray | None = None


class _Filter:
    def __init__(self):
        self.pts: list[tuple[float, float]] = []

    def accepts(self, th: float, ph: float) -> bool:
        for th_i, ph_i in self.pts:
            if th > (1.0 - _GAMMA_TH) * th_i and ph > ph_i - _GAMMA_PH * th_i:
                return False
        return True

    def add(self, th: float, ph: float) -> None:
        self.pts = [(a, b) for a, b in self.pts if not (a >= th and b >= ph)]
        self.pts.append((float(th), float(ph)))

    def clear(self) -> None:
        self.pts.clear()


def solve_ipm(model: NLPModel, tol: float = 1e-8, time_limit: float = 30.0,
              x0: np.ndarray | None = None, lb: np.ndarray | None = None,
              ub: np.ndarray | None = None, verbose: bool = False, max_iter: int = 80) -> NLPResult:
    """Solve ``model`` by the filter interior-point method.

    Bounds default to the graph bounds. Passing ``lb`` and ``ub`` fixes
    variables (equal finite bounds are removed from the Newton system) without
    changing the graph, which is how outer approximation freezes integers.
    """
    t0 = time.perf_counter()
    g = model.graph
    n0 = g.n_var
    lb_a = np.asarray(g.lb if lb is None else lb, dtype=np.float64).copy()
    ub_a = np.asarray(g.ub if ub is None else ub, dtype=np.float64).copy()
    if lb_a.shape != (n0,) or ub_a.shape != (n0,):
        raise ValueError("bound vectors must match the number of variables")
    n_in = len(model.inequalities)
    N = n0 + n_in
    full_lb = np.concatenate([lb_a, np.zeros(n_in)])
    full_ub = np.concatenate([ub_a, np.full(n_in, np.inf)])
    v = _start(full_lb, full_ub, x0, n0)
    for j, ineq in enumerate(model.inequalities):
        gv = _value(ineq, v[:n0])
        v[n0 + j] = max(1.0, -gv) if np.isfinite(gv) else 1.0
    fixed = np.array([np.isfinite(full_lb[i]) and np.isfinite(full_ub[i]) and full_ub[i] - full_lb[i] <= 1e-9
                      for i in range(N)])
    for i in np.flatnonzero(fixed):
        v[i] = full_lb[i]
    free = np.flatnonzero(~fixed)
    nf = int(free.size)
    m = len(model.equalities) + n_in
    has_bounds = any(np.isfinite(full_lb[i]) or np.isfinite(full_ub[i]) for i in free)
    mu = 0.1 if has_bounds else 0.0
    zL = np.zeros(N)
    zU = np.zeros(N)
    _reset_slacks(zL, zU, v, full_lb, full_ub, free, mu if mu > 0 else 0.1)
    lam = np.zeros(m)
    filt = _Filter()
    notes: list[str] = []
    iterations = 0
    restorations = 0
    inertia_fixes = 0
    th0 = _theta(model, v, n0)
    theta_max = max(1e4, 10.0 * th0 + 1.0)
    status = "iteration_limit"
    cons_exprs = [*model.equalities, *model.inequalities]

    def remaining() -> float:
        return time_limit - (time.perf_counter() - t0)

    while iterations < max_iter and remaining() > 0:
        iterations += 1
        _absorb_fixed(model, v[:n0], lam, zL[:n0], zU[:n0], lb_a, ub_a)
        res_now = kkt_residual(model, v[:n0], lam, zL[:n0], zU[:n0], lb_a, ub_a)
        if res_now["max_res"] <= tol and (not has_bounds or mu <= max(tol, 1e-12)):
            status = "optimal"
            break
        if has_bounds and _barrier_error(model, v, lam, zL, zU, n0, full_lb, full_ub, mu) <= _KAPPA_EPS * mu:
            mu = max(tol / 10.0, min(_KAPPA_MU * mu, mu ** _THETA_MU))
            if verbose:
                print(f"nlp-ipm mu -> {mu:.3e}")
            continue
        J = _jacobian(model, v, n0)
        sigma = _sigma(v, zL, zU, full_lb, full_ub, free)
        H = _reduced_hessian(model, cons_exprs, lam, v[:n0], free, n0)
        rhs_x, cval = _rhs(model, v, lam, mu, n0, full_lb, full_ub, free, J)
        fac, dw, dc, nfix = _corrected_factor(H, J[:, free], sigma, mu)
        inertia_fixes += nfix
        if fac is None:
            notes.append("inertia correction did not produce the required inertia")
            status = "not_converged"
            break
        step = fac.solve(np.concatenate([rhs_x, -cval]))
        if not np.all(np.isfinite(step)):
            notes.append("KKT solve was not finite")
            status = "not_converged"
            break
        d = np.zeros(N)
        d[free] = step[:nf]
        dlam = step[nf:]
        alpha_max = _fraction_primal(v, d, full_lb, full_ub, free, mu)
        th = _theta(model, v, n0)
        ph = _phi(model, v, mu, full_lb, full_ub, free, n0)
        slope = _slope(model, v, d, mu, full_lb, full_ub, free, n0)
        accepted = False
        augment = False
        alpha = alpha_max
        tried_soc = False
        d_used = d
        dlam_used = dlam
        for _bt in range(40):
            if alpha <= 1e-16:
                break
            trial = v + alpha * d_used
            if _acceptable(model, trial, v, alpha, slope, th, ph, mu, full_lb, full_ub, free, n0,
                           filt, theta_max, d_used):
                accepted = True
                augment = not _is_f_step(slope, alpha, th)
                break
            if not tried_soc and m > 0 and alpha == alpha_max:
                tried_soc = True
                soc = _soc_direction(fac, model, trial, nf, N, free)
                if soc is not None:
                    d_soc = d + soc[0]
                    dlam_soc = dlam + soc[1]
                    a_soc = _fraction_primal(v, d_soc, full_lb, full_ub, free, mu)
                    trial_s = v + a_soc * d_soc
                    # The paper tests the corrected point against the filter.
                    if _h_accept(model, trial_s, th, ph, mu, full_lb, full_ub, free, n0, filt, theta_max):
                        d_used, dlam_used, alpha = d_soc, dlam_soc, a_soc
                        accepted = True
                        augment = True
                        break
            alpha *= 0.5
        if not accepted:
            restorations += 1
            restored = _restore(model, v, free, full_lb, full_ub, n0, remaining())
            if restored is None:
                notes.append("restoration did not reduce the constraint violation")
                status = "not_converged"
                break
            v = restored
            filt.clear()
            notes.append("restoration phase reduced the constraint violation and cleared the filter")
            _reset_slacks(zL, zU, v, full_lb, full_ub, free, max(mu, 1e-8))
            lam[:] = 0.0
            continue
        if augment:
            filt.add(th, ph)
        dzL, dzU = _dual_direction(v, d_used, zL, zU, mu, full_lb, full_ub, free)
        a_d = _fraction_dual(zL, zU, dzL, dzU, free, mu)
        v = v + alpha * d_used
        lam = lam + a_d * dlam_used
        zL = zL + a_d * dzL
        zU = zU + a_d * dzU
        for i in free:
            if np.isfinite(full_lb[i]):
                zL[i] = max(zL[i], 0.0)
            if np.isfinite(full_ub[i]):
                zU[i] = max(zU[i], 0.0)
        if verbose:
            print(f"nlp-ipm it {iterations} th {th:.3e} alpha {alpha:.3e} mu {mu:.3e} dw {dw:.1e}")
    x = v[:n0].copy()
    zL_o = zL[:n0].copy()
    zU_o = zU[:n0].copy()
    _absorb_fixed(model, x, lam, zL_o, zU_o, lb_a, ub_a)
    cert = kkt_residual(model, x, lam, zL_o, zU_o, lb_a, ub_a)
    if cert["max_res"] <= tol:
        status = "optimal"
    elif status == "optimal":
        status = "not_converged"
        notes.append(f"step met the barrier test but the KKT residual is {cert['max_res']:.3e}")
    engine = {
        "engine": "nlp-ipm",
        "backend": "numpy",
        "iterations": iterations,
        "time": time.perf_counter() - t0,
        "mu": mu,
        "restorations": restorations,
        "inertia_corrections": inertia_fixes,
        "reason": "nonlinear program, filter line-search interior point",
    }
    obj = cert["objective"] if np.isfinite(cert["objective"]) else None
    return NLPResult(status=status, objective=obj, x=x, lam=lam, residuals=cert, engine=engine,
                     iterations=iterations, notes=notes, z_lower=zL_o, z_upper=zU_o,
                     extra={"mu": mu, "restorations": restorations})


def _start(lb, ub, x0, n0) -> np.ndarray:
    v = np.zeros(len(lb))
    for i, (lo, hi) in enumerate(zip(lb, ub)):
        if np.isfinite(lo) and np.isfinite(hi) and hi - lo <= 1e-12:
            v[i] = lo
        elif np.isfinite(lo) and np.isfinite(hi):
            v[i] = 0.5 * (lo + hi)
        elif np.isfinite(lo):
            v[i] = lo + 1.0
        elif np.isfinite(hi):
            v[i] = hi - 1.0
    if x0 is not None:
        x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
        v[:n0] = x0
        for i in range(n0):
            lo, hi = lb[i], ub[i]
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo + 1e-12:
                margin = 1e-3 * (hi - lo)
                v[i] = min(max(v[i], lo + margin), hi - margin)
            elif np.isfinite(lo) and np.isfinite(hi):
                v[i] = lo
            elif np.isfinite(lo):
                v[i] = max(v[i], lo + 1e-3)
            elif np.isfinite(hi):
                v[i] = min(v[i], hi - 1e-3)
    return v


def _value(expr, x) -> float:
    try:
        return evaluate(expr, x)
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return float("nan")


def _reset_slacks(zL, zU, v, lb, ub, free, mu: float) -> None:
    zL[:] = 0.0
    zU[:] = 0.0
    for i in free:
        if np.isfinite(lb[i]) and v[i] > lb[i]:
            zL[i] = mu / (v[i] - lb[i])
        if np.isfinite(ub[i]) and ub[i] > v[i]:
            zU[i] = mu / (ub[i] - v[i])


def _theta(model: NLPModel, v: np.ndarray, n0: int) -> float:
    c = _constraints(model, v, n0)
    if not np.all(np.isfinite(c)):
        return float("inf")
    return float(np.sum(np.abs(c)))


def _constraints(model: NLPModel, v: np.ndarray, n0: int) -> np.ndarray:
    x = v[:n0]
    vals = [_value(e, x) for e in model.equalities]
    for j, ineq in enumerate(model.inequalities):
        vals.append(_value(ineq, x) + v[n0 + j])
    return np.asarray(vals, dtype=np.float64) if vals else np.zeros(0)


def _phi(model, v, mu, lb, ub, free, n0) -> float:
    f = _value(model.objective, v[:n0])
    if not np.isfinite(f):
        return float("inf")
    s = 0.0
    if mu > 0:
        for i in free:
            if np.isfinite(lb[i]):
                gap = v[i] - lb[i]
                if gap <= 0:
                    return float("inf")
                s += np.log(gap)
            if np.isfinite(ub[i]):
                gap = ub[i] - v[i]
                if gap <= 0:
                    return float("inf")
                s += np.log(gap)
    return f - mu * s


def _jacobian(model: NLPModel, v: np.ndarray, n0: int) -> np.ndarray:
    x = v[:n0]
    m = len(model.equalities) + len(model.inequalities)
    J = np.zeros((m, len(v)))
    for i, e in enumerate(model.equalities):
        J[i, :n0] = gradient(e, x)
    for j, ineq in enumerate(model.inequalities):
        J[len(model.equalities) + j, :n0] = gradient(ineq, x)
        J[len(model.equalities) + j, n0 + j] = 1.0
    return J


def _sigma(v, zL, zU, lb, ub, free) -> np.ndarray:
    s = np.zeros(len(free))
    for k, i in enumerate(free):
        if np.isfinite(lb[i]) and v[i] > lb[i]:
            s[k] += zL[i] / (v[i] - lb[i])
        if np.isfinite(ub[i]) and ub[i] > v[i]:
            s[k] += zU[i] / (ub[i] - v[i])
    return s


def _reduced_hessian(model, cons, lam, x, free, n0) -> np.ndarray:
    nf = len(free)
    # Lagrangian multipliers hit original equalities and inequalities only.
    # Slack rows of the Hessian are zero because slacks enter linearly.
    n_con = len(cons)
    lam_c = lam[:n_con]
    H, _info = lagrangian_hessian(model.objective, cons, lam_c, x)
    orig = [int(i) for i in free if int(i) < n0]
    pos = {int(i): k for k, i in enumerate(free)}
    Hr = np.zeros((nf, nf))
    for i in orig:
        for j in orig:
            Hr[pos[i], pos[j]] = H[i, j]
    return (Hr + Hr.T) * 0.5


def _rhs(model, v, lam, mu, n0, lb, ub, free, J):
    g = np.zeros(len(v))
    gf = gradient(model.objective, v[:n0])
    g[:n0] = gf
    # J is (m, N); stationarity uses J.T @ lam
    grad = g + J.T @ lam
    rhs = np.zeros(len(free))
    for k, i in enumerate(free):
        term = grad[i]
        if mu > 0 and np.isfinite(lb[i]) and v[i] > lb[i]:
            term -= mu / (v[i] - lb[i])
        if mu > 0 and np.isfinite(ub[i]) and ub[i] > v[i]:
            term += mu / (ub[i] - v[i])
        rhs[k] = -term
    cval = _constraints(model, v, n0)
    return rhs, cval


def _corrected_factor(H, J, sigma, mu):
    """Return a factor whose inertia is (n, m, 0), plus the regularisation used."""
    n, m = H.shape[0], J.shape[0]
    dw = 0.0
    dc = 0.0
    last = None
    for _ in range(20):
        K = np.zeros((n + m, n + m))
        K[:n, :n] = H + np.diag(sigma)
        if dw:
            K[:n, :n] += dw * np.eye(n)
        if m:
            K[:n, n:] = J.T
            K[n:, :n] = J
            if dc:
                K[n:, n:] = -dc * np.eye(m)
        last = factor(K)
        pos, neg, zer = last.inertia
        if pos == n and neg == m and zer == 0 and np.all(np.isfinite(last.M)):
            return last, dw, dc, int(dw > 0 or dc > 0)
        if zer > 0 or neg < m:
            dc = (dc * 8.0) if dc > 0 else max(1e-8 * max(mu, 1e-8) ** 0.25, 1e-12)
        if pos < n or neg > m or zer > 0:
            dw = (dw * 8.0) if dw > 0 else 1e-4 * max(1.0, float(np.linalg.norm(H, ord=np.inf)))
    return None, dw, dc, 1


def _fraction_primal(v, d, lb, ub, free, mu) -> float:
    tau = _TAU_MIN if mu <= 0 else max(_TAU_MIN, 1.0 - mu)
    alpha = 1.0
    for i in free:
        if d[i] < 0 and np.isfinite(lb[i]):
            gap = v[i] - lb[i]
            if gap > 0:
                alpha = min(alpha, tau * gap / (-d[i]))
        if d[i] > 0 and np.isfinite(ub[i]):
            gap = ub[i] - v[i]
            if gap > 0:
                alpha = min(alpha, tau * gap / d[i])
    return max(0.0, float(alpha))


def _fraction_dual(zL, zU, dzL, dzU, free, mu) -> float:
    tau = _TAU_MIN if mu <= 0 else max(_TAU_MIN, 1.0 - mu)
    alpha = 1.0
    for i in free:
        if dzL[i] < 0 and zL[i] > 0:
            alpha = min(alpha, tau * zL[i] / (-dzL[i]))
        if dzU[i] < 0 and zU[i] > 0:
            alpha = min(alpha, tau * zU[i] / (-dzU[i]))
    return max(0.0, float(alpha))


def _dual_direction(v, d, zL, zU, mu, lb, ub, free):
    dzL = np.zeros_like(v)
    dzU = np.zeros_like(v)
    if mu <= 0:
        # No barrier: drive bound multipliers to zero if they were unused.
        return dzL, dzU
    for i in free:
        if np.isfinite(lb[i]) and v[i] > lb[i]:
            sl = v[i] - lb[i]
            dzL[i] = mu / sl - zL[i] - (zL[i] / sl) * d[i]
        if np.isfinite(ub[i]) and ub[i] > v[i]:
            su = ub[i] - v[i]
            dzU[i] = mu / su - zU[i] + (zU[i] / su) * d[i]
    return dzL, dzU


def _slope(model, v, d, mu, lb, ub, free, n0) -> float:
    g = np.zeros(len(v))
    g[:n0] = gradient(model.objective, v[:n0])
    if mu > 0:
        for i in free:
            if np.isfinite(lb[i]) and v[i] > lb[i]:
                g[i] -= mu / (v[i] - lb[i])
            if np.isfinite(ub[i]) and ub[i] > v[i]:
                g[i] += mu / (ub[i] - v[i])
    return float(g @ d)


def _is_f_step(slope: float, alpha: float, th: float) -> bool:
    if not np.isfinite(slope) or slope >= 0:
        return False
    return alpha * ((-slope) ** _S_PH) > _DELTA * (max(th, 0.0) ** _S_TH)


def _h_accept(model, trial, th, ph, mu, lb, ub, free, n0, filt, theta_max) -> bool:
    th_t = _theta(model, trial, n0)
    ph_t = _phi(model, trial, mu, lb, ub, free, n0)
    if not np.isfinite(th_t) or not np.isfinite(ph_t) or th_t > theta_max:
        return False
    if not (th_t <= (1.0 - _GAMMA_TH) * th or ph_t <= ph - _GAMMA_PH * th):
        return False
    return filt.accepts(th_t, ph_t)


def _acceptable(model, trial, v, alpha, slope, th, ph, mu, lb, ub, free, n0, filt, theta_max, d) -> bool:
    if _is_f_step(slope, alpha, th):
        ph_t = _phi(model, trial, mu, lb, ub, free, n0)
        if np.isfinite(ph_t) and ph_t <= ph + _ETA * alpha * slope:
            return True
        return False
    return _h_accept(model, trial, th, ph, mu, lb, ub, free, n0, filt, theta_max)


def _soc_direction(fac, model, trial, nf, N, free):
    c = _constraints(model, trial, model.graph.n_var)
    if not np.all(np.isfinite(c)) or c.size == 0:
        return None
    rhs = np.zeros(nf + c.size)
    rhs[nf:] = -c
    step = fac.solve(rhs)
    if not np.all(np.isfinite(step)):
        return None
    d = np.zeros(N)
    d[free] = step[:nf]
    return d, step[nf:]


def _restore(model, v, free, lb, ub, n0, time_left) -> np.ndarray | None:
    """Gauss-Newton on 1/2 ||c||^2, kept inside the bounds by fraction-to-boundary."""
    if time_left <= 0 or len(model.equalities) + len(model.inequalities) == 0:
        return None
    t0 = time.perf_counter()
    best = v.copy()
    best_th = _theta(model, v, n0)
    cur = v.copy()
    for _ in range(12):
        if time.perf_counter() - t0 > time_left:
            break
        c = _constraints(model, cur, n0)
        th = float(np.sum(np.abs(c))) if np.all(np.isfinite(c)) else float("inf")
        if th < best_th:
            best, best_th = cur.copy(), th
        if th <= 0.1 * _theta(model, v, n0):
            break
        J = _jacobian(model, cur, n0)[:, free]
        H = J.T @ J
        g = J.T @ c
        dw = 0.0
        fac = None
        nf = len(free)
        for _try in range(8):
            fac = factor(H + dw * np.eye(nf))
            if fac.inertia[0] == nf and fac.inertia[2] == 0:
                break
            dw = 1e-4 * max(1.0, float(np.linalg.norm(H, ord=np.inf))) if dw == 0 else dw * 8
        else:
            break
        d_red = fac.solve(-g)
        if not np.all(np.isfinite(d_red)):
            break
        d = np.zeros_like(cur)
        d[free] = d_red
        alpha = _fraction_primal(cur, d, lb, ub, free, 0.0)
        improved = False
        while alpha > 1e-16:
            trial = cur + alpha * d
            th_t = _theta(model, trial, n0)
            if th_t < th:
                cur = trial
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break
    if best_th < _theta(model, v, n0) * (1.0 - 1e-4):
        return best
    return None


def _barrier_error(model, v, lam, zL, zU, n0, lb, ub, mu) -> float:
    """Infinity-norm of the primal-dual residual at the current barrier parameter."""
    x = v[:n0]
    stat = gradient(model.objective, x)
    c = _constraints(model, v, n0)
    J = _jacobian(model, v, n0)
    stat_full = np.zeros(len(v))
    stat_full[:n0] = stat
    stat_full = stat_full + J.T @ lam - zL + zU
    comp = []
    for i in range(len(v)):
        if np.isfinite(lb[i]):
            comp.append((v[i] - lb[i]) * zL[i] - mu)
        if np.isfinite(ub[i]):
            comp.append((ub[i] - v[i]) * zU[i] - mu)
    pieces = [float(np.linalg.norm(stat_full, ord=np.inf)),
              float(np.linalg.norm(c, ord=np.inf)) if c.size else 0.0]
    if comp:
        pieces.append(float(np.max(np.abs(comp))))
    return max(pieces)


def _absorb_fixed(model, x, lam, zL, zU, lb, ub) -> None:
    """Put the reduced gradient of a fixed variable into its bound multipliers.

    A variable with equal bounds is an equality. Stationarity holds by choosing
    nonnegative z_L, z_U with z_U - z_L = -∂L/∂x_i, and complementarity holds
    because the slack of each bound is zero.
    """
    n = len(x)
    stat = gradient(model.objective, x)
    cons = [*model.equalities, *model.inequalities]
    for w, c in zip(lam, cons):
        if w != 0.0:
            stat = stat + float(w) * gradient(c, x)
    for i in range(n):
        if np.isfinite(lb[i]) and np.isfinite(ub[i]) and ub[i] - lb[i] <= 1e-8:
            r = float(stat[i])
            if r >= 0.0:
                zL[i], zU[i] = r, 0.0
            else:
                zL[i], zU[i] = 0.0, -r
