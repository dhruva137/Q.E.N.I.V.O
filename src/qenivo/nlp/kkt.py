"""First-order KKT residual for a local NLP solution.

For min f(x) subject to c_E(x) = 0, c_I(x) <= 0 and l <= x <= u, a point with
multipliers is a local first-order candidate when the stationarity residual

    ∇f + Σ λ_i ∇c_i - z_L + z_U

the primal violation, bound-multiplier signs, and the complementarity
products z_L (x-l), z_U (u-x) and λ_I c_I are all small. This certifies a
local stationary point. It is not a global optimality proof; the pooling
branch-and-bound carries that proof as a relaxation bound.
"""
from __future__ import annotations

import numpy as np

from .ad import gradient
from .problem import NLPModel


def kkt_residual(model: NLPModel, x: np.ndarray, lam: np.ndarray | None,
                 z_lower: np.ndarray | None, z_upper: np.ndarray | None,
                 lb: np.ndarray | None = None, ub: np.ndarray | None = None) -> dict:
    """Absolute KKT residuals of ``(x, λ, z)`` on ``model``.

    ``lam`` is ordered as equalities then inequalities. Missing multipliers are
    treated as zero, which makes a non-stationary point fail the stationarity test.
    ``lb`` and ``ub`` override the graph bounds when a variable was fixed for the solve.
    """
    g = model.graph
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape != (g.n_var,):
        raise ValueError(f"x has length {x.shape[0]}, model has {g.n_var} variables")
    lb = np.asarray(g.lb if lb is None else lb, dtype=np.float64)
    ub = np.asarray(g.ub if ub is None else ub, dtype=np.float64)
    n_eq = len(model.equalities)
    n_in = len(model.inequalities)
    m = n_eq + n_in
    lam_v = np.zeros(m) if lam is None else np.asarray(lam, dtype=np.float64).reshape(-1)
    if lam_v.shape != (m,):
        raise ValueError(f"lambda has length {lam_v.shape[0]}, expected {m}")
    zL = np.zeros(g.n_var) if z_lower is None else np.asarray(z_lower, dtype=np.float64).reshape(-1)
    zU = np.zeros(g.n_var) if z_upper is None else np.asarray(z_upper, dtype=np.float64).reshape(-1)
    from .expr import evaluate
    cons = [*model.equalities, *model.inequalities]
    stat = gradient(model.objective, x)
    cval = np.zeros(m)
    for i, c in enumerate(cons):
        cval[i] = evaluate(c, x)
        stat = stat + lam_v[i] * gradient(c, x)
    for i in range(g.n_var):
        if np.isfinite(lb[i]):
            stat[i] -= zL[i]
        if np.isfinite(ub[i]):
            stat[i] += zU[i]
    eq_viol = np.max(np.abs(cval[:n_eq])) if n_eq else 0.0
    in_viol = np.max(np.maximum(cval[n_eq:], 0.0)) if n_in else 0.0
    bound_viol = 0.0
    comp = []
    dual_neg = 0.0
    for i in range(g.n_var):
        if np.isfinite(lb[i]):
            bound_viol = max(bound_viol, float(lb[i] - x[i]))
            comp.append(zL[i] * (x[i] - lb[i]))
            dual_neg = max(dual_neg, float(-zL[i]))
        if np.isfinite(ub[i]):
            bound_viol = max(bound_viol, float(x[i] - ub[i]))
            comp.append(zU[i] * (ub[i] - x[i]))
            dual_neg = max(dual_neg, float(-zU[i]))
    for j in range(n_in):
        comp.append(lam_v[n_eq + j] * cval[n_eq + j])
        dual_neg = max(dual_neg, float(-lam_v[n_eq + j]))
    stationarity = float(np.linalg.norm(stat, ord=np.inf))
    complementarity = float(np.max(np.abs(comp))) if comp else 0.0
    primal = float(max(eq_viol, in_viol, bound_viol))
    return {
        "stationarity": stationarity,
        "primal": primal,
        "complementarity": complementarity,
        "dual_feasibility": float(dual_neg),
        "max_res": max(stationarity, primal, complementarity, float(dual_neg)),
        "objective": evaluate(model.objective, x),
    }
