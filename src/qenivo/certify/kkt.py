"""Residuals that decide every verdict, on the original (unscaled) problem.

One formula scores every answer (inside the engine loop, at the end, and for a comparator's
returned point). Works on 1-D vectors or on batches (n, S) / (m, S), with NumPy or CuPy.

Optimality (primal x, row duals y):
    r_p = [Ax - clip(Ax, lc, uc) ; x - clip(x, lx, ux)]
    lam = Qx + c - A'y                           reduced costs
    r_d = part of lam the variable bounds do not allow  +  row-dual cone violation of y
    P   = 0.5 x'Qx + c'x + c0
    D   = -0.5 x'Qx + sum(y+ lc - y- uc) + sum(lam+ lx - lam- ux) + c0   (finite terms only)
    optimal iff ||r_p|| <= eps(1+||b||), ||r_d|| <= eps(1+||c||), |P-D| <= eps(1+|P|+|D|)

Infeasibility (Farkas ray y):   with lam = -A'y, the homogeneous dual objective
    phi(y) = sum(y+ lc - y- uc) + sum(lam+ lx - lam- ux)  > 0
while y and lam respect the sign rules of the finite bounds. Such a y proves no x exists.

Unboundedness (primal ray d, LP):  c'd < 0, (Ad)_i >= 0 where lc_i is finite,
(Ad)_i <= 0 where uc_i is finite, d_j >= 0 where lx_j is finite, d_j <= 0 where ux_j is finite.
"""
from __future__ import annotations

import numpy as np


def rhs_norm(lc, uc, xp=np):
    """||bhat||_2 with bhat_i = the finite bound(s) of row i (max of |l|,|u| if ranged)."""
    lf = xp.where(xp.isfinite(lc), xp.abs(lc), 0.0)
    uf = xp.where(xp.isfinite(uc), xp.abs(uc), 0.0)
    return xp.sqrt(xp.sum(xp.maximum(lf, uf) ** 2, axis=0))


def _finite(v, xp):
    return xp.where(xp.isfinite(v), v, 0.0)


def kkt_from_products(x, y, Ax, ATy, c, c0, lc, uc, lx, ux, bnorm, cnorm, Qx=None, xp=np) -> dict:
    """Residuals given the products Ax and A'y (and Qx for QP) in original space."""
    rp_con = Ax - xp.clip(Ax, lc, uc)
    rp_var = x - xp.clip(x, lx, ux)
    primal_res = xp.sqrt(xp.sum(rp_con ** 2, axis=0) + xp.sum(rp_var ** 2, axis=0))

    lam = c - ATy
    if Qx is not None:
        lam = lam + Qx
    lam_pos, lam_neg = xp.maximum(lam, 0.0), xp.maximum(-lam, 0.0)
    rd_var = xp.where(xp.isfinite(lx), 0.0, lam_pos) - xp.where(xp.isfinite(ux), 0.0, lam_neg)
    y_pos, y_neg = xp.maximum(y, 0.0), xp.maximum(-y, 0.0)
    rd_row = xp.where(xp.isfinite(lc), 0.0, y_pos) - xp.where(xp.isfinite(uc), 0.0, y_neg)
    dual_res = xp.sqrt(xp.sum(rd_var ** 2, axis=0) + xp.sum(rd_row ** 2, axis=0))

    pobj = xp.sum(c * x, axis=0) + c0
    dobj = (xp.sum(y_pos * _finite(lc, xp) - y_neg * _finite(uc, xp), axis=0)
            + xp.sum(lam_pos * _finite(lx, xp) - lam_neg * _finite(ux, xp), axis=0) + c0)
    if Qx is not None:
        quad = 0.5 * xp.sum(x * Qx, axis=0)
        pobj = pobj + quad
        dobj = dobj - quad
    gap = xp.abs(pobj - dobj)
    return {
        "primal_res": primal_res, "dual_res": dual_res, "gap": gap, "pobj": pobj, "dobj": dobj,
        "rel_primal": primal_res / (1.0 + bnorm),
        "rel_dual": dual_res / (1.0 + cnorm),
        "rel_gap": gap / (1.0 + xp.abs(pobj) + xp.abs(dobj)),
    }


def is_optimal(k: dict, eps: float, xp=np):
    return (k["rel_primal"] <= eps) & (k["rel_dual"] <= eps) & (k["rel_gap"] <= eps)


def kkt_residuals(prob, x, y) -> dict:
    """Host-side check in float64. Returns plain floats; objectives in the user's sense."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    Qx = prob.Q @ x if prob.is_qp else None
    k = kkt_from_products(x, y, prob.A @ x, prob.A.T @ y, prob.c, prob.c0, prob.lc, prob.uc,
                          prob.lx, prob.ux, rhs_norm(prob.lc, prob.uc), np.linalg.norm(prob.c), Qx=Qx)
    out = {key: float(v) for key, v in k.items()}
    out["objective"] = prob.obj_sign * out["pobj"]
    out["dual_objective"] = prob.obj_sign * out["dobj"]
    out["max_rel"] = max(out["rel_primal"], out["rel_dual"], out["rel_gap"])
    return out


# ---------------------------------------------------------------------- rays
def farkas_products(y, ATy, lc, uc, lx, ux, xp=np):
    """Score a candidate Farkas ray. Returns (value, violation) per column of y.

    value     = phi(y) as defined above (must be > 0)
    violation = size of the sign-rule violations (must be ~0 relative to value)
    """
    lam = -ATy
    y_pos, y_neg = xp.maximum(y, 0.0), xp.maximum(-y, 0.0)
    lam_pos, lam_neg = xp.maximum(lam, 0.0), xp.maximum(-lam, 0.0)
    viol_row = xp.where(xp.isfinite(lc), 0.0, y_pos) + xp.where(xp.isfinite(uc), 0.0, y_neg)
    viol_col = xp.where(xp.isfinite(lx), 0.0, lam_pos) + xp.where(xp.isfinite(ux), 0.0, lam_neg)
    value = (xp.sum(y_pos * _finite(lc, xp) - y_neg * _finite(uc, xp), axis=0)
             + xp.sum(lam_pos * _finite(lx, xp) - lam_neg * _finite(ux, xp), axis=0))
    violation = xp.sqrt(xp.sum(viol_row ** 2, axis=0) + xp.sum(viol_col ** 2, axis=0))
    return value, violation


def ray_products(d, Ad, c, lc, uc, lx, ux, xp=np):
    """Score a candidate unbounded ray. Returns (descent, violation): descent = -c'd must be > 0."""
    descent = -xp.sum(c * d, axis=0)
    v_row = xp.where(xp.isfinite(lc), xp.maximum(-Ad, 0.0), 0.0) + xp.where(xp.isfinite(uc), xp.maximum(Ad, 0.0), 0.0)
    v_col = xp.where(xp.isfinite(lx), xp.maximum(-d, 0.0), 0.0) + xp.where(xp.isfinite(ux), xp.maximum(d, 0.0), 0.0)
    violation = xp.sqrt(xp.sum(v_row ** 2, axis=0) + xp.sum(v_col ** 2, axis=0))
    return descent, violation


def check_farkas(prob, y, tol: float = 1e-8) -> dict:
    """Host check of a Farkas certificate for primal infeasibility of `prob`."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    scale = np.linalg.norm(y)
    if not np.isfinite(scale) or scale == 0.0:
        return {"valid": False, "reason": "zero or non-finite ray"}
    y = y / scale
    value, violation = farkas_products(y, prob.A.T @ y, prob.lc, prob.uc, prob.lx, prob.ux)
    value, violation = float(value), float(violation)
    ok = value > 0 and violation <= tol * max(1.0, value) * 1e3 and violation < value
    return {"valid": bool(ok), "value": value, "violation": violation,
            "ratio": violation / value if value > 0 else float("inf")}


def check_ray(prob, d, tol: float = 1e-8) -> dict:
    """Host check of an unbounded ray for an LP."""
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    scale = np.linalg.norm(d)
    if not np.isfinite(scale) or scale == 0.0:
        return {"valid": False, "reason": "zero or non-finite ray"}
    d = d / scale
    descent, violation = ray_products(d, prob.A @ d, prob.c, prob.lc, prob.uc, prob.lx, prob.ux)
    descent, violation = float(descent), float(violation)
    qd = float(np.linalg.norm(prob.Q @ d)) if prob.is_qp else 0.0
    ok = descent > 0 and violation <= tol * 1e3 * max(1.0, descent) and violation < descent and qd <= 1e-8
    return {"valid": bool(ok), "descent": descent, "violation": violation}
