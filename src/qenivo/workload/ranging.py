"""Exact sensitivity ranging from the optimal simplex basis.

Standard form used by the simplex engine: z = [x; s], [A -I] z = 0, bounds on x and on the
row activities s. At an optimal basis B (index set of m basic columns):

Cost ranging (minimisation form). For a nonbasic column k the basis stays optimal while its
reduced cost keeps its sign: at its lower bound d_k >= 0, at its upper bound d_k <= 0.
  * nonbasic j:  the range of c_j is [c_j - d_j, +inf) at lower, (-inf, c_j - d_j] at upper
  * basic j in row r:  c_j + delta changes every nonbasic reduced cost by -delta * alpha_rk,
    alpha_r = row r of B^-1 N; delta is limited by the first reduced cost to change sign.

Right-hand-side ranging. For a row i whose activity s_i is nonbasic at a limit, moving that
limit by delta moves the basic values by -delta * B^-1 a_(s_i); the row's marginal value
(shadow price) holds while every basic variable stays inside its bounds.

Values are reported in the user's objective sense (a maximisation model's cost ranges and
marginal values are flipped back).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from ..model import Problem

_TOL = 1e-9


def _basis(prob: Problem, sol):
    basis = (sol.extra or {}).get("basis")
    if basis is None:
        raise ValueError("ranging needs a vertex solution with a basis (solve with engine='simplex')")
    m = prob.m
    st = list(basis["col_statuses"]) + list(basis["row_statuses"])
    Az = sp.hstack([prob.A, -sp.eye(m)], format="csc")
    B_idx = [k for k, s in enumerate(st) if s == "basic"]
    if len(B_idx) != m:
        raise ValueError(f"basis has {len(B_idx)} basic columns, expected {m}")
    return Az, np.array(B_idx), st


def cost_ranging(prob: Problem, sol) -> list[dict]:
    Az, B_idx, st = _basis(prob, sol)
    n, m = prob.n, prob.m
    cz = np.concatenate([prob.c, np.zeros(m)])
    B = Az[:, B_idx].toarray()
    lu = la.lu_factor(B)
    y = la.lu_solve(lu, cz[B_idx], trans=1)
    d = cz - Az.T @ y
    N_idx = np.array([k for k in range(n + m) if st[k] != "basic"])
    pos = {k: r for r, k in enumerate(B_idx)}
    fixed = np.concatenate([prob.lx == prob.ux, prob.lc == prob.uc])
    _, cn = prob.names()
    out = []
    for j in range(n):
        if st[j] != "basic":
            if st[j] == "at_lower":
                lo, hi = prob.c[j] - d[j], np.inf
            elif st[j] == "at_upper":
                lo, hi = -np.inf, prob.c[j] - d[j]
            else:
                lo, hi = prob.c[j], prob.c[j]
        else:
            r = pos[j]
            e = np.zeros(m); e[r] = 1.0
            rho = la.lu_solve(lu, e, trans=1)          # row r of B^-1
            alpha = Az[:, N_idx].T @ rho               # alpha_rk for nonbasic k
            up, dn = np.inf, np.inf
            for a, k in zip(alpha, N_idx):
                if abs(a) <= _TOL or fixed[k]:            # a fixed column cannot move: its sign never binds
                    continue
                dk = d[k]
                s = st[k]
                # need d_k - delta*a >= 0 (at_lower) or <= 0 (at_upper); free/fixed: = 0
                if s == "at_lower":
                    if a > 0:
                        up = min(up, dk / a)
                    else:
                        dn = min(dn, dk / -a)
                elif s == "at_upper":
                    if a < 0:
                        up = min(up, -dk / -a)
                    else:
                        dn = min(dn, -dk / a)
            lo, hi = prob.c[j] - max(dn, 0.0), prob.c[j] + max(up, 0.0)
        if prob.obj_sign < 0:
            lo, hi = -hi, -lo
        out.append({"column": cn[j], "value": float(sol.x[j]), "cost": float(prob.obj_sign * prob.c[j]),
                    "cost_min": float(lo), "cost_max": float(hi), "status": st[j],
                    "reduced_cost": float(prob.obj_sign * d[j])})
    return out


def rhs_ranging(prob: Problem, sol) -> list[dict]:
    Az, B_idx, st = _basis(prob, sol)
    n, m = prob.n, prob.m
    B = Az[:, B_idx].toarray()
    lu = la.lu_factor(B)
    lo_z = np.concatenate([prob.lx, prob.lc])
    hi_z = np.concatenate([prob.ux, prob.uc])
    xz = np.concatenate([sol.x, prob.A @ sol.x])
    rn, _ = prob.names()
    out = []
    for i in range(m):
        k = n + i
        if st[k] == "basic":
            out.append({"row": rn[i], "marginal": 0.0, "binding": False, "activity": float(xz[k])})
            continue
        a = Az[:, k].toarray().ravel()
        dxB = -la.lu_solve(lu, a)                   # change of basic values per unit of s_i
        xB = xz[B_idx]
        # x_B + t*dxB must stay within [lo, hi] for t in [-dec, inc]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_hi = np.where(dxB > _TOL, (hi_z[B_idx] - xB) / dxB, np.where(dxB < -_TOL, (lo_z[B_idx] - xB) / dxB, np.inf))
            t_lo = np.where(dxB > _TOL, (xB - lo_z[B_idx]) / dxB, np.where(dxB < -_TOL, (xB - hi_z[B_idx]) / dxB, np.inf))
        inc = float(np.min(np.maximum(t_hi, 0.0), initial=np.inf))
        dec = float(np.min(np.maximum(t_lo, 0.0), initial=np.inf))
        limit = prob.lc[i] if st[k] == "at_lower" else prob.uc[i]
        y = float(sol.y[i]) if sol.y is not None else 0.0
        out.append({"row": rn[i], "marginal": y, "binding": True, "activity": float(xz[k]),
                    "limit": float(limit), "limit_min": float(limit - dec), "limit_max": float(limit + inc)})
    return out
