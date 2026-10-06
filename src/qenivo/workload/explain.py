"""Explain an infeasible plan, and prove that it is infeasible.

A planner's worst LP outcome is "infeasible" with no reason. QENIVO answers with the
smallest total relaxation of the row limits that makes the plan feasible, row by row, in the
model's own names ("demand_PMF_t3 short by 1,240 bbl"), plus a proof.

Elastic LP (column bounds kept hard; they are physical limits):
    min  sum_i w_i (p_i + q_i)
    s.t. lc <= A x + p - q <= uc,   lx <= x <= ux,   p, q >= 0
It is always feasible. If its optimum V > 0 the original plan is infeasible, and the row
duals y of the elastic LP are a Farkas certificate for the original rows: the dual
constraints of the p and q columns give |y_i| <= w_i, those of x give that lam = -A'y
respects the column bounds, and strong duality gives phi(y) = V > 0.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..model import Problem


def _elastic(prob: Problem, weights=None):
    m, n = prob.m, prob.n
    w = np.ones(m) if weights is None else np.asarray(weights, float)
    fin = np.isfinite(prob.lc) | np.isfinite(prob.uc)
    rows = np.flatnonzero(fin)
    k = len(rows)
    P = sp.csr_matrix((np.ones(k), (rows, np.arange(k))), shape=(m, k))
    A = sp.hstack([prob.A, P, -P], format="csr")
    c = np.concatenate([np.zeros(n), w[rows], w[rows]])
    lx = np.concatenate([prob.lx, np.zeros(2 * k)])
    ux = np.concatenate([prob.ux, np.full(2 * k, np.inf)])
    rn, cn = prob.names()
    e = Problem(c=c, A=A, lc=prob.lc, uc=prob.uc, lx=lx, ux=ux, name=f"{prob.name}_elastic",
                row_names=rn, col_names=cn + [f"relax_up[{rn[i]}]" for i in rows] + [f"relax_down[{rn[i]}]" for i in rows])
    return e, rows


def _solve_lp(e: Problem, time_limit):
    # Prefer simplex first: the native dual core routinely finishes elastic LPs
    # of a few thousand rows in well under a second (e.g. Netlib gran), while the
    # previous m<=800 gate fell through to IPM/PDHG and timed out without a ray.
    if e.m <= 8000:
        from ..engines.simplex import solve_simplex
        out = solve_simplex(e, tol=1e-10, time_limit=time_limit)
        if out["status"] == "optimal":
            return out
    from ..engines.ipm import solve_ipm
    if e.m <= 4000:
        out = solve_ipm(e, tol=1e-10, time_limit=time_limit)
        if out["status"] == "optimal":
            return out
    from ..engines import pdhg
    br = pdhg.solve(e, tol=1e-8, time_limit=time_limit, detect_infeasibility=False)
    col = br.column(0)
    return {"status": col["status"], "x": col["x"], "y": col["y"]}


def elastic_farkas(prob: Problem, time_limit: float = 600.0):
    """Return {'ray', 'violation', 'scale', 'relax'} or None if the elastic LP failed."""
    if np.any(prob.lx > prob.ux):
        return None
    e, rows = _elastic(prob)
    out = _solve_lp(e, time_limit)
    if out["status"] != "optimal":
        return None
    x = out["x"]
    k = len(rows)
    n = prob.n
    p, q = x[n:n + k], x[n + k:n + 2 * k]
    V = float(np.sum(p) + np.sum(q))
    return {"ray": np.asarray(out["y"], float), "violation": V,
            "scale": float(np.max(np.abs(np.concatenate([prob.lc[np.isfinite(prob.lc)],
                                                         prob.uc[np.isfinite(prob.uc)], [1.0]])))),
            "relax": (rows, p, q), "x": x[:n]}


def explain_infeasibility(prob: Problem, time_limit: float = 600.0, top: int = 20) -> dict:
    """Planner-readable diagnosis: conflicting limits, amounts, and a Farkas proof."""
    rn, cn = prob.names()
    bad_cols = np.flatnonzero(prob.lx > prob.ux)
    if bad_cols.size:
        return {"feasible": False, "kind": "bound_conflict",
                "items": [{"constraint": cn[j], "issue": f"lower bound {prob.lx[j]:g} > upper bound {prob.ux[j]:g}"}
                          for j in bad_cols[:top]], "proof": "direct: contradictory bounds on one column"}
    bad_rows = np.flatnonzero(prob.lc > prob.uc)
    if bad_rows.size:
        return {"feasible": False, "kind": "bound_conflict",
                "items": [{"constraint": rn[i], "issue": f"row lower limit {prob.lc[i]:g} > upper limit {prob.uc[i]:g}"}
                          for i in bad_rows[:top]], "proof": "direct: contradictory limits on one row"}
    out = elastic_farkas(prob, time_limit)
    if out is None:
        return {"feasible": None, "kind": "unknown", "items": [], "proof": "elastic LP did not solve"}
    tol = 1e-9 * (1 + out["scale"])
    if out["violation"] <= tol:
        return {"feasible": True, "kind": "feasible", "items": [], "total_relaxation": out["violation"]}
    rows, p, q = out["relax"]
    Ax = prob.A @ out["x"]
    items = []
    for i, up, dn in zip(rows, p, q):
        if up > tol:          # activity had to be pushed UP: the row's lower limit cannot be met
            items.append({"constraint": rn[i], "limit": "lower", "limit_value": float(prob.lc[i]),
                          "best_achievable": float(Ax[i]), "shortfall": float(up),
                          "issue": f"cannot reach lower limit {prob.lc[i]:,.6g}; best is {Ax[i]:,.6g} (short by {up:,.6g})"})
        if dn > tol:
            items.append({"constraint": rn[i], "limit": "upper", "limit_value": float(prob.uc[i]),
                          "best_achievable": float(Ax[i]), "shortfall": float(dn),
                          "issue": f"cannot stay under upper limit {prob.uc[i]:,.6g}; best is {Ax[i]:,.6g} (over by {dn:,.6g})"})
    items.sort(key=lambda d: -d["shortfall"])
    y = out["ray"]
    involved = [rn[i] for i in np.argsort(-np.abs(y))[:top] if abs(y[i]) > 1e-9]
    from ..certify.kkt import check_farkas
    chk = check_farkas(prob, y)
    return {"feasible": False, "kind": "conflict", "total_relaxation": out["violation"],
            "items": items[:top], "conflict_rows": involved, "farkas_check": chk,
            "proof": "elastic LP optimum > 0; its row duals are a Farkas ray (checked)"}


def format_explanation(rep: dict) -> str:
    if rep.get("feasible"):
        return "The model is feasible (elastic relaxation needed: 0)."
    if rep.get("feasible") is None:
        return "Could not decide feasibility: " + rep.get("proof", "")
    lines = ["The model is INFEASIBLE."]
    if rep["kind"] == "bound_conflict":
        lines += [f"  - {it['constraint']}: {it['issue']}" for it in rep["items"]]
        return "\n".join(lines)
    lines.append(f"Smallest total relaxation that makes it feasible: {rep['total_relaxation']:,.6g}")
    lines.append("Limits that cannot all be met (largest first):")
    lines += [f"  - {it['constraint']}: {it['issue']}" for it in rep["items"]]
    if rep.get("conflict_rows"):
        lines.append("Constraints in the conflict (Farkas support): " + ", ".join(rep["conflict_rows"][:12]))
    chk = rep.get("farkas_check") or {}
    lines.append(f"Proof: {rep['proof']}  (value {chk.get('value', float('nan')):.3e}, "
                 f"violation {chk.get('violation', float('nan')):.1e})")
    return "\n".join(lines)
