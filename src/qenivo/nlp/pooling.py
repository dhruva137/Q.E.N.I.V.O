"""Spatial branch-and-bound for the Haverly pooling problems.

C. A. Haverly, Studies of the behavior of recursion for the pooling problem,
ACM SIGMAP Bulletin 25 (1978). The three standard instances and their maximum
profits 400, 600 and 750 are the ones tabulated by Adhya, Tawarmalani and
Sahinidis, Industrial & Engineering Chemistry Research 38 (1999) 1956-1972.
Instance 2 raises the demand of product X from 100 to 600. Instance 3 prices
crude B at 13 instead of 16. Crude A is 3% sulphur at cost 6, crude B is 1%
sulphur, crude C is 2% sulphur at cost 10 and bypasses the pool. Product X
sells at 9 with sulphur at most 2.5%. Product Y sells at 15 with sulphur at
most 1.5% and demand at most 200.

This module minimises feed cost minus product revenue, so those profits are
the negated optimal values -400, -600 and -750. The relaxation replaces each
product q * flow by a variable w inside the four McCormick envelopes
(F. A. Al-Khayyal and J. E. Falk, Jointly constrained biconvex programming,
Mathematics of Operations Research 8, 1983). Feasibility-based bound tightening
and two rounds of optimality-based bound tightening shrink the bounds. Any
feasible bilinear point satisfies the envelopes, so a certified optimum of the
linear relaxation is a lower bound on the minimised objective. The LP is solved
by ``qenivo.engines.simplex.solve_simplex``, and the bound is kept only when
the KKT residual on that LP is within 1e-6.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

PUBLISHED_PROFIT = {1: 400.0, 2: 600.0, 3: 750.0}
_NAMES = ("A", "B", "PX", "PY", "CX", "CY", "q", "wX", "wY")


@dataclass
class PoolingResult:
    """Global pooling solve. ``objective`` is minimised cost minus revenue."""

    status: str
    objective: float | None
    profit: float | None
    lower_bound: float | None
    root_relaxation: float | None
    x: dict | None
    residuals: dict
    engine: dict
    extra: dict = field(default_factory=dict)
    source: str | None = None
    notes: list = field(default_factory=list)


def haverly_data(instance: int) -> dict:
    """Published costs and demands for Haverly instance 1, 2 or 3."""
    if instance not in (1, 2, 3):
        raise ValueError("Haverly instance must be 1, 2 or 3")
    return {
        "instance": instance,
        "cost_b": 13.0 if instance == 3 else 16.0,
        "demand_x": 600.0 if instance == 2 else 100.0,
        "demand_y": 200.0,
        "published_profit": PUBLISHED_PROFIT[instance],
    }


def reference_point(instance: int) -> np.ndarray:
    """A published minimiser, in the variable order A, B, PX, PY, CX, CY, q, wX, wY.

    The points realise profits 400, 600 and 750. They are feasible for the
    bilinear model; the branch-and-bound does not start from them.
    """
    if instance == 1:
        # Pool is pure B at 1% sulphur; Y is half pool and half crude C.
        return np.array([0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 1.0, 0.0, 100.0])
    if instance == 2:
        # Product X at its sulphur limit from crude A and crude C; no Y.
        return np.array([300.0, 0.0, 300.0, 0.0, 300.0, 0.0, 3.0, 900.0, 0.0])
    # Instance 3: pool sulphur 1.5 from A:B = 1:3, all of it sold as Y.
    return np.array([50.0, 150.0, 0.0, 200.0, 0.0, 0.0, 1.5, 0.0, 300.0])


def haverly_nlp(instance: int):
    """The bilinear Haverly model as a nonconvex NLP, without the w-variables.

    Pool balance and sulphur balance are equalities. Demands and product
    sulphur limits are inequalities. A local interior-point solve of this
    model is not a global proof; :func:`solve_haverly` is.
    """
    from .expr import Graph
    from .problem import NLPModel
    d = haverly_data(instance)
    cap = d["demand_x"] + d["demand_y"]
    g = Graph()
    A = g.var("A", 0.0, cap)
    B = g.var("B", 0.0, cap)
    PX = g.var("PX", 0.0, d["demand_x"])
    PY = g.var("PY", 0.0, d["demand_y"])
    CX = g.var("CX", 0.0, d["demand_x"])
    CY = g.var("CY", 0.0, d["demand_y"])
    q = g.var("q", 1.0, 3.0)
    objective = 6.0 * A + d["cost_b"] * B - 9.0 * PX - 15.0 * PY + CX - 5.0 * CY
    equalities = [A + B - PX - PY, 3.0 * A + B - q * (A + B)]
    inequalities = [
        PX + CX - d["demand_x"],
        PY + CY - d["demand_y"],
        q * PX + 2.0 * CX - 2.5 * (PX + CX),
        q * PY + 2.0 * CY - 1.5 * (PY + CY),
    ]
    return NLPModel(g, objective, equalities, inequalities, name=f"haverly-{instance}-nlp")


def minimised_objective(instance: int, x: np.ndarray) -> float:
    """Feed cost minus product revenue. Published profit is the negation."""
    d = haverly_data(instance)
    A, B, PX, PY, CX, CY = [float(x[i]) for i in range(6)]
    return 6.0 * A + d["cost_b"] * B - 9.0 * PX - 15.0 * PY + 1.0 * CX - 5.0 * CY


def bilinear_feasible(instance: int, x: np.ndarray, tol: float = 1e-7) -> bool:
    """True when ``x`` meets balances, demands and sulphur limits at the pool quality."""
    d = haverly_data(instance)
    A, B, PX, PY, CX, CY, q, wX, wY = [float(v) for v in x]
    if min(A, B, PX, PY, CX, CY) < -tol:
        return False
    if abs((A + B) - (PX + PY)) > tol * (1.0 + abs(A + B)):
        return False
    if PX + CX > d["demand_x"] + tol or PY + CY > d["demand_y"] + tol:
        return False
    pool = A + B
    if pool > tol:
        q_true = (3.0 * A + B) / pool
        if abs(q_true - q) > 1e-5 and abs(wX - q * PX) + abs(wY - q * PY) > 1e-4:
            q = q_true
    if wX + 2.0 * CX > 2.5 * (PX + CX) + tol * (1.0 + PX + CX):
        # Fall back to the true quality if the stored w is only a relaxation value.
        if pool <= tol or (q * PX + 2.0 * CX > 2.5 * (PX + CX) + tol * (1.0 + PX + CX)):
            return False
    if wY + 2.0 * CY > 1.5 * (PY + CY) + tol * (1.0 + PY + CY):
        if pool <= tol or (q * PY + 2.0 * CY > 1.5 * (PY + CY) + tol * (1.0 + PY + CY)):
            return False
    if pool > tol and abs((3.0 * A + B) - (wX + wY)) > 1e-4 * (1.0 + pool):
        # Allow a repaired point whose w was not stored, as long as flows work.
        q_true = (3.0 * A + B) / pool
        if q_true * PX + 2.0 * CX > 2.5 * (PX + CX) + 1e-6 * (1.0 + PX + CX):
            return False
        if q_true * PY + 2.0 * CY > 1.5 * (PY + CY) + 1e-6 * (1.0 + PY + CY):
            return False
    return True


def profit_at_quality(instance: int, q: float) -> float | None:
    """Minimised objective of the pooling LP with the pool sulphur fixed at ``q``.

    Fixing q makes every bilinear product linear, so the LP is a restriction of
    the nonconvex model and its optimum is a valid incumbent.
    """
    val, _x = _fixed_q(instance, float(q), *_box(instance))
    return val


def solve_haverly(instance: int, tol: float = 1e-4, time_limit: float = 30.0,
                  verbose: bool = False, node_limit: int = 2000) -> PoolingResult:
    """Global minimum of Haverly instance ``instance`` by spatial branch-and-bound."""
    t0 = time.perf_counter()
    d = haverly_data(instance)
    lb0, ub0 = _box(instance)
    notes: list[str] = []
    inc, inc_x = _search_incumbent(instance, lb0, ub0)
    root = _relax(instance, lb0, ub0, None)
    root_obj = root["obj"] if root and root.get("status") == "optimal" else None
    root_kkt = root["kkt"] if root and root.get("status") == "optimal" else None
    if root_obj is None:
        notes.append("root McCormick relaxation was not certified")
    elif np.isfinite(inc) and root_obj > inc + 1e-5:
        notes.append("root relaxation is above the incumbent; envelopes are not being used as an outer approximation")
    tight = _obbt(instance, lb0, ub0, inc if np.isfinite(inc) else None)
    if tight is not None:
        lb0, ub0 = tight
    nodes = 0
    counter = 0
    heap: list = []
    state = {"inc": inc, "x": inc_x, "notes": notes, "incomplete": False}

    def consider(val, xvec):
        if val is not None and val < state["inc"]:
            state["inc"] = val
            state["x"] = xvec

    def process(lb, ub):
        nonlocal nodes, counter
        if time.perf_counter() - t0 > time_limit or nodes >= node_limit:
            return
        tightened = _tighten(instance, lb, ub)
        # A failed tightening is not a proof of infeasibility. Solve the LP on the
        # incoming bounds; only a certified infeasible LP prunes the node.
        use_lb, use_ub = (lb, ub) if tightened is None else tightened
        sol = _relax(instance, use_lb, use_ub, None)
        if sol is None or sol.get("status") == "unknown":
            j = int(np.argmax(ub - lb))
            if ub[j] - lb[j] > 1e-5:
                mid = 0.5 * (lb[j] + ub[j])
                for child_lb, child_ub in _split(lb, ub, j, mid):
                    counter += 1
                    heapq.heappush(heap, (-1e100, counter, child_lb, child_ub))
            else:
                state["incomplete"] = True
            return
        if sol["status"] == "infeasible":
            return
        lb, ub = use_lb, use_ub
        nodes += 1
        if sol["obj"] >= state["inc"] - tol:
            return
        repaired = _repair(instance, sol["x"])
        if repaired is not None:
            consider(minimised_objective(instance, repaired), repaired)
        if sol["obj"] >= state["inc"] - tol:
            return
        q_mid = 0.5 * (lb[6] + ub[6])
        for q in (float(sol["x"][6]), q_mid):
            if lb[6] - 1e-9 <= q <= ub[6] + 1e-9:
                val, xvec = _fixed_q(instance, q, lb, ub)
                consider(val, xvec)
        if _gap(sol["x"]) <= 1e-6:
            consider(sol["obj"], sol["x"].copy())
            return
        j = _branch_var(lb, ub, sol["x"])
        if j is None:
            if sol["obj"] < state["inc"] - tol:
                state["incomplete"] = True
            return
        span = ub[j] - lb[j]
        cut = float(sol["x"][j])
        if not (lb[j] + 0.1 * span < cut < ub[j] - 0.1 * span):
            cut = 0.5 * (lb[j] + ub[j])
        for child_lb, child_ub in _split(lb, ub, j, cut):
            counter += 1
            # The parent relaxation remains a lower bound on every child.
            heapq.heappush(heap, (sol["obj"], counter, child_lb, child_ub))
        if verbose:
            print(f"pool node {nodes} lb {sol['obj']:.6g} inc {state['inc']:.6g} branch {_NAMES[j]}")

    process(lb0, ub0)
    while heap and nodes < node_limit and time.perf_counter() - t0 <= time_limit:
        bound, _i, lb, ub = heapq.heappop(heap)
        if bound >= state["inc"] - tol:
            continue
        process(lb, ub)
    inc = state["inc"]
    timed_out = time.perf_counter() - t0 > time_limit or nodes >= node_limit
    open_lb = min((h[0] for h in heap), default=inc)
    if np.isfinite(inc) and not heap and not timed_out and not state["incomplete"] and root_obj is not None:
        status = "optimal"
        lower = inc
    elif np.isfinite(inc):
        status = "time_limit" if time.perf_counter() - t0 > time_limit else "node_limit"
        lower = min(open_lb, inc)
        if timed_out and nodes >= node_limit:
            status = "node_limit"
    else:
        status = "not_converged"
        lower = root_obj
    if root_obj is not None and np.isfinite(inc) and root_obj > inc + 1e-4:
        status = "not_converged"
    profit = None if not np.isfinite(inc) else -inc
    xdict = None
    if state["x"] is not None:
        xdict = {n: float(state["x"][i]) for i, n in enumerate(_NAMES)}
    engine = {
        "engine": "nlp-global",
        "backend": "numpy",
        "iterations": nodes,
        "time": time.perf_counter() - t0,
        "reason": "bilinear pooling, McCormick spatial branch-and-bound",
    }
    return PoolingResult(
        status=status, objective=None if not np.isfinite(inc) else float(inc), profit=profit,
        lower_bound=None if lower is None else float(lower),
        root_relaxation=None if root_obj is None else float(root_obj),
        x=xdict,
        residuals={"root_kkt": root_kkt, "gap": None if not np.isfinite(inc) or lower is None else float(inc - lower),
                   "published_profit": d["published_profit"]},
        engine=engine, notes=notes,
        extra={"nodes": nodes, "instance": instance, "formulation": "minimise cost minus revenue"},
    )


def _box(instance: int) -> tuple[np.ndarray, np.ndarray]:
    d = haverly_data(instance)
    dx, dy = d["demand_x"], d["demand_y"]
    lb = np.zeros(9)
    ub = np.array([dx + dy, dx + dy, dx, dy, dx, dy, 3.0, 3.0 * dx, 3.0 * dy])
    lb[6] = 1.0
    return lb, ub


def _cost(instance: int) -> np.ndarray:
    d = haverly_data(instance)
    return np.array([6.0, d["cost_b"], -9.0, -15.0, 1.0, -5.0, 0.0, 0.0, 0.0])


def _structural(instance: int) -> list[tuple[np.ndarray, float, float]]:
    d = haverly_data(instance)
    rows = []

    def row(coefs, lo, hi):
        a = np.zeros(9)
        for j, v in coefs.items():
            a[j] = v
        rows.append((a, lo, hi))

    row({0: 1.0, 1: 1.0, 2: -1.0, 3: -1.0}, 0.0, 0.0)
    row({0: 3.0, 1: 1.0, 7: -1.0, 8: -1.0}, 0.0, 0.0)
    row({2: 1.0, 4: 1.0}, -np.inf, d["demand_x"])
    row({3: 1.0, 5: 1.0}, -np.inf, d["demand_y"])
    row({7: 1.0, 2: -2.5, 4: -0.5}, -np.inf, 0.0)
    row({8: 1.0, 3: -1.5, 5: 0.5}, -np.inf, 0.0)
    return rows


def _mccormick(lb, ub) -> list[tuple[np.ndarray, float, float]]:
    rows = []
    for w, q, p in ((7, 6, 2), (8, 6, 3)):
        qL, qU, pL, pU = float(lb[q]), float(ub[q]), float(lb[p]), float(ub[p])
        # w >= qL p + pL q - qL pL
        rows.append((_coef({w: 1.0, p: -qL, q: -pL}), -qL * pL, np.inf))
        rows.append((_coef({w: 1.0, p: -qU, q: -pU}), -qU * pU, np.inf))
        rows.append((_coef({w: 1.0, p: -qU, q: -pL}), -np.inf, -qU * pL))
        rows.append((_coef({w: 1.0, p: -qL, q: -pU}), -np.inf, -qL * pU))
    return rows


def _coef(terms: dict) -> np.ndarray:
    a = np.zeros(9)
    for j, v in terms.items():
        a[j] = v
    return a


def _relax(instance, lb, ub, cut):
    rows = _structural(instance) + _mccormick(lb, ub)
    if cut is not None:
        rows.append((_cost(instance), -np.inf, cut))
    return _solve(_cost(instance), rows, lb, ub)


def _solve(c, rows, lb, ub):
    """Solve one relaxation with the simplex engine and accept it only if KKT holds.

    ``api.solve`` may hand a failed vertex solve to other engines. Those are not
    used here: a pooling bound has to be a certified simplex optimum.
    """
    import scipy.sparse as sp

    from ..certify.kkt import kkt_residuals
    from ..engines.simplex import solve_simplex
    from ..model import Problem
    lb = np.asarray(lb, dtype=float).copy()
    ub = np.asarray(ub, dtype=float).copy()
    if np.any(lb > ub + 1e-8):
        return {"status": "infeasible"}
    A = sp.csr_matrix(np.vstack([a for a, _, _ in rows]))
    lc = np.array([lo for _a, lo, _hi in rows], dtype=float)
    uc = np.array([hi for _a, _lo, hi in rows], dtype=float)
    cost = np.asarray(c, dtype=float)
    prob = Problem(c=cost, A=A, lc=lc, uc=uc, lx=lb, ux=ub, name="mccormick")
    out = solve_simplex(prob, tol=1e-8, time_limit=2.0)
    if out["status"] == "infeasible":
        return {"status": "infeasible"}
    if out["status"] != "optimal" or out.get("x") is None or out.get("y") is None:
        return {"status": "unknown"}
    k = kkt_residuals(prob, out["x"], out["y"])
    kkt = float(k["max_rel"])
    if kkt > 1e-6:
        return {"status": "unknown"}
    x = np.asarray(out["x"], dtype=float).reshape(-1)[:9]
    return {"status": "optimal", "obj": float(cost @ x), "x": x, "kkt": kkt, "certified": True}


def _fixed_q(instance, q, lb, ub):
    """LP with pool quality fixed. Returns (objective, full point) or (None, None)."""
    import scipy.sparse as sp

    from ..model import Problem
    d = haverly_data(instance)
    # variables A B PX PY CX CY
    c = np.array([6.0, d["cost_b"], -9.0, -15.0, 1.0, -5.0])
    lx = np.array([lb[0], lb[1], lb[2], lb[3], lb[4], lb[5]], dtype=float)
    ux = np.array([ub[0], ub[1], ub[2], ub[3], ub[4], ub[5]], dtype=float)
    rows = [
        (np.array([3.0 - q, 1.0 - q, 0, 0, 0, 0]), 0.0, 0.0),
        (np.array([1.0, 1.0, -1.0, -1.0, 0, 0]), 0.0, 0.0),
        (np.array([0, 0, 1.0, 0, 1.0, 0]), -np.inf, d["demand_x"]),
        (np.array([0, 0, 0, 1.0, 0, 1.0]), -np.inf, d["demand_y"]),
        (np.array([0, 0, q - 2.5, 0, -0.5, 0]), -np.inf, 0.0),
        (np.array([0, 0, 0, q - 1.5, 0, 0.5]), -np.inf, 0.0),
    ]
    A = sp.csr_matrix(np.vstack([a for a, _, _ in rows]))
    prob = Problem(c=c, A=A, lc=np.array([r[1] for r in rows]), uc=np.array([r[2] for r in rows]),
                   lx=lx, ux=ux, name=f"fixed-q-{q}")
    from ..certify.kkt import kkt_residuals
    from ..engines.simplex import solve_simplex
    out = solve_simplex(prob, tol=1e-8, time_limit=2.0)
    if out["status"] != "optimal" or out.get("x") is None or out.get("y") is None:
        return None, None
    if kkt_residuals(prob, out["x"], out["y"])["max_rel"] > 1e-6:
        return None, None
    A_, B, PX, PY, CX, CY = [float(v) for v in out["x"]]
    full = np.array([A_, B, PX, PY, CX, CY, q, q * PX, q * PY])
    if not _flows_meet_specs(instance, full):
        return None, None
    return float(c @ out["x"]), full


def _flows_meet_specs(instance, x, tol=1e-6) -> bool:
    d = haverly_data(instance)
    A, B, PX, PY, CX, CY, q = [float(x[i]) for i in range(7)]
    if min(A, B, PX, PY, CX, CY) < -tol:
        return False
    if abs(A + B - PX - PY) > tol * (1 + abs(A + B)):
        return False
    if PX + CX > d["demand_x"] + tol or PY + CY > d["demand_y"] + tol:
        return False
    pool = A + B
    if pool > tol and abs((3 - q) * A + (1 - q) * B) > tol * (1 + pool):
        return False
    if q * PX + 2 * CX > 2.5 * (PX + CX) + tol * (1 + PX + CX):
        return False
    if q * PY + 2 * CY > 1.5 * (PY + CY) + tol * (1 + PY + CY):
        return False
    return True


def _search_incumbent(instance, lb, ub):
    best, best_x = np.inf, None
    grid = [1.0, 1.5, 2.0, 2.5, 3.0, float(lb[6]), float(ub[6]), 0.5 * (lb[6] + ub[6])]
    for q in grid:
        if q < lb[6] - 1e-9 or q > ub[6] + 1e-9:
            continue
        val, x = _fixed_q(instance, q, lb, ub)
        if val is not None and val < best:
            best, best_x = val, x
    return best, best_x


def _repair(instance, x):
    A, B, PX, PY, CX, CY = [float(x[i]) for i in range(6)]
    pool = A + B
    if pool <= 1e-8:
        q = float(x[6])
    else:
        q = (3.0 * A + B) / pool
    full = np.array([A, B, PX, PY, CX, CY, q, q * PX, q * PY])
    if _flows_meet_specs(instance, full, tol=1e-5):
        return full
    return None


def _gap(x) -> float:
    return abs(float(x[7] - x[6] * x[2])) + abs(float(x[8] - x[6] * x[3]))


def _branch_var(lb, ub, x) -> int | None:
    scores = []
    gx = abs(float(x[7] - x[6] * x[2]))
    gy = abs(float(x[8] - x[6] * x[3]))
    span = ub - lb
    if gx > 1e-8:
        scores.append((gx * span[6], 6))
        scores.append((gx * span[2], 2))
    if gy > 1e-8:
        scores.append((gy * span[6], 6))
        scores.append((gy * span[3], 3))
    scores = [(s, j) for s, j in scores if span[j] > 1e-6]
    if not scores:
        return None
    return max(scores)[1]


def _split(lb, ub, j, cut):
    left_ub = ub.copy()
    left_ub[j] = cut
    right_lb = lb.copy()
    right_lb[j] = cut
    return (lb.copy(), left_ub), (right_lb, ub.copy())


def _tighten(instance, lb, ub):
    lb, ub = lb.copy(), ub.copy()
    for _ in range(6):
        rows = _structural(instance) + _mccormick(lb, ub)
        updated = _fbbt(lb, ub, rows)
        if updated is None:
            return None
        nlb, nub = updated
        if np.max(np.abs(nlb - lb) + np.abs(nub - ub)) < 1e-8:
            return nlb, nub
        lb, ub = nlb, nub
    return lb, ub


def _fbbt(lb, ub, rows):
    lb, ub = lb.copy(), ub.copy()
    for _ in range(4):
        for a, lo, hi in rows:
            for j in range(9):
                aj = a[j]
                if aj == 0.0:
                    continue
                slo = shi = 0.0
                for k in range(9):
                    if k == j or a[k] == 0.0:
                        continue
                    tlo, thi = (a[k] * lb[k], a[k] * ub[k]) if a[k] > 0 else (a[k] * ub[k], a[k] * lb[k])
                    slo += tlo
                    shi += thi
                tgt_lo = lo - shi if np.isfinite(lo) else -np.inf
                tgt_hi = hi - slo if np.isfinite(hi) else np.inf
                if aj > 0:
                    nlo = tgt_lo / aj if np.isfinite(tgt_lo) else -np.inf
                    nhi = tgt_hi / aj if np.isfinite(tgt_hi) else np.inf
                else:
                    nlo = tgt_hi / aj if np.isfinite(tgt_hi) else -np.inf
                    nhi = tgt_lo / aj if np.isfinite(tgt_lo) else np.inf
                lb[j] = max(lb[j], nlo)
                ub[j] = min(ub[j], nhi)
        for w, q, p in ((7, 6, 2), (8, 6, 3)):
            vals = [lb[q] * lb[p], lb[q] * ub[p], ub[q] * lb[p], ub[q] * ub[p]]
            lb[w] = max(lb[w], min(vals))
            ub[w] = min(ub[w], max(vals))
            if lb[q] > 1e-8:
                qv = [lb[w] / lb[q], lb[w] / ub[q], ub[w] / lb[q], ub[w] / ub[q]]
                lb[p] = max(lb[p], min(qv))
                ub[p] = min(ub[p], max(qv))
            if lb[p] > 1e-8:
                pv = [lb[w] / lb[p], lb[w] / ub[p], ub[w] / lb[p], ub[w] / ub[p]]
                lb[q] = max(lb[q], min(pv))
                ub[q] = min(ub[q], max(pv))
        qvals = []
        for A in (lb[0], ub[0]):
            for B in (lb[1], ub[1]):
                if A + B > 1e-12:
                    qvals.append((3.0 * A + B) / (A + B))
        if qvals:
            lb[6] = max(lb[6], min(qvals))
            ub[6] = min(ub[6], max(qvals))
        if np.any(lb > ub + 1e-7):
            return None
    for j in range(9):
        if lb[j] > ub[j]:
            if lb[j] - ub[j] < 1e-7:
                mid = 0.5 * (lb[j] + ub[j])
                lb[j] = ub[j] = mid
            else:
                return None
    return lb, ub


def _obbt(instance, lb, ub, inc):
    """Two rounds of bound tightening by optimising each branching variable."""
    lb, ub = lb.copy(), ub.copy()
    for _ in range(2):
        for j in (6, 0, 1, 2, 3):
            for sense in (1.0, -1.0):
                c = np.zeros(9)
                c[j] = sense
                rows = _structural(instance) + _mccormick(lb, ub)
                if inc is not None and np.isfinite(inc):
                    rows.append((_cost(instance), -np.inf, inc + 1e-6))
                sol = _solve(c, rows, lb, ub)
                if sol is None or sol.get("status") != "optimal" or not sol["certified"]:
                    continue
                val = float(sol["x"][j])
                if sense > 0 and val > lb[j] + 1e-7:
                    lb[j] = val
                if sense < 0 and val < ub[j] - 1e-7:
                    ub[j] = val
        if np.any(lb > ub + 1e-8):
            return None
    return lb, ub
