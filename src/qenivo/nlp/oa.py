"""Outer approximation for convex mixed-integer nonlinear programs.

M. A. Duran and I. E. Grossmann, An outer-approximation algorithm for a class
of mixed-integer nonlinear programs, Mathematical Programming 36 (1986) 307-339.

Each NLP subproblem freezes the integer variables and is solved by the filter
interior-point method in this package. The master is the MILP of accumulated
gradient cuts, solved by ``qenivo.engines.milp.solve_milp``. The cuts are an
outer approximation when the objective and the inequalities are convex and the
equalities are affine. This module does not claim that property for a model
it has not been given.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .ad import gradient
from .expr import evaluate
from .ipm import solve_ipm
from .problem import NLPModel


@dataclass
class OAResult:
    """Outer-approximation solve. ``objective`` is the best feasible NLP value."""

    status: str
    objective: float | None
    lower_bound: float | None
    x: np.ndarray | None
    residuals: dict
    engine: dict
    extra: dict = field(default_factory=dict)
    source: str | None = None
    notes: list = field(default_factory=list)
    nlp_solves: int = 0
    master_solves: int = 0


def convex_demo() -> NLPModel:
    """min (x-2)^2 + 3(y-1)^2, x in [0, 5], y integer in {0, 1, 2}.

    Enumerating y gives 3, 0 and 3. The minimum is 0 at x=2, y=1.
    """
    from .expr import Graph
    g = Graph()
    x = g.var("x", 0.0, 5.0)
    y = g.var("y", 0.0, 2.0)
    objective = (x - 2.0) ** 2 + 3.0 * ((y - 1.0) ** 2)
    return NLPModel(g, objective, integers=(1,), name="convex-oa-demo")


def outer_approximation(model: NLPModel, tol: float = 1e-4, time_limit: float = 30.0,
                        max_rounds: int = 12) -> OAResult:
    """Duran-Grossmann outer approximation. Integers start at their lower bounds."""
    t0 = time.perf_counter()
    notes: list[str] = []
    if not model.integers:
        raise ValueError("outer approximation needs at least one integer variable")
    lb0 = np.asarray(model.graph.lb, dtype=np.float64)
    ub0 = np.asarray(model.graph.ub, dtype=np.float64)
    integers = tuple(int(j) for j in model.integers)
    cuts: list[tuple] = []
    best = np.inf
    best_x = None
    best_res: dict = {}
    nlp_solves = 0
    master_solves = 0
    visited: set[tuple[int, ...]] = set()
    x_seed = None

    def remaining() -> float:
        return time_limit - (time.perf_counter() - t0)

    def nlp_fixed(assign: tuple[int, ...]):
        nonlocal nlp_solves, x_seed
        lb, ub = lb0.copy(), ub0.copy()
        for j, yj in zip(integers, assign):
            lb[j] = ub[j] = float(yj)
        x0 = np.zeros(model.graph.n_var)
        for i, (lo, hi) in enumerate(zip(lb, ub)):
            if np.isfinite(lo) and np.isfinite(hi):
                x0[i] = 0.5 * (lo + hi)
            elif np.isfinite(lo):
                x0[i] = lo + 0.5
            elif np.isfinite(hi):
                x0[i] = hi - 0.5
        if x_seed is not None:
            x0 = x_seed.copy()
            for j, yj in zip(integers, assign):
                x0[j] = float(yj)
            x0 = np.minimum(np.maximum(x0, lb), ub)
        nlp_solves += 1
        return solve_ipm(model, tol=min(1e-8, tol), time_limit=max(1.0, remaining()),
                         x0=x0, lb=lb, ub=ub, max_iter=60)

    def add_cuts(x: np.ndarray) -> None:
        g = gradient(model.objective, x)
        cuts.append(("obj", g, x.copy(), evaluate(model.objective, x)))
        for expr in model.inequalities:
            cuts.append(("ineq", gradient(expr, x), x.copy(), evaluate(expr, x)))
        for expr in model.equalities:
            cuts.append(("eq", gradient(expr, x), x.copy(), evaluate(expr, x)))

    y0 = tuple(int(np.ceil(lb0[j] - 1e-9)) for j in integers)
    first = nlp_fixed(y0)
    status = "not_converged"
    lower = None
    if first.status != "optimal" or first.x is None or first.objective is None:
        notes.append(f"NLP at the integer lower bound {y0} returned {first.status}")
    else:
        add_cuts(first.x)
        best, best_x, best_res = float(first.objective), first.x.copy(), dict(first.residuals)
        x_seed = first.x.copy()
        visited.add(y0)
        while master_solves < max_rounds and remaining() > 0.5:
            master = _master(model, cuts, lb0, ub0, max(1.0, remaining()))
            master_solves += 1
            if master is None or master.x is None or master.objective is None:
                notes.append("MILP master was not proven optimal")
                break
            lower = float(master.objective)
            assign = _integers_from(master.x, integers, lb0, ub0)
            if lower >= best - tol * max(1.0, abs(best)):
                status = "optimal"
                break
            if assign in visited:
                notes.append(f"master repeated integers {assign} with predicted value {lower:.6g}")
                break
            sub = nlp_fixed(assign)
            visited.add(assign)
            if sub.status != "optimal" or sub.x is None or sub.objective is None:
                notes.append(f"NLP at integers {assign} returned {sub.status}")
                break
            add_cuts(sub.x)
            x_seed = sub.x.copy()
            if sub.objective < best:
                best, best_x, best_res = float(sub.objective), sub.x.copy(), dict(sub.residuals)
        else:
            if lower is not None and np.isfinite(best) and lower >= best - tol * max(1.0, abs(best)):
                status = "optimal"
    if status != "optimal":
        notes.append("outer approximation stopped before the master bound met the incumbent")
    engine = {
        "engine": "nlp-global",
        "backend": "numpy",
        "iterations": nlp_solves + master_solves,
        "time": time.perf_counter() - t0,
        "reason": "convex MINLP, Duran-Grossmann outer approximation",
    }
    return OAResult(
        status=status,
        objective=None if not np.isfinite(best) else float(best),
        lower_bound=None if lower is None else float(lower),
        x=best_x,
        residuals=best_res,
        engine=engine,
        notes=notes,
        nlp_solves=nlp_solves,
        master_solves=master_solves,
        extra={"nlp_solves": nlp_solves, "master_solves": master_solves, "integers_tried": [list(v) for v in visited]},
    )


def _integers_from(x, integers, lb, ub) -> tuple[int, ...]:
    out = []
    for j in integers:
        lo = int(np.ceil(lb[j] - 1e-8))
        hi = int(np.floor(ub[j] + 1e-8))
        out.append(int(np.clip(np.rint(x[j]), lo, hi)))
    return tuple(out)


def _master(model: NLPModel, cuts, lb, ub, time_limit):
    from ..engines.milp import solve_milp
    from ..model import ModelBuilder
    names = list(model.graph.var_names)
    built = None
    for presolve, cuts_on in ((True, True), (False, False)):
        mb = ModelBuilder("oa-master")
        for j, name in enumerate(names):
            lo = float(lb[j]) if np.isfinite(lb[j]) else -1e6
            hi = float(ub[j]) if np.isfinite(ub[j]) else 1e6
            mb.var(name, lo=lo, hi=hi, integer=j in model.integers)
        mb.var("alpha", lo=-1e6, hi=1e6)
        for k, (kind, grad, xk, value) in enumerate(cuts):
            dot = float(grad @ xk)
            coefs = {}
            for j, gj in enumerate(grad):
                if abs(gj) > 1e-14:
                    coefs[names[j]] = float(-gj if kind == "obj" else gj)
            if kind == "obj":
                coefs["alpha"] = 1.0
                mb.row(f"c{k}", coefs, lo=float(value - dot))
            elif kind == "ineq":
                if coefs:
                    mb.row(f"c{k}", coefs, hi=float(-value + dot))
            else:
                rhs = float(-value + dot)
                if coefs:
                    mb.row(f"c{k}", coefs, lo=rhs, hi=rhs)
                elif abs(value) > 1e-8:
                    return None
        built = mb.build()
        sol = solve_milp(built, tol=1e-8, time_limit=time_limit, gap_tol=1e-8,
                         presolve=presolve, cuts=cuts_on)
        if sol.verdict == "optimal" and sol.x is not None:
            return sol
    return None
