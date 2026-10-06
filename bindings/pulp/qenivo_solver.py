"""PuLP solver class. Linear models are translated to the Qenivo C ABI.

Mitchell, O'Sullivan, Dunning, PuLP: A Linear Programming Toolkit for Python, 2011.
The coefficient walk is original. A FakeModel is solved when PuLP is not installed.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _affine():
    name = "qenivo_bindings_affine"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = Path(__file__).resolve().parents[1] / "affine.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def translate(model):
    affine = _affine()
    if affine.is_fake(model):
        return model if isinstance(model, affine.Model) else affine.from_fake(model)
    return _from_pulp(model)


def solve(model, engine=None, tol=1e-8, time_limit=60.0):
    return _affine().solve(translate(model), engine=engine, tol=tol, time_limit=time_limit)


class QenivoSolver:
    """PuLP LpSolver-shaped object. assign it with `prob.solve(QenivoSolver())` or call solve(prob)."""

    name = "Qenivo"
    def __init__(self, engine=None, tol=1e-8, time_limit=60.0, msg=False):
        self.engine = engine
        self.tol = tol
        self.time_limit = time_limit
        self.msg = msg
        self.result = None

    def available(self):
        return True

    def solve(self, lp, **kwargs):
        self.result = solve(lp, engine=self.engine, tol=self.tol, time_limit=self.time_limit)
        # PuLP status: 1 optimal, 0 not solved, -1 infeasible, -2 unbounded (same codes in 3.x and 4.x).
        code = {0: 1, 1: -1, 2: -2}.get(self.result.status, 0)
        if type(lp).__module__.split(".")[0] != "pulp":
            return code
        import pulp
        lp.solver = self
        # The C ABI hands back an x even for an infeasible model; only an optimal one is a solution.
        has_solution = code == 1 and self.result.x is not None
        if has_solution:
            by_name = dict(zip(self.result.names, self.result.x))
            for var in lp.variables():
                if var.name in by_name:
                    var.varValue = float(by_name[var.name])
        if hasattr(pulp, "LpSolveStatus"):
            # PuLP 4 drops the LpStatus* constants; a solver returns an LpSolveStats record instead.
            return pulp.LpSolveStats(
                solver=self.name, status=pulp.LpSolveStatus(code), has_solution=has_solution,
                objective=self.result.objective if code == 1 else None,
                num_variables=len(lp.variables()), num_constraints=len(_constraints(lp)),
                is_mip=any(v.cat in (pulp.LpInteger, pulp.LpBinary) for v in lp.variables()))
        status = {1: pulp.LpStatusOptimal, -1: pulp.LpStatusInfeasible, -2: pulp.LpStatusUnbounded,
                  0: pulp.LpStatusNotSolved}[code]
        lp.status = status
        return status


def _constraints(lp) -> list:
    """(name, constraint) pairs: a dict attribute up to PuLP 3, a method returning a list in PuLP 4."""
    cons = lp.constraints() if callable(lp.constraints) else lp.constraints
    if isinstance(cons, dict):
        return list(cons.items())
    return [(c.name, c) for c in cons]


def _from_pulp(lp):
    import pulp

    sense = "max" if lp.sense == pulp.LpMaximize else "min"
    # `prob += expr + 5` keeps the 5 as the objective's constant; drop it and every objective is off by 5.
    constant = 0.0 if lp.objective is None else float(getattr(lp.objective, "constant", 0.0) or 0.0)
    out = _affine().Model(name=getattr(lp, "name", "pulp") or "pulp", sense=sense, constant=constant)
    for var in lp.variables():
        integer = var.cat in (pulp.LpInteger, pulp.LpBinary)
        if var.cat == pulp.LpBinary:
            lower, upper, integer = 0.0, 1.0, True
        else:
            # None is an absent bound. PuLP's own default of 0 is stored as 0, not None.
            lower = -float("inf") if var.lowBound is None else float(var.lowBound)
            upper = float("inf") if var.upBound is None else float(var.upBound)
        coef = 0.0 if lp.objective is None else float(lp.objective.get(var, 0.0))
        out.column(var.name, lower, upper, coef, integer)
    for name, con in _constraints(lp):
        # PuLP stores sum(a_j x_j) + constant  ?  0, so the right-hand side is -constant.
        terms = {var.name: float(coef) for var, coef in con.items()}
        rhs = -float(con.constant)
        if con.sense == pulp.LpConstraintLE:
            lower, upper = -float("inf"), rhs
        elif con.sense == pulp.LpConstraintGE:
            lower, upper = rhs, float("inf")
        else:
            lower = upper = rhs
        out.row(terms, lower, upper, name)
    return out
