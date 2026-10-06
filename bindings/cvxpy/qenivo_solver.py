"""CVXPY interface for scalar linear and convex-quadratic problems.

Diamond and Boyd, CVXPY: A Python-embedded modeling language for convex
optimization, JMLR 2016. The expression walk is original and only accepts affine
atoms it can read off the expression tree. A FakeModel does not need CVXPY.
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
    return _from_cvxpy(model)


def solve(model, engine=None, tol=1e-8, time_limit=60.0):
    return _affine().solve(translate(model), engine=engine, tol=tol, time_limit=time_limit)


def _merge(into, const, terms):
    into_c, into_t = into
    for key, coef in terms.items():
        into_t[key] = into_t.get(key, 0.0) + coef
    return into_c + const, into_t


def _affine_expr(expr):
    name = type(expr).__name__
    if name == "Variable":
        if int(expr.size) != 1:
            raise ValueError("this CVXPY plugin solves scalar variables; stack them as separate Variables")
        return 0.0, {expr.name(): 1.0}
    if name in {"Constant", "Parameter"}:
        value = expr.value
        return float(value), {}
    if name == "NegExpression":
        const, terms = _affine_expr(expr.args[0])
        return -const, {k: -v for k, v in terms.items()}
    if name in {"AddExpression"}:
        const, terms = 0.0, {}
        for arg in expr.args:
            ci, ti = _affine_expr(arg)
            const, terms = _merge((const, terms), ci, ti)
        return const, terms
    if name in {"MulExpression", "DivExpression"}:
        left, right = expr.args
        if name == "DivExpression":
            scale = 1.0 / float(right.value)
            const, terms = _affine_expr(left)
            return const * scale, {k: v * scale for k, v in terms.items()}
        lc, lt = _affine_expr(left)
        rc, rt = _affine_expr(right)
        if lt and rt:
            raise ValueError("a product of two non-constants is not accepted as a linear term")
        if not lt:
            return lc * rc, {k: lc * v for k, v in rt.items()}
        return rc * lc, {k: rc * v for k, v in lt.items()}
    raise ValueError(f"CVXPY atom {name} is not translated by this plugin")


def _from_cvxpy(problem):
    variables = list(problem.variables())
    sense = "max" if type(problem.objective).__name__ == "Maximize" else "min"
    out = _affine().Model(name="cvxpy", sense=sense)
    for var in variables:
        if int(var.size) != 1:
            raise ValueError("vector variables are not translated; declare one Variable per column")
        attrs = var.attributes if hasattr(var, "attributes") else {}
        integer = bool(attrs.get("integer", False) or attrs.get("boolean", False))
        lower = 0.0 if attrs.get("boolean", False) else -float("inf")
        upper = 1.0 if attrs.get("boolean", False) else float("inf")
        # Bounds attached as constraints are picked up below; these are the domain flags.
        if attrs.get("nonneg", False):
            lower = 0.0
        if attrs.get("nonpos", False):
            upper = 0.0
        out.column(var.name(), lower, upper, 0.0, integer)
    expr = problem.objective.expr if hasattr(problem.objective, "expr") else problem.objective.args[0]
    const, terms = _affine_expr(expr)
    out.constant = const
    for col in out.columns:
        col.cost = terms.get(col.name, 0.0)
    for con in problem.constraints:
        kind = type(con).__name__
        body = con.args[0] if con.args else con.expr
        c, t = _affine_expr(body)
        # CVXPY canonicalises lhs <= rhs into (lhs - rhs) <= 0, and equality into == 0.
        if kind in {"Inequality", "NonPos", "NonPositive"}:
            out.row(t, -float("inf"), -c)
        elif kind in {"NonNeg", "NonNegative"}:
            out.row(t, -c, float("inf"))
        elif kind in {"Equality", "Zero"}:
            out.row(t, -c, -c)
        else:
            raise ValueError(f"CVXPY constraint {kind} is not translated")
    return out
