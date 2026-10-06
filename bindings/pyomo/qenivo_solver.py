"""Pyomo solver plugin. Linear models are translated to the Qenivo C ABI.

Hart, Laird, Watson, Woodruff, Hackebeil, Nicholson, Siirola, Pyomo — Optimization
Modeling in Python, Springer, 2017. The translation below is original; it uses
Pyomo's public standard representation when that package is installed. A FakeModel
from bindings/affine.py is solved without Pyomo.
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
    """Return an affine Model for a FakeModel or a linear Pyomo ConcreteModel."""
    affine = _affine()
    if affine.is_fake(model):
        return model if isinstance(model, affine.Model) else affine.from_fake(model)
    return _from_pyomo(model)


def solve(model, engine=None, tol=1e-8, time_limit=60.0):
    return _affine().solve(translate(model), engine=engine, tol=tol, time_limit=time_limit)


def _from_pyomo(model):
    import pyomo.environ as pyo
    from pyomo.repn.standard_repn import generate_standard_repn

    objectives = list(model.component_data_objects(pyo.Objective, active=True, descend_into=True))
    if len(objectives) != 1:
        raise ValueError("Qenivo's Pyomo plugin expects one active objective")
    obj = objectives[0]
    sense = "max" if obj.sense == pyo.maximize else "min"
    out = _affine().Model(name=getattr(model, "name", "pyomo") or "pyomo", sense=sense)
    ids = {}
    for var in model.component_data_objects(pyo.Var, sort=True):
        if getattr(var, "is_indexed", lambda: False)():
            continue
        lower = var.lb if var.lb is not None else -float("inf")
        upper = var.ub if var.ub is not None else float("inf")
        integer = bool(var.is_integer() or var.is_binary())
        if var.is_binary():
            lower, upper = 0.0, 1.0
        col = out.column(var.name, lower, upper, 0.0, integer)
        ids[id(var)] = col.name
    repn = generate_standard_repn(obj.expr, compute_values=True, quadratic=True)
    if repn.nonlinear_expr is not None:
        raise ValueError("nonlinear Pyomo objectives are outside this plugin")
    costs = {col.name: 0.0 for col in out.columns}
    for coef, var in zip(repn.linear_coefs or [], repn.linear_vars or []):
        costs[ids[id(var)]] = float(coef)
    for col in out.columns:
        col.cost = costs[col.name]
    out.constant = float(repn.constant or 0.0)
    if getattr(repn, "quadratic_vars", None):
        for (vi, vj), coef in zip(repn.quadratic_vars, repn.quadratic_coefs):
            out.quadratic(ids[id(vi)], ids[id(vj)], float(coef))
    for con in model.component_data_objects(pyo.Constraint, active=True, descend_into=True):
        if con.body is None:
            continue
        body = generate_standard_repn(con.body, compute_values=True)
        if body.nonlinear_expr is not None:
            raise ValueError(f"nonlinear constraint {con.name}")
        terms = {}
        for coef, var in zip(body.linear_coefs or [], body.linear_vars or []):
            terms[ids[id(var)]] = terms.get(ids[id(var)], 0.0) + float(coef)
        shift = float(body.constant or 0.0)
        lower = -float("inf") if con.lower is None else float(pyo.value(con.lower)) - shift
        upper = float("inf") if con.upper is None else float(pyo.value(con.upper)) - shift
        out.row(terms, lower, upper, con.name)
    return out
