"""Affine models shared by the modelling plugins.

A plugin translates Pyomo, PuLP, CVXPY, JuMP, or a tiny stand-in object into
this form and solves it through the C ABI. The C ABI minimises. A maximisation
model is negated here and the objective, duals and reduced costs are negated back.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_CAPI = Path(__file__).resolve().parents[1] / "capi"
if str(_CAPI) not in sys.path:
    sys.path.insert(0, str(_CAPI))

import capi_loader  # noqa: E402


def _bound(value, default):
    if value is None:
        return default
    return float(value)


class Column:
    def __init__(self, name: str, lower: float, upper: float, cost: float, integer: bool):
        self.name = name
        self.lower = lower
        self.upper = upper
        self.cost = cost
        self.integer = integer


class Row:
    def __init__(self, terms: dict, lower: float, upper: float, name: str = ""):
        self.terms = {str(k): float(v) for k, v in terms.items() if float(v) != 0.0}
        self.lower = lower
        self.upper = upper
        self.name = name


class Model:
    """Minimisation or maximisation of a linear (or quadratic) function on bounds."""

    def __init__(self, name: str = "model", sense: str = "min", constant: float = 0.0):
        if sense not in ("min", "max"):
            raise ValueError("sense must be min or max")
        self.name = name
        self.sense = sense
        self.constant = float(constant)
        self.columns: list[Column] = []
        self.rows: list[Row] = []
        self.quad: list[tuple[str, str, float]] = []
        self._index: dict[str, int] = {}

    def column(self, name: str, lower: float = 0.0, upper: float = math.inf, cost: float = 0.0,
               integer: bool = False) -> Column:
        if name in self._index:
            raise ValueError(f"duplicate column {name}")
        col = Column(name, float(lower), float(upper), float(cost), bool(integer))
        self._index[name] = len(self.columns)
        self.columns.append(col)
        return col

    def row(self, terms: dict, lower: float = -math.inf, upper: float = math.inf, name: str = "") -> Row:
        missing = [k for k in terms if k not in self._index]
        if missing:
            raise ValueError(f"unknown columns {missing}")
        item = Row(terms, float(lower), float(upper), name)
        self.rows.append(item)
        return item

    def quadratic(self, name_i: str, name_j: str, coef: float) -> None:
        """Add coef to Q_ij and Q_ji (the C ABI objective uses 0.5 x'Qx)."""
        if name_i not in self._index or name_j not in self._index:
            raise ValueError("quadratic term names an unknown column")
        self.quad.append((name_i, name_j, float(coef)))


class FakeVar:
    """Stand-in variable used when Pyomo, PuLP, CVXPY or JuMP is not installed."""

    def __init__(self, name, lb=0.0, ub=math.inf, obj=0.0, integer=False, binary=False):
        self.name = name
        self.lb = 0.0 if binary else lb
        self.ub = 1.0 if binary else ub
        self.obj = obj
        self.integer = bool(integer or binary)
        self.binary = bool(binary)


class FakeCons:
    def __init__(self, coeffs, lb=-math.inf, ub=math.inf, name=""):
        self.coeffs = dict(coeffs)
        self.lb = lb
        self.ub = ub
        self.name = name


class FakeModel:
    def __init__(self, sense="min", name="fake"):
        self.sense = sense
        self.name = name
        self.vars: list[FakeVar] = []
        self.cons: list[FakeCons] = []

    def var(self, name, lb=0.0, ub=math.inf, obj=0.0, integer=False, binary=False) -> FakeVar:
        item = FakeVar(name, lb, ub, obj, integer, binary)
        self.vars.append(item)
        return item

    def cons_add(self, coeffs, lb=-math.inf, ub=math.inf, name="") -> FakeCons:
        item = FakeCons(coeffs, lb, ub, name)
        self.cons.append(item)
        return item


def from_fake(model) -> Model:
    """Translate a FakeModel, or any object with .vars and .cons, into an affine Model."""
    sense = getattr(model, "sense", "min")
    out = Model(name=getattr(model, "name", "fake"), sense=sense, constant=float(getattr(model, "constant", 0.0)))
    variables = list(getattr(model, "vars"))
    for var in variables:
        lower = _bound(getattr(var, "lb", getattr(var, "lower", 0.0)), 0.0)
        upper = _bound(getattr(var, "ub", getattr(var, "upper", math.inf)), math.inf)
        cost = float(getattr(var, "obj", getattr(var, "cost", 0.0)))
        integer = bool(getattr(var, "integer", False) or getattr(var, "binary", False))
        if getattr(var, "binary", False):
            lower, upper = 0.0, 1.0
        out.column(str(var.name), lower, upper, cost, integer)
    for con in list(getattr(model, "cons")):
        if hasattr(con, "coeffs"):
            terms = con.coeffs
        elif hasattr(con, "terms"):
            terms = con.terms
        else:
            raise ValueError("a constraint needs coeffs or terms")
        lower = _bound(getattr(con, "lb", getattr(con, "lower", -math.inf)), -math.inf)
        upper = _bound(getattr(con, "ub", getattr(con, "upper", math.inf)), math.inf)
        out.row(terms, lower, upper, getattr(con, "name", ""))
    return out


def is_fake(model) -> bool:
    if isinstance(model, (Model, FakeModel)):
        return True
    module = type(model).__module__.split(".")[0]
    if module in {"pyomo", "pulp", "cvxpy"}:
        return False
    return hasattr(model, "vars") and hasattr(model, "cons")


def _csr(model: Model):
    n = len(model.columns)
    m = len(model.rows)
    ri, ci, vv = [], [], []
    for i, row in enumerate(model.rows):
        for name, coef in row.terms.items():
            ri.append(i)
            ci.append(model._index[name])
            vv.append(coef)
    matrix = sp.csr_matrix((vv, (ri, ci)), shape=(m, n), dtype=float)
    matrix.sum_duplicates()
    matrix.sort_indices()
    q = None
    if model.quad:
        qr, qc, qv = [], [], []
        for a, b, coef in model.quad:
            i, j = model._index[a], model._index[b]
            qr.append(i)
            qc.append(j)
            qv.append(coef)
            if i != j:
                qr.append(j)
                qc.append(i)
                qv.append(coef)
        q = sp.csr_matrix((qv, (qr, qc)), shape=(n, n), dtype=float)
        q.sum_duplicates()
    return matrix, q


class PluginResult:
    def __init__(self, raw, sense: str, constant: float):
        sign = -1.0 if sense == "max" else 1.0
        self.status = raw.status
        self.status_name = raw.status_name
        self.objective = sign * raw.objective + constant
        self.x = raw.x
        self.y = None if raw.y is None else sign * raw.y
        self.reduced_costs = None if raw.reduced_costs is None else sign * raw.reduced_costs
        self.column_basis = raw.column_basis
        self.row_basis = raw.row_basis
        self.iterations = raw.iterations
        self.certificate = raw.certificate
        self.names = []


def solve(model: Model, engine: str | None = None, tol: float = 1e-8, time_limit: float = 60.0) -> PluginResult:
    """Solve an affine model through the C ABI."""
    if any(col.integer for col in model.columns) and engine in (None, "simplex", "native", "native-simplex"):
        engine = "milp"
    engine = engine or "simplex"
    sign = -1.0 if model.sense == "max" else 1.0
    matrix, quad = _csr(model)
    cost = sign * np.array([col.cost for col in model.columns], dtype=float)
    lower = np.array([col.lower for col in model.columns], dtype=float)
    upper = np.array([col.upper for col in model.columns], dtype=float)
    row_lower = np.array([row.lower for row in model.rows], dtype=float) if model.rows else np.zeros(0)
    row_upper = np.array([row.upper for row in model.rows], dtype=float) if model.rows else np.zeros(0)
    integer = np.array([1 if col.integer else 0 for col in model.columns], dtype=np.int32)
    q_ptr = q_idx = q_val = None
    if quad is not None:
        q_ptr, q_idx, q_val = quad.indptr, quad.indices, quad.data
        if sign < 0:
            q_val = sign * np.array(q_val, dtype=float)
    raw = capi_loader.solve_csr(
        cost, matrix.indptr, matrix.indices, matrix.data, row_lower, row_upper, lower, upper,
        integrality=integer if integer.any() else None, q_ptr=q_ptr, q_idx=q_idx, q_val=q_val,
        engine=engine, tol=tol, time_limit=time_limit, name="".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in model.name) or "model")
    result = PluginResult(raw, model.sense, model.constant)
    result.names = [col.name for col in model.columns]
    return result
