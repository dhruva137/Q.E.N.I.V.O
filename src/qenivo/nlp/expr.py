"""Expression graph for nonlinear models.

Operators are addition, subtraction, multiplication, division and powers, plus
``exp``, ``log``, ``sqrt``, ``sin`` and ``cos``. Products of two variables are
bilinear; a variable squared is quadratic. Every value is produced by a walk
of this tape in Python. A compiled evaluator is optional and is not required
for any result reported here.

The tape is a Wengert list: children are created before their parents, so
node order is a topological order. ``qenivo.nlp.ad`` differentiates it.
"""
from __future__ import annotations

import numpy as np


class Graph:
    """Tape of operations shared by every expression built from it."""

    def __init__(self):
        self.ops: list[tuple] = []
        self.var_names: list[str] = []
        self.lb: list[float] = []
        self.ub: list[float] = []

    @property
    def n_var(self) -> int:
        return len(self.var_names)

    def var(self, name: str, lb: float = -np.inf, ub: float = np.inf) -> "Expr":
        """Append a decision variable and return the expression that reads it."""
        if name in self.var_names:
            raise ValueError(f"duplicate variable {name!r}")
        idx = len(self.var_names)
        self.var_names.append(name)
        self.lb.append(float(lb))
        self.ub.append(float(ub))
        self.ops.append(("var", idx))
        return Expr(self, len(self.ops) - 1)

    def const(self, value: float) -> "Expr":
        """Append a constant node."""
        self.ops.append(("const", float(value)))
        return Expr(self, len(self.ops) - 1)

    def _as(self, other) -> "Expr":
        if isinstance(other, Expr):
            if other.graph is not self:
                raise ValueError("cannot combine expressions from different graphs")
            return other
        return self.const(float(other))


class Expr:
    """One node in a :class:`Graph`. Arithmetic builds new nodes."""

    def __init__(self, graph: Graph, node: int):
        self.graph = graph
        self.node = node

    def _bin(self, kind: str, other) -> "Expr":
        g = self.graph
        rhs = g._as(other)
        g.ops.append((kind, self.node, rhs.node))
        return Expr(g, len(g.ops) - 1)

    def __add__(self, other) -> "Expr":
        return self._bin("add", other)

    def __radd__(self, other) -> "Expr":
        return self.graph._as(other)._bin("add", self)

    def __sub__(self, other) -> "Expr":
        return self._bin("sub", other)

    def __rsub__(self, other) -> "Expr":
        return self.graph._as(other)._bin("sub", self)

    def __mul__(self, other) -> "Expr":
        return self._bin("mul", other)

    def __rmul__(self, other) -> "Expr":
        return self.graph._as(other)._bin("mul", self)

    def __truediv__(self, other) -> "Expr":
        return self._bin("div", other)

    def __rtruediv__(self, other) -> "Expr":
        return self.graph._as(other)._bin("div", self)

    def __neg__(self) -> "Expr":
        g = self.graph
        g.ops.append(("neg", self.node))
        return Expr(g, len(g.ops) - 1)

    def __pos__(self) -> "Expr":
        return self

    def __pow__(self, other) -> "Expr":
        g = self.graph
        if isinstance(other, (int, float, np.floating)):
            g.ops.append(("powc", self.node, float(other)))
            return Expr(g, len(g.ops) - 1)
        # a**b = exp(b * log(a)) for a > 0; derivatives come from those ops.
        return exp(other * log(self))


def _unary(kind: str, expr: Expr) -> Expr:
    if not isinstance(expr, Expr):
        raise TypeError(f"{kind} expects an expression")
    expr.graph.ops.append((kind, expr.node))
    return Expr(expr.graph, len(expr.graph.ops) - 1)


def exp(expr: Expr) -> Expr:
    """Natural exponential."""
    return _unary("exp", expr)


def log(expr: Expr) -> Expr:
    """Natural logarithm. The argument must be positive at evaluation points."""
    return _unary("log", expr)


def sqrt(expr: Expr) -> Expr:
    """Square root. The argument must be positive at evaluation points."""
    return _unary("sqrt", expr)


def sin(expr: Expr) -> Expr:
    """Sine, argument in radians."""
    return _unary("sin", expr)


def cos(expr: Expr) -> Expr:
    """Cosine, argument in radians."""
    return _unary("cos", expr)


def evaluate(expr: Expr, x: np.ndarray) -> float:
    """Value of ``expr`` at the variable vector ``x``."""
    from .ad import forward_values
    val, _ = forward_values(expr.graph, np.asarray(x, dtype=np.float64), None)
    out = float(val[expr.node])
    if not np.isfinite(out):
        raise ValueError("expression is not finite at this point")
    return out
