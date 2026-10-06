"""A nonlinear program built on one expression graph."""
from __future__ import annotations

from dataclasses import dataclass, field

from .expr import Expr, Graph


@dataclass
class NLPModel:
    """min f(x) s.t. equalities = 0 and inequalities <= 0, with bounds on the graph.

    ``integers`` lists variable indices that an outer-approximation master treats
    as integer. The continuous solver leaves them alone unless bounds are fixed.
    """

    graph: Graph
    objective: Expr
    equalities: list = field(default_factory=list)
    inequalities: list = field(default_factory=list)
    integers: tuple = ()
    name: str = ""

    def __post_init__(self):
        exprs = [self.objective, *self.equalities, *self.inequalities]
        for e in exprs:
            if e.graph is not self.graph:
                raise ValueError("every expression must belong to the model graph")
