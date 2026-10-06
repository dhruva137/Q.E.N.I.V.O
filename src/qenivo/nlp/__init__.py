"""Nonlinear models: expression graphs, derivatives, and solvers.

Public pieces live in the submodules (``expr``, ``ad``, ``dense_kkt``, ``ipm``,
``pooling``, ``oa``, ``kkt``). Engines register themselves from
``qenivo.engines.nlp_ipm`` and ``qenivo.engines.nlp_global``.
"""

from .expr import Expr, Graph, cos, exp, log, sin, sqrt
from .problem import NLPModel

__all__ = ["Expr", "Graph", "NLPModel", "cos", "exp", "log", "sin", "sqrt"]
