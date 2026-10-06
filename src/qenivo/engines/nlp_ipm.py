"""Filter line-search interior-point NLP, registered as the engine "nlp-ipm".

A. Waechter and L. T. Biegler, Mathematical Programming 106 (2006) 25-57.
The Newton matrix is factored by the dense LDL routine in ``qenivo.nlp.dense_kkt``.
"""
from __future__ import annotations

from ..nlp.ipm import solve_ipm
from ..nlp.problem import NLPModel
from .registry import register_engine


def solve_nlp_ipm(prob, tol=1e-8, time_limit=30.0, backend="numpy", verbose=False, **options):
    """Solve an :class:`~qenivo.nlp.problem.NLPModel` by the filter interior-point method."""
    if not isinstance(prob, NLPModel):
        raise TypeError("nlp-ipm solves an NLPModel")
    return solve_ipm(prob, tol=tol, time_limit=time_limit, verbose=verbose,
                     x0=options.get("x0"), lb=options.get("lb"), ub=options.get("ub"))


register_engine(
    "nlp-ipm",
    solve_nlp_ipm,
    classes=("NLP", "MINLP"),
    description="filter line-search interior point (Waechter and Biegler 2006) with a dense Bunch-Kaufman factorisation",
)
