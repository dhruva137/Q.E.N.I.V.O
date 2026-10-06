"""Global nonlinear engine: spatial branch-and-bound for pooling, outer approximation for convex MINLP.

Haverly instances are the three problems in C. A. Haverly, ACM SIGMAP Bulletin 25
(1978), with the profits tabulated by Adhya, Tawarmalani and Sahinidis,
Industrial & Engineering Chemistry Research 38 (1999). Convex MINLPs go through
Duran and Grossmann, Mathematical Programming 36 (1986). The master MILP is the
existing ``solve_milp`` engine.
"""
from __future__ import annotations

from ..nlp.oa import outer_approximation
from ..nlp.pooling import solve_haverly
from ..nlp.problem import NLPModel
from .registry import register_engine


def solve_nlp_global(prob=None, tol=1e-6, time_limit=30.0, backend="numpy", verbose=False, **options):
    """Dispatch a Haverly instance, a convex MINLP, or a continuous NLP."""
    instance = options.get("instance")
    if isinstance(prob, str):
        key = prob.strip().lower().replace("-", "").replace("_", "")
        if key.startswith("haverly"):
            instance = int(key.replace("haverly", "") or "1")
    if isinstance(prob, int):
        instance = prob
    if instance is not None:
        return solve_haverly(int(instance), tol=tol, time_limit=time_limit, verbose=verbose,
                             node_limit=int(options.get("node_limit", 2000)))
    if isinstance(prob, NLPModel):
        if prob.integers:
            return outer_approximation(prob, tol=tol, time_limit=time_limit)
        from ..nlp.ipm import solve_ipm
        return solve_ipm(prob, tol=tol, time_limit=time_limit, verbose=verbose)
    raise TypeError("nlp-global expects an NLPModel or a Haverly instance id")


register_engine(
    "nlp-global",
    solve_nlp_global,
    classes=("NLP", "MINLP"),
    description="McCormick spatial branch-and-bound for Haverly pooling, and Duran-Grossmann outer approximation",
)
