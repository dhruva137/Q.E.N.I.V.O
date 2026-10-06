"""QENIVO: a sovereign, certified optimisation engine for LP, MILP and convex QP.

    import qenivo
    sol = qenivo.solve("plan.mps")           # route, solve, check, certify
    print(sol.summary())
    sol.save_certificate("plan.cert.json")

Layers: kernels (own CUDA kernels) -> engines (simplex, interior point, batched PDHG, PDQP,
branch and bound) -> certify (KKT / Farkas / ray certificates, independent verifier) ->
workload (case stacks, distributive recursion, infeasibility explanation, ranging,
crude valuation).
"""
__version__ = "0.1.0"

from .api import Solution, read, solve  # noqa: E402
from .edition import detect_edition, edition_meta  # noqa: E402
from .engines.registry import engines, register_engine  # noqa: E402
from .model import ModelBuilder, Problem  # noqa: E402

try:  # the crude-valuation engine ships only in the full edition (see docs/EDITIONS.md)
    from .workload.crude_value import CrudeValueResult, crude_value  # noqa: E402
except ImportError:  # public edition
    CrudeValueResult = crude_value = None

__all__ = [
    "Problem", "ModelBuilder", "Solution", "read", "solve",
    "crude_value", "CrudeValueResult",
    "register_engine", "engines", "detect_edition", "edition_meta", "__version__",
]
