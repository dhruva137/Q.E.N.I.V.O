"""Provenance guard: prove that no optimisation-solver or sparse-direct-solver code runs.

The problem statement requires a core "not built upon any existing open source solver
library". QENIVO's rule: the solve path may use only array and dense linear-algebra
primitives (NumPy, SciPy's LAPACK wrappers and sparse matrix *containers*, CuPy arrays and
NVRTC-compiled kernels written in this repository).

Two checks, because "imported" and "used" are different things:
  1. PACKAGES: separately installed solver packages (HiGHS, Gurobi, CPLEX, Xpress, MOSEK,
     OR-Tools, SCIP, OSQP, ...) must not even be imported.
  2. TRIPWIRES: SciPy imports its SuperLU wrapper as a side effect of `import scipy.sparse`,
     so its presence proves nothing. Instead every sparse direct solver and optimiser entry
     point (splu, spsolve, spilu, factorized, linprog, milp, minimize, ...) is replaced by a
     wrapper that records the call and raises. A clean run has ZERO tripwire hits.
The test suite installs the tripwires, runs every engine, and asserts both checks pass.
"""
from __future__ import annotations

import sys

FORBIDDEN_PACKAGES = (
    "highspy", "gurobipy", "cplex", "docplex", "xpress", "mosek", "ortools", "pyscipopt",
    "osqp", "scs", "ecos", "clarabel", "cyipopt", "cvxopt", "cvxpy", "pulp", "sksparse",
    "scikits.umfpack", "nvidia.cudss", "pypardiso", "qdldl",
    # AD, NLP and sparse-direct libraries later engines must not import.
    "casadi", "jax", "autograd", "petsc4py", "nlopt",
)

_TRIP_TARGETS = {
    "scipy.sparse.linalg": ("splu", "spsolve", "spilu", "factorized", "spsolve_triangular", "use_solver"),
    "scipy.optimize": ("linprog", "milp", "minimize", "quadratic_assignment", "lsq_linear", "nnls"),
}
_HITS: list = []
_INSTALLED = False


def _wrap(mod_name, fn_name, fn):
    def tripwire(*args, **kwargs):
        _HITS.append(f"{mod_name}.{fn_name}")
        raise RuntimeError(f"provenance tripwire: {mod_name}.{fn_name} called from the solve path")
    tripwire.__wrapped__ = fn
    return tripwire


def install_tripwires() -> None:
    """Replace forbidden entry points with recording tripwires (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    import importlib
    for mod_name, names in _TRIP_TARGETS.items():
        try:
            mod = importlib.import_module(mod_name) if mod_name == "scipy.sparse.linalg" else sys.modules.get(mod_name)
        except Exception:
            mod = None
        if mod is None:
            continue
        for n in names:
            if hasattr(mod, n) and not hasattr(getattr(mod, n), "__wrapped__"):
                setattr(mod, n, _wrap(mod_name, n, getattr(mod, n)))
    _INSTALLED = True


def guard() -> dict:
    pkgs = sorted(m for m in list(sys.modules) if m.split(".")[0] in {p.split(".")[0] for p in FORBIDDEN_PACKAGES}
                  and m.startswith(FORBIDDEN_PACKAGES))
    optimize_loaded = "scipy.optimize" in sys.modules
    return {"forbidden_packages": pkgs, "tripwire_hits": list(_HITS), "tripwires_installed": _INSTALLED,
            "scipy_optimize_imported": optimize_loaded,
            "clean": not pkgs and not _HITS}


def assert_clean() -> None:
    g = guard()
    if not g["clean"]:
        raise RuntimeError(f"provenance guard failed: packages {g['forbidden_packages']}, calls {g['tripwire_hits']}")
