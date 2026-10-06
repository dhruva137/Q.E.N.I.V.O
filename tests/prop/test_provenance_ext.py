"""The provenance guard must notice AD and factorisation libraries added for later modules."""
import sys
import types

from qenivo.provenance import FORBIDDEN_PACKAGES, guard

EXTRA = ("casadi", "jax", "autograd", "petsc4py", "nlopt", "cvxpy")


def test_expansion_libraries_are_forbidden():
    missing = [name for name in EXTRA if name not in FORBIDDEN_PACKAGES]
    assert not missing, missing


def test_synthetic_import_is_not_clean():
    sentinel = "casadi"
    added = sentinel not in sys.modules
    if added:
        sys.modules[sentinel] = types.ModuleType(sentinel)
    try:
        report = guard()
        assert sentinel in report["forbidden_packages"]
        assert report["clean"] is False
    finally:
        if added:
            del sys.modules[sentinel]
