"""Crude-unloading MILP (Lee et al. 1996): build sizes and prove small optimal."""
from __future__ import annotations

import pytest

from qenivo import solve
from qenivo.engines import native_simplex as ns
from qenivo.models.industry_ext.crude_unloading import (
    SIZES,
    crude_unloading,
    model_sizes,
    size_ok_to_solve,
)

# Branch and bound re-solves an LP relaxation per node. The pure-Python simplex reaches the same
# proven optimum (2056.6) but took 188 s on a 4-core laptop (1 Oct 2026), against 7.3 s natively
# in the W07 audit.
NEEDS_NATIVE = pytest.mark.skipif(
    ns.library() is None,
    reason="needs the native simplex to prove optimality inside the time limit: " + ns.status())


def test_sizes_build_and_report_dimensions():
    sizes = model_sizes(seed=0)
    assert set(sizes) == {"small", "medium", "large"}
    assert sizes["small"]["rows"] < sizes["medium"]["rows"] < sizes["large"]["rows"]
    assert sizes["small"]["integers"] > 0
    assert sizes["small"]["meta"]["periods"] == SIZES["small"]["periods"]
    assert sizes["medium"]["meta"]["storage"] == 6
    assert sizes["large"]["meta"]["charging"] == 4
    assert sizes["large"]["meta"]["cdus"] == 3


def test_seed_is_deterministic():
    a = crude_unloading("small", seed=3)
    b = crude_unloading("small", seed=3)
    assert a.fingerprint() == b.fingerprint()
    c = crude_unloading("small", seed=4)
    assert a.fingerprint() != c.fingerprint()


@NEEDS_NATIVE
def test_small_proven_optimal_native_milp():
    p = crude_unloading("small", seed=0)
    assert p.is_mip
    assert "Lee" in p.notes[0] or "Lee" in p.meta.get("citation", "")
    sol = solve(p, engine="milp", time_limit=60)
    assert sol.verdict == "optimal", sol.summary()
    assert sol.objective is not None
    assert sol.engine.get("gap", 1.0) <= 1e-9
    # Re-solve: same proven objective (deterministic BnB on this instance).
    again = solve(p, engine="milp", time_limit=60)
    assert again.verdict == "optimal"
    assert abs(again.objective - sol.objective) <= 1e-5 * (1 + abs(sol.objective))


@pytest.mark.parametrize("size", ["medium", "large"])
def test_medium_large_build_only_under_ram_gate(size):
    p = crude_unloading(size, seed=0)
    assert p.m > 0 and p.n > 0 and p.is_mip
    if not size_ok_to_solve(size):
        pytest.skip(f"free RAM < 2.5 GB; not solving {size}")
