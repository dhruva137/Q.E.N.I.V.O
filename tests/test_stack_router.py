"""Stack router: profiles and policy on synthetic W01 stack kinds (CPU only, tiny S)."""
from __future__ import annotations

import pytest

from qenivo.models.refinery import refinery_level
from qenivo.workload.router import (
    RANK_CPU_MAX,
    StackProfile,
    classify_stack,
    route_stack,
)
from qenivo.workload.stacks import make_stack

KINDS = ("one_crude", "one_unit", "factor_price", "factor_mixed", "independent", "cargo_menu")

# Expected route when no GPU is offered (backend=numpy / prefer_gpu still consults gpu_available;
# we force prefer_gpu=False so the independent lane is deterministic on CPU-only machines).
EXPECTED_NO_GPU = {
    "one_crude": "cpu_warm_ray",
    "cargo_menu": "cpu_warm_ray",
    "one_unit": "cpu_warm_ray",
    "factor_price": "cpu_warm",      # rank ~3, within RANK_CPU_MAX
    "factor_mixed": "mixed",
    "independent": "cpu_warm",      # would be gpu_batch if a GPU were present
}


@pytest.fixture(scope="module")
def tiny_prob():
    return refinery_level(1, seed=0)


@pytest.mark.parametrize("kind", KINDS)
def test_classify_stack_shapes(tiny_prob, kind):
    S = 8
    deltas = make_stack(tiny_prob, kind, S, seed=0)
    prof = classify_stack(tiny_prob, deltas)
    assert isinstance(prof, StackProfile)
    assert prof.n_cases == S
    assert prof.n_changed_coords >= 1
    assert 0 <= prof.delta_rank <= min(prof.n_changed_coords, S)
    assert prof.max_rel_move >= 0.0
    d = prof.to_dict()
    assert d["n_cases"] == S


def test_one_factor_kinds_are_rank_one(tiny_prob):
    for kind in ("one_crude", "one_unit", "cargo_menu"):
        deltas = make_stack(tiny_prob, kind, 6, seed=1)
        prof = classify_stack(tiny_prob, deltas)
        assert prof.delta_rank == 1, kind
        assert prof.is_one_factor_ray
        assert prof.only_costs or prof.only_bounds
        assert not prof.costs_and_bounds


def test_independent_is_high_rank(tiny_prob):
    deltas = make_stack(tiny_prob, "independent", 8, seed=2)
    prof = classify_stack(tiny_prob, deltas)
    assert prof.only_costs
    assert prof.delta_rank > RANK_CPU_MAX
    assert prof.n_changed_coords > 10


def test_factor_mixed_is_costs_and_bounds(tiny_prob):
    deltas = make_stack(tiny_prob, "factor_mixed", 8, seed=3)
    prof = classify_stack(tiny_prob, deltas)
    assert prof.costs_and_bounds
    assert not prof.only_costs and not prof.only_bounds


@pytest.mark.parametrize("kind,expected", list(EXPECTED_NO_GPU.items()))
def test_policy_picks_expected_route_no_gpu(tiny_prob, kind, expected):
    from qenivo.workload.router import StackRoutingPolicy
    deltas = make_stack(tiny_prob, kind, 8, seed=0)
    prof = classify_stack(tiny_prob, deltas)
    # Force the no-GPU branch so independent is cpu_warm (W01: GPU when present).
    pol = StackRoutingPolicy(prefer_gpu=False)
    dec = route_stack(prof, base=tiny_prob, deltas=deltas, backend="numpy",
                      threads=1, policy=pol)
    assert dec["route"] == expected, (kind, dec["route"], dec["reason"])
    assert "profile" in dec and dec["profile"]["n_cases"] == 8
    assert dec["max_workers"] == 1  # threads=1
    assert "reason" in dec


def test_gpu_policy_sends_independent_to_gpu_batch(tiny_prob, monkeypatch):
    from qenivo.workload import router as R
    monkeypatch.setattr(R, "gpu_available", lambda: True)
    deltas = make_stack(tiny_prob, "independent", 8, seed=0)
    prof = classify_stack(tiny_prob, deltas)
    dec = route_stack(prof, base=tiny_prob, deltas=deltas, backend="auto", threads=1)
    assert dec["route"] == "gpu_batch"
    assert dec["engine"] == "pdhg"
    assert dec["backend"] == "cupy"


def test_forced_engine_still_works(tiny_prob):
    deltas = make_stack(tiny_prob, "one_crude", 4, seed=0)
    prof = classify_stack(tiny_prob, deltas)
    dec = route_stack(prof, base=tiny_prob, deltas=deltas, engine="pdhg", backend="numpy")
    assert dec["route"] == "forced"
    assert dec["engine"] == "pdhg"
    assert dec["backend"] == "numpy"


def test_solve_cases_records_route(tiny_prob):
    """Auto path stamps route/reason/profile into CaseReport.engine (no GPU, tiny S)."""
    from qenivo.workload.cases import solve_cases
    deltas = make_stack(tiny_prob, "one_crude", 3, seed=0)
    cases = [d.to_case(tiny_prob) for d in deltas]
    rep = solve_cases(tiny_prob, cases, backend="numpy", threads=1)
    assert rep.engine.get("route") in ("cpu_warm_ray", "cpu_warm")
    assert rep.engine.get("profile") is not None
    assert "reason" in rep.engine
    assert all(s.verdict == "optimal" for s in rep.cases)
