"""Case-stack generators: shapes, determinism, sparsity, apply shares A."""
from __future__ import annotations

import numpy as np
import pytest

from qenivo.models.refinery import refinery_level
from qenivo.workload.stacks import CaseDelta, groups, make_stack, nn_order, refinery_groups

KINDS = ("one_crude", "one_unit", "factor_price", "factor_mixed", "independent", "cargo_menu")


@pytest.fixture(scope="module")
def small_prob():
    return refinery_level(1, seed=0)


def test_groups_cost_replay(small_prob):
    g = groups(small_prob)
    assert g["K"] >= 1 and g["crude_cols"].size == g["R"] * g["T"] * g["K"]
    g2 = refinery_groups(1, seed=0)
    assert np.array_equal(g["crude_cols"], g2["crude_cols"])
    assert np.array_equal(g["ship_cols"], g2["ship_cols"])


@pytest.mark.parametrize("kind", KINDS)
def test_make_stack_shapes(small_prob, kind):
    S = 8
    deltas = make_stack(small_prob, kind, S, seed=7)
    assert len(deltas) == S
    assert all(isinstance(d, CaseDelta) for d in deltas)
    for d in deltas:
        assert d.c_idx.size == d.c_val.size
        assert d.uc_idx.size == d.uc_val.size
        assert d.lc_idx.size == d.lc_val.size
        assert d.shock.size >= 1 or kind == "independent"  # independent stores full mult shock


def test_determinism_by_seed(small_prob):
    a = make_stack(small_prob, "independent", 16, seed=42)
    b = make_stack(small_prob, "independent", 16, seed=42)
    c = make_stack(small_prob, "factor_price", 8, seed=3)
    d = make_stack(small_prob, "factor_price", 8, seed=3)
    for x, y in zip(a, b):
        assert np.array_equal(x.c_idx, y.c_idx)
        assert np.allclose(x.c_val, y.c_val)
        assert np.allclose(x.shock, y.shock)
    for x, y in zip(c, d):
        assert np.allclose(x.c_val, y.c_val)
        assert np.allclose(x.shock, y.shock)
    e = make_stack(small_prob, "independent", 16, seed=43)
    assert not np.allclose(a[0].shock, e[0].shock)


def test_deltas_sparse(small_prob):
    S = 5
    one = make_stack(small_prob, "one_crude", S, seed=0, crude_k=0)
    g = groups(small_prob)
    n_crude0 = int((g["crude_k"] == 0).sum())
    for d in one:
        assert d.c_idx.size == n_crude0
        assert d.c_idx.size < small_prob.n
        assert d.uc_idx.size == 0 and d.lc_idx.size == 0
    unit = make_stack(small_prob, "one_unit", S, seed=0)
    for d in unit:
        assert d.uc_idx.size > 0
        assert d.uc_idx.size < small_prob.m
        assert d.c_idx.size == 0


def test_independent_matches_uniform_multipliers(small_prob):
    """Same seed → same U(0.9, 1.1) multipliers as rng.uniform(0.9, 1.1, (n, S))."""
    S, seed, a = 12, 99, 0.1
    deltas = make_stack(small_prob, "independent", S, seed=seed, a=a)
    rng = np.random.default_rng(seed)
    f = rng.uniform(1.0 - a, 1.0 + a, (small_prob.n, S))
    for s, d in enumerate(deltas):
        c_full = small_prob.c.copy()
        c_full[d.c_idx] = d.c_val
        assert np.allclose(c_full, small_prob.c * f[:, s])
        # multipliers on every coefficient are in [0.9, 1.1]
        assert np.all(f[:, s] >= 0.9 - 1e-15) and np.all(f[:, s] <= 1.1 + 1e-15)
        # zeros stay zero → not stored
        assert np.all(small_prob.c[d.c_idx] != 0) or d.c_idx.size == 0


def test_one_crude_only_touches_one_k(small_prob):
    g = groups(small_prob)
    k = 1 if g["K"] > 1 else 0
    deltas = make_stack(small_prob, "one_crude", 6, seed=0, crude_k=k, lo=-0.1, hi=0.1)
    allowed = set(g["crude_cols"][g["crude_k"] == k].tolist())
    other = set(g["crude_cols"][g["crude_k"] != k].tolist())
    for d in deltas:
        assert set(d.c_idx.tolist()) == allowed
        assert not (set(d.c_idx.tolist()) & other)


def test_apply_shares_A(small_prob):
    deltas = make_stack(small_prob, "one_crude", 3, seed=1, lo=-0.2, hi=0.2)
    for d in deltas:
        p = d.apply(small_prob)
        assert p.A is small_prob.A
        assert id(p.A) == id(small_prob.A)
        if abs(float(d.shock[0])) > 1e-15:
            assert p.c is not small_prob.c
            assert not np.allclose(p.c, small_prob.c)


def test_nn_order_is_permutation(small_prob):
    deltas = make_stack(small_prob, "factor_price", 10, seed=5)
    order = nn_order(deltas)
    assert sorted(order) == list(range(10))
    assert len(set(order)) == 10
    # also accepts raw shock matrix
    X = np.stack([d.shock for d in deltas])
    order2 = nn_order(X)
    assert sorted(order2) == list(range(10))


def test_to_case_roundtrip_names(small_prob):
    d = make_stack(small_prob, "one_crude", 1, seed=0)[0]
    case = d.to_case(small_prob)
    assert case.name == d.name
    assert len(case.cost) == d.c_idx.size


@pytest.mark.parametrize("kind", KINDS)
def test_smoke_all_kinds_level2(kind):
    """Optional smoke: make_stack on refinery_level(2) for all kinds, S=8."""
    prob = refinery_level(2, seed=0)
    deltas = make_stack(prob, kind, 8, seed=1)
    assert len(deltas) == 8
    p = deltas[0].apply(prob)
    assert p.A is prob.A
    assert deltas[0].n_changed() >= 1 or kind == "independent"
