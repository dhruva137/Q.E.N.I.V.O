"""Property tests: random LPs, QPs and MILPs with an optimum planted by construction.

Claessen and Hughes, QuickCheck, ICFP 2000. The generator is tests/prop/support.py.
"""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from support import planted_lp, planted_milp, planted_qp, solve_checked

_fast = settings(max_examples=3, deadline=None, derandomize=True)
_night = settings(max_examples=15, deadline=None, derandomize=True)


def _lp(m, n, seed, engines):
    prob, value = planted_lp(np.random.default_rng(seed), m, max(n, m + 1))
    for engine in engines:
        solve_checked(prob, engine, "optimal", value)


@_fast
@given(st.integers(1, 3), st.integers(2, 5), st.integers(0, 10_000))
def test_planted_lp(m, n, seed):
    _lp(m, n, seed, ("simplex", "ipm", "pdhg"))


@pytest.mark.nightly
@_night
@given(st.integers(1, 5), st.integers(2, 8), st.integers(0, 10_000))
def test_planted_lp_nightly(m, n, seed):
    _lp(m, n, seed + 50_000, ("simplex", "ipm", "pdhg"))


@_fast
@given(st.integers(2, 4), st.integers(0, 10_000))
def test_planted_qp(n, seed):
    prob, value = planted_qp(np.random.default_rng(seed), n)
    solve_checked(prob, "pdqp", "optimal", value, time_limit=5.0)


@_fast
@given(st.integers(1, 2), st.integers(2, 4), st.integers(0, 10_000))
def test_planted_milp(m, n, seed):
    prob, value = planted_milp(np.random.default_rng(seed), m, max(n, m + 1))
    solve_checked(prob, "milp", "optimal", value, time_limit=8.0)
