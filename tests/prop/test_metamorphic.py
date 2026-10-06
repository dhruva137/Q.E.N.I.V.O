"""Metamorphic tests: row and column changes that must not move the optimal value.

T. Y. Chen, S. C. Cheung and S. M. Yiu, 1998. A proven answer is checked
against the planted value. ``not_proven`` is allowed; a different proven
value is not.
"""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from support import (
    duplicate_row,
    flip_column,
    permute_lp,
    planted_lp,
    scale_row,
    shift_constant,
    solve_checked,
)

from qenivo.io.mps import read_mps, write_mps

_fast = settings(max_examples=3, deadline=None, derandomize=True)


def _images(prob, rng):
    i = int(rng.integers(0, prob.m))
    j = int(rng.integers(0, prob.n))
    alpha = float(rng.uniform(0.5, 2.0))
    return (
        permute_lp(prob, rng),
        scale_row(prob, i, alpha),
        flip_column(prob, j),
        duplicate_row(prob),
    )


@_fast
@given(st.integers(1, 3), st.integers(2, 5), st.integers(0, 10_000))
def test_metamorphic_lp(m, n, seed):
    rng = np.random.default_rng(seed)
    prob, value = planted_lp(rng, m, max(n, m + 1))
    solve_checked(prob, "simplex", "optimal", value)
    for image in _images(prob, rng):
        solve_checked(image, "simplex", "optimal", value, time_limit=4.0)


@_fast
@given(st.integers(0, 10_000), st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False))
def test_objective_constant_shifts_value(seed, c0):
    rng = np.random.default_rng(seed)
    prob, value = planted_lp(rng, 2, 4)
    shifted, delta = shift_constant(prob, float(c0))
    solve_checked(shifted, "simplex", "optimal", value + delta)


def test_mps_roundtrip_keeps_planted_value(tmp_path):
    prob, value = planted_lp(np.random.default_rng(7), 2, 4)
    path = tmp_path / "planted.mps"
    write_mps(prob, path)
    solve_checked(read_mps(path), "simplex", "optimal", value, time_limit=4.0)


@pytest.mark.nightly
@settings(max_examples=10, deadline=None, derandomize=True)
@given(st.integers(1, 4), st.integers(3, 7), st.integers(0, 10_000))
def test_metamorphic_lp_nightly(m, n, seed):
    rng = np.random.default_rng(seed + 9)
    prob, value = planted_lp(rng, m, max(n, m + 1))
    for image in _images(prob, rng):
        for engine in ("simplex", "ipm"):
            solve_checked(image, engine, "optimal", value, time_limit=4.0)
