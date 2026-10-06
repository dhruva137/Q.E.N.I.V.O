"""Infeasible and unbounded families. A wrong proven verdict fails; not_proven does not."""
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from support import infeasible_box, solve_checked, unbounded_ray

_fast = settings(max_examples=3, deadline=None, derandomize=True)


@_fast
@given(st.integers(1, 4), st.integers(0, 10_000))
def test_infeasible_by_construction(n, seed):
    prob = infeasible_box(np.random.default_rng(seed), n)
    for engine in ("simplex", "ipm", "pdhg"):
        solve_checked(prob, engine, "infeasible", time_limit=3.0)


@_fast
@given(st.integers(0, 10_000))
def test_unbounded_by_construction(seed):
    prob = unbounded_ray(np.random.default_rng(seed))
    for engine in ("simplex", "pdhg"):
        solve_checked(prob, engine, "unbounded", time_limit=3.0)
