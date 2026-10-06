"""Differential testing: engines that both prove optimality must agree.

W. M. McKeeman, Digital Technical Journal, 1998. A disagreement is written
under tests/diff/findings/ and fails the test. ``not_proven`` is not a
disagreement.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from support import LP_ENGINES, planted_lp, solve_checked

FINDINGS = Path(__file__).resolve().parent / "findings"


def _record(payload: dict) -> Path:
    FINDINGS.mkdir(exist_ok=True)
    path = FINDINGS / f"disagree_seed{payload['seed']}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _compare(seed, m, n):
    prob, value = planted_lp(np.random.default_rng(seed), m, max(n, m + 1))
    proven = []
    for engine in LP_ENGINES:
        sol = solve_checked(prob, engine, "optimal", value, time_limit=3.0)
        if sol.verdict == "optimal":
            proven.append({"engine": engine, "objective": sol.objective})
    if len(proven) < 2:
        return
    values = [row["objective"] for row in proven]
    spread = max(values) - min(values)
    if spread > 1e-5 * (1.0 + abs(values[0])):
        path = _record({"seed": int(seed), "m": m, "n": n, "planted": value, "proven": proven,
                        "spread": spread})
        pytest.fail(f"engines disagree by {spread:.3e}; wrote {path}")


@settings(max_examples=3, deadline=None, derandomize=True)
@given(st.integers(1, 3), st.integers(2, 5), st.integers(0, 10_000))
def test_engines_agree_on_planted_lp(m, n, seed):
    _compare(seed, m, n)


@pytest.mark.nightly
@settings(max_examples=12, deadline=None, derandomize=True)
@given(st.integers(1, 4), st.integers(2, 7), st.integers(0, 10_000))
def test_engines_agree_nightly(m, n, seed):
    _compare(seed + 80_000, m, n)
