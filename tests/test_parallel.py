"""Multi-core: the parallel case pool agrees with serial solving; the race returns a certified winner."""
import dataclasses

import numpy as np

from conftest import infeasible_lp
from qenivo.models.williams import PUBLISHED_OPTIMUM, williams_lp
from qenivo.workload.parallel import race, solve_parallel


def test_parallel_pool_matches_serial():
    base = williams_lp()
    rng = np.random.default_rng(0)
    probs = [dataclasses.replace(base, c=base.c * rng.uniform(0.95, 1.05, base.n), name=f"c{k}") for k in range(6)]
    one = solve_parallel(probs, workers=1)
    two = solve_parallel(probs, workers=2)
    assert all(r["verdict"] == "optimal" for r in two["results"])
    for a, b in zip(one["results"], two["results"]):
        assert abs(a["objective"] - b["objective"]) <= 1e-8 * (1 + abs(a["objective"]))


def test_race_returns_a_certified_winner():
    r = race(williams_lp(), time_limit=120)
    assert r["winner"]["verdict"] == "optimal"
    assert abs(r["winner"]["objective"] - PUBLISHED_OPTIMUM) <= 0.01 + 1e-6 * PUBLISHED_OPTIMUM   # any certified engine may win


def test_race_proves_infeasibility():
    r = race(infeasible_lp(), time_limit=120)
    assert r["winner"]["verdict"] == "infeasible"
