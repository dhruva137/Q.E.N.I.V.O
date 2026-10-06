"""Solve the tiny hand-checked refinery and industry models and match EXPECTED."""
from itertools import permutations

import pytest

from qenivo import solve
from qenivo.io.mps import read_mps, write_mps
from qenivo.models.industry import LIBRARY
from qenivo.models.industry_ext import EXPECTED as EXT_EXPECTED
from qenivo.models.refinery_full.planner import EXPECTED as REF_EXPECTED

EXPECTED = {**REF_EXPECTED, **EXT_EXPECTED}
NAMES = (
    "refinery_pims",
    "refinery_delta",
    "refinery_pool_relax",
    "petrochemical",
    "sced",
    "hydro_thermal",
    "vrptw",
    "multi_echelon",
    "rcpsp",
)
MILP = {"vrptw", "rcpsp"}


def assert_close(got, exp):
    assert got is not None
    assert abs(got - exp) <= 1e-6 * max(1.0, abs(exp)), (got, exp)


def _solve(prob, name):
    if name in MILP:
        return solve(prob, engine="milp", time_limit=15)
    return solve(prob, engine="simplex", polish=False, time_limit=15)


@pytest.fixture(scope="module")
def solved():
    out = {}
    for name in NAMES:
        prob = LIBRARY[name]()
        out[name] = (prob, _solve(prob, name))
    return out


def test_names_are_in_the_library():
    assert set(NAMES) <= set(LIBRARY)
    assert set(EXPECTED) == set(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_proven_objective_matches_expected(name, solved):
    prob, sol = solved[name]
    assert sol.verdict == "optimal", sol.summary()
    assert_close(sol.objective, EXPECTED[name])
    assert prob.is_qp is False
    assert prob.is_mip is (name in MILP)
    if name == "refinery_pool_relax":
        term = prob.meta["bilinear"][0]
        assert term["w"] == "sour" and term["x"] == "frac" and term["y"] == "vol"
        assert term["coef"] == 1.0


def _route_cost(seq, spec):
    travel, window, demand = spec["travel"], spec["window"], spec["demand"]
    capacity, service = spec["capacity"], spec["service"]
    load = 0
    clock = 0.0
    prev = 0
    cost = 0
    for node in seq:
        cost += travel[(prev, node)]
        clock += travel[(prev, node)] + service
        lo, hi = window[node]
        if clock < lo:
            clock = lo
        if clock > hi or load + demand[node] > capacity:
            return None
        load += demand[node]
        prev = node
    cost += travel[(prev, 0)]
    clock += travel[(prev, 0)]
    depot_lo, depot_hi = window[0]
    if clock < depot_lo or clock > depot_hi:
        return None
    return cost


def enumerate_tours(spec):
    """Every ordered partition of the clients into capacity-feasible routes."""
    customers = list(spec["customers"])
    best = None
    seen = 0
    for perm in permutations(customers):
        def walk(start, cost_so_far):
            nonlocal best, seen
            if best is not None and cost_so_far >= best and start < len(perm):
                return
            if start == len(perm):
                seen += 1
                if best is None or cost_so_far < best:
                    best = cost_so_far
                return
            for length in range(1, len(perm) - start + 1):
                cost = _route_cost(perm[start:start + length], spec)
                if cost is None:
                    continue
                walk(start + length, cost_so_far + cost)

        walk(0, 0)
    return best, seen


def test_vrptw_enumeration_matches_expected(solved):
    spec = solved["vrptw"][0].meta["vrptw"]
    best, seen = enumerate_tours(spec)
    assert seen > 0
    assert best == EXPECTED["vrptw"]
    assert_close(solved["vrptw"][1].objective, best)


def _precedence_orders(jobs, prec):
    preds = {job: set() for job in jobs}
    for earlier, later in prec:
        preds[later].add(earlier)
    found = []

    def rec(done):
        if len(done) == len(jobs):
            found.append(done)
            return
        for job in jobs:
            if job in done or not preds[job].issubset(done):
                continue
            rec(done + (job,))

    rec(())
    return found


def _scheduled_makespan(order, spec):
    preds = {job: set() for job in spec["jobs"]}
    for earlier, later in spec["precedences"]:
        preds[later].add(earlier)
    finish = {}
    busy = []
    for job in order:
        earliest = max((finish[p] for p in preds[job]), default=0)
        duration = spec["duration"][job]
        need = spec["resource"][job]
        start = earliest
        while True:
            if all(
                sum(amount for s, e, amount in busy if s <= tau < e) + need <= spec["capacity"]
                for tau in range(start, start + duration)
            ):
                busy.append((start, start + duration, need))
                finish[job] = start + duration
                break
            start += 1
            if start > 50:
                raise AssertionError(order)
    return max(finish.values())


def test_rcpsp_feasible_orders_match_expected(solved):
    spec = solved["rcpsp"][0].meta["rcpsp"]
    orders = _precedence_orders(spec["jobs"], spec["precedences"])
    assert set(orders) == {("j1", "j2", "j3", "j4"), ("j1", "j3", "j2", "j4")}
    spans = [_scheduled_makespan(order, spec) for order in orders]
    assert min(spans) == EXPECTED["rcpsp"] == 4
    assert_close(solved["rcpsp"][1].objective, min(spans))


def test_refinery_pims_mps_roundtrip(tmp_path, solved):
    prob, sol = solved["refinery_pims"]
    path = tmp_path / "refinery_pims.mps"
    write_mps(prob, path)
    again = solve(read_mps(path), engine="simplex", polish=False, time_limit=15)
    assert again.verdict == "optimal", again.summary()
    assert_close(again.objective, sol.objective)
    assert_close(again.objective, EXPECTED["refinery_pims"])
