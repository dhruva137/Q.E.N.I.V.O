"""Planner workloads: case stacks, breakeven, recursion with certified bounds, explain, ranging."""
import numpy as np
import pytest

from conftest import infeasible_lp
from qenivo import solve
from qenivo.models.refinery import refinery_lp
from qenivo.models.williams import williams_lp
from qenivo.workload.cases import Case, crude_breakeven, solve_cases
from qenivo.workload.explain import explain_infeasibility
from qenivo.workload.ranging import cost_ranging, rhs_ranging
from qenivo.workload.recursion import (
    HAVERLY_OPTIMA,
    certified_recursion,
    haverly,
    pooling_bound,
    refinery_pools,
    run_recursion,
)


def test_case_stack_all_certified_and_consistent():
    p = williams_lp()
    cases = [Case("base"), Case("pmf+1", cost={"PMF": 8.0}), Case("reform70", row_hi={"reform_cap": 7000})]
    rep = solve_cases(p, cases, tol=1e-8, backend="numpy")
    assert all(s.verdict == "optimal" for s in rep.cases)
    # a better price can only raise a maximised profit
    assert rep.cases[1].objective >= rep.cases[0].objective - 1e-6
    # each case equals a stand-alone solve of the changed model
    from qenivo.workload.cases import case_problem
    alone = solve(case_problem(p, cases[1]), engine="simplex", tol=1e-10)
    assert abs(alone.objective - rep.cases[1].objective) <= 1e-6 * abs(alone.objective)


def test_batched_pdhg_case_stack_matches_sequential():
    p = refinery_lp(1, 2, 3, 3, seed=1, names=True)
    rn, cn = p.names()
    cases = [Case(f"s{k}", cost={cn[0]: float(p.c[0] * (1 + 0.1 * k))}) for k in range(4)]
    seq = solve_cases(p, cases, tol=1e-6, backend="numpy")
    bat = solve_cases(p, cases, tol=1e-6, engine="pdhg", backend="numpy")
    for a, b in zip(seq.cases, bat.cases):
        assert a.verdict == b.verdict == "optimal"
        assert abs(a.objective - b.objective) <= 1e-4 * (1 + abs(a.objective))


def test_crude_breakeven_monotone():
    p = refinery_lp(1, 1, 2, 2, seed=3, names=True)
    j = 0                                   # first crude purchase column of block 0
    col = p.names()[1][j]
    base = p.c[j]
    out = crude_breakeven(p, col, [base * f for f in (0.5, 1.0, 2.0, 4.0, 8.0)], backend="numpy")
    assert out["all_certified"]
    vols = out["volume"]
    assert all(vols[i] + 1e-6 >= vols[i + 1] for i in range(len(vols) - 1))   # dearer crude, no more bought


@pytest.mark.parametrize("inst", [1, 2, 3])
def test_mccormick_bounds_match_literature(inst):
    prob, pools = haverly(inst)
    bound, sol = pooling_bound(prob, pools)
    assert sol.verdict == "optimal"
    assert abs(bound - {1: 500.0, 2: 1000.0, 3: 800.0}[inst]) <= 1e-6 * bound
    assert bound >= HAVERLY_OPTIMA[inst] - 1e-9       # a valid upper bound on the global optimum


def test_recursion_reaches_haverly2_global_optimum():
    prob, pools = haverly(2)
    for m in ("simplex", "ipm", "pdhg_warm", "simplex_step"):
        r = run_recursion(prob, pools, method=m, q0=1.0)
        assert r.converged, m
        assert abs(r.solution.objective - 600.0) <= 1e-2, (m, r.solution.objective)


def test_certified_recursion_reports_gap():
    prob, pools = haverly(1)
    out = certified_recursion(prob, pools, starts=(1.0, 3.0))
    assert out["bound_certified"]
    assert out["best_objective"] <= out["bound"] + 1e-6
    assert 0 <= out["gap"] <= 1


def test_recursion_on_pooled_refinery_converges():
    p = refinery_lp(1, 2, 3, 3, seed=0, names=True, pooled=True)
    pools = refinery_pools(p)
    r = run_recursion(p, pools, method="pdhg_warm", q0=80.0, tol=1e-6, q_tol=1e-4, max_passes=80)
    assert r.converged


def test_explain_names_the_conflict():
    rep = explain_infeasibility(infeasible_lp())
    assert rep["feasible"] is False
    assert abs(rep["total_relaxation"] - 12.0) <= 1e-6       # 25 needed, 3 + 10 possible
    assert rep["farkas_check"]["valid"]
    assert {"need", "cap_x"} <= set(rep["conflict_rows"])


def test_explain_feasible_model():
    assert explain_infeasibility(williams_lp())["feasible"] is True


def test_cost_ranging_is_exact():
    p = williams_lp()
    sol = solve(p, engine="simplex", tol=1e-10)
    rng = {d["column"]: d for d in cost_ranging(p, sol)}
    d = rng["PMF"]
    _, cn = p.names()
    j = cn.index("PMF")
    for coef in (d["cost_min"] + 1e-3 if np.isfinite(d["cost_min"]) else d["cost"] - 1.0,
                 d["cost_max"] - 1e-3 if np.isfinite(d["cost_max"]) else d["cost"] + 1.0):
        q = p.copy(); q.c[j] = p.obj_sign * coef
        s2 = solve(q, engine="simplex", tol=1e-10)
        # inside the range the old plan is still optimal (the engine may return an alternative optimum)
        assert abs(q.objective(sol.x) - s2.objective) <= 1e-7 * (1 + abs(s2.objective)), coef
    # just outside the upper end the old BASIS is no longer optimal: the plan changes, or (the model is
    # degenerate, so one plan can have several optimal bases) the optimum is reached by another basis
    if np.isfinite(d["cost_max"]):
        q = p.copy(); q.c[j] = p.obj_sign * (d["cost_max"] + 0.05)
        s3 = solve(q, engine="simplex", tol=1e-10)
        assert s3.objective > q.objective(sol.x) + 1e-6 or s3.extra["basis"] != sol.extra["basis"]


def test_rhs_ranging_marginals_hold():
    p = williams_lp()
    sol = solve(p, engine="simplex", tol=1e-10)
    rr = [d for d in rhs_ranging(p, sol) if d["binding"] and abs(d["marginal"]) > 1e-9]
    assert rr
    d = rr[0]
    rn, _ = p.names()
    i = rn.index(d["row"])
    step = 0.25 * min(d["limit_max"] - d["limit"], 100.0)
    if step <= 0:
        pytest.skip("no room to move this limit")
    q = p.copy()
    if abs(p.uc[i] - d["limit"]) < 1e-9:
        q.uc[i] += step
        if abs(p.lc[i] - d["limit"]) < 1e-9:
            q.lc[i] += step
    else:
        q.lc[i] += step
    s2 = solve(q, engine="simplex", tol=1e-10)
    assert abs((s2.objective - sol.objective) - d["marginal"] * step) <= 1e-6 * (1 + abs(sol.objective))
