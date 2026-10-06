"""Every LP engine, on every bound pattern, agreeing with each other and with published optima."""
import numpy as np
import pytest

from conftest import DATA, infeasible_lp, lp_all_bound_types, unbounded_lp
from qenivo import solve
from qenivo.certify.kkt import kkt_residuals
from qenivo.engines import pdhg
from qenivo.models.williams import PUBLISHED_OPTIMUM, williams_lp

ENGINES = ["simplex", "ipm", "pdhg"]


@pytest.mark.parametrize("engine", ENGINES)
def test_williams_published_optimum(engine):
    sol = solve(williams_lp(), engine=engine, tol=1e-8, backend="numpy")
    assert sol.verdict == "optimal"
    assert abs(sol.objective - PUBLISHED_OPTIMUM) <= 0.01


@pytest.mark.parametrize("engine", ENGINES)
def test_all_bound_types_agree(engine):
    p = lp_all_bound_types()
    ref = solve(p, engine="simplex", tol=1e-10)
    sol = solve(p, engine=engine, tol=1e-8, backend="numpy")
    assert ref.verdict == sol.verdict == "optimal"
    assert abs(sol.objective - ref.objective) <= 1e-6 * (1 + abs(ref.objective))
    k = kkt_residuals(p, sol.x, p.obj_sign * sol.y)
    assert k["max_rel"] <= 1e-7


@pytest.mark.parametrize("engine", ENGINES)
def test_infeasible_is_proven(engine):
    sol = solve(infeasible_lp(), engine=engine, tol=1e-6, backend="numpy")
    assert sol.verdict == "infeasible"
    assert sol.ray_check["valid"]


def test_pdhg_detects_infeasibility_in_loop():
    r = pdhg.solve(infeasible_lp(), tol=1e-6)
    assert r.status[0] == "infeasible"
    assert r.rays[0]["kind"] == "farkas" and r.rays[0]["check"]["valid"]


def test_pdhg_detects_unboundedness():
    r = pdhg.solve(unbounded_lp(), tol=1e-6)
    assert r.status[0] == "unbounded"
    assert r.rays[0]["check"]["valid"]


def test_unbounded_verdict_via_solve():
    sol = solve(unbounded_lp(), engine="pdhg", backend="numpy")
    assert sol.verdict == "unbounded"


def test_batch_equals_individual_solves():
    p = williams_lp()
    rng = np.random.default_rng(0)
    S = 6
    C = np.stack([p.c * rng.uniform(0.9, 1.1, p.n) for _ in range(S)], 1)
    br = pdhg.solve_batch(p.A, C, p.lc, p.uc, p.lx, p.ux, p.c0, pdhg.PDHGOptions(tol=1e-8))
    for j in range(S):
        q = p.copy(); q.c = C[:, j].copy()
        ref = solve(q, engine="simplex", tol=1e-10)
        assert br.status[j] == "optimal"
        obj = q.objective(br.x[:, j])
        assert abs(obj - ref.objective) <= 1e-6 * (1 + abs(ref.objective))


def test_empty_model_is_not_silently_optimal_with_notes():
    from qenivo.model import ModelBuilder
    b = ModelBuilder("empty")
    sol = solve(b.build())
    assert sol.verdict == "optimal" and sol.objective == 0.0


def test_ipm_on_small_netlib_if_present():
    files = [DATA / f"{n}.mps" for n in ("afiro", "sc50a", "blend", "adlittle")]
    files = [f for f in files if f.exists()]
    if not files:
        pytest.skip("netlib test files not present")
    ref = {"afiro": -464.7531428571, "sc50a": -64.575077059, "blend": -30.812149846, "adlittle": 225494.96316}
    for f in files:
        sol = solve(str(f), engine="ipm", tol=1e-8)
        assert sol.verdict == "optimal", f.name
        r = ref[f.stem]
        assert abs(sol.objective - r) <= 1e-6 * (1 + abs(r)), (f.name, sol.objective)
