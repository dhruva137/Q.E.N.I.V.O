"""Float32 iterations with float64 refinement (engine pdhg-rf), and the float32 PDHG path."""
import dataclasses

import numpy as np
import pytest

from conftest import DATA, lp_all_bound_types
from qenivo import read, solve
from qenivo.certify.kkt import kkt_residuals
from qenivo.engines import pdhg, refine
from qenivo.engines.simplex import solve_simplex
from qenivo.models.refinery import LEVELS, refinery_lp
from qenivo.models.williams import PUBLISHED_OPTIMUM, williams_lp

# Netlib published optima (netlib.org/lp/data/readme)
NETLIB = {"afiro": -4.6475314286e+02, "blend": -3.0812149846e+01, "sc50a": -6.4575077059e+01,
          "adlittle": 2.2549496316e+05}


@pytest.mark.parametrize("name", sorted(NETLIB))
def test_refined_float32_reaches_published_optimum(name):
    p = read(DATA / f"{name}.mps")
    r = refine.solve(p, tol=1e-8, backend="numpy", time_limit=120)
    assert r.status[0] == "optimal"
    k = kkt_residuals(p, r.x[:, 0], r.y[:, 0])
    assert k["max_rel"] <= 1e-8
    assert abs(k["objective"] - NETLIB[name]) <= 1e-7 * (1 + abs(NETLIB[name]))


def test_refinement_beats_float64_pdhg_accuracy():
    p = read(DATA / "blend.mps")
    f64 = pdhg.solve(p, tol=1e-6)
    rf = refine.solve(p, tol=1e-9, time_limit=120)
    e64 = abs(kkt_residuals(p, f64.x[:, 0], f64.y[:, 0])["objective"] - NETLIB["blend"])
    erf = abs(kkt_residuals(p, rf.x[:, 0], rf.y[:, 0])["objective"] - NETLIB["blend"])
    assert erf < 1e-3 * e64


def test_float32_pdhg_stays_float32_and_converges_loosely():
    p = williams_lp()
    r = pdhg.solve(p, tol=1e-4, dtype="float32", detect_infeasibility=False)
    assert r.status[0] == "optimal"
    assert abs(p.obj_sign * (p.c @ r.x[:, 0] + p.c0) - PUBLISHED_OPTIMUM) <= 1e-3 * abs(PUBLISHED_OPTIMUM)


def test_refined_batch_certifies_every_case():
    p = refinery_lp(*LEVELS[1], seed=0)
    C = p.c[:, None] * np.random.default_rng(3).uniform(0.95, 1.05, (p.n, 6))
    r = refine.solve_refined(p.A, C, p.lc, p.uc, p.lx, p.ux, p.c0, tol=1e-7, time_limit=300)
    assert all(s == "optimal" for s in r.status)
    for j in range(C.shape[1]):
        q = dataclasses.replace(p, c=C[:, j])
        ref = solve_simplex(q)
        assert abs(C[:, j] @ r.x[:, j] - C[:, j] @ ref["x"]) <= 1e-6 * (1 + abs(C[:, j] @ ref["x"]))


def test_api_engine_pdhg_rf_all_bound_types():
    p = lp_all_bound_types()
    ref = solve(p, engine="simplex", tol=1e-10)
    sol = solve(p, engine="pdhg-rf", tol=1e-9, backend="numpy")
    assert sol.verdict == "optimal"
    assert abs(sol.objective - ref.objective) <= 1e-8 * (1 + abs(ref.objective))


def test_simplex_warm_start_after_price_change():
    p = refinery_lp(*LEVELS[1], seed=0)
    base = solve_simplex(p)
    q = dataclasses.replace(p, c=p.c * np.random.default_rng(1).uniform(0.97, 1.03, p.n))
    cold, warm = solve_simplex(q), solve_simplex(q, start=base)
    assert cold["status"] == warm["status"] == "optimal"
    assert abs(q.c @ cold["x"] - q.c @ warm["x"]) <= 1e-8 * (1 + abs(q.c @ cold["x"]))
    assert warm["iterations"] < cold["iterations"]
