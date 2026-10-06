"""The native C++ simplex agrees with the Python engine and the published optima."""
import dataclasses

import numpy as np
import pytest

from conftest import DATA, infeasible_lp, lp_all_bound_types
from qenivo import read
from qenivo.certify.kkt import kkt_residuals
from qenivo.engines import native_simplex as ns
from qenivo.engines.simplex import solve_simplex_python as solve_simplex
from qenivo.models.refinery import LEVELS, refinery_lp

pytestmark = pytest.mark.skipif(ns.library() is None, reason="native library unavailable: " + ns.status())

NETLIB = {"afiro": -4.6475314286e+02, "blend": -3.0812149846e+01, "sc50a": -6.4575077059e+01,
          "sc105": -5.2202061212e+01, "adlittle": 2.2549496316e+05, "kb2": -1.7499001299e+03}


@pytest.mark.parametrize("name", sorted(NETLIB))
def test_native_reaches_published_optimum(name):
    p = read(DATA / f"{name}.mps")
    r = ns.solve_simplex_native(p)
    assert r["status"] == "optimal"
    k = kkt_residuals(p, r["x"], r["y"])
    assert k["max_rel"] <= 1e-8
    assert abs(k["objective"] - NETLIB[name]) <= 1e-8 * (1 + abs(NETLIB[name]))


def test_native_all_bound_types_match_python():
    p = lp_all_bound_types()
    a, b = ns.solve_simplex_native(p), solve_simplex(p)
    assert a["status"] == b["status"] == "optimal"
    assert abs(p.c @ a["x"] - p.c @ b["x"]) <= 1e-9 * (1 + abs(p.c @ b["x"]))


def test_native_proves_infeasible():
    assert ns.solve_simplex_native(infeasible_lp())["status"] == "infeasible"


def test_native_warm_start_after_price_and_bound_change():
    p = refinery_lp(*LEVELS[1], seed=0)
    base = ns.solve_simplex_native(p)
    q = dataclasses.replace(p, c=p.c * np.random.default_rng(1).uniform(0.97, 1.03, p.n))
    cold, warm = ns.solve_simplex_native(q), ns.solve_simplex_native(q, start=base)
    assert cold["status"] == warm["status"] == "optimal"
    assert abs(q.c @ cold["x"] - q.c @ warm["x"]) <= 1e-8 * (1 + abs(q.c @ cold["x"]))
    assert warm["iterations"] < cold["iterations"]
    ux = p.ux.copy()
    j = int(np.argmax(base["x"]))
    ux[j] = 0.5 * base["x"][j]                   # a branching-style bound change: dual simplex
    r = dataclasses.replace(p, ux=ux)
    d = ns.solve_simplex_native(r, start=base)
    ref = solve_simplex(r)
    assert d["status"] == ref["status"]
    if ref["status"] == "optimal":
        assert abs(r.c @ d["x"] - r.c @ ref["x"]) <= 1e-8 * (1 + abs(r.c @ ref["x"]))
