"""The extension point: plugin engines through the registry, and the MIQP plugin."""
import dataclasses
import itertools

import numpy as np
import pytest
import scipy.sparse as sp

import qenivo
from qenivo import ModelBuilder, register_engine, solve
from qenivo.engines.pdqp import solve_qp
from qenivo.models.williams import williams_lp


def _integer_qp():
    mb = ModelBuilder("iqp")
    mb.var("x", 0, 5, -5.2, integer=True)
    mb.var("y", 0, 5, -2.8, integer=True)
    mb.row("r", {"x": 1, "y": 1}, hi=10)
    p = mb.build()
    return dataclasses.replace(p, Q=sp.identity(2, format="csr") * 2.0, c0=8.72)


def test_miqp_is_routed_to_the_plugin_and_solved():
    s = solve(_integer_qp(), time_limit=60)
    assert s.engine["engine"] == "miqp"
    assert s.verdict == "optimal"
    assert np.allclose(s.x, [3, 1])
    assert abs(s.objective - 0.32) <= 1e-6


def _portfolio(n=6, k=3, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, 2))
    cov = F @ F.T * 0.02 + np.diag(rng.uniform(0.01, 0.04, n))
    mu = rng.uniform(0.02, 0.12, n)
    mb = ModelBuilder("portfolio")
    for i in range(n):
        mb.var(f"w{i}", 0, 1, -mu[i])
        mb.var(f"z{i}", 0, 1, 0.0, integer=True)
        mb.row(f"link{i}", {f"w{i}": 1, f"z{i}": -1}, hi=0)
    mb.row("budget", {f"w{i}": 1 for i in range(n)}, 1, 1)
    mb.row("card", {f"z{i}": 1 for i in range(n)}, hi=k)
    p = mb.build()
    Q = np.zeros((2 * n, 2 * n))
    Q[0::2, 0::2] = 2 * cov
    return dataclasses.replace(p, Q=sp.csr_matrix(Q)), n, k


def test_miqp_cardinality_portfolio_matches_enumeration():
    p, n, k = _portfolio()
    s = solve(p, time_limit=120)
    best = np.inf
    for sub in itertools.combinations(range(n), k):
        lx, ux = p.lx.copy(), p.ux.copy()
        for i in range(n):
            lx[2 * i + 1] = ux[2 * i + 1] = 1.0 if i in sub else 0.0
        r = solve_qp(dataclasses.replace(p, lx=lx, ux=ux, integer=None), tol=1e-9)
        x = r["x"]
        best = min(best, 0.5 * x @ (p.Q @ x) + p.c @ x)
    assert s.verdict == "optimal"
    assert abs(s.objective - best) <= 1e-6 * (1 + abs(best))


def test_register_a_third_party_engine():
    calls = []

    def my_engine(prob, tol, time_limit, backend, verbose, **kw):
        calls.append(prob.name)
        return solve(prob, engine="simplex", tol=tol)

    register_engine("my_simplex_wrapper", my_engine, classes=("LP",), description="test plugin")
    s = solve(williams_lp(), engine="my_simplex_wrapper")
    assert s.verdict == "optimal" and calls == [williams_lp().name]
    assert "my_simplex_wrapper" in qenivo.engines()


def test_unknown_engine_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown engine"):
        solve(williams_lp(), engine="does_not_exist")
