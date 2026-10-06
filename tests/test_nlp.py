"""Derivatives, the filter interior-point method, Haverly pooling, and outer approximation.

A wrong derivative or a wrong published optimum fails one of these tests.
Haverly profits 400, 600 and 750 are the maxima in Adhya, Tawarmalani and
Sahinidis, Industrial & Engineering Chemistry Research 38 (1999); this code
minimises cost minus revenue, so the optimal values are the negations.
"""
from __future__ import annotations

import numpy as np
import pytest

from qenivo.engines.nlp_global import solve_nlp_global  # noqa: F401  (registers the engine)
from qenivo.engines.nlp_ipm import solve_nlp_ipm  # noqa: F401
from qenivo.engines.registry import get_engine
from qenivo.nlp.ad import gradient, hessian, hvp, sparsity
from qenivo.nlp.dense_kkt import factor
from qenivo.nlp.expr import Graph, cos, evaluate, exp, log, sin, sqrt
from qenivo.nlp.ipm import solve_ipm
from qenivo.nlp.kkt import kkt_residual
from qenivo.nlp.oa import convex_demo, outer_approximation
from qenivo.nlp.pooling import (
    bilinear_feasible,
    minimised_objective,
    profit_at_quality,
    reference_point,
    solve_haverly,
)
from qenivo.nlp.problem import NLPModel


def test_analytic_derivatives():
    g = Graph()
    x = g.var("x")
    cube = x ** 3
    assert gradient(cube, np.array([2.0]))[0] == pytest.approx(12.0)
    H, _ = hessian(cube, np.array([2.0]))
    assert H[0, 0] == pytest.approx(12.0)

    g = Graph()
    x = g.var("x")
    s = sin(x)
    assert gradient(s, np.array([0.5]))[0] == pytest.approx(np.cos(0.5))

    g = Graph()
    x = g.var("x", 0.1, 10.0)
    assert gradient(log(x), np.array([2.0]))[0] == pytest.approx(0.5)

    g = Graph()
    x, y = g.var("x"), g.var("y")
    quot = x / y
    gr = gradient(quot, np.array([3.0, 4.0]))
    assert gr[0] == pytest.approx(0.25)
    assert gr[1] == pytest.approx(-3.0 / 16.0)

    g = Graph()
    x, y = g.var("x"), g.var("y")
    quad = (x ** 2) * y
    point = np.array([2.0, 3.0])
    gr = gradient(quad, point)
    assert gr[0] == pytest.approx(12.0)
    assert gr[1] == pytest.approx(4.0)
    H, _ = hessian(quad, point)
    assert H[0, 0] == pytest.approx(6.0)
    assert H[1, 1] == pytest.approx(0.0)
    assert H[0, 1] == pytest.approx(4.0)
    assert (0, 1) in sparsity(quad)


def test_gradient_matches_central_differences():
    g = Graph()
    x = g.var("x", 0.2, 5.0)
    y = g.var("y", 0.2, 5.0)
    f = sin(x) * exp(y) + (x ** 2) * y + log(x) + sqrt(y) + x / y + cos(x) - y ** 3
    p = np.array([1.3, 0.8])
    got = gradient(f, p)
    eps = 1e-6
    fd = np.zeros(2)
    for i in range(2):
        step = np.zeros(2)
        step[i] = eps
        fd[i] = (evaluate(f, p + step) - evaluate(f, p - step)) / (2.0 * eps)
    assert np.max(np.abs(got - fd)) < 1e-5
    direction = np.array([0.4, -0.2])
    hv = hvp(f, p, direction)
    fd_h = (gradient(f, p + eps * direction) - gradient(f, p - eps * direction)) / (2.0 * eps)
    assert np.max(np.abs(hv - fd_h)) < 1e-4


def test_sparse_hessian_uses_fewer_products_than_variables():
    g = Graph()
    x0, x1, x2, x3 = (g.var(f"x{i}") for i in range(4))
    f = (x0 + x1) ** 2 + x2 ** 2 + sin(x3)
    point = np.array([0.3, -0.4, 1.1, 0.7])
    H, info = hessian(f, point)
    assert info["n_products"] < 4
    assert H[0, 0] == pytest.approx(2.0)
    assert H[1, 1] == pytest.approx(2.0)
    assert H[0, 1] == pytest.approx(2.0)
    assert H[2, 2] == pytest.approx(2.0)
    assert H[3, 3] == pytest.approx(-np.sin(0.7))
    assert H[0, 2] == pytest.approx(0.0)
    v = np.array([1.0, -1.0, 0.5, 0.25])
    assert np.linalg.norm(H @ v - hvp(f, point, v)) < 1e-8


def test_ldl_inertia_and_kkt_solve():
    indefinite = factor(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert indefinite.inertia == (1, 1, 0)
    rng = np.random.default_rng(0)
    base = rng.normal(size=(5, 5))
    spd = base.T @ base + np.eye(5)
    rhs = rng.normal(size=5)
    fac = factor(spd)
    sol = fac.solve(rhs)
    assert np.linalg.norm(spd @ sol - rhs) < 1e-8
    assert fac.inertia == (5, 0, 0)

    n, m = 4, 2
    raw = rng.normal(size=(n, n))
    hess = raw.T @ raw + np.eye(n)
    jac = rng.normal(size=(m, n))
    kkt = np.zeros((n + m, n + m))
    kkt[:n, :n] = hess
    kkt[:n, n:] = jac.T
    kkt[n:, :n] = jac
    fac = factor(kkt)
    signs = np.sign(np.linalg.eigvalsh(0.5 * (kkt + kkt.T)))
    npos = int(np.sum(signs > 0))
    nneg = int(np.sum(signs < 0))
    assert fac.inertia == (npos, nneg, (n + m) - npos - nneg)
    rhs = rng.normal(size=n + m)
    sol = fac.solve(rhs)
    assert np.linalg.norm(kkt @ sol - rhs) < 1e-7


def test_ipm_unconstrained_equality_and_active_bound():
    g = Graph()
    x, y = g.var("x"), g.var("y")
    free = NLPModel(g, (x - 2.0) ** 2 + (y + 1.0) ** 2, name="unconstrained")
    wrong = kkt_residual(free, np.zeros(2), None, None, None)
    assert wrong["stationarity"] > 1.0
    sol = solve_ipm(free, tol=1e-8)
    assert sol.status == "optimal"
    assert sol.x[0] == pytest.approx(2.0, abs=1e-6)
    assert sol.x[1] == pytest.approx(-1.0, abs=1e-6)
    assert sol.residuals["stationarity"] < 1e-6
    assert sol.residuals["max_res"] < 1e-6

    g = Graph()
    x, y = g.var("x"), g.var("y")
    eq = NLPModel(g, (x - 1.0) ** 2 + (y - 1.0) ** 2, equalities=[x + y - 1.0])
    sol = solve_ipm(eq, tol=1e-8)
    assert sol.status == "optimal"
    assert sol.objective == pytest.approx(0.5, abs=1e-6)
    assert sol.x[0] == pytest.approx(0.5, abs=1e-5)
    assert sol.residuals["max_res"] < 1e-6

    g = Graph()
    x = g.var("x", ub=1.0)
    bound = NLPModel(g, (x - 3.0) ** 2)
    sol = solve_ipm(bound, tol=1e-7)
    assert sol.status == "optimal"
    assert sol.x[0] == pytest.approx(1.0, abs=1e-5)
    assert sol.objective == pytest.approx(4.0, abs=1e-5)
    assert sol.residuals["max_res"] < 1e-5


def test_haverly_reference_points_and_fixed_quality():
    for instance, profit in ((1, 400.0), (2, 600.0), (3, 750.0)):
        x = reference_point(instance)
        assert minimised_objective(instance, x) == pytest.approx(-profit)
        assert bilinear_feasible(instance, x)
        assert x[7] == pytest.approx(x[6] * x[2])
        assert x[8] == pytest.approx(x[6] * x[3])
    assert profit_at_quality(1, 1.0) == pytest.approx(-400.0, abs=1e-6)
    assert profit_at_quality(2, 3.0) == pytest.approx(-600.0, abs=1e-6)
    assert profit_at_quality(3, 1.5) == pytest.approx(-750.0, abs=1e-6)


@pytest.mark.parametrize("instance,profit", [(1, 400.0), (2, 600.0), (3, 750.0)])
def test_haverly_branch_and_bound(instance, profit):
    result = solve_haverly(instance, tol=1e-4, time_limit=45.0)
    assert result.status == "optimal"
    assert result.profit == pytest.approx(profit, abs=1e-3)
    assert result.objective == pytest.approx(-profit, abs=1e-3)
    assert result.lower_bound == pytest.approx(-profit, abs=1e-3)
    assert result.root_relaxation <= -profit + 1e-3
    if result.residuals.get("root_kkt") is not None:
        assert result.residuals["root_kkt"] <= 1e-4


def test_outer_approximation_matches_enumeration():
    enumerated = min(3.0 * (y - 1.0) ** 2 for y in (0, 1, 2))
    assert enumerated == 0.0
    result = outer_approximation(convex_demo(), tol=1e-4, time_limit=30.0)
    assert result.nlp_solves >= 1
    assert result.master_solves >= 1
    assert result.status == "optimal"
    assert result.objective == pytest.approx(0.0, abs=1e-3)
    assert result.objective < 2.0
    assert result.lower_bound == pytest.approx(0.0, abs=1e-3)
    assert result.lower_bound <= result.objective + 1e-6


def test_nlp_engines_are_registered():
    assert get_engine("nlp-ipm") is not None
    assert get_engine("nlp-global") is not None
    assert solve_nlp_global is not None
    assert solve_nlp_ipm is not None
