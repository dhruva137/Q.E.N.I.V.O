"""Planted models and the verdict check used by the E6 suites.

A feasible point and multipliers are built so the KKT residuals in
``qenivo.certify.kkt`` are zero by algebra (complementary slackness as in
Vanderbei, Linear Programming, 4th ed.). The solver is then required to
return that verdict, or an honest ``not_proven``. A proven answer with the
wrong status or the wrong value fails the test.

Original construction. It is not taken from another solver's test suite.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from qenivo import solve
from qenivo.certify.kkt import kkt_residuals
from qenivo.model import ModelBuilder, Problem

LP_ENGINES = ("simplex", "ipm", "pdhg")


def _problem(c, A, lc, uc, lx, ux, name, Q=None, integer=None, c0=0.0) -> Problem:
    return Problem(c=np.asarray(c, dtype=np.float64), A=sp.csr_matrix(A, dtype=np.float64),
                   lc=np.asarray(lc, dtype=np.float64), uc=np.asarray(uc, dtype=np.float64),
                   lx=np.asarray(lx, dtype=np.float64), ux=np.asarray(ux, dtype=np.float64),
                   c0=float(c0), name=name, Q=None if Q is None else sp.csr_matrix(Q, dtype=np.float64),
                   integer=integer)


def planted_lp(rng: np.random.Generator, m: int, n: int) -> tuple[Problem, float]:
    """Equality-form LP whose unique optimum is known: basic x > 0, nonbasic x = 0, lam > 0."""
    if m < 1 or n <= m:
        raise ValueError("need 1 <= m < n")
    A = rng.normal(scale=0.25, size=(m, n))
    basis = np.sort(rng.choice(n, size=m, replace=False))
    for i, j in enumerate(basis):
        A[i, j] += 1.5
    if np.linalg.matrix_rank(A[:, basis], tol=1e-8) < m:
        A[:, basis] = A[:, basis] + np.eye(m)
    x = np.zeros(n)
    x[basis] = rng.uniform(0.4, 1.6, size=m)
    y = rng.normal(scale=0.4, size=m)
    lam = np.zeros(n)
    nonbasic = np.ones(n, dtype=bool)
    nonbasic[basis] = False
    lam[nonbasic] = rng.uniform(0.3, 1.4, size=int(nonbasic.sum()))
    c = lam + A.T @ y
    b = A @ x
    prob = _problem(c, A, b, b.copy(), np.zeros(n), np.full(n, np.inf), "planted-lp")
    _require_kkt(prob, x, y)
    return prob, float(prob.objective(x))


def planted_qp(rng: np.random.Generator, n: int = 3) -> tuple[Problem, float]:
    """Strictly feasible convex QP: diagonal Q, one equality, multipliers chosen so lam = 0."""
    if n < 1:
        raise ValueError("n >= 1")
    x = rng.uniform(0.5, 1.5, size=n)
    diag = rng.uniform(0.5, 2.0, size=n)
    Q = sp.diags(diag, format="csr")
    a = rng.normal(size=n)
    norm = np.linalg.norm(a)
    a = np.ones(n) if norm < 1e-8 else a / norm
    y = np.array([float(rng.normal(scale=0.3))])
    c = a * y[0] - diag * x
    A = sp.csr_matrix(a.reshape(1, n))
    b = np.asarray(A @ x).reshape(-1)
    prob = _problem(c, A, b, b.copy(), x - 1.0, x + 1.0, "planted-qp", Q=Q)
    _require_kkt(prob, x, y)
    return prob, float(prob.objective(x))


def planted_milp(rng: np.random.Generator, m: int, n: int) -> tuple[Problem, float]:
    """Same planted LP with an integral basic solution, so the MILP optimum equals the LP optimum."""
    if m < 1 or n <= m:
        raise ValueError("need 1 <= m < n")
    A = rng.normal(scale=0.2, size=(m, n))
    basis = np.sort(rng.choice(n, size=m, replace=False))
    for i, j in enumerate(basis):
        A[i, j] += 2.0
    x = np.zeros(n)
    x[basis] = rng.integers(1, 3, size=m).astype(np.float64)
    y = rng.normal(scale=0.3, size=m)
    lam = np.zeros(n)
    nonbasic = np.ones(n, dtype=bool)
    nonbasic[basis] = False
    lam[nonbasic] = rng.uniform(0.4, 1.2, size=int(nonbasic.sum()))
    c = lam + A.T @ y
    b = A @ x
    integer = np.ones(n, dtype=bool)
    prob = _problem(c, A, b, b.copy(), np.zeros(n), np.full(n, np.inf), "planted-milp", integer=integer)
    _require_kkt(prob, x, y)
    return prob, float(prob.objective(x))


def infeasible_box(rng: np.random.Generator, n: int) -> Problem:
    """Variables in [0, 1] forced to sum to n + 1. Infeasible by inspection."""
    n = max(int(n), 1)
    b = ModelBuilder("infeasible-box")
    names = [b.var(f"x{j}", 0.0, 1.0, obj=float(rng.normal())) for j in range(n)]
    b.row("too_much", {name: 1.0 for name in names}, lo=n + 1.0, hi=n + 1.0)
    return b.build()


def unbounded_ray(rng: np.random.Generator) -> Problem:
    """min -x with x >= 0 and no upper bound. Improving ray d = e_x."""
    b = ModelBuilder("unbounded-ray")
    b.var("x", 0.0, np.inf, obj=-1.0)
    extras = int(rng.integers(0, 3))
    coefs = {"x": 1.0}
    for j in range(extras):
        name = b.var(f"y{j}", 0.0, 1.0, obj=float(rng.uniform(-0.5, 0.5)))
        coefs[name] = float(rng.uniform(-0.2, 0.2))
    b.row("does_not_cap_x", coefs, lo=-np.inf, hi=np.inf)
    return b.build()


def permute_lp(prob: Problem, rng: np.random.Generator) -> Problem:
    row = rng.permutation(prob.m)
    col = rng.permutation(prob.n)
    A = prob.A[row][:, col]
    return _problem(prob.c[col], A, prob.lc[row], prob.uc[row], prob.lx[col], prob.ux[col],
                    prob.name + "-perm", c0=prob.c0)


def scale_row(prob: Problem, i: int, alpha: float) -> Problem:
    A = prob.A.tolil()
    A[i, :] = np.asarray(A[i, :].todense()).ravel() * alpha
    lc, uc = prob.lc.copy(), prob.uc.copy()
    lc[i] *= alpha
    uc[i] *= alpha
    return _problem(prob.c, A, lc, uc, prob.lx, prob.ux, prob.name + "-scale", c0=prob.c0)


def flip_column(prob: Problem, j: int) -> Problem:
    """Replace x_j by -z_j. The optimal value is unchanged."""
    A = prob.A.tolil()
    A[:, j] = np.asarray(A[:, j].todense()).ravel() * -1.0
    c = prob.c.copy()
    c[j] *= -1.0
    lx, ux = prob.lx.copy(), prob.ux.copy()
    lx[j], ux[j] = -prob.ux[j], -prob.lx[j]
    return _problem(c, A, prob.lc, prob.uc, lx, ux, prob.name + "-flip", c0=prob.c0)


def duplicate_row(prob: Problem) -> Problem:
    A = sp.vstack([prob.A, prob.A.getrow(0)], format="csr")
    return _problem(prob.c, A, np.append(prob.lc, prob.lc[0]), np.append(prob.uc, prob.uc[0]),
                    prob.lx, prob.ux, prob.name + "-dup", c0=prob.c0)


def shift_constant(prob: Problem, c0: float) -> tuple[Problem, float]:
    """Adding c0 to a minimisation model shifts the optimal value by c0."""
    q = prob.copy()
    q.c0 = prob.c0 + c0
    q.name = prob.name + "-c0"
    return q, c0


def solve_checked(prob: Problem, engine: str, kind: str, value: float | None = None,
                  tol: float = 1e-5, time_limit: float = 2.0):
    """``kind`` is optimal, infeasible or unbounded. ``not_proven`` is always acceptable."""
    sol = solve(prob, engine=engine, tol=tol, backend="numpy", time_limit=time_limit, polish=False)
    allowed = {"optimal": ("optimal", "not_proven"),
               "infeasible": ("infeasible", "not_proven"),
               "unbounded": ("unbounded", "not_proven")}[kind]
    assert sol.verdict in allowed, sol.summary()
    if kind == "optimal" and sol.verdict == "optimal":
        if value is None:
            raise AssertionError("optimal check needs the planted value")
        assert abs(sol.objective - value) <= tol * (1.0 + abs(value)), sol.summary()
    if kind == "infeasible" and sol.verdict == "infeasible":
        assert sol.ray_check and sol.ray_check.get("valid"), sol.ray_check
    return sol


def _require_kkt(prob: Problem, x: np.ndarray, y: np.ndarray) -> None:
    k = kkt_residuals(prob, x, y)
    if k["max_rel"] > 1e-8:
        raise RuntimeError(f"planted point is not optimal: {k}")
