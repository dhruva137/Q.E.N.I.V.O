"""Ordering, symbolic factor, numeric LDL' and a hand-solved LP for ipm-native."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from qenivo.certify.kkt import kkt_residuals
from qenivo.engines.native_ipm import (
    factor_solve,
    order_matrix,
    solve_ipm_native,
    symbolic_matrix,
)
from qenivo.engines.registry import get_engine
from qenivo.model import ModelBuilder, Problem


def _csc(A):
    C = sp.csc_matrix(A, dtype=float)
    C.sum_duplicates()
    return C


def _star(n=8):
    rows, cols, data = [], [], []
    for i in range(n):
        rows.append(i)
        cols.append(i)
        data.append(1.0)
    for i in range(1, n):
        rows.extend([0, i])
        cols.extend([i, 0])
        data.extend([1.0, 1.0])
    return _csc(sp.csc_matrix((data, (rows, cols)), shape=(n, n)))


def _spd(n, seed):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n))
    mask = rng.random((n, n)) < 0.15
    B = np.tril(B * mask, -1)
    A = B @ B.T + np.eye(n)
    return _csc(A)


def _kkt(q, m, seed):
    rng = np.random.default_rng(seed)
    C = (rng.random((m, q)) < 0.4) * rng.normal(scale=0.2, size=(m, q))
    n = q + m
    A = np.zeros((n, n))
    A[:q, :q] = 2.0 * np.eye(q)
    A[q:, :q] = C
    A[:q, q:] = C.T
    A[q:, q:] = -0.5 * np.eye(m)
    return _csc(A)


def _rel(A, x, b):
    return float(np.linalg.norm(A @ x - b) / np.linalg.norm(b))


def test_amd_eliminates_the_star_centre_last():
    A = _star()
    for impl in ("auto", "python"):
        perm = order_matrix(A.shape[0], A.indptr, A.indices, "amd", impl=impl)
        assert sorted(perm.tolist()) == list(range(A.shape[0]))
        assert int(perm[-1]) == 0


def test_nested_dissection_is_a_permutation():
    A = _spd(18, 3)
    for impl in ("auto", "python"):
        perm = order_matrix(A.shape[0], A.indptr, A.indices, "nd", impl=impl)
        assert sorted(perm.tolist()) == list(range(A.shape[0]))


def test_symbolic_diagonal_and_column_counts():
    A = _csc(sp.eye(6))
    sym = symbolic_matrix(6, A.indptr, A.indices, np.arange(6, dtype=np.int32))
    assert np.all(sym["parent"] == -1)
    assert np.all(sym["colcount"] == 1)
    assert sym["supernodes"] == 6
    chain = _csc(
        sp.diags(
            [np.ones(5), 2 * np.ones(6), np.ones(5)], offsets=[-1, 0, 1], format="csc"
        )
    )
    perm = order_matrix(6, chain.indptr, chain.indices, "amd")
    sym = symbolic_matrix(6, chain.indptr, chain.indices, perm)
    assert np.all(sym["colcount"] >= 1)
    assert np.all((sym["parent"] == -1) | (sym["parent"] > np.arange(6)))


def test_spd_and_kkt_residual():
    spd = _spd(22, 4)
    kkt = _kkt(8, 4, 5)
    rng = np.random.default_rng(6)
    for impl in ("auto", "python"):
        for method in ("amd", "nd"):
            b = rng.normal(size=spd.shape[0])
            x = factor_solve(
                spd.shape[0],
                spd.indptr,
                spd.indices,
                spd.data,
                b,
                method=method,
                spd=True,
                impl=impl,
            )
            assert _rel(spd, x, b) <= 1e-12
        b = rng.normal(size=kkt.shape[0])
        x = factor_solve(
            kkt.shape[0],
            kkt.indptr,
            kkt.indices,
            kkt.data,
            b,
            method="amd",
            spd=False,
            impl=impl,
        )
        assert _rel(kkt, x, b) <= 1e-12


def test_hand_solved_lp():
    # min 2x + y s.t. x + y = 1, x, y >= 0. Unique optimum (0, 1), value 1.
    built = ModelBuilder("hand")
    built.var("x", 0.0, np.inf, obj=2.0)
    built.var("y", 0.0, np.inf, obj=1.0)
    built.row("eq", {"x": 1.0, "y": 1.0}, 1.0, 1.0)
    prob = built.build()
    for impl in ("auto", "python"):
        out = solve_ipm_native(prob, tol=1e-8, impl=impl)
        assert out["status"] == "optimal"
        assert abs(out["x"][0]) <= 1e-5
        assert abs(out["x"][1] - 1.0) <= 1e-5
        assert kkt_residuals(prob, out["x"], out["y"])["max_rel"] <= 1e-7


def test_diagonal_and_offdiagonal_qp():
    A = sp.csr_matrix([[1.0, 1.0]])
    bounds = dict(
        lc=np.array([1.0]),
        uc=np.array([1.0]),
        lx=np.array([-10.0, -10.0]),
        ux=np.array([10.0, 10.0]),
    )
    diag = Problem(c=np.zeros(2), A=A, Q=sp.eye(2, format="csr"), **bounds)
    out = solve_ipm_native(diag, tol=1e-8)
    assert out["status"] == "optimal"
    assert np.allclose(out["x"], [0.5, 0.5], atol=1e-5)
    Q = sp.csr_matrix([[2.0, 0.5], [0.5, 2.0]])
    full = Problem(c=np.zeros(2), A=A, Q=Q, **bounds)
    out = solve_ipm_native(full, tol=1e-8)
    assert out["status"] == "optimal"
    assert np.allclose(out["x"], [0.5, 0.5], atol=1e-5)
    assert abs(full.objective(out["x"]) - 0.625) <= 1e-5


def test_homogeneous_optimum_and_infeasible_row():
    built = ModelBuilder("hsd")
    built.var("x", 0.0, np.inf, obj=2.0)
    built.var("y", 0.0, np.inf, obj=1.0)
    built.row("eq", {"x": 1.0, "y": 1.0}, 1.0, 1.0)
    good = solve_ipm_native(built.build(), tol=1e-7, homogeneous=True)
    assert good["status"] == "optimal"
    assert abs(good["x"][1] - 1.0) <= 1e-4
    bad = ModelBuilder("empty")
    bad.var("x", 0.0, np.inf, obj=1.0)
    bad.row("neg", {"x": 1.0}, hi=-1.0)
    out = solve_ipm_native(bad.build(), tol=1e-7, homogeneous=True, max_iter=80)
    assert out["status"] == "infeasible"


def test_engine_is_registered():
    spec = get_engine("ipm-native")
    assert spec is not None
    assert "LP" in spec.classes and "QP" in spec.classes
    from qenivo import solve

    built = ModelBuilder("public")
    built.var("x", 0.0, np.inf, obj=2.0)
    built.var("y", 0.0, np.inf, obj=1.0)
    built.row("eq", {"x": 1.0, "y": 1.0}, 1.0, 1.0)
    sol = solve(built.build(), engine="ipm-native", tol=1e-7, backend="numpy", polish=False)
    assert sol.verdict == "optimal"
    assert abs(sol.objective - 1.0) <= 1e-5
