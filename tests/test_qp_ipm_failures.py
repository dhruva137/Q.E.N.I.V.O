"""Regression hooks for Maros-style QP failure classes on ipm-native.

Covers free variables, extreme scaling, and a dense (non-diagonal) Q. Each case must
reach a KKT-certified optimal verdict through the public solve() path (default route).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from qenivo import Problem, solve
from qenivo.certify.kkt import kkt_residuals
from qenivo.engines.native_ipm import solve_ipm_native
from qenivo.workload.router import RoutingPolicy, route


def _cert_optimal(prob, sol, tol=1e-6):
    assert sol.verdict == "optimal", (sol.status, sol.verdict, sol.engine)
    assert sol.residuals and sol.residuals["max_rel"] <= tol * 1.000001
    k = kkt_residuals(prob, sol.x, prob.obj_sign * sol.y)
    assert k["max_rel"] <= tol * 1.000001
    return k


def test_default_route_is_ipm_native_for_small_qp():
    Q = sp.eye(3, format="csr")
    p = Problem(
        c=np.zeros(3),
        A=sp.csr_matrix([[1.0, 1.0, 1.0]]),
        lc=np.array([1.0]),
        uc=np.array([1.0]),
        lx=np.full(3, -10.0),
        ux=np.full(3, 10.0),
        Q=Q,
        name="route_qp",
    )
    r = route(p, 1, backend="numpy")
    assert r["engine"] == "ipm-native"
    # Tall + high nnz -> PDQP. Force both knobs.
    tall = Problem(
        c=np.zeros(3),
        A=sp.csr_matrix(np.eye(4)[:, :3]),
        lc=np.zeros(4),
        uc=np.zeros(4),
        lx=np.full(3, -10.0),
        ux=np.full(3, 10.0),
        Q=Q,
        name="tall_qp",
    )
    r2 = route(
        tall,
        1,
        backend="numpy",
        policy=RoutingPolicy(qp_ipm_nnz_max=1, ipm_rows_max=2),
    )
    assert r2["engine"] == "pdqp"
    # Wide (few rows, many cols) stays on IPM even when nnz is huge.
    wide = Problem(
        c=np.zeros(3),
        A=sp.csr_matrix([[1.0, 1.0, 1.0]]),
        lc=np.array([1.0]),
        uc=np.array([1.0]),
        lx=np.full(3, -10.0),
        ux=np.full(3, 10.0),
        Q=Q,
        name="wide_qp",
    )
    r3 = route(
        wide,
        1,
        backend="numpy",
        policy=RoutingPolicy(qp_ipm_nnz_max=1, ipm_rows_max=3000),
    )
    assert r3["engine"] == "ipm-native"


def test_free_variables_qp_kkt():
    # min 0.5 (x^2 + y^2)  s.t. x - y = 1; both free. Optimum (0.5, -0.5), value 0.25.
    Q = sp.eye(2, format="csr")
    p = Problem(
        c=np.zeros(2),
        A=sp.csr_matrix([[1.0, -1.0]]),
        lc=np.array([1.0]),
        uc=np.array([1.0]),
        lx=np.full(2, -np.inf),
        ux=np.full(2, np.inf),
        Q=Q,
        name="free_qp",
    )
    sol = solve(p, tol=1e-8, backend="numpy", time_limit=30.0)
    k = _cert_optimal(p, sol, tol=1e-8)
    assert abs(k["objective"] - 0.25) <= 1e-6
    assert np.allclose(sol.x, [0.5, -0.5], atol=1e-5)


def test_extreme_scaling_qp_kkt():
    # Same geometry as a unit QP but with huge / tiny row and objective scales.
    # min 0.5*(1e8 x^2 + 1e-8 y^2) s.t. 1e6 x + 1e-6 y = 1, bounds [-1,1].
    Q = sp.diags([1e8, 1e-8], format="csr")
    p = Problem(
        c=np.zeros(2),
        A=sp.csr_matrix([[1e6, 1e-6]]),
        lc=np.array([1.0]),
        uc=np.array([1.0]),
        lx=np.array([-1.0, -1.0]),
        ux=np.array([1.0, 1.0]),
        Q=Q,
        name="scale_qp",
    )
    sol = solve(p, tol=1e-6, backend="numpy", time_limit=30.0)
    k = _cert_optimal(p, sol, tol=1e-6)
    # Feasible point near x=1e-6, y≈0 when the equality is scaled; just require KKT + finite.
    assert np.isfinite(k["objective"])
    ax = float(np.asarray(p.A @ sol.x).reshape(-1)[0])
    assert abs(ax - 1.0) <= 1e-5 * (1.0 + abs(ax))


def test_dense_q_qp_kkt():
    # Dense SPD Q (all entries nonzero): exercises the Friedlander–Orban augmented system.
    rng = np.random.default_rng(7)
    B = rng.normal(size=(4, 4))
    Qd = B.T @ B + np.eye(4)
    Q = sp.csr_matrix(Qd)
    assert Q.nnz == 16
    A = sp.csr_matrix([[1.0, 1.0, 1.0, 1.0]])
    p = Problem(
        c=np.zeros(4),
        A=A,
        lc=np.array([1.0]),
        uc=np.array([1.0]),
        lx=np.full(4, -5.0),
        ux=np.full(4, 5.0),
        Q=Q,
        name="dense_q",
    )
    out = solve_ipm_native(p, tol=1e-8, time_limit=30.0)
    assert out["status"] == "optimal"
    k = kkt_residuals(p, out["x"], out["y"])
    assert k["max_rel"] <= 1e-7
    sol = solve(p, engine="ipm-native", tol=1e-8, backend="numpy", time_limit=30.0)
    _cert_optimal(p, sol, tol=1e-8)


def test_auto_falls_back_to_pdqp_when_ipm_stalls(monkeypatch):
    """If ipm-native cannot certify, auto must try PDQP rather than leave not_proven."""
    Q = sp.eye(2, format="csr")
    p = Problem(
        c=np.array([-1.0, -1.0]),
        A=sp.csr_matrix([[1.0, 1.0]]),
        lc=np.array([1.0]),
        uc=np.array([1.0]),
        lx=np.zeros(2),
        ux=np.full(2, np.inf),
        Q=Q,
        name="fb_qp",
    )

    def fake_native(prob, tol=1e-6, time_limit=3600.0, backend="numpy", verbose=False, **kw):
        from qenivo.api import _finish
        return _finish(
            prob, "stalled", np.array([0.4, 0.4]), np.array([0.0]), tol,
            {"engine": "ipm-native", "backend": "numpy", "iterations": 40, "time": 0.01,
             "reason": "engine chosen by the caller"},
        )

    from qenivo.engines import registry
    registry._builtin_plugins()
    monkeypatch.setitem(registry._ENGINES, "ipm-native",
                        registry.EngineSpec("ipm-native", fake_native, ("LP", "QP"), "stub"))
    sol = solve(p, engine="auto", tol=1e-6, backend="numpy", time_limit=30.0)
    assert sol.verdict == "optimal"
    assert sol.engine.get("engine") == "pdqp"
    assert "ipm-native" in (sol.engine.get("ladder") or [])

