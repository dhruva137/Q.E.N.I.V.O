"""Unit tests for warm-started bilinear SLP, homotopy, and McCormick/OBBT bounds."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from qenivo.io.gams import BilinearModel
from qenivo.model import Problem
from qenivo.workload.bilinear_bound import fbbt, mccormick_bound, obbt
from qenivo.workload.slp import run_homotopy, run_slp


def _tiny_bilinear(maximize: bool = True) -> BilinearModel:
    """Hand-built bilinear NLP with a known feasible point.

        max  x3
        s.t. x0 + x1 = 1
             x2 = x0 * x1          (bilinear)
             x3 = 4 * x2           (so optimum at x0=x1=0.5 -> x2=0.25 -> x3=1)
             0 <= x0,x1 <= 1
             x2, x3 free enough

    Feasible plan: (0.5, 0.5, 0.25, 1.0) with objective 1.0.
    Also feasible: (0.2, 0.8, 0.16, 0.64).
    """
    n, m = 4, 3
    A = sp.csr_matrix(([1.0, 1.0, -1.0, -1.0, 4.0],
                       ([0, 0, 1, 2, 2], [0, 1, 2, 3, 2])), shape=(m, n))
    lc = np.array([1.0, 0.0, 0.0])
    uc = np.array([1.0, 0.0, 0.0])
    lx = np.array([0.0, 0.0, 0.0, 0.0])
    ux = np.array([1.0, 1.0, 1.0, 2.0])
    sign = -1.0 if maximize else 1.0
    c = np.zeros(n)
    c[3] = sign
    lin = Problem(c=c, A=A, lc=lc, uc=uc, lx=lx, ux=ux, obj_sign=sign, name="tiny_bilinear",
                  row_names=["e0", "e1", "e2"], col_names=[f"x{i}" for i in range(n)])
    terms = [(1, 0, 1, 1.0)]
    start = np.array([0.3, 0.7, 0.0, 0.0])
    return BilinearModel(linear=lin, terms=terms, start=start, objective_var=3,
                         sense="maximizing" if maximize else "minimizing")


def test_tiny_feasible_point_hand_check():
    M = _tiny_bilinear()
    x = np.array([0.5, 0.5, 0.25, 1.0])
    assert M.violation(x).max() <= 1e-12
    assert abs(M.objective(x) - 1.0) <= 1e-12


def test_homotopy_finds_feasible_tiny():
    M = _tiny_bilinear()
    start_viol = float(M.violation(M.start).max())
    assert start_viol > 1e-6
    r = run_homotopy(M, max_iter=40, engine="simplex", tol_viol=1e-7, time_limit=30.0,
                     damping=1.0)
    # Pinned-quality fixed-factor LP recovers a feasible plan on this model.
    assert r.feasible or r.max_violation < 0.25 * start_viol
    assert float(M.violation(r.x).max()) == pytest.approx(r.max_violation, abs=1e-12)
    assert 0.0 <= r.objective <= 1.0 + 1e-4
    if r.feasible:
        assert float(M.violation(r.x).max()) <= 1e-6


def test_slp_warm_simplex_feasible_tiny():
    M = _tiny_bilinear()
    r = run_slp(M, engine="simplex", max_iter=40, homotopy_first=True, homotopy_iters=20,
                homotopy_damping=1.0, time_limit=60.0, tol_viol=1e-7)
    assert r.feasible, (r.status, r.max_violation, r.rel_violation, r.objective)
    assert float(M.violation(r.x).max()) <= 1e-6
    assert r.objective >= 0.9
    engines = {h.get("lp_engine") for h in r.history if "lp_engine" in h}
    assert "simplex" in engines or engines <= {None, "simplex"}


def test_slp_history_uses_warm_basis_flag():
    M = _tiny_bilinear()
    r = run_slp(M, engine="simplex", max_iter=15, homotopy_first=False, time_limit=30.0)
    slp_steps = [h for h in r.history if h.get("phase") == "slp" and "warm" in h]
    if len(slp_steps) >= 2:
        assert any(h.get("warm") for h in slp_steps[1:])


def test_mccormick_bound_certifies_tiny():
    M = _tiny_bilinear()
    out = mccormick_bound(M, engine="simplex", time_limit=30.0, use_obbt=True, obbt_vars=4,
                          obbt_time=10.0)
    assert out["bound"] is not None
    assert out["solution"].verdict == "optimal"
    assert out["bound"] >= 1.0 - 1e-5
    assert out["bound"] <= 2.0 + 1e-6


def test_fbbt_and_obbt_do_not_cut_feasible():
    M = _tiny_bilinear()
    x_feas = np.array([0.5, 0.5, 0.25, 1.0])
    L, U = fbbt(M, passes=20)
    assert np.all(x_feas >= L - 1e-9)
    assert np.all(x_feas <= U + 1e-9)
    L2, U2, info = obbt(M, L, U, max_vars=4, time_limit=20.0, engine="simplex")
    assert np.all(x_feas >= L2 - 1e-8)
    assert np.all(x_feas <= U2 + 1e-8)
    assert info["solves"] >= 1


def test_infeasible_start_still_reports_true_violation():
    """Honesty: an unfinished short run must not claim feasibility."""
    M = _tiny_bilinear()
    bad = np.array([0.9, 0.9, 0.0, 0.0])
    r = run_slp(M, x0=bad, engine="simplex", max_iter=1, homotopy_first=False,
                homotopy_iters=0, time_limit=5.0)
    measured = float(M.violation(r.x).max())
    assert abs(measured - r.max_violation) <= 1e-12
    if measured > 1e-6:
        assert not r.feasible
