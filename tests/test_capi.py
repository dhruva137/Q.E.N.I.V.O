"""The C ABI solves a known LP and a known MILP, and the modelling plugins translate a fake model."""
from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "capi"))

import capi_loader  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


affine = _load(ROOT / "bindings" / "affine.py", "qenivo_bindings_affine")

pytestmark = pytest.mark.skipif(capi_loader.compiler() is None, reason="no C++ compiler for the C ABI")


def _plugin(rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location("plugin_" + path.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bound():
    return capi_loader.solve_csr(
        np.array([-1.0]), np.array([0], dtype=np.int32), None, None,
        np.zeros(0), np.zeros(0), np.array([0.0]), np.array([1.0]), engine="simplex", name="bound")


def _milp():
    return capi_loader.solve_csr(
        np.array([-1.0, -2.0]), np.array([0, 2], dtype=np.int32), np.array([0, 1], dtype=np.int32),
        np.array([1.0, 1.0]), np.array([-np.inf]), np.array([1.0]), np.zeros(2), np.ones(2),
        integrality=np.array([1, 1], dtype=np.int32), engine="milp", name="two_binary")


def test_abi_version():
    lib = capi_loader.library()
    assert lib.qenivo_abi_version() == 1


def test_bound_lp_optimum_is_minus_one():
    sol = _bound()
    assert sol.status_name == "optimal"
    assert sol.objective == pytest.approx(-1.0)
    assert sol.x == pytest.approx([1.0])
    assert sol.reduced_costs == pytest.approx([-1.0])
    assert int(sol.column_basis[0]) == 2  # at upper bound
    cert = json.loads(sol.certificate)
    assert cert["schema"] == "qenivo.capi.certificate/1"
    assert cert["impl"] == "native simplex"
    assert cert["objective"] == pytest.approx(-1.0)


def test_two_binary_milp():
    sol = _milp()
    assert sol.status_name == "optimal"
    assert sol.objective == pytest.approx(-2.0)
    assert sol.x == pytest.approx([0.0, 1.0])
    cert = json.loads(sol.certificate)
    assert cert["schema"] == "qenivo.certificate/1"
    assert cert["objective"] == pytest.approx(-2.0)


def test_vertex_matches_python_simplex():
    import qenivo
    from qenivo.model import Problem

    # A unique optimum (x = (0, 1), y = -2): with equal costs every point of x1 + x2 = 1 is optimal, and the
    # Python simplex (QENIVO_NATIVE=0) and the C ABI may then return different, equally valid vertices.
    matrix = sp.csr_matrix(([1.0, 1.0], ([0, 0], [0, 1])), shape=(1, 2))
    prob = Problem(c=np.array([-1.0, -2.0]), A=matrix, lc=np.array([-np.inf]), uc=np.array([1.0]),
                   lx=np.zeros(2), ux=np.full(2, np.inf), name="vertex")
    py = qenivo.solve(prob, engine="simplex", tol=1e-8)
    cap = capi_loader.solve_csr(
        prob.c, matrix.indptr, matrix.indices, matrix.data, prob.lc, prob.uc, prob.lx, prob.ux, engine="native")
    assert cap.objective == pytest.approx(py.objective)
    assert cap.x == pytest.approx(py.x)
    assert cap.y == pytest.approx(py.y, abs=1e-7)


def test_bad_csr_is_an_integer_error():
    with pytest.raises(capi_loader.CapiError) as caught:
        capi_loader.solve_csr(np.array([-1.0]), np.array([1], dtype=np.int32), None, None,
                              np.zeros(0), np.zeros(0), np.array([0.0]), np.array([1.0]))
    assert caught.value.code == 2


def test_simplex_refuses_integrality_and_quadratic():
    with pytest.raises(capi_loader.CapiError) as caught:
        capi_loader.solve_csr(np.array([-1.0]), np.array([0], dtype=np.int32), None, None,
                              np.zeros(0), np.zeros(0), np.array([0.0]), np.array([1.0]),
                              integrality=np.array([1], dtype=np.int32), engine="simplex")
    assert caught.value.code == 4
    with pytest.raises(capi_loader.CapiError) as caught:
        capi_loader.solve_csr(np.array([1.0]), np.array([0], dtype=np.int32), None, None,
                              np.zeros(0), np.zeros(0), np.array([0.0]), np.array([1.0]),
                              q_ptr=np.array([0, 1], dtype=np.int32), q_idx=np.array([0], dtype=np.int32),
                              q_val=np.array([2.0]), engine="simplex")
    assert caught.value.code == 4


def test_progress_callback_sees_start_and_finish():
    seen = []

    def progress(_user, iteration, _objective, status):
        seen.append((int(iteration), int(status)))

    sol = capi_loader.solve_csr(
        np.array([-1.0]), np.array([0], dtype=np.int32), None, None, np.zeros(0), np.zeros(0),
        np.array([0.0]), np.array([1.0]), progress=progress)
    assert sol.objective == pytest.approx(-1.0)
    assert seen[0][1] == -1
    assert seen[-1][1] == 0


def test_two_threads_solve_two_handles():
    def once(_i):
        return _bound().objective

    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(once, (1, 2)))
    assert values == pytest.approx([-1.0, -1.0])


def _plugins():
    return [
        _plugin("bindings/pyomo/qenivo_solver.py"),
        _plugin("bindings/pulp/qenivo_solver.py"),
        _plugin("bindings/cvxpy/qenivo_solver.py"),
        _plugin("bindings/jump/translate.py"),
    ]


def test_plugins_translate_a_fake_lp_and_milp():
    for plugin in _plugins():
        lp = affine.FakeModel(name="bound")
        lp.var("x", 0.0, 1.0, obj=-1.0)
        sol = plugin.solve(lp)
        assert sol.objective == pytest.approx(-1.0)
        assert sol.x == pytest.approx([1.0])
        milp = affine.FakeModel(name="bins")
        milp.var("x", obj=-1.0, binary=True)
        milp.var("y", obj=-2.0, binary=True)
        milp.cons_add({"x": 1.0, "y": 1.0}, ub=1.0)
        sol = plugin.solve(milp)
        assert sol.objective == pytest.approx(-2.0)
        assert sol.x == pytest.approx([0.0, 1.0])


def test_pyomo_model_when_installed():
    pyo = pytest.importorskip("pyomo.environ")      # ConcreteModel lives in pyomo.environ
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, 1))
    model.obj = pyo.Objective(expr=-model.x)
    sol = _plugin("bindings/pyomo/qenivo_solver.py").solve(model)
    assert sol.objective == pytest.approx(-1.0)
    assert sol.x == pytest.approx([1.0])


def _pulp_var(pulp, prob, name, **kw):
    """PuLP 4 creates variables on the problem; up to PuLP 3 they are standalone."""
    return prob.add_variable(name, **kw) if hasattr(prob, "add_variable") else pulp.LpVariable(name, **kw)


def test_pulp_model_when_installed():
    pulp = pytest.importorskip("pulp")
    prob = pulp.LpProblem("bound", pulp.LpMinimize)
    x = _pulp_var(pulp, prob, "x", lowBound=0, upBound=1)
    prob += -x
    sol = _plugin("bindings/pulp/qenivo_solver.py").solve(prob)
    assert sol.objective == pytest.approx(-1.0)
    assert sol.x == pytest.approx([1.0])


def test_pulp_solver_constraints_constant_and_status():
    """Rows of every sense, an objective constant, a binary, and PuLP's own prob.solve(solver) path.

    PuLP 4 made `constraints` a method returning a list and replaced the LpStatus* constants with
    LpSolveStatus; the binding read the dict attribute and dropped the objective constant.
    """
    pulp = pytest.importorskip("pulp")
    mod = _plugin("bindings/pulp/qenivo_solver.py")
    prob = pulp.LpProblem("mix", pulp.LpMaximize)
    x = _pulp_var(pulp, prob, "x", lowBound=0, upBound=1)
    z = _pulp_var(pulp, prob, "z", cat="Binary")
    prob += 3 * x + 2 * z + 5
    prob += (x + z <= 1.5, "cap")
    prob += (2 * x - z >= -1, "lo")
    prob += (x == 0.25, "fix")
    status = prob.solve(mod.QenivoSolver())
    code = getattr(status, "status", status)          # LpSolveStats in PuLP 4, an int before
    assert int(getattr(code, "value", code)) == 1
    assert (x.varValue, z.varValue) == pytest.approx((0.25, 1.0))
    assert mod.solve(prob).objective == pytest.approx(7.75)      # 0.75 + 2 + 5

    bad = pulp.LpProblem("infeasible", pulp.LpMinimize)
    y = _pulp_var(pulp, bad, "y", lowBound=0, upBound=1)
    bad += y
    bad += (y >= 2, "need")
    status = bad.solve(mod.QenivoSolver())
    code = getattr(status, "status", status)
    assert int(getattr(code, "value", code)) == -1
    assert getattr(status, "has_solution", False) is False
    assert y.varValue is None                      # no values written back for an infeasible model


def test_cvxpy_model_when_installed():
    cp = pytest.importorskip("cvxpy")
    x = cp.Variable(nonneg=True)
    prob = cp.Problem(cp.Minimize(-x), [x <= 1])
    sol = _plugin("bindings/cvxpy/qenivo_solver.py").solve(prob)
    assert sol.objective == pytest.approx(-1.0)


def test_jump_example_when_julia_is_installed():
    julia = __import__("shutil").which("julia")
    if julia is None:
        pytest.skip("julia is not installed")
    import subprocess
    script = ROOT / "bindings" / "julia" / "example.jl"
    proc = subprocess.run([julia, "--project=", str(script)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "-1" in proc.stdout
