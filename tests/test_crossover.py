"""Crossover: a moderate-accuracy PDHG point finished to an exact, verified vertex."""
import numpy as np
import pytest

from conftest import DATA
from qenivo.certify.kkt import kkt_residuals
from qenivo.engines import pdhg
from qenivo.engines.crossover import basis_from_point, crossover
from qenivo.engines.simplex import solve_simplex
from qenivo.io.mps import read_mps
from qenivo.models import industry
from qenivo.models.refinery import refinery_level

MPS = ["afiro", "adlittle", "blend", "kb2", "sc50a", "sc105"]


def _check(prob, r, ref_obj):
    assert r["status"] == "optimal", r.get("crossover_log")
    k = kkt_residuals(prob, r["x"], r["y"])
    assert k["max_rel"] <= 1e-9, k
    assert abs(k["objective"] - ref_obj) <= 1e-7 * (1 + abs(ref_obj))
    nb = sum(v == "basic" for v in r["col_statuses"]) + sum(v == "basic" for v in r["row_statuses"])
    assert nb == prob.m
    assert len(r["col_statuses"]) == prob.n and len(r["row_statuses"]) == prob.m


@pytest.mark.parametrize("name", MPS)
@pytest.mark.parametrize("tol", [1e-2, 1e-3])
def test_crossover_netlib(name, tol):
    prob = read_mps(DATA / f"{name}.mps")
    ref = solve_simplex(prob)
    br = pdhg.solve(prob, tol=tol, backend="numpy", time_limit=60)
    r = crossover(prob, br.x[:, 0], br.y[:, 0])
    _check(prob, r, prob.objective(ref["x"]))


@pytest.mark.parametrize("name", ["afiro", "sc105"])
def test_crossover_full_lp_path(name):
    prob = read_mps(DATA / f"{name}.mps")
    ref = solve_simplex(prob)
    br = pdhg.solve(prob, tol=1e-3, backend="numpy", time_limit=60)
    r = crossover(prob, br.x[:, 0], br.y[:, 0], reduce=False)
    _check(prob, r, prob.objective(ref["x"]))


def test_crossover_from_a_bad_point_still_exact():
    prob = read_mps(DATA / "adlittle.mps")
    ref = solve_simplex(prob)
    r = crossover(prob, np.zeros(prob.n), np.zeros(prob.m))
    _check(prob, r, prob.objective(ref["x"]))


@pytest.mark.parametrize("make", [lambda: industry.transportation(10, 30), lambda: industry.crude_blending(12),
                                  lambda: industry.multi_commodity_flow(12, 4), lambda: refinery_level(1)])
def test_crossover_sector_lps(make):
    prob = make()
    ref = solve_simplex(prob)
    br = pdhg.solve(prob, tol=1e-3, backend="numpy", time_limit=60)
    r = crossover(prob, br.x[:, 0], br.y[:, 0])
    _check(prob, r, prob.objective(ref["x"]))


def test_basis_guess_has_m_basics():
    prob = read_mps(DATA / "sc50a.mps")
    st = basis_from_point(prob, np.zeros(prob.n), np.zeros(prob.m))
    assert sum(v == "basic" for v in st["col_statuses"] + st["row_statuses"]) == prob.m
