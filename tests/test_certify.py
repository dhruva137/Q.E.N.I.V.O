"""Certificates are re-checked by an independent verifier that rejects tampered answers."""
import json

import numpy as np
import pytest

from conftest import ROOT, infeasible_lp, lp_all_bound_types, unbounded_lp
from qenivo import solve
from qenivo.certify.certificate import load
from qenivo.certify.verify import read_mps as v_read
from qenivo.certify.verify import verify
from qenivo.io.mps import read_mps, write_mps
from qenivo.models.williams import williams_lp


def _roundtrip(tmp_path, prob, **kw):
    mps = tmp_path / f"{prob.name}.mps"
    cert = tmp_path / f"{prob.name}.cert.json"
    write_mps(prob, mps)
    sol = solve(read_mps(mps), **kw)
    sol.save_certificate(cert)
    return mps, cert, sol


@pytest.mark.parametrize("exact", [False, True])
def test_optimal_certificate_verifies(tmp_path, exact):
    mps, cert, sol = _roundtrip(tmp_path, williams_lp(), tol=1e-9)
    rep = verify(mps, load(cert), exact=exact)
    assert rep["passed"], rep["checks"]


def test_mixed_bounds_certificate_verifies(tmp_path):
    p = lp_all_bound_types()
    keep = np.isfinite(p.lc) | np.isfinite(p.uc)          # free rows are dropped by any MPS reader
    from qenivo.model import Problem
    p = Problem(c=p.c, A=p.A[keep], lc=p.lc[keep], uc=p.uc[keep], lx=p.lx, ux=p.ux, name=p.name,
                row_names=[r for r, k in zip(p.row_names, keep) if k], col_names=p.col_names)
    mps, cert, sol = _roundtrip(tmp_path, p, tol=1e-9)
    rep = verify(mps, load(cert), exact=True)
    assert rep["passed"], rep["checks"]


def test_tampered_primal_is_rejected(tmp_path):
    mps, cert, sol = _roundtrip(tmp_path, williams_lp(), tol=1e-9)
    c = load(cert)
    k = next(iter(c["solution"]["x"]))
    c["solution"]["x"][k] += 500.0
    assert not verify(mps, c)["passed"]


def test_tampered_duals_are_rejected(tmp_path):
    mps, cert, sol = _roundtrip(tmp_path, williams_lp(), tol=1e-9)
    c = load(cert)
    for r in c["solution"]["y"]:
        c["solution"]["y"][r] = 0.0
    assert not verify(mps, c)["passed"]


def test_claimed_objective_must_match(tmp_path):
    mps, cert, sol = _roundtrip(tmp_path, williams_lp(), tol=1e-9)
    c = load(cert)
    c["objective"] += 1.0
    assert not verify(mps, c)["passed"]


def test_farkas_certificate_verifies(tmp_path):
    mps, cert, sol = _roundtrip(tmp_path, infeasible_lp())
    c = load(cert)
    assert c["verdict"] == "infeasible" and c["evidence"]["kind"] == "farkas"
    assert verify(mps, c, exact=True)["passed"]
    c["evidence"]["vector"] = {k: -v for k, v in c["evidence"]["vector"].items()}
    assert not verify(mps, c)["passed"]


def test_ray_certificate_verifies(tmp_path):
    mps, cert, sol = _roundtrip(tmp_path, unbounded_lp(), engine="pdhg", backend="numpy")
    c = load(cert)
    assert c["verdict"] == "unbounded"
    assert verify(mps, c, exact=True)["passed"]


def test_not_proven_never_verifies(tmp_path):
    mps, cert, sol = _roundtrip(tmp_path, williams_lp(), tol=1e-9)
    c = load(cert)
    c["verdict"] = "not_proven"
    assert not verify(mps, c)["passed"]


def test_certificate_is_strict_json(tmp_path):
    _, cert, _ = _roundtrip(tmp_path, williams_lp(), tol=1e-9)
    json.loads(cert.read_text())      # allow_nan=False on write: no NaN / Infinity tokens


def test_verifier_parser_matches_engine_parser(tmp_path):
    p = lp_all_bound_types()
    p.lc[-1], p.uc[-1] = -1.0, 7.0          # make the last row constrained so both parsers keep it
    mps = tmp_path / "m.mps"
    write_mps(p, mps)
    V = v_read(mps)
    q = read_mps(mps)
    assert V.rows == q.row_names and V.cols == q.col_names
    for i, r in enumerate(V.rows):
        lo, up = V.row_bounds(r)
        assert (lo, up) == (q.lc[i], q.uc[i])


def test_fingerprint_changes_with_data():
    p = williams_lp()
    q = p.copy(); q.c[0] += 1e-9
    assert p.fingerprint() != q.fingerprint()


@pytest.mark.network
@pytest.mark.parametrize("name,time_limit", [
    ("cplex2", 60.0),   # tiny elastic V; host check_farkas must still accept the ray
    ("gosh", 300.0),    # verify sign-rule tol must match host check_farkas
    ("gran", 120.0),    # elastic must prefer native simplex on ~2.6k-row models
])
def test_netlib_hard_infeas_verified_farkas(tmp_path, name, time_limit):
    """W09 gate cases: never report infeasible without an independently verified Farkas ray."""
    import sys
    sys.path.insert(0, str(ROOT / "bench"))
    from fetch import fetch  # noqa: E402

    path = fetch(name, verbose=False)
    sol = solve(path, engine="auto", tol=1e-6, time_limit=time_limit, backend="numpy")
    assert sol.verdict == "infeasible", (name, sol.status, sol.verdict, sol.engine)
    assert sol.ray is not None and sol.ray_check and sol.ray_check.get("valid")
    cert = tmp_path / f"{name}.cert.json"
    sol.source = str(path)
    sol.save_certificate(cert)
    rep = verify(path, load(cert))
    assert rep["passed"], (name, rep.get("checks"))
