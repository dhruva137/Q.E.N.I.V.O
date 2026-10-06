"""QP, MILP, routing, CLI and the provenance guard."""
import json
import subprocess
import sys

import numpy as np
import scipy.sparse as sp

from conftest import ROOT
from qenivo import ModelBuilder, Problem, solve
from qenivo.io.mps import write_mps
from qenivo.models.williams import williams_lp
from qenivo.provenance import guard
from qenivo.workload.router import route


def _qp():
    # min (x-1)^2 + (y-2)^2  s.t. x + y <= 2, x,y >= 0   -> x = 0.5, y = 1.5
    Q = sp.csr_matrix(np.diag([2.0, 2.0]))
    return Problem(c=np.array([-2.0, -4.0]), A=sp.csr_matrix([[1.0, 1.0]]), lc=np.array([-np.inf]),
                   uc=np.array([2.0]), lx=np.zeros(2), ux=np.full(2, np.inf), c0=5.0, Q=Q, name="qp")


def test_qp_known_solution():
    sol = solve(_qp(), tol=1e-8, backend="numpy")
    assert sol.verdict == "optimal"
    assert np.allclose(sol.x, [0.5, 1.5], atol=1e-5)
    assert abs(sol.objective - 0.5) <= 1e-6


def _knapsack():
    b = ModelBuilder("knap", maximize=True)
    w = [12, 7, 11, 8, 9]
    v = [24, 13, 23, 15, 16]
    for i, (wi, vi) in enumerate(zip(w, v)):
        b.var(f"x{i}", 0, 1, obj=vi, integer=True)
    b.row("cap", {f"x{i}": w[i] for i in range(5)}, hi=26)
    return b.build()


def test_milp_knapsack_optimum():
    sol = solve(_knapsack(), tol=1e-6)
    assert sol.verdict == "optimal"
    assert abs(sol.objective - 51.0) <= 1e-6         # items 0 and 2 (w 23, v 47) ... brute force below
    import itertools
    best = max(sum(v for v, s in zip([24, 13, 23, 15, 16], bits) if s)
               for bits in itertools.product([0, 1], repeat=5)
               if sum(w for w, s in zip([12, 7, 11, 8, 9], bits) if s) <= 26)
    assert abs(sol.objective - best) <= 1e-6


def test_milp_integer_infeasible_is_not_claimed_optimal():
    b = ModelBuilder("odd")
    b.var("x", 0, 10, integer=True)
    b.row("half", {"x": 2}, lo=3, hi=3)            # 2x = 3 has no integer solution
    sol = solve(b.build())
    assert sol.verdict != "optimal"


def test_router_decisions():
    p = williams_lp()
    assert route(p, 1, backend="numpy")["engine"] == "simplex"
    r = route(p, 32, backend="numpy")
    assert r["backend"] == "numpy" and "case by case" in r["reason"]
    r_qp = route(_qp(), 1, backend="numpy")
    assert r_qp["engine"] == "ipm-native"
    assert "KKT" in r_qp["reason"] or "native" in r_qp["reason"]


def test_cli_solve_verify_roundtrip(tmp_path):
    mps = tmp_path / "w.mps"
    cert = tmp_path / "w.cert.json"
    write_mps(williams_lp(), mps)
    env = {"PYTHONPATH": str(ROOT / "src")}
    import os
    env = {**os.environ, **env}
    r = subprocess.run([sys.executable, "-m", "qenivo.cli", "solve", str(mps), "--cert", str(cert)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "-m", "qenivo.cli", "verify", str(mps), str(cert), "--exact"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "VERIFIED" in r.stdout, r.stdout + r.stderr
    c = json.loads(cert.read_text())
    c["solution"]["x"]["crude1"] += 100
    cert.write_text(json.dumps(c))
    r = subprocess.run([sys.executable, "-m", "qenivo.cli", "verify", str(mps), str(cert)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 4 and "REJECTED" in r.stdout


_ALL_ENGINES = """
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from qenivo.provenance import guard, install_tripwires
install_tripwires()
from conftest import infeasible_lp
from qenivo import solve
from qenivo.models.williams import williams_lp
from test_qp_milp_system import _knapsack, _qp
for eng in ("simplex", "ipm", "pdhg"):
    solve(williams_lp(), engine=eng, backend="numpy")
    solve(infeasible_lp(), engine=eng, backend="numpy")
solve(_qp(), backend="numpy")
solve(_knapsack())
print(json.dumps(guard()))
"""


def test_provenance_guard_clean_after_all_engines():
    """Every engine runs in a process that imported nothing else, and no forbidden package loads.

    The check reads sys.modules, so it runs in a fresh interpreter: in the pytest process the PuLP
    and Pyomo binding tests import those libraries on purpose (PuLP 4 also pulls in highspy).
    """
    import os
    r = subprocess.run([sys.executable, "-c", _ALL_ENGINES, str(ROOT / "src"), str(ROOT / "tests")],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                       timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    g = json.loads(r.stdout.strip().splitlines()[-1])
    assert g["tripwires_installed"]
    assert g["clean"], g


def test_no_tripwire_hit_in_this_process():
    """A forbidden entry point (splu, linprog, ...) was never called by any test run so far."""
    g = guard()
    assert g["tripwires_installed"]
    assert g["tripwire_hits"] == [], g["tripwire_hits"]
