"""qenivo-io shell in scripted mode: every command, certificates beside each solve, honest refusals."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from qenivo.certify.certificate import load
from qenivo.certify.verify import verify
from qenivo.compat import shell as qs

DATA = Path(__file__).resolve().parent / "data"
AFIRO = DATA / "afiro.mps"
AFIRO_OPT = -464.7531428571

KNAPSACK = """\\ binary knapsack: optimum 21 (b, c, d), LP relaxation 22.0
Maximize
 value: 8 a + 11 b + 6 c + 4 d
Subject To
 cap: 5 a + 7 b + 4 c + 3 d <= 14
Binary
 a b c d
End
"""

INFEASIBLE = """\\ x + y >= 3 with x, y <= 1
Minimize
 obj: x
Subject To
 need: x + y >= 3
Bounds
 0 <= x <= 1
 0 <= y <= 1
End
"""


def _shell(tmp_path):
    out = io.StringIO()
    return qs.Shell(out=out, certdir=str(tmp_path / "certs")), out


def _run(sh, out, *lines) -> str:
    out.seek(0)
    out.truncate()
    sh.run_lines(lines)
    return out.getvalue()


@pytest.fixture
def afiro(tmp_path):
    sh, out = _shell(tmp_path)
    _run(sh, out, f'read "{AFIRO}"', "optimize")
    return sh, out


def test_read_and_problem_stats(tmp_path):
    sh, out = _shell(tmp_path)
    text = _run(sh, out, f'read "{AFIRO}"', "display problem stats")
    assert "LP, 27 rows x 32 cols, 83 nnz" in text
    assert "less 19, greater 0, equal 8, ranged 0, free 0" in text
    assert "continuous 32, integer 0, binary 0" in text
    assert "objective nonzeros 5" in text
    assert sh.errors == 0


def test_optimize_writes_a_certificate_that_verifies(afiro, tmp_path):
    sh, out = afiro
    assert sh.sol.verdict == "optimal"
    assert sh.sol.objective == pytest.approx(AFIRO_OPT, rel=1e-9)
    assert "Proven optimal." in out.getvalue()
    cert_path = tmp_path / "certs" / "afiro.cert.json"
    assert cert_path.exists() and sh.cert_path == cert_path
    cert = load(cert_path)
    assert cert["verdict"] == "optimal"
    assert cert["shell"]["command"] == "optimize"
    assert verify(str(AFIRO), cert)["passed"]
    text = _run(sh, out, "display solution quality")
    assert "independent check  passed" in text


@pytest.mark.parametrize("cmd,engine", [("primopt", "simplex"), ("tranopt", "simplex"), ("baropt", "ipm")])
def test_each_algorithm_command(tmp_path, cmd, engine):
    sh, out = _shell(tmp_path)
    text = _run(sh, out, f'read "{AFIRO}"', cmd)
    assert sh.errors == 0, text
    assert sh.sol.verdict == "optimal"
    assert sh.sol.objective == pytest.approx(AFIRO_OPT, rel=1e-6)
    assert sh.sol.engine["engine"] == engine
    assert load(sh.cert_path)["shell"]["command"] == cmd
    if cmd == "primopt":
        assert sh.sol.engine["method"].startswith("primal simplex")


def test_display_objective_variables_and_selection(afiro):
    sh, out = afiro
    text = _run(sh, out, "display solution objective")
    assert "Objective = -4.6475314286e+02" in text and "Proven optimal" in text
    _, cn = sh.prob.names()
    text = _run(sh, out, "display solution variables -")
    shown = {ln.split()[0]: float(ln.split()[1]) for ln in text.splitlines()[1:] if ln and not ln.startswith("(")}
    nz = {cn[j]: v for j, v in enumerate(sh.sol.x) if abs(v) > 1e-12}
    assert shown.keys() == nz.keys()
    for k, v in nz.items():
        assert shown[k] == pytest.approx(v, rel=1e-10)
    assert f"({32 - len(nz)} of 32 selected values are zero" in text
    # ranges by name and by position, globs, abbreviations
    assert sh._select(cn, ["X01-X04"]) == [0, 1, 2, 3]
    assert sh._select(cn, ["2-3"]) == [1, 2]
    assert sh._select(cn, ["X0?"]) == [j for j, nm in enumerate(cn) if len(nm) == 3 and nm.startswith("X0")]
    text = _run(sh, out, "disp sol var X01")
    assert "X01" in text and sh.errors == 0
    text = _run(sh, out, "d s v NOPE")
    assert "no such item 'NOPE'" in text and sh.errors == 1


def test_display_dual_slacks_reduced(afiro):
    sh, out = afiro
    p, sol = sh.prob, sh.sol
    rn, cn = p.names()
    text = _run(sh, out, "display solution dual *")
    rows = {ln.split()[0]: float(ln.split()[1]) for ln in text.splitlines()[1:] if ln and not ln.startswith("(")}
    for i, nm in enumerate(rn):
        if abs(sol.y[i]) > 1e-12:
            assert rows[nm] == pytest.approx(sol.y[i], rel=1e-10)
    s = qs.slacks(p, sol.x)
    act = p.A @ sol.x
    for i in range(p.m):            # afiro has only <= and = rows
        rhs = p.uc[i] if np.isinf(p.lc[i]) else p.lc[i]
        assert s[i] == pytest.approx(rhs - act[i], abs=1e-9)
    assert np.all(s[np.isinf(p.lc)] >= -1e-9)         # <= rows: slack is spare capacity
    text = _run(sh, out, "display solution slacks -")
    assert "Constraint" in text
    text = _run(sh, out, "display solution reduced -")
    rc = sol.reduced_costs
    shown = {ln.split()[0]: float(ln.split()[1]) for ln in text.splitlines()[1:] if ln and not ln.startswith("(")}
    for j, nm in enumerate(cn):
        if abs(rc[j]) > 1e-12:
            assert shown[nm] == pytest.approx(rc[j], rel=1e-8, abs=1e-12)
    assert sh.errors == 0


def test_slack_conventions_on_every_row_kind():
    import scipy.sparse as sp

    from qenivo.model import Problem
    A = sp.csr_matrix(np.ones((5, 1)))
    p = Problem(c=[1.0], A=A, lc=[-np.inf, 1.0, 2.0, 0.5, -np.inf], uc=[4.0, np.inf, 2.0, 3.0, np.inf],
                lx=[0.0], ux=[10.0])
    s = qs.slacks(p, [2.0])
    assert s[:4] == pytest.approx([2.0, -1.0, 0.0, 1.5])   # L: rhs-act, G/E: rhs-act, R: act-lower
    assert np.isnan(s[4])


def test_mipopt_gap_setting_and_lp_only_commands(tmp_path):
    f = tmp_path / "knap.lp"
    f.write_text(KNAPSACK)
    sh, out = _shell(tmp_path)
    text = _run(sh, out, f'read "{f}"', "set mip tolerances mipgap 0.001", "mipopt", "display solution objective")
    assert "MILP" in text and sh.errors == 0, text
    assert sh.sol.verdict == "optimal"
    assert sh.sol.objective == pytest.approx(21.0)
    assert "best bound" in text
    cert = load(sh.cert_path)
    assert cert["shell"]["settings"]["mipgap"] == 0.001
    xs = cert["solution"]["x"]
    assert [round(xs[k]) for k in "abcd"] == [0, 1, 1, 1]
    text = _run(sh, out, "primopt")
    assert "integer variables" in text and sh.errors == 1
    text = _run(sh, out, "display solution dual")
    assert "no dual values" in text


def test_mipgap_reaches_the_tree(tmp_path, monkeypatch):
    import importlib
    milp = importlib.import_module("qenivo.engines.milp")    # qenivo.engines is also a function name
    seen = {}
    real = milp.solve_milp

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(milp, "solve_milp", spy)
    f = tmp_path / "knap.lp"
    f.write_text(KNAPSACK)
    sh, out = _shell(tmp_path)
    _run(sh, out, f'read "{f}"', "set mip tol mipgap 0.25", "set timelimit 30", "optimize")
    assert seen["gap_tol"] == 0.25 and seen["time_limit"] == 30.0


def test_mipopt_on_lp_falls_back_to_optimize(tmp_path):
    sh, out = _shell(tmp_path)
    text = _run(sh, out, f'read "{AFIRO}"', "mipopt")
    assert "No integer variables" in text
    assert sh.sol.verdict == "optimal"


def test_infeasible_is_proven(tmp_path):
    f = tmp_path / "bad.lp"
    f.write_text(INFEASIBLE)
    for cmd in ("optimize", "primopt"):
        sh, out = _shell(tmp_path)
        text = _run(sh, out, f'read "{f}"', cmd, "display solution objective")
        assert sh.sol.verdict == "infeasible", text
        assert "Proven infeasible (Farkas certificate)." in text
        ev = load(sh.cert_path)["evidence"]
        assert ev["kind"] == "farkas" and ev["check"]["valid"] is True and ev["vector"]
        assert "No objective value: verdict infeasible" in text


def test_set_and_display_settings(tmp_path):
    sh, out = _shell(tmp_path)
    text = _run(sh, out, "set timelimit 12.5", "set threads 4", "set tolerance 1e-7", "display settings")
    assert sh.settings["timelimit"] == 12.5 and sh.settings["threads"] == 4 and sh.settings["tolerance"] == 1e-7
    assert "recorded in the certificate only" in text
    assert "timelimit  12.5   (default 3600.0)" in text
    text = _run(sh, out, "set emphasis mip 1", "set mip strategy branch 1")
    assert text.count("has no QENIVO equivalent") == 2 and sh.errors == 0
    text = _run(sh, out, "set timelimit -3", "set threads two", "set mip tolerances mipgap 2")
    assert sh.errors == 3 and text.count("invalid value") == 3
    _run(sh, out, "set defaults")
    assert sh.settings == dict(qs.DEFAULTS, certdir=str(tmp_path / "certs"))
    text = _run(sh, out, "display settings changed")
    assert "All parameters are at their defaults." in text


def test_timelimit_and_tolerance_reach_the_engine(tmp_path, monkeypatch):
    import qenivo.api as api
    seen = {}
    real = api.solve

    def spy(prob, **k):
        seen.update(k)
        return real(prob, **k)

    monkeypatch.setattr(api, "solve", spy)
    sh, out = _shell(tmp_path)
    _run(sh, out, f'read "{AFIRO}"', "set timelimit 42", "set tolerance 1e-7", "tranopt")
    assert seen == {"engine": "simplex", "tol": 1e-7, "time_limit": 42.0}


def test_write_every_format_and_read_back(afiro, tmp_path):
    sh, out = afiro
    files = {k: tmp_path / f"out.{k}" for k in ("sol", "mps", "lp")}
    cert = tmp_path / "copy.json"
    text = _run(sh, out, *[f'write "{p}"' for p in files.values()], f'write "{cert}"')
    assert sh.errors == 0, text
    sol_text = files["sol"].read_text()
    assert "status optimal" in sol_text and "objective -464.75314285" in sol_text
    assert json.loads(cert.read_text())["verdict"] == "optimal"
    for kind in ("mps", "lp"):
        sh2, out2 = _shell(tmp_path)
        _run(sh2, out2, f'read "{files[kind]}"', "optimize")
        assert sh2.sol.objective == pytest.approx(AFIRO_OPT, rel=1e-9), kind
    text = _run(sh, out, f'write "{tmp_path / "noext"}"')
    assert "cannot tell the file type" in text


def test_errors_and_quit(tmp_path):
    sh, out = _shell(tmp_path)
    text = _run(sh, out, "optimize", "display solution objective", "frobnicate", "read nothing_here.mps")
    assert "no problem loaded" in text and "unknown command 'frobnicate'" in text and "no such file" in text
    assert sh.errors == 4
    assert sh.run_line("q") is False and sh.run_line("exit") is False
    assert sh.run_line("# a comment") is True and sh.run_line("") is True


def test_help_lists_commands_and_limits(tmp_path):
    sh, out = _shell(tmp_path)
    text = _run(sh, out, "help")
    for cmd in ("read", "optimize", "primopt", "tranopt", "baropt", "mipopt", "display", "set", "write", "quit"):
        assert f"\n{cmd}" in "\n" + text
    assert "threads    accepted and recorded" in text and "Not available" in text
    assert "dual simplex" in _run(sh, out, "help tranopt")


def test_main_command_file_stdin_and_exit_codes(tmp_path, monkeypatch, capsys):
    script = tmp_path / "cmds.txt"
    script.write_text("optimize\ndisplay solution objective\nquit\ndisplay problem stats\n")
    certs = tmp_path / "c"
    assert qs.main([str(AFIRO), "-f", str(script), "--certdir", str(certs)]) == 0
    text = capsys.readouterr().out
    assert "Objective = -4.6475314286e+02" in text
    assert "Problem name" not in text                    # nothing runs after quit
    assert (certs / "afiro.cert.json").exists()

    assert qs.main(["-c", f'read "{AFIRO}"', "bogus", "--certdir", str(certs)]) == 1

    monkeypatch.setattr("sys.stdin", io.StringIO(f'read "{AFIRO}"\ntranopt\ndisp sol obj\n'))
    assert qs.main(["--certdir", str(certs)]) == 0
    text = capsys.readouterr().out
    assert qs.PROMPT not in text and "Objective = -4.6475314286e+02" in text


# ---------------------------------------------------------------------- C subset (capi/compat)
def _qnv_build():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "capi" / "compat" / "build_qnv.py"
    spec = importlib.util.spec_from_file_location("qnv_build", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_qnv_c_program_passes_every_check():
    qb = _qnv_build()
    if qb.toolchain() is None:
        pytest.skip("no gcc/g++ or clang/clang++ for the QNV C program")
    r = qb.run(timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert ", 0 failures" in r.stdout
    assert int(r.stdout.split()[0]) >= 60          # the program really ran its checks
