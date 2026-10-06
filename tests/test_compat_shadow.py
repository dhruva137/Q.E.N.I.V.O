"""Shadow mode: an external solver run as a separate command, diffed against certified QENIVO answers.

HiGHS (highspy, a bench-only comparator) is the external solver when it is installed; it only ever
runs in a child process, so this test process never imports it. Mismatch detection is also tested
with a fixed-answer external command that needs no solver at all.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from qenivo.compat import shadow as sh

DATA = Path(__file__).resolve().parent / "data"
AFIRO = DATA / "afiro.mps"

HIGHS_ADAPTER = r'''
"""Run HiGHS on an MPS file and write the shadow-mode JSON answer (optionally falsified)."""
import json, sys
import highspy

model, out = sys.argv[1], sys.argv[2]
cheat = sys.argv[3] if len(sys.argv) > 3 else ""
h = highspy.Highs()
h.setOptionValue("output_flag", False)
h.readModel(model)
h.run()
lp = h.getLp()
st = h.modelStatusToString(h.getModelStatus()).lower()
st = {"optimal": "optimal", "infeasible": "infeasible", "unbounded": "unbounded"}.get(st, st)
s = h.getSolution()
ans = {"status": st, "objective": h.getInfo().objective_function_value,
       "x": dict(zip(lp.col_names_, map(float, s.col_value)))}
if s.dual_valid:
    ans["y"] = dict(zip(lp.row_names_, map(float, s.row_dual)))
if cheat == "objective":
    ans["objective"] += 1.0
elif cheat == "x":
    ans["x"] = {k: 1.5 * v for k, v in ans["x"].items()}
elif cheat == "status":
    ans["status"] = "infeasible"
elif cheat == "dual":
    ans["y"] = {k: v + 0.5 for k, v in ans["y"].items()}
if out == "-":
    print(json.dumps(ans))
else:
    open(out, "w").write(json.dumps(ans))
'''

FIXED_ANSWER = r'''
import json, sys
open(sys.argv[2], "w").write(open(sys.argv[1] + ".answer.json").read())
'''

needs_highs = pytest.mark.skipif(importlib.util.find_spec("highspy") is None,
                                 reason="no external solver (highspy) installed")


@pytest.fixture
def adapter(tmp_path):
    p = tmp_path / "highs_adapter.py"
    p.write_text(HIGHS_ADAPTER)
    return p


def _cmd(script, *extra):
    return [sys.executable, str(script), "{model}", "{solution}", *extra]


@needs_highs
def test_agreement_on_a_small_lp(adapter, tmp_path):
    cert = tmp_path / "afiro.cert.json"
    had_highspy = "highspy" in sys.modules      # other tests (PuLP 4) may have loaded it already
    rep = sh.shadow(str(AFIRO), _cmd(adapter), cert_path=str(cert))
    assert rep.agree, rep.text()
    names = [n for n, _, _ in rep.checks]
    for must in ("verdicts agree", "objectives agree", "external objective matches its x",
                 "external x is feasible on the model", "external duals pass QENIVO's dual check"):
        assert must in names
    assert rep.qenivo["verdict"] == "optimal"
    assert json.loads(cert.read_text())["verdict"] == "optimal"
    assert had_highspy or "highspy" not in sys.modules          # the comparator never ran in-process
    assert "AGREE" in rep.text()


@needs_highs
def test_stdout_answer_and_string_command(adapter, tmp_path):
    cmd = f'"{sys.executable}" "{adapter}" {{model}} -'
    rep = sh.shadow(str(AFIRO), cmd, cert_path=str(tmp_path / "c.json"))
    assert rep.agree, rep.text()


@needs_highs
@pytest.mark.parametrize("cheat,failing", [
    ("objective", {"objectives agree", "external objective matches its x"}),
    ("x", {"external x is feasible on the model"}),
    ("status", {"verdicts agree"}),
    ("dual", {"external duals pass QENIVO's dual check"}),
])
def test_wrong_external_answer_is_a_mismatch(adapter, tmp_path, cheat, failing):
    rep = sh.shadow(str(AFIRO), _cmd(adapter, cheat), cert_path=str(tmp_path / "c.json"))
    assert not rep.agree
    bad = {n for n, ok, _ in rep.checks if not ok}
    assert failing <= bad, rep.text()
    assert "MISMATCH" in rep.text()


@needs_highs
def test_milp_shadow_checks_integrality(adapter, tmp_path):
    from qenivo.io.mps import write_mps
    from qenivo.model import ModelBuilder
    b = ModelBuilder("knap", maximize=True)
    for nm, v in zip("abcd", (8, 11, 6, 4)):
        b.var(nm, 0, 1, obj=v, integer=True)
    b.row("cap", dict(zip("abcd", (5, 7, 4, 3))), hi=14)
    f = tmp_path / "knap.mps"
    write_mps(b.build(), f)
    rep = sh.shadow(str(f), _cmd(adapter), cert_path=str(tmp_path / "k.json"))
    assert rep.agree, rep.text()
    assert rep.qenivo["objective"] == pytest.approx(21.0)
    assert "external x is integral" in [n for n, _, _ in rep.checks]


def _fixed(tmp_path, answer: dict):
    model = tmp_path / "afiro.mps"
    model.write_bytes(AFIRO.read_bytes())
    (tmp_path / "afiro.mps.answer.json").write_text(json.dumps(answer))
    script = tmp_path / "fixed.py"
    script.write_text(FIXED_ANSWER)
    return model, _cmd(script)


def test_fixed_wrong_answer_without_any_solver(tmp_path):
    from qenivo.api import solve
    good = solve(str(AFIRO))
    _, cn = good.problem.names()
    rn, _ = good.problem.names()
    x = dict(zip(cn, good.x.tolist()))
    y = dict(zip(rn, good.y.tolist()))
    model, cmd = _fixed(tmp_path, {"status": "optimal", "objective": good.objective, "x": x, "y": y})
    rep = sh.shadow(str(model), cmd, cert_path=str(tmp_path / "c.json"))
    assert rep.agree, rep.text()                       # QENIVO's own answer replayed agrees
    model, cmd = _fixed(tmp_path, {"status": "optimal", "objective": good.objective * 0.99, "x": x})
    rep = sh.shadow(str(model), cmd, cert_path=str(tmp_path / "c.json"))
    assert not rep.agree
    assert {"objectives agree", "external objective matches its x"} <= {n for n, ok, _ in rep.checks if not ok}
    assert any("no duals" in n for n in rep.notes)
    x.pop(cn[0])
    model, cmd = _fixed(tmp_path, {"status": "optimal", "objective": good.objective, "x": x})
    rep = sh.shadow(str(model), cmd, cert_path=str(tmp_path / "c.json"))
    assert not rep.agree and any("missing from external x" in m for m in rep.mismatches)


def test_alternative_optimum_is_a_note_not_a_mismatch(tmp_path):
    from qenivo.io.mps import write_mps
    from qenivo.model import ModelBuilder
    b = ModelBuilder("tie")                              # min -x - y, x + y <= 1: a whole edge is optimal
    b.var("x", 0, 1, obj=-1)
    b.var("y", 0, 1, obj=-1)
    b.row("cap", {"x": 1, "y": 1}, hi=1)
    model = tmp_path / "tie.mps"
    write_mps(b.build(), model)
    from qenivo.api import solve
    q = solve(str(model))
    other = {"x": 1.0 - q.x[0], "y": 1.0 - q.x[1]}       # the opposite vertex of the optimal edge
    (tmp_path / "tie.mps.answer.json").write_text(json.dumps(
        {"status": "optimal", "objective": -1.0, "x": other, "y": {"cap": -1.0}}))
    script = tmp_path / "fixed.py"
    script.write_text(FIXED_ANSWER)
    rep = sh.shadow(str(model), _cmd(script), cert_path=str(tmp_path / "c.json"))
    assert np.max(np.abs(q.x - np.array([other["x"], other["y"]]))) > 0.5
    assert rep.agree, rep.text()
    assert any("alternative optimum" in n for n in rep.notes)


def test_external_failures_and_cli_exit_codes(tmp_path, capsys):
    bad = tmp_path / "boom.py"
    bad.write_text("import sys; sys.exit(7)")
    with pytest.raises(RuntimeError, match="exited with 7"):
        sh.run_external([sys.executable, str(bad)], str(AFIRO))
    junk = tmp_path / "junk.py"
    junk.write_text("print('not json')")
    with pytest.raises(RuntimeError, match="not readable JSON"):
        sh.run_external([sys.executable, str(junk)], str(AFIRO))
    with pytest.raises(RuntimeError, match="not found"):
        sh.run_external(["no_such_solver_binary_x3"], str(AFIRO))
    assert sh.main([str(AFIRO), "--external", f'"{sys.executable}" "{bad}"', "--cert", str(tmp_path / "c.json")]) == 3
    assert sh.main([str(AFIRO), "--external", ""]) == 2
    assert sh.main([str(tmp_path / "missing.mps"), "--external", "x"]) == 2
    capsys.readouterr()


def test_cli_report_file(tmp_path, capsys):
    from qenivo.api import solve
    good = solve(str(AFIRO))
    _, cn = good.problem.names()
    model, cmd = _fixed(tmp_path, {"status": "optimal", "objective": good.objective + 5,
                                   "x": dict(zip(cn, good.x.tolist()))})
    report = tmp_path / "diff.json"
    ext = " ".join(f'"{c}"' if " " in c or "\\" in c else c for c in cmd)
    code = sh.main([str(model), "--external", ext, "--report", str(report), "--cert", str(tmp_path / "c.json")])
    assert code == 1
    data = json.loads(report.read_text())
    assert data["schema"] == "qenivo.shadow/1" and data["agree"] is False
    assert "MISMATCH" in capsys.readouterr().out
