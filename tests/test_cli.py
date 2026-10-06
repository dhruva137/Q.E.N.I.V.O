"""CLI packaging surface: doctor, engines, convert, completion, --json, exit codes."""
from __future__ import annotations

import json

import numpy as np
import pytest

from qenivo.cli import build_parser, main
from qenivo.model import ModelBuilder


def _run(argv):
    return main(list(argv))


def test_build_parser_has_new_commands():
    p = build_parser()
    # reach into subparsers
    subs = p._subparsers._group_actions[0].choices
    for name in ("bench", "convert", "presolve", "doctor", "engines", "completion",
                 "solve", "verify", "explain", "range", "crude-value", "cases", "recursion",
                 "serve", "model", "info", "demo"):
        assert name in subs


def test_doctor_json_exit():
    code = _run(["doctor", "--json"])
    assert code in (0, 1)


def test_doctor_global_json(capsys):
    code = _run(["--json", "doctor"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "checks" in data
    assert code in (0, 1)


def test_engines_lists_builtin_and_plugins(capsys):
    assert _run(["engines", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    names = {e["name"] for e in data["engines"]}
    assert "simplex" in names
    assert "pdhg" in names
    assert "miqp" in names  # plugin


def test_completion_shells(capsys):
    for shell in ("bash", "zsh", "powershell"):
        assert _run(["completion", shell]) == 0
        text = capsys.readouterr().out
        assert "qenivo" in text.lower() or "Register-ArgumentCompleter" in text


def test_convert_mps_to_json_and_back(tmp_path, capsys):
    b = ModelBuilder("tiny")
    b.var("x", 0, 1, obj=-1.0)
    b.row("r", {"x": 1}, hi=1)
    prob = b.build()
    from qenivo.io.mps import write_mps
    mps = tmp_path / "tiny.mps"
    js = tmp_path / "tiny.json"
    mps2 = tmp_path / "tiny2.mps"
    write_mps(prob, mps, name="tiny")
    assert _run(["convert", str(mps), str(js), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == 1 and payload["cols"] == 1
    assert js.is_file()
    assert _run(["convert", str(js), str(mps2)]) == 0
    assert mps2.is_file()


def test_convert_lp_exits_usage(tmp_path, capsys):
    b = ModelBuilder("tiny")
    b.var("x", 0, 1, obj=-1.0)
    b.row("r", {"x": 1}, hi=1)
    from qenivo.io.mps import write_mps
    mps = tmp_path / "tiny.mps"
    write_mps(b.build(), mps, name="tiny")
    code = _run(["convert", str(mps), str(tmp_path / "out.lp")])
    err = capsys.readouterr().err
    assert code == 2
    assert "LP format" in err or "lpformat" in err.lower() or "not available" in err.lower()


def test_presolve_report_json(tmp_path, capsys):
    b = ModelBuilder("p")
    b.var("x", 0, 0, obj=1.0)  # fixed column — presolve should remove
    b.var("y", 0, 5, obj=-1.0)
    b.row("r", {"x": 1, "y": 1}, hi=5)
    from qenivo.io.mps import write_mps
    mps = tmp_path / "p.mps"
    out = tmp_path / "p_red.mps"
    write_mps(b.build(), mps, name="p")
    code = _run(["presolve", str(mps), str(out), "--report", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert code in (0, 1)
    assert "log" in data
    assert data["original"]["cols"] == 2


def test_json_format_roundtrip():
    from qenivo.benchmarks.json_format import problem_from_json, problem_to_json
    b = ModelBuilder("j")
    b.var("x", 0, np.inf, obj=1.0)
    b.var("z", 0, 1, obj=0.0, integer=True)
    b.row("eq", {"x": 1, "z": 1}, 2, 2)
    p = b.build()
    p2 = problem_from_json(problem_to_json(p))
    assert p2.m == p.m and p2.n == p.n
    assert p2.is_mip
    np.testing.assert_allclose(p2.c, p.c)


def test_available_sets():
    from qenivo.benchmarks import available_sets
    sets = available_sets()
    assert "small" in sets
    assert "netlib" in sets


def test_bench_unknown_set():
    code = _run(["bench", "no_such_set_xyz", "--time-limit", "1", "--json"])
    assert code == 2


@pytest.mark.network
def test_bench_small_smoke(tmp_path):
    """Optional network fetch; skipped in default CI unless marker selected with net."""
    out = tmp_path / "small.json"
    code = _run(["bench", "small", "--time-limit", "5", "--out", str(out), "--json"])
    assert code == 0
    assert out.is_file()


def test_info_json(capsys):
    assert _run(["info", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "qenivo" in data
