"""Fuzz the readers that exist on this branch. A crash that is not a clear error fails.

B. P. Miller, L. Fredriksen and B. So, CACM 1990. LP and JSON readers are not
in this tree; MPS, GAMS scalar and the spreadsheet reader are.
"""
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from qenivo.io.gams import read_gams
from qenivo.io.mps import read_mps
from qenivo.io.sheets import read_case_table, read_rows
from qenivo.model import Problem

CLEAR = (ValueError, OSError, UnicodeError, NotImplementedError)
_fast = settings(max_examples=25, deadline=None, derandomize=True,
               suppress_health_check=[HealthCheck.function_scoped_fixture])


def _accept(fn, path):
    try:
        return fn(path)
    except CLEAR:
        return None


@_fast
@given(st.text(max_size=240))
def test_mps_random_text(tmp_path, text):
    path = tmp_path / "fuzz.mps"
    path.write_text(text, encoding="utf-8")
    parsed = _accept(read_mps, path)
    if parsed is not None:
        assert isinstance(parsed, Problem)


@_fast
@given(st.text(max_size=240))
def test_gams_random_text(tmp_path, text):
    path = tmp_path / "fuzz.gms"
    path.write_text(text, encoding="utf-8")
    _accept(read_gams, path)


@_fast
@given(st.binary(max_size=180))
def test_xlsx_random_bytes(tmp_path, blob):
    path = tmp_path / "fuzz.xlsx"
    path.write_bytes(blob)
    parsed = _accept(read_rows, path)
    if parsed is not None:
        assert isinstance(parsed, list)


@pytest.mark.parametrize("text", [
    "",
    "ROWS\n E\n",
    "NAME X\nBOUNDS\n UP\nENDATA\n",
    "NAME X\nROWS\n E R1\nCOLUMNS\n C R1 1\nQUADOBJ\n NOPE ALSO 1\nENDATA\n",
    "NAME X\nSOS\nENDATA\n",
    "\x00\x01not mps",
])
def test_mps_malformed_is_a_clear_error_or_a_model(tmp_path, text):
    path = tmp_path / "bad.mps"
    path.write_text(text, encoding="utf-8")
    parsed = _accept(read_mps, path)
    if parsed is not None:
        assert isinstance(parsed, Problem)


def test_gams_unknown_variable_and_deep_nesting(tmp_path):
    unknown = tmp_path / "unknown.gms"
    unknown.write_text(
        "Variables x1;\nEquations e1;\ne1.. x2 =E= 1;\nSolve m using lp minimizing x1;\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="unknown variable"):
        read_gams(unknown)
    nested = tmp_path / "nested.gms"
    expr = "(" * 40 + "x1" + ")" * 40
    nested.write_text(
        f"Variables x1;\nEquations e1;\ne1.. {expr} =E= 1;\n"
        "Solve m using lp minimizing x1;\n",
        encoding="utf-8")
    parsed = read_gams(nested)
    assert parsed.linear.n == 1
    broken = tmp_path / "broken.gms"
    broken.write_text(
        "Variables x1;\nEquations e1;\ne1.. (x1 =E= 1;\nSolve m using lp minimizing x1;\n",
        encoding="utf-8")
    with pytest.raises(ValueError):
        read_gams(broken)


def test_case_table_bad_header_and_broken_xlsx(tmp_path):
    csv_path = tmp_path / "cases.csv"
    csv_path.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        read_case_table(csv_path)
    xlsx = tmp_path / "not.xlsx"
    xlsx.write_bytes(b"this is not a workbook")
    with pytest.raises(ValueError):
        read_rows(xlsx)
