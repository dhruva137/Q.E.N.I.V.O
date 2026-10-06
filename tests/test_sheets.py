"""Case tables from Excel / CSV and result tables back, with the standard library only."""
from qenivo.io.sheets import read_case_table, read_rows, write_case_report, write_rows
from qenivo.models.williams import williams_lp
from qenivo.workload.cases import solve_cases


def _table(p):
    cn = p.col_names
    return [["case", "kind", "name", "value"],
            ["base", "cost", cn[0], p.obj_sign * p.c[0]],
            ["dear", "cost", cn[0], p.obj_sign * p.c[0] * 1.5],
            ["dear", "col_hi", cn[1], 1000.0]]


def test_xlsx_roundtrip_and_case_stack(tmp_path):
    p = williams_lp()
    f = tmp_path / "cases.xlsx"
    write_rows(f, _table(p))
    assert [str(x) for x in read_rows(f)[0]] == ["case", "kind", "name", "value"]
    cases = read_case_table(f)
    assert [c.name for c in cases] == ["base", "dear"]
    rep = solve_cases(p, cases, backend="numpy")
    assert all(s.verdict == "optimal" for s in rep.cases)
    out = tmp_path / "result.xlsx"
    write_case_report(out, rep)
    rows = read_rows(out)
    assert rows[0][0] == "case" and rows[1][0] == "base" and rows[2][1] == "optimal"


def test_csv_case_table(tmp_path):
    p = williams_lp()
    f = tmp_path / "cases.csv"
    write_rows(f, _table(p))
    assert len(read_case_table(f)) == 2
