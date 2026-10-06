"""Spreadsheet exchange for planners: case tables in, result tables out (CSV or .xlsx).

Planners work in Excel. A case table is one sheet in long form, the way a PIMS CASE table lists
changes to a base model:

    case        kind     name              value
    HighBrent   cost     buy_crude1        78.5
    HighBrent   cost     buy_crude2        81.0
    CDUdown     row_hi   cdu_capacity      36000
    Crude2Lim   col_hi   buy_crude2        12000

kind is one of cost, col_lo, col_hi, row_lo, row_hi; names are the model's column / row names.
The .xlsx reader and writer use only the Python standard library (an .xlsx file is a zip of XML),
so nothing extra has to be installed on an air-gapped planning desktop.
"""
from __future__ import annotations

import csv
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

KINDS = ("cost", "col_lo", "col_hi", "row_lo", "row_hi")
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


# ---------------------------------------------------------------------- generic rows
def read_rows(path) -> list[list]:
    """Read a CSV or .xlsx sheet. A malformed file raises ValueError."""
    path = Path(path)
    try:
        if path.suffix.lower() == ".xlsx":
            return _read_xlsx(path)
        with open(path, newline="", encoding="utf-8-sig") as f:
            return [row for row in csv.reader(f)]
    except ValueError:
        raise
    except (OSError, UnicodeError, csv.Error, zipfile.BadZipFile, ET.ParseError,
            IndexError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"unreadable table {path}: {exc}") from exc


def write_rows(path, rows: list[list]):
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        return _write_xlsx(path, rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def write_sheets(path, sheets: dict[str, list[list]]):
    """Write a multi-sheet .xlsx (name -> rows). CSV falls back to the first sheet."""
    path = Path(path)
    if path.suffix.lower() != ".xlsx":
        first = next(iter(sheets.values()), [])
        return write_rows(path, first)
    return _write_xlsx_multi(path, sheets)


# ---------------------------------------------------------------------- case tables
def read_case_table(path):
    """Read a case table (CSV or .xlsx) into a list of workload.cases.Case, in order of appearance."""
    from ..workload.cases import Case
    rows = [r for r in read_rows(path) if any(str(c).strip() for c in r)]
    if not rows:
        return []
    head = [str(c).strip().lower() for c in rows[0]]
    need = ["case", "kind", "name", "value"]
    if head[:4] != need:
        raise ValueError(f"case table header must be {need}, got {head[:4]}")
    cases: dict = {}
    for k, r in enumerate(rows[1:], start=2):
        case, kind, name, value = (str(c).strip() for c in (list(r) + ["", "", "", ""])[:4])
        if kind not in KINDS:
            raise ValueError(f"row {k}: kind {kind!r} is not one of {KINDS}")
        c = cases.setdefault(case, Case(name=case))
        getattr(c, kind)[name] = float(value)
    return list(cases.values())


def write_case_report(path, report):
    """Write a case-stack result table (workload.cases report) as CSV or .xlsx."""
    table = report.table()
    cols = list(table[0].keys()) if table else ["case"]
    write_rows(path, [cols] + [[d.get(c) for c in cols] for d in table])


# ---------------------------------------------------------------------- minimal xlsx
def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _read_xlsx(path: Path) -> list[list]:
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
        sheets = sorted(n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n))
        root = ET.fromstring(z.read(sheets[0]))
    rows = []
    for row in root.iter(f"{{{_NS['m']}}}row"):
        out = []
        for c in row.findall("m:c", _NS):
            idx = _col_index(c.get("r")) if c.get("r") else len(out)
            while len(out) < idx:
                out.append("")
            t = c.get("t")
            if t == "s":
                out.append(shared[int(c.find("m:v", _NS).text)])
            elif t == "inlineStr":
                out.append("".join(x.text or "" for x in c.iter(f"{{{_NS['m']}}}t")))
            else:
                v = c.find("m:v", _NS)
                out.append("" if v is None else v.text)
        rows.append(out)
    return rows


def _cell(ref, v):
    if v is None:
        return f'<c r="{ref}"/>'
    if isinstance(v, bool):
        return f'<c r="{ref}" t="b"><v>{int(v)}</v></c>'
    if isinstance(v, (int, float)) and v == v and abs(v) != float("inf"):
        return f'<c r="{ref}"><v>{repr(float(v)) if isinstance(v, float) else v}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(v))}</t></is></c>'


def _ref(r, c):
    s = ""
    c += 1
    while c:
        c, rem = divmod(c - 1, 26)
        s = chr(65 + rem) + s
    return f"{s}{r + 1}"


def _sheet_xml(rows: list[list]) -> str:
    body = "".join(f'<row r="{i + 1}">' + "".join(_cell(_ref(i, j), v) for j, v in enumerate(r)) + "</row>"
                   for i, r in enumerate(rows))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<worksheet xmlns="{_NS["m"]}"><sheetData>{body}</sheetData></worksheet>')


def _safe_sheet_name(name: str, used: set[str]) -> str:
    base = re.sub(r'[:\\/?*\[\]]', "_", str(name or "Sheet"))[:31] or "Sheet"
    cand, k = base, 1
    while cand in used:
        suffix = f"_{k}"
        cand = (base[: 31 - len(suffix)] + suffix)
        k += 1
    used.add(cand)
    return cand


def _write_xlsx(path: Path, rows: list[list]):
    return _write_xlsx_multi(path, {"QENIVO": rows})


def _write_xlsx_multi(path: Path, sheets: dict[str, list[list]]):
    if not sheets:
        sheets = {"QENIVO": []}
    used: set[str] = set()
    items = [(_safe_sheet_name(name, used), rows) for name, rows in sheets.items()]
    overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    ]
    sheet_tags = []
    rels = []
    files: dict[str, str] = {}
    for i, (name, rows) in enumerate(items, start=1):
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
        sheet_tags.append(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>')
        rels.append(
            f'<Relationship Id="rId{i}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{i}.xml"/>'
        )
        files[f"xl/worksheets/sheet{i}.xml"] = _sheet_xml(rows)
    files["[Content_Types].xml"] = (
        '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(overrides) + "</Types>"
    )
    files["_rels/.rels"] = (
        '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    files["xl/workbook.xml"] = (
        f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{_NS["m"]}" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(sheet_tags)}</sheets></workbook>'
    )
    files["xl/_rels/workbook.xml.rels"] = (
        '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels) + "</Relationships>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in files.items():
            z.writestr(name, text)
