#!/usr/bin/env python3
"""Generate docs/site/benchmarks.md from bench/results/ (no invented numbers).

Run from the qenivo/ package root (directory that contains mkdocs.yml)::

    python docs/site/build_benchmarks.py

Every table cell that carries a measured value also lists the source filename.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

QENIVO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = QENIVO_ROOT / "bench" / "results"
OUT = Path(__file__).resolve().parent / "benchmarks.md"


def _rel(path: Path) -> str:
    try:
        return path.relative_to(QENIVO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _fmt(v: object) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if abs(v) >= 100 or abs(v) == 0:
            return f"{v:.4g}"
        return f"{v:.6g}"
    return str(v)


def _load_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def netlib_section(lines: list[str]) -> None:
    path = RESULTS / "netlib_native_20260930.json"
    if not path.is_file():
        candidates = sorted(
            RESULTS.glob("netlib_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        path = next((p for p in candidates if "infeas" not in p.name), path)
    src = _rel(path)
    lines.append("## Netlib LP (headline counts)")
    lines.append("")
    data = _load_json(path)
    if not isinstance(data, dict) or "runs" not in data:
        lines.append(f"_Could not parse `{src}`._")
        lines.append("")
        return
    runs = data["runs"]
    if not isinstance(runs, list):
        lines.append(f"_Unexpected shape in `{src}`._")
        lines.append("")
        return
    verdicts: Counter[str] = Counter()
    engines: Counter[str] = Counter()
    verified = 0
    for r in runs:
        if not isinstance(r, dict):
            continue
        n = r.get("qenivo") if isinstance(r.get("qenivo"), dict) else r
        v = n.get("verdict") or n.get("status") or "unknown"
        verdicts[str(v)] += 1
        eng = n.get("engine")
        engines[str(eng) if eng is not None else "unknown"] += 1
        if n.get("verified") is True:
            verified += 1
    lines.append("| metric | value | source |")
    lines.append("|---|---|---|")
    lines.append(f"| instances in file | {len(runs)} | `{src}` |")
    lines.append(f"| verified true | {verified} | `{src}` |")
    for k, c in sorted(verdicts.items()):
        lines.append(f"| verdict `{k}` | {c} | `{src}` |")
    for k, c in sorted(engines.items()):
        lines.append(f"| engine `{k}` | {c} | `{src}` |")
    lines.append("")

    infeas = RESULTS / "netlib_infeas_20260930.json"
    if infeas.is_file():
        isrc = _rel(infeas)
        idata = _load_json(infeas)
        iruns: list = []
        if isinstance(idata, dict):
            iruns = idata.get("runs") or []
        elif isinstance(idata, list):
            iruns = idata
        iv: Counter[str] = Counter()
        for r in iruns:
            if not isinstance(r, dict):
                continue
            n = r.get("qenivo") if isinstance(r.get("qenivo"), dict) else r
            iv[str(n.get("verdict") or n.get("status") or "unknown")] += 1
        lines.append("### Netlib infeasible set")
        lines.append("")
        lines.append("| metric | value | source |")
        lines.append("|---|---|---|")
        lines.append(f"| instances in file | {len(iruns)} | `{isrc}` |")
        for k, c in sorted(iv.items()):
            lines.append(f"| verdict `{k}` | {c} | `{isrc}` |")
        lines.append("")


def e2_section(lines: list[str]) -> None:
    path = RESULTS / "a100_evidence_run1" / "e2_large_lp.jsonl"
    src = _rel(path)
    lines.append("## Large LPs vs cuPDLPx (e2)")
    lines.append("")
    if not path.is_file():
        lines.append(f"_Missing `{src}`._")
        lines.append("")
        return
    rows = _load_jsonl(path)
    lines.append(
        "| instance | tol | our time_median (s) | cuPDLPx wall (s) | status | certified | source |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        if "tol" not in r or "time_median" not in r:
            continue
        cup = r.get("cupdlpx") if isinstance(r.get("cupdlpx"), dict) else {}
        inst = _fmt(r.get("instance") or r.get("key"))
        lines.append(
            f"| {inst} | {_fmt(r.get('tol'))} | {_fmt(r.get('time_median'))} | "
            f"{_fmt(cup.get('wall'))} | {_fmt(r.get('status'))} | {_fmt(r.get('certified'))} | `{src}` |"
        )
    lines.append("")


def e3_section(lines: list[str]) -> None:
    path = RESULTS / "a100_evidence_run1" / "e3_batch.jsonl"
    src = _rel(path)
    lines.append("## What-if batch speedups (e3)")
    lines.append("")
    if not path.is_file():
        lines.append(f"_Missing `{src}`._")
        lines.append("")
        return
    rows = _load_jsonl(path)
    with_speed = [r for r in rows if r.get("speedup_vs_highs_all_cores") is not None]
    if len(with_speed) > 40:
        by_level: dict[int, list[dict]] = {}
        for r in with_speed:
            by_level.setdefault(int(r.get("level") or 0), []).append(r)
        keep: list[dict] = []
        for _lvl, group in sorted(by_level.items()):
            group = sorted(group, key=lambda r: int(r.get("S") or 0))
            keep.extend(group[:3])
            if len(group) > 6:
                mid = len(group) // 2
                keep.append(group[mid])
            keep.extend(group[-3:])
        seen: set[str] = set()
        uniq: list[dict] = []
        for r in keep:
            k = str(r.get("key") or id(r))
            if k in seen:
                continue
            seen.add(k)
            uniq.append(r)
        with_speed = uniq
    lines.append(
        "| key | level | S | tol | gpu_median (s) | HiGHS all-cores (s) | speedup | certified | source |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in with_speed:
        lines.append(
            f"| {_fmt(r.get('key'))} | {_fmt(r.get('level'))} | {_fmt(r.get('S'))} | "
            f"{_fmt(r.get('tol'))} | {_fmt(r.get('gpu_median'))} | "
            f"{_fmt(r.get('highs_parallel_time'))} | {_fmt(r.get('speedup_vs_highs_all_cores'))} | "
            f"{_fmt(r.get('certified'))} | `{src}` |"
        )
    lines.append("")
    lines.append(
        f"_Rows without a measured `speedup_vs_highs_all_cores` are omitted; "
        f"{len(rows)} total records in `{src}`._"
    )
    lines.append("")


def e4_section(lines: list[str]) -> None:
    path = RESULTS / "a100_evidence_run1" / "e4_refinery.jsonl"
    src = _rel(path)
    lines.append("## Refinery cases (e4)")
    lines.append("")
    if not path.is_file():
        lines.append(f"_Missing `{src}`._")
        lines.append("")
        return
    rows = _load_jsonl(path)
    lines.append(
        "| key | case | engine | objective | feasible | time (s) | BARON best found | BARON bound | source |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {_fmt(r.get('key'))} | {_fmt(r.get('case'))} | {_fmt(r.get('engine'))} | "
            f"{_fmt(r.get('objective'))} | {_fmt(r.get('feasible'))} | {_fmt(r.get('time'))} | "
            f"{_fmt(r.get('baron_best_found'))} | {_fmt(r.get('baron_best_possible'))} | `{src}` |"
        )
    lines.append("")


def root_cuts_section(lines: list[str]) -> None:
    path = RESULTS / "root_cuts.jsonl"
    src = _rel(path)
    lines.append("## Root cuts")
    lines.append("")
    if not path.is_file():
        lines.append(f"_No `{src}` present in this tree; section omitted._")
        lines.append("")
        return
    rows = _load_jsonl(path)
    if not rows:
        lines.append(f"_File `{src}` is empty or unparsable._")
        lines.append("")
        return
    keys = list(rows[0].keys())
    preferred = [
        "instance",
        "name",
        "gap_closed",
        "root_gap_closed",
        "gap_closed_pct",
        "cuts",
        "n_cuts",
        "time",
        "time_s",
    ]
    cols = [k for k in preferred if k in keys] or keys[:8]
    lines.append("| " + " | ".join(cols + ["source"]) + " |")
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for r in rows:
        cells = [_fmt(r.get(c)) for c in cols]
        lines.append("| " + " | ".join(cells + [f"`{src}`"]) + " |")
    lines.append("")


def native_simplex_section(lines: list[str]) -> None:
    path = RESULTS / "native_simplex_netlib.jsonl"
    src = _rel(path)
    lines.append("## Native simplex vs Python / HiGHS (sample)")
    lines.append("")
    if not path.is_file():
        lines.append(f"_Missing `{src}`._")
        lines.append("")
        return
    rows = _load_jsonl(path)
    lines.append("| metric | value | source |")
    lines.append("|---|---|---|")
    lines.append(f"| records | {len(rows)} | `{src}` |")
    lines.append("")
    if not rows:
        return
    sample = rows[:15]
    cols: list[str] = []
    for cand in (
        "instance",
        "name",
        "native_status",
        "native_time",
        "python_time",
        "highs_time",
        "time_native",
        "time_python",
        "time_highs",
        "speedup",
        "status",
    ):
        if any(cand in r for r in sample):
            cols.append(cand)
    if not cols:
        cols = list(sample[0].keys())[:6]
    lines.append("| " + " | ".join(cols + ["source"]) + " |")
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for r in sample:
        cells = [_fmt(r.get(c)) for c in cols]
        lines.append("| " + " | ".join(cells + [f"`{src}`"]) + " |")
    if len(rows) > len(sample):
        lines.append("")
        lines.append(f"_Showing {len(sample)} of {len(rows)} rows from `{src}`._")
    lines.append("")


def inventory_section(lines: list[str]) -> None:
    lines.append("## Results inventory")
    lines.append("")
    lines.append("JSON / JSONL / markdown artefacts under `bench/results/` (one subdir deep):")
    lines.append("")
    lines.append("| path | bytes |")
    lines.append("|---|---|")
    paths = list(RESULTS.glob("*"))
    a100 = RESULTS / "a100_evidence_run1"
    if a100.is_dir():
        paths.extend(a100.glob("*"))
    for p in sorted(paths, key=lambda x: x.as_posix().lower()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".json", ".jsonl", ".md"}:
            continue
        lines.append(f"| `{_rel(p)}` | {p.stat().st_size} |")
    lines.append("")


def main() -> None:
    lines: list[str] = [
        "# Benchmarks",
        "",
        "Generated by `docs/site/build_benchmarks.py` from files under `bench/results/`.",
        "Every number below cites its source path. No invented figures.",
        "",
        "Regenerate:",
        "",
        "```bash",
        "python docs/site/build_benchmarks.py",
        "```",
        "",
        "Declared sets live in `bench/sets.py`. Narrative of published measurements:",
        "`docs/EVIDENCE.md` (package docs, not rewritten here).",
        "",
    ]
    netlib_section(lines)
    e2_section(lines)
    e3_section(lines)
    e4_section(lines)
    root_cuts_section(lines)
    native_simplex_section(lines)
    inventory_section(lines)
    lines.append("## Protocol")
    lines.append("")
    lines.append("1. Record git commit and machine in the results file (the runner writes `meta`).")
    lines.append("2. Keep failures and time-outs in the file.")
    lines.append("3. Use HiGHS only as an optional comparator, never on the solve path.")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT.relative_to(QENIVO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
