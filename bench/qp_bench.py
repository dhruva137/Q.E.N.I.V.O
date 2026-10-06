"""Maros-Meszaros convex QP bench through ipm-native (default QP route).

    python bench/qp_bench.py --time-limit 120
    python bench/qp_bench.py --limit 10 --time-limit 120          # smoke
    python bench/qp_bench.py --resume bench/results/maros_native_ipm_2026-10-01.jsonl

Every instance prints ``[{i}/138] name status time`` (flushed). Results append as JSONL
with a HiGHS column when highspy is importable. An answer is never counted as agreeing
unless the independent KKT certificate passes (verdict == optimal and verified).
Wrong answers (KKT-optimal but far from the published optimum) are flagged separately.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

# Single-thread BLAS / OpenMP for reproducible wall times and to leave cores for others.
for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
):
    os.environ.setdefault(_k, _v)

WORKER_QENIVO = r"""
import json, os, sys, time
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, "1")
sys.path.insert(0, {src!r})
from qenivo.provenance import install_tripwires, guard
install_tripwires()
import qenivo
from qenivo.certify.certificate import load
from qenivo.certify.verify import verify
path, engine, tol, tl, cert = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
t0 = time.perf_counter()
prob = qenivo.read(path)
t_read = time.perf_counter() - t0
assert prob.is_qp, "expected a convex QP (QUADOBJ/QMATRIX present)"
print(f"PROGRESS stage=solve engine={{engine}} rows={{prob.m}} cols={{prob.n}} tl={{tl}}", flush=True)
sol = qenivo.solve(prob, engine=engine, tol=tol, time_limit=tl, backend="numpy", verbose=True)
t_solve = time.perf_counter() - t0 - t_read
sol.source = path
sol.save_certificate(cert)
v = verify(path, load(cert))
g = guard()
res = sol.residuals or {{}}
print(f"PROGRESS stage=done status={{sol.status}} verdict={{sol.verdict}} "
      f"engine={{sol.engine.get('engine')}} it={{sol.engine.get('iterations')}} "
      f"t={{t_solve:.2f}}s", flush=True)
print("RESULT " + json.dumps({{
    "rows": prob.m, "cols": prob.n, "nnz": int(prob.nnz),
    "q_nnz": int(prob.Q.nnz) if prob.Q is not None else 0,
    "read_time": t_read, "solve_time": t_solve,
    "verdict": sol.verdict, "status": sol.status, "objective": sol.objective,
    "engine": sol.engine.get("engine"), "backend": sol.engine.get("backend"),
    "reason": sol.engine.get("reason"), "iterations": sol.engine.get("iterations"),
    "residuals": {{k: res.get(k) for k in
                  ("rel_primal","rel_dual","rel_gap","max_rel","objective","dual_objective")}},
    "kkt_certified": bool(sol.verdict == "optimal" and res.get("max_rel") is not None
                          and res["max_rel"] <= tol * 1.000001),
    "verified": v["passed"],
    "verify_checks": [[c[0], c[1], c[2]] for c in v["checks"]],
    "provenance_clean": g["clean"],
}}, default=str), flush=True)
"""

WORKER_HIGHS = r"""
import json, os, sys, time
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import highspy
h = highspy.Highs()
h.setOptionValue("output_flag", False)
h.setOptionValue("time_limit", float(sys.argv[2]))
h.setOptionValue("threads", 1)
# Convex QP: let HiGHS pick; IPM is the usual QP path in recent highspy.
try:
    h.setOptionValue("solver", "choose")
except Exception:
    pass
t0 = time.perf_counter()
rc = h.readModel(sys.argv[1])
if rc != highspy.HighsStatus.kOk:
    print("RESULT " + json.dumps({"status": "read_error", "objective": None, "time": 0.0,
                                  "error": str(rc)}))
    raise SystemExit(0)
h.run()
t = time.perf_counter() - t0
st = h.modelStatusToString(h.getModelStatus())
info = h.getInfo()
obj = None
try:
    if st in ("Optimal", "ObjectiveBound"):
        obj = float(info.objective_function_value)
except Exception:
    obj = None
print("RESULT " + json.dumps({"status": st, "objective": obj, "time": t}))
"""


def _run(code: str, args: list, wall: float) -> dict:
    """Run worker; stream PROGRESS/epoch lines live so long solves are not a blind wait."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        p = subprocess.Popen(
            [sys.executable, "-u", "-c", code, *map(str, args)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT),
            env=env,
            bufsize=1,
        )
    except OSError as exc:
        return {"error": f"spawn failed: {exc!r}"[:400], "status": "error"}

    deadline = time.perf_counter() + wall
    result = None
    last_lines: list[str] = []
    assert p.stdout is not None
    while True:
        if time.perf_counter() > deadline:
            p.kill()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return {"error": f"killed at hard wall {wall:.0f}s", "status": "time_limit"}
        line = p.stdout.readline()
        if not line:
            if p.poll() is not None:
                break
            time.sleep(0.05)
            continue
        line = line.rstrip("\r\n")
        if not line:
            continue
        last_lines.append(line)
        if len(last_lines) > 40:
            last_lines = last_lines[-40:]
        if line.startswith("RESULT "):
            try:
                result = json.loads(line[7:])
            except json.JSONDecodeError as exc:
                return {"error": f"bad RESULT: {exc}", "status": "error"}
        elif line.startswith("PROGRESS ") or line.startswith("STAGE "):
            print(f"    {line}", flush=True)
        # else: keep quiet (tripwire noise); last_lines retained for errors

    if result is not None:
        return result
    err = (last_lines or ["no output"])[-1]
    return {"error": err[:400], "status": "error"}


def _highs_available() -> bool:
    r = subprocess.run(
        [sys.executable, "-c", "import highspy"],
        capture_output=True,
        cwd=str(ROOT),
    )
    return r.returncode == 0


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _strip_bom(path: Path) -> None:
    """Rewrite JSONL without a UTF-8 BOM (PowerShell/editors sometimes inject one)."""
    if not path.exists():
        return
    raw = path.read_bytes()
    if not raw.startswith(b"\xef\xbb\xbf"):
        return
    path.write_bytes(raw.lstrip(b"\xef\xbb\xbf"))


def _load_done(path: Path) -> dict[str, dict]:
    done = {}
    if not path.exists():
        return done
    _strip_bom(path)
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = rec.get("instance")
            if name and "error" not in rec.get("qenivo", {}) and rec.get("qenivo", {}).get("verdict"):
                done[name] = rec
            elif name and rec.get("qenivo", {}).get("status") not in (None, "error"):
                done[name] = rec
    return done


def _agree(rec: dict, published_tol: float = 1e-6) -> bool:
    """Agree with published optimum only when KKT-certified and verified."""
    q = rec.get("qenivo") or {}
    if not q.get("kkt_certified") or not q.get("verified"):
        return False
    if q.get("verdict") != "optimal":
        return False
    rel = rec.get("rel_obj_diff_published")
    return rel is not None and rel <= published_tol


def _wrong(rec: dict, published_tol: float = 1e-4) -> bool:
    """KKT-certified 'optimal' but objective far from the published value."""
    q = rec.get("qenivo") or {}
    if not q.get("kkt_certified") or q.get("verdict") != "optimal":
        return False
    pub = rec.get("published_objective")
    rel = rec.get("rel_obj_diff_published")
    if pub is None or rel is None:
        return False
    return rel > published_tol


def run_one(name: str, a, highs_ok: bool) -> dict:
    from fetch import fetch, maros_key, maros_references

    rec = {
        "instance": name,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "commit": _git_head(),
    }
    try:
        path = fetch(name, verbose=False)
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"fetch failed: {exc!r}"[:300]
        rec["qenivo"] = {"status": "fetch_error", "verdict": "not_proven"}
        return rec

    code = WORKER_QENIVO.format(src=str(ROOT / "src"))
    with tempfile.TemporaryDirectory() as td:
        cert = Path(td) / f"{name}.cert.json"
        t0 = time.perf_counter()
        q = _run(code, [path, a.engine, a.tol, a.time_limit, cert], a.time_limit * 1.5 + 90)
        q["wall"] = time.perf_counter() - t0
        rec["qenivo"] = q
        # Never claim optimal without a KKT certificate on the original model.
        if q.get("verdict") == "optimal" and not q.get("kkt_certified"):
            q["verdict"] = "not_proven"
            q["note"] = "engine claimed optimal without passing KKT; demoted"

    if highs_ok and not a.no_ref:
        rec["highs"] = _run(WORKER_HIGHS, [path, a.time_limit], a.time_limit * 1.5 + 60)
    else:
        rec["highs"] = {"status": "skipped", "objective": None, "time": None}

    pub = maros_references().get(maros_key(name))
    rec["published_objective"] = pub
    obj = (rec.get("qenivo") or {}).get("objective")
    if pub is not None and obj is not None:
        rec["rel_obj_diff_published"] = abs(obj - pub) / (1.0 + abs(pub))
    hobj = (rec.get("highs") or {}).get("objective")
    if obj is not None and hobj is not None:
        rec["rel_obj_diff_highs"] = abs(obj - hobj) / (1.0 + abs(hobj))
    rec["agree_published"] = _agree(rec)
    rec["wrong_answer"] = _wrong(rec)
    return rec


def summarise(recs: list[dict], total: int) -> str:
    agree = sum(1 for r in recs if r.get("agree_published"))
    wrong = [r["instance"] for r in recs if r.get("wrong_answer")]
    kkt = sum(1 for r in recs if (r.get("qenivo") or {}).get("kkt_certified"))
    verified = sum(1 for r in recs if (r.get("qenivo") or {}).get("verified"))
    highs_ok = sum(1 for r in recs if (r.get("highs") or {}).get("status") == "Optimal")
    lines = [
        f"Maros QP ipm-native: {len(recs)}/{total} recorded",
        f"  KKT-certified optimal:     {kkt}",
        f"  independently verified:    {verified}",
        f"  agree published (rel<=1e-6, cert+verify): {agree}/{total}",
        f"  HiGHS Optimal:             {highs_ok}",
        f"  wrong answers:             {len(wrong)}" + (f"  {wrong}" if wrong else ""),
    ]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default="auto", help="auto routes QP -> ipm-native")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N names (smoke)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--no-ref", action="store_true", help="skip HiGHS column")
    ap.add_argument(
        "--defer",
        default="BOYD2,CONT-300",
        help="comma-separated names to run last (large / high-RAM)",
    )
    ap.add_argument(
        "--out",
        default=str(HERE / "results" / "maros_native_ipm_2026-10-01.jsonl"),
    )
    ap.add_argument("--resume", default=None, help="alias for --out (resume JSONL)")
    a = ap.parse_args(argv)

    from sets import MAROS

    names = list(MAROS)
    defer = {n.strip() for n in (a.defer or "").split(",") if n.strip()}
    if a.offset:
        names = names[a.offset :]
    if a.limit and a.limit > 0:
        names = names[: a.limit]
    # Run deferred (RAM-heavy) instances after the rest.
    names = [n for n in names if n not in defer] + [n for n in names if n in defer]
    total_set = len(MAROS)
    out = Path(a.resume or a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = _load_done(out)
    todo = [n for n in names if n not in done]
    print(
        f"qp_bench: {len(done)} done, {len(todo)} to run, out={out}, "
        f"engine={a.engine}, tl={a.time_limit}s, threads=1",
        flush=True,
    )
    highs_ok = False if a.no_ref else _highs_available()
    print(f"HiGHS column: {'yes' if highs_ok else 'no'}", flush=True)

    # Header meta line once if file is new.
    if not out.exists():
        meta = {
            "meta": True,
            "suite": "maros_native_ipm",
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "commit": _git_head(),
            "tol": a.tol,
            "time_limit": a.time_limit,
            "engine": a.engine,
            "threads": 1,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "highs": highs_ok,
            "n_maros": total_set,
        }
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(meta) + "\n")

    index_map = {n: i + 1 for i, n in enumerate(MAROS)}
    recs = list(done.values())
    for name in todo:
        i = index_map.get(name, 0)
        t0 = time.perf_counter()
        rec = run_one(name, a, highs_ok)
        elapsed = time.perf_counter() - t0
        q = rec.get("qenivo") or {}
        status = q.get("verdict") or q.get("status") or rec.get("error") or "?"
        stime = q.get("solve_time")
        if stime is None:
            stime = elapsed
        tag = "AGREE" if rec.get("agree_published") else (
            "WRONG" if rec.get("wrong_answer") else status
        )
        print(f"[{i}/{total_set}] {name} {tag} {stime:.2f}s", flush=True)
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
        recs.append(rec)

    # Re-load full file for summary over all completed instances in this out file.
    all_recs = [r for r in _load_done(out).values()]
    print("\n" + summarise(all_recs, total_set), flush=True)
    print(f"results -> {out}", flush=True)
    wrong_n = sum(1 for r in all_recs if r.get("wrong_answer"))
    if wrong_n:
        sys.exit(2)


if __name__ == "__main__":
    main()
