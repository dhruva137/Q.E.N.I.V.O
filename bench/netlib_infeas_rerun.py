"""W09 Netlib infeasible re-run: verified Farkas rays only (never bare 'infeasible').

Farkas path inventory (dual simplex phase 1 + certify)
------------------------------------------------------
1. Native dual (src/qenivo/native/simplex_core.cpp):
   - solve() when primal-infeasible: dual_phase1() on an auxiliary box problem, then dual().
   - dual() with no entering column (q < 0) returns S_INFEASIBLE — that is the unbounded dual /
     Farkas ray case; y_out is written as the current pi, but the Python wrapper does not yet
     pass that vector as an evidence ray into _finish.
   - Fallback: primal(phase1=true); if residual infeas > 1e-6*(1+m) -> S_INFEASIBLE (status only).

2. Python revised simplex (engines/simplex.py):
   - _run_dual: no repairable leaving row -> "infeasible" (dual ray / Farkas).
   - Cold dual currently demotes that status and falls back to composite primal phase 1;
     warm dual trusts the dual-ray infeasible. Phase-1 residual > tol -> "infeasible".

3. Certify (api.py + workload/explain.py + certify/kkt.py):
   - Engine "infeasible" without a ray -> verdict stays not_proven.
   - _certify_infeasible builds an elastic phase-1 LP; its row duals are the Farkas candidate;
     check_farkas(prob, y) must pass (phi>0, sign rules) or the verdict is not_proven.
   - Solution.save_certificate writes evidence_kind=farkas only for verdict=infeasible.

4. Independent re-check (certify/verify.py, optional engines/exact_lp.py):
   - verify() re-scores the saved ray against a fresh MPS parse.
   - For small models this runner also asks exact_lp to rational-recheck the float ray when cheap.

Usage
-----
    python bench/netlib_infeas_rerun.py                  # RAM gate + smoke or full
    python bench/netlib_infeas_rerun.py --force-full     # ignore RAM gate (parent resume)
    python bench/netlib_infeas_rerun.py --smoke-only     # always 3 easy models
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

from sets import NETLIB_INFEAS  # noqa: E402

# Known easy infeasibles (small, previously verified Farkas on 2026-09-30 run).
SMOKE = ["bgprtr", "itest2", "woodinfe"]

# Failures from bench/results/netlib_infeas_20260930.json (24/29 proven).
OLD_FAILURES = ["cplex2", "gosh", "gran", "greenbea", "pang"]
# Kaggle-era names from W09 brief (for cross-reference in meta).
KAGGLE_ERA_FAILURES = ["bgindy", "cplex2", "gran", "greenbea-infeas", "pang", "refinery"]

MIN_FREE_GB = 2.0
DEFAULT_OUT = HERE / "results" / "netlib_infeas_2026-10-01.jsonl"

WORKER = r"""
import json, sys, time
sys.path.insert(0, {src!r})
from qenivo.provenance import install_tripwires, guard
install_tripwires()
import qenivo
from qenivo.certify.certificate import load
from qenivo.certify.verify import verify
from qenivo.certify.kkt import check_farkas
path, engine, tol, tl, cert = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
t0 = time.perf_counter()
prob = qenivo.read(path)
t_read = time.perf_counter() - t0
sol = qenivo.solve(prob, engine=engine, tol=tol, time_limit=tl, backend="numpy")
t_solve = time.perf_counter() - t0 - t_read
sol.source = path
sol.save_certificate(cert)
v = verify(path, load(cert))
g = guard()
# Independent host re-score of any ray the solution carried (even if verify failed).
host = None
if sol.ray is not None:
    host = check_farkas(prob, sol.ray)
exact = None
# Cheap rational re-check only for tiny models (exact_lp is heavy).
if sol.verdict == "infeasible" and sol.ray is not None and prob.m <= 80 and prob.n <= 120:
    try:
        from qenivo.engines.exact_lp import _ExactSolver, rat_from_float
        elp = _ExactSolver(prob, arithmetic="python", candidate=False, time_limit=30.0)
        guess = [rat_from_float(float(x)) for x in sol.ray]
        phi, viol = elp._farkas_of(guess)
        if not (phi.is_positive() and viol.is_zero()):
            flipped = [-g for g in guess]
            phi2, viol2 = elp._farkas_of(flipped)
            if phi2.is_positive() and viol2.is_zero():
                phi, viol = phi2, viol2
        exact = {{"phi_positive": bool(phi.is_positive()), "viol_zero": bool(viol.is_zero()),
                 "ok": bool(phi.is_positive() and viol.is_zero()),
                 "phi": str(phi), "viol": str(viol)}}
    except Exception as e:  # noqa: BLE001
        exact = {{"error": repr(e)[:200]}}
# Never promote to infeasible without a verified ray (host + independent verify).
# Exact re-check is recorded when cheap; float->Rat conversion may leave a tiny viol,
# so it does not demote a host-verified ray on its own.
verified_ray = bool(sol.verdict == "infeasible" and sol.ray is not None
                    and v.get("passed") and (host is None or host.get("valid")))
reported = "infeasible" if verified_ray else "not_proven"
print("RESULT " + json.dumps({{
    "rows": prob.m, "cols": prob.n, "nnz": prob.nnz, "read_time": t_read, "solve_time": t_solve,
    "engine_status": sol.status, "engine_verdict": sol.verdict, "reported": reported,
    "objective": sol.objective, "engine": sol.engine.get("engine"),
    "backend": sol.engine.get("backend"), "reason": sol.engine.get("reason"),
    "ladder": sol.engine.get("ladder"), "iterations": sol.engine.get("iterations"),
    "certificate_method": sol.engine.get("certificate_method"),
    "has_ray": sol.ray is not None, "ray_check": sol.ray_check,
    "host_farkas": host, "exact_farkas": exact,
    "verified": verified_ray, "verify_passed": v.get("passed"),
    "verify_checks": [[c[0], c[1], c[2]] for c in v.get("checks", [])],
    "provenance_clean": g["clean"],
}}, default=str))
"""


def free_ram_gb() -> float:
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                text=True).strip()
            return float(out) / (1024 * 1024)  # KB -> GB
        except Exception:  # noqa: BLE001
            pass
    try:
        # Linux: MemAvailable from /proc/meminfo
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / (1024 * 1024)
    except Exception:  # noqa: BLE001
        pass
    return 99.0  # unknown: do not block


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _run_worker(path: Path, engine: str, tol: float, tl: float, cert: Path, wall: float) -> dict:
    code = WORKER.format(src=str(ROOT / "src"))
    try:
        p = subprocess.run([sys.executable, "-c", code, str(path), engine, str(tol), str(tl), str(cert)],
                           capture_output=True, text=True, timeout=wall)
    except subprocess.TimeoutExpired:
        return {"error": f"killed at hard wall {wall:.0f}s", "reported": "not_proven", "verified": False}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    err = (p.stderr.strip().splitlines() or p.stdout.strip().splitlines() or ["no output"])[-1][:400]
    return {"error": err, "reported": "not_proven", "verified": False}


def run_one(name: str, *, engine: str, tol: float, time_limit: float) -> dict:
    from fetch import fetch
    rec = {"instance": name, "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    try:
        path = fetch(name, verbose=False)
    except Exception as e:  # noqa: BLE001
        rec.update(error=f"fetch failed: {e!r}"[:300], reported="not_proven", verified=False)
        return rec
    with tempfile.TemporaryDirectory() as td:
        cert = Path(td) / f"{name}.cert.json"
        t0 = time.perf_counter()
        q = _run_worker(path, engine, tol, time_limit, cert, wall=time_limit * 1.5 + 60)
        q["wall"] = time.perf_counter() - t0
    # Force the W09 contract at the record boundary.
    if q.get("reported") == "infeasible" and not q.get("verified"):
        q["reported"] = "not_proven"
    if q.get("reported") != "infeasible":
        q["reported"] = "not_proven"
        q["verified"] = False
    rec["qenivo"] = q
    return rec


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, default=str) + "\n")


def summarise(recs: list[dict], mode: str) -> str:
    proven = [r for r in recs if r.get("qenivo", {}).get("reported") == "infeasible"
              and r.get("qenivo", {}).get("verified")]
    not_p = [r["instance"] for r in recs if r not in proven]
    old_still = [n for n in OLD_FAILURES if n in not_p]
    lines = [
        f"mode={mode}  n={len(recs)}  proven_verified_ray={len(proven)}  not_proven={len(not_p)}",
        f"  proven: {' '.join(r['instance'] for r in proven) or '(none)'}",
        f"  not_proven: {' '.join(not_p) or '(none)'}",
        f"  old failures still unproven: {' '.join(old_still) or '(none of cplex2/gosh/gran/greenbea/pang)'}",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--force-full", action="store_true",
                    help="run all 29 even if free RAM < 2 GB")
    ap.add_argument("--instances", help="comma list override")
    a = ap.parse_args(argv)

    free = free_ram_gb()
    if a.instances:
        names = [n.strip() for n in a.instances.split(",") if n.strip()]
        mode = "custom"
    elif a.smoke_only or (free < MIN_FREE_GB and not a.force_full):
        names = list(SMOKE)
        mode = "smoke"
    else:
        names = list(NETLIB_INFEAS)
        mode = "full"

    meta = {
        "kind": "netlib_infeas_rerun_meta",
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "free_ram_gb": round(free, 3),
        "min_free_gb": MIN_FREE_GB,
        "n_planned": len(names),
        "instances": names,
        "time_limit": a.time_limit,
        "tol": a.tol,
        "engine": a.engine,
        "threads": 1,
        "jobs": 1,
        "commit": git_head(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "old_failures_20260930": OLD_FAILURES,
        "kaggle_era_failures": KAGGLE_ERA_FAILURES,
        "farkas_path": (
            "native dual_phase1+dual ray -> S_INFEASIBLE; certify via elastic_farkas + check_farkas; "
            "independent verify(); never report infeasible without verified ray"
        ),
        "note": ("Phase 1 light: free RAM < 2 GB -> smoke only; parent resumes full when RAM frees"
                 if mode == "smoke" and free < MIN_FREE_GB else None),
    }
    print(json.dumps({k: meta[k] for k in ("mode", "free_ram_gb", "n_planned", "instances", "note")},
                     indent=2), flush=True)
    append_jsonl(a.out, meta)

    recs = []
    for name in names:
        print(f"--- {name} ---", flush=True)
        rec = run_one(name, engine=a.engine, tol=a.tol, time_limit=a.time_limit)
        recs.append(rec)
        append_jsonl(a.out, rec)
        q = rec.get("qenivo", {})
        print(f"{name:12s} reported={q.get('reported')} verified={q.get('verified')} "
              f"engine={q.get('engine')} t={q.get('solve_time', 0) or 0:.2f}s "
              f"err={q.get('error')}", flush=True)

    summary = summarise(recs, mode)
    print("\n" + summary)
    append_jsonl(a.out, {"kind": "summary", "mode": mode, "text": summary,
                         "proven": sum(1 for r in recs if r.get("qenivo", {}).get("verified")),
                         "not_proven": sum(1 for r in recs if not r.get("qenivo", {}).get("verified")),
                         "n": len(recs)})
    print(f"results -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
