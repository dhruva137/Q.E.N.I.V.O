"""Benchmark runner: QENIVO vs an established solver, every answer independently verified.

    python bench/run.py --set netlib --tol 1e-6 --time-limit 120 --jobs 3
    python bench/run.py --set netlib_infeas --time-limit 120
    python bench/run.py --instances afiro,blend --engine ipm

For each instance, in separate processes with a hard wall-clock limit:
  1. QENIVO solves it (engine routed automatically unless --engine), writes a certificate;
  2. the independent verifier (qenivo.certify.verify, own MPS reader) re-checks the certificate;
  3. HiGHS (comparator only, never imported by QENIVO) solves it for the reference objective.
Results go to bench/results/<set>_<utc>.json with the QENIVO git commit, machine and versions.
Failures, time-outs and read errors stay in the file.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
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

WORKER_QENIVO = r"""
import json, sys, time
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
sol = qenivo.solve(prob, engine=engine, tol=tol, time_limit=tl, backend={backend!r})
t_solve = time.perf_counter() - t0 - t_read
sol.source = path
sol.save_certificate(cert)
v = verify(path, load(cert))
g = guard()
print("RESULT " + json.dumps({{"rows": prob.m, "cols": prob.n, "nnz": prob.nnz, "read_time": t_read,
    "solve_time": t_solve, "verdict": sol.verdict, "status": sol.status, "objective": sol.objective,
    "engine": sol.engine.get("engine"), "backend": sol.engine.get("backend"),
    "reason": sol.engine.get("reason"), "ladder": sol.engine.get("ladder"),
    "iterations": sol.engine.get("iterations"), "residuals": sol.residuals,
    "verified": v["passed"], "verify_checks": [[c[0], c[1], c[2]] for c in v["checks"]],
    "provenance_clean": g["clean"]}}, default=str))
"""

WORKER_HIGHS = r"""
import json, sys, time
import highspy
h = highspy.Highs()
h.setOptionValue("output_flag", False)
h.setOptionValue("time_limit", float(sys.argv[2]))
h.setOptionValue("threads", 1)
t0 = time.perf_counter()
h.readModel(sys.argv[1])
h.run()
t = time.perf_counter() - t0
st = h.modelStatusToString(h.getModelStatus())
obj = h.getInfo().objective_function_value if st == "Optimal" else None
print("RESULT " + json.dumps({"status": st, "objective": obj, "time": t}))
"""


def _run(code: str, args: list, wall: float) -> dict:
    try:
        p = subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True, text=True,
                           timeout=wall)
    except subprocess.TimeoutExpired:
        return {"error": f"killed at hard wall {wall:.0f}s"}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"error": (p.stderr.strip().splitlines() or ["no output"])[-1][:300]}


def run_instance(name: str, a) -> dict:
    from fetch import fetch
    rec = {"instance": name}
    try:
        path = fetch(name, verbose=False)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"fetch failed: {e!r}"[:300]
        return rec
    with tempfile.TemporaryDirectory() as td:
        cert = Path(td) / f"{name}.cert.json"
        code = WORKER_QENIVO.format(src=str(ROOT / "src"), backend=a.backend)
        t0 = time.perf_counter()
        rec["qenivo"] = _run(code, [path, a.engine, a.tol, a.time_limit, cert], a.time_limit * 1.5 + 60)
        rec["qenivo"]["wall"] = time.perf_counter() - t0
        if a.keep_certs and cert.exists():
            out = HERE / "results" / "certs"
            out.mkdir(parents=True, exist_ok=True)
            (out / cert.name).write_text(cert.read_text())
    if not a.no_ref:
        rec["highs"] = _run(WORKER_HIGHS, [path, a.time_limit], a.time_limit * 1.5 + 60)
    n, h = rec.get("qenivo", {}), rec.get("highs", {})
    if n.get("objective") is not None and h.get("objective") is not None:
        ref = h["objective"]
        rec["rel_obj_diff"] = abs(n["objective"] - ref) / (1 + abs(ref))
    from fetch import _is_maros, maros_key, maros_references
    if _is_maros(name):
        pub = maros_references().get(maros_key(name))
        rec["published_objective"] = pub
        if pub is not None and n.get("objective") is not None:
            # the published values carry 8 significant digits
            rec["rel_obj_diff_published"] = abs(n["objective"] - pub) / (1 + abs(pub))
    return rec


def _meta(a) -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    import numpy
    import scipy
    sys.path.insert(0, str(ROOT / "src"))
    from qenivo import __version__
    from qenivo.kernels.backend import gpu_info
    try:
        import highspy  # noqa: F401
        hv = "installed"
    except Exception:  # noqa: BLE001
        hv = None
    return {"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "qenivo": __version__, "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "argv": sys.argv[1:], "tol": a.tol, "time_limit": a.time_limit, "engine": a.engine,
            "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "highspy": hv, "platform": platform.platform(), "cpu": platform.processor(),
            "cpu_count": os.cpu_count(), "gpu": gpu_info(), "comparator_threads": 1}


def summarise(recs: list, set_name: str) -> str:
    tot = len(recs)
    ok = [r for r in recs if r.get("qenivo", {}).get("verdict") in ("optimal", "infeasible", "unbounded")]
    ver = [r for r in ok if r["qenivo"].get("verified")]
    agree = [r for r in ver if r.get("rel_obj_diff") is not None and r["rel_obj_diff"] <= 1e-6]
    pub = [r for r in ver if r.get("rel_obj_diff_published") is not None and r["rel_obj_diff_published"] <= 1e-5]
    infeas = [r for r in ver if r["qenivo"]["verdict"] == "infeasible"]
    lines = [f"{set_name}: {tot} instances",
             f"  proven verdict (optimal/infeasible/unbounded): {len(ok)}",
             f"  ...and independently VERIFIED:                 {len(ver)}",
             f"  optimal and within 1e-6 of HiGHS:              {len(agree)}",
             f"  infeasible with verified Farkas certificate:   {len(infeas)}"]
    if any("published_objective" in r for r in recs):
        lines.append(f"  optimal and within 1e-5 of the PUBLISHED optimum: {len(pub)}")
    bad = [r["instance"] for r in recs if r not in ver]
    if bad:
        lines.append(f"  not proven / not verified ({len(bad)}): {' '.join(bad)}")
    wrong = [r["instance"] for r in ver if r.get("rel_obj_diff") is not None and r["rel_obj_diff"] > 1e-4]
    if wrong:
        lines.append(f"  !! verified but objective differs from HiGHS by >1e-4: {' '.join(wrong)}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set")
    ap.add_argument("--instances")
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--keep-certs", action="store_true")
    ap.add_argument("--out-file", help="fixed results path; an existing file is resumed (done instances skipped)")
    a = ap.parse_args(argv)
    from sets import SETS
    names = SETS[a.set] if a.set else a.instances.split(",")
    label = a.set or "custom"
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(a.out_file) if a.out_file else out_dir / f"{label}_{a.engine}_{stamp}.json"
    meta = _meta(a)
    recs = []
    if out.exists():
        prev = json.load(open(out))
        recs = [r for r in prev.get("runs", []) if "error" not in r and "error" not in r.get("qenivo", {})]
        meta["resumed_from"] = prev.get("meta", {}).get("commit")
        done = {r["instance"] for r in recs}
        names = [n for n in names if n not in done]
        print(f"resuming {out}: {len(done)} done, {len(names)} to go")
    out.parent.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_instance, n, a): n for n in names}
        for f in cf.as_completed(futs):
            r = f.result()
            recs.append(r)
            n = r.get("qenivo", {})
            print(f"{r['instance']:12s} {n.get('verdict', r.get('error', n.get('error', '?'))):12s} "
                  f"{str(n.get('engine')):8s} {n.get('solve_time', 0) or 0:8.2f}s  "
                  f"verified={n.get('verified')}  d_obj={r.get('rel_obj_diff')}  "
                  f"highs={r.get('highs', {}).get('status')} {r.get('highs', {}).get('time', 0) or 0:.2f}s", flush=True)
            tmp = out.with_suffix(".tmp")
            with open(tmp, "w") as fh:
                json.dump({"meta": meta, "runs": sorted(recs, key=lambda x: x["instance"])}, fh, indent=1,
                          default=str)
            for attempt in range(20):                # atomic replace; Windows may hold the file briefly
                try:
                    os.replace(tmp, out)             # (antivirus, indexer, a reader): retry
                    break
                except PermissionError:
                    time.sleep(0.5)
    print("\n" + summarise(recs, label))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
