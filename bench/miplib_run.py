"""MIPLIB 2017 collection subset: our branch and cut vs HiGHS, scored against the official optima.

    python bench/miplib_run.py --time-limit 120 --out bench/results/miplib.jsonl

Instances: small classic models from the MIPLIB 2017 collection (the ones that are still served
at miplib.zib.de/WebData/instances). The reference value of each is the "=opt=" entry of the
official miplib2017-v33.solu file. Both solvers get the same time limit and ONE thread, so the
comparison is algorithm against algorithm. Our incumbent is re-checked on the original model
(rows, bounds, integrality 1e-6). Rows are appended as they finish (resume by name).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

MIPLIB = ("p0201 pk1 misc07 gt2 flugpl dcmulti khb05250 gen-ip002 markshare_4_0 neos5 mas76 noswot "
          "blend2 qnet1 rout").split()
SOLU_URL = "https://miplib.zib.de/downloads/miplib2017-v33.solu"
INST = HERE / "instances"


def fetch_miplib(name: str) -> Path:
    import gzip
    out = INST / f"{name}.mps"
    if out.exists() and out.stat().st_size:
        return out
    INST.mkdir(parents=True, exist_ok=True)
    raw = INST / f"{name}.mps.gz"
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(f"https://miplib.zib.de/WebData/instances/{name}.mps.gz", raw)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3)
    with gzip.open(raw, "rb") as fi, open(out, "wb") as fo:
        fo.write(fi.read())
    raw.unlink()
    return out


def optima() -> dict:
    f = INST / "miplib2017.solu"
    if not f.exists():
        INST.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOLU_URL, f)
    ref = {}
    for line in f.read_text().splitlines():
        p = line.split()
        if len(p) >= 3 and p[0] == "=opt=":
            ref[p[1]] = float(p[2])
    return ref


HIGHS = ("import highspy,sys,time;h=highspy.Highs();h.setOptionValue('output_flag',False);"
         "h.setOptionValue('threads',1);h.setOptionValue('time_limit',float(sys.argv[2]));h.readModel(sys.argv[1]);"
         "t=time.perf_counter();h.run();i=h.getInfo();"
         "print('RES',h.modelStatusToString(h.getModelStatus()).replace(' ','_'),i.objective_function_value,"
         "i.mip_dual_bound,i.mip_node_count,time.perf_counter()-t)")


def run_highs(path, tl):
    py = os.environ.get("CMP_PYTHON", sys.executable)
    try:
        p = subprocess.run([py, "-c", HIGHS, str(path), str(tl)], capture_output=True, text=True, timeout=tl + 120)
    except subprocess.TimeoutExpired:
        return {"status": "killed"}
    for line in p.stdout.splitlines():
        if line.startswith("RES"):
            _, st, obj, bnd, nodes, t = line.split()
            return {"status": st, "objective": float(obj), "bound": float(bnd), "nodes": int(nodes), "time": float(t)}
    return {"status": "error", "stderr": p.stderr[-300:]}


def run_ours(path, tl):
    import qenivo
    from qenivo.engines.milp import solve_milp
    prob = qenivo.read(path)
    t = time.perf_counter()
    s = solve_milp(prob, time_limit=tl)
    e = s.engine
    return {"verdict": s.verdict, "status": s.status, "objective": s.objective if s.x is not None else None,
            "bound": e.get("bound"), "gap": e.get("gap"), "nodes": e.get("nodes"), "time": time.perf_counter() - t,
            "feasible_on_original": (s.extra.get("mip") or {}).get("incumbent_feasible"),
            "cuts": e.get("cuts"), "presolve": e.get("presolve"), "rows": prob.m, "cols": prob.n,
            "integers": int(prob.integer.sum()) if prob.integer is not None else 0}


def score(obj, opt):
    if obj is None or opt is None:
        return None
    return abs(obj - opt) / max(1.0, abs(opt))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=120)
    ap.add_argument("--out", default=str(HERE / "results" / "miplib.jsonl"))
    ap.add_argument("--only", default="")
    ap.add_argument("--no-highs", action="store_true", help="skip the HiGHS comparison (quick A/B runs)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line)["instance"])
            except Exception:
                pass
    ref = optima()
    names = a.only.split(",") if a.only else MIPLIB
    for name in names:
        if name in done:
            continue
        try:
            path = fetch_miplib(name)
        except Exception as e:
            row = {"instance": name, "status": "missing", "error": str(e)[:200]}
        else:
            ours = run_ours(path, a.time_limit)
            highs = {"status": "skipped"} if a.no_highs else run_highs(path, a.time_limit)
            opt = ref.get(name)
            row = {"instance": name, "optimum": opt, "ours": ours, "highs": highs,
                   "ours_rel_err": score(ours.get("objective"), opt), "highs_rel_err": score(highs.get("objective"), opt),
                   "time_limit": a.time_limit}
            row["ours_solved"] = bool(row["ours_rel_err"] is not None and row["ours_rel_err"] <= 1e-6
                                      and ours["verdict"] == "optimal")
            row["ours_found_optimum"] = bool(row["ours_rel_err"] is not None and row["ours_rel_err"] <= 1e-6)
            row["highs_solved"] = bool(highs.get("status") == "Optimal" and row["highs_rel_err"] is not None
                                       and row["highs_rel_err"] <= 1e-6)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        o = row.get("ours", {})
        print(f"{name:14s} opt {row.get('optimum')}  ours {o.get('verdict')} {o.get('objective')} gap {o.get('gap')} "
              f"{o.get('time', 0):.1f}s nodes {o.get('nodes')} | HiGHS {row.get('highs', {}).get('status')} "
              f"{row.get('highs', {}).get('time')}", flush=True)


if __name__ == "__main__":
    main()
