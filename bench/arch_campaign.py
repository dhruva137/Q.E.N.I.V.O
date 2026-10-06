"""Notebook 2 campaign: decide the architecture from data, and measure against the world.

    python bench/arch_campaign.py --run-dir <dir>            # everything, resumable
    python bench/arch_campaign.py --run-dir <dir> --only a3  # one part
    python bench/arch_campaign.py --run-dir runs/x --quick   # CPU smoke test

Parts
  a1 preflight  download every benchmark set up front (retries + mirrors); manifest of what is
                available; everything later uses only what is on disk, so no part fails on the network
  a2 parity     full coverage through the production pipeline (route -> solve -> certify -> independent
                verify): Netlib 98, Netlib infeasible 29, Kennington 16, Maros-Meszaros QP 138,
                large Mittelmann LPs. HiGHS alongside as the reference. Uses bench/run.py (resumable).
  a3 engine_map every engine on every LP up to a size cap, fixed time limit: which engine wins where.
                Output: the routing thresholds, fitted from the measurements, for the router
  a4 ablation   PDHG design choices on the GPU (restarts, Halpern, reflection, preconditioning,
                check interval, own vs vendor kernels) on hard LPs at 1e-6: the defaults, decided by data
  a5 world      the same instances through solvers from other countries, on this machine, each in
                its own process: HiGHS (UK), OR-Tools GLOP / PDLP (Google, USA), SCIP (ZIB, Germany),
                cuPDLPx (MIT, USA, GPU), NVIDIA cuOpt (USA, GPU, when it installs)
  a7 precision  float64 PDHG vs float32 PDHG with float64 refinement (pdhg-rf) on the hard LPs at 1e-6
                and 1e-8: on T4-class GPUs float32 runs far faster than float64; is the answer as good?
  a8 multigpu   one refinery what-if batch on 1 GPU and split across all GPUs (e.g. Kaggle's two T4s),
                against HiGHS on all CPU cores; every scenario certified in float64
  m1 miplib     MIPLIB 2017 subset: our branch and cut vs HiGHS, one thread each, official optima
  a6 report     ARCHITECTURE_DECISIONS.md, PARITY.md, WORLD.md, T4.md from the recorded rows only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from gpu_campaign import GPU, Recorder, env, free_gpu, sync  # noqa: E402
from sets import SETS  # noqa: E402

import qenivo  # noqa: E402
from qenivo.certify.kkt import kkt_residuals  # noqa: E402

PARITY_SETS = ["netlib", "netlib_infeas", "kennington", "maros", "gate"]
HARD_LPS = ["25fv47", "80bau3b", "d2q06c", "degen3", "dfl001", "fit2p", "greenbea", "maros-r7", "pilot", "pilot87",
            "perold", "qap15", "stocfor3", "ken-13", "osa-60", "pds-20", "cre-b", "nug08-3rd", "cont1", "savsched1"]


def _manifest(run_dir):
    f = Path(run_dir) / "manifest.json"
    return json.loads(f.read_text()) if f.exists() else {}


# ---------------------------------------------------------------------- a1
def a1_preflight(rec, run_dir, quick):
    from fetch import fetch
    sets = {"small": SETS["small"]} if quick else {k: SETS[k] for k in PARITY_SETS}
    manifest = _manifest(run_dir)
    for s, names in sets.items():
        for n in names:
            if n in manifest and manifest[n].get("ok"):
                continue
            ok, err, size = False, None, None
            for attempt in range(3):
                try:
                    p = fetch(n, verbose=False)
                    size = p.stat().st_size
                    qenivo.read(p)                          # readable, not just downloaded
                    ok = True
                    break
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"[:200]
                    time.sleep(2 * (attempt + 1))
            manifest[n] = {"set": s, "ok": ok, "error": None if ok else err, "bytes": size}
            (Path(run_dir) / "manifest.json").write_text(json.dumps(manifest, indent=1))
    have = sum(v["ok"] for v in manifest.values())
    rec.add(f"manifest/{int(time.time())}", {"available": have, "missing": [k for k, v in manifest.items() if not v["ok"]]})
    print(f"a1 preflight: {have}/{len(manifest)} instances on disk and readable")


# ---------------------------------------------------------------------- a2
def a2_parity(rec, run_dir, quick):
    manifest = _manifest(run_dir)
    sets = ["small"] if quick else PARITY_SETS
    tl = 60 if quick else 150
    for s in sets:
        names = [n for n in SETS[s] if manifest.get(n, {}).get("ok", quick)]
        out = Path(run_dir) / f"parity_{s}.json"
        cmd = [sys.executable, "-u", str(HERE / "run.py"), "--instances", ",".join(names), "--tol", "1e-6",
               "--time-limit", str(tl), "--jobs", "1" if quick else "4", "--out-file", str(out)]
        print(f"a2 {s}: {len(names)} instances -> {out.name}", flush=True)
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=6 * 3600)
        tail = [line for line in p.stdout.splitlines() if line.strip()][-8:]
        print("\n".join(tail))
        rec.add(f"{s}/{int(time.time())}", {"set": s, "returncode": p.returncode, "time": time.perf_counter() - t0,
                                            "summary": tail})


# ---------------------------------------------------------------------- a3
def a3_engine_map(rec, run_dir, quick):
    from fetch import fetch
    manifest = _manifest(run_dir)
    names = SETS["small"] if quick else [n for n in SETS["netlib"] + SETS["kennington"] if manifest.get(n, {}).get("ok")]
    if not quick:                                   # keep the map inside its budget: models up to 40k rows
        from fetch import fetch as _f
        names = [n for n in names if qenivo.read(_f(n, verbose=False)).m <= 40000]
    engines = [("simplex", "numpy"), ("ipm", "numpy"), ("pdhg", "numpy")] + ([("pdhg", "cupy")] if GPU == "cupy" else [])
    tl = 30 if quick else 60
    for name in names:
        p = qenivo.read(fetch(name, verbose=False))
        for eng, be in engines:
            key = f"{name}/{eng}/{be}"
            if rec.done(key):
                continue
            if eng == "simplex" and p.m > 3000:
                rec.add(key, {"instance": name, "engine": eng, "backend": be, "skipped": "dense simplex above 3000 rows"})
                continue
            code = (f"import sys,time,json;sys.path.insert(0,{str(ROOT / 'src')!r});import qenivo\n"
                    f"p=qenivo.read(sys.argv[1]);t=time.perf_counter()\n"
                    f"s=qenivo.solve(p,engine={eng!r},backend={be!r},tol=1e-6,time_limit={tl},polish=False)\n"
                    f"print('RESULT '+json.dumps({{'verdict':s.verdict,'time':time.perf_counter()-t,'iterations':s.engine.get('iterations'),'max_rel':(s.residuals or {{}}).get('max_rel')}}))")
            try:
                r = subprocess.run([sys.executable, "-c", code, str(fetch(name, verbose=False))], capture_output=True,
                                   text=True, timeout=tl * 2 + 60)
                res = next((json.loads(line[7:]) for line in r.stdout.splitlines() if line.startswith("RESULT ")),
                           {"verdict": "error", "error": r.stderr[-200:]})
            except subprocess.TimeoutExpired:
                res = {"verdict": "killed_at_wall", "time": tl * 2 + 60}
            rec.add(key, {"instance": name, "rows": p.m, "cols": p.n, "nnz": p.nnz, "engine": eng, "backend": be, **res})
        print(f"a3 {name:10s} " + " ".join(f"{e}/{b}:{(r.get('verdict') or '')[:3]}:{(r.get('time') or 0):.1f}s"
                                           for r in rec.rows if r.get("instance") == name for e, b in [(r["engine"], r["backend"])]))


def fit_routing(rows):
    """From the engine map, the size thresholds at which each engine is fastest among verified answers."""
    by = {}
    for r in rows:
        if r.get("verdict") == "optimal":
            by.setdefault(r["instance"], []).append(r)
    winners = []
    for inst, rs in by.items():
        best = min(rs, key=lambda r: r["time"])
        winners.append((best["rows"], best["nnz"], f"{best['engine']}/{best['backend']}"))
    winners.sort()
    table = {}
    for lo, hi in ((0, 600), (600, 3000), (3000, 20000), (20000, 10 ** 9)):
        w = [e for m, _, e in winners if lo <= m < hi]
        if w:
            table[f"{lo}-{hi} rows"] = {e: w.count(e) for e in sorted(set(w))}
    return table, winners


# ---------------------------------------------------------------------- a4
ABLATIONS = {"default": {}, "no_restarts": {"restarts": False}, "no_halpern": {"halpern": False},
             "no_reflection": {"reflection": 0.5}, "no_scaling": {"geo_iters": 0, "ruiz_iters": 0, "pc_alpha": None},
             "check_32": {"check_every": 32}, "check_256": {"check_every": 256}, "vendor_kernels": {"kernels": "vendor"}}


def a4_ablation(rec, run_dir, quick):
    from fetch import fetch

    from qenivo.engines import pdhg
    manifest = _manifest(run_dir)
    names = ["afiro", "sc50a"] if quick else [n for n in HARD_LPS if manifest.get(n, {}).get("ok")]
    tl = 30 if quick else 120
    for name in names:
        p = qenivo.read(fetch(name, verbose=False))
        for cfg, over in ABLATIONS.items():
            key = f"{name}/{cfg}"
            if rec.done(key) or (cfg == "vendor_kernels" and GPU != "cupy"):
                continue
            opts = pdhg.PDHGOptions(tol=1e-6, time_limit=tl, **over)
            t = time.perf_counter()
            try:
                br = pdhg.solve(p, tol=1e-6, backend=GPU, opts=opts)
                sync()
                col = br.column(0)
                k = kkt_residuals(p, col["x"], col["y"])
                row = {"status": col["status"], "iterations": col["iterations"], "time": time.perf_counter() - t,
                       "max_rel": k["max_rel"]}
            except Exception as e:  # noqa: BLE001
                row = {"status": "error", "error": repr(e)[:200], "time": time.perf_counter() - t}
            rec.add(key, {"instance": name, "rows": p.m, "config": cfg, "backend": GPU, **row})
            print(f"a4 {name:10s} {cfg:15s} {row['status']:15s} {row['time']:7.2f}s it {row.get('iterations')}")
            free_gpu()


# ---------------------------------------------------------------------- a5
WORLD = ["highs_simplex", "highs_ipm", "highs_pdlp", "ortools_glop", "ortools_pdlp", "scip", "cupdlpx", "cuopt"]


def a5_world(rec, run_dir, quick):
    import comparators as C
    from fetch import fetch
    manifest = _manifest(run_dir)
    avail = C.available()
    if not rec.done("available"):
        rec.add("available", avail)
    names = ["afiro"] if quick else [n for n in HARD_LPS + SETS["maros"][:20] if manifest.get(n, {}).get("ok")]
    tl = 60 if quick else 240
    for name in names:
        path = fetch(name, verbose=False)
        key = f"{name}/qenivo"
        if not rec.done(key):
            t = time.perf_counter()
            s = qenivo.solve(qenivo.read(path), tol=1e-6, time_limit=tl)
            rec.add(key, {"instance": name, "solver": "qenivo", "status": s.verdict, "objective": s.objective,
                          "time": time.perf_counter() - t, "engine": s.engine.get("engine"), "backend": s.engine.get("backend"),
                          "our_rel_primal": (s.residuals or {}).get("rel_primal")})
        for c in WORLD:
            key = f"{name}/{c}"
            if rec.done(key):
                continue
            rec.add(key, {"instance": name, **C.run(c, path, 1e-6, tl)})
        print(f"a5 {name:10s} " + " ".join(f"{r['solver']}:{str(r.get('status'))[:8]}:{(r.get('time') or 0):.1f}s"
                                           for r in rec.rows if r.get("instance") == name))


# ---------------------------------------------------------------------- a7
def a7_precision(rec, run_dir, quick):
    from fetch import fetch

    from qenivo.engines import pdhg, refine
    manifest = _manifest(run_dir)
    names = ["afiro", "sc50a"] if quick else [n for n in HARD_LPS if manifest.get(n, {}).get("ok")]
    tl = 30 if quick else 150
    for name in names:
        p = qenivo.read(fetch(name, verbose=False))
        for tol in (1e-6, 1e-8):
            for eng in ("pdhg", "pdhg-rf"):
                key = f"{name}/{eng}/{tol:g}"
                if rec.done(key):
                    continue
                t = time.perf_counter()
                try:
                    if eng == "pdhg":
                        br = pdhg.solve(p, tol=tol, backend=GPU, time_limit=tl)
                    else:
                        br = refine.solve(p, tol=tol, backend=GPU, time_limit=tl)
                    sync()
                    col = br.column(0)
                    k = kkt_residuals(p, col["x"], col["y"])
                    row = {"status": col["status"], "iterations": col["iterations"], "time": time.perf_counter() - t,
                           "max_rel": k["max_rel"], "objective": k["objective"]}
                    if eng == "pdhg-rf":
                        row["timing"] = br.rays.get("_timing")
                except Exception as e:  # noqa: BLE001
                    row = {"status": "error", "error": repr(e)[:200], "time": time.perf_counter() - t}
                rec.add(key, {"instance": name, "rows": p.m, "nnz": p.nnz, "engine": eng, "tol": tol, "backend": GPU, **row})
                print(f"a7 {name:10s} {eng:8s} tol {tol:.0e}: {row['status']:15s} {row['time']:7.2f}s "
                      f"it {row.get('iterations')} maxrel {row.get('max_rel', float('nan')):.1e}", flush=True)
                free_gpu()


# ---------------------------------------------------------------------- a8
def a8_multigpu(rec, run_dir, quick):
    from gpu_campaign import _scenarios, certify_batch, highs_pool_time

    from qenivo.engines import pdhg
    from qenivo.engines.multigpu import device_count, solve_batch_devices
    ndev = device_count()
    if not rec.done("devices"):
        rec.add("devices", {"gpus": ndev})
    if ndev == 0:
        print("a8: no GPU, skipped")
        return
    levels = (1,) if quick else (3, 4)
    batches = (8,) if quick else (64, 256)
    for lv in levels:
        for S in batches:
            p, C = _scenarios(lv, S)
            for devs in sorted({1, ndev}):
                key = f"L{lv}/S{S}/gpus{devs}"
                if rec.done(key):
                    continue
                r = solve_batch_devices(p.A, C, p.lc, p.uc, p.lx, p.ux, p.c0, devices=list(range(devs)),
                                        engine="pdhg", tol=1e-4, time_limit=1800)
                br = pdhg.BatchResult(x=r["x"], y=r["y"], status=r["status"], iterations=r["iterations"],
                                      restarts=np.zeros(S), kkt={}, solve_time=r["wall"], setup_time=0.0,
                                      backend="cupy", eta=float("nan"), primal_weight=np.zeros(S))
                ok, worst = certify_batch(p, C, br, 1e-4)
                row = {"level": lv, "rows": p.m, "S": S, "gpus": devs, "wall": r["wall"], "device_times": r["device_times"],
                       "certified": ok, "worst_rel": worst, "errors": r["errors"]}
                if devs == ndev and not rec.done(f"L{lv}/S{S}/highs"):
                    try:
                        th, _ = highs_pool_time(p, C, os.cpu_count() or 1)
                        rec.add(f"L{lv}/S{S}/highs", {"level": lv, "S": S, "highs_all_cores": th,
                                                      "cpu_count": os.cpu_count()})
                    except Exception as e:  # noqa: BLE001
                        rec.add(f"L{lv}/S{S}/highs", {"level": lv, "S": S, "error": repr(e)[:200]})
                rec.add(key, row)
                print(f"a8 L{lv} S={S:4d} on {devs} GPU(s): {r['wall']:7.2f}s  certified {ok}/{S}", flush=True)
                free_gpu()


# ---------------------------------------------------------------------- m1
def m1_miplib(rec, run_dir, quick):
    out = Path(run_dir) / "miplib.jsonl"
    names = "flugpl,khb05250" if quick else ""
    cmd = [sys.executable, "-u", str(HERE / "miplib_run.py"), "--time-limit", "30" if quick else "120", "--out", str(out)]
    if names:
        cmd += ["--only", names]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=4 * 3600)
    tail = [line for line in p.stdout.splitlines() if line.strip()][-20:]
    print("\n".join(tail))
    rec.add(f"miplib/{int(time.time())}", {"returncode": p.returncode, "summary": tail})


# ---------------------------------------------------------------------- a6
def a6_report(run_dir):
    def load(n):
        f = Path(run_dir) / f"{n}.jsonl"
        return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()] if f.exists() else []
    L = ["# Architecture decisions (from measurements)", ""]
    em = load("a3_engine_map")
    if em:
        table, winners = fit_routing(em)
        L += ["## Router: fastest engine among verified answers, by model size", "",
              "| size band | wins per engine |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in table.items()]
        L += ["", "Use: set RoutingPolicy thresholds to the band edges where the winner changes.", ""]
    ab = load("a4_ablation")
    if ab:
        cfgs = list(ABLATIONS)
        L += ["## PDHG design ablation (1e-6, GPU): solved count and median time over solved", "",
              "| config | solved | median time (s) | median iterations |", "|---|---|---|---|"]
        for c in cfgs:
            rs = [r for r in ab if r.get("config") == c]
            ok = [r for r in rs if r.get("status") == "optimal"]
            if rs:
                L.append(f"| {c} | {len(ok)}/{len(rs)} | {np.median([r['time'] for r in ok]) if ok else float('nan'):.2f} | "
                         f"{np.median([r['iterations'] for r in ok]) if ok else float('nan'):.0f} |")
        L.append("")
    (Path(run_dir) / "ARCHITECTURE_DECISIONS.md").write_text("\n".join(L), encoding="utf-8")
    w = [r for r in load("a5_world") if r.get("instance")]
    if w:
        solvers = ["qenivo"] + WORLD
        insts = sorted({r["instance"] for r in w})
        M = ["# Same machine, same instances, same 1e-6 target", "",
             "| instance | " + " | ".join(solvers) + " |", "|---" * (len(solvers) + 1) + "|"]
        for i in insts:
            cells = []
            for s in solvers:
                r = next((r for r in w if r["instance"] == i and r.get("solver") == s), None)
                if not r or r.get("status") == "unavailable":
                    cells.append("n/a")
                else:
                    st = str(r.get("status"))[:10]
                    cells.append(f"{st} {r.get('time', 0) or 0:.1f}s")
            M.append(f"| {i} | " + " | ".join(cells) + " |")
        (Path(run_dir) / "WORLD.md").write_text("\n".join(M), encoding="utf-8")
    try:
        import parity_summary
        parity_summary.main(run_dir)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    _t4_report(run_dir, load)
    for f in ("ARCHITECTURE_DECISIONS.md", "WORLD.md", "T4.md"):
        p = Path(run_dir) / f
        if p.exists():
            print(p.read_text(encoding="utf-8"))


def _t4_report(run_dir, load):
    L = ["# Precision, multi-GPU and MIPLIB (from the recorded rows only)", ""]
    pr = load("a7_precision")
    if pr:
        L += ["## float64 PDHG vs float32 PDHG + float64 refinement", "",
              "| instance | tol | pdhg (s) | pdhg max_rel | pdhg-rf (s) | pdhg-rf max_rel |", "|---|---|---|---|---|---|"]
        for (inst, tol) in sorted({(r["instance"], r["tol"]) for r in pr if "tol" in r}):
            a = next((r for r in pr if r.get("instance") == inst and r.get("tol") == tol and r["engine"] == "pdhg"), {})
            b = next((r for r in pr if r.get("instance") == inst and r.get("tol") == tol and r["engine"] == "pdhg-rf"), {})
            f = lambda r: (f"{r.get('time', float('nan')):.1f}" + ("" if r.get("status") == "optimal" else f" ({r.get('status')})"),  # noqa: E731
                           f"{r.get('max_rel', float('nan')):.1e}")
            L.append(f"| {inst} | {tol:g} | {f(a)[0]} | {f(a)[1]} | {f(b)[0]} | {f(b)[1]} |")
        L.append("")
    mg = load("a8_multigpu")
    if mg:
        L += ["## One what-if batch on 1 GPU vs all GPUs (1e-4, every scenario certified)", "",
              "| level | S | GPUs | wall (s) | certified | HiGHS all cores (s) |", "|---|---|---|---|---|---|"]
        for r in mg:
            if "gpus" in r and "wall" in r:
                h = next((x.get("highs_all_cores") for x in mg if x.get("level") == r["level"] and x.get("S") == r["S"]
                          and "highs_all_cores" in x), None)
                L.append(f"| L{r['level']} | {r['S']} | {r['gpus']} | {r['wall']:.1f} | {r['certified']}/{r['S']} | "
                         f"{'' if h is None else f'{h:.1f}'} |")
        L.append("")
    mf = Path(run_dir) / "miplib.jsonl"
    if mf.exists():
        rows = [json.loads(line) for line in mf.read_text().splitlines() if line.strip()]
        L += ["## MIPLIB 2017 subset, 120 s, one thread each, scored against the official optima", "",
              f"Proven optimal: ours {sum(r.get('ours_solved', False) for r in rows)}/{len(rows)}, "
              f"HiGHS {sum(r.get('highs_solved', False) for r in rows)}/{len(rows)}", "",
              "| instance | optimum | ours | ours gap | HiGHS |", "|---|---|---|---|---|"]
        for r in rows:
            o, h = r.get("ours", {}), r.get("highs", {})
            L.append(f"| {r['instance']} | {r.get('optimum')} | {o.get('verdict')} {o.get('objective')} | {o.get('gap')} | "
                     f"{h.get('status')} {h.get('time')} |")
    (Path(run_dir) / "T4.md").write_text("\n".join(L), encoding="utf-8")


PARTS = {"a1": ("a1_preflight", a1_preflight), "a2": ("a2_parity", a2_parity), "a3": ("a3_engine_map", a3_engine_map),
         "a4": ("a4_ablation", a4_ablation), "a5": ("a5_world", a5_world), "a7": ("a7_precision", a7_precision),
         "a8": ("a8_multigpu", a8_multigpu), "m1": ("m1_miplib", m1_miplib)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--only", default="a1,a2,a3,a4,a5")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    envrec = Recorder(run_dir, "a0_env")
    if not envrec.done("env"):
        envrec.add("env", env())
    for tag in a.only.split(","):
        name, fn = PARTS[tag]
        rec = Recorder(run_dir, name)
        t0 = time.perf_counter()
        print(f"\n===== {tag} {name} ({len(rec.rows)} rows already recorded) =====", flush=True)
        try:
            fn(rec, run_dir, a.quick)
            status = "ok"
        except Exception as e:  # noqa: BLE001
            status = f"failed: {type(e).__name__}: {e}"
            traceback.print_exc()
        Recorder(run_dir, "_progress").add(f"{tag}/{int(time.time())}", {"part": tag, "status": status,
                                                                          "seconds": time.perf_counter() - t0})
    a6_report(run_dir)


if __name__ == "__main__":
    main()
