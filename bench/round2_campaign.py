"""Round-2 A100 campaign: measure only what the first evidence run did not cover.

    python bench/round2_campaign.py --run-dir /content/qenivo_results/round2_run1
    python bench/round2_campaign.py --run-dir runs/r2_smoke --quick
    python bench/round2_campaign.py --run-dir <same> --only stamp,tight

Parts (append-only JSONL, resumable, no invented numbers)
  stamp          GPU, driver, commit, gpu_status(), native simplex status
  tight          Linf_520c / rmine15 / cont1 at 1e-6 (60 s cap): PDHG persistent off/on,
                 refine float32+refinement, PDHG-to-1e-3 then crossover; host KKT
  l5             refinery level 5 batch S=8,32 vs HiGHS all cores (8 min hard cap per batch)
  refinery       case 1 T4 homotopy (CPU), objective / max violation / feasible
  headline       HEADLINE_ROUND2.md from recorded rows + existing e2 cuPDLPx rows
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

# Import pool helpers from the existing campaign (spawn-safe workers live there).
from gpu_campaign import certify_batch, highs_pool_time  # noqa: E402

import qenivo  # noqa: E402
from qenivo.certify.kkt import kkt_residuals  # noqa: E402
from qenivo.engines import crossover, native_simplex, pdhg, refine  # noqa: E402
from qenivo.kernels.backend import gpu_available, gpu_info, gpu_status  # noqa: E402
from qenivo.models.refinery import LEVELS, refinery_lp  # noqa: E402

GPU = "cupy" if gpu_available() else "numpy"
E2_CUPDLPX = HERE / "results" / "a100_evidence_run1" / "e2_large_lp.jsonl"


# ====================================================================== recording
class Recorder:
    """Append-only JSONL with fsync; `done(key)` makes every loop resumable."""

    def __init__(self, run_dir: Path, name: str):
        self.path = Path(run_dir) / f"{name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self.rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        self.keys = {r.get("key") for r in self.rows}

    def done(self, key: str) -> bool:
        return key in self.keys

    def add(self, key: str, row: dict):
        row = {"key": key, "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **row}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=_json) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.rows.append(row)
        self.keys.add(key)


def _json(o):
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


def sync():
    if GPU == "cupy":
        import cupy
        cupy.cuda.Device().synchronize()


def free_gpu():
    if GPU == "cupy":
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()


def _git(*a):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _host_kkt(prob, x, y):
    k = kkt_residuals(prob, x, y)
    return {kk: float(k[kk]) for kk in ("objective", "rel_primal", "rel_dual", "rel_gap", "max_rel")}


# ====================================================================== stamp
def part_stamp(rec, quick):
    if rec.done("stamp"):
        return
    smi = None
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        smi = f"unavailable: {type(e).__name__}"
    row = {
        "qenivo": qenivo.__version__,
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "backend": GPU,
        "gpu_info": gpu_info(),
        "gpu_status": gpu_status(),
        "gpu_available": gpu_available(),
        "nvidia_smi": smi,
        "native_simplex": native_simplex.status(),
        "quick": bool(quick),
    }
    rec.add("stamp", row)
    print("stamp:", row["nvidia_smi"] or "no GPU", "|", row["gpu_status"],
          "| native", row["native_simplex"], "|", row["commit"][:12] if row["commit"] else "?")


# ====================================================================== tight tolerance
def _trial_pdhg(prob, *, tol, persistent, time_limit):
    t0 = time.perf_counter()
    br = pdhg.solve(prob, tol=tol, backend=GPU, time_limit=time_limit, persistent=persistent)
    sync()
    elapsed = time.perf_counter() - t0
    col = br.column(0)
    kkt = _host_kkt(prob, col["x"], col["y"])
    return {
        "method": f"pdhg_persistent_{persistent}",
        "time": elapsed,
        "iterations": col["iterations"],
        "status": col["status"],
        "tol_request": tol,
        "host_kkt": kkt,
        "x": col["x"],
        "y": col["y"],
    }


def _trial_refine(prob, *, tol, time_limit):
    t0 = time.perf_counter()
    br = refine.solve(prob, tol=tol, backend=GPU, time_limit=time_limit, inner_dtype="float32")
    sync()
    elapsed = time.perf_counter() - t0
    col = br.column(0)
    kkt = _host_kkt(prob, col["x"], col["y"])
    return {
        "method": "refine_float32",
        "time": elapsed,
        "iterations": col["iterations"],
        "status": col["status"],
        "tol_request": tol,
        "host_kkt": kkt,
        "backend": br.backend,
    }


def _trial_crossover(prob, *, time_limit):
    """PDHG to 1e-3, then engines.crossover.crossover(prob, x, y)."""
    t_pdhg0 = time.perf_counter()
    br = pdhg.solve(prob, tol=1e-3, backend=GPU, time_limit=min(time_limit, 60.0), persistent="auto")
    sync()
    t_pdhg = time.perf_counter() - t_pdhg0
    col = br.column(0)
    remain = max(1.0, time_limit - t_pdhg)
    t_x0 = time.perf_counter()
    out = crossover.crossover(prob, col["x"], col["y"], time_limit=remain)
    t_x = time.perf_counter() - t_x0
    x, y = out["x"], out["y"]
    kkt = _host_kkt(prob, x, y)
    return {
        "method": "pdhg_1e-3_then_crossover",
        "time": t_pdhg + t_x,
        "pdhg_time": t_pdhg,
        "crossover_time": out.get("crossover_time", t_x),
        "crossover_guess_time": out.get("crossover_guess_time"),
        "pdhg_iterations": col["iterations"],
        "pdhg_status": col["status"],
        "iterations": out.get("iterations"),
        "status": out.get("status"),
        "tol_request": 1e-6,
        "host_kkt": kkt,
    }


def part_tight(rec, quick):
    from fetch import fetch

    names = ["afiro"] if quick else ["Linf_520c", "rmine15", "cont1"]
    tol = 1e-6
    trial_cap = 15.0 if quick else 60.0
    methods = (
        ("pdhg_off", lambda p: _trial_pdhg(p, tol=tol, persistent="off", time_limit=trial_cap)),
        ("pdhg_on", lambda p: _trial_pdhg(p, tol=tol, persistent="on", time_limit=trial_cap)),
        ("refine", lambda p: _trial_refine(p, tol=tol, time_limit=trial_cap)),
        ("crossover", lambda p: _trial_crossover(p, time_limit=trial_cap)),
    )
    for name in names:
        path = fetch(name, verbose=False)
        prob = qenivo.read(path)
        for tag, fn in methods:
            key = f"{name}/{tag}/tol{tol:g}"
            if rec.done(key):
                continue
            try:
                row = fn(prob)
                # never persist primal/dual vectors in the JSONL
                row.pop("x", None)
                row.pop("y", None)
                row.update({"instance": name, "rows": prob.m, "cols": prob.n, "nnz": int(prob.A.nnz),
                            "backend": GPU, "trial_cap_s": trial_cap})
                rec.add(key, row)
                hk = row["host_kkt"]
                print(f"tight {name:10s} {tag:10s}: {row['status']} {row['time']:.2f}s "
                      f"it {row.get('iterations')} maxrel {hk['max_rel']:.1e}")
            except Exception as e:  # noqa: BLE001
                rec.add(key, {"instance": name, "method": tag, "status": "error",
                              "error": f"{type(e).__name__}: {e}"[:500], "traceback": traceback.format_exc()[-800:]})
                print(f"tight {name} {tag}: ERROR {type(e).__name__}: {e}")
            free_gpu()


# ====================================================================== L5 stack
def _scenarios(level, S, seed=0):
    R, T, K, Mk = LEVELS[level]
    p = refinery_lp(R, T, K, Mk, seed=seed)
    rng = np.random.default_rng(1000 + seed + level)
    return p, p.c[:, None] * rng.uniform(0.9, 1.1, (p.n, S))


def _highs_pool_worker(queue, problem, costs, nworkers):
    """Module-level so spawn can pickle it (Windows)."""
    try:
        tp, objs = highs_pool_time(problem, costs, nworkers)
        queue.put(("ok", float(tp), np.asarray(objs, float)))
    except Exception as e:  # noqa: BLE001
        queue.put(("err", f"{type(e).__name__}: {e}", None))


def _highs_with_cap(p, C, workers, highs_cap):
    """Run highs_pool_time with a hard wall; return (note, time_or_None, objs_or_None)."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_highs_pool_worker, args=(q, p, C, workers))
    proc.start()
    proc.join(timeout=highs_cap)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        return "cap reached", None, None
    try:
        kind, a, b = q.get_nowait()
    except Exception:  # noqa: BLE001
        return "no result", None, None
    if kind == "ok":
        return None, a, b
    return a, None, None


def part_l5(rec, quick):
    level = 1 if quick else 5
    batches = (2, 4) if quick else (8, 32)
    tol = 1e-4
    highs_cap = 30.0 if quick else 8 * 60.0
    workers = os.cpu_count() or 1
    try:
        import highspy  # noqa: F401
        has_highs = True
    except Exception:  # noqa: BLE001
        has_highs = False

    for S in batches:
        key = f"L{level}/S{S}/tol{tol:g}"
        if rec.done(key):
            continue
        p, C = _scenarios(level, S)
        opts = pdhg.PDHGOptions(tol=tol, time_limit=3600)
        t0 = time.perf_counter()
        br = pdhg.solve_batch(p.A, C, p.lc, p.uc, p.lx, p.ux, p.c0, opts, GPU)
        sync()
        gpu_t = time.perf_counter() - t0
        ok, worst = certify_batch(p, C, br, tol)
        row = {
            "level": level, "rows": p.m, "cols": p.n, "nnz": int(p.A.nnz), "S": S, "tol": tol,
            "gpu_time": gpu_t, "certified": ok, "worst_rel": worst,
            "iterations_max": int(br.iterations.max()), "cpu_workers": workers,
            "statuses": list(br.status), "highs_cap_s": highs_cap,
        }
        if has_highs:
            note, ht, objs = _highs_with_cap(p, C, workers, highs_cap)
            if note == "cap reached":
                row["highs_parallel_time"] = None
                row["highs_note"] = "cap reached"
                print(f"l5 L{level} S={S}: HiGHS cap reached ({highs_cap:.0f}s)")
            elif ht is not None:
                row["highs_parallel_time"] = float(ht)
                ours = np.array([float(C[:, j] @ br.x[:, j] + p.c0) for j in range(S)])
                row["max_rel_obj_diff_vs_highs"] = float(
                    np.max(np.abs(ours - np.asarray(objs, float)) / (1 + np.abs(objs)))
                )
                row["speedup_vs_highs_all_cores"] = (ht / gpu_t) if gpu_t > 0 else None
            else:
                row["highs_error"] = note or "unknown"
        else:
            row["highs_note"] = "highspy not installed"
        rec.add(key, row)
        print(f"l5 L{level} S={S:3d}: GPU {gpu_t:.2f}s certified {ok}/{S} "
              f"HiGHS {row.get('highs_parallel_time', row.get('highs_note'))}")
        free_gpu()


# ====================================================================== refinery case 1 (homotopy)
def part_refinery(rec, quick):
    from qenivo.io.gams import read_gams
    from qenivo.workload.slp import run_homotopy

    key = "case1/homotopy"
    if rec.done(key):
        return
    d = HERE / "instances" / "refinery_benchmark"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "case1.gms"
    if not f.exists():
        url = ("https://raw.githubusercontent.com/EMRPS/refinery-planning-benchmark/"
               "main/case1/case1.gms")
        urllib.request.urlretrieve(url, f)
    M = read_gams(f)
    # Full run: 20 min CPU homotopy. Quick: a few iterations on the same model.
    tl = 60.0 if quick else 20 * 60.0
    max_iter = 3 if quick else 80
    t0 = time.perf_counter()
    r = run_homotopy(M, engine="simplex", backend="numpy", max_iter=max_iter,
                     time_limit=tl, lp_tol=1e-8, tol_viol=1e-6)
    elapsed = time.perf_counter() - t0
    row = {
        "case": 1,
        "size": M.size_str(),
        "phase": r.phase,
        "objective": r.objective,
        "max_violation": r.max_violation,
        "rel_violation": r.rel_violation,
        "feasible": r.feasible,
        "iterations": r.iterations,
        "time": elapsed,
        "status": r.status,
        "time_limit_s": tl,
        "quick": bool(quick),
        "engine": "simplex",
        "note": "T4 homotopy (run_homotopy) on refinery case 1",
    }
    rec.add(key, row)
    print(f"refinery case1 homotopy: obj {r.objective:,.0f} feasible={r.feasible} "
          f"maxviol {r.max_violation:.2e} in {elapsed:.1f}s ({r.status})")


# ====================================================================== headline
def _load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _cupdlpx_from_e2(instance: str, tol: float = 1e-6):
    """Read the existing A100 e2 row; do not rerun cuPDLPx."""
    key = f"{instance}/tol{tol:g}"
    for r in _load_jsonl(E2_CUPDLPX):
        if r.get("key") == key:
            return r.get("cupdlpx"), r
    return None, None


def part_headline(run_dir: Path):
    stamp = _load_jsonl(run_dir / "stamp.jsonl")
    tight = _load_jsonl(run_dir / "tight.jsonl")
    l5 = _load_jsonl(run_dir / "l5.jsonl")
    ref = _load_jsonl(run_dir / "refinery.jsonl")
    L = ["# QENIVO · Round 2 headline (A100)", "",
         "Numbers below are copied from recorded JSONL rows only. Nothing is typed by hand.", ""]

    env = next((r for r in stamp if r.get("key") == "stamp"), None)
    if env:
        L += ["## Environment",
              f"- commit `{env.get('commit')}` · qenivo `{env.get('qenivo')}` · backend `{env.get('backend')}`",
              f"- GPU: `{env.get('nvidia_smi')}` · kernels: `{env.get('gpu_status')}`",
              f"- native simplex: `{env.get('native_simplex')}`", ""]

    L.append("## Tight tolerance (1e-6), 60 s trial cap")
    L.append("")
    L.append("| instance | method | status | time (s) | iters | host max_rel | cuPDLPx wall (from e2) |")
    L.append("|---|---|---|---:|---:|---:|---:|")
    for r in tight:
        if "host_kkt" not in r:
            continue
        inst = r["instance"]
        cup, e2 = _cupdlpx_from_e2(inst)
        if isinstance(cup, dict):
            cup_wall = cup.get("wall", cup.get("status"))
        elif e2 is not None:
            cup_wall = "(e2 row has no cupdlpx field)"
        else:
            cup_wall = "(no e2 row)"
        hk = r["host_kkt"]
        L.append(
            f"| {inst} | {r.get('method')} | {r.get('status')} | "
            f"{r.get('time'):.2f} | {r.get('iterations')} | {hk.get('max_rel'):.2e} | {cup_wall} |"
        )
    L.append("")
    L.append(f"cuPDLPx comparison source: `{E2_CUPDLPX.as_posix()}` (not re-run).")
    L.append("")

    L.append("## L5 stack (tol 1e-4, one repeat, every case certified)")
    L.append("")
    if not l5:
        L.append("_no l5 rows yet_")
    else:
        L.append("| level | rows | S | GPU (s) | certified | HiGHS all-cores (s) | note | speedup |")
        L.append("|---:|---:|---:|---:|---:|---:|---|---:|")
        for r in l5:
            note = r.get("highs_note") or ""
            ht = r.get("highs_parallel_time")
            ht_s = f"{ht:.2f}" if isinstance(ht, (int, float)) else "—"
            sp = r.get("speedup_vs_highs_all_cores")
            sp_s = f"{sp:.2f}" if isinstance(sp, (int, float)) else "—"
            L.append(
                f"| {r.get('level')} | {r.get('rows')} | {r.get('S')} | {r.get('gpu_time'):.2f} | "
                f"{r.get('certified')}/{r.get('S')} | {ht_s} | {note} | {sp_s} |"
            )
    L.append("")

    L.append("## Refinery case 1 · T4 homotopy (CPU)")
    L.append("")
    for r in ref:
        if r.get("key") != "case1/homotopy":
            continue
        L += [
            f"- size `{r.get('size')}` · status `{r.get('status')}` · phase `{r.get('phase')}`",
            f"- objective `{r.get('objective')}` · max violation `{r.get('max_violation')}` "
            f"· feasible `{r.get('feasible')}` · time `{r.get('time'):.1f}s`",
            "",
        ]
    if not any(r.get("key") == "case1/homotopy" for r in ref):
        L.append("_no refinery rows yet_")
        L.append("")

    out = run_dir / "HEADLINE_ROUND2.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


# ====================================================================== driver
PARTS = {
    "stamp": ("stamp", part_stamp),
    "tight": ("tight", part_tight),
    "l5": ("l5", part_l5),
    "refinery": ("refinery", part_refinery),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--only", default="stamp,tight,l5,refinery,headline")
    ap.add_argument("--quick", action="store_true",
                    help="tiny substitutes for CPU smoke (afiro, L1 batches, short homotopy)")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"round2_campaign backend={GPU} quick={a.quick} run_dir={run_dir}")
    for tag in a.only.split(","):
        tag = tag.strip()
        if not tag:
            continue
        if tag == "headline":
            part_headline(run_dir)
            continue
        if tag not in PARTS:
            raise SystemExit(f"unknown part {tag!r}; choose from {list(PARTS)+['headline']}")
        name, fn = PARTS[tag]
        rec = Recorder(run_dir, name)
        t0 = time.perf_counter()
        try:
            fn(rec, a.quick)
            print(f"--- {tag} done in {time.perf_counter()-t0:.1f}s")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print(f"--- {tag} FAILED after {time.perf_counter()-t0:.1f}s (saved rows kept)")
            raise


if __name__ == "__main__":
    # Required: highs_pool_time uses multiprocessing spawn; keep the script body here.
    main()
