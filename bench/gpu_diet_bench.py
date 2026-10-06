"""W04 GPU byte-diet ablation bench.

    python bench/gpu_diet_bench.py --quick
    python bench/gpu_diet_bench.py --time-limit 1800 --out bench/results/gpu_diet_rtx5050_2026-10-01.jsonl

Progress is printed (and flushed) for EVERY config. Resume skips keys already in the JSONL.
Compulsory GB/s uses the 04-report model: (10n + 9m) * itemsize * S * iterations / wall.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from qenivo.certify.kkt import is_optimal, kkt_from_products, rhs_norm  # noqa: E402
from qenivo.engines import pdhg  # noqa: E402
from qenivo.kernels.backend import gpu_available, gpu_info  # noqa: E402
from qenivo.kernels.byte_diet import ablation_flags, device_l2_bytes, s_tile  # noqa: E402
from qenivo.models.refinery import LEVELS, refinery_lp  # noqa: E402

ABLATIONS = ("baseline", "A1", "A2", "A3", "A4")


def _log(msg: str):
    print(msg, flush=True)


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


class Recorder:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keys = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self.keys.add(json.loads(line).get("key"))
                except json.JSONDecodeError:
                    pass

    def done(self, key: str) -> bool:
        return key in self.keys

    def add(self, key: str, row: dict):
        row = {"key": key, "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **row}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=_json) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.keys.add(key)


def smi() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.free,memory.used,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False)
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        if len(parts) >= 6:
            return {"gpu_name": parts[0], "mem_free_mib": float(parts[1]), "mem_used_mib": float(parts[2]),
                    "temp_c": float(parts[3]), "sm_clock_mhz": float(parts[4]),
                    "mem_clock_mhz": float(parts[5]),
                    "power_w": float(parts[6]) if len(parts) > 6 and parts[6] not in ("", "[N/A]") else None}
    except Exception:  # noqa: BLE001
        pass
    return {}


def compulsory_gbs(n, m, S, iters, wall, dtype):
    """04-report model: (10n+9m) words per case-iteration."""
    if wall <= 0:
        return None
    vs = np.dtype(dtype).itemsize
    bytes_ = (10 * n + 9 * m) * vs * S * float(np.mean(iters))
    return bytes_ / wall / 1e9


def certify_batch(p, C, br, tol, chunk: int = 16):
    """Host f64 KKT in column chunks (avoids one huge A@X on L4×S)."""
    X, Y = br.x, br.y
    S = X.shape[1]
    ok = np.zeros(S, dtype=bool)
    worst = 0.0
    status_ok = np.array([s == "optimal" for s in br.status])
    for lo in range(0, S, max(1, int(chunk))):
        hi = min(S, lo + chunk)
        sl = slice(lo, hi)
        k = kkt_from_products(X[:, sl], Y[:, sl], p.A @ X[:, sl], p.A.T @ Y[:, sl],
                              C[:, sl], p.c0, p.lc[:, None], p.uc[:, None],
                              p.lx[:, None], p.ux[:, None],
                              rhs_norm(p.lc, p.uc), np.linalg.norm(C[:, sl], axis=0))
        ok[sl] = np.asarray(is_optimal(k, tol * 1.0001)) & status_ok[sl]
        worst = max(worst, float(np.max(np.maximum(np.maximum(
            k["rel_primal"], k["rel_dual"]), k["rel_gap"]))))
    return int(ok.sum()), worst


def _free_gpu():
    try:
        import gc

        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()
    except Exception:  # noqa: BLE001
        pass


def _ensure_cuda_path():
    """Silence CuPy CUDA_PATH warnings when a toolkit tree exists; no-op otherwise."""
    if os.environ.get("CUDA_PATH"):
        return
    base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if not base.is_dir():
        return
    for child in sorted(base.iterdir(), reverse=True):
        if child.is_dir() and (child / "bin").is_dir():
            os.environ["CUDA_PATH"] = str(child)
            return


def run_one(p, C, ablation, tol, backend, time_limit, reps):
    flags = ablation_flags(ablation)
    # f64 @ 1e-6: force baseline-like precision even under A1+ names when tol is tight
    if tol < 1e-4:
        flags = {**flags, "dtype": "float64", "diet_f32_cert": False}
    walls, iters, certified, worsts = [], [], [], []
    last = None
    for r in range(reps):
        if backend == "cupy":
            _free_gpu()
        opts = pdhg.PDHGOptions(
            tol=tol, time_limit=time_limit, persistent="auto", detect_infeasibility=False,
            **{k: v for k, v in flags.items() if k in pdhg.PDHGOptions.__dataclass_fields__})
        t0 = time.perf_counter()
        br = pdhg.solve_batch(p.A, C, p.lc, p.uc, p.lx, p.ux, p.c0, opts, backend)
        if backend == "cupy":
            import cupy as cp
            cp.cuda.Device().synchronize()
        wall = time.perf_counter() - t0
        n_ok, worst = certify_batch(p, C, br, tol)
        walls.append(wall)
        iters.append(float(np.mean(br.iterations)))
        certified.append(n_ok)
        worsts.append(worst)
        last = br
        _log(f"    rep {r+1}/{reps}: wall={wall:.3f}s iters_mean={iters[-1]:.0f} "
             f"certified={n_ok}/{C.shape[1]} worst_kkt={worst:.2e} status0={br.status[0]}")
        if backend == "cupy":
            _free_gpu()
    order = np.argsort(walls)
    mid = order[len(order) // 2]
    dtype = flags.get("dtype", "float64")
    return {
        "wall_s": walls[mid], "walls": walls, "iters_mean": iters[mid], "iters_all": iters,
        "certified": certified[mid], "certified_all": certified, "worst_kkt": worsts[mid],
        "dtype": dtype, "setup_s": last.setup_time if last else None,
        "solve_s": last.solve_time if last else None,
        "us_per_iter": 1e6 * walls[mid] / max(iters[mid], 1.0),
        "gbs_compulsory": compulsory_gbs(p.n, p.m, C.shape[1], np.array([iters[mid]]), walls[mid], dtype),
        "s_tile": s_tile(p.n, p.m, dtype, shared_bounds=bool(flags.get("diet_shared_bounds"))),
        "l2_bytes": device_l2_bytes() if backend == "cupy" else None,
        "statuses": list(last.status) if last else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=HERE / "results" / "gpu_diet_rtx5050_2026-10-01.jsonl")
    ap.add_argument("--time-limit", type=float, default=600.0, help="per-solve wall budget (s)")
    ap.add_argument("--budget", type=float, default=None, help="total wall budget for the whole run")
    ap.add_argument("--quick", action="store_true", help="L3×S={1,16} × baseline,A1; 1 rep; numpy if no GPU")
    ap.add_argument("--levels", default=None, help="comma levels, default 3,4")
    ap.add_argument("--S", default=None, help="comma batch sizes")
    ap.add_argument("--ablations", default=None, help="comma names from baseline,A1,A2,A3,A4")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--also-1e-6", action="store_true", help="also run f64 @ 1e-6 on baseline")
    ap.add_argument("--max-temp", type=float, default=87.0)
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    _ensure_cuda_path()
    backend = args.backend or ("cupy" if gpu_available() else "numpy")
    if args.quick:
        levels = [3]
        Ss = [1, 16]
        ablations = ["baseline", "A1"]
        reps = 1
        args.time_limit = min(args.time_limit, 90.0)
    else:
        levels = [int(x) for x in (args.levels or "3,4").split(",")]
        Ss = [int(x) for x in (args.S or "1,16,64,256").split(",")]
        ablations = [a.strip() for a in (args.ablations or ",".join(ABLATIONS)).split(",") if a.strip()]
        reps = args.reps

    rec = Recorder(args.out)
    info = {"gpu": gpu_info(), "backend": backend, **smi()}
    _log(f"gpu_diet_bench start backend={backend} out={args.out} levels={levels} S={Ss} "
         f"ablations={ablations} reps={reps} tol={args.tol}")
    _log(f"env: {json.dumps(info, default=_json)}")
    if backend == "cupy":
        free_gb = (info.get("mem_free_mib") or 0) / 1024.0
        temp = info.get("temp_c") or 0
        _log(f"GPU free={free_gb:.2f} GiB temp={temp}C")
        if free_gb < 2.0:
            _log("ABORT: free GPU RAM < 2.0 GiB")
            return 2
        if temp > args.max_temp:
            _log(f"ABORT: GPU temp {temp}C > {args.max_temp}C")
            return 3

    t_run = time.perf_counter()
    for lv in levels:
        p = refinery_lp(*LEVELS[lv], seed=0)
        _log(f"level L{lv}: m={p.m} n={p.n} nnz={p.nnz}")
        for S in Ss:
            C = p.c[:, None] * np.random.default_rng(10 + lv).uniform(0.9, 1.1, (p.n, S))
            for abl in ablations:
                key = f"L{lv}|S{S}|{abl}|tol{args.tol:g}|{backend}"
                _log(f"CONFIG {key}")
                if rec.done(key):
                    _log("  skip (resume)")
                    continue
                if args.budget is not None and time.perf_counter() - t_run > args.budget:
                    _log("ABORT: total --budget exceeded")
                    return 0
                if backend == "cupy":
                    snap = smi()
                    if (snap.get("temp_c") or 0) > args.max_temp:
                        _log(f"ABORT: GPU temp {snap['temp_c']}C")
                        return 3
                    if (snap.get("mem_free_mib") or 0) < 2048:
                        _log(f"ABORT: free GPU RAM {snap.get('mem_free_mib')} MiB < 2 GiB")
                        return 2
                    _free_gpu()
                try:
                    row = run_one(p, C, abl, args.tol, backend, args.time_limit, reps)
                    row.update(level=lv, S=S, ablation=abl, tol=args.tol, backend=backend,
                               n=p.n, m=p.m, nnz=p.nnz, **smi())
                    rec.add(key, row)
                    _log(f"  DONE median_wall={row['wall_s']:.3f}s certified={row['certified']}/{S} "
                         f"us/iter={row['us_per_iter']:.1f} GBps={row['gbs_compulsory']}")
                except Exception as e:  # noqa: BLE001
                    rec.add(key, {"error": f"{type(e).__name__}: {e}", "level": lv, "S": S,
                                  "ablation": abl, "tol": args.tol, "backend": backend, **smi()})
                    _log(f"  ERROR {type(e).__name__}: {e}")
                    if backend == "cupy":
                        _free_gpu()
                if backend == "cupy":
                    _free_gpu()
            if args.also_1e_6 and not args.quick:
                key = f"L{lv}|S{S}|baseline|tol1e-06|{backend}"
                _log(f"CONFIG {key}")
                if not rec.done(key):
                    try:
                        row = run_one(p, C, "baseline", 1e-6, backend, args.time_limit, reps)
                        row.update(level=lv, S=S, ablation="baseline", tol=1e-6, backend=backend,
                                   n=p.n, m=p.m, nnz=p.nnz, **smi())
                        rec.add(key, row)
                        _log(f"  DONE 1e-6 wall={row['wall_s']:.3f}s certified={row['certified']}/{S}")
                    except Exception as e:  # noqa: BLE001
                        rec.add(key, {"error": f"{type(e).__name__}: {e}", "tol": 1e-6, **smi()})
                        _log(f"  ERROR {e}")
    _log(f"gpu_diet_bench finished in {time.perf_counter()-t_run:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
