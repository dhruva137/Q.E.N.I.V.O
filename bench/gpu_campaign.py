"""The A100 evidence campaign: one run, every GPU claim measured fairly, certified and stamped.

    python bench/gpu_campaign.py --run-dir /content/drive/MyDrive/qenivo/run1          # everything
    python bench/gpu_campaign.py --run-dir runs/smoke --quick                               # CPU smoke test
    python bench/gpu_campaign.py --run-dir <same dir> --only e3                            # resume one part

Crash safety: every result row is appended to <run-dir>/<experiment>.jsonl and fsync'ed the moment
it exists. On restart with the same --run-dir, rows already present are skipped (resume). Each
experiment has a wall-clock budget; an experiment that fails is recorded and the next one starts.

Experiments
  e0 env        environment, GPU, driver, library versions, the test suite on this machine
  e1 kernels    own CUDA SpMV/SpMM vs the vendor library: agreement, bit-for-bit repeatability, GB/s
  e2 large_lp   large Mittelmann LPs at 1e-4 and 1e-6 on the GPU, 3 repeats, host-checked;
                cuPDLPx on the SAME GPU (its answer re-checked by us); HiGHS IPM on all CPU cores
  e3 batch      refinery families x batch size x tolerance: GPU batch vs HiGHS in parallel on ALL
                CPU cores (the fair baseline), the serial warm chain, and our CPU path;
                every scenario certified (vectorised batch KKT check)
  e4 refinery   the open industrial refinery-petrochemical benchmark (Du et al. 2025): penalty SLP
                with warm-started GPU LPs vs the published BARON / ANTIGONE 18,000 s results
  e5 recursion  distributive recursion on pooled refinery families: warm first-order vs cold vs IPM
  e6 learned    learned warm starts: a network trained on GPU-solved scenarios; every answer certified
  e7 headline   HEADLINE.md + PNG charts for the slides, built only from the rows above
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import multiprocessing as mp
import os
import platform
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

import qenivo  # noqa: E402
from qenivo.certify.kkt import is_optimal, kkt_from_products, kkt_residuals, rhs_norm  # noqa: E402
from qenivo.engines import pdhg  # noqa: E402
from qenivo.kernels.backend import gpu_available, gpu_info  # noqa: E402
from qenivo.models.refinery import LEVELS, refinery_lp  # noqa: E402

GPU = "cupy" if gpu_available() else "numpy"


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
                    pass                                   # a line cut by a crash is dropped
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


def env() -> dict:
    def run(*a):
        try:
            return subprocess.run(list(a), capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    import scipy
    out = {"qenivo": qenivo.__version__, "commit": run("git", "-C", str(ROOT), "rev-parse", "HEAD"),
           "dirty": bool(run("git", "-C", str(ROOT), "status", "--porcelain")), "python": platform.python_version(),
           "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
           "cpu": run("bash", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2") or platform.processor(),
           "cpu_count": os.cpu_count(), "gpu": gpu_info(), "backend": GPU,
           "nvidia_smi": run("nvidia-smi", "--query-gpu=name,memory.total,driver_version,clocks.max.sm", "--format=csv,noheader")}
    try:
        import cupy
        out["cupy"] = cupy.__version__
    except Exception:  # noqa: BLE001
        pass
    return out


def sync():
    if GPU == "cupy":
        import cupy
        cupy.cuda.Device().synchronize()


def free_gpu():
    if GPU == "cupy":
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()


def certify_batch(p, C, br, tol):
    """Vectorised KKT check of every scenario of a batch on the host (one pass over A)."""
    X, Y = br.x, br.y
    k = kkt_from_products(X, Y, p.A @ X, p.A.T @ Y, C, p.c0, p.lc[:, None], p.uc[:, None],
                          p.lx[:, None], p.ux[:, None], rhs_norm(p.lc, p.uc), np.linalg.norm(C, axis=0))
    ok = is_optimal(k, tol * 1.0001) & np.array([s == "optimal" for s in br.status])
    return int(ok.sum()), float(np.max(np.maximum(np.maximum(k["rel_primal"], k["rel_dual"]), k["rel_gap"])))


# ====================================================================== e0
def e0_env(rec, quick):
    if not rec.done("env"):
        rec.add("env", env())
    if not rec.done("tests"):
        t = time.perf_counter()
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")],
                           capture_output=True, text=True, timeout=1800)
        tail = [line for line in p.stdout.splitlines() if line.strip()][-3:]
        rec.add("tests", {"returncode": p.returncode, "summary": tail, "time": time.perf_counter() - t})
        print("e0 tests:", tail[-1] if tail else p.stderr[-300:])


# ====================================================================== e1
def e1_kernels(rec, quick):
    if GPU != "cupy":
        rec.add("skipped", {"reason": "no GPU"}) if not rec.done("skipped") else None
        return
    import cupy as cp
    from fetch import fetch

    from qenivo.kernels.cuda_kernels import GPUMatrix
    names = ["afiro", "sc50a"] if quick else ["qap15", "nug08-3rd", "savsched1", "rmine15", "cont1"]
    for name in names:
        A = qenivo.read(fetch(name, verbose=False)).A
        own, ven = GPUMatrix(A), GPUMatrix(A, use_vendor=True)
        for S in (1, 8, 32, 128):
            key = f"{name}/S{S}"
            if rec.done(key):
                continue
            X = cp.asarray(np.random.default_rng(0).standard_normal((A.shape[1], S)))
            ref = A @ cp.asnumpy(X)
            y1, y2 = own.mv(X), own.mv(X)
            row = {"instance": name, "S": S, "rows": A.shape[0], "nnz": A.nnz,
                   "bitwise_repeatable": bool(cp.all(y1 == y2)),
                   "rel_err_vs_cpu": float(np.max(np.abs(cp.asnumpy(y1) - ref)) / (1 + np.max(np.abs(ref))))}
            for label, M in (("qenivo", own), ("vendor", ven)):
                for _ in range(3):
                    M.mv(X)
                sync()
                ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
                reps = 50
                ev0.record()
                for _ in range(reps):
                    M.mv(X)
                ev1.record(); ev1.synchronize()
                ms = cp.cuda.get_elapsed_time(ev0, ev1) / reps
                bytes_moved = A.nnz * 12 + A.shape[0] * 4 + (A.nnz + A.shape[0]) * 8 * S
                row[f"{label}_ms"] = ms
                row[f"{label}_GBps"] = bytes_moved / (ms / 1e3) / 1e9
            rec.add(key, row)
            print(f"e1 {name:10s} S={S:4d} own {row['qenivo_ms']:.3f} ms ({row['qenivo_GBps']:.0f} GB/s) "
                  f"vendor {row['vendor_ms']:.3f} ms  bitwise={row['bitwise_repeatable']}")
        free_gpu()


# ====================================================================== e2
def _highs_ipm_all_cores(path, tl):
    code = ("import highspy,sys,time;h=highspy.Highs();h.setOptionValue('output_flag',False);"
            f"h.setOptionValue('time_limit',{tl});h.setOptionValue('solver','ipm');h.readModel(sys.argv[1]);"
            "t=time.perf_counter();h.run();print('RES',h.modelStatusToString(h.getModelStatus()).replace(' ','_'),"
            "h.getInfo().objective_function_value,time.perf_counter()-t)")
    try:
        p = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, timeout=tl + 120)
        for line in p.stdout.splitlines():
            if line.startswith("RES"):
                _, st, obj, t = line.split()
                return {"status": st, "objective": float(obj), "time": float(t)}
        return {"status": "error", "stderr": p.stderr[-300:]}
    except subprocess.TimeoutExpired:
        return {"status": "killed", "time": tl + 120}


def _cupdlpx(path, tol, tl):
    exe = os.environ.get("CUPDLPX_BIN")
    if not exe or not Path(exe).exists():
        return {"status": "not available"}
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        t = time.perf_counter()
        try:
            p = subprocess.run([exe, "-q", "--eps_opt", str(tol), "--eps_feas", str(tol), "--time_limit", str(tl),
                                str(path), td], capture_output=True, text=True, timeout=tl + 120)
        except subprocess.TimeoutExpired:
            return {"status": "killed", "wall": tl + 120}
        out = {"status": "ran", "wall": time.perf_counter() - t, "returncode": p.returncode}
        stem = Path(path).name.split(".")[0]
        s = Path(td) / f"{stem}_summary.txt"
        if s.exists():
            out["summary"] = s.read_text()[:3000]
        xs = Path(td) / f"{stem}_primal_solution.txt"
        if xs.exists():
            x = np.loadtxt(xs).reshape(-1)
            prob = qenivo.read(path)
            if x.size == prob.n:
                ax = prob.A @ x
                viol = np.maximum(prob.lc - ax, 0) + np.maximum(ax - prob.uc, 0)
                out["our_check_rel_primal"] = float(np.linalg.norm(viol) / (1 + rhs_norm(prob.lc, prob.uc)))
                out["objective"] = prob.objective(x)
        return out


def e2_large_lp(rec, quick):
    from fetch import fetch
    names = ["afiro"] if quick else ["qap15", "nug08-3rd", "savsched1", "rmine15", "Linf_520c", "cont1", "pds-100"]
    tl = 60 if quick else 900
    reps = 1 if quick else 3
    for name in names:
        path = fetch(name, verbose=False)
        prob = qenivo.read(path)
        for tol in (1e-4, 1e-6):
            key = f"{name}/tol{tol:g}"
            if rec.done(key):
                continue
            ts, col = [], None
            for r in range(reps):
                t = time.perf_counter()
                br = pdhg.solve(prob, tol=tol, backend=GPU, time_limit=tl)
                sync()
                ts.append(time.perf_counter() - t)
                col = br.column(0)
                if col["status"] != "optimal":
                    break
            k = kkt_residuals(prob, col["x"], col["y"])
            row = {"instance": name, "rows": prob.m, "cols": prob.n, "nnz": prob.nnz, "tol": tol, "backend": GPU,
                   "status": col["status"], "iterations": col["iterations"], "times": ts,
                   "time_median": float(np.median(ts)), "objective": k["objective"], "max_rel_host": k["max_rel"],
                   "certified": bool(col["status"] == "optimal" and k["max_rel"] <= tol * 1.0001)}
            row["cupdlpx"] = _cupdlpx(path, tol, tl)
            rec.add(key, row)
            print(f"e2 {name:10s} tol {tol:.0e}: {row['status']} {row['time_median']:.2f}s it {row['iterations']} "
                  f"maxrel {row['max_rel_host']:.1e} | cuPDLPx {row['cupdlpx'].get('wall', row['cupdlpx']['status'])}")
            free_gpu()
        key = f"{name}/highs_ipm"
        if not rec.done(key):
            rec.add(key, {"instance": name, **_highs_ipm_all_cores(path, tl)})


# ====================================================================== e3
def _scenarios(level, S, seed=0):
    R, T, K, Mk = LEVELS[level]
    p = refinery_lp(R, T, K, Mk, seed=seed)
    rng = np.random.default_rng(1000 + seed + level)
    return p, p.c[:, None] * rng.uniform(0.9, 1.1, (p.n, S))


def _highs_slice(args):
    A, C, lc, uc, lx, ux = args
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = C[:, 0], lx, ux
    lp.row_lower_, lp.row_upper_ = lc, uc
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    h.passModel(lp)
    idx = np.arange(A.shape[1], dtype=np.int32)
    objs = []
    for s in range(C.shape[1]):
        if s:
            h.changeColsCost(A.shape[1], idx, C[:, s])       # warm start from the previous basis
        h.run()
        objs.append(h.getInfo().objective_function_value)
    return objs


def highs_pool_time(p, C, workers):
    """HiGHS on `workers` cores, each warm-starting along its slice. Pool started and warmed
    BEFORE the clock: the time is solve time only (the fairest CPU baseline)."""
    S = C.shape[1]
    chunks = [ch for ch in np.array_split(np.arange(S), min(workers, S)) if len(ch)]
    A = p.A.tocsc()
    with mp.get_context("spawn").Pool(len(chunks)) as pool:
        pool.map(_highs_slice, [(A, C[:, ch[:1]], p.lc, p.uc, p.lx, p.ux) for ch in chunks])
        t = time.perf_counter()
        res = pool.map(_highs_slice, [(A, C[:, ch], p.lc, p.uc, p.lx, p.ux) for ch in chunks])
        return time.perf_counter() - t, np.array([o for r in res for o in r])


def e3_batch(rec, quick):
    try:
        import highspy  # noqa: F401
        has_highs = True
    except Exception:  # noqa: BLE001
        has_highs = False
    workers = os.cpu_count() or 1
    levels = (1,) if quick else (3, 4, 5)
    batches = (1, 4) if quick else (1, 8, 32, 128, 512)
    tols = (1e-4,) if quick else (1e-4, 1e-6)
    reps = 1 if quick else 3
    for lv in levels:
        for S in batches:
            p, C = _scenarios(lv, S)
            for tol in tols:
                key = f"L{lv}/S{S}/tol{tol:g}"
                if rec.done(key):
                    continue
                opts = pdhg.PDHGOptions(tol=tol, time_limit=3600)
                ts, br = [], None
                for _ in range(reps):
                    t = time.perf_counter()
                    br = pdhg.solve_batch(p.A, C, p.lc, p.uc, p.lx, p.ux, p.c0, opts, GPU)
                    sync()
                    ts.append(time.perf_counter() - t)
                ok, worst = certify_batch(p, C, br, tol)
                row = {"level": lv, "rows": p.m, "cols": p.n, "nnz": p.nnz, "S": S, "tol": tol,
                       "gpu_times": ts, "gpu_median": float(np.median(ts)), "certified": ok, "worst_rel": worst,
                       "iterations_max": int(br.iterations.max()), "cpu_workers": workers}
                ours = np.array([float(C[:, j] @ br.x[:, j] + p.c0) for j in range(S)])
                if has_highs and tol == tols[0]:
                    tp, objs = highs_pool_time(p, C, workers)
                    row["highs_parallel_time"] = tp
                    row["max_rel_obj_diff_vs_highs"] = float(np.max(np.abs(ours - objs) / (1 + np.abs(objs))))
                    if S <= 32:
                        row["highs_serial_time"] = highs_pool_time(p, C, 1)[0]
                base = row.get("highs_parallel_time")
                row["speedup_vs_highs_all_cores"] = (base / row["gpu_median"]) if base else None
                rec.add(key, row)
                print(f"e3 L{lv} S={S:4d} tol {tol:.0e}: GPU {row['gpu_median']:7.2f}s  certified {ok}/{S}  "
                      f"HiGHS x{workers} {row.get('highs_parallel_time')}  speedup {row['speedup_vs_highs_all_cores']}")
                free_gpu()


# ====================================================================== e4
BARON = {1: {"best_found": 34168000.0, "best_possible": 37910846.9074, "time_s": 18015},
         2: {"best_found": 67303200.0, "best_possible": 76374974.0096, "time_s": 18020},
         3: {"best_found": 125250000.0, "best_possible": 135294529.942, "time_s": 18026}}


def e4_refinery(rec, quick):
    import urllib.request

    from qenivo.io.gams import read_gams
    from qenivo.workload.slp import run_slp
    base = "https://raw.githubusercontent.com/EMRPS/refinery-planning-benchmark/main"
    d = HERE / "instances" / "refinery_benchmark"
    d.mkdir(parents=True, exist_ok=True)
    cases = (1,) if quick else (1, 2, 3)
    for c in cases:
        f = d / f"case{c}.gms"
        if not f.exists():
            urllib.request.urlretrieve(f"{base}/case{c}/case{c}.gms", f)
        M = read_gams(f)
        for eng in (("pdhg",) if quick else ("pdhg", "ipm")):
            key = f"case{c}/{eng}"
            if rec.done(key) or (eng == "ipm" and c != 1):       # dense IPM only fits case 1 in budget
                continue
            be = GPU if eng == "pdhg" else "numpy"
            integer = M.linear.integer
            t0 = time.perf_counter()
            r = run_slp(M, engine=eng, backend=be, max_iter=3 if quick else 120,
                        time_limit=300 if quick else 1500, lp_tol=1e-6 if eng == "pdhg" else 1e-8)
            note = "continuous"
            if integer is not None and integer.any():
                # binaries: round the relaxed plan, fix them, re-run SLP on the continuous rest
                x = r.x.copy()
                x[integer] = np.round(x[integer])
                M.linear.lx[integer] = M.linear.ux[integer] = x[integer]
                r = run_slp(M, x0=x, engine=eng, backend=be, max_iter=3 if quick else 80,
                            time_limit=300 if quick else 900, lp_tol=1e-6 if eng == "pdhg" else 1e-8)
                note = "binaries fixed by rounding the relaxed SLP plan"
            ref = BARON[c]
            row = {"case": c, "size": M.size_str(), "engine": eng, "backend": be, "objective": r.objective,
                   "max_violation": r.max_violation, "rel_violation": r.rel_violation, "feasible": r.feasible,
                   "iterations": r.iterations, "time": time.perf_counter() - t0, "status": r.status, "note": note,
                   "baron_best_found": ref["best_found"], "baron_best_possible": ref["best_possible"],
                   "baron_time_s": ref["time_s"],
                   "vs_baron_best_found": r.objective / ref["best_found"] - 1.0,
                   "gap_to_baron_bound": (ref["best_possible"] - r.objective) / ref["best_possible"]}
            rec.add(key, row)
            print(f"e4 case{c} {eng}: objective {r.objective:,.0f} feasible={r.feasible} ({r.rel_violation:.1e}) "
                  f"in {row['time']:.0f}s | BARON {ref['best_found']:,.0f} in {ref['time_s']} s")
            free_gpu()


# ====================================================================== e5
def e5_recursion(rec, quick):
    from qenivo.workload.recursion import refinery_pools, run_recursion
    levels = (1,) if quick else (2, 3, 4)
    seeds = (0,) if quick else (0, 1, 2)
    for lv in levels:
        R, T, K, Mk = LEVELS[lv]
        for seed in seeds:
            p = refinery_lp(R, T, K, Mk, seed=seed, names=True, pooled=True)
            pools = refinery_pools(p)
            for mth in ("pdhg_warm", "pdhg_cold", "ipm"):
                key = f"L{lv}/s{seed}/{mth}"
                if rec.done(key) or (mth == "ipm" and p.m > 5500):
                    continue
                be = GPU if mth.startswith("pdhg") else "numpy"
                r = run_recursion(p, pools, method=mth, q0=80.0, tol=1e-6, q_tol=1e-4, max_passes=60, backend=be)
                rec.add(key, {"level": lv, "seed": seed, "rows": p.m, "method": mth, "backend": be,
                              "converged": r.converged, "passes": len(r.passes),
                              "time": sum(x["time"] for x in r.passes),
                              "objective": r.passes[-1]["objective"] if r.passes else None,
                              "dq_trace": [x["max_dq"] for x in r.passes]})
                print(f"e5 L{lv} s{seed} {r.summary()}")
                free_gpu()


# ====================================================================== e6
def e6_learned(rec, quick):
    if rec.done("result"):
        return
    try:
        import torch
    except Exception:  # noqa: BLE001
        rec.add("result", {"skipped": "torch not installed"})
        return
    level, n_train, n_test, epochs = (1, 64, 16, 3) if quick else (3, 8192, 512, 80)
    R, T, K, Mk = LEVELS[level]
    p = refinery_lp(R, T, K, Mk, seed=0)
    rng = np.random.default_rng(7)
    n_par = 16
    groups = rng.integers(0, n_par, p.n)

    def make(nS):
        P = rng.uniform(0.85, 1.15, (n_par, nS))
        return P, p.c[:, None] * P[groups, :]
    opts = pdhg.PDHGOptions(tol=1e-6, time_limit=3600)
    Ptr, Ctr = make(n_train)
    t = time.perf_counter()
    Xs, Ys = [], []
    for b0 in range(0, n_train, 512):
        br = pdhg.solve_batch(p.A, Ctr[:, b0:b0 + 512], p.lc, p.uc, p.lx, p.ux, p.c0, opts, GPU)
        Xs.append(br.x); Ys.append(br.y)
    t_data = time.perf_counter() - t
    X, Y = np.concatenate(Xs, 1), np.concatenate(Ys, 1)
    dev = "cpu"
    if torch.cuda.is_available():
        try:                                             # self-test: some driver/card pairs fail in cuBLAS
            a = torch.randn(64, 64, device="cuda"); (a @ a).sum().item()
            dev = "cuda"
        except Exception as e:  # noqa: BLE001
            print("e6: CUDA self-test failed, training on CPU:", e)
    inp = torch.tensor(Ptr.T, dtype=torch.float32, device=dev)
    tgt = torch.tensor(np.concatenate([X, Y], 0).T, dtype=torch.float32, device=dev)
    mu, sd = tgt.mean(0), tgt.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(n_par, 512), torch.nn.GELU(), torch.nn.Linear(512, 512),
                              torch.nn.GELU(), torch.nn.Linear(512, tgt.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3)
    t = time.perf_counter()
    for _ in range(epochs):
        perm = torch.randperm(n_train, device=dev)
        for b in range(0, n_train, 256):
            idx = perm[b:b + 256]
            loss = torch.nn.functional.mse_loss(net(inp[idx]), (tgt[idx] - mu) / sd)
            opt.zero_grad(); loss.backward(); opt.step()
    t_train = time.perf_counter() - t
    Pte, Cte = make(n_test)
    with torch.no_grad():
        pred = (net(torch.tensor(Pte.T, dtype=torch.float32, device=dev)) * sd + mu).cpu().numpy().T.astype(float)
    t = time.perf_counter()
    cold = pdhg.solve_batch(p.A, Cte, p.lc, p.uc, p.lx, p.ux, p.c0, opts, GPU); sync()
    t_cold = time.perf_counter() - t
    t = time.perf_counter()
    warm = pdhg.solve_batch(p.A, Cte, p.lc, p.uc, p.lx, p.ux, p.c0, opts, GPU, warm_x=pred[:p.n], warm_y=pred[p.n:]); sync()
    t_warm = time.perf_counter() - t
    ok_c, _ = certify_batch(p, Cte, cold, 1e-6)
    ok_w, worst = certify_batch(p, Cte, warm, 1e-6)
    row = {"level": level, "rows": p.m, "n_train": n_train, "n_test": n_test, "data_time": t_data,
           "train_time": t_train, "cold_iters_median": float(np.median(cold.iterations)),
           "warm_iters_median": float(np.median(warm.iterations)), "cold_time": t_cold, "warm_time": t_warm,
           "cold_certified": ok_c, "warm_certified": ok_w, "warm_worst_rel": worst}
    rec.add("result", row)
    print(f"e6 learned warm start: iterations {row['cold_iters_median']:.0f} -> {row['warm_iters_median']:.0f}, "
          f"time {t_cold:.1f}s -> {t_warm:.1f}s, certified {ok_w}/{n_test}")


# ====================================================================== e7
def e7_headline(run_dir: Path):
    """Build HEADLINE.md and charts from the recorded rows only."""
    def load(n):
        f = run_dir / f"{n}.jsonl"
        return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()] if f.exists() else []
    envr = next((r for r in load("e0_env") if r["key"] == "env"), {})
    L = [f"# Headline evidence  (run {run_dir.name})", "",
         f"GPU: {envr.get('nvidia_smi')}  |  commit {str(envr.get('commit'))[:10]}  |  backend {envr.get('backend')}", ""]
    e2 = [r for r in load("e2_large_lp") if "tol" in r]
    if e2:
        L += ["## Large LPs on the GPU (certified on the host)", "",
              "| instance | nnz | tol | status | time (s) | cuPDLPx wall (s) |", "|---|---|---|---|---|---|"]
        for r in e2:
            L.append(f"| {r['instance']} | {r['nnz']:,} | {r['tol']:g} | {r['status']} | {r['time_median']:.2f} | "
                     f"{r['cupdlpx'].get('wall', r['cupdlpx'].get('status'))} |")
        L.append("")
    e3 = load("e3_batch")
    if e3:
        L += ["## What-if batches vs HiGHS on ALL CPU cores", "",
              "| level | rows | S | tol | GPU (s) | certified | HiGHS all cores (s) | speed-up |", "|---|---|---|---|---|---|---|---|"]
        for r in e3:
            sp = r.get("speedup_vs_highs_all_cores")
            L.append(f"| L{r['level']} | {r['rows']:,} | {r['S']} | {r['tol']:g} | {r['gpu_median']:.2f} | "
                     f"{r['certified']}/{r['S']} | {r.get('highs_parallel_time') or '-'} | {f'{sp:.2f}x' if sp else '-'} |")
        L.append("")
    e4 = load("e4_refinery")
    if e4:
        L += ["## Open industrial refinery benchmark (Du et al. 2025) vs BARON 18,000 s", "",
              "| case | engine | our plan | feasible | our time (s) | BARON plan | BARON bound |", "|---|---|---|---|---|---|---|"]
        for r in e4:
            L.append(f"| {r['case']} | {r['engine']} | {r['objective']:,.0f} | {r['feasible']} | {r['time']:.0f} | "
                     f"{r['baron_best_found']:,.0f} | {r['baron_best_possible']:,.0f} |")
        L.append("")
    e6 = [r for r in load("e6_learned") if r.get("key") == "result" and "cold_iters_median" in r]
    if e6:
        r = e6[0]
        L += ["## Learned warm starts (answers still certified)", "",
              f"iterations {r['cold_iters_median']:.0f} -> {r['warm_iters_median']:.0f}; time {r['cold_time']:.1f}s -> "
              f"{r['warm_time']:.1f}s; certified {r['warm_certified']}/{r['n_test']}", ""]
    (run_dir / "HEADLINE.md").write_text("\n".join(L), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if e3:
            fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
            for lv in sorted({r["level"] for r in e3}):
                rs = sorted([r for r in e3 if r["level"] == lv and r["tol"] == 1e-4 and r.get("speedup_vs_highs_all_cores")],
                            key=lambda r: r["S"])
                if rs:
                    ax.plot([r["S"] for r in rs], [r["speedup_vs_highs_all_cores"] for r in rs], marker="o",
                            label=f"L{lv} ({rs[0]['rows']:,} rows)")
            ax.axhline(1, color="grey", lw=0.8, ls="--")
            ax.set_xscale("log", base=2); ax.set_xlabel("what-if cases solved together")
            ax.set_ylabel("speed-up vs HiGHS on all CPU cores"); ax.legend(frameon=False)
            ax.set_title("GPU batch speed-up, every case certified")
            fig.tight_layout(); fig.savefig(run_dir / "speedup.png"); plt.close(fig)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    print((run_dir / "HEADLINE.md").read_text(encoding="utf-8"))


# ====================================================================== driver
EXPERIMENTS = {"e0": ("e0_env", e0_env, 1800), "e1": ("e1_kernels", e1_kernels, 1200),
               "e2": ("e2_large_lp", e2_large_lp, 5400), "e3": ("e3_batch", e3_batch, 7200),
               "e4": ("e4_refinery", e4_refinery, 9000), "e5": ("e5_recursion", e5_recursion, 3600),
               "e6": ("e6_learned", e6_learned, 2400)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--only", default="e0,e1,e2,e3,e4,e5,e6")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for tag in a.only.split(","):
        name, fn, budget = EXPERIMENTS[tag]
        rec = Recorder(run_dir, name)
        t0 = time.perf_counter()
        print(f"\n===== {tag} {name} (budget {budget}s, {len(rec.rows)} rows already recorded) =====", flush=True)
        try:
            fn(rec, a.quick)
            status = "ok"
        except Exception as e:  # noqa: BLE001
            status = f"failed: {type(e).__name__}: {e}"
            traceback.print_exc()
        Recorder(run_dir, "_progress").add(f"{tag}/{int(time.time())}", {"experiment": tag, "status": status,
                                                                          "seconds": time.perf_counter() - t0})
    e7_headline(run_dir)


if __name__ == "__main__":
    main()
