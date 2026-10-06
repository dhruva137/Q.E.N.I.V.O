"""Fair stack benchmark: HiGHS cold + HiGHS warm vs QENIVO cold / warm / GPU batch.

    python bench/stack_bench.py --level 3 --kinds all --S 256 --workers 2 --time-limit 120 \\
        --out bench/results/stack_bench_2026-10-01.jsonl --no-gpu
    python bench/stack_bench.py --quick --workers 2 --no-gpu \\
        --out bench/results/stack_bench_2026-10-01.jsonl

Columns (identical case set and NN order for every method):
  highs_cold          highspy cold, threads=1, ProcessPool (workers<=6)
  highs_warm_chain    NN order; same Highs object; change* + getBasis/setBasis; N chains
  qenivo_cold         native simplex per case, pool (KKT inside timed region)
  qenivo_warm_chain   native simplex start=previous statuses, N NN chains
  qenivo_gpu_batch    PDHG batch; skipped with --no-gpu / skipped_gpu_busy

HiGHS warm API:
  passModel once; per case changeColsCost / changeColsBounds / changeRowsBounds;
  getBasis() after solve; setBasis(prev) before next run(); no clearSolver() on warm path;
  threads=1. Workers set OMP/OPENBLAS/MKL_NUM_THREADS=1.
"""
from __future__ import annotations

import argparse
import copy
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

_LOCK_CANDIDATES = [
    ROOT / "bench" / ".locks" / "gpu.lock",
]

ALL_KINDS = ["one_crude", "one_unit", "factor_price", "factor_mixed", "independent", "cargo_menu"]

import numpy as np  # noqa: E402

from qenivo.workload.stacks import CaseDelta, make_stack, nn_order  # noqa: E402

STACKS_SOURCE = "qenivo.workload.stacks"


# ====================================================================== recording
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


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def machine_stamp() -> dict:
    import scipy

    import qenivo
    out = {
        "commit": git_commit(),
        "machine": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "qenivo": qenivo.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "date": dt.date.today().isoformat(),
        "cpu_count": os.cpu_count(),
        "stacks_source": STACKS_SOURCE,
    }
    try:
        import highspy
        out["highspy"] = getattr(highspy, "__version__", "present")
    except Exception:  # noqa: BLE001
        out["highspy"] = "missing"
    try:
        import psutil
        vm = psutil.virtual_memory()
        out["ram_total_gb"] = round(vm.total / 1e9, 2)
        out["ram_available_gb"] = round(vm.available / 1e9, 2)
        out["cpu_percent"] = psutil.cpu_percent(interval=0.2)
    except Exception:  # noqa: BLE001
        pass
    return out


def gpu_lock_free() -> bool:
    return not any(p.exists() for p in _LOCK_CANDIDATES)


def rel_obj_diff(a, b):
    if a is None or b is None:
        return None
    return float(abs(a - b) / (1.0 + abs(b)))


# ====================================================================== materialise
def materialise_vectors(prob, delta: CaseDelta):
    """Return full (c, lc, uc, lx, ux) for one delta; shares semantics with CaseDelta.apply."""
    c, lc, uc, lx, ux = prob.c.copy(), prob.lc.copy(), prob.uc.copy(), prob.lx.copy(), prob.ux.copy()
    if delta.c_idx.size:
        c[delta.c_idx] = delta.c_val
    if delta.lx_idx.size:
        lx[delta.lx_idx] = delta.lx_val
    if delta.ux_idx.size:
        ux[delta.ux_idx] = delta.ux_val
    if delta.lc_idx.size:
        lc[delta.lc_idx] = delta.lc_val
    if delta.uc_idx.size:
        uc[delta.uc_idx] = delta.uc_val
    return c, lc, uc, lx, ux


def pack_case_vectors(prob, deltas):
    S = len(deltas)
    C = np.repeat(prob.c[:, None], S, 1)
    LC = np.repeat(prob.lc[:, None], S, 1)
    UC = np.repeat(prob.uc[:, None], S, 1)
    LX = np.repeat(prob.lx[:, None], S, 1)
    UX = np.repeat(prob.ux[:, None], S, 1)
    for s, d in enumerate(deltas):
        c, lc, uc, lx, ux = materialise_vectors(prob, d)
        C[:, s], LC[:, s], UC[:, s], LX[:, s], UX[:, s] = c, lc, uc, lx, ux
    return C, LC, UC, LX, UX


def _prob_payload(prob):
    A = prob.A.tocsr()
    return {
        "c": np.asarray(prob.c), "lc": np.asarray(prob.lc), "uc": np.asarray(prob.uc),
        "lx": np.asarray(prob.lx), "ux": np.asarray(prob.ux), "c0": float(prob.c0),
        "obj_sign": float(prob.obj_sign), "name": prob.name,
        "A_data": A.data.copy(), "A_indices": A.indices.copy(), "A_indptr": A.indptr.copy(),
        "A_shape": A.shape,
    }


def _rebuild_prob(payload):
    import scipy.sparse as sp

    from qenivo.model import Problem
    A = sp.csr_matrix((payload["A_data"], payload["A_indices"], payload["A_indptr"]),
                      shape=payload["A_shape"])
    return Problem(c=payload["c"], A=A, lc=payload["lc"], uc=payload["uc"],
                   lx=payload["lx"], ux=payload["ux"], c0=payload["c0"],
                   obj_sign=payload["obj_sign"], name=payload.get("name", ""))


# ====================================================================== workers
_W = {}


def _worker_init(prob_payload, time_limit):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, str(ROOT / "src"))
    _W["prob"] = prob_payload
    _W["tl"] = time_limit


def _highs_cold_one(args):
    idx, c, lc, uc, lx, ux = args
    import highspy
    base = _rebuild_prob(_W["prob"])
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", float(_W["tl"]))
    A = base.A.tocsc()
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = base.n, base.m
    lp.col_cost_ = np.ascontiguousarray(c)
    lp.col_lower_, lp.col_upper_ = np.ascontiguousarray(lx), np.ascontiguousarray(ux)
    lp.row_lower_, lp.row_upper_ = np.ascontiguousarray(lc), np.ascontiguousarray(uc)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    h.passModel(lp)
    t0 = time.perf_counter()
    h.run()
    wall = time.perf_counter() - t0
    info = h.getInfo()
    st = h.modelStatusToString(h.getModelStatus())
    return {
        "index": idx, "status": st,
        "objective": float(info.objective_function_value) if "Optimal" in st else None,
        "iterations": int(info.simplex_iteration_count), "time": wall,
        "zero_pivot": int(info.simplex_iteration_count) == 0,
        "verdict": "optimal" if "Optimal" in st else st, "max_rel": None,
    }


def _qenivo_cold_one(args):
    idx, c, lc, uc, lx, ux = args
    from qenivo.certify.kkt import kkt_residuals
    from qenivo.engines.native_simplex import solve_simplex_native
    base = _rebuild_prob(_W["prob"])
    p = copy.copy(base)
    p.c, p.lc, p.uc, p.lx, p.ux = map(np.ascontiguousarray, (c, lc, uc, lx, ux))
    t0 = time.perf_counter()
    r = solve_simplex_native(p, tol=1e-9, time_limit=float(_W["tl"]))
    verdict, max_rel, obj = "not_proven", None, None
    if r["status"] == "optimal":
        k = kkt_residuals(p, r["x"], r["y"])
        max_rel = k["max_rel"]
        obj = k["objective"]
        verdict = "optimal" if max_rel <= 1e-7 else "not_proven"
    wall = time.perf_counter() - t0
    return {
        "index": idx, "status": r["status"], "objective": obj,
        "iterations": int(r["iterations"]), "time": wall,
        "zero_pivot": int(r["iterations"]) == 0, "verdict": verdict, "max_rel": max_rel,
    }


def _highs_warm_chain_chunk(args):
    order, vectors = args
    C, LC, UC, LX, UX = vectors
    import highspy
    base = _rebuild_prob(_W["prob"])
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", float(_W["tl"]))
    A = base.A.tocsc()
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = base.n, base.m
    s0 = int(order[0])
    lp.col_cost_ = np.ascontiguousarray(C[:, s0])
    lp.col_lower_, lp.col_upper_ = np.ascontiguousarray(LX[:, s0]), np.ascontiguousarray(UX[:, s0])
    lp.row_lower_, lp.row_upper_ = np.ascontiguousarray(LC[:, s0]), np.ascontiguousarray(UC[:, s0])
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    h.passModel(lp)
    ci = np.arange(base.n, dtype=np.int32)
    ri = np.arange(base.m, dtype=np.int32)
    prev_basis = None
    out, t_sum = [], 0.0
    for k, s in enumerate(order):
        s = int(s)
        if k:
            h.changeColsCost(base.n, ci, np.ascontiguousarray(C[:, s]))
            h.changeColsBounds(base.n, ci, np.ascontiguousarray(LX[:, s]), np.ascontiguousarray(UX[:, s]))
            h.changeRowsBounds(base.m, ri, np.ascontiguousarray(LC[:, s]), np.ascontiguousarray(UC[:, s]))
            if prev_basis is not None:
                h.setBasis(prev_basis)
        t0 = time.perf_counter()
        h.run()
        wall = time.perf_counter() - t0
        t_sum += wall
        info = h.getInfo()
        st = h.modelStatusToString(h.getModelStatus())
        try:
            prev_basis = h.getBasis()
        except Exception:  # noqa: BLE001
            prev_basis = None
        out.append({
            "index": s, "status": st,
            "objective": float(info.objective_function_value) if "Optimal" in st else None,
            "iterations": int(info.simplex_iteration_count), "time": wall,
            "zero_pivot": int(info.simplex_iteration_count) == 0,
            "verdict": "optimal" if "Optimal" in st else st, "max_rel": None,
            "highs_api": "changeColsCost+changeColsBounds+changeRowsBounds+getBasis/setBasis",
        })
    return out, t_sum


def _qenivo_warm_chain_chunk(args):
    order, vectors = args
    C, LC, UC, LX, UX = vectors
    from qenivo.certify.kkt import kkt_residuals
    from qenivo.engines.native_simplex import solve_simplex_native
    base = _rebuild_prob(_W["prob"])
    start = None
    out, t_sum = [], 0.0
    for s in order:
        s = int(s)
        p = copy.copy(base)
        p.c = np.ascontiguousarray(C[:, s])
        p.lc, p.uc = np.ascontiguousarray(LC[:, s]), np.ascontiguousarray(UC[:, s])
        p.lx, p.ux = np.ascontiguousarray(LX[:, s]), np.ascontiguousarray(UX[:, s])
        t0 = time.perf_counter()
        r = solve_simplex_native(p, tol=1e-9, time_limit=float(_W["tl"]), start=start)
        verdict, max_rel, obj = "not_proven", None, None
        if r["status"] == "optimal":
            k = kkt_residuals(p, r["x"], r["y"])
            max_rel = k["max_rel"]
            obj = k["objective"]
            verdict = "optimal" if max_rel <= 1e-7 else "not_proven"
            start = {"col_statuses": r["col_statuses"], "row_statuses": r["row_statuses"]}
        else:
            start = None
        wall = time.perf_counter() - t0
        t_sum += wall
        out.append({
            "index": s, "status": r["status"], "objective": obj,
            "iterations": int(r["iterations"]), "time": wall,
            "zero_pivot": int(r["iterations"]) == 0, "verdict": verdict, "max_rel": max_rel,
        })
    return out, t_sum


def _fill_missing(results):
    for i, r in enumerate(results):
        if r is None:
            results[i] = {
                "index": i, "status": "timeout_budget", "verdict": "timeout_budget",
                "objective": None, "iterations": None, "time": None,
                "zero_pivot": False, "max_rel": None,
            }
    return results


def run_pool_cold(method, prob, deltas, workers, time_limit, budget_deadline):
    C, LC, UC, LX, UX = pack_case_vectors(prob, deltas)
    payload = _prob_payload(prob)
    fn = _highs_cold_one if method == "highs_cold" else _qenivo_cold_one
    tasks = [(s, C[:, s], LC[:, s], UC[:, s], LX[:, s], UX[:, s]) for s in range(len(deltas))]
    results = [None] * len(deltas)
    wall0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                              initializer=_worker_init, initargs=(payload, time_limit)) as ex:
        futs = {ex.submit(fn, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            if time.perf_counter() > budget_deadline:
                break
            try:
                row = fut.result()
                results[row["index"]] = row
            except Exception as e:  # noqa: BLE001
                idx = futs[fut]
                results[idx] = {
                    "index": idx, "status": "error", "error": str(e)[:300], "time": None,
                    "verdict": "error", "objective": None, "iterations": None,
                    "zero_pivot": False, "max_rel": None,
                }
    wall = time.perf_counter() - wall0
    _fill_missing(results)
    return results, wall, sum(r["time"] or 0.0 for r in results)


def _split_chains(order, workers):
    w = max(1, min(workers, len(order)))
    return [list(ch) for ch in np.array_split(np.asarray(order, int), w) if len(ch)]


def run_warm_chains(method, prob, deltas, order, workers, time_limit, budget_deadline):
    vectors = pack_case_vectors(prob, deltas)
    payload = _prob_payload(prob)
    chunks = _split_chains(order, workers)
    fn = _highs_warm_chain_chunk if method == "highs_warm_chain" else _qenivo_warm_chain_chunk
    results = [None] * len(deltas)
    wall0 = time.perf_counter()
    tsum = 0.0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx,
                              initializer=_worker_init, initargs=(payload, time_limit)) as ex:
        futs = [ex.submit(fn, (ch, vectors)) for ch in chunks]
        for fut in as_completed(futs):
            if time.perf_counter() > budget_deadline:
                break
            try:
                rows, partial = fut.result()
                tsum += partial
                for r in rows:
                    results[r["index"]] = r
            except Exception as e:  # noqa: BLE001
                print(f"  warm chain error: {e}")
    wall = time.perf_counter() - wall0
    _fill_missing(results)
    return results, wall, tsum


def run_gpu_batch(prob, deltas, tol, time_limit):
    from qenivo.certify.kkt import is_optimal, kkt_from_products, rhs_norm
    from qenivo.engines import pdhg
    from qenivo.kernels.backend import gpu_available
    if not gpu_available():
        return {"status": "skipped_no_gpu", "cases": []}
    if not gpu_lock_free():
        return {"status": "skipped_gpu_busy", "cases": []}
    C, LC, UC, LX, UX = pack_case_vectors(prob, deltas)
    opts = pdhg.PDHGOptions(tol=tol, time_limit=time_limit)
    t0 = time.perf_counter()
    br = pdhg.solve_batch(prob.A, C, LC, UC, LX, UX, prob.c0, opts, "cupy")
    try:
        import cupy
        cupy.cuda.Device().synchronize()
    except Exception:  # noqa: BLE001
        pass
    k = kkt_from_products(br.x, br.y, prob.A @ br.x, prob.A.T @ br.y, C, prob.c0,
                          LC, UC, LX, UX, rhs_norm(LC, UC), np.linalg.norm(C, axis=0))
    max_rel = np.maximum(np.maximum(k["rel_primal"], k["rel_dual"]), k["rel_gap"])
    ok = is_optimal(k, tol * 1.0001) & np.array([s == "optimal" for s in br.status])
    wall = time.perf_counter() - t0
    cases = [{
        "index": j, "status": br.status[j], "objective": float(k["pobj"][j]) * prob.obj_sign,
        "iterations": int(br.iterations[j]), "time": None, "zero_pivot": False,
        "verdict": "optimal" if ok[j] else "not_proven", "max_rel": float(max_rel[j]),
    } for j in range(len(deltas))]
    return {"status": "ok", "wall": wall, "sum_time": wall, "tol": tol,
            "cases": cases, "certified": int(ok.sum())}


# ====================================================================== markdown
def write_markdown(rec: Recorder, md_path: Path):
    stacks = [r for r in rec.rows if r.get("row_type") == "stack_aggregate"]
    lines = [
        f"# Stack bench summary ({dt.date.today().isoformat()})",
        "",
        "HiGHS warm API: `changeColsCost` / `changeColsBounds` / `changeRowsBounds` on one "
        "`Highs` object, then `getBasis()` / `setBasis(prev)` between cases; no `clearSolver()` "
        "on the warm path; `threads=1`.",
        "",
        "| level | kind | S | highs_cold wall/sum | highs_warm wall/sum | "
        "qenivo_cold wall/sum | qenivo_warm wall/sum | gpu | zp% q-warm | max |obj| rel |",
        "|---|---|---:|---|---|---|---|---|---:|---:|",
    ]

    def fmt(col, r):
        d = r.get(col) or {}
        st = d.get("status", "")
        if st and str(st).startswith("skipped"):
            return st
        w, s = d.get("wall"), d.get("sum_time")
        if w is None:
            return "n/a"
        return f"{w:.2f}/{s:.2f}"

    for r in stacks:
        lines.append(
            f"| L{r.get('level')} | {r.get('kind')} | {r.get('S')} | "
            f"{fmt('highs_cold', r)} | {fmt('highs_warm_chain', r)} | "
            f"{fmt('qenivo_cold', r)} | {fmt('qenivo_warm_chain', r)} | "
            f"{(r.get('qenivo_gpu_batch') or {}).get('status', 'n/a')} | "
            f"{100 * (r.get('zero_pivot_frac_qenivo_warm') or 0):.1f} | "
            f"{r.get('max_rel_obj_vs_highs_cold')} |"
        )
    lines.append("")
    for d in rec.rows:
        if d.get("row_type") == "deferred":
            lines.append(f"- Deferred: {d.get('reason')}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ====================================================================== one stack
def run_one_stack(rec, stamp, cmd, level, kind, S, seed, workers, time_limit,
                  budget_deadline, no_gpu):
    from qenivo.models.refinery import refinery_level
    key = f"L{level}/{kind}/S{S}/seed{seed}"
    if rec.done(key):
        print(f"skip (resume) {key}")
        return

    t_gen0 = time.perf_counter()
    prob = refinery_level(level, seed=0)
    try:
        deltas = make_stack(prob, kind, S, seed)
    except Exception as e:  # noqa: BLE001
        rec.add(key, {
            "row_type": "stack_aggregate", "level": level, "kind": kind, "S": S,
            "status": "error", "error": str(e), "traceback": traceback.format_exc()[-800:],
            **stamp, "command": cmd,
        })
        print(f"FAIL make_stack {key}: {e}")
        return
    order = nn_order(deltas)
    gen_t = time.perf_counter() - t_gen0
    cols = {}

    def maybe_stop(tag):
        if time.perf_counter() > budget_deadline:
            _finish_stack(rec, key, stamp, cmd, level, kind, S, seed, order, cols, gen_t, tag)
            return True
        return False

    print(f"{key}: highs_cold ({prob.m}x{prob.n}) ...")
    hc, hw, hs = run_pool_cold("highs_cold", prob, deltas, workers, time_limit, budget_deadline)
    cols["highs_cold"] = {"wall": hw, "sum_time": hs, "cases": hc, "status": "ok"}
    if maybe_stop("budget"):
        return

    print(f"{key}: highs_warm_chain ...")
    wc, ww, ws = run_warm_chains("highs_warm_chain", prob, deltas, order, workers,
                                 time_limit, budget_deadline)
    cols["highs_warm_chain"] = {
        "wall": ww, "sum_time": ws, "cases": wc, "status": "ok",
        "highs_api": "changeColsCost+changeColsBounds+changeRowsBounds+getBasis/setBasis; "
                     f"{workers} NN-ordered contiguous chains; threads=1",
    }
    if maybe_stop("budget"):
        return

    print(f"{key}: qenivo_cold ...")
    qc, qw, qs = run_pool_cold("qenivo_cold", prob, deltas, workers, time_limit, budget_deadline)
    cols["qenivo_cold"] = {"wall": qw, "sum_time": qs, "cases": qc, "status": "ok"}
    if maybe_stop("budget"):
        return

    print(f"{key}: qenivo_warm_chain ...")
    yc, yw, ys = run_warm_chains("qenivo_warm_chain", prob, deltas, order, workers,
                                 time_limit, budget_deadline)
    cols["qenivo_warm_chain"] = {"wall": yw, "sum_time": ys, "cases": yc, "status": "ok"}

    if no_gpu:
        cols["qenivo_gpu_batch"] = {"status": "skipped_no_gpu_flag", "cases": []}
    elif not gpu_lock_free():
        cols["qenivo_gpu_batch"] = {"status": "skipped_gpu_busy", "cases": []}
    else:
        try:
            cols["qenivo_gpu_batch"] = run_gpu_batch(prob, deltas, 1e-4, time_limit)
        except Exception as e:  # noqa: BLE001
            cols["qenivo_gpu_batch"] = {"status": "error", "error": str(e)[:300], "cases": []}

    _finish_stack(rec, key, stamp, cmd, level, kind, S, seed, order, cols, gen_t, "ok")


def _finish_stack(rec, key, stamp, cmd, level, kind, S, seed, order, cols, gen_t, status):
    highs = (cols.get("highs_cold") or {}).get("cases") or []
    href = {r["index"]: r.get("objective") for r in highs}

    for name in list(cols):
        block = cols[name]
        if "cases" not in block:
            continue
        annotated = []
        for r in block["cases"] or []:
            rr = dict(r)
            rr["obj_rel_vs_highs_cold"] = rel_obj_diff(r.get("objective"), href.get(r["index"]))
            annotated.append(rr)
        cols[name] = dict(block)
        cols[name]["cases"] = annotated

    for name, block in cols.items():
        for r in block.get("cases") or []:
            rec.add(f"{key}/{name}/case{r['index']}", {
                "row_type": "case", "level": level, "kind": kind, "S": S, "seed": seed,
                "column": name, **r, **stamp, "command": cmd,
            })

    qw = (cols.get("qenivo_warm_chain") or {}).get("cases") or []
    zp = [r for r in qw if r.get("zero_pivot")]
    rels = []
    for name in ("qenivo_cold", "qenivo_warm_chain"):
        for r in (cols.get(name) or {}).get("cases") or []:
            if r.get("obj_rel_vs_highs_cold") is not None:
                rels.append(r["obj_rel_vs_highs_cold"])

    agg = {
        "row_type": "stack_aggregate", "level": level, "kind": kind, "S": S, "seed": seed,
        "status": status, "nn_order": order, "gen_time": gen_t,
        "zero_pivot_frac_qenivo_warm": (len(zp) / len(qw)) if qw else None,
        "max_rel_obj_vs_highs_cold": (max(rels) if rels else None),
        "stacks_source": STACKS_SOURCE,
        **stamp, "command": cmd,
    }
    for name in ("highs_cold", "highs_warm_chain", "qenivo_cold", "qenivo_warm_chain", "qenivo_gpu_batch"):
        block = cols.get(name) or {}
        slim = {k: v for k, v in block.items() if k != "cases"}
        slim["n_cases"] = len(block.get("cases") or [])
        slim["n_optimal"] = sum(1 for r in (block.get("cases") or []) if r.get("verdict") == "optimal")
        agg[name] = slim
    rec.add(key, agg)
    hc, hw = cols.get("highs_cold") or {}, cols.get("highs_warm_chain") or {}
    qc, qw = cols.get("qenivo_cold") or {}, cols.get("qenivo_warm_chain") or {}
    print(
        f"done {key}: H_cold {hc.get('wall', 0):.1f}s H_warm {hw.get('wall', 0):.1f}s "
        f"Q_cold {qc.get('wall', 0):.1f}s Q_warm {qw.get('wall', 0):.1f}s "
        f"zp={agg['zero_pivot_frac_qenivo_warm']} gpu={agg['qenivo_gpu_batch'].get('status')}"
    )


def parse_kinds(s: str) -> list[str]:
    if s.strip().lower() == "all":
        return list(ALL_KINDS)
    return [k.strip() for k in s.split(",") if k.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--kinds", type=str, default="all")
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--budget-min", type=float, default=55.0, help="global wall budget minutes")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-gpu", action="store_true", default=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--after-quick-continue", action="store_true",
                    help="after --quick, continue into L3 S=64/S=256 (owner campaign mode)")
    args = ap.parse_args(argv)

    if args.gpu:
        args.no_gpu = False
    args.workers = max(1, min(6, args.workers))
    # Parent / workers: keep BLAS single-threaded; pool size controls parallelism.
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"

    today = dt.date.today().isoformat()
    out = Path(args.out) if args.out else HERE / "results" / f"stack_bench_{today}.jsonl"
    rec = Recorder(out)
    stamp = machine_stamp()
    cmd = " ".join([sys.executable] + (["bench/stack_bench.py"] + (argv or sys.argv[1:])))
    stamp["command"] = cmd
    if not rec.done("env"):
        rec.add("env", {"row_type": "env", **stamp})

    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"resources: ram_avail={vm.available/1e9:.2f}GB/{vm.total/1e9:.2f}GB "
              f"cpu={psutil.cpu_percent(interval=0.3):.0f}% workers={args.workers}")
    except Exception:  # noqa: BLE001
        print(f"workers={args.workers}")

    budget_deadline = time.perf_counter() + args.budget_min * 60.0

    # Smoke L2 S=4 (one_crude + independent) first; resume-safe via Recorder keys.
    smoke_kinds = ["one_crude", "independent"]
    for kind in smoke_kinds:
        if time.perf_counter() > budget_deadline:
            break
        run_one_stack(rec, stamp, cmd, 2, kind, 4, args.seed, args.workers,
                      min(args.time_limit, 60.0), budget_deadline, no_gpu=True)
    write_markdown(rec, out.with_suffix(".md"))
    if args.quick and not args.after_quick_continue:
        print("wrote", out, "and", out.with_suffix(".md"), "(quick only)")
        return

    kinds = parse_kinds(args.kinds)
    level = args.level
    # S=64 then S=256 when targeting >=256; otherwise just the requested S.
    sizes = [64, 256] if args.S >= 256 else [args.S]
    sizes = [s for s in sizes if s <= args.S]
    for S in sizes:
        for kind in kinds:
            if time.perf_counter() > budget_deadline:
                rec.add(f"deferred/L{level}/{kind}/S{S}", {
                    "row_type": "deferred",
                    "reason": f"global budget exhausted before L{level} {kind} S={S}",
                    **stamp, "command": cmd,
                })
                write_markdown(rec, out.with_suffix(".md"))
                print("budget exhausted; wrote", out)
                return
            run_one_stack(rec, stamp, cmd, level, kind, S, args.seed, args.workers,
                          args.time_limit, budget_deadline, args.no_gpu)
        write_markdown(rec, out.with_suffix(".md"))

    write_markdown(rec, out.with_suffix(".md"))
    print("wrote", out, "and", out.with_suffix(".md"))


if __name__ == "__main__":
    main()
