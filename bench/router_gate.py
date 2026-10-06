"""W03 gate measurement: routed stack time vs best single path and HiGHS warm.

Uses W01 generators + warm-chain helpers from stack_bench.py (not owned by W03).
Writes bench/results/router_<date>.jsonl. RAM-safe defaults: workers=1, one kind at a time.

  python bench/router_gate.py --level 3 --S 64 --workers 1 --kinds all
  python bench/router_gate.py --smoke   # L2 S=4 all kinds, no GPU
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import stack_bench as SB  # noqa: E402

from qenivo.models.refinery import refinery_level  # noqa: E402
from qenivo.workload.router import classify_stack, route_stack  # noqa: E402
from qenivo.workload.stacks import make_stack, nn_order  # noqa: E402

LOCK = Path(__file__).resolve().parents[1] / "bench" / ".locks" / "gpu.lock"
ALL_KINDS = ["one_crude", "one_unit", "factor_price", "factor_mixed", "independent", "cargo_menu"]


def _status(msg: str) -> None:
    print(f"STATUS: {msg}", flush=True)


def _acquire_gpu(owner: str) -> bool:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        age = time.time() - LOCK.stat().st_mtime
        if age < 45 * 60:
            _status(f"gpu.lock held age={age:.0f}s — skip GPU")
            return False
        _status(f"gpu.lock stale age={age:.0f}s — taking over")
    LOCK.write_text(f"{owner}\n{dt.datetime.now(dt.timezone.utc).isoformat()}\n", encoding="utf-8")
    return True


def _release_gpu() -> None:
    try:
        if LOCK.exists():
            LOCK.unlink()
            _status("gpu.lock released")
    except OSError as e:
        _status(f"gpu.lock release failed: {e}")


def _shock_order(deltas) -> list[int]:
    shocks = []
    for d in deltas:
        if getattr(d, "shock", None) is not None and len(d.shock):
            shocks.append(np.asarray(d.shock, dtype=np.float64))
        else:
            shocks.append(np.asarray([0.0], dtype=np.float64))
    return nn_order(shocks)


def _factor_order(deltas) -> list[int]:
    # Order along first shock coordinate when present (1-D ray).
    keys = []
    for i, d in enumerate(deltas):
        s = getattr(d, "shock", None)
        if s is not None and len(s):
            keys.append((float(s[0]), i))
        else:
            keys.append((float(i), i))
    keys.sort()
    return [i for _, i in keys]


def run_qenivo_warm(prob, deltas, order, workers, time_limit):
    budget = time.perf_counter() + max(30.0, time_limit * len(deltas) * 2)
    return SB.run_warm_chains("qenivo_warm_chain", prob, deltas, order, workers, time_limit, budget)


def run_highs_warm(prob, deltas, order, workers, time_limit):
    budget = time.perf_counter() + max(30.0, time_limit * len(deltas) * 2)
    return SB.run_warm_chains("highs_warm_chain", prob, deltas, order, workers, time_limit, budget)


def run_gpu_batch(prob, deltas, time_limit, tol=1e-4):
    from qenivo.engines import pdhg
    C, LC, UC, LX, UX = SB.pack_case_vectors(prob, deltas)
    opts = pdhg.PDHGOptions(tol=tol, time_limit=time_limit, verbose=False)
    t0 = time.perf_counter()
    br = pdhg.solve_batch(prob.A, C, LC, UC, LX, UX, prob.c0, opts, "cupy")
    try:
        import cupy as cp
        cp.cuda.Device().synchronize()
    except Exception:  # noqa: BLE001
        pass
    wall = time.perf_counter() - t0
    n_ok = sum(1 for s in br.status if s == "optimal")
    return {
        "wall": wall,
        "sum_time": float(br.setup_time + br.solve_time),
        "n_ok": n_ok,
        "S": len(deltas),
        "tol": tol,
        "status_sample": list(br.status[: min(5, len(br.status))]),
    }


def measure_kind(rec: SB.Recorder, stamp: dict, level: int, kind: str, S: int,
                 seed: int, workers: int, time_limit: float, use_gpu: bool) -> dict:
    key = f"L{level}/{kind}/S{S}/seed{seed}"
    _status(f"build {key}")
    prob = refinery_level(level, seed=seed)
    deltas = make_stack(prob, kind, S, seed=seed)
    profile = classify_stack(prob, deltas)
    decision = route_stack(profile, base=prob, deltas=deltas, engine="auto",
                           backend="auto" if use_gpu else "numpy", threads=workers)
    route_name = decision["route"]
    _status(f"{key} route={route_name} rank={profile.delta_rank} "
            f"coords={profile.n_changed_coords} max_rel={profile.max_rel_move:.3f}")

    nn = _shock_order(deltas)
    factor = _factor_order(deltas) if route_name == "cpu_warm_ray" else nn
    order_for_route = factor if route_name == "cpu_warm_ray" else nn

    # --- single paths ---
    paths: dict[str, float] = {}
    detail: dict = {"profile": profile.to_dict(), "decision": {
        "route": decision["route"], "reason": decision["reason"],
        "engine": decision["engine"], "backend": decision["backend"],
        "order": decision.get("order"), "gpu_available": decision.get("gpu_available"),
    }}

    _status(f"{key} qenivo_warm workers={workers}")
    qw_rows, qw_wall, qw_sum = run_qenivo_warm(prob, deltas, nn, workers, time_limit)
    paths["qenivo_warm"] = qw_wall
    detail["qenivo_warm"] = {
        "wall": qw_wall, "sum": qw_sum,
        "n_ok": sum(r.get("verdict") == "optimal" for r in qw_rows),
        "zp_frac": sum(bool(r.get("zero_pivot")) for r in qw_rows) / max(1, S),
    }
    _status(f"{key} qenivo_warm wall={qw_wall:.2f}s")

    _status(f"{key} highs_warm workers={workers}")
    hw_rows, hw_wall, hw_sum = run_highs_warm(prob, deltas, nn, workers, time_limit)
    paths["highs_warm"] = hw_wall
    detail["highs_warm"] = {
        "wall": hw_wall, "sum": hw_sum,
        "n_ok": sum(r.get("verdict") == "optimal" for r in hw_rows),
        "highs_api": "changeColsCost+Bounds+changeRowsBounds+getBasis/setBasis threads=1",
    }
    _status(f"{key} highs_warm wall={hw_wall:.2f}s")

    gpu_ok = False
    if use_gpu and (route_name in ("gpu_batch", "mixed") or kind == "independent"):
        if _acquire_gpu("W03-router-gate"):
            try:
                _status(f"{key} gpu_batch")
                g = run_gpu_batch(prob, deltas, time_limit)
                paths["qenivo_gpu_batch"] = g["wall"]
                detail["qenivo_gpu_batch"] = g
                gpu_ok = True
                _status(f"{key} gpu_batch wall={g['wall']:.2f}s n_ok={g['n_ok']}")
            except Exception as e:  # noqa: BLE001
                detail["qenivo_gpu_batch"] = {"error": str(e)[:400]}
                _status(f"{key} gpu_batch FAILED: {e}")
            finally:
                _release_gpu()
        else:
            detail["qenivo_gpu_batch"] = {"skipped": "gpu_busy"}

    # --- routed ---
    routed_wall = None
    if route_name in ("cpu_warm", "cpu_warm_ray", "mixed") and decision["backend"] == "numpy":
        # mixed without GPU falls back to CPU warm; use NN or cluster order (NN for fair wall).
        ord_ = order_for_route
        _status(f"{key} routed={route_name} via qenivo_warm order={decision.get('order')}")
        rr, routed_wall, rsum = run_qenivo_warm(prob, deltas, ord_, workers, time_limit)
        detail["routed"] = {
            "how": route_name, "wall": routed_wall, "sum": rsum,
            "n_ok": sum(r.get("verdict") == "optimal" for r in rr),
            "order": decision.get("order"),
        }
        # Avoid double-counting if same as qenivo_warm with same order
        if route_name == "cpu_warm" and ord_ == nn:
            routed_wall = qw_wall
            detail["routed"]["wall"] = qw_wall
            detail["routed"]["reused_qenivo_warm"] = True
    elif route_name == "gpu_batch" and gpu_ok:
        routed_wall = paths["qenivo_gpu_batch"]
        detail["routed"] = {"how": "gpu_batch", "wall": routed_wall,
                            "reused_gpu_batch": True}
    elif route_name == "mixed" and decision["backend"] == "cupy" and gpu_ok:
        routed_wall = paths["qenivo_gpu_batch"]
        detail["routed"] = {"how": "mixed_gpu_batch", "wall": routed_wall,
                            "reused_gpu_batch": True, "clusters": decision.get("clusters")}
    else:
        # Fallback: CPU warm
        _status(f"{key} routed fallback cpu_warm")
        rr, routed_wall, rsum = run_qenivo_warm(prob, deltas, nn, workers, time_limit)
        detail["routed"] = {"how": "fallback_cpu_warm", "wall": routed_wall, "sum": rsum,
                            "intended_route": route_name}

    # Best single *product* path (exclude HiGHS)
    product_paths = {k: v for k, v in paths.items() if k != "highs_warm" and v is not None}
    best_name = min(product_paths, key=product_paths.get) if product_paths else None
    best_wall = product_paths[best_name] if best_name else None

    vs_best = (routed_wall / best_wall) if (routed_wall and best_wall and best_wall > 0) else None
    vs_highs = (routed_wall / hw_wall) if (routed_wall and hw_wall and hw_wall > 0) else None
    gate_vs_best = vs_best is not None and vs_best <= 1.15
    gate_vs_highs = vs_highs is not None and vs_highs <= 1.20
    gate_ok = bool(gate_vs_best and gate_vs_highs)

    row = {
        "row_type": "stack_gate",
        "level": level, "kind": kind, "S": S, "seed": seed,
        "route": route_name, "routed_wall": routed_wall,
        "best_single_path": best_name, "best_single_wall": best_wall,
        "vs_best": vs_best, "vs_highs_warm": vs_highs,
        "highs_warm_wall": hw_wall,
        "paths": paths,
        "gate_within_15pct_best": gate_vs_best,
        "gate_le_1_2x_highs_warm": gate_vs_highs,
        "gate_ok": gate_ok,
        "workers": workers, "time_limit": time_limit,
        "detail": detail,
        **stamp,
    }
    rec.add(key, row)
    _status(f"{key} GATE {'OK' if gate_ok else 'FAIL'} "
            f"routed={routed_wall:.2f} best={best_name}:{best_wall:.2f} "
            f"vs_best={vs_best:.3f} vs_highs={vs_highs:.3f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--S", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--kinds", default="all")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.level, args.S, args.no_gpu = 2, 4, True
        _status("smoke mode L2 S=4 --no-gpu")

    kinds = ALL_KINDS if args.kinds == "all" else [k.strip() for k in args.kinds.split(",")]
    date = dt.date.today().isoformat()
    out = Path(args.out or (ROOT / "bench" / "results" / f"router_{date}.jsonl"))
    rec = SB.Recorder(out)
    stamp = SB.machine_stamp()
    stamp["command"] = " ".join(sys.argv)
    stamp["gate"] = "routed <= 1.15*best_product_path and routed <= 1.20*highs_warm"
    if "env" not in rec.keys:
        rec.add("env", {"row_type": "env", **stamp,
                        "kinds": kinds, "level": args.level, "S": args.S,
                        "workers": args.workers, "no_gpu": args.no_gpu})

    _status(f"out={out} kinds={kinds} L{args.level} S={args.S} workers={args.workers}")
    results = []
    for kind in kinds:
        kkey = f"L{args.level}/{kind}/S{args.S}/seed{args.seed}"
        if rec.done(kkey):
            _status(f"skip done {kkey}")
            continue
        try:
            results.append(measure_kind(
                rec, stamp, args.level, kind, args.S, args.seed,
                max(1, min(2, args.workers)), args.time_limit, use_gpu=not args.no_gpu,
            ))
        except Exception as e:  # noqa: BLE001
            import traceback
            _status(f"ERROR {kkey}: {e}")
            rec.add(kkey + "/error", {
                "row_type": "error", "level": args.level, "kind": kind,
                "S": args.S, "error": str(e)[:500],
                "traceback": traceback.format_exc()[-800:], **stamp,
            })

    # Scoreboard for this (level, S) only — do not mix smoke / other levels.
    rows = [r for r in rec.rows if r.get("row_type") == "stack_gate"
            and r.get("level") == args.level and r.get("S") == args.S]
    n_ok = sum(1 for r in rows if r.get("gate_ok"))
    summary = {
        "row_type": "scoreboard",
        "level": args.level,
        "S": args.S,
        "n_kinds": len(rows),
        "n_gate_ok": n_ok,
        "gate_met_this_level": bool(rows) and n_ok == len(rows),
        "note": ("partial: L4 not in this invocation" if args.level < 4
                 else "L4 included in this invocation"),
        "per_kind": [
            {"kind": r["kind"], "level": r["level"], "S": r["S"], "route": r["route"],
             "routed_wall": r["routed_wall"], "best": r["best_single_path"],
             "best_wall": r["best_single_wall"], "vs_best": r["vs_best"],
             "vs_highs": r["vs_highs_warm"], "gate_ok": r["gate_ok"]}
            for r in rows
        ],
        **stamp,
    }
    sk = f"scoreboard/L{args.level}/S{args.S}"
    if not rec.done(sk):
        rec.add(sk, summary)
    _status(f"SCOREBOARD L{args.level}/S{args.S} gate_met={summary['gate_met_this_level']} "
            f"{n_ok}/{len(rows)} kinds OK")
    for p in summary["per_kind"]:
        _status(f"  {p['kind']}: route={p['route']} routed={p['routed_wall']:.2f} "
                f"vs_best={p['vs_best']:.3f} vs_highs={p['vs_highs']:.3f} "
                f"{'OK' if p['gate_ok'] else 'FAIL'}")


if __name__ == "__main__":
    main()
