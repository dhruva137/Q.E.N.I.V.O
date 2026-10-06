"""Round-3 A100 campaign skeleton (W11): 5-8 GPU hours; predict-vs-measure.

    python bench/round3_campaign.py --run-dir /content/qenivo_results/round3_run1
    python bench/round3_campaign.py --run-dir runs/r3_smoke --quick
    python bench/round3_campaign.py --run-dir <same> --only stamp,stack

Parts (append-only JSONL, resumable, no invented numbers)
  stamp      GPU/driver/commit + pre-flight probes; print W10 predictions (PENDING if missing)
  crude      Crude Valuation Engine at scale (L4/L5 full; L2 tiny under --quick)
  stack      Stack matrix: L4/L5 x W01 kinds x S; L2 tiny S under --quick
  diet       GPU byte-diet ablation (skeleton; W04 module may be PENDING)
  simplex    Stack Simplex vertices/s (skeleton; only if W05 gate passed)
  large_lp   Persistent + preconditioning on large LPs (afiro under --quick)
  qp_milp    QP (W08) + MILP (W07) on VM CPU (skeleton / tiny under --quick)
  headline   HEADLINE_ROUND3.md from recorded rows + prediction vs measured

--quick: every part on L2 with tiny S on CPU; no GPU required.
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
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import qenivo  # noqa: E402
from qenivo.certify.kkt import kkt_residuals  # noqa: E402
from qenivo.engines import crossover, native_simplex, pdhg  # noqa: E402
from qenivo.kernels.backend import gpu_available, gpu_info, gpu_status  # noqa: E402
from qenivo.models.refinery import LEVELS, refinery_lp  # noqa: E402

try:
    from gpu_campaign import certify_batch, highs_pool_time  # noqa: E402
except Exception:  # noqa: BLE001
    certify_batch = None
    highs_pool_time = None

GPU = "cupy" if gpu_available() else "numpy"
E2_CUPDLPX = HERE / "results" / "a100_evidence_run1" / "e2_large_lp.jsonl"
PREDICTIONS_CANDIDATES = (
    ROOT / "research_prototypes" / "transfer" / "PREDICTIONS.md",
)
TRANSFER_MODEL_CANDIDATES = (
    ROOT / "research_prototypes" / "transfer" / "transfer_model_2026-10-01.json",
    ROOT / "bench" / "results" / "transfer_model_2026-10-01.json",
)

# W10 gate: only these four spend A100 / CPU campaign time under the laptop-beats-comparator rule.
ELIGIBLE_IDS = ("C-BATCH-L4", "C-PERSIST-S1", "C-STACK-CARGO", "C-STACK-FPRICE")
# Map W11 notebook parts -> eligible candidate IDs (empty => PENDING / blocked gate).
PART_ELIGIBLE = {
    "stamp": ELIGIBLE_IDS,  # pre-flight always runs; print every eligible prediction
    "crude": (),            # P-W02-CRUDE pending
    "stack": ("C-BATCH-L4", "C-STACK-CARGO", "C-STACK-FPRICE"),
    "diet": (),             # P-W04-DIET pending
    "simplex": (),          # P-W05-SSX pending
    "large_lp": ("C-PERSIST-S1",),
    "qp_milp": (),          # B-W07 / W08 pending
    "headline": ELIGIBLE_IDS,
}
PART_PENDING_REASON = {
    "crude": "PENDING — P-W02-CRUDE (L3/L4 HiGHS midpoint gate open)",
    "diet": "PENDING — P-W04-DIET (byte-diet hold; no laptop ≥2× gate)",
    "simplex": "PENDING — P-W05-SSX (Stack Simplex kill test not run)",
    "qp_milp": "PENDING — B-W07-MILP / W08 (laptop does not beat HiGHS)",
}

# Stack kinds: omit blocked warm losers (one_crude, one_unit, factor_mixed).
STACK_KINDS_FULL = ("cargo_menu", "factor_price", "independent")  # cargo+fprice CPU; independent→GPU C-BATCH
STACK_KINDS_QUICK = ("cargo_menu", "factor_price")
STACK_S_FULL = (256, 1024, 4096)
STACK_S_QUICK = (2, 4)
LARGE_LP_FULL = ("qap15",)  # C-PERSIST-S1 narrow; do not chase cuPDLPx losses
LARGE_LP_QUICK = ("afiro",)

BUDGETS_S = {
    "stamp": 5 * 60,
    "crude": 30 * 60,
    "stack": 90 * 60,
    "diet": 45 * 60,
    "simplex": 60 * 60,
    "large_lp": 45 * 60,
    "qp_milp": 60 * 60,
    "headline": 2 * 60,
}


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


def _predictions_path() -> Path | None:
    for p in PREDICTIONS_CANDIDATES:
        if p.exists():
            return p
    return None


def _transfer_model_path() -> Path | None:
    for p in TRANSFER_MODEL_CANDIDATES:
        if p.exists():
            return p
    return None


def _load_transfer_model() -> dict:
    path = _transfer_model_path()
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"transfer_model load failed: {type(e).__name__}: {e}")
        return {}


def _parse_md_candidate_rows(text: str) -> dict[str, str]:
    """Pull pipe-table rows that mention **C-…** IDs from PREDICTIONS.md."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "C-" not in line or "|" not in line:
            continue
        for cid in ELIGIBLE_IDS:
            token = f"**{cid}**"
            if token in line or f"| {cid} |" in line or f"| **{cid}** |" in line:
                # Collapse markdown bold for a compact one-liner.
                cleaned = line.strip().strip("|").replace("**", "")
                out[cid] = " | ".join(c.strip() for c in cleaned.split("|") if c.strip())
    return out


def _format_json_candidate(entry: dict) -> str:
    cid = entry.get("id", "?")
    bits = [
        f"id={cid}",
        f"rank={entry.get('rank')}",
        f"regime={entry.get('regime')}",
        f"confidence={entry.get('confidence')}",
        f"comparator={entry.get('comparator')}",
        f"gpu_hours={entry.get('gpu_hours')}",
    ]
    for key in (
        "predicted_a100_s",
        "predicted_a100_us_iter",
        "predicted_a100_ratio_vs_highs_warm",
        "predicted_ratio_vs_f64",
        "predicted_ratio_vs_untiled",
        "factor_used",
        "note",
    ):
        if key in entry:
            bits.append(f"{key}={json.dumps(entry[key], default=str)}")
    return "; ".join(bits)


def load_predictions() -> dict:
    """Return per-part prediction text from W10 PREDICTIONS.md + transfer_model JSON.

    Eligible IDs only: C-BATCH-L4, C-PERSIST-S1, C-STACK-CARGO, C-STACK-FPRICE.
    Parts whose laptop gate is still open stay PENDING with the W10 reason.
    """
    parts = list(BUDGETS_S)
    md_path = _predictions_path()
    model_path = _transfer_model_path()
    model = _load_transfer_model()
    md_text = md_path.read_text(encoding="utf-8") if md_path else ""
    md_rows = _parse_md_candidate_rows(md_text) if md_text else {}

    by_id = {}
    for entry in model.get("w11_eligible") or []:
        cid = entry.get("id")
        if cid in ELIGIBLE_IDS:
            by_id[cid] = _format_json_candidate(entry)
    for cid, row in md_rows.items():
        if cid not in by_id:
            by_id[cid] = row
        else:
            by_id[cid] = by_id[cid] + "\nMD: " + row

    if md_path:
        print(f"W10 PREDICTIONS.md: loaded {md_path}")
    else:
        print("W10 PREDICTIONS.md: PENDING (file not found)")
    if model_path:
        print(f"W10 transfer_model: loaded {model_path} "
              f"(eligible={[e.get('id') for e in (model.get('w11_eligible') or []) if e.get('id') in ELIGIBLE_IDS]})")
    else:
        print("W10 transfer_model: PENDING (file not found)")

    # Fallback blurbs if JSON/MD missing but we still know the ID is eligible.
    FALLBACK = {
        "C-BATCH-L4": "Batch PDHG L4/L5×{512,1024,4096} iid/independent @1e-4; "
                      "pred L4×512≈13.66s (measured A100), L4×1024≈22–30s, L4×4096≈70–110s; BW factor 3.5×",
        "C-PERSIST-S1": "Persistent vs launch S∈{1,4,16}; pred qap15≈10–20 µs/iter, L4≈15–35, L5≈20–45",
        "C-STACK-CARGO": "CPU warm cargo_menu/one-factor; pred wall ≈ laptop×0.9–1.2; beats HiGHS warm",
        "C-STACK-FPRICE": "CPU warm factor_price; marginal on L3; may flip on L4; vs HiGHS warm",
    }

    out = {}
    detail = {}  # part -> list of {id, text}
    for part in parts:
        ids = PART_ELIGIBLE.get(part, ())
        if not ids:
            msg = PART_PENDING_REASON.get(part, "PENDING")
            out[part] = msg
            detail[part] = []
            continue
        blocks = []
        detail[part] = []
        for cid in ids:
            text = by_id.get(cid) or FALLBACK.get(cid) or "PENDING"
            if md_path is None and model_path is None and cid not in FALLBACK:
                text = "PENDING"
            blocks.append(f"[{cid}] {text}")
            detail[part].append({"id": cid, "text": text})
        out[part] = "\n".join(blocks) if blocks else "PENDING"

    out["_meta"] = {
        "predictions_path": str(md_path) if md_path else None,
        "transfer_model_path": str(model_path) if model_path else None,
        "eligible_ids": list(ELIGIBLE_IDS),
        "detail": detail,
        "calibration": (model.get("calibration") or {}).get("calibrated_bandwidth_wall_factor"),
        "pending_gates": model.get("pending_laptop_gate") or [],
    }
    return out


def _ascii_safe(s: str) -> str:
    """Windows cp1252 consoles choke on ≈/–/× from PREDICTIONS.md."""
    if s is None:
        return "PENDING"
    repl = {
        "\u2248": "~", "\u2013": "-", "\u2014": "-", "\u00d7": "x",
        "\u2212": "-", "\u2264": "<=", "\u2265": ">=", "\u00b1": "+/-",
        "\u03bc": "u", "\u2192": "->", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    }
    out = str(s)
    for a, b in repl.items():
        out = out.replace(a, b)
    return out.encode("ascii", "replace").decode("ascii")


def print_prediction(preds: dict, part: str):
    body = preds.get(part, "PENDING")
    ids = PART_ELIGIBLE.get(part, ())
    print(f"\n=== W10 prediction for `{part}` ===")
    if ids:
        print(f"eligible IDs: {', '.join(ids)}")
    else:
        print("eligible IDs: (none - blocked / pending laptop gate)")
    print(_ascii_safe(body) if body else "PENDING")
    print("=== end prediction ===\n")


def prediction_status(preds: dict, part: str) -> str:
    """loaded | PENDING — for JSONL stamp rows."""
    if part in PART_PENDING_REASON and not PART_ELIGIBLE.get(part):
        return "PENDING"
    body = preds.get(part, "PENDING")
    if not body or body == "PENDING" or str(body).startswith("PENDING"):
        meta = preds.get("_meta") or {}
        if meta.get("predictions_path") or meta.get("transfer_model_path"):
            return "PENDING" if not PART_ELIGIBLE.get(part) else "loaded"
        return "PENDING"
    return "loaded"

def _level_dims(level: int):
    R, T, K, Mk = LEVELS[level]
    return R, T, K, Mk


def _scenarios(level: int, S: int, seed: int = 0):
    R, T, K, Mk = _level_dims(level)
    p = refinery_lp(R, T, K, Mk, seed=seed)
    rng = np.random.default_rng(1000 + seed + level)
    return p, p.c[:, None] * rng.uniform(0.9, 1.1, (p.n, S))


def _try_import(path_dots: str):
    try:
        mod = __import__(path_dots, fromlist=["*"])
        return mod, None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


# ====================================================================== stamp + pre-flight
def _probe_bandwidth(n: int = 1 << 20):
    """Host triad proxy (CPU). Full A100 run replaces with device triad."""
    a = np.ones(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float64)
    c = np.empty(n, dtype=np.float64)
    t0 = time.perf_counter()
    for _ in range(3):
        np.add(a, b, out=c)
        c *= 1.1
    elapsed = time.perf_counter() - t0
    bytes_moved = 3 * 3 * n * 8  # rough: 3 passes, triad-ish
    gbps = (bytes_moved / elapsed) / 1e9 if elapsed > 0 else None
    return {"probe": "host_triad_proxy", "n": n, "elapsed_s": elapsed, "approx_GB_s": gbps}


def _probe_fp64_gemm(n: int = 256):
    a = np.random.default_rng(0).standard_normal((n, n))
    b = np.random.default_rng(1).standard_normal((n, n))
    t0 = time.perf_counter()
    _ = a @ b
    elapsed = time.perf_counter() - t0
    flops = 2.0 * n * n * n
    tflops = (flops / elapsed) / 1e12 if elapsed > 0 else None
    return {"probe": "host_fp64_gemm", "n": n, "elapsed_s": elapsed, "approx_TFLOP_s": tflops}


def _probe_launch_latency(reps: int = 200):
    """Host proxy for kernel launch / sync latency (GPU path fills real numbers)."""
    x = np.zeros(1024)
    t0 = time.perf_counter()
    for _ in range(reps):
        x += 1.0
    elapsed = time.perf_counter() - t0
    return {"probe": "host_launch_proxy", "reps": reps, "elapsed_s": elapsed,
            "per_call_us": (elapsed / reps) * 1e6 if reps else None}


def _probe_overhead_profile(level: int, S: int):
    """Tiny L* x S PDHG wall vs kernel-ish work; skeleton uses host solve time share."""
    p, C = _scenarios(level, S, seed=7)
    opts = pdhg.PDHGOptions(tol=1e-3, time_limit=30.0)
    t0 = time.perf_counter()
    br = pdhg.solve_batch(p.A, C, p.lc, p.uc, p.lx, p.ux, p.c0, opts, GPU)
    sync()
    wall = time.perf_counter() - t0
    return {
        "probe": "overhead_L_x_S",
        "level": level,
        "S": S,
        "rows": p.m,
        "cols": p.n,
        "wall_s": wall,
        "iterations_max": int(br.iterations.max()) if hasattr(br, "iterations") else None,
        "statuses": list(br.status),
        "note": "skeleton: full e6_overhead_profile wires in when gpu_lab is present",
    }


def part_stamp(rec, quick, preds):
    print_prediction(preds, "stamp")
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

    level = 2 if quick else 4
    S = 4 if quick else 512
    probes = {
        "bandwidth": _probe_bandwidth(1 << 18 if quick else 1 << 22),
        "fp64_gemm": _probe_fp64_gemm(64 if quick else 512),
        "launch_latency": _probe_launch_latency(50 if quick else 500),
    }
    try:
        probes["overhead_profile"] = _probe_overhead_profile(level, S)
    except Exception as e:  # noqa: BLE001
        probes["overhead_profile"] = {"status": "error", "error": f"{type(e).__name__}: {e}"[:400]}

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
        "predictions_status": prediction_status(preds, "stamp"),
        "predictions_excerpt": (preds.get("stamp") or "PENDING")[:800],
        "eligible_ids": list(ELIGIBLE_IDS),
        "transfer_model_path": (preds.get("_meta") or {}).get("transfer_model_path"),
        "probes": probes,
        "quick": bool(quick),
        "budget_s": BUDGETS_S["stamp"],
    }
    rec.add("stamp", row)
    print("stamp:", row["nvidia_smi"] or "no GPU", "|", row["gpu_status"],
          "| native", row["native_simplex"], "| preds", row["predictions_status"],
          "|", (row["commit"] or "?")[:12])
    free_gpu()


# ====================================================================== crude valuation
def part_crude(rec, quick, preds):
    print_prediction(preds, "crude")
    # W10: blocked pending P-W02-CRUDE. Quick still smokes the harness; full run records PENDING only.
    if not quick and not PART_ELIGIBLE.get("crude"):
        key = "gate/P-W02-CRUDE"
        if not rec.done(key):
            rec.add(key, {
                "status": "PENDING",
                "prediction": preds.get("crude") or PART_PENDING_REASON["crude"],
                "note": "not on W11 burn list until W02 laptop gate passes",
            })
            print("crude: PENDING (P-W02-CRUDE)")
        return

    levels = (2,) if quick else (4, 5)
    n_crudes = 2 if quick else 10
    pct = 0.1 if quick else 0.30

    mod, err = _try_import("qenivo.workload.crude_value")
    for level in levels:
        key = f"L{level}/crudes{n_crudes}/pct{pct:g}"
        if rec.done(key):
            continue
        R, T, K, Mk = _level_dims(level)
        p = refinery_lp(R, T, K, Mk, seed=0)
        row = {
            "level": level, "rows": p.m, "cols": p.n, "nnz": int(p.A.nnz),
            "n_crudes": n_crudes, "pct": pct, "backend": GPU,
            "prediction": (preds.get("crude") or "PENDING")[:400],
            "quick": bool(quick), "budget_s": BUDGETS_S["crude"],
        }
        if mod is None:
            t0 = time.perf_counter()
            br = pdhg.solve(p, tol=1e-3, backend=GPU, time_limit=15.0 if quick else 120.0)
            sync()
            col = br.column(0)
            kkt = _host_kkt(p, col["x"], col["y"])
            row.update({
                "status": "skeleton_pdhg_smoke",
                "module": "PENDING (qenivo.workload.crude_value)",
                "module_error": err,
                "time": time.perf_counter() - t0,
                "pdhg_status": col["status"],
                "host_kkt": kkt,
                "note": "quick harness only; full CVE blocked pending W02 gate",
            })
        else:
            t0 = time.perf_counter()
            try:
                fn = getattr(mod, "crude_value", None) or getattr(qenivo, "crude_value", None)
                if fn is None:
                    raise AttributeError("crude_value not found")
                col = min(n_crudes - 1, p.n - 1)
                result = fn(p, col, (-pct, pct))
                row.update({
                    "status": "ok",
                    "time": time.perf_counter() - t0,
                    "n_segments": len(getattr(result, "segments", []) or []),
                    "n_breakpoints": len(getattr(result, "breakpoints", []) or []),
                })
            except Exception as e:  # noqa: BLE001
                row.update({
                    "status": "error",
                    "time": time.perf_counter() - t0,
                    "error": f"{type(e).__name__}: {e}"[:500],
                    "traceback": traceback.format_exc()[-800:],
                })
        rec.add(key, row)
        print(f"crude {key}: {row.get('status')} {row.get('time', 0):.2f}s")
        free_gpu()


# ====================================================================== stack matrix
def part_stack(rec, quick, preds):
    print_prediction(preds, "stack")
    levels = (2,) if quick else (4, 5)
    kinds = STACK_KINDS_QUICK if quick else STACK_KINDS_FULL
    Ss = STACK_S_QUICK if quick else STACK_S_FULL
    tol = 1e-3 if quick else 1e-4
    kind_candidate = {
        "cargo_menu": "C-STACK-CARGO",
        "factor_price": "C-STACK-FPRICE",
        "independent": "C-BATCH-L4",
        "one_crude": "C-STACK-CARGO",  # quick alias if present
    }

    stacks_mod, stacks_err = _try_import("qenivo.workload.stacks")
    router_mod, router_err = _try_import("qenivo.workload.router")

    for level in levels:
        for kind in kinds:
            for S in Ss:
                key = f"L{level}/{kind}/S{S}/tol{tol:g}"
                if rec.done(key):
                    continue
                cand = kind_candidate.get(kind, "C-BATCH-L4")
                try:
                    if stacks_mod is not None and hasattr(stacks_mod, "make_stack"):
                        p, C = stacks_mod.make_stack(level=level, kind=kind, S=S, seed=0)
                    else:
                        p, C = _scenarios(level, S, seed=hash(kind) % 10_000)
                    opts = pdhg.PDHGOptions(tol=tol, time_limit=60.0 if quick else 600.0)
                    t0 = time.perf_counter()
                    if router_mod is not None and hasattr(router_mod, "route"):
                        _ = router_mod.route(p, batch=S, engine="auto", backend=GPU)
                    br = pdhg.solve_batch(p.A, C, p.lc, p.uc, p.lx, p.ux, p.c0, opts, GPU)
                    sync()
                    gpu_t = time.perf_counter() - t0
                    if certify_batch is not None:
                        ok, worst = certify_batch(p, C, br, tol)
                    else:
                        ok, worst = None, None
                    row = {
                        "level": level, "kind": kind, "S": S, "tol": tol,
                        "candidate_id": cand,
                        "rows": p.m, "cols": p.n, "nnz": int(p.A.nnz),
                        "gpu_time": gpu_t, "certified": ok, "worst_rel": worst,
                        "iterations_max": int(br.iterations.max()),
                        "statuses": list(br.status),
                        "stacks_module": "ok" if stacks_mod else f"PENDING ({stacks_err})",
                        "router_module": "ok" if router_mod else f"PENDING ({router_err})",
                        "prediction": (preds.get("stack") or "PENDING")[:500],
                        "quick": bool(quick), "backend": GPU,
                        "budget_s": BUDGETS_S["stack"],
                        "note": "HiGHS cold/warm columns fill when stack_bench helpers land",
                    }
                    rec.add(key, row)
                    print(f"stack {key} [{cand}]: GPU {gpu_t:.2f}s certified {ok}/{S}")
                except Exception as e:  # noqa: BLE001
                    rec.add(key, {
                        "level": level, "kind": kind, "S": S, "status": "error",
                        "candidate_id": cand,
                        "error": f"{type(e).__name__}: {e}"[:500],
                        "traceback": traceback.format_exc()[-800:],
                        "prediction": (preds.get("stack") or "PENDING")[:500],
                    })
                    print(f"stack {key}: ERROR {type(e).__name__}: {e}")
                free_gpu()


# ====================================================================== GPU byte diet
def part_diet(rec, quick, preds):
    print_prediction(preds, "diet")
    if not quick and not PART_ELIGIBLE.get("diet"):
        key = "gate/P-W04-DIET"
        if not rec.done(key):
            rec.add(key, {
                "status": "PENDING",
                "prediction": preds.get("diet") or PART_PENDING_REASON["diet"],
                "note": "not on W11 burn list until W04 ≥2× certified gate",
            })
            print("diet: PENDING (P-W04-DIET)")
        return

    levels = (2,) if quick else (4, 5)
    Ss = (2,) if quick else (512, 4096)
    tols = (1e-3,) if quick else (1e-4, 1e-6)
    diet_mod, diet_err = _try_import("qenivo.kernels.byte_diet")

    for level in levels:
        for S in Ss:
            for tol in tols:
                for mode in ("baseline", "diet"):
                    key = f"L{level}/S{S}/tol{tol:g}/{mode}"
                    if rec.done(key):
                        continue
                    if mode == "diet" and diet_mod is None:
                        rec.add(key, {
                            "level": level, "S": S, "tol": tol, "mode": mode,
                            "status": "PENDING",
                            "module": "qenivo.kernels.byte_diet",
                            "module_error": diet_err,
                            "prediction": (preds.get("diet") or "PENDING")[:400],
                            "quick": bool(quick),
                            "note": "W04 byte-diet not in this tree yet",
                        })
                        print(f"diet {key}: PENDING")
                        continue
                    try:
                        p, C = _scenarios(level, S)
                        opts = pdhg.PDHGOptions(tol=tol, time_limit=30.0 if quick else 300.0)
                        t0 = time.perf_counter()
                        br = pdhg.solve_batch(p.A, C, p.lc, p.uc, p.lx, p.ux, p.c0, opts, GPU)
                        sync()
                        wall = time.perf_counter() - t0
                        ok, worst = (certify_batch(p, C, br, tol) if certify_batch else (None, None))
                        rec.add(key, {
                            "level": level, "S": S, "tol": tol, "mode": mode,
                            "rows": p.m, "cols": p.n, "time": wall,
                            "certified": ok, "worst_rel": worst,
                            "status": "ok", "backend": GPU,
                            "prediction": (preds.get("diet") or "PENDING")[:400],
                            "quick": bool(quick),
                        })
                        print(f"diet {key}: {wall:.2f}s")
                    except Exception as e:  # noqa: BLE001
                        rec.add(key, {
                            "level": level, "S": S, "tol": tol, "mode": mode,
                            "status": "error", "error": f"{type(e).__name__}: {e}"[:500],
                        })
                        print(f"diet {key}: ERROR {e}")
                    free_gpu()


# ====================================================================== stack simplex
def part_simplex(rec, quick, preds):
    print_prediction(preds, "simplex")
    key = "stack_simplex/gate"
    if rec.done(key):
        return
    mod, err = _try_import("qenivo.engines.stack_simplex")
    w05_gate = os.environ.get("W05_PASSED", "").strip().lower() in ("1", "true", "yes")
    if not PART_ELIGIBLE.get("simplex") or mod is None or not w05_gate:
        rec.add(key, {
            "status": "PENDING",
            "module": "qenivo.engines.stack_simplex",
            "module_error": err,
            "w05_passed": w05_gate,
            "prediction": preds.get("simplex") or PART_PENDING_REASON["simplex"],
            "quick": bool(quick),
            "note": "P-W05-SSX — run only if W05 kill test passed; set W05_PASSED=1",
            "budget_s": BUDGETS_S["simplex"],
        })
        print("simplex: PENDING (P-W05-SSX / module / gate)")
        return

    Ss = (4,) if quick else (512, 1024, 2048, 4096)
    level = 2 if quick else 4
    for S in Ss:
        k = f"L{level}/S{S}/vertices"
        if rec.done(k):
            continue
        t0 = time.perf_counter()
        p, C = _scenarios(level, min(S, 4 if quick else S))
        elapsed = time.perf_counter() - t0
        rec.add(k, {
            "level": level, "S": S, "status": "skeleton",
            "time": elapsed, "rows": p.m, "cols": p.n,
            "prediction": (preds.get("simplex") or "PENDING")[:400],
            "quick": bool(quick),
        })
        print(f"simplex {k}: skeleton {elapsed:.2f}s")


# ====================================================================== large LPs
def part_large_lp(rec, quick, preds):
    print_prediction(preds, "large_lp")
    from fetch import fetch

    names = LARGE_LP_QUICK if quick else LARGE_LP_FULL
    tols = (1e-3,) if quick else (1e-4,)
    trial_cap = 15.0 if quick else 60.0
    cand = "C-PERSIST-S1"

    for name in names:
        path = fetch(name, verbose=False)
        prob = qenivo.read(path)
        for tol in tols:
            for tag, kwargs in (
                ("pdhg_persistent_on", {"persistent": "on"}),
                ("pdhg_persistent_off", {"persistent": "off"}),
            ):
                key = f"{name}/{tag}/tol{tol:g}"
                if rec.done(key):
                    continue
                try:
                    t0 = time.perf_counter()
                    br = pdhg.solve(prob, tol=tol, backend=GPU, time_limit=trial_cap, **kwargs)
                    sync()
                    elapsed = time.perf_counter() - t0
                    col = br.column(0)
                    kkt = _host_kkt(prob, col["x"], col["y"])
                    rec.add(key, {
                        "instance": name, "method": tag, "tol_request": tol,
                        "candidate_id": cand,
                        "time": elapsed, "iterations": col["iterations"],
                        "status": col["status"], "host_kkt": kkt,
                        "rows": prob.m, "cols": prob.n, "backend": GPU,
                        "prediction": (preds.get("large_lp") or "PENDING")[:500],
                        "quick": bool(quick), "trial_cap_s": trial_cap,
                    })
                    print(f"large_lp {name} {tag} [{cand}]: {col['status']} {elapsed:.2f}s")
                except Exception as e:  # noqa: BLE001
                    rec.add(key, {
                        "instance": name, "method": tag, "status": "error",
                        "candidate_id": cand,
                        "error": f"{type(e).__name__}: {e}"[:500],
                    })
                    print(f"large_lp {name} {tag}: ERROR {e}")
                free_gpu()

            key = f"{name}/crossover/tol{tol:g}"
            if not rec.done(key):
                try:
                    t0 = time.perf_counter()
                    br = pdhg.solve(prob, tol=1e-3, backend=GPU, time_limit=min(trial_cap, 30.0))
                    sync()
                    col = br.column(0)
                    out = crossover.crossover(prob, col["x"], col["y"],
                                              time_limit=max(1.0, trial_cap - (time.perf_counter() - t0)))
                    kkt = _host_kkt(prob, out["x"], out["y"])
                    rec.add(key, {
                        "instance": name, "method": "pdhg_then_crossover",
                        "candidate_id": cand,
                        "time": time.perf_counter() - t0, "status": out.get("status"),
                        "host_kkt": kkt, "backend": GPU,
                        "prediction": (preds.get("large_lp") or "PENDING")[:500],
                        "quick": bool(quick),
                        "cupdlpx_ref": "see e2_large_lp.jsonl (not re-run)",
                    })
                    print(f"large_lp {name} crossover [{cand}]: {out.get('status')}")
                except Exception as e:  # noqa: BLE001
                    rec.add(key, {"instance": name, "method": "crossover", "status": "error",
                                  "candidate_id": cand,
                                  "error": f"{type(e).__name__}: {e}"[:500]})
                    print(f"large_lp {name} crossover: ERROR {e}")
                free_gpu()


# ====================================================================== QP + MILP (CPU)
def part_qp_milp(rec, quick, preds):
    print_prediction(preds, "qp_milp")
    key = "qp_milp/skeleton"
    if rec.done(key):
        return
    if not quick and not PART_ELIGIBLE.get("qp_milp"):
        rec.add(key, {
            "status": "PENDING",
            "prediction": preds.get("qp_milp") or PART_PENDING_REASON["qp_milp"],
            "quick": False,
            "budget_s": BUDGETS_S["qp_milp"],
            "note": "blocked — laptop MILP/QP does not beat HiGHS",
        })
        print("qp_milp: PENDING (B-W07 / W08)")
        return

    qp_mod, qp_err = _try_import("qenivo.engines.pdqp")
    milp_mod, milp_err = _try_import("qenivo.engines.milp")

    row = {
        "prediction": (preds.get("qp_milp") or "PENDING")[:400],
        "quick": bool(quick),
        "budget_s": BUDGETS_S["qp_milp"],
        "qp_module": "ok" if qp_mod else f"PENDING ({qp_err})",
        "milp_module": "ok" if milp_mod else f"PENDING ({milp_err})",
        "note": "quick harness smoke only; full W07/W08 blocked pending laptop gate",
    }

    level = 2 if quick else 3
    p = refinery_lp(*_level_dims(level), seed=0)
    t0 = time.perf_counter()
    try:
        from qenivo.engines.simplex import solve_simplex
        sol = solve_simplex(p, tol=1e-6, time_limit=20.0 if quick else 120.0)
        row.update({
            "status": "skeleton_simplex_smoke",
            "time": time.perf_counter() - t0,
            "level": level, "rows": p.m, "cols": p.n,
            "simplex_status": sol.get("status") if isinstance(sol, dict) else str(type(sol)),
        })
    except Exception as e:  # noqa: BLE001
        row.update({
            "status": "error",
            "time": time.perf_counter() - t0,
            "error": f"{type(e).__name__}: {e}"[:500],
        })
    rec.add(key, row)
    print(f"qp_milp: {row.get('status')} {row.get('time', 0):.2f}s")


# ====================================================================== headline
def _cupdlpx_from_e2(instance: str, tol: float = 1e-4):
    key = f"{instance}/tol{tol:g}"
    for r in _load_jsonl(E2_CUPDLPX):
        if r.get("key") == key or (r.get("instance") == instance and r.get("tol") == tol):
            return r.get("cupdlpx"), r
    return None, None


def part_headline(run_dir: Path, preds: dict):
    stamp = _load_jsonl(run_dir / "stamp.jsonl")
    crude = _load_jsonl(run_dir / "crude.jsonl")
    stack = _load_jsonl(run_dir / "stack.jsonl")
    diet = _load_jsonl(run_dir / "diet.jsonl")
    simplex = _load_jsonl(run_dir / "simplex.jsonl")
    large = _load_jsonl(run_dir / "large_lp.jsonl")
    qp = _load_jsonl(run_dir / "qp_milp.jsonl")

    L = [
        "# QENIVO — Round 3 headline (A100)",
        "",
        "Numbers below are copied from recorded JSONL rows only. Nothing is typed by hand.",
        f"W10 predictions file: `{(preds.get('_meta') or {}).get('predictions_path') or _predictions_path() or 'PENDING'}`",
        f"W10 transfer model: `{(preds.get('_meta') or {}).get('transfer_model_path') or _transfer_model_path() or 'PENDING'}`",
        f"Eligible IDs: `{', '.join(ELIGIBLE_IDS)}`",
        "",
    ]

    env = next((r for r in stamp if r.get("key") == "stamp"), None)
    if env:
        L += [
            "## Environment",
            f"- commit `{env.get('commit')}` · qenivo `{env.get('qenivo')}` · backend `{env.get('backend')}`",
            f"- GPU: `{env.get('nvidia_smi')}` · kernels: `{env.get('gpu_status')}`",
            f"- native simplex: `{env.get('native_simplex')}`",
            f"- predictions: `{env.get('predictions_status')}` · eligible `{env.get('eligible_ids')}`",
            "",
        ]

    def section(title: str, part: str, rows: list):
        L.append(f"## {title}")
        L.append("")
        ids = PART_ELIGIBLE.get(part, ())
        if ids:
            L.append(f"**Eligible:** {', '.join(ids)}")
        else:
            L.append("**Eligible:** none (PENDING / blocked gate)")
        L.append("")
        L.append(f"**Prediction:** {(preds.get(part) or 'PENDING')[:600]}")
        L.append("")
        if not rows:
            L.append("_no rows yet_")
            L.append("")
            return
        L.append("| key | status / note | time (s) | extras |")
        L.append("|---|---|---:|---|")
        for r in rows:
            st = r.get("status") or r.get("mode") or r.get("method") or r.get("candidate_id") or ""
            tm = r.get("time", r.get("gpu_time"))
            tm_s = f"{tm:.2f}" if isinstance(tm, (int, float)) else "—"
            extra = r.get("candidate_id") or r.get("module") or r.get("certified") or r.get("host_kkt") or ""
            if isinstance(extra, dict):
                extra = f"max_rel={extra.get('max_rel')}"
            L.append(f"| {r.get('key')} | {st} | {tm_s} | {extra} |")
        L.append("")

    section("Stamp / pre-flight", "stamp", stamp)
    section("Crude Valuation Engine", "crude", crude)
    section("Stack matrix", "stack", stack)
    section("GPU byte diet", "diet", diet)
    section("Stack Simplex", "simplex", simplex)
    section("Large LPs", "large_lp", large)
    section("QP + MILP (CPU)", "qp_milp", qp)

    L += [
        "## cuPDLPx reference",
        "",
        f"Large-LP comparator rows (not re-run): `{E2_CUPDLPX.as_posix()}`",
        "",
    ]

    out = run_dir / "HEADLINE_ROUND3.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(_ascii_safe(out.read_text(encoding="utf-8")))


# ====================================================================== driver
PARTS = {
    "stamp": ("stamp", part_stamp),
    "crude": ("crude", part_crude),
    "stack": ("stack", part_stack),
    "diet": ("diet", part_diet),
    "simplex": ("simplex", part_simplex),
    "large_lp": ("large_lp", part_large_lp),
    "qp_milp": ("qp_milp", part_qp_milp),
}


def main(argv=None):
    # A cp1252 console (Windows) cannot print every character in the predictions and headline; a long
    # run must not die at its last print. Unencodable characters become '?' instead.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--only", default="stamp,crude,stack,diet,simplex,large_lp,qp_milp,headline")
    ap.add_argument("--quick", action="store_true",
                    help="L2 + tiny S on CPU for every part (no GPU required)")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    preds = load_predictions()
    print(f"round3_campaign backend={GPU} quick={a.quick} run_dir={run_dir}")
    credit = sum(BUDGETS_S.get(t.strip(), 0) for t in a.only.split(",") if t.strip() in BUDGETS_S)
    print(f"budget credit estimate (full-run caps): {credit / 60:.0f} min for selected parts")

    for tag in a.only.split(","):
        tag = tag.strip()
        if not tag:
            continue
        if tag == "headline":
            part_headline(run_dir, preds)
            continue
        if tag not in PARTS:
            raise SystemExit(f"unknown part {tag!r}; choose from {list(PARTS) + ['headline']}")
        name, fn = PARTS[tag]
        rec = Recorder(run_dir, name)
        t0 = time.perf_counter()
        try:
            fn(rec, a.quick, preds)
            print(f"--- {tag} done in {time.perf_counter() - t0:.1f}s")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print(f"--- {tag} FAILED after {time.perf_counter() - t0:.1f}s (saved rows kept)")
            raise


if __name__ == "__main__":
    main()
