"""Dynamic engine bench: PlanTicker on market tick paths vs a warm re-solve per tick.

Paths (refinery L2 / L3, deterministic in --seed):
  one_crude   lattice random walk of one crude's purchase price, u_k = k * h, k -> k +- 1, |u| <= 20%
  factor3     3-factor price path (crude level, product crack, freight) as an Ornstein-Uhlenbeck walk,
              loadings as workload/stacks.py factor_price
  capacity    unit-capacity shocks: each tick every (unit, site) multiplier relaxes 10% toward 1 and,
              with probability 0.15, one random group suffers an outage to U(0.85, 1.0) of its level

Per tick the same data go to
  ticker      qenivo.workload.dynamic.PlanTicker (instant answer from the held basis, else warm simplex)
  warm        our simplex warm-started from the previous tick's basis, plus the same KKT check
  highs_warm  (--highs) one highspy object kept across ticks (costs / bounds changed in place, so HiGHS
              warm-starts from its own basis), threads=1; timed run() only
and the ticker's objective is compared with the warm answer. --speculate adds a second ticker pass on the
one_crude lattice with a Speculator that pre-solves the two lattice neighbours in idle time
(run_pending between ticks; its work time is recorded separately).

Every tick is a row in the results file (all rows kept), plus a summary row per (level, path, mode).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qenivo.api import _finish  # noqa: E402
from qenivo.engines.simplex import solve_simplex  # noqa: E402
from qenivo.models.refinery import refinery_level  # noqa: E402
from qenivo.workload import dynamic as D  # noqa: E402
from qenivo.workload.stacks import groups  # noqa: E402


def stamp(args) -> dict:
    import scipy
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=30).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"],
                                    capture_output=True, text=True, timeout=30).stdout.strip())
    except Exception:  # noqa: BLE001
        commit, dirty = "unknown", None
    out = {"kind": "header", "commit": commit, "src_dirty": dirty, "machine": platform.node(),
           "platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
           "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
           "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "command": " ".join([Path(sys.executable).name] + sys.argv),
           "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
           "args": vars(args)}
    try:
        import psutil
        vm = psutil.virtual_memory()
        out.update(ram_gb=round(vm.total / 2 ** 30, 1), free_ram_gb=round(vm.available / 2 ** 30, 2))
    except Exception:  # noqa: BLE001
        pass
    try:
        import highspy
        out["highspy"] = getattr(highspy, "__version__", None) or "present"
    except Exception:  # noqa: BLE001
        out["highspy"] = "missing"
    from qenivo.engines.native_simplex import library
    out["native_simplex"] = library() is not None
    return out


# ------------------------------------------------------------------------------ paths
def path_one_crude(p, g, T, rng, h=0.002, lim=0.2):
    sel = g["crude_cols"][g["crude_k"] == 0]
    kmax = int(round(lim / h))
    k = 0
    for _ in range(T):
        k = int(np.clip(k + rng.choice([-1, 1]), -kmax, kmax))
        c = p.c.copy()
        c[sel] = p.c[sel] * (1.0 + k * h)
        yield D.with_data(p, c=c), {"k": k}


def lattice_problem(p, g, k, h=0.002):
    sel = g["crude_cols"][g["crude_k"] == 0]
    c = p.c.copy()
    c[sel] = p.c[sel] * (1.0 + k * h)
    return D.with_data(p, c=c)


def path_factor3(p, g, T, rng, sigma=0.002, kappa=0.01):
    cc, sc = g["crude_cols"], g["ship_cols"]
    tr = g["transport"][g["ship_e"]]
    f = np.zeros(3)
    for _ in range(T):
        f = (1.0 - kappa) * f + rng.normal(0.0, sigma, 3)
        c = p.c.copy()
        c[cc] = p.c[cc] * (1.0 + f[0])
        c[sc] = p.c[sc] - g["ship_price"] * f[1] + tr * f[2]
        yield D.with_data(p, c=c), {"f": [round(float(v), 6) for v in f]}


def path_capacity(p, g, T, rng, p_shock=0.15):
    units = list(g["cap_rows"])
    R = g["R"]
    mult = np.ones((len(units), R))
    for _ in range(T):
        mult = 1.0 + 0.9 * (mult - 1.0)
        ev = None
        if rng.random() < p_shock:
            u, s = int(rng.integers(len(units))), int(rng.integers(R))
            mult[u, s] *= rng.uniform(0.85, 1.0)
            ev = [units[u], s]
        uc = p.uc.copy()
        for ui, name in enumerate(units):
            rows = g["cap_rows"][name]
            uc[rows] = p.uc[rows] * mult[ui][g["site_of_block"]]
        yield D.with_data(p, uc=uc), {"shock": ev}


PATHS = {"one_crude": path_one_crude, "factor3": path_factor3, "capacity": path_capacity}


# ------------------------------------------------------------------------------ HiGHS warm
class HighsWarm:
    def __init__(self, p):
        import highspy
        self.hp = highspy
        inf = highspy.kHighsInf
        A = p.A.tocsc()
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = p.n, p.m
        f = lambda v: np.where(np.isfinite(v), v, np.sign(v) * inf)  # noqa: E731
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = p.c, f(p.lx), f(p.ux)
        lp.row_lower_, lp.row_upper_ = f(p.lc), f(p.uc)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("threads", 1)
        self.h.setOptionValue("presolve", "off")
        self.h.passModel(lp)
        self.f = f
        self.cur = p
        self.h.run()

    def tick(self, q):
        cur = self.cur
        ci = np.flatnonzero(q.c != cur.c)
        if ci.size:
            self.h.changeColsCost(ci.size, ci.astype(np.int32), q.c[ci])
        ri = np.flatnonzero((q.lc != cur.lc) | (q.uc != cur.uc))
        if ri.size:
            self.h.changeRowsBounds(ri.size, ri.astype(np.int32), self.f(q.lc[ri]), self.f(q.uc[ri]))
        self.cur = q
        t0 = time.perf_counter()
        self.h.run()
        dtm = time.perf_counter() - t0
        info = self.h.getInfo()
        return dtm, float(info.objective_function_value), int(info.simplex_iteration_count), \
            str(self.h.modelStatusToString(self.h.getModelStatus()))


# ------------------------------------------------------------------------------ runner
def run_path(p, g, level, path, T, seed, args, write, speculate=False):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    spec = None
    pos = {"k": 0}
    if speculate:
        spec = D.Speculator(predict=lambda tk, q: [lattice_problem(p, g, pos["k"] - 1),
                                                   lattice_problem(p, g, pos["k"] + 1)])
    tk = D.PlanTicker(p, tol=args.tol, speculator=spec)
    init = time.perf_counter() - t0
    warm = {"col_statuses": tk.first.extra["basis"]["col_statuses"],
            "row_statuses": tk.first.extra["basis"]["row_statuses"]}
    hw = HighsWarm(p) if args.highs and not speculate else None
    mode = "speculate" if speculate else "plain"
    agg = {"ticker": [], "warm": [], "warm_raw": [], "highs": [], "agree": [], "spec_work": 0.0}
    t_wall = time.perf_counter()
    for i, (q, info) in enumerate(PATHS[path](p, g, T, rng)):
        if speculate:
            pos["k"] = info["k"]
            ts = time.perf_counter()
            spec.run_pending()
            agg["spec_work"] += time.perf_counter() - ts
        s = tk.tick_problem(q)
        row = {"kind": "tick", "level": level, "path": path, "mode": mode, "i": i, **info,
               "t_path": s.engine.get("path"), "t_latency": s.engine["latency"], "t_verdict": s.verdict,
               "t_obj": s.objective, "t_iters": s.engine.get("iterations"),
               "t_max_rel": (s.residuals or {}).get("max_rel")}
        agg["ticker"].append(s.engine["latency"])
        if not speculate:
            tw = time.perf_counter()
            r = solve_simplex(q, tol=min(args.tol, 1e-9), time_limit=args.time_limit, start=warm)
            raw = time.perf_counter() - tw
            ws = _finish(q, r["status"], r["x"], r["y"], args.tol, {"engine": "simplex"})
            wt = time.perf_counter() - tw
            if r["status"] == "optimal":
                warm = r
            agree = None
            if ws.objective is not None and s.objective is not None:
                agree = abs(ws.objective - s.objective) / (1.0 + abs(ws.objective))
                agg["agree"].append(agree)
            agg["warm"].append(wt)
            agg["warm_raw"].append(raw)
            row.update(w_latency=wt, w_raw=raw, w_iters=r["iterations"], w_verdict=ws.verdict, w_obj=ws.objective,
                       rel_obj_diff=agree)
            if hw is not None:
                ht, hobj, hit, hst = hw.tick(q)
                agg["highs"].append(ht)
                row.update(h_latency=ht, h_obj=hobj, h_iters=hit, h_status=hst)
        write(row)
        if time.perf_counter() - t_wall > args.path_budget:
            write({"kind": "note", "level": level, "path": path, "mode": mode,
                   "note": f"path budget {args.path_budget}s reached after {i + 1} ticks"})
            break
    st = tk.stats()
    q = lambda v, a: float(np.quantile(v, a)) if len(v) else None  # noqa: E731
    summ = {"kind": "summary", "level": level, "path": path, "mode": mode, "m": p.m, "n": p.n,
            "ticks": st["ticks"], "init_s": init, "instant_fraction": st["instant_fraction"],
            "counts": {k: st.get(k) for k in ("instant", "speculative", "warm", "cold", "instant_rejected",
                                              "not_optimal")},
            "factor_time_total": st["factor_time"],
            "ticker_p50": q(agg["ticker"], 0.5), "ticker_p95": q(agg["ticker"], 0.95),
            "ticker_mean": float(np.mean(agg["ticker"])) if agg["ticker"] else None,
            "ticker_total": float(np.sum(agg["ticker"])),
            **{f"{k}_{a}": st.get(f"{k}_{a}") for k in ("instant_latency", "warm_latency", "speculative_latency")
               for a in ("p50", "p95", "max")}}
    if agg["warm"]:
        summ.update(warm_p50=q(agg["warm"], 0.5), warm_p95=q(agg["warm"], 0.95),
                    warm_mean=float(np.mean(agg["warm"])), warm_total=float(np.sum(agg["warm"])),
                    warm_raw_total=float(np.sum(agg["warm_raw"])),
                    speedup_total=float(np.sum(agg["warm"]) / max(np.sum(agg["ticker"]), 1e-12)),
                    speedup_p50=q(agg["warm"], 0.5) / max(q(agg["ticker"], 0.5), 1e-12),
                    max_rel_obj_diff=float(np.max(agg["agree"])) if agg["agree"] else None)
    if agg["highs"]:
        summ.update(highs_p50=q(agg["highs"], 0.5), highs_p95=q(agg["highs"], 0.95),
                    highs_total=float(np.sum(agg["highs"])))
    if spec is not None:
        spec.close()
        summ.update(spec_stats=dict(spec.stats), spec_idle_work_s=agg["spec_work"])
    write(summ)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--paths", nargs="+", default=list(PATHS))
    ap.add_argument("--ticks", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--path-budget", type=float, default=1500.0, help="wall seconds per (level, path)")
    ap.add_argument("--highs", action="store_true")
    ap.add_argument("--speculate", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / f"dynamic_{dt.date.today().isoformat()}.jsonl"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def write(row):
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=float) + "\n")

    write(stamp(args))
    for level in args.levels:
        p = refinery_level(level)
        g = groups(p)
        for k, path in enumerate(args.paths):
            s = run_path(p, g, level, path, args.ticks, args.seed + k, args, write)
            print(json.dumps({k2: s.get(k2) for k2 in ("level", "path", "ticks", "instant_fraction", "ticker_p50",
                                                       "warm_p50", "speedup_total", "highs_p50",
                                                       "max_rel_obj_diff", "counts")}, default=float), flush=True)
        if args.speculate and "one_crude" in args.paths:
            s = run_path(p, g, level, "one_crude", args.ticks, args.seed, args, write, speculate=True)
            print(json.dumps({k2: s.get(k2) for k2 in ("level", "mode", "ticks", "instant_fraction", "ticker_p50",
                                                       "counts", "spec_stats", "spec_idle_work_s")}, default=float),
                  flush=True)


if __name__ == "__main__":
    main()
