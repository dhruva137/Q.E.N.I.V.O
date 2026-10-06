"""Multi-core execution: a parallel case pool and a verifier-gated engine race.

  solve_parallel(problems, workers)   independent models (what-if cases, a case stack, a batch of
                                      files) spread over CPU cores, one process per core
                                      (processes, not threads: each engine is single-threaded
                                      Python and must not share the interpreter lock)
  race(problem, engines)              the same model on several engines at once, one per core;
                                      the first answer whose verdict passes the independent
                                      float64 check wins and the other processes are stopped

Every answer returned is the certified Solution of the process that produced it; nothing is
trusted from an engine's word alone. Processes use the spawn start method, so the pool behaves
the same on Windows (the planners' desktops) and Linux servers.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait


def _solve_one(args):
    prob, engine, tol, time_limit, backend = args[:5]
    want_cert = len(args) > 5 and args[5]
    from ..api import solve
    t = time.perf_counter()
    s = solve(prob, engine=engine, tol=tol, time_limit=time_limit, backend=backend)
    return {"name": prob.name, "verdict": s.verdict, "objective": s.objective, "engine": s.engine.get("engine"),
            "x": s.x, "time": time.perf_counter() - t, "pid": os.getpid(), "summary": s.summary(),
            "certificate": s.certificate() if want_cert else None}


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def solve_parallel(problems, workers: int | None = None, engine: str = "auto", tol: float = 1e-6,
                   time_limit: float = 3600.0) -> dict:
    """Solve independent models on `workers` processes. Returns results in input order plus the
    wall time; `speedup_estimate` compares against the sum of the per-model times."""
    workers = workers or default_workers()
    jobs = [(p, engine, tol, time_limit, "numpy") for p in problems]
    t0 = time.perf_counter()
    if workers == 1:
        out = [_solve_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            out = list(ex.map(_solve_one, jobs, chunksize=max(1, len(jobs) // (4 * workers))))
    wall = time.perf_counter() - t0
    busy = sum(r["time"] for r in out)
    return {"results": out, "wall": wall, "workers": workers, "sum_of_solve_times": busy,
            "speedup_estimate": busy / wall if wall > 0 else None,
            "processes_used": len({r["pid"] for r in out})}


def race(prob, engines=("simplex", "ipm", "pdhg"), tol: float = 1e-6, time_limit: float = 600.0) -> dict:
    """Run several engines on the same model at once; return the first certified verdict.

    A verdict counts only if it is 'optimal', 'infeasible' or 'unbounded' after the float64 check
    (so a fast engine that stops short cannot win). The losers are cancelled or terminated."""
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    ex = ProcessPoolExecutor(len(engines), mp_context=ctx)
    futs = {ex.submit(_solve_one, (prob, e, tol, time_limit, "numpy", True)): e for e in engines}
    winner, finished = None, []
    pending = set(futs)
    try:
        while pending and winner is None:
            done, pending = wait(pending, timeout=max(0.1, time_limit - (time.perf_counter() - t0)),
                                 return_when=FIRST_COMPLETED)
            if not done:
                break
            for f in done:
                try:
                    r = f.result()
                except Exception as e:  # noqa: BLE001
                    r = {"engine": futs[f], "verdict": "error", "error": repr(e)}
                r["engine_requested"] = futs[f]
                finished.append(r)
                if r.get("verdict") in ("optimal", "infeasible", "unbounded") and winner is None:
                    winner = r
    finally:
        for f in pending:
            f.cancel()
        for p in list(getattr(ex, "_processes", {}).values()):   # stop engines still running
            if p.is_alive():
                p.terminate()
        ex.shutdown(wait=False, cancel_futures=True)
    return {"winner": winner, "finished": [{k: v for k, v in r.items() if k not in ("x", "certificate")}
                                           for r in finished],
            "wall": time.perf_counter() - t0}
