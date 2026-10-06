"""Public API: read a model, solve it, get a certified answer.

    import qenivo
    prob = qenivo.read("plan.mps")
    sol = qenivo.solve(prob)                 # routed, solved, checked, certified
    print(sol.status, sol.objective)
    sol.save_certificate("plan.cert.json")

Every Solution carries its verdict ("optimal", "infeasible", "unbounded" or "not_proven"),
the residuals that justify it, the engine that produced it and the reason that engine was
chosen. A verdict is never taken from an engine's word alone: optimality is re-checked on the
original model, and infeasible / unbounded verdicts need a ray that passes its own check.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .certify import certificate as _cert
from .certify.kkt import check_farkas, check_ray, kkt_residuals
from .model import Problem

PROVEN = ("optimal", "infeasible", "unbounded")


@dataclass
class Solution:
    status: str                       # engine status
    verdict: str                      # optimal | infeasible | unbounded | not_proven
    objective: float | None
    x: np.ndarray | None
    y: np.ndarray | None              # row duals, user's sense
    residuals: dict
    engine: dict
    problem: Problem
    tol: float
    ray: np.ndarray | None = None
    ray_check: dict | None = None
    source: str | None = None
    extra: dict = field(default_factory=dict)
    history: list | None = None       # first-order engines: residuals per check, for display only

    @property
    def is_proven(self) -> bool:
        return self.verdict in PROVEN

    @property
    def reduced_costs(self) -> np.ndarray | None:
        if self.x is None or self.y is None:
            return None
        p = self.problem
        lam = p.c - p.A.T @ (p.obj_sign * self.y)
        if p.is_qp:
            lam = lam + p.Q @ self.x
        return p.obj_sign * lam

    def certificate(self, include_solution: bool = True) -> dict:
        kind = {"optimal": "kkt", "infeasible": "farkas", "unbounded": "ray"}.get(self.verdict, "none")
        return _cert.build(self.problem, verdict=self.verdict, status=self.status, tol=self.tol,
                           residuals=self.residuals, evidence_kind=kind, x=self.x,
                           y=None if self.y is None else self.problem.obj_sign * self.y,
                           ray=self.ray, ray_check=self.ray_check, engine=self.engine,
                           source=self.source, include_solution=include_solution,
                           extra=self.extra or None)

    def save_certificate(self, path, include_solution: bool = True) -> None:
        _cert.save(self.certificate(include_solution), path)

    def table(self, which: str = "x", top: int | None = None):
        """(name, value) pairs of the primal values or the row duals (marginal values)."""
        rn, cn = self.problem.names()
        names, vals = (cn, self.x) if which == "x" else (rn, self.y)
        if vals is None:
            return []
        pairs = list(zip(names, np.asarray(vals).tolist()))
        if top:
            pairs = sorted(pairs, key=lambda t: -abs(t[1]))[:top]
        return pairs

    def summary(self) -> str:
        e = self.engine
        obj = "n/a" if self.objective is None else f"{self.objective:.10g}"
        r = self.residuals or {}
        if "max_integrality" in r:          # MILP: primal feasibility, integrality and the proven gap
            res = (f"max row violation {r.get('max_row_violation', 0):.2e}  max integrality "
                   f"{r.get('max_integrality', 0):.2e}  gap {r.get('gap', float('nan')):.2e}  "
                   f"bound {r.get('bound')}")
        else:
            res = (f"rel_primal {r.get('rel_primal', float('nan')):.2e}  rel_dual {r.get('rel_dual', float('nan')):.2e}"
                   f"  rel_gap {r.get('rel_gap', float('nan')):.2e}") if r else "no residuals"
        return (f"{self.problem.name}: {self.verdict.upper()}  objective {obj}\n"
                f"  engine {e.get('engine')} on {e.get('backend')}  iterations {e.get('iterations')}  "
                f"time {e.get('time', 0):.3f}s\n  {res}\n  routing: {e.get('reason')}")


def read(path, name: str | None = None) -> Problem:
    """Read an MPS file (free or fixed, .gz / .bz2 accepted)."""
    from .io.mps import read_mps
    return read_mps(path, name=name)


def _residuals(prob, x, y_internal):
    k = kkt_residuals(prob, x, y_internal)
    return {key: k[key] for key in ("objective", "dual_objective", "rel_primal", "rel_dual", "rel_gap",
                                    "max_rel", "primal_res", "dual_res", "gap")}


def _finish(prob, status, x, y_int, tol, engine, ray=None, ray_kind=None, source=None) -> Solution:
    """Turn an engine result into a Solution whose verdict is checked, not trusted."""
    res, ray_chk, verdict, obj = {}, None, "not_proven", None
    if x is not None and prob.n and np.all(np.isfinite(x)) and y_int is not None and np.all(np.isfinite(y_int)):
        res = _residuals(prob, x, y_int)
    if prob.is_empty:
        verdict = "optimal" if not prob.notes else "not_proven"
        res = {"objective": prob.obj_sign * prob.c0, "rel_primal": 0.0, "rel_dual": 0.0, "rel_gap": 0.0, "max_rel": 0.0}
        obj = res["objective"]
        engine = dict(engine, note="model has no columns")
    elif status == "optimal" and res and res["max_rel"] <= tol * 1.000001:
        verdict, obj = "optimal", res["objective"]
    elif status == "infeasible" and ray is not None:
        ray_chk = check_farkas(prob, ray)
        verdict = "infeasible" if ray_chk["valid"] else "not_proven"
    elif status == "unbounded" and ray is not None:
        ray_chk = check_ray(prob, ray)
        if ray_chk["valid"]:
            # a ray proves unboundedness only if the model also has a feasible point
            from .workload.explain import elastic_farkas
            fe = elastic_farkas(prob)
            if fe is None:
                verdict = "not_proven"
                engine = dict(engine, note="improving ray found, feasibility undecided")
            elif fe["violation"] <= 1e-9 * (1 + fe["scale"]):
                verdict = "unbounded"
                ray_chk = dict(ray_chk, feasible_point_violation=fe["violation"])
                x = fe["x"]
            else:
                far = check_farkas(prob, fe["ray"])
                if far["valid"]:
                    verdict, ray, ray_chk, status = "infeasible", fe["ray"], far, "infeasible"
                    engine = dict(engine, note="an improving ray exists, but the model is infeasible (Farkas proof)")
                else:
                    verdict = "not_proven"
        else:
            verdict = "not_proven"
    elif status == "optimal" and res:
        engine = dict(engine, note=f"engine said optimal but max relative residual {res['max_rel']:.2e} > {tol:g}")
    y_user = None if y_int is None else prob.obj_sign * np.asarray(y_int)
    return Solution(status=status, verdict=verdict, objective=obj, x=x, y=y_user, residuals=res,
                    engine=engine, problem=prob, tol=tol, ray=ray, ray_check=ray_chk, source=source)


BUILTIN_ENGINES = {"simplex", "ipm", "pdhg", "pdhg-rf", "pdqp", "milp"}


def _trace(history, case: int) -> list:
    """One case's residuals at each PDHG check (the batch records every still-active case)."""
    out = []
    for h in history or []:
        cols = list(h.get("cols") or [])
        if case in cols:
            k = cols.index(case)
            out.append({"iter": int(h["iter"]), "time": float(h["time"]), "rel_primal": float(h["rel_primal"][k]),
                        "rel_dual": float(h["rel_dual"][k]), "rel_gap": float(h["rel_gap"][k])})
    return out

def solve(prob: Problem | str, engine: str = "auto", tol: float = 1e-6, backend: str = "auto",
          time_limit: float = 3600.0, warm: "Solution | None" = None, polish: bool = True,
          verbose: bool = False, **options) -> Solution:
    """Solve one model. engine: auto | simplex | ipm | ipm-native | pdhg | pdhg-rf | pdqp | milp.

    auto routes convex QP to ipm-native (PDQP only for very large QPs / GPU batches).
    pdhg-rf iterates in float32 (fast on any GPU, including T4 and RTX cards) and refines the
    answer in float64 until the float64 KKT check passes; use it for tight tolerances (1e-8)."""
    from .workload.router import route
    source = None
    if isinstance(prob, (str, Path)):
        source = str(prob)
        prob = read(prob)
    t0 = time.perf_counter()
    r = route(prob, 1, engine, backend)
    eng, dev = r["engine"], r["backend"]
    meta = dict(r)

    if prob.is_empty:
        return _finish(prob, "optimal", np.zeros(0), np.zeros(prob.m), tol,
                       dict(meta, iterations=0, time=0.0), source=source)

    if eng not in BUILTIN_ENGINES:                  # a registered plugin engine (engines/registry.py)
        from .engines.registry import engines as _engines
        from .engines.registry import get_engine, problem_class
        spec = get_engine(eng)
        if spec is None:
            raise ValueError(f"unknown engine {eng!r}; available: {sorted(BUILTIN_ENGINES | set(_engines()))}")
        if problem_class(prob) not in spec.classes:
            raise ValueError(f"engine {eng!r} solves {spec.classes}, this model is {problem_class(prob)}")
        # Auto QP: budget Mehrotra, then let PDQP use a full time_limit on fallback.
        ipm_budget = time_limit
        if engine == "auto" and polish and prob.is_qp and eng == "ipm-native":
            ipm_budget = min(45.0, 0.4 * time_limit)
        sol = spec.fn(prob, tol=tol, time_limit=ipm_budget, backend=dev, verbose=verbose, **options)
        sol.source = source
        # Prefer the router's reason when engine was auto (plugins often hard-code "caller").
        if engine == "auto" and meta.get("reason"):
            sol.engine["reason"] = meta["reason"]
        else:
            sol.engine.setdefault("reason", meta.get("reason"))
        # Convex QP: if ipm-native stalls on dense Q / scaling, PDQP often certifies.
        if (engine == "auto" and polish and prob.is_qp and eng == "ipm-native"
                and sol.verdict != "optimal"):
            from .engines.pdqp import solve_qp
            # Full PDQP budget (IPM may have already burned time_limit). Use a tighter
            # inner tol so near-miss objectives (e.g. Maros DUALC*) clear published 1e-6
            # without relaxing the agree threshold — KKT still gates the claim.
            fb_tol = min(float(tol), 1e-8)
            if verbose:
                print(
                    f"PROGRESS stage=pdqp-fallback after ipm-native ({sol.status}) "
                    f"fb_tol={fb_tol:g} budget={time_limit:.0f}s",
                    flush=True,
                )
            out = solve_qp(
                prob, tol=fb_tol, backend="numpy", time_limit=time_limit, verbose=verbose,
                warm_x=sol.x, warm_y=None if sol.y is None else (prob.obj_sign * sol.y),
            )
            m2 = dict(
                meta, engine="pdqp", backend="numpy", iterations=out["iterations"],
                time=out["time"], restarts=out.get("restarts"),
                reason=f"fallback after ipm-native ({sol.status})",
                ladder=["ipm-native", "pdqp"],
                fallback_tol=fb_tol,
            )
            sol2 = _finish(prob, out["status"], out["x"], out["y"], fb_tol, m2, source=source)
            if sol2.verdict == "optimal":
                sol2.extra["wall_time"] = time.perf_counter() - t0
                return sol2
        sol.extra["wall_time"] = time.perf_counter() - t0
        return sol

    if eng == "simplex":
        from .engines.simplex import solve_simplex
        out = solve_simplex(prob, tol=min(tol, 1e-9), time_limit=time_limit)
        meta.update(iterations=out["iterations"], time=out["time"])
        sol = _finish(prob, out["status"], out["x"], out["y"], tol, meta, source=source)
        if out["status"] == "infeasible" and sol.verdict != "infeasible":
            sol = _certify_infeasible(prob, tol, time_limit, meta, source) or sol
        if sol.verdict != "optimal" and out["status"] in ("iteration_limit", "time_limit", "numerical_error"):
            return _fallback(prob, tol, t0 + time_limit, meta, source, "simplex did not finish")
        sol.extra["basis"] = {"col_statuses": out["col_statuses"], "row_statuses": out["row_statuses"]}
        return sol

    if eng == "ipm":
        from .engines.ipm import solve_ipm
        out = solve_ipm(prob, tol=tol, time_limit=time_limit)
        meta.update(iterations=out["iterations"], time=out["time"], regularisation=out["regularisation"],
                    factorisation=out.get("factor"))
        sol = _finish(prob, out["status"], out["x"], out["y"], tol, meta, source=source)
        if sol.verdict == "optimal":
            return sol
        if out["status"] in ("infeasible", "unbounded"):
            cert = _certify_infeasible(prob, tol, time_limit, meta, source)
            if cert is not None:
                return cert
        return _fallback(prob, tol, t0 + time_limit, meta, source, f"ipm ended with {out['status']}")

    if eng == "pdqp":
        from .engines.pdqp import solve_qp
        out = solve_qp(prob, tol=tol, backend=dev, time_limit=time_limit, verbose=verbose,
                       warm_x=None if warm is None else warm.x,
                       warm_y=None if warm is None else prob.obj_sign * warm.y)
        meta.update(iterations=out["iterations"], time=out["time"], restarts=out["restarts"])
        return _finish(prob, out["status"], out["x"], out["y"], tol, meta, source=source)

    if eng == "milp":
        from .engines.milp import solve_milp
        return solve_milp(prob, tol=tol, time_limit=time_limit, verbose=verbose, meta=meta, source=source)

    if eng == "pdhg-rf":
        from .engines import refine
        br = refine.solve(prob, tol=tol, backend=dev, time_limit=time_limit)
        col = br.column(0)
        tm = br.rays.get("_timing", {})
        meta.update(iterations=col["iterations"], refinement_rounds=col["restarts"], time=br.solve_time,
                    precision="float32 iterations, float64 refinement", polished=bool(tm.get("polished", [False])[0]))
        sol = _finish(prob, col["status"], col["x"], col["y"], tol, meta, source=source)
        if sol.verdict == "not_proven" and polish:
            return _fallback(prob, tol, t0 + time_limit, meta, source, f"pdhg-rf stopped at {col['status']}")
        sol.extra["wall_time"] = time.perf_counter() - t0
        return sol

    # pdhg
    from .engines import pdhg
    opts = pdhg.PDHGOptions(tol=tol, time_limit=time_limit, verbose=verbose,
                            **{k: v for k, v in options.items() if hasattr(pdhg.PDHGOptions, k)})
    br = pdhg.solve(prob, tol=tol, backend=dev, opts=opts,
                    warm_x=None if warm is None else warm.x,
                    warm_y=None if warm is None else prob.obj_sign * warm.y)
    col = br.column(0)
    meta.update(iterations=col["iterations"], restarts=col["restarts"],
                time=br.setup_time + br.solve_time, kernels=opts.kernels,
                primal_weight=col["primal_weight"])
    ray = col["ray"]["vector"] if col["ray"] else None
    sol = _finish(prob, col["status"], col["x"], col["y"], tol, meta, ray=ray, source=source)
    sol.history = _trace(br.history, 0)
    if sol.verdict == "not_proven" and col["status"] in ("iteration_limit", "time_limit") and polish \
            and prob.m <= 3000:
        return _fallback(prob, tol, t0 + time_limit, meta, source, f"pdhg stopped at {col['status']}")
    sol.extra["wall_time"] = time.perf_counter() - t0
    return sol


def _fallback(prob, tol, deadline, meta, source, why) -> Solution:
    """Robustness ladder: the next engine, and an infeasibility certificate if needed.

    deadline is the caller's start + time_limit (time.perf_counter clock): the whole solve, ladder
    included, keeps to the user's limit. Each rung gets the time left; with none left the ladder stops
    and the answer is not_proven. (Each rung used to get the full limit again, so a 900 s solve could
    run for 45 minutes.)
    """
    from .engines.ipm import solve_ipm
    from .engines.simplex import solve_simplex
    tried = [meta.get("engine")]

    def left() -> float:
        return deadline - time.perf_counter()

    for name in ("simplex", "ipm", "pdhg"):
        if name in tried or (name == "simplex" and prob.m > 3000):
            continue
        if left() <= 0:
            why = f"{why}; time limit reached"
            break
        t = time.perf_counter()
        if name == "pdhg":
            from .engines import pdhg as _pd
            br = _pd.solve(prob, tol=tol, time_limit=left())
            col = br.column(0)
            out = {"status": col["status"], "x": col["x"], "y": col["y"], "iterations": col["iterations"]}
            ray = col["ray"]["vector"] if col["ray"] else None
            m2 = dict(meta, engine="pdhg", backend="numpy", iterations=col["iterations"],
                      time=time.perf_counter() - t, reason=f"fallback: {why}", ladder=tried + [name])
            sol = _finish(prob, out["status"], out["x"], out["y"], tol, m2, ray=ray, source=source)
            if sol.is_proven:
                return sol
            tried.append(name)
            continue
        out = (solve_simplex(prob, tol=1e-9, time_limit=left()) if name == "simplex"
               else solve_ipm(prob, tol=tol, time_limit=left()))
        m2 = dict(meta, engine=name, backend="numpy", iterations=out["iterations"],
                  time=time.perf_counter() - t, reason=f"fallback: {why}", ladder=tried + [name])
        sol = _finish(prob, out["status"], out["x"], out["y"], tol, m2, source=source)
        if sol.verdict == "optimal":
            return sol
        tried.append(name)
        if out["status"] == "infeasible" and left() > 0:
            cert = _certify_infeasible(prob, tol, left(), m2, source)
            if cert is not None:
                return cert
    if left() > 0:
        cert = _certify_infeasible(prob, tol, left(), dict(meta, ladder=tried), source)
        if cert is not None:
            return cert
    return _finish(prob, "not_proven", None, None, tol, dict(meta, reason=f"no engine proved a verdict ({why})",
                                                             ladder=tried), source=source)


def _certify_infeasible(prob, tol, time_limit, meta, source) -> Solution | None:
    """Prove infeasibility with the elastic (phase-1) LP: its dual is a Farkas ray."""
    from .workload.explain import elastic_farkas
    out = elastic_farkas(prob, time_limit=time_limit)
    if out is None:
        return None
    # Trust check_farkas on the elastic duals. Do not require a large elastic
    # objective V: near-margin infeasibles (Netlib cplex2) can have V ~ 1e-9
    # while the unit ray still validates. Feasible models yield a zero ray.
    m2 = dict(meta, certificate_method="elastic phase-1 LP; its row duals form the Farkas ray",
              total_violation=out["violation"])
    sol = _finish(prob, "infeasible", None, None, tol, m2, ray=out["ray"], source=source)
    return sol if sol.verdict == "infeasible" else None
