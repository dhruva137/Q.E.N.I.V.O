"""Case stacks: one plan, many what-ifs, every answer certified.

A refinery re-plans with the same matrix and different data: crude prices, product prices,
demand, unit capacities, crude availability. A `Case` states only what differs from the base
plan (by column / row NAME). `solve_cases` runs the whole stack

  * on the GPU as ONE batched first-order solve (the matrix is read once per iteration for
    all cases; finished cases leave the batch), or
  * on the CPU, case by case, each warm-started from the base answer,

then checks every case on the original data and returns a certificate per case plus a
comparison table (objective change vs base, marginal values, decisions that moved).

`crude_breakeven` answers the spot-cargo question: over a range of offer prices for one crude,
how much does the plan buy, and what is the highest price at which the cargo still enters
the optimal slate?
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..api import Solution, _finish
from ..model import Problem
from .router import classify_stack, route, route_stack


@dataclass
class Case:
    name: str
    cost: dict = field(default_factory=dict)        # column -> new objective coefficient (user's sense)
    col_lo: dict = field(default_factory=dict)      # column -> new lower bound
    col_hi: dict = field(default_factory=dict)      # column -> new upper bound
    row_lo: dict = field(default_factory=dict)      # row -> new lower limit
    row_hi: dict = field(default_factory=dict)      # row -> new upper limit


@dataclass
class CaseReport:
    base: str
    cases: list                  # [Solution]
    names: list
    engine: dict
    wall_time: float

    def table(self, rows_of_interest=None) -> list[dict]:
        base = self.cases[0].objective if self.cases and self.cases[0].objective is not None else None
        out = []
        for name, s in zip(self.names, self.cases):
            d = {"case": name, "verdict": s.verdict, "objective": s.objective,
                 "delta_vs_first": (None if (s.objective is None or base is None) else s.objective - base),
                 "max_rel": (s.residuals or {}).get("max_rel"), "iterations": s.engine.get("iterations")}
            if rows_of_interest and s.y is not None:
                rn, _ = s.problem.names()
                idx = {r: i for i, r in enumerate(rn)}
                for r in rows_of_interest:
                    d[f"marginal[{r}]"] = float(s.y[idx[r]])
            out.append(d)
        return out

    def summary(self) -> str:
        n_ok = sum(s.verdict == "optimal" for s in self.cases)
        lines = [f"{len(self.cases)} cases on {self.base}: {n_ok} certified optimal, "
                 f"engine {self.engine.get('engine')} on {self.engine.get('backend')} "
                 f"({self.engine.get('reason')}), wall {self.wall_time:.2f}s"]
        for d in self.table():
            obj = "n/a" if d["objective"] is None else f"{d['objective']:.10g}"
            dl = "" if d["delta_vs_first"] is None else f"  ({d['delta_vs_first']:+.6g})"
            lines.append(f"  {d['case']:28s} {d['verdict']:11s} {obj}{dl}")
        return "\n".join(lines)


def _materialise(base: Problem, case: Case):
    rn, cn = base.names()
    ci = {c: j for j, c in enumerate(cn)}
    ri = {r: i for i, r in enumerate(rn)}
    c, lx, ux = base.c.copy(), base.lx.copy(), base.ux.copy()
    lc, uc = base.lc.copy(), base.uc.copy()
    for k, v in case.cost.items():
        c[ci[k]] = base.obj_sign * float(v)
    for k, v in case.col_lo.items():
        lx[ci[k]] = float(v)
    for k, v in case.col_hi.items():
        ux[ci[k]] = float(v)
    for k, v in case.row_lo.items():
        lc[ri[k]] = float(v)
    for k, v in case.row_hi.items():
        uc[ri[k]] = float(v)
    return c, lc, uc, lx, ux


def case_problem(base: Problem, case: Case) -> Problem:
    c, lc, uc, lx, ux = _materialise(base, case)
    p = Problem(c=c, A=base.A, lc=lc, uc=uc, lx=lx, ux=ux, c0=base.c0, obj_sign=base.obj_sign,
                name=f"{base.name}[{case.name}]", row_names=base.row_names, col_names=base.col_names,
                integer=base.integer)
    return p


def solve_cases(base: Problem, cases: list[Case], tol: float = 1e-6, engine: str = "auto",
                backend: str = "auto", time_limit: float = 3600.0, verbose: bool = False,
                exact_duals: bool = False, threads: int = 1) -> CaseReport:
    if base.is_qp or base.is_mip:
        raise ValueError("case stacks are LP-only in this release; solve QP/MILP cases one by one")
    t0 = time.perf_counter()
    S = len(cases)
    # Stack-structure router by default; engine= / --engine still forces a path.
    if engine == "auto":
        profile = classify_stack(base, cases)
        r = route_stack(profile, base=base, deltas=cases, engine="auto", backend=backend,
                        exact_duals=exact_duals, threads=threads)
    else:
        r = route(base, S, engine, backend)
        r = dict(r, route="forced", profile=None, order=None, clusters=None,
                 max_workers=1, exact_duals=exact_duals)
    probs = [case_problem(base, cs) for cs in cases]
    sols: list[Solution] = []
    if r["engine"] == "pdhg":
        from ..engines import pdhg
        mats = [_materialise(base, cs) for cs in cases]
        C = np.stack([m_[0] for m_ in mats], 1)
        LC, UC = np.stack([m_[1] for m_ in mats], 1), np.stack([m_[2] for m_ in mats], 1)
        LX, UX = np.stack([m_[3] for m_ in mats], 1), np.stack([m_[4] for m_ in mats], 1)
        opts = pdhg.PDHGOptions(tol=tol, time_limit=time_limit, verbose=verbose)
        br = pdhg.solve_batch(base.A, C, LC, UC, LX, UX, base.c0, opts, r["backend"])
        for j, p in enumerate(probs):
            col = br.column(j)
            meta = dict(r, iterations=col["iterations"], time=br.setup_time + br.solve_time,
                        batch_index=j, batch_size=S)
            ray = col["ray"]["vector"] if col["ray"] else None
            s = _finish(p, col["status"], col["x"], col["y"], tol, meta, ray=ray)
            if s.verdict == "not_proven":
                from ..api import solve as _solve
                s2 = _solve(p, tol=tol, time_limit=time_limit)
                s2.engine["reason"] = f"batch lane did not converge; re-solved alone ({s2.engine.get('reason')})"
                s = s2
            sols.append(s)
    else:
        from ..api import solve as _solve
        warm = None
        order = list(range(S))
        if r.get("order") in ("nn", "factor") and S > 1:
            try:
                from .stacks import nn_order
                shocks = []
                for cs in cases:
                    vec = [float(v) for d in (cs.cost, cs.col_lo, cs.col_hi, cs.row_lo, cs.row_hi)
                           for v in (d or {}).values()]
                    shocks.append(np.asarray(vec if vec else [0.0], dtype=np.float64))
                order = nn_order(shocks)
            except Exception:  # noqa: BLE001
                order = list(range(S))
        ordered_sols: list[Solution | None] = [None] * S
        for idx in order:
            p = probs[idx]
            s = _solve(p, engine=r["engine"] if r["engine"] != "auto" else "auto", tol=tol,
                       backend="numpy", time_limit=time_limit, warm=warm)
            s.engine = dict(s.engine or {}, route=r.get("route"), stack_reason=r.get("reason"),
                            profile=r.get("profile"), order=r.get("order"))
            ordered_sols[idx] = s
            if s.verdict == "optimal":
                warm = s
        sols = ordered_sols  # type: ignore[assignment]
    return CaseReport(base=base.name, cases=sols, names=[c.name for c in cases], engine=r,
                      wall_time=time.perf_counter() - t0)


def crude_breakeven(base: Problem, purchase_col: str, prices, tol: float = 1e-6,
                    backend: str = "auto") -> dict:
    """Sweep the offer price of one crude (objective coefficient of `purchase_col`, user's sense).

    Returns the purchased volume and plan value at each price, and the breakeven: the highest
    price in the sweep at which the plan still buys the cargo.
    """
    prices = [float(p) for p in prices]
    cases = [Case(name=f"{purchase_col}@{p:g}", cost={purchase_col: p}) for p in prices]
    rep = solve_cases(base, cases, tol=tol, backend=backend)
    _, cn = base.names()
    j = cn.index(purchase_col)
    vols = [None if s.x is None else float(s.x[j]) for s in rep.cases]
    objs = [s.objective for s in rep.cases]
    thresh = 1e-6 * (1 + (abs(base.ux[j]) if np.isfinite(base.ux[j]) else 1.0))
    buy = [p for p, v in zip(prices, vols) if v is not None and v > thresh]
    # objective coefficient = cost per unit in a minimise model, -(cost) in a maximise model;
    # the breakeven is the least favourable coefficient at which the cargo is still bought
    if buy:
        breakeven = max(buy) if base.obj_sign > 0 else min(buy)
    else:
        breakeven = None
    return {"column": purchase_col, "prices": prices, "volume": vols, "objective": objs,
            "breakeven_coefficient": breakeven, "bought_at": buy,
            "all_certified": all(s.verdict == "optimal" for s in rep.cases), "report": rep}
