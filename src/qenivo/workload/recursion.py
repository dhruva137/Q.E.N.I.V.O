"""Distributive recursion (successive LP) for pooled blend qualities.

Refinery planning systems (Aspen PIMS, Honeywell RPMS, Haverly GRTMPS) handle the bilinear
pooling terms "pool quality x pool flow" by recursion: guess each pool's quality, solve the
LP, recompute the qualities from the solution, repeat. Each pass is an LP that differs from
the previous one only in the quality coefficients of the specification rows.

A pool is described by a `Pool`:
    inputs   [(column, quality)]            streams entering the pool, with known quality
    outputs  [(column, row, spec)]          pool streams leaving to products; in `row`
                                            the coefficient of `column` is (q - spec)
The model is written with the pool coefficients at any placeholder; recursion overwrites them.

Engines per pass (the research question is which of these converge, and how fast):
    pdhg_warm          first-order LP, warm-started from the previous pass (x, y)
    pdhg_cold          first-order LP from zero every pass
    ipm                interior point (an interior, "central" optimal point)
    simplex            vertex solution, cold each pass
    simplex_step       vertex solution with SLP step bounds on the pool flows (trust region,
                       shrunk when the quality error grows): the standard industrial damping,
                       and the fair control for any claim about first-order engines
Quality update: damped successive substitution q <- q + damping (q_actual - q).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..model import ModelBuilder, Problem


@dataclass
class Pool:
    name: str
    inputs: list                 # [(col_name, quality)]
    outputs: list                # [(col_name, row_name, spec)]


@dataclass
class RecursionResult:
    method: str
    converged: bool
    passes: list = field(default_factory=list)
    q: dict = field(default_factory=dict)
    solution: object = None      # qenivo.api.Solution of the final pass

    def summary(self) -> str:
        last = self.passes[-1] if self.passes else {}
        obj = last.get("objective")
        return (f"{self.method:13s} {'CONVERGED' if self.converged else 'not converged':13s} "
                f"passes {len(self.passes):3d}  objective {obj if obj is None else f'{obj:.8g}'}  "
                f"max|dq| {last.get('max_dq', float('nan')):.2e}  time {sum(p['time'] for p in self.passes):.2f}s")


def _index(prob):
    rn, cn = prob.names()
    return {r: i for i, r in enumerate(rn)}, {c: j for j, c in enumerate(cn)}


def _set_quality(prob, pools, q, ri, ci):
    A = prob.A.tolil()
    for pl in pools:
        for col, row, spec in pl.outputs:
            A[ri[row], ci[col]] = q[pl.name] - spec
    prob.A = A.tocsr()
    prob.A.sort_indices()


def _actual_quality(x, pools, ci, q_prev):
    out = {}
    for pl in pools:
        flows = np.array([max(x[ci[c]], 0.0) for c, _ in pl.inputs])
        quals = np.array([qq for _, qq in pl.inputs])
        tot = flows.sum()
        out[pl.name] = float(quals @ flows / tot) if tot > 1e-9 else q_prev[pl.name]
    return out


def run_recursion(prob: Problem, pools: list[Pool], method: str = "pdhg_warm", q0: dict | float | None = None,
                  tol: float = 1e-8, damping: float = 0.5, q_tol: float = 1e-7, max_passes: int = 200,
                  backend: str = "numpy", step_frac: float = 0.5, time_limit: float = 600.0,
                  verbose: bool = False) -> RecursionResult:
    from ..api import solve
    prob = prob.copy()
    ri, ci = _index(prob)
    if q0 is None:
        q = {pl.name: float(np.mean([qq for _, qq in pl.inputs])) for pl in pools}
    elif isinstance(q0, dict):
        q = dict(q0)
    else:
        q = {pl.name: float(q0) for pl in pools}
    pool_cols = sorted({ci[c] for pl in pools for c, _ in pl.inputs} | {ci[c] for pl in pools for c, _, _ in pl.outputs})
    lx0, ux0 = prob.lx.copy(), prob.ux.copy()
    step = None
    prev_dq = np.inf
    warm = None
    res = RecursionResult(method=method, converged=False)
    t_start = time.perf_counter()
    for k in range(max_passes):
        _set_quality(prob, pools, q, ri, ci)
        if method == "simplex_step" and warm is not None and warm.x is not None:
            xp_ = warm.x[pool_cols]
            if step is None:
                step = step_frac * np.maximum(np.abs(xp_), 1.0 + 0.1 * np.abs(xp_).max())
            prob.lx[pool_cols] = np.maximum(lx0[pool_cols], xp_ - step)
            prob.ux[pool_cols] = np.minimum(ux0[pool_cols], xp_ + step)
        engine = {"pdhg_warm": "pdhg", "pdhg_cold": "pdhg", "ipm": "ipm",
                  "simplex": "simplex", "simplex_step": "simplex"}[method]
        t0 = time.perf_counter()
        sol = solve(prob, engine=engine, tol=tol, backend=backend, time_limit=time_limit,
                    warm=warm if method == "pdhg_warm" else None, polish=False)
        widen = 0
        while method == "simplex_step" and step is not None and sol.verdict != "optimal" and widen < 6:
            # trust region cut off every feasible point at the new quality: widen and retry
            step = 4.0 * step
            xp_ = warm.x[pool_cols]
            prob.lx[pool_cols] = np.maximum(lx0[pool_cols], xp_ - step)
            prob.ux[pool_cols] = np.minimum(ux0[pool_cols], xp_ + step)
            sol = solve(prob, engine=engine, tol=tol, backend=backend, time_limit=time_limit, polish=False)
            widen += 1
        dt = time.perf_counter() - t0
        if sol.x is None:
            res.passes.append({"pass": k, "time": dt, "status": sol.verdict, "objective": None, "max_dq": np.nan})
            break
        qa = _actual_quality(sol.x, pools, ci, q)
        dq = max(abs(qa[n] - q[n]) for n in q)
        res.passes.append({"pass": k, "time": dt, "status": sol.verdict, "objective": sol.objective,
                           "iterations": sol.engine.get("iterations"), "max_dq": dq, "q": dict(q)})
        if verbose:
            print(f"  pass {k:2d} {method:12s} {dt:7.3f}s obj {sol.objective:.8g} max|dq| {dq:.3e}")
        res.solution = sol
        if dq < q_tol and sol.verdict == "optimal":
            # the step-bounded variant must also be off its trust-region bounds to count
            if method != "simplex_step" or not _at_step_bound(sol.x, prob, pool_cols, lx0, ux0):
                res.converged = True
                break
        if method == "simplex_step" and step is not None and dq > prev_dq:
            step = 0.5 * step
        prev_dq = dq
        q = {n: q[n] + damping * (qa[n] - q[n]) for n in q}
        warm = sol
        if time.perf_counter() - t_start > time_limit:
            break
    res.q = q
    return res


def _at_step_bound(x, prob, cols, lx0, ux0):
    lo_active = (np.abs(x[cols] - prob.lx[cols]) <= 1e-7 * (1 + np.abs(x[cols]))) & (prob.lx[cols] > lx0[cols])
    hi_active = (np.abs(x[cols] - prob.ux[cols]) <= 1e-7 * (1 + np.abs(x[cols]))) & (prob.ux[cols] < ux0[cols])
    return bool(np.any(lo_active | hi_active))


# ---------------------------------------------------------------------- certified global bound
def _implied_upper(prob: Problem, j: int) -> float:
    """Upper bound on column j from its own bound and from rows with nonnegative data."""
    ub = prob.ux[j]
    A = prob.A.tocsc()
    Acsr = prob.A
    for k in range(A.indptr[j], A.indptr[j + 1]):
        i, a = A.indices[k], A.data[k]
        if a <= 0 or not np.isfinite(prob.uc[i]):
            continue
        s, e = Acsr.indptr[i], Acsr.indptr[i + 1]
        cols, vals = Acsr.indices[s:e], Acsr.data[s:e]
        if np.all(vals >= 0) and np.all(prob.lx[cols] >= 0):
            ub = min(ub, prob.uc[i] / a)
    return float(ub)


def pooling_bound(prob: Problem, pools: list[Pool], tol: float = 1e-9):
    """McCormick relaxation of the pooling bilinear terms, solved and certified.

    For each pool: quality q in [min input quality, max input quality]; for each pool output p_k
    with 0 <= p_k <= U_k, a new variable w_k stands for q * p_k and the specification row uses
    w_k - spec * p_k. Quality is conserved: sum_i qual_i f_i = sum_k w_k. The McCormick
    envelopes
        w >= qL p,   w >= qU p + U q - U qU,   w <= qU p,   w <= qL p + U q - U qL
    make the problem an LP whose optimum bounds every feasible pooled plan (an upper bound for
    a maximisation, a lower bound for a minimisation). Returns (bound, Solution).
    """
    from ..api import solve
    rn, cn = prob.names()
    ri, ci = {r: i for i, r in enumerate(rn)}, {c: j for j, c in enumerate(cn)}
    A = prob.A.tolil()
    extra_cols, extra_rows = [], []            # (name, lo, hi), (name, {col: coef}, lo, hi)
    spec_adds = []                             # (existing row index, new column, coefficient)
    for pl in pools:
        quals = [qq for _, qq in pl.inputs]
        qL, qU = min(quals), max(quals)
        qn = f"q[{pl.name}]"
        extra_cols.append((qn, qL, qU))
        bal = {c: qq for c, qq in pl.inputs}
        for col, row, spec in pl.outputs:
            j, i = ci[col], ri[row]
            U = _implied_upper(prob, j)
            if not np.isfinite(U):
                raise ValueError(f"pool output {col} has no finite upper bound; McCormick needs one")
            wn = f"w[{pl.name},{col}]"
            extra_cols.append((wn, qL * 0.0 if qL >= 0 else qL * U, qU * U))
            A[i, j] = -spec                      # spec row: w - spec * p   (w carries q * p)
            spec_adds.append((i, wn, 1.0))
            bal[wn] = bal.get(wn, 0.0) - 1.0
            extra_rows.append((f"mc1[{wn}]", {wn: 1.0, col: -qL}, 0.0, np.inf))
            extra_rows.append((f"mc2[{wn}]", {wn: 1.0, col: -qU, qn: -U}, -U * qU, np.inf))
            extra_rows.append((f"mc3[{wn}]", {wn: 1.0, col: -qU}, -np.inf, 0.0))
            extra_rows.append((f"mc4[{wn}]", {wn: 1.0, col: -qL, qn: -U}, -np.inf, -U * qL))
        extra_rows.append((f"quality_balance[{pl.name}]", bal, 0.0, 0.0))
    m0 = prob.m
    names = cn + [e[0] for e in extra_cols]
    idx = {c: j for j, c in enumerate(names)}
    A = A.tocsr()
    import scipy.sparse as sps
    A = sps.hstack([A, sps.csr_matrix((m0, len(extra_cols)))], format="lil")
    new_rows, lcs, ucs, rnames = [], [], [], []
    for i, c, v in spec_adds:
        A[i, idx[c]] = A[i, idx[c]] + v
    for name, coefs, lo, hi in extra_rows:
        r = np.zeros(len(names))
        for c, v in coefs.items():
            r[idx[c]] += v
        new_rows.append(r); lcs.append(lo); ucs.append(hi); rnames.append(name)
    A = sps.vstack([A.tocsr(), sps.csr_matrix(np.array(new_rows))], format="csr")
    relax = Problem(c=np.concatenate([prob.c, np.zeros(len(extra_cols))]), A=A,
                    lc=np.concatenate([prob.lc, lcs]), uc=np.concatenate([prob.uc, ucs]),
                    lx=np.concatenate([prob.lx, [e[1] for e in extra_cols]]),
                    ux=np.concatenate([prob.ux, [e[2] for e in extra_cols]]),
                    c0=prob.c0, obj_sign=prob.obj_sign, name=f"{prob.name}_mccormick",
                    row_names=rn + rnames, col_names=names)
    sol = solve(relax, tol=tol)
    return (sol.objective if sol.verdict == "optimal" else None), sol


def certified_recursion(prob: Problem, pools: list[Pool], starts=(None,), method: str = "simplex",
                        **kw) -> dict:
    """Multi-start recursion plus the certified McCormick bound: best plan, bound and gap."""
    bound, bsol = pooling_bound(prob, pools)
    runs = [run_recursion(prob, pools, method=method, q0=q0, **kw) for q0 in starts]
    ok = [r for r in runs if r.solution is not None and r.solution.verdict == "optimal"]
    best = None
    if ok:
        pick = max if prob.obj_sign < 0 else min
        best = pick(ok, key=lambda r: r.solution.objective)
    val = None if best is None else best.solution.objective
    gap = None
    if val is not None and bound is not None:
        gap = abs(bound - val) / max(1.0, abs(bound))
    return {"best_objective": val, "bound": bound, "gap": gap, "bound_certified": bsol.verdict == "optimal",
            "runs": runs, "best": best, "bound_solution": bsol}


# ---------------------------------------------------------------------- Haverly pooling
HAVERLY_OPTIMA = {1: 400.0, 2: 600.0, 3: 750.0}


def haverly(instance: int = 1):
    """Haverly (1978) pooling problems in P-formulation with a placeholder pool sulphur.

    Crudes A (3% S, cost 6) and B (1% S, cost 16 in instances 1-2, 13 in 3) enter one pool;
    crude C (2% S, cost 10) goes direct. Product X (price 9, S <= 2.5%, demand <= 100, or
    600 in instance 2); product Y (price 15, S <= 1.5%, demand <= 200). Maximise profit.
    Published global optima: 400 / 600 / 750.
    """
    cost_b = 13.0 if instance == 3 else 16.0
    dx = 600.0 if instance == 2 else 100.0
    b = ModelBuilder(f"haverly{instance}", maximize=True)
    b.var("A", obj=-6.0); b.var("B", obj=-cost_b)
    b.var("P_X", obj=9.0); b.var("P_Y", obj=15.0)
    b.var("C_X", obj=9.0 - 10.0); b.var("C_Y", obj=15.0 - 10.0)
    b.row("pool_balance", {"A": 1.0, "B": 1.0, "P_X": -1.0, "P_Y": -1.0}, 0.0, 0.0)
    b.row("demand_X", {"P_X": 1.0, "C_X": 1.0}, hi=dx)
    b.row("demand_Y", {"P_Y": 1.0, "C_Y": 1.0}, hi=200.0)
    b.row("sulphur_X", {"P_X": 0.0, "C_X": 2.0 - 2.5}, hi=0.0)
    b.row("sulphur_Y", {"P_Y": 0.0, "C_Y": 2.0 - 1.5}, hi=0.0)
    prob = b.build()
    pools = [Pool("pool", inputs=[("A", 3.0), ("B", 1.0)],
                  outputs=[("P_X", "sulphur_X", 2.5), ("P_Y", "sulphur_Y", 1.5)])]
    return prob, pools


def refinery_pools(prob: Problem) -> list[Pool]:
    """Pools of the pooled refinery generator (models.refinery, pooled=True): one naphtha pool
    per (site, period) block, octane specifications on premium and regular motor fuel."""
    from ..models import williams as W
    meta = prob.meta.get("pool")
    if meta is None:
        raise ValueError("model has no pooling layout (build it with refinery_lp(..., pooled=True))")
    rn, cn = prob.names()
    pools = []
    nblk = len(meta["rows"]["PMF"])
    oct_ = {"LN_POOL": W.OCTANE["LN"], "MN_POOL": W.OCTANE["MN"], "HN_POOL": W.OCTANE["HN"]}
    for b_ in range(nblk):
        ins = [(cn[meta["cols"][nm][b_]], oct_[nm]) for nm in ("LN_POOL", "MN_POOL", "HN_POOL")]
        outs = [(cn[meta["cols"]["POOL_" + p][b_]], rn[meta["rows"][p][b_]], W.OCT_MIN[p]) for p in ("PMF", "RMF")]
        pools.append(Pool(f"pool{b_}", ins, outs))
    return pools
