"""Realistic case-stack generators for fair warm/cold benchmarking.

Planner what-ifs keep the matrix fixed and change prices, capacities, demand or
availability. `CaseDelta` stores only the changed entries (indices + values) and
`apply` shares the base `A`. Stack kinds range from one-factor sweeps (high basis
reuse) to the old iid cost stack (GPU-campaign distribution).

Column/row groups are rebuilt by replaying `refinery_lp`'s RNG (same order as
research/cpu_lab/caselab.py `refinery_with_meta`). Deltas are index-based; default
names from `Problem.names()` are C0/R0-style when the generator left names unset.

Refs: Williams (refinery block); research note 05_cpu_case_structure (stack kinds).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

import numpy as np

from ..model import Problem
from ..models import refinery as RF
from .cases import Case

_UNIT_NAMES = ("distillation_cap", "reform_cap", "crack_cap")


# -------------------------------------------------------------------------------- groups
def refinery_groups(level: int, seed: int = 0) -> dict:
    """Index groups for `refinery_level(level, seed)`. Replays RNG; asserts cost match."""
    R, T, K, M = RF.LEVELS[level]
    return _groups_from_params(R, T, K, M, seed)


def groups(prob: Problem) -> dict:
    """Column/row index groups for a refinery `Problem` (parse name, replay RNG)."""
    R, T, K, M, seed = _parse_refinery_name(prob.name)
    return _groups_from_params(R, T, K, M, seed, prob=prob)


def _parse_refinery_name(name: str) -> tuple[int, int, int, int, int]:
    m = re.match(r"refinery_R(\d+)_T(\d+)_K(\d+)_M(\d+)_s(\d+)", name or "")
    if not m:
        raise ValueError(f"not a named refinery problem: {name!r}")
    return tuple(int(x) for x in m.groups())  # type: ignore[return-value]


def _groups_from_params(R: int, T: int, K: int, M: int, seed: int,
                        prob: Problem | None = None) -> dict:
    """Replay `refinery_lp` draws in order; return crude/ship/cap/demand/avail indices."""
    if prob is None:
        prob = RF.refinery_lp(R, T, K, M, seed=seed)
    rng = np.random.default_rng(seed)
    P, deg = len(RF.PRODUCTS), min(3, R)
    y = RF.crude_yields(K, rng)
    B, blo, bhi, brow, bcol = RF._block_template(y)
    NR, NB = B.shape
    nb = R * T
    size = rng.uniform(0.5, 1.5, R)
    E = np.array([(r, m) for m in range(M) for r in rng.choice(R, deg, replace=False)])
    nE = len(E)
    avail = rng.uniform(0.6, 1.4, (K, T))
    dmaxu = rng.uniform(0.5, 1.5, (M, P, T))
    dminu = rng.uniform(0.0, 1.0, (M, P, T))
    crude_u = rng.uniform(0.85, 1.15, (K, T))
    price = (np.array([RF.W.PROFIT[p] for p in RF.PRODUCTS]) + 1.0)[None, :, None] * rng.uniform(
        0.9, 1.1, (M, P, T))
    transport = rng.uniform(0.05, 0.3, nE)
    nI, nS = R * T * P, nE * P * T
    oI = nb * NB
    oS, oU = oI + nI, oI + nI + nS
    ee, pp2, tt2 = [a.ravel() for a in np.meshgrid(np.arange(nE), np.arange(P), np.arange(T),
                                                    indexing="ij")]
    sc = oS + (ee * P + pp2) * T + tt2
    ship_price = price[E[ee, 1], pp2, tt2]
    assert np.allclose(prob.c[sc], transport[ee] - ship_price, rtol=0, atol=1e-12), \
        "price replay failed"
    rk, tk, kk = [a.ravel() for a in np.meshgrid(np.arange(R), np.arange(T), np.arange(K),
                                                  indexing="ij")]
    crude_cols = (rk * T + tk) * NB + kk
    oIB = nb * NR
    oM = oIB + R * T * P
    oC = oM + M * P * T
    cap_rows = {u: np.arange(nb) * NR + brow.index(u) for u in _UNIT_NAMES}
    site_of_block = np.repeat(np.arange(R), T)
    dem_p = np.repeat(np.arange(P), T)[None, :].repeat(M, 0).ravel()
    # silence unused RNG-advanced locals (kept for exact replay order)
    _ = (avail, dmaxu, dminu, crude_u, blo, bhi, bcol, size, nS)
    return dict(
        R=R, T=T, K=K, M=M, P=P, NB=NB, NR=NR, nE=nE,
        crude_cols=crude_cols, crude_k=kk, crude_t=tk,
        ship_cols=sc, ship_p=pp2, ship_e=ee, ship_price=ship_price,
        cap_rows=cap_rows, site_of_block=site_of_block,
        dem_rows=np.arange(oM, oC), dem_p=dem_p,
        unmet_cols=np.arange(oU, oU + M * P * T),
        avail_rows=np.arange(oC, oC + K * T),
        avail_k=np.repeat(np.arange(K), T),
        E=E, transport=transport,
    )


# -------------------------------------------------------------------------------- CaseDelta
@dataclass
class CaseDelta:
    """Sparse change of costs / bounds against a base problem (matrix not copied).

    Indices are integer positions. When names exist (or default C0/R0 from
    `Problem.names()`), `to_case` converts to the name-keyed `Case` used by
    `solve_cases`. Unnamed refinery models from `refinery_level` leave names
    unset; prefer index-based deltas for those.
    """
    name: str
    c_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    c_val: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    lx_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    lx_val: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    ux_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    ux_val: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    lc_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    lc_val: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    uc_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    uc_val: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    shock: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    def apply(self, base: Problem) -> Problem:
        """Shallow-copy `base` and overwrite only changed vectors; shares `base.A`."""
        p = copy.copy(base)
        if self.c_idx.size:
            p.c = np.ascontiguousarray(base.c.copy())
            p.c[self.c_idx] = self.c_val
        if self.lx_idx.size:
            p.lx = np.ascontiguousarray(base.lx.copy())
            p.lx[self.lx_idx] = self.lx_val
        if self.ux_idx.size:
            p.ux = np.ascontiguousarray(base.ux.copy())
            p.ux[self.ux_idx] = self.ux_val
        if self.lc_idx.size:
            p.lc = np.ascontiguousarray(base.lc.copy())
            p.lc[self.lc_idx] = self.lc_val
        if self.uc_idx.size:
            p.uc = np.ascontiguousarray(base.uc.copy())
            p.uc[self.uc_idx] = self.uc_val
        p.name = f"{base.name}[{self.name}]"
        return p

    def to_case(self, base: Problem) -> Case:
        """Convert index deltas to a name-keyed `Case` via `base.names()`."""
        rn, cn = base.names()
        cost = {cn[j]: float(v) * base.obj_sign for j, v in zip(self.c_idx, self.c_val)}
        col_lo = {cn[j]: float(v) for j, v in zip(self.lx_idx, self.lx_val)}
        col_hi = {cn[j]: float(v) for j, v in zip(self.ux_idx, self.ux_val)}
        row_lo = {rn[i]: float(v) for i, v in zip(self.lc_idx, self.lc_val)}
        row_hi = {rn[i]: float(v) for i, v in zip(self.uc_idx, self.uc_val)}
        return Case(name=self.name, cost=cost, col_lo=col_lo, col_hi=col_hi,
                    row_lo=row_lo, row_hi=row_hi)

    def n_changed(self) -> int:
        return (self.c_idx.size + self.lx_idx.size + self.ux_idx.size
                + self.lc_idx.size + self.uc_idx.size)


def _delta_c(name: str, idx: np.ndarray, val: np.ndarray, shock: np.ndarray) -> CaseDelta:
    idx = np.asarray(idx, dtype=np.int64).ravel()
    val = np.asarray(val, dtype=np.float64).ravel()
    return CaseDelta(name=name, c_idx=idx, c_val=val, shock=np.asarray(shock, dtype=np.float64).ravel())


def _delta_bounds(name: str, shock: np.ndarray, *,
                  lc_idx=None, lc_val=None, uc_idx=None, uc_val=None,
                  lx_idx=None, lx_val=None, ux_idx=None, ux_val=None,
                  c_idx=None, c_val=None) -> CaseDelta:
    empty_i = np.array([], dtype=np.int64)
    empty_v = np.array([], dtype=np.float64)

    def iv(i, v):
        if i is None:
            return empty_i, empty_v
        return np.asarray(i, dtype=np.int64).ravel(), np.asarray(v, dtype=np.float64).ravel()

    c_i, c_v = iv(c_idx, c_val)
    lx_i, lx_v = iv(lx_idx, lx_val)
    ux_i, ux_v = iv(ux_idx, ux_val)
    lc_i, lc_v = iv(lc_idx, lc_val)
    uc_i, uc_v = iv(uc_idx, uc_val)
    return CaseDelta(name=name, c_idx=c_i, c_val=c_v, lx_idx=lx_i, lx_val=lx_v,
                     ux_idx=ux_i, ux_val=ux_v, lc_idx=lc_i, lc_val=lc_v,
                     uc_idx=uc_i, uc_val=uc_v,
                     shock=np.asarray(shock, dtype=np.float64).ravel())


# -------------------------------------------------------------------------------- stacks
def make_stack(prob: Problem, kind: str, S: int, seed: int = 0, **params) -> list[CaseDelta]:
    """Build `S` sparse case deltas of the given kind. Deterministic in `seed`."""
    if S < 1:
        raise ValueError("S must be >= 1")
    kind = kind.lower()
    if kind == "independent":
        return _stack_independent(prob, S, seed, **params)
    g = groups(prob)
    if kind == "one_crude":
        return _stack_one_crude(prob, g, S, seed, **params)
    if kind == "one_unit":
        return _stack_one_unit(prob, g, S, seed, **params)
    if kind == "factor_price":
        return _stack_factor_price(prob, g, S, seed, **params)
    if kind == "factor_mixed":
        return _stack_factor_mixed(prob, g, S, seed, **params)
    if kind == "cargo_menu":
        return _stack_cargo_menu(prob, g, S, seed, **params)
    raise ValueError(f"unknown stack kind: {kind!r}")


def _stack_independent(prob: Problem, S: int, seed: int, a: float = 0.1, **_) -> list[CaseDelta]:
    """Every cost × U(1-a, 1+a); a=0.1 matches gpu_campaign `_scenarios` distribution."""
    rng = np.random.default_rng(seed)
    f = rng.uniform(1.0 - a, 1.0 + a, (prob.n, S))
    out = []
    for s in range(S):
        c_new = prob.c * f[:, s]
        idx = np.flatnonzero(c_new != prob.c)
        out.append(_delta_c(f"ind_{s}", idx, c_new[idx], f[:, s] - 1.0))
    return out


def _stack_one_crude(prob: Problem, g: dict, S: int, seed: int,
                     crude_k: int = 0, lo: float = -0.2, hi: float = 0.2, **_) -> list[CaseDelta]:
    """Sweep one crude's purchase price over a grid of relative moves [lo, hi]."""
    _ = seed  # grid is deterministic; seed kept for API uniformity
    fracs = np.linspace(lo, hi, S)
    sel = g["crude_cols"][g["crude_k"] == int(crude_k)]
    if sel.size == 0:
        raise ValueError(f"no columns for crude_k={crude_k}")
    out = []
    for s, u in enumerate(fracs):
        val = prob.c[sel] * (1.0 + u)
        out.append(_delta_c(f"crude{crude_k}_{u:+.4f}", sel, val, np.array([u])))
    return out


def _stack_one_unit(prob: Problem, g: dict, S: int, seed: int,
                    unit: str = "distillation_cap", site: int = 0,
                    lo: float = -0.2, hi: float = 0.2, **_) -> list[CaseDelta]:
    """Sweep one unit capacity (all periods at `site`) over a relative grid."""
    _ = seed
    if unit not in g["cap_rows"]:
        raise ValueError(f"unknown unit {unit!r}; choose from {list(g['cap_rows'])}")
    fracs = np.linspace(lo, hi, S)
    rows = g["cap_rows"][unit][g["site_of_block"] == int(site)]
    if rows.size == 0:
        raise ValueError(f"no capacity rows for unit={unit} site={site}")
    out = []
    for s, u in enumerate(fracs):
        val = prob.uc[rows] * (1.0 + u)
        out.append(_delta_bounds(f"{unit}_s{site}_{u:+.4f}", np.array([u]),
                                 uc_idx=rows, uc_val=val))
    return out


def _stack_factor_price(prob: Problem, g: dict, S: int, seed: int,
                       k: int = 3, sigma: float = 0.05, **_) -> list[CaseDelta]:
    """c = c0 + B f with f ~ N(0, sigma^2 I); loadings on crude and product/ship groups."""
    rng = np.random.default_rng(seed)
    k = int(k)
    f = rng.normal(0.0, sigma, (k, S))  # factors × cases
    cc, sc = g["crude_cols"], g["ship_cols"]
    # B columns: 0 Brent→crudes, 1 crack→product revenue in ship cols, 2 freight→ship
    out = []
    for s in range(S):
        c_new = prob.c.copy()
        fs = f[:, s]
        # additive: B f with B[:,0] = c0 on crude cols (relative-scale loading)
        c_new[cc] = prob.c[cc] + prob.c[cc] * fs[0]
        if k >= 2:
            c_new[sc] = prob.c[sc] - g["ship_price"] * fs[1]
        if k >= 3:
            # freight: push transport component of ship cost
            c_new[sc] = c_new[sc] + g["transport"][g["ship_e"]] * fs[2]
        # extra factors (k>3): small iid on crudes
        for j in range(3, k):
            c_new[cc] = c_new[cc] + prob.c[cc] * (fs[j] / max(k - 2, 1))
        idx = np.flatnonzero(c_new != prob.c)
        out.append(_delta_c(f"fprice_{s}", idx, c_new[idx], fs))
    return out


def _stack_factor_mixed(prob: Problem, g: dict, S: int, seed: int,
                       k: int = 3, sigma: float = 0.05, shock: float = 0.05, **_) -> list[CaseDelta]:
    """Factor prices plus capacity and demand shocks."""
    rng = np.random.default_rng(seed)
    price = _stack_factor_price(prob, g, S, seed, k=k, sigma=sigma)
    R, K, P = g["R"], g["K"], g["P"]
    fu = 1.0 + rng.uniform(-shock, shock, (3, R, S))
    gd = rng.uniform(-shock, shock, S)
    dp = gd[None, :] + rng.uniform(-shock / 3, shock / 3, (P, S))
    fk = 1.0 + rng.uniform(-shock, shock, (K, S))
    out = []
    for s in range(S):
        d = price[s]
        # capacity
        uc_idx, uc_val = [], []
        for u, (name, rows) in enumerate(g["cap_rows"].items()):
            r = rows
            scale = fu[u][g["site_of_block"], s]
            uc_idx.append(r)
            uc_val.append(prob.uc[r] * scale)
        # demand
        dr = g["dem_rows"]
        fd = 1.0 + dp[g["dem_p"], s]
        lc_d = prob.lc[dr] * fd
        uc_d = prob.uc[dr] * fd
        # avail
        ar = g["avail_rows"]
        uc_a = prob.uc[ar] * fk[g["avail_k"], s]
        uc_idx.append(ar)
        uc_val.append(uc_a)
        uc_idx.append(dr)
        uc_val.append(uc_d)
        uc_i = np.concatenate(uc_idx)
        uc_v = np.concatenate(uc_val)
        # unmet upper = dmin
        ux_i = g["unmet_cols"]
        ux_v = lc_d.copy()
        shock_v = np.concatenate([d.shock, fu[:, :, s].ravel(), dp[:, s], fk[:, s] - 1.0])
        out.append(_delta_bounds(
            f"fmix_{s}", shock_v,
            c_idx=d.c_idx, c_val=d.c_val,
            lc_idx=dr, lc_val=lc_d,
            uc_idx=uc_i, uc_val=uc_v,
            ux_idx=ux_i, ux_val=ux_v,
        ))
    return out


def _stack_cargo_menu(prob: Problem, g: dict, S: int, seed: int,
                      cargo_k: int = 0, lo: float = -0.2, hi: float = 0.2,
                      scale: float = 1.0, **_) -> list[CaseDelta]:
    """Base slate + one candidate cargo (existing crude columns, optional scale) price sweep."""
    _ = seed
    fracs = np.linspace(lo, hi, S)
    sel = g["crude_cols"][g["crude_k"] == int(cargo_k)]
    if sel.size == 0:
        raise ValueError(f"no columns for cargo_k={cargo_k}")
    # assay scale: treat purchase cost as scaled assay proxy on the same columns
    base_c = prob.c[sel] * float(scale)
    out = []
    for s, u in enumerate(fracs):
        val = base_c * (1.0 + u)
        out.append(_delta_c(f"cargo{cargo_k}_{u:+.4f}", sel, val, np.array([u])))
    return out


def nn_order(deltas_or_shocks) -> list[int]:
    """Greedy nearest-neighbour tour from the case closest to the origin (shock 0)."""
    if len(deltas_or_shocks) == 0:
        return []
    if isinstance(deltas_or_shocks[0], CaseDelta):
        shocks = [d.shock for d in deltas_or_shocks]
        # pad to equal length
        w = max(s.size for s in shocks)
        X = np.zeros((len(shocks), w))
        for i, s in enumerate(shocks):
            X[i, :s.size] = s
    else:
        X = np.asarray(deltas_or_shocks, float)
        if X.ndim == 1:
            X = X[:, None]
    S = len(X)
    left = np.ones(S, dtype=bool)
    cur = int(np.argmin((X ** 2).sum(1)))
    order = [cur]
    left[cur] = False
    for _ in range(S - 1):
        idx = np.flatnonzero(left)
        nxt = int(idx[np.argmin(((X[idx] - X[cur]) ** 2).sum(1))])
        order.append(nxt)
        left[nxt] = False
        cur = nxt
    return order
