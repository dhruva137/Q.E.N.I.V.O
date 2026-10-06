"""Industrial model library: one generator per sector named in PS 26119.

Each generator builds a documented, seeded, scalable instance in the structure of a standard
open-literature formulation, so the engines can be exercised on every sector the problem
statement lists. Data are synthetic (no plant data is published), sized by a scale argument.

  sector (PS wording)        generator                   class   formulation source
  power system dispatch      economic_dispatch           QP      quadratic fuel cost, ramping, reserve (Wood & Wollenberg, ch. 3)
  power system dispatch      unit_commitment             MILP    on/off, start-up cost, min up time (Carrion & Arroyo 2006, compact form)
  transportation             transportation              LP      Hitchcock transportation problem
  supply chain management    facility_location           MILP    capacitated facility location (Cornuejols, Sridharan & Thizy 1991)
  production planning        lot_sizing                  MILP    multi-item capacitated lot sizing with setups (Pochet & Wolsey 2006)
  crude blending             crude_blending              LP      crude slate blend to sulphur / API / TAN specs
  refinery scheduling        crude_scheduling            MILP    discrete-time crude unloading, tank inventory and CDU charging
                                                                 (after Lee, Pinto, Grossmann & Park 1996, linear variant)
  logistics                  multi_commodity_flow        LP      multi-commodity network flow with shared arc capacity
The refinery planning LP itself lives in models/refinery.py and Williams' model in williams.py.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import scipy.sparse as sp

from ..model import ModelBuilder, Problem


def economic_dispatch(units=10, periods=24, seed=0) -> Problem:
    """Multi-period economic dispatch, a convex QP: min sum a P^2 + b P subject to demand,
    spinning reserve, unit limits and ramp limits."""
    rng = np.random.default_rng(seed)
    pmin = rng.uniform(50, 150, units)
    pmax = pmin + rng.uniform(150, 450, units)
    a = rng.uniform(0.001, 0.01, units)
    b = rng.uniform(10, 40, units)
    ramp = rng.uniform(0.2, 0.5, units) * (pmax - pmin)
    load = 0.55 * pmax.sum() * (1 + 0.25 * np.sin(np.linspace(0, 2 * np.pi, periods, endpoint=False)))
    reserve = 0.08 * load
    mb = ModelBuilder("economic_dispatch")
    for t in range(periods):
        for g in range(units):
            mb.var(f"P_{g}_{t}", pmin[g], pmax[g], b[g])
            mb.var(f"R_{g}_{t}", 0.0, pmax[g] - pmin[g], 0.0)
            mb.row(f"cap_{g}_{t}", {f"P_{g}_{t}": 1, f"R_{g}_{t}": 1}, hi=pmax[g])
        mb.row(f"demand_{t}", {f"P_{g}_{t}": 1 for g in range(units)}, load[t], load[t])
        mb.row(f"reserve_{t}", {f"R_{g}_{t}": 1 for g in range(units)}, lo=reserve[t])
        if t:
            for g in range(units):
                mb.row(f"ramp_{g}_{t}", {f"P_{g}_{t}": 1, f"P_{g}_{t-1}": -1}, -ramp[g], ramp[g])
    p = mb.build()
    q = np.zeros(p.n)
    for t in range(periods):
        for g in range(units):
            q[mb.col[f"P_{g}_{t}"]] = 2 * a[g]                  # 0.5 x'Qx = a P^2
    return dataclasses.replace(p, Q=sp.diags(q).tocsr(), notes=["convex QP: quadratic fuel costs"])


def unit_commitment(units=10, periods=24, seed=0) -> Problem:
    """Unit commitment MILP: binary on/off u, start-up v, min up time, piecewise-free linear cost."""
    rng = np.random.default_rng(seed)
    pmin = rng.uniform(50, 150, units)
    pmax = pmin + rng.uniform(150, 450, units)
    cost = rng.uniform(15, 45, units)
    noload = rng.uniform(100, 600, units)
    start = rng.uniform(500, 3000, units)
    minup = rng.integers(2, 5, units)
    load = 0.55 * pmax.sum() * (1 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, periods, endpoint=False)))
    mb = ModelBuilder("unit_commitment")
    for g in range(units):
        for t in range(periods):
            mb.var(f"P_{g}_{t}", 0.0, pmax[g], cost[g])
            mb.var(f"u_{g}_{t}", 0, 1, noload[g], integer=True)
            mb.var(f"v_{g}_{t}", 0, 1, start[g], integer=True)
            mb.row(f"max_{g}_{t}", {f"P_{g}_{t}": 1, f"u_{g}_{t}": -pmax[g]}, hi=0.0)
            mb.row(f"min_{g}_{t}", {f"P_{g}_{t}": 1, f"u_{g}_{t}": -pmin[g]}, lo=0.0)
            prev = {f"u_{g}_{t-1}": 1} if t else {}
            mb.row(f"start_{g}_{t}", {f"v_{g}_{t}": 1, f"u_{g}_{t}": -1, **prev}, lo=0.0)
            window = {f"v_{g}_{s}": 1 for s in range(max(0, t - int(minup[g]) + 1), t + 1)}
            mb.row(f"minup_{g}_{t}", {**window, f"u_{g}_{t}": -1}, hi=0.0)
    for t in range(periods):
        mb.row(f"demand_{t}", {f"P_{g}_{t}": 1 for g in range(units)}, load[t], load[t])
    return mb.build()


def transportation(sources=50, sinks=200, seed=0) -> Problem:
    """Transportation LP: ship from supply points to demand points at least distance cost."""
    rng = np.random.default_rng(seed)
    supply = rng.uniform(100, 500, sources)
    demand = rng.uniform(10, 100, sinks)
    demand *= 0.9 * supply.sum() / demand.sum()
    xs, ys = rng.random((sources, 2)), rng.random((sinks, 2))
    cost = np.linalg.norm(xs[:, None, :] - ys[None, :, :], axis=2) * 100
    mb = ModelBuilder("transportation")
    for i in range(sources):
        for j in range(sinks):
            mb.var(f"x_{i}_{j}", 0.0, np.inf, cost[i, j])
    for i in range(sources):
        mb.row(f"supply_{i}", {f"x_{i}_{j}": 1 for j in range(sinks)}, hi=supply[i])
    for j in range(sinks):
        mb.row(f"demand_{j}", {f"x_{i}_{j}": 1 for i in range(sources)}, lo=demand[j])
    return mb.build()


def facility_location(facilities=20, customers=80, seed=0) -> Problem:
    """Capacitated facility location MILP with the strong linking constraints x_ij <= d_j y_i."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(5, 35, customers)
    cap = rng.uniform(10, 160, facilities)
    cap *= 3.0 * d.sum() / cap.sum()
    fixed = rng.uniform(300, 700, facilities) * np.sqrt(cap / cap.mean())
    xs, ys = rng.random((facilities, 2)), rng.random((customers, 2))
    unit = np.linalg.norm(xs[:, None, :] - ys[None, :, :], axis=2) * 10
    mb = ModelBuilder("facility_location")
    for i in range(facilities):
        mb.var(f"y_{i}", 0, 1, fixed[i], integer=True)
        for j in range(customers):
            mb.var(f"x_{i}_{j}", 0.0, d[j], unit[i, j])
            mb.row(f"link_{i}_{j}", {f"x_{i}_{j}": 1, f"y_{i}": -d[j]}, hi=0.0)
        mb.row(f"cap_{i}", {**{f"x_{i}_{j}": 1 for j in range(customers)}, f"y_{i}": -cap[i]}, hi=0.0)
    for j in range(customers):
        mb.row(f"serve_{j}", {f"x_{i}_{j}": 1 for i in range(facilities)}, d[j], d[j])
    return mb.build()


def lot_sizing(items=6, periods=12, seed=0) -> Problem:
    """Multi-item capacitated lot sizing: production x, stock s, setup y (binary)."""
    rng = np.random.default_rng(seed)
    dem = rng.uniform(20, 120, (items, periods))
    setup = rng.uniform(200, 900, items)
    hold = rng.uniform(1, 5, items)
    unit_time = rng.uniform(0.5, 1.5, items)
    cap = 1.1 * float((unit_time[:, None] * dem).sum(axis=0).max())   # lot-for-lot always fits
    mb = ModelBuilder("lot_sizing")
    for i in range(items):
        big = dem[i].sum()
        for t in range(periods):
            mb.var(f"x_{i}_{t}", 0.0, big, 0.0)
            mb.var(f"s_{i}_{t}", 0.0, np.inf, hold[i])
            mb.var(f"y_{i}_{t}", 0, 1, setup[i], integer=True)
            prev = {f"s_{i}_{t-1}": 1} if t else {}
            mb.row(f"bal_{i}_{t}", {**prev, f"x_{i}_{t}": 1, f"s_{i}_{t}": -1}, dem[i, t], dem[i, t])
            mb.row(f"setup_{i}_{t}", {f"x_{i}_{t}": 1, f"y_{i}_{t}": -min(big, dem[i, t:].sum())}, hi=0.0)
    for t in range(periods):
        mb.row(f"cap_{t}", {f"x_{i}_{t}": unit_time[i] for i in range(items)}, hi=cap)
    return mb.build()


def crude_blending(crudes=12, seed=0) -> Problem:
    """Blend a crude slate for the distillation unit at minimum cost, within the unit's limits on
    sulphur, API gravity and total acid number, with availability limits per crude."""
    rng = np.random.default_rng(seed)
    price = rng.uniform(68, 90, crudes)
    sulphur = rng.uniform(0.1, 3.5, crudes)
    api = rng.uniform(20, 45, crudes)
    tan = rng.uniform(0.05, 1.2, crudes)
    avail = rng.uniform(15, 60, crudes)
    need = 0.6 * avail.sum()
    mb = ModelBuilder("crude_blending")
    for k in range(crudes):
        mb.var(f"c_{k}", 0.0, avail[k], price[k])
    all_ = {f"c_{k}": 1 for k in range(crudes)}
    mb.row("throughput", all_, need, need)
    w = avail / avail.sum()                     # specs at the slate average: a proportional blend meets them
    s_max, api_min, tan_max = float(w @ sulphur) + 0.1, float(w @ api) - 1.0, float(w @ tan) + 0.05
    mb.row("sulphur", {f"c_{k}": sulphur[k] - s_max for k in range(crudes)}, hi=0.0)
    mb.row("api_min", {f"c_{k}": api[k] - api_min for k in range(crudes)}, lo=0.0)
    mb.row("tan", {f"c_{k}": tan[k] - tan_max for k in range(crudes)}, hi=0.0)
    return mb.build()


def crude_scheduling(tanks=4, crudes=3, periods=12, seed=0) -> Problem:
    """Discrete-time crude unloading and charging: vessels unload crude into tanks, one crude per
    tank at a time (binary), tanks feed the CDU at a fixed rate, and the CDU charge must meet
    blended sulphur limits. Costs: tank changeovers and inventory."""
    rng = np.random.default_rng(seed)
    arrivals = np.zeros((crudes, periods))
    for k in range(crudes):
        arrivals[k, rng.choice(periods, 2, replace=False)] = rng.uniform(250, 450, 2)
    sulphur = rng.uniform(0.3, 2.8, crudes)
    cap = 600.0
    opening = rng.uniform(100, 300, (crudes, tanks)) * (rng.random((crudes, tanks)) < 1.0 / crudes)
    rate = (arrivals.sum() + opening.sum()) / periods
    spec = float((arrivals.sum(axis=1) + opening.sum(axis=1)) @ sulphur / (arrivals.sum() + opening.sum())) + 0.3
    mb = ModelBuilder("crude_scheduling")
    for t in range(periods):
        for j in range(tanks):
            for k in range(crudes):
                mb.var(f"in_{k}_{j}_{t}", 0.0, cap, 0.0)             # unloading into tank
                mb.var(f"inv_{k}_{j}_{t}", 0.0, cap, 0.2)            # inventory by crude
                mb.var(f"out_{k}_{j}_{t}", 0.0, rate, 0.0)           # charge to the CDU
                mb.var(f"a_{k}_{j}_{t}", 0, 1, 5.0, integer=True)     # tank j holds crude k
                mb.row(f"hold_{k}_{j}_{t}", {f"inv_{k}_{j}_{t}": 1, f"a_{k}_{j}_{t}": -cap}, hi=0.0)
                prev = {f"inv_{k}_{j}_{t-1}": 1} if t else {}
                rhs = -opening[k, j] if t == 0 else 0.0          # opening stock enters period 0
                mb.row(f"bal_{k}_{j}_{t}", {**prev, f"in_{k}_{j}_{t}": 1, f"out_{k}_{j}_{t}": -1,
                                            f"inv_{k}_{j}_{t}": -1}, rhs, rhs)
            mb.row(f"one_{j}_{t}", {f"a_{k}_{j}_{t}": 1 for k in range(crudes)}, hi=1.0)
        for k in range(crudes):
            mb.row(f"arrive_{k}_{t}", {f"in_{k}_{j}_{t}": 1 for j in range(tanks)},
                   arrivals[k, t], arrivals[k, t])
        charge = {f"out_{k}_{j}_{t}": 1 for k in range(crudes) for j in range(tanks)}
        mb.row(f"cdu_{t}", charge, 0.9 * rate, rate)
        mb.row(f"sulphur_{t}", {f"out_{k}_{j}_{t}": sulphur[k] - spec for k in range(crudes) for j in range(tanks)},
               hi=0.0)
    return mb.build()


def multi_commodity_flow(nodes=30, commodities=8, seed=0) -> Problem:
    """Multi-commodity network flow LP: route several origin-destination demands over shared arcs."""
    rng = np.random.default_rng(seed)
    arcs = set()
    for i in range(nodes):                                       # a ring plus random chords
        arcs.add((i, (i + 1) % nodes))
        arcs.add(((i + 1) % nodes, i))
    for _ in range(3 * nodes):
        i, j = rng.integers(0, nodes, 2)
        if i != j:
            arcs.add((int(i), int(j)))
    arcs = sorted(arcs)
    cost = {a: rng.uniform(1, 10) for a in arcs}
    capa = {a: rng.uniform(40, 120) for a in arcs}
    od = [(int(s), int(t), rng.uniform(10, 40)) for s, t in (rng.choice(nodes, 2, replace=False) for _ in range(commodities))]
    mb = ModelBuilder("multi_commodity_flow")
    for k in range(commodities):
        for (i, j) in arcs:
            mb.var(f"f_{k}_{i}_{j}", 0.0, np.inf, cost[(i, j)])
    for k, (s, t, q) in enumerate(od):
        for v in range(nodes):
            coefs = {}
            for (i, j) in arcs:
                if i == v:
                    coefs[f"f_{k}_{i}_{j}"] = coefs.get(f"f_{k}_{i}_{j}", 0) + 1
                if j == v:
                    coefs[f"f_{k}_{i}_{j}"] = coefs.get(f"f_{k}_{i}_{j}", 0) - 1
            rhs = q if v == s else -q if v == t else 0.0
            mb.row(f"flow_{k}_{v}", coefs, rhs, rhs)
    for (i, j) in arcs:
        mb.row(f"cap_{i}_{j}", {f"f_{k}_{i}_{j}": 1 for k in range(commodities)}, hi=capa[(i, j)])
    return mb.build()


# Re-exported here, after the generators above, because these modules import from this one.
from .industry_ext import hydro_thermal, multi_echelon, petrochemical, rcpsp, sced, vrptw  # noqa: E402
from .industry_ext.crude_unloading import crude_unloading  # noqa: E402
from .refinery_full.planner import refinery_delta, refinery_pims, refinery_pool_relax  # noqa: E402

LIBRARY = {
    "economic_dispatch": economic_dispatch, "unit_commitment": unit_commitment,
    "transportation": transportation, "facility_location": facility_location,
    "lot_sizing": lot_sizing, "crude_blending": crude_blending,
    "crude_scheduling": crude_scheduling, "multi_commodity_flow": multi_commodity_flow,
    "crude_unloading": crude_unloading,
    "refinery_pims": refinery_pims, "refinery_delta": refinery_delta,
    "refinery_pool_relax": refinery_pool_relax, "petrochemical": petrochemical,
    "sced": sced, "hydro_thermal": hydro_thermal, "vrptw": vrptw,
    "multi_echelon": multi_echelon, "rcpsp": rcpsp,
}
