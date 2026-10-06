"""Vehicle routing with time windows as a binary-arc MILP.

Four customers, two identical vehicles, capacity 2 (demand 1 each). The formulation is
the arc-flow model of Desrochers, Desrosiers & Solomon, Operations Research 40, 1992:
binary arcs, flow conservation, a big-M travel-time recursion, and a route-load inequality. The problem
statement is Solomon's, Operations Research 35, 1987. Travel times are the costs.
Synthetic distances, not a copied Solomon instance.

``scale=1`` is this 4-customer instance. Larger ``scale`` adds disjoint customer pairs
and one vehicle per pair. ``seed`` only lengthens the travel time on arcs that touch
those extra customers.
"""
from __future__ import annotations

import numpy as np

from ...model import ModelBuilder

# Node 0 is the depot. Textbook costs are also the travel times.
VRPTW_COST = (
    (0, 1, 1, 10, 10),
    (1, 0, 1, 5, 5),
    (1, 1, 0, 5, 5),
    (10, 5, 5, 0, 1),
    (10, 5, 5, 1, 0),
)
# (release, deadline). Depot departs at time 0.
VRPTW_WINDOW = (
    (0, 0),
    (0, 20),
    (0, 20),
    (5, 40),
    (5, 40),
)
VRPTW_DEMAND = (0, 1, 1, 1, 1)
VRPTW_CAPACITY = 2
VRPTW_VEHICLES = 2
_BIG_M = 60.0


def vrptw(scale=1, seed=0):
    """4-customer VRPTW at ``scale=1``; each extra unit of scale adds a customer pair.

    Arguments are integers. Capacity is the total demand a vehicle may carry.
    """
    if int(scale) < 1:
        raise ValueError("scale must be a positive integer")
    scale = int(scale)
    seed = int(seed)
    cost, window, demand, n_veh = _instance(scale, seed)
    n = len(demand)
    mb = ModelBuilder("vrptw")
    for k in range(n_veh):
        for i in range(n):
            for j in range(n):
                if i != j:
                    mb.var(f"x_{k}_{i}_{j}", 0, 1, float(cost[i][j]), integer=True)
        for i in range(n):
            lo, hi = window[i]
            mb.var(f"T_{k}_{i}", float(lo), float(hi), 0.0)
    for h in range(1, n):
        visit = {}
        for k in range(n_veh):
            for i in range(n):
                if i != h:
                    visit[f"x_{k}_{i}_{h}"] = 1.0
        mb.row(f"visit_{h}", visit, 1.0, 1.0)
    for k in range(n_veh):
        leave = {f"x_{k}_0_{j}": 1.0 for j in range(1, n)}
        back = {f"x_{k}_{i}_0": 1.0 for i in range(1, n)}
        mb.row(f"leave_{k}", leave, hi=1.0)
        mb.row(f"return_{k}", {**back, **{f"x_{k}_0_{j}": -1.0 for j in range(1, n)}}, 0.0, 0.0)
        for h in range(1, n):
            flow = {}
            for i in range(n):
                if i == h:
                    continue
                flow[f"x_{k}_{i}_{h}"] = 1.0
                flow[f"x_{k}_{h}_{i}"] = -1.0
            mb.row(f"flow_{k}_{h}", flow, 0.0, 0.0)
        cap = {}
        for h in range(1, n):
            for i in range(n):
                if i != h:
                    cap[f"x_{k}_{i}_{h}"] = float(demand[h])
        mb.row(f"capacity_{k}", cap, hi=float(VRPTW_CAPACITY))
        for i in range(n):
            for j in range(1, n):
                if i == j:
                    continue
                # T_j >= T_i + travel_ij when the arc is used.
                mb.row(
                    f"time_{k}_{i}_{j}",
                    {f"T_{k}_{j}": 1.0, f"T_{k}_{i}": -1.0, f"x_{k}_{i}_{j}": -_BIG_M},
                    lo=float(cost[i][j]) - _BIG_M,
                )
    # Identical vehicles: put customer 1 on vehicle 0. This does not change the optimum.
    mb.row("symmetry", {f"x_0_{i}_1": 1.0 for i in range(n) if i != 1}, 1.0, 1.0)
    # Four unit demands and capacity 2 need both textbook vehicles on the road.
    if n_veh >= 2:
        both = {f"x_{k}_0_{j}": 1.0 for k in range(2) for j in range(1, 5)}
        mb.row("both_vehicles", both, lo=2.0)
    problem = mb.build()
    problem.meta = {
        "seed": seed,
        "vehicles": n_veh,
        "capacity": VRPTW_CAPACITY,
        "demand": list(demand),
        "window": [list(w) for w in window],
        "cost": [list(row) for row in cost],
    }
    problem.notes.append("VRPTW arc-flow MILP (Desrochers, Desrosiers & Solomon 1992)")
    return problem


def _instance(scale, seed):
    """Textbook 5-node instance, plus ``scale - 1`` disjoint pairs."""
    cost = [list(row) for row in VRPTW_COST]
    window = list(VRPTW_WINDOW)
    demand = list(VRPTW_DEMAND)
    n_veh = VRPTW_VEHICLES
    if scale == 1:
        return cost, window, demand, n_veh
    bump = 15 + int(np.random.default_rng(seed).integers(0, 3))
    for _ in range(scale - 1):
        a = len(demand)
        b = a + 1
        n_old = len(demand)
        for row in cost:
            row.append(bump)
            row.append(bump)
        cost.append([bump] * n_old + [0, 1])
        cost.append([bump] * n_old + [1, 0])
        cost[0][a] = 1
        cost[0][b] = 1
        cost[a][0] = 1
        cost[b][0] = 1
        window.append((0, 20))
        window.append((0, 20))
        demand.append(1)
        demand.append(1)
        n_veh += 1
    return cost, window, demand, n_veh
