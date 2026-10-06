"""Tiny industry models with optima computed by hand.

Petrochemical planning, 3-bus DC economic dispatch, hydro-thermal scheduling, a
vehicle-routing MILP with time windows, a two-demand supply chain, and a small
resource-constrained project schedule. Data are synthetic. Arguments are ints.
``scale`` adds periods or jobs on top of the size argument; the checked call is
the default, which does not use ``scale``.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from ...model import ModelBuilder, Problem

EXPECTED: dict[str, float] = {}


def petrochemical(seed: int = 0, scale: int = 0) -> Problem:
    """Two feeds, one reactor with fixed yields, one product demand. Linear program.

    Feed A costs 3 per unit, yields 2 product, and has capacity 1. Feed B costs 4 and
    yields 1 product. Demand is an equality. ``scale`` adds two product units and the
    feed-B capacity to cover them. Minimise feed cost.
    """
    extra = int(scale)
    demand = 4 + 2 * extra
    mb = ModelBuilder("petrochemical")
    mb.var("feed_a", 0.0, 1.0, 3.0)
    mb.var("feed_b", 0.0, 10.0 + 2.0 * extra, 4.0)
    mb.var("product", 0.0, np.inf, 0.0)
    mb.row("reactor", {"feed_a": 2.0, "feed_b": 1.0, "product": -1.0}, 0.0, 0.0)
    mb.row("demand", {"product": 1.0}, float(demand), float(demand))
    prob = mb.build()
    prob.meta["seed"] = int(seed)
    prob.notes.append("fixed reactor yields; synthetic feed prices")
    return prob


# Feed A is cheaper per product (3/2 = 1.5 versus 4) but its capacity is 1, so it makes 2 product.
# Demand is 4, so feed B makes the other 2. Cost 3*1 + 4*2 = 11. The demand equality binds,
# and feed A's capacity binds.
EXPECTED["petrochemical"] = 11.0


def sced(buses: int = 3, seed: int = 0, scale: int = 0) -> Problem:
    """Three-bus DC economic dispatch with linear costs and one line limit. Linear program.

    Buses 1 and 2 have generators (costs 10 and 30, limits 10 MW). Bus 3 has a 6 MW load.
    Every line reactance is 1. Angle at bus 1 is the reference. The only flow limit is
    3 MW on line 1-3, written on the angle difference. Power balance is B theta = P with
    B_ii = degree and B_ij = -1. Extra buses (``buses`` or ``scale`` above 3) are
    zero-injection buses tied to bus 1, so they do not change the 3-bus cost.
    Angle bounds of +/- 20 are a wide box; the checked optimum sits inside it.
    """
    nbus = int(buses) + int(scale)
    if nbus < 3:
        nbus = 3
    mb = ModelBuilder("sced")
    mb.var("g1", 0.0, 10.0, 10.0)
    mb.var("g2", 0.0, 10.0, 30.0)
    mb.var("th1", 0.0, 0.0, 0.0)
    mb.var("th2", -20.0, 20.0, 0.0)
    mb.var("th3", -20.0, 20.0, 0.0)
    mb.row("bal1", {"g1": 1.0, "th1": -2.0, "th2": 1.0, "th3": 1.0}, 0.0, 0.0)
    mb.row("bal2", {"g2": 1.0, "th1": 1.0, "th2": -2.0, "th3": 1.0}, 0.0, 0.0)
    # (B theta)_3 = -load, rearranged to th1 + th2 - 2 th3 = 6.
    mb.row("bal3", {"th1": 1.0, "th2": 1.0, "th3": -2.0}, 6.0, 6.0)
    mb.row("line13", {"th1": 1.0, "th3": -1.0}, -3.0, 3.0)
    for b in range(4, nbus + 1):
        mb.var(f"th{b}", -20.0, 20.0, 0.0)
        mb.row(f"bal{b}", {f"th{b}": 1.0, "th1": -1.0}, 0.0, 0.0)
    prob = mb.build()
    prob.meta["seed"] = int(seed)
    prob.meta["buses"] = nbus
    prob.notes.append("DC load flow, linear generation cost, one binding line")
    return prob


# With theta1 fixed at 0 the balances give g1 = -6 - 3*theta3 and g2 = 12 + 3*theta3.
# Cost = 10*g1 + 30*g2 = 300 + 60*theta3, so the smallest feasible theta3 is best.
# Line 1-3 limits theta1 - theta3 to [-3, 3], hence theta3 >= -3. That bound is active:
# theta3 = -3, theta2 = 0, g1 = 3, g2 = 3. Both generator limits (10) are slack.
# Cost 10*3 + 30*3 = 120.
EXPECTED["sced"] = 120.0


def hydro_thermal(periods: int = 3, seed: int = 0, scale: int = 0) -> Problem:
    """One reservoir and one thermal unit over a few periods. Linear program.

    Storage starts at 3 energy units. Each period has inflow 2, demand 5, hydro limit 4
    and a thermal unit of cost 5 and limit 10. Storage stays in [0, 4], and the last
    period must end at or above 1. Balance: storage_t = storage_{t-1} + inflow - hydro_t
    (the opening storage enters the first right-hand side). Hydro is free, so the cost
    is thermal energy only. ``scale`` adds periods with the same per-period data; the
    checked instance is three periods.
    """
    horizon = int(periods) + int(scale)
    if horizon < 1:
        raise ValueError("periods must be at least 1")
    start, inflow, demand = 3.0, 2.0, 5.0
    mb = ModelBuilder("hydro_thermal")
    for t in range(horizon):
        mb.var(f"h{t}", 0.0, 4.0, 0.0)
        mb.var(f"g{t}", 0.0, 10.0, 5.0)
        end = t == horizon - 1
        mb.var(f"s{t}", 1.0 if end else 0.0, 4.0, 0.0)
        prev = {f"s{t - 1}": -1.0} if t else {}
        rhs = inflow + (start if t == 0 else 0.0)
        mb.row(f"res{t}", {**prev, f"s{t}": 1.0, f"h{t}": 1.0}, rhs, rhs)
        mb.row(f"dem{t}", {f"h{t}": 1.0, f"g{t}": 1.0}, demand, demand)
    prob = mb.build()
    prob.meta["seed"] = int(seed)
    prob.meta["periods"] = horizon
    prob.notes.append("single reservoir, linear thermal cost")
    return prob


# Three periods, demand 5+5+5 = 15. Opening storage 3 plus inflow 2*3 = 6, and at least 1
# must remain, so hydro energy is at most 8. Thermal energy is then at least 7.
# Hydro 4, 3, 1 leaves storage 1, 0, 1, all inside [0, 4], with thermal 1, 2, 4.
# Cost 5*7 = 35. The terminal storage bound binds; the hydro limit binds only in period 0.
EXPECTED["hydro_thermal"] = 35.0


def _travel(i: int, j: int) -> int:
    """Depot legs cost 2, the pairs (1, 2) and (3, 4) cost 1, every other leg costs 10."""
    if (i, j) in {(1, 2), (2, 1), (3, 4), (4, 3)}:
        return 1
    if i == 0 or j == 0:
        return 2
    return 10


def vrptw(customers: int = 4, seed: int = 0, scale: int = 0) -> Problem:
    """Vehicle routing with time windows, as a MILP on binary arcs.

    Depot 0 and ``customers`` clients (plus ``scale`` extra clients). Each client demands
    1 and the vehicle capacity is 2, so a route has at most two clients. Arc costs and
    travel times are :func:`_travel`. Service time is 0. Departure from the depot is fixed
    at time 0; each client time lies in [0, 20]; a return arrives by time 20.
    Load and time are propagated with Miller-Tucker-Zemlin inequalities. Because there
    are four clients, every subset also gets a rounded capacity cut
    (incoming arcs at least ceil(demand / capacity)). Minimise travel cost.
    The checked instance is four clients. Its feasible tours are enumerated in the test.
    """
    n = int(customers) + int(scale)
    if n < 1:
        raise ValueError("customers must be at least 1")
    nodes = range(n + 1)
    clients = list(range(1, n + 1))
    capacity = 2.0
    window_hi = 20.0
    big_m = 40.0
    service = 0.0
    demand = {j: 1 for j in clients}
    mb = ModelBuilder("vrptw")
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            mb.var(f"x_{i}_{j}", 0, 1, float(_travel(i, j)), integer=True)
    for j in clients:
        mb.var(f"tm{j}", 0.0, window_hi, 0.0)
        mb.var(f"ld{j}", float(demand[j]), capacity, 0.0)
        mb.var(f"arr{j}", 0.0, window_hi, 0.0)
    for j in clients:
        mb.row(f"in_{j}", {f"x_{i}_{j}": 1.0 for i in nodes if i != j}, 1.0, 1.0)
        mb.row(f"out_{j}", {f"x_{j}_{k}": 1.0 for k in nodes if k != j}, 1.0, 1.0)
    mb.row(
        "depot",
        {**{f"x_0_{j}": 1.0 for j in clients}, **{f"x_{j}_0": -1.0 for j in clients}},
        0.0,
        0.0,
    )
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            arc = f"x_{i}_{j}"
            travel = float(_travel(i, j))
            if j != 0 and i != 0:
                mb.row(f"tm_{i}_{j}", {f"tm{j}": 1.0, f"tm{i}": -1.0, arc: -big_m}, lo=travel + service - big_m)
                mb.row(
                    f"ld_{i}_{j}",
                    {f"ld{j}": 1.0, f"ld{i}": -1.0, arc: -capacity},
                    lo=float(demand[j]) - capacity,
                )
            elif j != 0:
                mb.row(f"tm_0_{j}", {f"tm{j}": 1.0, arc: -big_m}, lo=travel + service - big_m)
                mb.row(f"ld_0_{j}", {f"ld{j}": 1.0, arc: -capacity}, lo=float(demand[j]) - capacity)
            else:
                mb.row(f"arr_{i}", {f"arr{i}": 1.0, f"tm{i}": -1.0, arc: -big_m}, lo=travel - big_m)
    for r in range(2, n + 1):
        for subset in combinations(clients, r):
            group = set(subset)
            need = (sum(demand[j] for j in subset) + int(capacity) - 1) // int(capacity)
            entering = {
                f"x_{i}_{j}": 1.0
                for j in subset
                for i in nodes
                if i not in group
            }
            mb.row("cap_" + "_".join(str(j) for j in subset), entering, lo=float(need))
    travel = {(i, j): _travel(i, j) for i in nodes for j in nodes if i != j}
    prob = mb.build()
    prob.meta["seed"] = int(seed)
    prob.meta["vrptw"] = {
        "customers": clients,
        "capacity": int(capacity),
        "demand": demand,
        "travel": travel,
        "window": {i: (0, int(window_hi)) for i in nodes},
        "service": service,
    }
    prob.notes.append("binary arcs, capacity, time windows; synthetic travel costs")
    return prob


# Capacity 2 and demand 1, so a route has at most two clients.
# Pair routes 0-1-2-0 and 0-3-4-0 cost 2+1+2 = 5 each, total 10. The swapped orders cost the same.
# The other pairings use a cross arc of cost 10 and total 28.
# One pair plus two depot singletons (each 2+2 = 4) costs 5+4+4 = 13. Four singletons cost 16.
# Every capacity-feasible route returns by time 14, inside the window [0, 20], so the windows
# do not remove the cost optimum. Two such routes are feasible together, so the optimum is 10.
EXPECTED["vrptw"] = 10.0


def multi_echelon(seed: int = 0, scale: int = 0) -> Problem:
    """One plant, one warehouse, two demand sites. Linear program.

    Production costs 2, the plant-to-warehouse leg costs 1, and the two outbound legs
    cost 3 and 4. Demands are equalities of 4 and 3. Plant and warehouse capacities are
    10. ``scale`` adds extra unit demands shipped at cost 4 and raises both capacities
    by the same amount. Minimise production plus freight.
    """
    extra = int(scale)
    mb = ModelBuilder("multi_echelon")
    mb.var("prod", 0.0, 10.0 + extra, 2.0)
    mb.var("to_wh", 0.0, 10.0 + extra, 1.0)
    mb.var("ship1", 0.0, np.inf, 3.0)
    mb.var("ship2", 0.0, np.inf, 4.0)
    outbound = {"to_wh": 1.0, "ship1": -1.0, "ship2": -1.0}
    for k in range(extra):
        name = f"shipx{k}"
        mb.var(name, 0.0, np.inf, 4.0)
        outbound[name] = -1.0
        mb.row(f"demx{k}", {name: 1.0}, 1.0, 1.0)
    mb.row("plant", {"prod": 1.0, "to_wh": -1.0}, 0.0, 0.0)
    mb.row("warehouse", outbound, 0.0, 0.0)
    mb.row("dem1", {"ship1": 1.0}, 4.0, 4.0)
    mb.row("dem2", {"ship2": 1.0}, 3.0, 3.0)
    prob = mb.build()
    prob.meta["seed"] = int(seed)
    prob.notes.append("plant, one warehouse, two demand equalities")
    return prob


# Demands 4 and 3 must be produced and must pass through the warehouse.
# cost = 7*2 + 7*1 + 4*3 + 3*4 = 14 + 7 + 12 + 12 = 45.
# Plant and warehouse capacities of 10 are slack.
EXPECTED["multi_echelon"] = 45.0


def rcpsp(jobs: int = 4, seed: int = 0, scale: int = 0) -> Problem:
    """Resource-constrained project scheduling MILP, time-indexed, one renewable resource.

    The checked instance has four unit-duration jobs, each using one unit of a resource
    of capacity 1. Precedences are j1 before j2, j1 before j3, and both j2 and j3 before j4.
    A binary ``u_job_t`` selects the start period. The horizon is the number of jobs, which
    is enough for a serial schedule. Makespan is minimised.
    If ``jobs + scale`` is not 4, the jobs form a single chain instead; the test calls the default.
    Feasible orders of the checked instance are listed in the test.
    """
    n = int(jobs) + int(scale)
    if n < 1:
        raise ValueError("jobs must be at least 1")
    names = [f"j{j}" for j in range(1, n + 1)]
    if n == 4:
        prec = [("j1", "j2"), ("j1", "j3"), ("j2", "j4"), ("j3", "j4")]
    else:
        prec = [(f"j{j}", f"j{j + 1}") for j in range(1, n)]
    duration = {name: 1 for name in names}
    usage = {name: 1 for name in names}
    capacity = 1
    mb = ModelBuilder("rcpsp")
    mb.var("makespan", 0.0, np.inf, 1.0)
    for name in names:
        for t in range(n):
            mb.var(f"u_{name}_{t}", 0, 1, 0.0, integer=True)
        mb.row(f"once_{name}", {f"u_{name}_{t}": 1.0 for t in range(n)}, 1.0, 1.0)
        mb.row(
            f"finish_{name}",
            {"makespan": 1.0, **{f"u_{name}_{t}": -float(t + duration[name]) for t in range(n)}},
            lo=0.0,
        )
    for t in range(n):
        mb.row(f"res_{t}", {f"u_{name}_{t}": float(usage[name]) for name in names}, hi=float(capacity))
    for earlier, later in prec:
        coefs = {}
        for t in range(n):
            if t == 0:
                continue
            coefs[f"u_{later}_{t}"] = float(t)
            coefs[f"u_{earlier}_{t}"] = -float(t)
        mb.row(f"prec_{earlier}_{later}", coefs, lo=float(duration[earlier]))
    prob = mb.build()
    prob.meta["seed"] = int(seed)
    prob.meta["rcpsp"] = {
        "jobs": names,
        "duration": duration,
        "resource": usage,
        "capacity": capacity,
        "precedences": [list(pair) for pair in prec],
    }
    prob.notes.append("time-indexed RCPSP, one renewable resource")
    return prob


# Each job takes one period and one unit of resource, and the capacity is 1, so four jobs
# need four periods: makespan >= 4. The precedence chain j1 -> j2 -> j4 is only three
# periods, so the resource is what forces the fourth.
# j1 at 0, j2 at 1, j3 at 2, j4 at 3 finishes at 4 and respects both precedences and the resource.
# j1 at 0, j3 at 1, j2 at 2, j4 at 3 does too. No other order respects the precedences.
EXPECTED["rcpsp"] = 4.0
