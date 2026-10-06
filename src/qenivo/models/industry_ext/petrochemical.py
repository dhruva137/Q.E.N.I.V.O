"""Fixed-yield petrochemical process-network LP.

Two feeds (ethane, naphtha) and two crackers with constant mass yields. Ethylene and
propylene have minimum demands; fuel gas is a free byproduct. The pattern is the
linear yield network used for petrochemical planning (Al-Sharrah, Alatiqi, Elkamel
& Alper, Industrial & Engineering Chemistry Research 40, 2001) and the process-network
LPs in Biegler, Grossmann & Westerberg, Systematic Methods of Chemical Process Design,
Prentice Hall, 1997, chapters 5-7.

``scale`` repeats the textbook period. When ``scale`` > 1, ``seed`` prices an extra
gas-oil feed strictly above the ethane route, so that feed stays idle. Synthetic
data. The scale=1 optimum is the hand value in ``tests/test_e5_models.py``.
"""
from __future__ import annotations

import numpy as np

from ...model import ModelBuilder

ETHANE_COST = 6.0
NAPHTHA_COST = 10.0
FEED_CAP = 10.0
ETHYLENE_DEMAND = 3.0
PROPYLENE_DEMAND = 1.0
# Mass yields. Ethane -> 1/2 ethylene + 1/2 fuel. Naphtha -> 1/2 ethylene + 1/2 propylene.
_HALF = 0.5


def petrochemical(scale=1, seed=0):
    """Multi-period fixed-yield cracker network. ``scale`` periods, integer ``seed``.

    Scale 1 is one period of the textbook network. Reactor capacity is the feed
    upper bound.
    """
    if int(scale) < 1:
        raise ValueError("scale must be a positive integer")
    periods = int(scale)
    seed = int(seed)
    gasoil_cost = None
    if periods > 1:
        gasoil_cost = 30.0 + float(int(np.random.default_rng(seed).integers(0, 5)))
    mb = ModelBuilder("petrochemical")
    for t in range(periods):
        mb.var(f"ethane_t{t}", 0.0, FEED_CAP, ETHANE_COST)
        mb.var(f"naphtha_t{t}", 0.0, FEED_CAP, NAPHTHA_COST)
        streams = ["eth_c2", "eth_fuel", "naph_c2", "naph_c3", "c2", "c3", "fuel"]
        if gasoil_cost is not None:
            mb.var(f"gasoil_t{t}", 0.0, FEED_CAP, gasoil_cost)
            streams.append("gasoil_fuel")
        for stream in streams:
            mb.var(f"{stream}_t{t}", 0.0, 2.0 * FEED_CAP, 0.0)
        mb.row(f"ethane_c2_t{t}", {f"eth_c2_t{t}": 1.0, f"ethane_t{t}": -_HALF}, 0.0, 0.0)
        mb.row(f"ethane_fuel_t{t}", {f"eth_fuel_t{t}": 1.0, f"ethane_t{t}": -_HALF}, 0.0, 0.0)
        mb.row(f"naphtha_c2_t{t}", {f"naph_c2_t{t}": 1.0, f"naphtha_t{t}": -_HALF}, 0.0, 0.0)
        mb.row(f"naphtha_c3_t{t}", {f"naph_c3_t{t}": 1.0, f"naphtha_t{t}": -_HALF}, 0.0, 0.0)
        c2 = {f"c2_t{t}": 1.0, f"eth_c2_t{t}": -1.0, f"naph_c2_t{t}": -1.0}
        fuel = {f"fuel_t{t}": 1.0, f"eth_fuel_t{t}": -1.0}
        if gasoil_cost is not None:
            c2[f"gasoil_t{t}"] = -_HALF
            mb.row(f"gasoil_fuel_t{t}",
                   {f"gasoil_fuel_t{t}": 1.0, f"gasoil_t{t}": -_HALF}, 0.0, 0.0)
            fuel[f"gasoil_fuel_t{t}"] = -1.0
        mb.row(f"ethylene_t{t}", c2, 0.0, 0.0)
        mb.row(f"propylene_t{t}", {f"c3_t{t}": 1.0, f"naph_c3_t{t}": -1.0}, 0.0, 0.0)
        mb.row(f"fuel_t{t}", fuel, 0.0, 0.0)
        mb.row(f"ethylene_demand_t{t}", {f"c2_t{t}": 1.0}, lo=ETHYLENE_DEMAND)
        mb.row(f"propylene_demand_t{t}", {f"c3_t{t}": 1.0}, lo=PROPYLENE_DEMAND)
    problem = mb.build()
    problem.meta = {
        "seed": seed,
        "periods": periods,
        "yields": {
            "ethane": {"ethylene": _HALF, "fuel": _HALF},
            "naphtha": {"ethylene": _HALF, "propylene": _HALF},
        },
        "demands": {"ethylene": ETHYLENE_DEMAND, "propylene": PROPYLENE_DEMAND},
        "feed_cost": {"ethane": ETHANE_COST, "naphtha": NAPHTHA_COST, "gasoil": gasoil_cost},
    }
    problem.notes.append("fixed-yield petrochemical process network (LP)")
    return problem
