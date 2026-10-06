"""Two-stage inventory LP: plant, warehouse, demand.

Finite-horizon deterministic form of the multi-echelon inventory balance
(Clark & Scarf, Management Science 6, 1960). The network LP is the one in
Shapiro, Modeling the Supply Chain, Duxbury, 2001, chapter 4: production and
shipping costs, holding at the plant and at the warehouse, no backorders.

The textbook instance has two periods, plant capacity 4, demands 3 then 5.
``scale`` repeats that horizon (``periods * scale``). ``seed`` is recorded and,
when the horizon is longer than two periods, does not change the demand pattern.
"""
from __future__ import annotations

from ...model import ModelBuilder

PROD_COST = 4.0
SHIP_COST = 1.0
PLANT_HOLD = 2.0
WAREHOUSE_HOLD = 1.0
PLANT_CAP = 4.0
# Textbook demands. Longer horizons repeat the pair (3, 5), which fits the capacity.
_DEMAND_PAIR = (3.0, 5.0)
_STOCK_CAP = 20.0


def multi_echelon(scale=1, seed=0, periods=2):
    """Plant and warehouse inventory over ``periods * scale`` periods.

    Scale 1 with the default two periods is the hand-checked instance. Production
    in a period cannot exceed 4. Demand is served from the warehouse only.
    """
    if int(scale) < 1 or int(periods) < 1:
        raise ValueError("scale and periods must be positive integers")
    horizon = int(periods) * int(scale)
    seed = int(seed)
    demand = [_DEMAND_PAIR[t % 2] for t in range(horizon)]
    mb = ModelBuilder("multi_echelon")
    for t in range(horizon):
        mb.var(f"produce_t{t}", 0.0, PLANT_CAP, PROD_COST)
        mb.var(f"ship_t{t}", 0.0, _STOCK_CAP, SHIP_COST)
        mb.var(f"plant_inv_t{t}", 0.0, _STOCK_CAP, PLANT_HOLD)
        mb.var(f"wh_inv_t{t}", 0.0, _STOCK_CAP, WAREHOUSE_HOLD)
        plant = {f"plant_inv_t{t}": 1.0, f"produce_t{t}": -1.0, f"ship_t{t}": 1.0}
        warehouse = {f"wh_inv_t{t}": 1.0, f"ship_t{t}": -1.0}
        if t:
            plant[f"plant_inv_t{t-1}"] = -1.0
            warehouse[f"wh_inv_t{t-1}"] = -1.0
        mb.row(f"plant_t{t}", plant, 0.0, 0.0)
        mb.row(f"warehouse_t{t}", warehouse, -demand[t], -demand[t])
    problem = mb.build()
    problem.meta = {
        "seed": seed,
        "periods": horizon,
        "demand": demand,
        "plant_capacity": PLANT_CAP,
        "cost": {
            "produce": PROD_COST, "ship": SHIP_COST,
            "plant_hold": PLANT_HOLD, "warehouse_hold": WAREHOUSE_HOLD,
        },
    }
    problem.notes.append("two-stage (plant, warehouse) inventory LP")
    return problem
