"""Security-constrained economic dispatch and a one-reservoir hydro-thermal LP.

SCED is the DC load-flow dispatch of Wood, Wollenberg & Sheblé, Power Generation,
Operation, and Control, 3rd ed., Wiley, 2014, chapter 4, with the linear network
model reviewed by Stott, Jardim & Alsaç, IEEE Transactions on Power Systems 24, 2009.
Costs are linear, so the model is an LP. The textbook case is 3 buses.

Hydro-thermal coordination is the deterministic single-reservoir balance in the same
Wood & Wollenberg text, chapter 7. The multi-reservoir decomposition of Pereira &
Pinto, Water Resources Research 21, 1985, has this LP as its inner problem. One hydro
reservoir, one thermal unit, productivity 1 (power equals discharge).
"""
from __future__ import annotations

from ...model import ModelBuilder

# --- 3-bus textbook SCED -------------------------------------------------------
_GEN_COST = (10.0, 20.0)
_GEN_CAP = 10.0
_LOAD = 6.0
_LINE_CAP = 10.0
_TIGHT_LINE = 2.0          # |f_13| <= 2 is the security limit that binds
_ANGLE = 30.0

# --- hydro-thermal textbook period --------------------------------------------
_V0 = 2.0
_INFLOW = 1.0
_V_END = 2.0
_V_MAX = 20.0
_Q_MAX = 5.0
_LOAD_H = 4.0
_P_MAX = 10.0
_THERMAL_COST = 10.0


def security_constrained_dispatch(scale=1, seed=0):
    """DC security-constrained economic dispatch. ``scale=1`` is the 3-bus LP.

    Buses 1 and 2 have generators (costs 10 and 20). Bus 3 has a load of 6. Lines
    have susceptance 1. The flow on line 1-3 is limited to 2, which forces the
    expensive generator to serve the load. ``scale`` > 1 adds radial buses off bus 3,
    each with a load of 1 and a generator costed from ``seed`` above either textbook unit.
    """
    if int(scale) < 1:
        raise ValueError("scale must be a positive integer")
    return _sced(2 + int(scale), int(seed))


def _sced(n_bus, seed):
    """3-bus textbook core plus radial buses 4..n_bus, each balancing its own load."""
    extra_cost = 100.0 + float(seed % 7)
    mb = ModelBuilder("security_constrained_dispatch")
    mb.var("P_1", 0.0, _GEN_CAP, _GEN_COST[0])
    mb.var("P_2", 0.0, _GEN_CAP, _GEN_COST[1])
    mb.var("f_1_2", -_LINE_CAP, _LINE_CAP, 0.0)
    mb.var("f_1_3", -_TIGHT_LINE, _TIGHT_LINE, 0.0)
    mb.var("f_2_3", -_LINE_CAP, _LINE_CAP, 0.0)
    mb.var("theta_2", -_ANGLE, _ANGLE, 0.0)
    mb.var("theta_3", -_ANGLE, _ANGLE, 0.0)
    for b in range(4, n_bus + 1):
        mb.var(f"P_{b}", 0.0, _GEN_CAP, extra_cost)
        mb.var(f"f_3_{b}", -_LINE_CAP, _LINE_CAP, 0.0)
        mb.var(f"theta_{b}", -_ANGLE, _ANGLE, 0.0)
    bus3 = {"f_1_3": 1.0, "f_2_3": 1.0}
    for b in range(4, n_bus + 1):
        bus3[f"f_3_{b}"] = -1.0          # export from bus 3 toward bus b
    mb.row("balance_1", {"P_1": 1.0, "f_1_2": -1.0, "f_1_3": -1.0}, 0.0, 0.0)
    mb.row("balance_2", {"P_2": 1.0, "f_1_2": 1.0, "f_2_3": -1.0}, 0.0, 0.0)
    mb.row("balance_3", bus3, _LOAD, _LOAD)
    mb.row("ohm_1_2", {"f_1_2": 1.0, "theta_2": 1.0}, 0.0, 0.0)
    mb.row("ohm_1_3", {"f_1_3": 1.0, "theta_3": 1.0}, 0.0, 0.0)
    mb.row("ohm_2_3", {"f_2_3": 1.0, "theta_2": -1.0, "theta_3": 1.0}, 0.0, 0.0)
    for b in range(4, n_bus + 1):
        mb.row(f"balance_{b}", {f"P_{b}": 1.0, f"f_3_{b}": 1.0}, 1.0, 1.0)
        mb.row(f"ohm_3_{b}", {f"f_3_{b}": 1.0, "theta_3": -1.0, f"theta_{b}": 1.0}, 0.0, 0.0)
    problem = mb.build()
    problem.meta = {"seed": seed, "buses": n_bus, "load_bus3": _LOAD, "susceptance": 1.0}
    if n_bus > 3:
        problem.meta["radial_load"] = 1.0
        problem.meta["radial_cost"] = extra_cost
    problem.notes.append("DC security-constrained economic dispatch (linear costs)")
    return problem


def hydro_thermal(scale=1, seed=0, periods=3):
    """One reservoir and one thermal unit over ``periods * scale`` periods.

    Opening storage 2, inflow 1 per period, terminal storage at least 2, load 4,
    thermal cost 10, hydro productivity 1. Spill is allowed and has no value.
    The scale-1, 3-period instance is the hand-checked LP.
    """
    if int(scale) < 1 or int(periods) < 1:
        raise ValueError("scale and periods must be positive integers")
    horizon = int(periods) * int(scale)
    seed = int(seed)
    # seed shifts inflow by a dyadic amount only on horizons other than the textbook one,
    # and only upward, which cannot make the thermal unit short of capacity.
    inflow = _INFLOW if horizon == 3 else _INFLOW + 0.5 * float(seed % 2)
    mb = ModelBuilder("hydro_thermal")
    for t in range(horizon):
        mb.var(f"thermal_t{t}", 0.0, _P_MAX, _THERMAL_COST)
        mb.var(f"discharge_t{t}", 0.0, _Q_MAX, 0.0)
        mb.var(f"spill_t{t}", 0.0, _V_MAX, 0.0)
        vlo = _V_END if t == horizon - 1 else 0.0
        mb.var(f"storage_t{t}", vlo, _V_MAX, 0.0)
        mb.row(f"load_t{t}", {f"thermal_t{t}": 1.0, f"discharge_t{t}": 1.0}, _LOAD_H, _LOAD_H)
        bal = {f"storage_t{t}": 1.0, f"discharge_t{t}": 1.0, f"spill_t{t}": 1.0}
        if t == 0:
            mb.row(f"water_t{t}", bal, _V0 + inflow, _V0 + inflow)
        else:
            bal[f"storage_t{t-1}"] = -1.0
            mb.row(f"water_t{t}", bal, inflow, inflow)
    problem = mb.build()
    problem.meta = {
        "seed": seed, "periods": horizon, "opening": _V0, "inflow": inflow,
        "terminal_min": _V_END, "load": _LOAD_H, "thermal_cost": _THERMAL_COST,
        "productivity": 1.0,
    }
    problem.notes.append("single-reservoir hydro-thermal LP, productivity 1")
    return problem
