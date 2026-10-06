"""Tiny refinery planning LPs with optima computed by hand.

Data are synthetic. Every argument is an int because ``qenivo model`` casts parameters
with ``int()``. The checked call is the default (one crude, one period, seed 0).
``scale`` adds extra independent periods on top of ``periods``.
"""
from __future__ import annotations

import numpy as np

from ...model import ModelBuilder, Problem

# Shared plant for the linear and delta-base models. Yields are dyadic so the
# material balances are exact in float64.
CDU_CAP = 8.0
CRUDE_AVAIL = 100.0
YIELD_N = 0.25
YIELD_D = 0.5
YIELD_R = 0.25
CRUDE_COST = 10.0
EXTRA_CRUDE_COST = 1000.0
HT_CAP = 8.0
PRICE_GAS = 20.0
PRICE_DIESEL = 4.0
PRICE_FUEL = 2.0
GAS_MIN = 1.0
# Distillate sulphur is the crude's sulphur number. The hydrotreater returns the
# same volume and halves that sulphur, so treated blendstock has sulphur 1.
SULPHUR_DIST = 2.0
SULPHUR_TREATED = 1.0
SULPHUR_SPEC = 0.5
# Delta-base index: base + delta_naphtha * barrels + delta_treated * barrels.
DELTA_BASE = 0.25
DELTA_NAPH = -0.125
DELTA_TREATED = 0.25
DELTA_SPEC = 0.5

EXPECTED: dict[str, float] = {}


def _periods(periods: int, scale: int) -> int:
    horizon = int(periods) + int(scale)
    if horizon < 1:
        raise ValueError("periods must be at least 1")
    return horizon


def _linear_plant(kind: str, crudes: int, periods: int, seed: int) -> Problem:
    """One CDU, one hydrotreater, one gasoline pool, copied independently each period.

    Crude 0 costs CRUDE_COST. Any further crude costs EXTRA_CRUDE_COST, so the checked
    one-crude instance never buys it. Streams:
      naphtha  = 0.25 * crude, sulphur 0, gasoline blendstock only
      distillate = 0.50 * crude, sulphur SULPHUR_DIST
      residue  = 0.25 * crude, sold as fuel oil
    The hydrotreater converts a chosen amount of distillate (up to HT_CAP) one barrel
    out per barrel in and halves its sulphur. Treated distillate is gasoline blendstock.
    Untreated distillate is diesel. Gasoline demand is at least GAS_MIN.
    """
    crudes = int(crudes)
    if crudes < 1:
        raise ValueError("crudes must be at least 1")
    name = "refinery_pims" if kind == "pims" else "refinery_delta"
    mb = ModelBuilder(name)
    for t in range(periods):
        crude_names = []
        for k in range(crudes):
            cname = f"crude_c{k}_t{t}"
            crude_names.append(cname)
            mb.var(cname, 0.0, CRUDE_AVAIL, CRUDE_COST if k == 0 else EXTRA_CRUDE_COST)
        naph, dist, resid = f"naph_t{t}", f"dist_t{t}", f"resid_t{t}"
        ht, gas_n, gas_h = f"ht_t{t}", f"gas_n_t{t}", f"gas_h_t{t}"
        gas, diesel, fuel = f"gas_t{t}", f"diesel_t{t}", f"fuel_t{t}"
        mb.var(naph, 0.0, np.inf, 0.0)
        mb.var(dist, 0.0, np.inf, 0.0)
        mb.var(resid, 0.0, np.inf, 0.0)
        mb.var(ht, 0.0, HT_CAP, 0.0)
        mb.var(gas_n, 0.0, np.inf, 0.0)
        mb.var(gas_h, 0.0, np.inf, 0.0)
        mb.var(gas, 0.0, np.inf, -PRICE_GAS)
        mb.var(diesel, 0.0, np.inf, -PRICE_DIESEL)
        mb.var(fuel, 0.0, np.inf, -PRICE_FUEL)
        slate = {cname: 1.0 for cname in crude_names}
        mb.row(f"cdu_t{t}", slate, hi=CDU_CAP)
        mb.row(f"yield_n_t{t}", {**{c: YIELD_N for c in crude_names}, naph: -1.0}, 0.0, 0.0)
        mb.row(f"yield_d_t{t}", {**{c: YIELD_D for c in crude_names}, dist: -1.0}, 0.0, 0.0)
        mb.row(f"yield_r_t{t}", {**{c: YIELD_R for c in crude_names}, resid: -1.0}, 0.0, 0.0)
        # Hydrotreater feed plus diesel (untreated distillate) exhausts the cut.
        mb.row(f"split_t{t}", {ht: 1.0, diesel: 1.0, dist: -1.0}, 0.0, 0.0)
        mb.row(f"naph_bal_t{t}", {gas_n: 1.0, naph: -1.0}, 0.0, 0.0)
        mb.row(f"ht_bal_t{t}", {gas_h: 1.0, ht: -1.0}, 0.0, 0.0)
        mb.row(f"fuel_bal_t{t}", {fuel: 1.0, resid: -1.0}, 0.0, 0.0)
        mb.row(f"gas_vol_t{t}", {gas: 1.0, gas_n: -1.0, gas_h: -1.0}, 0.0, 0.0)
        mb.row(f"gas_dem_t{t}", {gas: 1.0}, lo=GAS_MIN)
        if kind == "pims":
            # Volume-weighted sulphur: SULPHUR_TREATED * gas_h <= SULPHUR_SPEC * gas
            # (naphtha sulphur is 0 and untreated distillate is not in the pool).
            mb.row(f"sulphur_t{t}", {gas_h: SULPHUR_TREATED, gas: -SULPHUR_SPEC}, hi=0.0)
        else:
            # Delta-base index, linear in the blendstock barrels, not divided by volume:
            # DELTA_BASE + DELTA_NAPH * gas_n + DELTA_TREATED * gas_h <= DELTA_SPEC.
            mb.row(
                f"delta_t{t}",
                {gas_n: DELTA_NAPH, gas_h: DELTA_TREATED},
                hi=DELTA_SPEC - DELTA_BASE,
            )
    prob = mb.build()
    prob.meta["seed"] = int(seed)
    prob.meta["crudes"] = crudes
    prob.meta["periods"] = periods
    prob.meta["sulphur_distillate"] = SULPHUR_DIST
    prob.notes.append("synthetic slate; objective is crude cost minus product revenue")
    return prob


def refinery_pims(crudes: int = 1, periods: int = 1, seed: int = 0, scale: int = 0) -> Problem:
    """Linear PIMS-style planning LP. Minimise crude cost minus gasoline, diesel and fuel revenue.

    Fixed yields, one sulphur number on the distillate cut, a CDU capacity, one hydrotreater
    that converts a fraction of that cut and halves its sulphur, and a gasoline pool with a
    maximum sulphur specification. Product demand is a minimum on gasoline. Each period is an
    independent copy of the plant (no inventory). The default is the tiny checked instance.
    """
    return _linear_plant("pims", crudes, _periods(periods, scale), seed)


# 8 barrels of crude at the CDU cap. Yields 0.25 / 0.50 / 0.25 give 2 naphtha, 4 distillate, 2 residue.
# The hydrotreater treats 2 distillate (sulphur 2 -> 1). Gasoline is 2 naphtha + 2 treated = 4.
# The sulphur spec binds: 1 * 2 = 0.5 * 4. The other 2 distillate barrels are diesel; residue is fuel.
# Each extra barrel earns 0.5*20 + 0.25*4 + 0.25*2 = 11.5 and costs 10, so the CDU stays full.
# Gasoline minimum 1 is slack. Hydrotreater capacity 8 is slack.
# cost 10*8 = 80; revenue 20*4 + 4*2 + 2*2 = 92; objective 80 - 92 = -12.
EXPECTED["refinery_pims"] = -12.0


def refinery_delta(seed: int = 0, crudes: int = 1, periods: int = 1, scale: int = 0) -> Problem:
    """Same plant as :func:`refinery_pims`, with a delta-base quality row instead of a weighted spec.

    The gasoline sulphur index is a base property plus a small delta per extra barrel of
    naphtha and of treated distillate. The row is linear, so the model stays an LP.
    ``scale`` adds independent periods. The checked instance is one period and one crude.
    """
    return _linear_plant("delta", crudes, _periods(periods, scale), seed)


# Index S = 0.25 - 0.125*naphtha + 0.25*treated <= 0.5.
# While every distillate barrel can enter gasoline, 0.5*C <= 1 + 0.125*C, i.e. C <= 8/3.
# Below that kink an extra barrel earns 20*0.75 + 2*0.25 = 15.5 and costs 10.
# Above it, treated barrels are 1 + 0.125*C, diesel is 0.375*C - 1, and an extra barrel
# earns 20*0.375 + 4*0.375 + 2*0.25 = 9.5, which is less than the crude cost 10.
# Optimum C = 8/3: naphtha 2/3, treated 4/3, gasoline 2, diesel 0, residue 2/3.
# Index = 0.25 - 0.125*(2/3) + 0.25*(4/3) = 0.5, so the quality row binds.
# CDU cap 8, hydrotreater cap 8 and gasoline minimum 1 are slack.
# cost 10*(8/3) = 80/3; revenue 20*2 + 2*(2/3) = 124/3; objective (80 - 124)/3 = -44/3.
EXPECTED["refinery_delta"] = -44.0 / 3.0


def refinery_pool_relax(seed: int = 0, scale: int = 0) -> Problem:
    """McCormick LP relaxation of one bilinear pooling product. Not a global pooling solve.

    A gasoline pool has sour fraction ``frac`` and volume ``vol``, both in [0, 1].
    Sour barrels are the product ``sour = frac * vol``. A downstream requirement asks for
    at least 1/2 a sour barrel. The synthetic cost is ``frac + vol`` (one per unit of sour
    fraction and one per barrel of pool). ``scale`` is accepted and does not change the
    tiny box, so the checked instance stays the one computed below.

    The nonconvex constraint is not enforced. ``problem.meta["bilinear"]`` stores the
    product, and the returned Problem is the four-inequality McCormick envelope on that
    box (Al-Khayyal and Falk), plus the sour-barrel minimum:

        sour >= 0
        sour >= frac + vol - 1
        sour <= frac
        sour <= vol
        sour >= 1/2

    EXPECTED is the optimum of this LP. It is not a claim about the nonconvex global
    optimum. At the LP solution frac = vol = 1/2 the product is 1/4, which misses
    frac*vol >= 1/2. The bilinear problem costs sqrt(2), at frac = vol = sqrt(1/2).
    """
    _ = int(scale)
    mb = ModelBuilder("refinery_pool_relax")
    mb.var("frac", 0.0, 1.0, 1.0)
    mb.var("vol", 0.0, 1.0, 1.0)
    mb.var("sour", 0.0, 1.0, 0.0)  # lower bound is the envelope sour >= 0
    mb.row("mc_under", {"sour": 1.0, "frac": -1.0, "vol": -1.0}, lo=-1.0)
    mb.row("mc_frac", {"sour": 1.0, "frac": -1.0}, hi=0.0)
    mb.row("mc_vol", {"sour": 1.0, "vol": -1.0}, hi=0.0)
    mb.row("sour_min", {"sour": 1.0}, lo=0.5)
    prob = mb.build()
    prob.meta["seed"] = int(seed)
    prob.meta["bilinear"] = [
        {
            "w": "sour",
            "x": "frac",
            "y": "vol",
            "coef": 1.0,
            "x_bounds": (0.0, 1.0),
            "y_bounds": (0.0, 1.0),
        }
    ]
    prob.notes.append(
        "McCormick LP of sour = frac*vol; optimum is the relaxation value, not the bilinear global optimum"
    )
    return prob


# Envelope: sour <= frac, sour <= vol, sour >= frac + vol - 1, sour >= 0, and sour >= 1/2.
# sour <= frac and sour <= vol force frac >= 1/2 and vol >= 1/2.
# frac = vol = 1/2, sour = 1/2 meets sour >= frac + vol - 1 = 0.
# Cost 1*frac + 1*vol = 1. This is the relaxation optimum only.
EXPECTED["refinery_pool_relax"] = 1.0
