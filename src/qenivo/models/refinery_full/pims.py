"""One-crude PIMS-style refinery planning LP.

The scale=1 model is one period, one crude assay, a CDU capacity, one hydrotreater
with a fixed volumetric yield, and one diesel pool. Three quality modes share the
material balances:

  0  linear volume-weighted sulphur (Pinto, Joly & Moro, Computers & Chemical
     Engineering 24, 2000, the planning-LP structure)
  1  delta-base linearisation around a 50/50 base blend (Rigby, Lasdon & Waren,
     Interfaces 25(5), 1995): base quality plus property deltas times share deviations
  2  bilinear pool sulphur stored in ``Problem.meta``, with the McCormick (1976)
     envelopes as the LP the simplex solves. No NLP solver is called. The pooling
     products are the treated and untreated distillate streams (Haverly, ACM SIGMAP
     Bulletin 25, 1978, is the classical pooling form; this instance is smaller).

Synthetic assay, not plant data. ``scale`` is the number of identical periods;
``seed`` only prices an extra, strictly dominated crude when ``scale`` > 1.
The textbook period (crude cost 20, hydrotreater cost 4, diesel price 10, demand 3)
is the plan used by the hand objective in ``tests/test_e5_models.py``.
"""
from __future__ import annotations

import numpy as np

from ...model import ModelBuilder

# Textbook period. Yields and sulphur levels are dyadic so the LP data is exact in float64.
CRUDE_COST = 20.0
HT_COST = 4.0
DIESEL_PRICE = 10.0
HOLD_COST = 1.0
CDU_CAP = 10.0
CRUDE_AVAIL = 20.0          # purchase limit; the CDU row is what caps throughput at CDU_CAP
HT_CAP = 6.0
DEMAND = 3.0
NAPHTHA_YIELD = 0.5
DISTILLATE_YIELD = 0.5
NAPHTHA_OCTANE = 90.0
NAPHTHA_SULPHUR = 0.0
DISTILLATE_OCTANE = 40.0
DISTILLATE_SULPHUR = 1.0
TREATED_SULPHUR = 0.25
HT_YIELD = 1.0
SULPHUR_MAX = 0.5
# Delta-base reference blend (half treated, half untreated). Shares are dyadic.
_BASE_TREATED = 0.5
_BASE_UNTREATED = 0.5


def refinery_pims(scale=1, seed=0, mode=0):
    """Linear (mode 0), delta-base (mode 1) or McCormick pooling (mode 2) refinery LP.

    Default ``mode=0`` is the volume-weighted sulphur spec. See the module docstring
    for the sources. ``scale`` is the period count; ``seed`` is an integer.
    """
    return _build(scale=scale, seed=seed, mode=mode, name="refinery_pims")


def refinery_pims_delta(scale=1, seed=0):
    """Delta-base diesel sulphur: base quality plus deltas on share deviations. Still an LP.

    Rigby, Lasdon & Waren, Interfaces 25(5), 1995. Same material balance as
    ``refinery_pims``; only the quality row changes. Integer ``scale`` and ``seed``.
    """
    return _build(scale=scale, seed=seed, mode=1, name="refinery_pims_delta")


def refinery_pims_pool(scale=1, seed=0):
    """Bilinear pool sulphur, emitted as its McCormick LP relaxation.

    Coefficients of ``pool_sulphur * diesel = 0.25 treated + untreated`` are stored in
    ``problem.meta['bilinear']``. McCormick, Mathematical Programming 10, 1976.
    Integer ``scale`` and ``seed``. The simplex solves the relaxation, not an NLP.
    """
    return _build(scale=scale, seed=seed, mode=2, name="refinery_pims_pool")


def _build(scale, seed, mode, name):
    if not isinstance(scale, (int, np.integer)) or int(scale) < 1:
        raise ValueError("scale must be a positive integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if mode not in (0, 1, 2):
        raise ValueError("mode must be 0 (linear), 1 (delta-base) or 2 (McCormick)")
    periods = int(scale)
    seed = int(seed)
    # A second crude is available only on multi-period instances, and it is priced
    # strictly above the textbook crude so a one-crude plan stays optimal.
    n_crudes = 1 if periods == 1 else 2
    extra_cost = None
    if n_crudes == 2:
        extra_cost = 40.0 + float(int(np.random.default_rng(seed).integers(0, 5)))
    crude_cost = [CRUDE_COST] + ([] if extra_cost is None else [extra_cost])

    mb = ModelBuilder(name)
    for t in range(periods):
        for k in range(n_crudes):
            mb.var(f"crude_{k}_t{t}", 0.0, CRUDE_AVAIL, crude_cost[k])
        mb.var(f"naphtha_t{t}", 0.0, CDU_CAP, 0.0)
        mb.var(f"ht_feed_t{t}", 0.0, HT_CAP, HT_COST)
        mb.var(f"treated_t{t}", 0.0, HT_CAP, 0.0)
        mb.var(f"treated_dump_t{t}", 0.0, HT_CAP, 0.0)
        mb.var(f"untreated_t{t}", 0.0, CDU_CAP, 0.0)
        mb.var(f"dist_dump_t{t}", 0.0, CDU_CAP, 0.0)
        # Mode 2 puts the pool volume on its demand lower bound so the McCormick
        # bounds below are exactly the variable bounds.
        diesel_lo = DEMAND if mode == 2 else 0.0
        mb.var(f"diesel_t{t}", diesel_lo, CDU_CAP, 0.0)
        mb.var(f"sales_t{t}", DEMAND, CDU_CAP, -DIESEL_PRICE)
        mb.var(f"inv_t{t}", 0.0, CDU_CAP, HOLD_COST)
        if mode == 2:
            mb.var(f"pool_w_t{t}", 0.0, CDU_CAP, 0.0)
            mb.var(f"pool_s_t{t}", TREATED_SULPHUR, SULPHUR_MAX, 0.0)

        naph = {f"naphtha_t{t}": 1.0}
        # Raw distillate splits into hydrotreater feed, untreated blendstock and dump.
        # Treated product is the hydrotreater outlet, not a second copy of the feed.
        dist = {f"ht_feed_t{t}": 1.0, f"untreated_t{t}": 1.0, f"dist_dump_t{t}": 1.0}
        cdu = {}
        for k in range(n_crudes):
            naph[f"crude_{k}_t{t}"] = -NAPHTHA_YIELD
            dist[f"crude_{k}_t{t}"] = -DISTILLATE_YIELD
            cdu[f"crude_{k}_t{t}"] = 1.0
        mb.row(f"cdu_t{t}", cdu, hi=CDU_CAP)
        mb.row(f"naphtha_yield_t{t}", naph, 0.0, 0.0)
        mb.row(f"distillate_t{t}", dist, 0.0, 0.0)
        mb.row(f"ht_yield_t{t}",
               {f"treated_t{t}": 1.0, f"treated_dump_t{t}": 1.0, f"ht_feed_t{t}": -HT_YIELD},
               0.0, 0.0)
        mb.row(f"diesel_blend_t{t}",
               {f"diesel_t{t}": 1.0, f"treated_t{t}": -1.0, f"untreated_t{t}": -1.0},
               0.0, 0.0)
        inv = {f"inv_t{t}": 1.0, f"sales_t{t}": 1.0, f"diesel_t{t}": -1.0}
        if t:
            inv[f"inv_t{t-1}"] = -1.0
        mb.row(f"inventory_t{t}", inv, 0.0, 0.0)
        _quality_row(mb, t, mode)

    problem = mb.build()
    problem.meta = _meta(periods, seed, mode, crude_cost)
    problem.notes.append(
        "PIMS-style planning LP; mode 0 linear sulphur, 1 delta-base, 2 McCormick pooling"
    )
    return problem


def _quality_row(mb, t, mode):
    treated, untreated, diesel = f"treated_t{t}", f"untreated_t{t}", f"diesel_t{t}"
    if mode == 0:
        mb.row(f"sulphur_linear_t{t}", {
            treated: TREATED_SULPHUR,
            untreated: DISTILLATE_SULPHUR,
            diesel: -SULPHUR_MAX,
        }, hi=0.0)
        return
    if mode == 1:
        q_base = _BASE_TREATED * TREATED_SULPHUR + _BASE_UNTREATED * DISTILLATE_SULPHUR
        d_treated = TREATED_SULPHUR - q_base
        d_untreated = DISTILLATE_SULPHUR - q_base
        # q_base * x + d_t (t - s_t x) + d_u (u - s_u x) <= Smax * x
        coef_x = (q_base - d_treated * _BASE_TREATED - d_untreated * _BASE_UNTREATED
                  - SULPHUR_MAX)
        mb.row(f"sulphur_delta_t{t}", {
            treated: d_treated,
            untreated: d_untreated,
            diesel: coef_x,
        }, hi=0.0)
        return
    # w = 0.25 * treated + 1 * untreated, and w envelopes pool_s * diesel.
    mb.row(f"pool_balance_t{t}", {
        f"pool_w_t{t}": 1.0,
        treated: -TREATED_SULPHUR,
        untreated: -DISTILLATE_SULPHUR,
    }, 0.0, 0.0)
    pL, pU = TREATED_SULPHUR, SULPHUR_MAX
    xL, xU = DEMAND, CDU_CAP
    w, p, x = f"pool_w_t{t}", f"pool_s_t{t}", diesel
    mb.row(f"mccormick_under_a_t{t}",
           {w: 1.0, x: -pL, p: -xL}, lo=-(pL * xL))
    mb.row(f"mccormick_under_b_t{t}",
           {w: 1.0, x: -pU, p: -xU}, lo=-(pU * xU))
    mb.row(f"mccormick_over_a_t{t}",
           {w: 1.0, x: -pU, p: -xL}, hi=-(pU * xL))
    mb.row(f"mccormick_over_b_t{t}",
           {w: 1.0, x: -pL, p: -xU}, hi=-(pL * xU))


def _meta(periods, seed, mode, crude_cost):
    meta = {
        "seed": seed,
        "mode": mode,
        "periods": periods,
        "assay": {
            "crudes": [
                {
                    "name": f"crude_{k}",
                    "price": crude_cost[k],
                    "cuts": {
                        "naphtha": {"yield": NAPHTHA_YIELD, "sulphur": NAPHTHA_SULPHUR,
                                    "octane": NAPHTHA_OCTANE},
                        "distillate": {"yield": DISTILLATE_YIELD, "sulphur": DISTILLATE_SULPHUR,
                                       "octane": DISTILLATE_OCTANE},
                    },
                }
                for k in range(len(crude_cost))
            ],
            "hydrotreater": {
                "feed": "distillate", "yield": HT_YIELD,
                "outlet_sulphur": TREATED_SULPHUR, "capacity": HT_CAP, "cost": HT_COST,
            },
            "cdu_capacity": CDU_CAP,
            "product": {"name": "diesel", "demand": DEMAND, "price": DIESEL_PRICE,
                        "sulphur_max": SULPHUR_MAX},
        },
    }
    if mode == 1:
        q_base = _BASE_TREATED * TREATED_SULPHUR + _BASE_UNTREATED * DISTILLATE_SULPHUR
        meta["delta_base"] = {
            "base_shares": {"treated": _BASE_TREATED, "untreated": _BASE_UNTREATED},
            "base_sulphur": q_base,
            "delta_sulphur": {
                "treated": TREATED_SULPHUR - q_base,
                "untreated": DISTILLATE_SULPHUR - q_base,
            },
        }
    if mode == 2:
        meta["bilinear"] = [
            {
                "w": f"pool_w_t{t}",
                "quality": f"pool_s_t{t}",
                "volume": f"diesel_t{t}",
                "equation": "w = quality * volume",
                "feeds": [
                    {"stream": f"treated_t{t}", "quality": TREATED_SULPHUR},
                    {"stream": f"untreated_t{t}", "quality": DISTILLATE_SULPHUR},
                ],
            }
            for t in range(periods)
        ]
        meta["mccormick_bounds"] = {
            "quality": [TREATED_SULPHUR, SULPHUR_MAX],
            "volume": [DEMAND, CDU_CAP],
        }
    return meta
