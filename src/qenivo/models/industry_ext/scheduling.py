"""Renewable-resource project scheduling (RCPSP) as a time-indexed MILP.

The zero-one formulation is Pritsker, Watters & Wolfe, Management Science 16, 1969:
a binary variable for each feasible start period, a makespan, finish-time inequalities,
and one renewable-resource knapsack per period. Precedence is a start-time inequality.
The classification of these constraints is Brucker, Drexl, Möhring, Neumann & Pesch,
European Journal of Operational Research 112, 1999.

The textbook project has four jobs, durations (2, 2, 2, 1), resource uses (1, 1, 2, 1)
and capacity 2. Job 2 follows job 0; job 3 follows job 1. ``scale`` appends unit-time
jobs on the same resource. ``seed`` is stored and does not change the four-job data.
"""
from __future__ import annotations

from ...model import ModelBuilder

_DUR = (2, 2, 2, 1)
_RES = (1, 1, 2, 1)
_CAP = 2
_PRED = {2: (0,), 3: (1,)}
_HORIZON = 8


def rcpsp(scale=1, seed=0):
    """Time-indexed RCPSP. ``scale=1`` is the 4-job textbook project.

    The objective is the makespan. Extra jobs added for ``scale`` > 1 have duration 1,
    resource use 1, and a single predecessor (the previous job).
    """
    if int(scale) < 1:
        raise ValueError("scale must be a positive integer")
    scale = int(scale)
    seed = int(seed)
    dur = list(_DUR)
    res = list(_RES)
    pred = {j: list(ps) for j, ps in _PRED.items()}
    for _ in range(scale - 1):
        j = len(dur)
        dur.append(1)
        res.append(1)
        pred[j] = [j - 1]
    horizon = _HORIZON + 2 * (scale - 1)
    starts = [list(range(horizon - dur[j] + 1)) for j in range(len(dur))]
    mb = ModelBuilder("rcpsp")
    mb.var("makespan", 0.0, float(horizon), 1.0)
    for j, times in enumerate(starts):
        for t in times:
            mb.var(f"start_{j}_{t}", 0, 1, 0.0, integer=True)
        mb.row(f"once_{j}", {f"start_{j}_{t}": 1.0 for t in times}, 1.0, 1.0)
        mb.row(
            f"finish_{j}",
            {"makespan": 1.0, **{f"start_{j}_{t}": -float(t + dur[j]) for t in times}},
            lo=0.0,
        )
    for j, preds in pred.items():
        for i in preds:
            coef = {f"start_{j}_{t}": float(t) for t in starts[j]}
            for t in starts[i]:
                coef[f"start_{i}_{t}"] = coef.get(f"start_{i}_{t}", 0.0) - float(t)
            mb.row(f"prec_{i}_{j}", coef, lo=float(dur[i]))
    for tau in range(horizon):
        coef = {}
        for j, times in enumerate(starts):
            for t in times:
                if t <= tau < t + dur[j]:
                    coef[f"start_{j}_{t}"] = float(res[j])
        if coef:
            mb.row(f"resource_{tau}", coef, hi=float(_CAP))
    problem = mb.build()
    problem.meta = {
        "seed": seed,
        "durations": dur,
        "resources": res,
        "capacity": _CAP,
        "predecessors": {str(j): ps for j, ps in pred.items()},
        "horizon": horizon,
    }
    problem.notes.append("time-indexed RCPSP (Pritsker, Watters & Wolfe 1969)")
    return problem
