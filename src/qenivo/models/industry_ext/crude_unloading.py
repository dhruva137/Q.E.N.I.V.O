"""Discrete-time crude-oil unloading / inventory scheduling MILP (Lee et al. 1996).

Citation
--------
Lee, H., Pinto, J. M., Grossmann, I. E. and Park, S. *Mixed-Integer Linear
Programming Model for Refinery Short-Term Scheduling of Crude Oil Unloading
with Inventory Management.* Industrial & Engineering Chemistry Research
35(5):1630–1641, 1996. doi:10.1021/ie950519h

Topology (MRPL-shaped, after Reddy, Karimi & Srinivasan 2004 / Kelly & Mann 2003)
---------------------------------------------------------------------------------
One single-buoy mooring (SBM) receives VLCC parcels. Parcels unload into storage
tanks, transfer into charging tanks, and feed multiple CDUs. Each storage and
charging tank holds at most one crude at a time (binary assignment). Blended CDU
charge quality (sulphur upper bound, API lower bound) is written as *linear*
component-flow inequalities, the Lee et al. linearisation of the bilinear blend
when tanks are dedicated by crude.

Sets
----
V vessels, I storage tanks, J charging tanks, L CDUs, C crudes, T periods.

Decision variables (period index ``t``)
---------------------------------------
``dock_v_t`` binary          vessel ``v`` occupies the SBM in ``t``
``unload_v_i_t`` continuous  unload flow vessel → storage
``as_i_c_t`` binary          storage ``i`` holds crude ``c``
``invs_i_c_t`` continuous    storage inventory of crude ``c``
``xfer_i_j_c_t`` continuous  transfer storage → charging
``ac_j_c_t`` binary          charging ``j`` holds crude ``c``
``invc_j_c_t`` continuous    charging inventory of crude ``c``
``feed_j_l_c_t`` continuous  charging → CDU feed of crude ``c``
``vinv_v_t`` continuous      remaining vessel cargo

Key constraints
---------------
* SBM exclusivity: ``sum_v dock_v_t ≤ 1``
* Unload only while docked, only after arrival, vessel emptied by horizon end
* Material balances on vessels, storage, charging
* Tank dedication: ``sum_c as ≤ 1``, ``invs ≤ cap * as`` (same for charging)
* CDU throughput bounds and linear sulphur / API blend inequalities

Objective: demurrage on residual vessel cargo + inventory holding on tanks.

Seeded sizes ``small`` / ``medium`` / ``large`` scale vessels, tanks, CDUs and
horizon (medium ≈ 7×8 h days; large ≈ 7×4 h days) while keeping the same
topology.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from ...model import ModelBuilder, Problem

SizeName = Literal["small", "medium", "large"]

# MRPL-shaped generators. Medium/large match the W07 brief (6–8 storage, 3–4
# charging, 3 CDUs, 7-day horizon). Small is a compact solvable instance.
SIZES: dict[str, dict] = {
    "small": dict(
        vessels=2, storage=3, charging=2, cdus=2, crudes=2, periods=6, period_hours=8.0
    ),
    "medium": dict(
        vessels=4, storage=6, charging=3, cdus=3, crudes=3, periods=21, period_hours=8.0
    ),
    "large": dict(
        vessels=6, storage=8, charging=4, cdus=3, crudes=4, periods=42, period_hours=4.0
    ),
}


def _free_ram_gb() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().available) / 1e9
    except Exception:
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return float(stat.ullAvailPhys) / 1e9
        except Exception:
            return float("inf")


def size_ok_to_solve(size: str, min_free_gb: float = 2.5) -> bool:
    """Medium/large solves are blocked when free RAM is below ``min_free_gb``."""
    size = str(size).lower()
    if size == "small":
        return True
    return _free_ram_gb() >= float(min_free_gb)


def _instance_data(size: str, seed: int) -> dict:
    cfg = dict(SIZES[size])
    size_salt = {"small": 0, "medium": 101, "large": 202}[size]
    rng = np.random.default_rng(int(seed) + 17 * size_salt)
    n_v, n_i, _, n_l, n_c, n_t = (
        cfg["vessels"],
        cfg["storage"],
        cfg["charging"],
        cfg["cdus"],
        cfg["crudes"],
        cfg["periods"],
    )
    sulphur = rng.uniform(0.4, 2.5, n_c)
    api = rng.uniform(25.0, 40.0, n_c)
    # One VLCC parcel per vessel; staggered arrivals; each vessel carries one crude.
    crude_of = np.arange(n_v) % n_c
    arrivals = np.zeros(n_v, dtype=int)
    for v in range(n_v):
        arrivals[v] = int(rng.integers(0, max(1, n_t // 3)))
    # Throughput targets: keep demand feasible from opening + parcels.
    cdu_mid = 80.0 if size == "small" else 120.0
    cdu_lo = 0.75 * cdu_mid
    cdu_hi = 1.15 * cdu_mid
    total_charge = n_l * n_t * cdu_mid
    parcel = np.full(n_v, total_charge / max(n_v, 1) * 0.55)
    # Opening inventory covers early periods before vessels finish unloading.
    opening = np.zeros((n_i, n_c))
    for c in range(n_c):
        tanks_for_c = [i for i in range(n_i) if i % n_c == c]
        if not tanks_for_c:
            tanks_for_c = [c % n_i]
        share = (total_charge * 0.55) / n_c / len(tanks_for_c)
        for i in tanks_for_c:
            opening[i, c] = share
    stor_cap = float(opening.max() + parcel.max() + 50.0)
    chg_cap = stor_cap * 0.6
    unload_rate = float(parcel.max() / 2.0 + 1.0)
    xfer_rate = unload_rate
    # Specs: slate-average sulphur + margin, API − margin (Lee-style linear blend).
    mass = opening.sum(axis=0) + np.bincount(crude_of, weights=parcel, minlength=n_c)
    mass = np.maximum(mass, 1e-9)
    s_max = float(mass @ sulphur / mass.sum()) + 0.35
    api_min = float(mass @ api / mass.sum()) - 2.0
    demurrage = 2.5
    hold_s = 0.05
    hold_c = 0.08
    return {
        **cfg,
        "sulphur": sulphur,
        "api": api,
        "crude_of": crude_of,
        "arrivals": arrivals,
        "parcel": parcel,
        "opening": opening,
        "stor_cap": stor_cap,
        "chg_cap": chg_cap,
        "unload_rate": unload_rate,
        "xfer_rate": xfer_rate,
        "cdu_lo": cdu_lo,
        "cdu_hi": cdu_hi,
        "s_max": s_max,
        "api_min": api_min,
        "demurrage": demurrage,
        "hold_s": hold_s,
        "hold_c": hold_c,
        "seed": int(seed),
        "size": size,
    }


def crude_unloading(size: SizeName | str = "small", seed: int = 0) -> Problem:
    """Build a seeded MRPL-shaped Lee et al. (1996) crude-unloading MILP.

    Parameters
    ----------
    size:
        ``"small"``, ``"medium"``, or ``"large"`` (see :data:`SIZES`).
    seed:
        RNG seed for parcel arrivals, qualities and opening inventories.
    """
    size = str(size).lower()
    if size not in SIZES:
        raise ValueError(f"size must be one of {sorted(SIZES)}, got {size!r}")
    data = _instance_data(size, int(seed))
    n_v = data["vessels"]
    n_i = data["storage"]
    n_j = data["charging"]
    n_l = data["cdus"]
    n_c = data["crudes"]
    n_t = data["periods"]
    sulphur = data["sulphur"]
    api = data["api"]
    crude_of = data["crude_of"]
    arrivals = data["arrivals"]
    parcel = data["parcel"]
    opening = data["opening"]
    stor_cap = data["stor_cap"]
    chg_cap = data["chg_cap"]
    unload_rate = data["unload_rate"]
    xfer_rate = data["xfer_rate"]
    cdu_lo = data["cdu_lo"]
    cdu_hi = data["cdu_hi"]
    s_max = data["s_max"]
    api_min = data["api_min"]
    demurrage = data["demurrage"]
    hold_s = data["hold_s"]
    hold_c = data["hold_c"]

    mb = ModelBuilder("crude_unloading")

    for v in range(n_v):
        for t in range(n_t):
            # Demurrage on residual cargo encourages early unloading.
            mb.var(f"vinv_{v}_{t}", 0.0, float(parcel[v]), demurrage)
            # Docking before arrival is fixed at 0 via bounds.
            if t < int(arrivals[v]):
                mb.var(f"dock_{v}_{t}", 0, 0, 0.0, integer=True)
            else:
                mb.var(f"dock_{v}_{t}", 0, 1, 0.0, integer=True)
            for i in range(n_i):
                mb.var(f"unload_{v}_{i}_{t}", 0.0, unload_rate, 0.0)

    for t in range(n_t):
        mb.row(f"sbm_{t}", {f"dock_{v}_{t}": 1.0 for v in range(n_v)}, hi=1.0)

    for v in range(n_v):
        for t in range(n_t):
            mb.row(
                f"unload_link_{v}_{t}",
                {
                    **{f"unload_{v}_{i}_{t}": 1.0 for i in range(n_i)},
                    f"dock_{v}_{t}": -unload_rate,
                },
                hi=0.0,
            )
            prev = {f"vinv_{v}_{t - 1}": 1.0} if t else {}
            rhs = -float(parcel[v]) if t == 0 else 0.0
            mb.row(
                f"vbal_{v}_{t}",
                {
                    **prev,
                    f"vinv_{v}_{t}": -1.0,
                    **{f"unload_{v}_{i}_{t}": -1.0 for i in range(n_i)},
                },
                rhs,
                rhs,
            )
        # Vessel must be empty by the end of the horizon.
        mb.row(f"vempty_{v}", {f"vinv_{v}_{n_t - 1}": 1.0}, 0.0, 0.0)

    for i in range(n_i):
        for c in range(n_c):
            for t in range(n_t):
                mb.var(f"as_{i}_{c}_{t}", 0, 1, 0.0, integer=True)
                mb.var(f"invs_{i}_{c}_{t}", 0.0, stor_cap, hold_s)
                mb.row(
                    f"shold_{i}_{c}_{t}",
                    {f"invs_{i}_{c}_{t}": 1.0, f"as_{i}_{c}_{t}": -stor_cap},
                    hi=0.0,
                )
        for t in range(n_t):
            mb.row(f"sone_{i}_{t}", {f"as_{i}_{c}_{t}": 1.0 for c in range(n_c)}, hi=1.0)

    for j in range(n_j):
        for c in range(n_c):
            for t in range(n_t):
                mb.var(f"ac_{j}_{c}_{t}", 0, 1, 0.0, integer=True)
                mb.var(f"invc_{j}_{c}_{t}", 0.0, chg_cap, hold_c)
                mb.row(
                    f"chold_{j}_{c}_{t}",
                    {f"invc_{j}_{c}_{t}": 1.0, f"ac_{j}_{c}_{t}": -chg_cap},
                    hi=0.0,
                )
        for t in range(n_t):
            mb.row(f"cone_{j}_{t}", {f"ac_{j}_{c}_{t}": 1.0 for c in range(n_c)}, hi=1.0)

    for i in range(n_i):
        for j in range(n_j):
            for c in range(n_c):
                for t in range(n_t):
                    mb.var(f"xfer_{i}_{j}_{c}_{t}", 0.0, xfer_rate, 0.0)

    for j in range(n_j):
        for l in range(n_l):
            for c in range(n_c):
                for t in range(n_t):
                    mb.var(f"feed_{j}_{l}_{c}_{t}", 0.0, cdu_hi, 0.0)

    # Storage balances.
    for i in range(n_i):
        for c in range(n_c):
            for t in range(n_t):
                unload_in = {
                    f"unload_{v}_{i}_{t}": 1.0
                    for v in range(n_v)
                    if int(crude_of[v]) == c
                }
                xfer_out = {f"xfer_{i}_{j}_{c}_{t}": -1.0 for j in range(n_j)}
                prev = {f"invs_{i}_{c}_{t - 1}": 1.0} if t else {}
                rhs = -float(opening[i, c]) if t == 0 else 0.0
                mb.row(
                    f"sbal_{i}_{c}_{t}",
                    {
                        **prev,
                        f"invs_{i}_{c}_{t}": -1.0,
                        **unload_in,
                        **xfer_out,
                    },
                    rhs,
                    rhs,
                )

    # Charging balances.
    for j in range(n_j):
        for c in range(n_c):
            for t in range(n_t):
                xfer_in = {f"xfer_{i}_{j}_{c}_{t}": 1.0 for i in range(n_i)}
                feed_out = {f"feed_{j}_{l}_{c}_{t}": -1.0 for l in range(n_l)}
                prev = {f"invc_{j}_{c}_{t - 1}": 1.0} if t else {}
                mb.row(
                    f"cbal_{j}_{c}_{t}",
                    {
                        **prev,
                        f"invc_{j}_{c}_{t}": -1.0,
                        **xfer_in,
                        **feed_out,
                    },
                    0.0,
                    0.0,
                )

    # CDU throughput and Lee-style linear quality inequalities.
    for l in range(n_l):
        for t in range(n_t):
            feeds = {f"feed_{j}_{l}_{c}_{t}": 1.0 for j in range(n_j) for c in range(n_c)}
            mb.row(f"cdu_{l}_{t}", feeds, cdu_lo, cdu_hi)
            mb.row(
                f"sulphur_{l}_{t}",
                {
                    f"feed_{j}_{l}_{c}_{t}": float(sulphur[c] - s_max)
                    for j in range(n_j)
                    for c in range(n_c)
                },
                hi=0.0,
            )
            mb.row(
                f"api_{l}_{t}",
                {
                    f"feed_{j}_{l}_{c}_{t}": float(api[c] - api_min)
                    for j in range(n_j)
                    for c in range(n_c)
                },
                lo=0.0,
            )

    # Transfer only when both tanks hold the same crude (big-M via assignment).
    for i in range(n_i):
        for j in range(n_j):
            for c in range(n_c):
                for t in range(n_t):
                    mb.row(
                        f"xfer_s_{i}_{j}_{c}_{t}",
                        {f"xfer_{i}_{j}_{c}_{t}": 1.0, f"as_{i}_{c}_{t}": -xfer_rate},
                        hi=0.0,
                    )
                    mb.row(
                        f"xfer_c_{i}_{j}_{c}_{t}",
                        {f"xfer_{i}_{j}_{c}_{t}": 1.0, f"ac_{j}_{c}_{t}": -xfer_rate},
                        hi=0.0,
                    )

    # Feed only from a charging tank that holds that crude.
    for j in range(n_j):
        for l in range(n_l):
            for c in range(n_c):
                for t in range(n_t):
                    mb.row(
                        f"feed_link_{j}_{l}_{c}_{t}",
                        {f"feed_{j}_{l}_{c}_{t}": 1.0, f"ac_{j}_{c}_{t}": -cdu_hi},
                        hi=0.0,
                    )

    prob = mb.build()
    n_int = int(np.sum(prob.integer)) if prob.integer is not None else 0
    meta = {
        "size": size,
        "seed": int(seed),
        "vessels": n_v,
        "storage": n_i,
        "charging": n_j,
        "cdus": n_l,
        "crudes": n_c,
        "periods": n_t,
        "period_hours": data["period_hours"],
        "sulphur": sulphur.tolist(),
        "api": api.tolist(),
        "crude_of_vessel": crude_of.tolist(),
        "arrivals": arrivals.tolist(),
        "parcel": parcel.tolist(),
        "s_max": s_max,
        "api_min": api_min,
        "cdu_lo": cdu_lo,
        "cdu_hi": cdu_hi,
        "rows": prob.m,
        "cols": prob.n,
        "nnz": prob.nnz,
        "integers": n_int,
        "citation": "Lee, Pinto, Grossmann & Park, Ind. Eng. Chem. Res. 35:1630 (1996)",
    }
    prob.meta = meta
    prob.notes.append(
        "Lee–Pinto–Grossmann–Park 1996 discrete-time crude unloading; "
        "MRPL-shaped SBM → storage → charging → CDUs; linear sulphur/API blend"
    )
    return prob


def model_sizes(seed: int = 0) -> dict[str, dict]:
    """Build each size and return row/col/nnz/integer counts (no solve)."""
    out = {}
    for name in ("small", "medium", "large"):
        p = crude_unloading(size=name, seed=seed)
        out[name] = {
            "rows": p.m,
            "cols": p.n,
            "nnz": p.nnz,
            "integers": int(np.sum(p.integer)) if p.integer is not None else 0,
            "meta": {
                k: p.meta[k]
                for k in (
                    "vessels",
                    "storage",
                    "charging",
                    "cdus",
                    "crudes",
                    "periods",
                    "period_hours",
                )
            },
        }
    return out
