"""Built-in planner models and planner-language plan tables for the desktop console.

The refinery models are `models.refinery.refinery_level` unchanged; this module only attaches
names a planner can read (crude grade, site, period, unit, product) by replaying the generator's
layout through `workload.stacks.groups`. Crude grade names are display labels on synthetic
assays (the same five labels the W02 crude-value bench uses, plus five more).
"""
from __future__ import annotations

import numpy as np

from ..model import Problem

CRUDE_LABELS = ("Basrah_Light", "Arab_Light", "Merey", "Kuwait", "Oman", "Murban", "Upper_Zakum",
                "Das_Blend", "CPC_Blend", "Azeri_Light", "Es_Sider", "Bonny_Light", "Forcados",
                "Qua_Iboe", "WTI", "Brent")
PRODUCT_LABELS = {"PMF": "Premium_petrol", "RMF": "Regular_petrol", "JF": "Jet_fuel", "FO": "Fuel_oil",
                  "LBO": "Lube_oil"}
UNIT_ROWS = {"distillation_cap": "CDU", "reform_cap": "Reformer", "crack_cap": "Cracker"}
UNIT_LABELS = {"CDU": "Crude distillation", "Reformer": "Catalytic reformer", "Cracker": "Cracker"}
STREAM = {"LN": "light naphtha", "MN": "medium naphtha", "HN": "heavy naphtha", "LO": "light oil",
          "HO": "heavy oil", "R": "residuum", "RG": "reformate", "CO": "cracked oil", "CG": "cracked gasoline",
          "CRACK": "cracker", "PMF": "premium petrol", "RMF": "regular petrol", "JF": "jet fuel",
          "LBO": "lube oil", "FO": "fuel oil"}

BUILTINS = {
    "refinery_l2": {"title": "Refinery network L2", "detail": "1 site, 12 periods, 6 crudes, 10 markets",
                    "level": 2},
    "refinery_l3": {"title": "Refinery network L3", "detail": "4 sites, 12 periods, 10 crudes, 30 markets",
                    "level": 3},
    "refinery_l4": {"title": "Refinery network L4", "detail": "8 sites, 26 periods, 12 crudes, 60 markets",
                    "level": 4},
    "williams": {"title": "Williams refinery", "detail": "H.P. Williams textbook refinery, one day"},
    "crude_blending": {"title": "Crude blending", "detail": "12 crudes into the CDU within sulphur, API, TAN"},
}


def builtin(key: str) -> Problem:
    """Build a catalog model with planner names and plan-table metadata."""
    if key.startswith("refinery_l"):
        from ..models.refinery import refinery_level
        return name_refinery(refinery_level(BUILTINS[key]["level"]))
    if key == "williams":
        from ..models.williams import williams_lp
        p = williams_lp()
        p.meta["planner"] = {"objective_label": "Daily profit", "objective_sign": 1.0,
                             "crude_prefix": "crude"}
        return p
    if key == "crude_blending":
        from ..models.industry import crude_blending
        p = crude_blending()
        p.meta["planner"] = {"objective_label": "Crude cost", "objective_sign": 1.0, "crude_prefix": "c_"}
        return p
    raise KeyError(f"unknown built-in model {key!r}")


def name_refinery(prob: Problem) -> Problem:
    """Give a `refinery_lp` model readable row and column names (layout replayed, values untouched)."""
    from ..models import refinery as RF
    from ..workload.stacks import groups
    g = groups(prob)
    R, T, K, M, P, NB, NR = g["R"], g["T"], g["K"], g["M"], g["P"], g["NB"], g["NR"]
    down, products = RF.DOWN, RF.PRODUCTS
    _, _, _, brow, _ = RF._block_template(np.full((K, 6), 0.1))
    crude = [CRUDE_LABELS[k] if k < len(CRUDE_LABELS) else f"Crude_{k + 1}" for k in range(K)]
    prod = [PRODUCT_LABELS[p] for p in products]
    cols, rows = [], []
    for r in range(R):
        for t in range(T):
            sfx = f"S{r + 1}_P{t + 1:02d}"
            cols += [f"BUY_{crude[k]}_{sfx}" for k in range(K)]
            cols += [(f"MAKE_{PRODUCT_LABELS[d]}_{sfx}" if d in PRODUCT_LABELS else f"{d}_{sfx}") for d in down]
            for b in brow:
                u = UNIT_ROWS.get(b)
                rows.append(f"CAP_{u}_{sfx}" if u else f"{b}_{sfx}")
    cols += [f"STOCK_{prod[p]}_S{r + 1}_P{t + 1:02d}" for r in range(R) for t in range(T) for p in range(P)]
    E = g["E"]
    cols += [f"SHIP_S{E[e, 0] + 1}_M{E[e, 1] + 1:02d}_{prod[p]}_P{t + 1:02d}"
             for e in range(len(E)) for p in range(P) for t in range(T)]
    cols += [f"SHORT_M{m + 1:02d}_{prod[p]}_P{t + 1:02d}" for m in range(M) for p in range(P) for t in range(T)]
    rows += [f"TANK_{prod[p]}_S{r + 1}_P{t + 1:02d}" for r in range(R) for t in range(T) for p in range(P)]
    rows += [f"DEMAND_M{m + 1:02d}_{prod[p]}_P{t + 1:02d}" for m in range(M) for p in range(P) for t in range(T)]
    rows += [f"AVAIL_{crude[k]}_P{t + 1:02d}" for k in range(K) for t in range(T)]
    if len(cols) != prob.n or len(rows) != prob.m or NB * R * T > prob.n or NR * R * T > prob.m:
        raise ValueError("refinery layout replay does not match the model")
    prob.col_names, prob.row_names = cols, rows
    prob.meta["planner"] = {"objective_label": "Net margin", "objective_sign": -1.0, "crude_prefix": "BUY_",
                            "crudes": crude, "sites": R, "periods": T}
    return prob


def objective_view(prob: Problem, objective: float | None) -> dict:
    pl = prob.meta.get("planner") or {}
    sign = float(pl.get("objective_sign", 1.0))
    return {"label": pl.get("objective_label", "Objective"),
            "value": None if objective is None else sign * float(objective), "sign": sign}


GENERIC_COLUMN_LIMIT = 2000      # models without planner names: list priced columns up to this many


def crude_columns(prob: Problem) -> dict:
    """Cargo columns a planner can value, grouped by crude grade.

    With planner names every purchase column is listed, priced or not. A cargo supplied at zero
    cost (the Williams crudes) has no price to move by a percentage, so it is valued on an
    absolute price ray instead: `relative_ok` says which. `cost` is what one unit adds to the
    minimised objective, c[j]; that is the purchase price whether the model was a min or a max.
    Each grade's default column is its first, the column the W02 bench valued.
    """
    _, cn = prob.names()
    pre = (prob.meta.get("planner") or {}).get("crude_prefix")
    cols, grades = [], {}
    for j, name in enumerate(cn):
        if pre:
            if not name.startswith(pre):
                continue
        elif prob.c[j] == 0:
            continue
        cost = float(prob.c[j])
        grade = name[len(pre):].rsplit("_S", 1)[0] if pre and name.startswith("BUY_") else name
        cols.append({"column": name, "label": label(name), "cost": cost, "grade": grade.replace("_", " "),
                     "relative_ok": cost != 0.0})
        grades.setdefault(cols[-1]["grade"], []).append(name)
    truncated = not pre and len(cols) > GENERIC_COLUMN_LIMIT
    return {"columns": cols[:GENERIC_COLUMN_LIMIT] if truncated else cols, "truncated": truncated,
            "grades": [{"grade": g, "default_column": names[0], "n_columns": len(names)}
                       for g, names in grades.items()]}


_PLACE = {"S": "site", "M": "market", "P": "period"}


def _is_place(tok: str) -> bool:
    return len(tok) > 1 and tok[0] in _PLACE and tok[1:].isdigit()


def label(name: str) -> str:
    """Planner reading of a generated name.

    BUY_Basrah_Light_S1_P01        -> Buy Basrah Light, site 1, period 1
    SHIP_S3_M13_Lube_oil_P01       -> Ship lube oil from site 3 to market 13, period 1
    DEMAND_M07_Premium_petrol_P09  -> Demand for premium petrol at market 7, period 9
    Site / market / period codes may sit anywhere in the name; the suffix form ", site r, period t"
    is kept for names that have no route, so plan_tables can split the location off.
    """
    parts = name.split("_")
    head = parts[0]
    verb = {"BUY": "Buy", "MAKE": "Make", "STOCK": "Stock of", "SHIP": "Ship", "SHORT": "Unmet demand",
            "CAP": "Capacity of", "TANK": "Tank balance of", "DEMAND": "Demand", "AVAIL": "Availability of"}.get(head)
    if verb is None or len(parts) < 2:
        tail = []
        while parts and len(parts) > 1 and _is_place(parts[-1]):
            q = parts.pop()
            tail.insert(0, f"{_PLACE[q[0]]} {int(q[1:])}")
        lead = {"bal": "Balance of", "def": "Blend of", "octane": "Octane spec of", "jet": "Jet",
                "premium": "Premium share of"}.get(parts[0])
        bits = ([lead] if lead else [STREAM.get(parts[0], parts[0])]) + [STREAM.get(b, b) for b in parts[1:]]
        text = " ".join(bits)
        return f"{text}, {', '.join(tail)}" if tail else text
    places = {q[0]: int(q[1:]) for q in parts[1:] if _is_place(q)}
    item = " ".join(q for q in parts[1:] if not _is_place(q))
    when = f", period {places['P']}" if "P" in places else ""
    if head == "SHIP" and "S" in places and "M" in places:
        return f"Ship {item.lower()} from site {places['S']} to market {places['M']}{when}"
    if head in ("DEMAND", "SHORT") and "M" in places:
        return f"{verb} for {item.lower()} at market {places['M']}{when}"
    where = "".join(f", {_PLACE[k]} {places[k]}" for k in ("S", "M", "P") if k in places)
    return f"{verb} {item}".strip() + where


def plan_tables(prob: Problem, x: np.ndarray, y: np.ndarray | None = None) -> dict:
    """Crude slate, unit throughputs and product outputs from a primal vector."""
    rn, cn = prob.names()
    x = np.asarray(x, dtype=float)
    act = prob.A @ x
    slate: dict[str, float] = {}
    products: dict[str, float] = {}
    for j, name in enumerate(cn):
        v = float(x[j])
        if abs(v) < 1e-9:
            continue
        if name.startswith("BUY_"):
            key = name[4:].rsplit("_S", 1)[0]
            slate[key] = slate.get(key, 0.0) + v
        elif name.startswith("MAKE_"):
            key = name[5:].rsplit("_S", 1)[0]
            products[key] = products.get(key, 0.0) + v
    pre = (prob.meta.get("planner") or {}).get("crude_prefix")
    if not slate and pre:
        for j, name in enumerate(cn):
            if name.startswith(pre) and abs(x[j]) > 1e-9:
                slate[name] = float(x[j])
    if not products:
        for p in PRODUCT_LABELS:
            if p in cn and abs(x[cn.index(p)]) > 1e-9:
                products[PRODUCT_LABELS[p]] = float(x[cn.index(p)])
    units = []
    for i, name in enumerate(rn):
        u = None
        if name.startswith("CAP_"):
            u = name.split("_")[1]
            where = label(name).split(", ", 1)[1] if ", " in label(name) else ""
        elif name in UNIT_ROWS:
            u, where = UNIT_ROWS[name], ""
        if u is None:
            continue
        cap = float(prob.uc[i])
        units.append({"unit": UNIT_LABELS.get(u, u), "where": where, "throughput": float(act[i]),
                      "limit": cap if np.isfinite(cap) else None,
                      "utilisation": float(act[i] / cap) if np.isfinite(cap) and cap > 0 else None,
                      "shadow_price": None if y is None else float(y[i])})
    tot = sum(slate.values()) or 1.0
    return {
        "crude_slate": sorted(({"crude": k.replace("_", " "), "volume": v, "share": v / tot}
                               for k, v in slate.items()), key=lambda d: -d["volume"]),
        "units": units,
        "products": sorted(({"product": k.replace("_", " "), "volume": v} for k, v in products.items()),
                           key=lambda d: -d["volume"]),
    }


def binding_limits(prob: Problem, x: np.ndarray, y: np.ndarray, top: int = 40) -> list[dict]:
    """Limits with a non-zero shadow price, ranked by |value|, in planner words."""
    rn, _ = prob.names()
    y = np.asarray(y, dtype=float)
    act = prob.A @ np.asarray(x, dtype=float)
    idx = [int(i) for i in np.argsort(-np.abs(y)) if abs(y[i]) > 1e-9][:top]
    out = []
    for i in idx:
        at_hi = np.isfinite(prob.uc[i]) and abs(act[i] - prob.uc[i]) <= 1e-7 * max(1.0, abs(prob.uc[i]))
        at_lo = np.isfinite(prob.lc[i]) and abs(act[i] - prob.lc[i]) <= 1e-7 * max(1.0, abs(prob.lc[i]))
        side = "fixed" if (at_hi and at_lo) else ("upper" if at_hi else ("lower" if at_lo else "-"))
        out.append({"row": rn[i], "label": label(rn[i]), "shadow_price": float(y[i]), "activity": float(act[i]),
                    "limit": float(prob.uc[i] if side == "upper" else prob.lc[i]) if side in ("upper", "lower")
                    else float(act[i]), "side": side})
    return out
