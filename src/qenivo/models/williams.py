"""H.P. Williams, *Model Building in Mathematical Programming* (5th ed., Wiley), problem
"Refinery Optimisation". Data cross-checked in research/REFINERY.md against two independent
open reproductions (Gurobi modeling-examples/refinery; naraqb/williams-problems).
Published optimum: profit 211,365.13 per day.

Stored in minimisation form (c = -profit, obj_sign = -1), so objective() reports profit.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..model import Problem as LPProblem

CRUDES = {"crude1": 20000.0, "crude2": 30000.0}
DIST_CAP = 45000.0
YIELD = {  # bbl cut per bbl crude
    "crude1": {"LN": 0.10, "MN": 0.20, "HN": 0.20, "LO": 0.12, "HO": 0.20, "R": 0.13},
    "crude2": {"LN": 0.15, "MN": 0.25, "HN": 0.18, "LO": 0.08, "HO": 0.19, "R": 0.12},
}
REFORM = {"LN": 0.60, "MN": 0.52, "HN": 0.45}
REFORM_CAP = 10000.0
CRACK = {"LO": {"CO": 0.68, "CG": 0.28}, "HO": {"CO": 0.75, "CG": 0.20}}
CRACK_CAP = 8000.0
OCTANE = {"LN": 90.0, "MN": 80.0, "HN": 70.0, "RG": 115.0, "CG": 105.0}
OCT_MIN = {"PMF": 94.0, "RMF": 84.0}
VAPOUR = {"LO": 1.0, "HO": 0.6, "CO": 1.5, "R": 0.05}
JET_VP_MAX = 1.0
FUEL_OIL_RATIO = {"LO": 10.0, "HO": 4.0, "CO": 3.0, "R": 1.0}
LUBE_YIELD = 0.5
LUBE_MIN, LUBE_MAX = 500.0, 1000.0
PROFIT = {"PMF": 7.0, "RMF": 6.0, "JF": 4.0, "FO": 3.5, "LBO": 1.5}
PREMIUM_MIN_FRACTION_OF_REGULAR = 0.4
PUBLISHED_OPTIMUM = 211365.13


class _Builder:
    def __init__(self):
        self.cols, self.lx, self.ux, self.c = [], [], [], []
        self.rows, self.lc, self.uc = [], [], []
        self.ri, self.ci, self.v = [], [], []
        self.col = {}

    def var(self, name, lo=0.0, hi=np.inf, obj=0.0):
        self.col[name] = len(self.cols)
        self.cols.append(name); self.lx.append(lo); self.ux.append(hi); self.c.append(obj)

    def row(self, name, coefs: dict, lo=-np.inf, hi=np.inf):
        i = len(self.rows)
        self.rows.append(name); self.lc.append(lo); self.uc.append(hi)
        for k, a in coefs.items():
            self.ri.append(i); self.ci.append(self.col[k]); self.v.append(a)

    def build(self, name) -> LPProblem:
        A = sp.csr_matrix((self.v, (self.ri, self.ci)), shape=(len(self.rows), len(self.cols)))
        A.eliminate_zeros()
        return LPProblem(c=np.array(self.c), A=A, lc=np.array(self.lc), uc=np.array(self.uc),
                         lx=np.array(self.lx), ux=np.array(self.ux), obj_sign=-1.0, name=name,
                         row_names=self.rows, col_names=self.cols)


def williams_lp() -> LPProblem:
    b = _Builder()
    for k, cap in CRUDES.items():
        b.var(k, 0.0, cap)
    for n in ("LN", "MN", "HN"):
        b.var(f"{n}_RG"); b.var(f"{n}_PMF"); b.var(f"{n}_RMF")
    b.var("RG_PMF"); b.var("RG_RMF")
    for o in ("LO", "HO"):
        b.var(f"{o}_CRACK"); b.var(f"{o}_JF")
    b.var("CO_JF"); b.var("CG_PMF"); b.var("CG_RMF")
    b.var("R_LBO"); b.var("R_JF")
    for p, pr in PROFIT.items():
        lo, hi = (LUBE_MIN, LUBE_MAX) if p == "LBO" else (0.0, np.inf)
        b.var(p, lo, hi, obj=-pr)                     # minimise -profit

    b.row("distillation_cap", {k: 1.0 for k in CRUDES}, hi=DIST_CAP)
    # cut balances: produced = used
    for n in ("LN", "MN", "HN"):
        b.row(f"bal_{n}", {**{k: YIELD[k][n] for k in CRUDES},
                           f"{n}_RG": -1.0, f"{n}_PMF": -1.0, f"{n}_RMF": -1.0}, 0.0, 0.0)
    fo = sum(FUEL_OIL_RATIO.values())
    for o in ("LO", "HO"):
        b.row(f"bal_{o}", {**{k: YIELD[k][o] for k in CRUDES}, f"{o}_CRACK": -1.0,
                           f"{o}_JF": -1.0, "FO": -FUEL_OIL_RATIO[o] / fo}, 0.0, 0.0)
    b.row("bal_R", {**{k: YIELD[k]["R"] for k in CRUDES}, "R_LBO": -1.0, "R_JF": -1.0,
                    "FO": -FUEL_OIL_RATIO["R"] / fo}, 0.0, 0.0)
    b.row("bal_RG", {**{f"{n}_RG": REFORM[n] for n in REFORM}, "RG_PMF": -1.0, "RG_RMF": -1.0}, 0.0, 0.0)
    b.row("bal_CO", {"LO_CRACK": CRACK["LO"]["CO"], "HO_CRACK": CRACK["HO"]["CO"],
                     "CO_JF": -1.0, "FO": -FUEL_OIL_RATIO["CO"] / fo}, 0.0, 0.0)
    b.row("bal_CG", {"LO_CRACK": CRACK["LO"]["CG"], "HO_CRACK": CRACK["HO"]["CG"],
                     "CG_PMF": -1.0, "CG_RMF": -1.0}, 0.0, 0.0)
    b.row("reform_cap", {f"{n}_RG": 1.0 for n in REFORM}, hi=REFORM_CAP)
    b.row("crack_cap", {"LO_CRACK": 1.0, "HO_CRACK": 1.0}, hi=CRACK_CAP)
    # product definitions
    for p in ("PMF", "RMF"):
        b.row(f"def_{p}", {"LN_" + p: 1.0, "MN_" + p: 1.0, "HN_" + p: 1.0, "RG_" + p: 1.0,
                           "CG_" + p: 1.0, p: -1.0}, 0.0, 0.0)
    b.row("def_JF", {"LO_JF": 1.0, "HO_JF": 1.0, "CO_JF": 1.0, "R_JF": 1.0, "JF": -1.0}, 0.0, 0.0)
    b.row("def_LBO", {"R_LBO": LUBE_YIELD, "LBO": -1.0}, 0.0, 0.0)
    # quality specifications (linear: fixed component qualities)
    for p in ("PMF", "RMF"):
        b.row(f"octane_{p}", {"LN_" + p: OCTANE["LN"] - OCT_MIN[p], "MN_" + p: OCTANE["MN"] - OCT_MIN[p],
                              "HN_" + p: OCTANE["HN"] - OCT_MIN[p], "RG_" + p: OCTANE["RG"] - OCT_MIN[p],
                              "CG_" + p: OCTANE["CG"] - OCT_MIN[p]}, lo=0.0)
    b.row("jet_vapour", {"LO_JF": VAPOUR["LO"] - JET_VP_MAX, "HO_JF": VAPOUR["HO"] - JET_VP_MAX,
                         "CO_JF": VAPOUR["CO"] - JET_VP_MAX, "R_JF": VAPOUR["R"] - JET_VP_MAX}, hi=0.0)
    b.row("premium_vs_regular", {"PMF": 1.0, "RMF": -PREMIUM_MIN_FRACTION_OF_REGULAR}, lo=0.0)
    return b.build("williams_refinery")
