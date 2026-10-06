"""Scalable multi-site, multi-period refinery supply-chain LP (SPEC §4).

Kernel: every (site, period) block is the Williams refinery (generator/williams.py) with
crude-specific distillation yields; blocks are linked by
  * crude availability per (crude, period), shared by all sites,
  * product inventory balances per (site, product, period) with tank limits,
  * a sparse site -> market shipping network (each market served by `deg` sites),
  * market demand rows per (market, product, period): dmin <= shipped + unmet <= dmax.
Minimise crude cost + transport + holding + penalty * unmet - revenue.

Feasible by construction (all flows 0, unmet = dmin) and bounded (every flow is limited by
crude availability, capacities or market caps). Deterministic for a given seed.
Blending is linear because blend-stock qualities are fixed per stream (Williams' assumption);
pooled-quality nonlinearity is the subject of generator/recursion.py.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..model import Problem as LPProblem
from . import williams as W

DOWN = ["LN_RG", "LN_PMF", "LN_RMF", "MN_RG", "MN_PMF", "MN_RMF", "HN_RG", "HN_PMF", "HN_RMF",
        "RG_PMF", "RG_RMF", "LO_CRACK", "LO_JF", "HO_CRACK", "HO_JF", "CO_JF", "CG_PMF", "CG_RMF",
        "R_LBO", "R_JF", "PMF", "RMF", "JF", "FO", "LBO"]
# pooled variant: the three naphthas enter one blending pool whose octane is unknown
DOWN_POOLED = ["LN_RG", "LN_POOL", "MN_RG", "MN_POOL", "HN_RG", "HN_POOL", "POOL_PMF", "POOL_RMF",
               "RG_PMF", "RG_RMF", "LO_CRACK", "LO_JF", "HO_CRACK", "HO_JF", "CO_JF", "CG_PMF", "CG_RMF",
               "R_LBO", "R_JF", "PMF", "RMF", "JF", "FO", "LBO"]
POOL_Q0 = 80.0          # placeholder pool octane written into the template
PRODUCTS = ["PMF", "RMF", "JF", "FO", "LBO"]
CUTS = ["LN", "MN", "HN", "LO", "HO", "R"]
# typical product output per bbl crude at the Williams optimum (6818+17045+15156+0+500)/45000
OUT_PER_CRUDE = {"PMF": 0.152, "RMF": 0.379, "JF": 0.337, "FO": 0.05, "LBO": 0.011}

# scale levels for the sweep: (sites, periods, crudes, markets)
LEVELS = [(1, 1, 2, 2), (1, 4, 4, 4), (2, 12, 6, 10), (4, 12, 10, 30), (8, 26, 12, 60),
          (10, 52, 16, 120), (20, 52, 20, 250), (30, 104, 24, 400), (40, 156, 30, 700),
          (50, 365, 40, 1000)]


def crude_yields(K, rng):
    """(K, 6) cut yields: mixes of Williams' two assays plus noise; waste 3-6%."""
    y1 = np.array([W.YIELD["crude1"][c] for c in CUTS])
    y2 = np.array([W.YIELD["crude2"][c] for c in CUTS])
    a = rng.uniform(0, 1, K)[:, None]
    y = a * y1 + (1 - a) * y2 + rng.normal(0, 0.01, (K, 6))
    y = np.clip(y, 0.02, None)
    y *= (1 - rng.uniform(0.03, 0.06, (K, 1))) / y.sum(axis=1, keepdims=True)
    return y


def _block_template(yields, pooled=False):
    """Rows x cols matrix of one Williams block with K crudes (cols: crudes then DOWN)."""
    K = yields.shape[0]
    down = DOWN_POOLED if pooled else DOWN
    col = {f"crude{k}": k for k in range(K)}
    col.update({n: K + i for i, n in enumerate(down)})
    rows, lo, hi, ri, ci, v = [], [], [], [], [], []

    def row(name, coefs, l=-np.inf, h=np.inf):
        i = len(rows)
        rows.append(name); lo.append(l); hi.append(h)
        for k, a in coefs.items():
            ri.append(i); ci.append(col[k]); v.append(a)

    crudes = [f"crude{k}" for k in range(K)]
    fo = sum(W.FUEL_OIL_RATIO.values())
    row("distillation_cap", {c: 1.0 for c in crudes}, h=W.DIST_CAP)
    for j, n in enumerate(CUTS):
        coefs = {crudes[k]: yields[k, j] for k in range(K)}
        if n in ("LN", "MN", "HN") and pooled:
            coefs.update({f"{n}_RG": -1.0, f"{n}_POOL": -1.0})
        elif n in ("LN", "MN", "HN"):
            coefs.update({f"{n}_RG": -1.0, f"{n}_PMF": -1.0, f"{n}_RMF": -1.0})
        elif n in ("LO", "HO"):
            coefs.update({f"{n}_CRACK": -1.0, f"{n}_JF": -1.0, "FO": -W.FUEL_OIL_RATIO[n] / fo})
        else:
            coefs.update({"R_LBO": -1.0, "R_JF": -1.0, "FO": -W.FUEL_OIL_RATIO["R"] / fo})
        row(f"bal_{n}", coefs, 0.0, 0.0)
    row("bal_RG", {**{f"{n}_RG": W.REFORM[n] for n in W.REFORM}, "RG_PMF": -1.0, "RG_RMF": -1.0}, 0.0, 0.0)
    row("bal_CO", {"LO_CRACK": W.CRACK["LO"]["CO"], "HO_CRACK": W.CRACK["HO"]["CO"], "CO_JF": -1.0,
                   "FO": -W.FUEL_OIL_RATIO["CO"] / fo}, 0.0, 0.0)
    row("bal_CG", {"LO_CRACK": W.CRACK["LO"]["CG"], "HO_CRACK": W.CRACK["HO"]["CG"],
                   "CG_PMF": -1.0, "CG_RMF": -1.0}, 0.0, 0.0)
    row("reform_cap", {f"{n}_RG": 1.0 for n in W.REFORM}, h=W.REFORM_CAP)
    row("crack_cap", {"LO_CRACK": 1.0, "HO_CRACK": 1.0}, h=W.CRACK_CAP)
    if pooled:
        row("bal_POOL", {"LN_POOL": 1.0, "MN_POOL": 1.0, "HN_POOL": 1.0, "POOL_PMF": -1.0, "POOL_RMF": -1.0}, 0.0, 0.0)
    for p in ("PMF", "RMF"):
        src = ("POOL", "RG", "CG") if pooled else ("LN", "MN", "HN", "RG", "CG")
        row(f"def_{p}", {**{s_ + "_" + p: 1.0 for s_ in src}, p: -1.0}, 0.0, 0.0)
    row("def_JF", {"LO_JF": 1.0, "HO_JF": 1.0, "CO_JF": 1.0, "R_JF": 1.0, "JF": -1.0}, 0.0, 0.0)
    row("def_LBO", {"R_LBO": W.LUBE_YIELD, "LBO": -1.0}, 0.0, 0.0)
    octane = dict(W.OCTANE, POOL=POOL_Q0)
    for p in ("PMF", "RMF"):
        src = ("POOL", "RG", "CG") if pooled else ("LN", "MN", "HN", "RG", "CG")
        row(f"octane_{p}", {s_ + "_" + p: octane[s_] - W.OCT_MIN[p] for s_ in src}, l=0.0)
    row("jet_vapour", {s + "_JF": W.VAPOUR[s] - W.JET_VP_MAX for s in ("LO", "HO", "CO", "R")}, h=0.0)
    row("premium_vs_regular", {"PMF": 1.0, "RMF": -W.PREMIUM_MIN_FRACTION_OF_REGULAR}, l=0.0)
    B = sp.coo_matrix((v, (ri, ci)), shape=(len(rows), K + len(DOWN)))
    return B, np.array(lo), np.array(hi), rows, col


def refinery_lp(sites=2, periods=4, crudes=4, markets=6, deg=3, seed=0, names=False,
                pooled=False) -> LPProblem:
    rng = np.random.default_rng(seed)
    R, T, K, M, P = sites, periods, crudes, markets, len(PRODUCTS)
    deg = min(deg, R)
    y = crude_yields(K, rng)
    B, blo, bhi, brow, bcol = _block_template(y, pooled)
    NR, NB = B.shape
    nb = R * T

    # --- per-site capacities (scale the Williams capacities)
    size = rng.uniform(0.5, 1.5, R)
    cap_rows = {brow.index("distillation_cap"): W.DIST_CAP, brow.index("reform_cap"): W.REFORM_CAP,
                brow.index("crack_cap"): W.CRACK_CAP}
    lc_b = np.tile(blo, nb).reshape(nb, NR)
    uc_b = np.tile(bhi, nb).reshape(nb, NR)
    site_of_block = np.repeat(np.arange(R), T)
    for r_i, base in cap_rows.items():
        uc_b[:, r_i] = base * size[site_of_block]
    lx_b = np.zeros((nb, NB))
    ux_b = np.full((nb, NB), np.inf)
    ux_b[:, bcol["LBO"]] = W.LUBE_MAX * size[site_of_block]

    # --- column offsets
    nI = R * T * P
    E = np.array([(r, m) for m in range(M) for r in rng.choice(R, deg, replace=False)])  # edges
    nE = len(E)
    nS = nE * P * T
    nU = M * P * T
    oI, oS, oU = nb * NB, nb * NB + nI, nb * NB + nI + nS
    n = oU + nU

    def cI(r, p, t):
        return oI + (r * T + t) * P + p

    def cS(e, p, t):
        return oS + (e * P + p) * T + t

    def cU(m, p, t):
        return oU + (m * P + p) * T + t

    # --- rows: blocks, inventory balances, market rows, crude availability
    oIB = nb * NR
    oM = oIB + R * T * P
    oC = oM + M * P * T
    m_rows = oC + K * T

    Ab = sp.kron(sp.identity(nb, format="csr"), B.tocsr(), format="coo")
    ri, ci, vv = [Ab.row], [Ab.col], [Ab.data]

    rr, tt, pp = np.meshgrid(np.arange(R), np.arange(T), np.arange(P), indexing="ij")
    rr, tt, pp = rr.ravel(), tt.ravel(), pp.ravel()
    ib = oIB + (rr * T + tt) * P + pp
    prod_local = np.array([bcol[p] for p in PRODUCTS])
    ri += [ib, ib]; ci += [(rr * T + tt) * NB + prod_local[pp], cI(rr, pp, tt)]; vv += [np.ones_like(ib, float), -np.ones_like(ib, float)]
    prev = tt > 0
    ri.append(ib[prev]); ci.append(cI(rr[prev], pp[prev], tt[prev] - 1)); vv.append(np.ones(prev.sum()))

    ee, pp2, tt2 = np.meshgrid(np.arange(nE), np.arange(P), np.arange(T), indexing="ij")
    ee, pp2, tt2 = ee.ravel(), pp2.ravel(), tt2.ravel()
    sc = cS(ee, pp2, tt2)
    er, em = E[ee, 0], E[ee, 1]
    ri += [oIB + (er * T + tt2) * P + pp2, oM + (em * P + pp2) * T + tt2]
    ci += [sc, sc]; vv += [-np.ones(len(sc)), np.ones(len(sc))]

    mm, pp3, tt3 = np.meshgrid(np.arange(M), np.arange(P), np.arange(T), indexing="ij")
    mm, pp3, tt3 = mm.ravel(), pp3.ravel(), tt3.ravel()
    ri.append(oM + (mm * P + pp3) * T + tt3); ci.append(cU(mm, pp3, tt3)); vv.append(np.ones(len(mm)))

    rk, kk, tk = np.meshgrid(np.arange(R), np.arange(K), np.arange(T), indexing="ij")
    rk, kk, tk = rk.ravel(), kk.ravel(), tk.ravel()
    ri.append(oC + kk * T + tk); ci.append((rk * T + tk) * NB + kk); vv.append(np.ones(len(rk)))

    A = sp.coo_matrix((np.concatenate(vv), (np.concatenate(ri), np.concatenate(ci))),
                      shape=(m_rows, n)).tocsr()
    A.eliminate_zeros()          # e.g. light oil's vapour pressure equals the jet limit (coefficient 0)

    # --- bounds and data
    lc = np.concatenate([lc_b.ravel(), np.zeros(R * T * P), np.zeros(M * P * T), np.full(K * T, -np.inf)])
    uc = np.concatenate([uc_b.ravel(), np.zeros(R * T * P), np.zeros(M * P * T), np.zeros(K * T)])
    total_crude_cap = W.DIST_CAP * size.sum()
    avail = total_crude_cap / K * rng.uniform(0.6, 1.4, (K, T))
    uc[oC:] = avail.ravel()
    supply = np.array([OUT_PER_CRUDE[p] for p in PRODUCTS]) * total_crude_cap * 0.9
    dmax = supply[None, :, None] / M * rng.uniform(0.5, 1.5, (M, P, T))
    dmin = 0.3 * dmax * rng.uniform(0.0, 1.0, (M, P, T))
    lc[oM:oC], uc[oM:oC] = dmin.ravel(), dmax.ravel()

    lx = np.concatenate([lx_b.ravel(), np.zeros(nI + nS + nU)])
    tank = 3.0 * np.array([OUT_PER_CRUDE[p] for p in PRODUCTS])[None, :] * W.DIST_CAP * size[:, None]
    uI = np.broadcast_to(tank[:, None, :], (R, T, P)).ravel()
    ux = np.concatenate([ux_b.ravel(), uI, np.full(nS, np.inf), dmin.ravel()])

    c = np.zeros(n)
    crude_cost = rng.uniform(0.85, 1.15, (K, T)) * (1.0 + 0.5 * (y[:, :3].sum(axis=1)[:, None] - 0.5))
    cb = np.zeros((R, T, NB))
    cb[:, :, :K] = crude_cost.T[None, :, :]
    cb[:, :, bcol["LN_RG"]] = cb[:, :, bcol["MN_RG"]] = cb[:, :, bcol["HN_RG"]] = 0.10   # reforming op cost
    cb[:, :, bcol["LO_CRACK"]] = cb[:, :, bcol["HO_CRACK"]] = 0.15                    # cracking op cost
    c[:oI] = cb.ravel()
    c[oI:oS] = 0.02                                                                       # holding
    price = (np.array([W.PROFIT[p] for p in PRODUCTS]) + 1.0)[None, :, None] * rng.uniform(0.9, 1.1, (M, P, T))
    transport = rng.uniform(0.05, 0.3, nE)
    c[sc] = transport[ee] - price[em, pp2, tt2]
    c[oU:] = 10.0                                                                         # unmet-demand penalty

    prob = LPProblem(c=c, A=A, lc=lc, uc=uc, lx=lx, ux=ux,
                     name=f"refinery_R{R}_T{T}_K{K}_M{M}_s{seed}" + ("_pooled" if pooled else ""))
    if pooled:   # positions needed by generator/recursion.py
        A.sort_indices()
        blocks = np.arange(nb)
        prob.meta["pool"] = {"cols": {n: blocks * NB + bcol[n] for n in ("LN_POOL", "MN_POOL", "HN_POOL", "POOL_PMF", "POOL_RMF")},
                     "rows": {p: blocks * NR + brow.index(f"octane_{p}") for p in ("PMF", "RMF")}}
    if names:
        prob.row_names = [f"r{i}" for i in range(m_rows)]
        prob.col_names = [f"x{j}" for j in range(n)]
    return prob


def refinery_level(level: int, seed: int = 0) -> LPProblem:
    R, T, K, M = LEVELS[level]
    return refinery_lp(R, T, K, M, seed=seed)


if __name__ == "__main__":
    for lv in range(len(LEVELS)):
        R, T, K, M = LEVELS[lv]
        if lv > 7:
            print(f"level {lv}: R={R} T={T} K={K} M={M} (large; generate on Colab)")
            continue
        p = refinery_level(lv)
        print(f"level {lv}: R={R} T={T} K={K} M={M}  {p.size_str()}")
