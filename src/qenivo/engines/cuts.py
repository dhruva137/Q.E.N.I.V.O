"""Cutting planes for the MILP root: Gomory mixed-integer (GMI) cuts and knapsack covers.

GMI (Gomory 1960; the mixed-integer form as in Cornuejols, "Revival of the Gomory cuts",
Math. Programming 2008). Standard form z = (x, s), [A -I] z = 0, lo <= z <= hi. For a basic
integer column z_i with fractional value, its tableau row is z_i = z_i* + sum_j g_j t_j with
t_j >= 0 the distance of nonbasic z_j from its bound (z_j = l_j + t_j or u_j - t_j). Writing
z_i + sum_j a_j t_j = z_i* (a_j = -g_j), f0 = frac(z_i*), f_j = frac(a_j):

    sum_{j integer} min(f_j/f0, (1-f_j)/(1-f0)) t_j + sum_{j continuous} max(a_j/f0, -a_j/(1-f0)) t_j >= 1

The cut is mapped back to x (slacks s = A x), scaled, and kept only if it is numerically safe:
bounded dynamism, violated by the LP point, rhs relaxed slightly against rounding.

Covers: for a row sum a_j x_j <= b over binaries (a_j > 0 after complementing), a set C with
sum_C a_j > b gives sum_C x_j <= |C| - 1; C is grown greedily in order of (1 - x_j*)/a_j.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

MAX_DYNAMISM = 1e6
MIN_VIOLATION = 1e-6


def _clean(px, b0, lx, ux, rel=1e-9):
    """Remove the tiny coefficients of a cut  px @ x >= b0, relaxing the rhs by the largest value each
    removed term can take on the column's box (so the cut stays valid). None if a removed term has an
    unbounded column."""
    mx = np.abs(px).max(initial=0.0)
    if mx == 0.0:
        return None
    small = (px != 0) & (np.abs(px) < rel * mx)
    if small.any():
        p = px[small]
        top = np.maximum(p * lx[small], p * ux[small])
        if not np.all(np.isfinite(top)):
            return None
        b0 = b0 - float(top.sum())
        px = px.copy()
        px[small] = 0.0
    return px, b0


def _frac(v):
    return v - np.floor(v)


def gmi_cuts(prob, lp, integer, lx, ux, max_cuts=50):
    """GMI cuts from an optimal simplex result `lp` (needs col/row statuses). Returns (P, rhs)
    meaning P @ x >= rhs, or None."""
    if "col_statuses" not in lp:
        return None
    m, n = prob.m, prob.n
    st = np.array(list(lp["col_statuses"]) + list(lp["row_statuses"]))
    basic = np.flatnonzero(st == "basic")
    if basic.size != m or m == 0:
        return None
    Astd = sp.hstack([prob.A.tocsc(), -sp.identity(m, format="csc")], format="csc")
    lo = np.concatenate([lx, prob.lc])
    hi = np.concatenate([ux, prob.uc])
    x = np.asarray(lp["x"], dtype=float)
    z = np.concatenate([x, prob.A @ x])
    isint = np.concatenate([integer, np.zeros(m, bool)])
    try:
        lu = la.lu_factor(Astd[:, basic].toarray())
    except (la.LinAlgError, ValueError):
        return None
    nb = np.flatnonzero(st != "basic")
    at_up = st[nb] == "at_upper"
    free_nb = st[nb] == "free"
    Nmat = Astd[:, nb]
    # candidate rows: basic integer structurals, most fractional first
    pos = np.flatnonzero(isint[basic])
    f_all = _frac(z[basic[pos]])
    dist = np.minimum(f_all, 1 - f_all)
    pos = pos[dist > 0.01]
    pos = pos[np.argsort(-np.minimum(_frac(z[basic[pos]]), 1 - _frac(z[basic[pos]])))][: 3 * max_cuts]
    cuts, rhs = [], []
    for r in pos:
        e = np.zeros(m)
        e[r] = 1.0
        rho = la.lu_solve(lu, e, trans=1)
        abar = Nmat.T @ rho                         # row r of B^-1 N ;  z_B = -B^-1 N z_N
        if np.any(free_nb & (np.abs(abar) > 1e-12)):
            continue
        # z_i = const + sum g_j t_j ,  g_j = -abar_j (at lower) ,  +abar_j (at upper)
        g = np.where(at_up, abar, -abar)
        a = -g
        f0 = _frac(z[basic[r]])
        if f0 < 0.01 or f0 > 0.99:
            continue
        intj = isint[nb] & np.isfinite(np.where(at_up, hi[nb], lo[nb]))
        fj = _frac(a)
        coef = np.where(intj, np.minimum(fj / f0, (1 - fj) / (1 - f0)),
                        np.maximum(a / f0, -a / (1 - f0)))
        coef[np.abs(coef) < 1e-12] = 0.0
        # sum coef_j t_j >= 1 with t_j = z_j - l_j (lower) or u_j - z_j (upper)
        bnd = np.where(at_up, hi[nb], lo[nb])
        if np.any(~np.isfinite(bnd) & (coef != 0)):
            continue
        pz = np.where(at_up, -coef, coef)           # coefficient on z_j
        b0 = 1.0 + float(np.sum(np.where(coef != 0, pz * bnd, 0.0)))
        # map slacks back: s = A x
        px = np.zeros(n)
        cz = np.zeros(n + m)
        cz[nb] = pz
        px += cz[:n]
        px += prob.A.T @ cz[n:]
        cl = _clean(px, b0, lx, ux)
        if cl is None:
            continue
        px, b0 = cl
        nzv = np.abs(px[px != 0])
        if nzv.size == 0 or nzv.max() / nzv.min() > MAX_DYNAMISM:
            continue
        s = nzv.max()
        px, b0 = px / s, b0 / s
        viol = b0 - px @ x
        if viol <= MIN_VIOLATION * (1 + abs(b0)):
            continue
        cuts.append(px)
        rhs.append(b0 - 1e-9 * (1 + abs(b0)))
        if len(cuts) >= max_cuts:
            break
    if not cuts:
        return None
    return sp.csr_matrix(np.array(cuts)), np.array(rhs)


def cover_cuts(prob, x, integer, lx, ux, max_cuts=50):
    """Knapsack cover cuts from rows whose columns are all binary. Returns (P, rhs) for P x >= rhs."""
    A = prob.A.tocsr()
    binary = integer & (lx >= -1e-9) & (ux <= 1 + 1e-9)
    cuts, rhs = [], []
    for i in range(prob.m):
        s, e = A.indptr[i], A.indptr[i + 1]
        idx, val = A.indices[s:e], A.data[s:e]
        if idx.size < 2 or not binary[idx].all():
            continue
        for sense in (1.0, -1.0):                    # a x <= uc   and   -a x <= -lc
            b = prob.uc[i] if sense > 0 else -prob.lc[i]
            if not np.isfinite(b):
                continue
            a = sense * val
            comp = a < 0                                # complement x_j -> 1 - x_j
            aa = np.abs(a)
            bb = b + float(np.sum(aa[comp]))
            xs = np.where(comp, 1 - x[idx], x[idx])
            if aa.sum() <= bb + 1e-9:
                continue
            order = np.argsort((1 - xs) / np.maximum(aa, 1e-12))
            tot, C = 0.0, []
            for k in order:
                C.append(k)
                tot += aa[k]
                if tot > bb + 1e-9:
                    break
            if tot <= bb + 1e-9:
                continue
            C = np.array(C)
            if xs[C].sum() <= len(C) - 1 + MIN_VIOLATION:
                continue
            # sum_{C} y_j <= |C|-1  with y = x or 1-x  ->  as P x >= rhs
            row = np.zeros(prob.n)
            r0 = -(len(C) - 1.0)
            for k in C:
                j = idx[k]
                if comp[k]:
                    row[j] += 1.0          # -(1 - x_j) -> x_j - 1
                    r0 += 1.0
                else:
                    row[j] -= 1.0
            cuts.append(row)
            rhs.append(r0)
            if len(cuts) >= max_cuts:
                return sp.csr_matrix(np.array(cuts)), np.array(rhs)
    if not cuts:
        return None
    return sp.csr_matrix(np.array(cuts)), np.array(rhs)


def _mir_row(a_int, xs_int, ub_int, s_star, b, delta):
    """MIR of  sum a_j x'_j - s <= b  (x' >= 0 integer, s >= 0) with divisor delta.
    Returns (F coefficients on x', coefficient on s, rhs, violation at the LP point) or None."""
    bd = b / delta
    f0 = bd - np.floor(bd)
    if f0 < 0.05 or f0 > 0.95:
        return None
    t = a_int / delta
    ft = t - np.floor(t)
    F = np.floor(t) + np.maximum(0.0, ft - f0) / (1.0 - f0)
    cs = -1.0 / (delta * (1.0 - f0))
    rhs = np.floor(bd)
    viol = float(F @ xs_int + cs * s_star - rhs)
    return F, cs, rhs, viol


def variable_bounds(prob, integer, lx, ux):
    """Variable bounds  y <= d z + d0  (VUB) and  y >= d z + d0  (VLB) of continuous y on an integer z
    with finite bounds, read from the two-nonzero rows. Returns {y: [(kind, z, d, d0), ...]}."""
    A = prob.A.tocsr()
    out = {}
    nnz = np.diff(A.indptr)
    for i in np.flatnonzero(nnz == 2):
        s = A.indptr[i]
        (j1, j2), (v1, v2) = A.indices[s:s + 2], A.data[s:s + 2]
        if integer[j1] == integer[j2]:
            continue
        y, ay, z, az = (j1, v1, j2, v2) if not integer[j1] else (j2, v2, j1, v1)
        if not (np.isfinite(lx[z]) and np.isfinite(ux[z])) or abs(ay) < 1e-9:
            continue
        for sense, bnd in ((1.0, prob.uc[i]), (-1.0, -prob.lc[i])):      # sense*(ay y + az z) <= bnd
            if not np.isfinite(bnd):
                continue
            a_y, a_z = sense * ay, sense * az
            d, d0 = -a_z / a_y, bnd / a_y
            if abs(d) < 1e-9 or abs(d) > 1e7 or abs(d0) > 1e7:
                continue
            out.setdefault(int(y), []).append(("vub" if a_y > 0 else "vlb", int(z), float(d), float(d0)))
    return out


def _mir_separate(idx, a, b, x, integer, lx, ux, max_divisors=8, vb=None):
    """c-MIR on one (possibly aggregated) row  sum a_j x_j <= b. Continuous columns are substituted by
    their closest bound, simple or variable (y <= d z + d0 brings the integer z into the row: with
    VUBs the c-MIR yields the flow cover family, Marchand & Wolsey 2001). Returns (efficacy, pi, pi0)
    with the cut pi x <= pi0 in the original variables, or None."""
    icoef = {}
    conts = []                                           # (c, back, const): y' = sum back + const >= 0
    s_star = 0.0
    for aj, j in zip(a, idx):
        if aj == 0.0:
            continue
        j = int(j)
        if integer[j]:
            icoef[j] = icoef.get(j, 0.0) + aj
            continue
        l, u, xj = lx[j], ux[j], x[j]
        opts = []                                        # (distance, rank, kind, z, d, d0)
        if np.isfinite(l):
            opts.append((xj - l, 1, "lo", -1, 0.0, l))
        if np.isfinite(u):
            opts.append((u - xj, 1, "up", -1, 0.0, u))
        for kind, z, d, d0 in (vb.get(j, ()) if vb else ()):
            dist = (xj - d * x[z] - d0) if kind == "vlb" else (d * x[z] + d0 - xj)
            opts.append((max(dist, 0.0), 0, kind, z, d, d0))
        if not opts:
            return None
        _, _, kind, z, d, d0 = min(opts, key=lambda t: (t[0] - 1e-9 * (1 + abs(t[0])) * (t[1] == 0), t[1]))
        b -= aj * d0
        if kind in ("vlb", "vub"):
            icoef[z] = icoef.get(z, 0.0) + aj * d
        if kind in ("lo", "vlb"):                        # y = d0 (+ d z) + y'
            c, back, const = aj, [(j, 1.0)] + ([(z, -d)] if kind == "vlb" else []), -d0
        else:                                            # y = d0 (+ d z) - y'
            c, back, const = -aj, [(j, -1.0)] + ([(z, d)] if kind == "vub" else []), d0
        if c < 0:                                        # kept: s = sum (-c) y' ; c > 0 terms are dropped
            ys = sum(m * x[q] for q, m in back) + const
            conts.append((c, back, const))
            s_star += -c * max(ys, 0.0)
    ints, a_int, xs_int, ub_int, comp_int = [], [], [], [], []
    for j, aj in icoef.items():
        if abs(aj) < 1e-12:
            continue
        l, u = lx[j], ux[j]
        if np.isfinite(u) and (not np.isfinite(l) or u - x[j] < x[j] - l):
            ints.append(j); a_int.append(-aj); xs_int.append(u - x[j]); ub_int.append(u - l)
            comp_int.append(True); b -= aj * u
        elif np.isfinite(l):
            ints.append(j); a_int.append(aj); xs_int.append(x[j] - l); ub_int.append(u - l)
            comp_int.append(False); b -= aj * l
        else:
            return None
    if not ints:
        return None
    a_int, xs_int, ub_int = np.array(a_int), np.array(xs_int), np.array(ub_int)
    frac_ok = (xs_int > 1e-6) & (xs_int < ub_int - 1e-6)
    cands = np.unique(np.abs(a_int[frac_ok & (np.abs(a_int) > 1e-9)]))[::-1][:max_divisors]
    csq = sum(c * c for c, _, _ in conts)
    best = None

    def consider(d):
        nonlocal best
        r = _mir_row(a_int, xs_int, ub_int, s_star, b, d)
        if r is None:
            return
        F, cs, rhs, viol = r
        eff = viol / max(np.sqrt(F @ F + cs * cs * csq), 1e-12)
        if viol > MIN_VIOLATION * (1 + abs(rhs)) and (best is None or eff > best[0]):
            best = (eff, d, F, cs, rhs)
    for d in list(cands) + [1.0]:
        consider(d)
    if best is not None:                                  # smaller divisors of the best one
        d0 = best[1]
        for k in (2.0, 4.0, 8.0):
            consider(d0 / k)
    if best is None:
        return None
    eff, d, F, cs, rhs = best
    pi = {}
    pi0 = rhs
    for Fj, j, comp in zip(F, ints, comp_int):            # back to the original variables
        if comp:                                          # x' = u - x
            pi[j] = pi.get(j, 0.0) - Fj
            pi0 -= Fj * ux[j]
        else:                                             # x' = x - l
            pi[j] = pi.get(j, 0.0) + Fj
            pi0 += Fj * lx[j]
    for c, back, const in conts:
        coef = cs * (-c)                                  # coefficient on y'
        for q, mlt in back:
            pi[q] = pi.get(q, 0.0) + coef * mlt
        pi0 -= coef * const
    return eff, pi, pi0


def mir_cuts(prob, x, integer, lx, ux, max_cuts=50, max_divisors=8, max_aggr=3, vbounds=True):
    """Complemented MIR cuts (Marchand & Wolsey 2001) from single rows and from aggregations of up to
    `max_aggr` further EQUALITY rows, each eliminating a continuous variable that sits strictly between
    its bounds at the LP point. Returns (P, rhs) for P x >= rhs."""
    A = prob.A.tocsr()
    AC = A.tocsc()
    act = A @ x
    eq = np.isfinite(prob.lc) & (prob.lc == prob.uc)
    vb = variable_bounds(prob, integer, lx, ux) if vbounds else None
    found = []
    for i in range(prob.m):
        s0, e0 = A.indptr[i], A.indptr[i + 1]
        idx0, val0 = A.indices[s0:e0], A.data[s0:e0]
        if idx0.size < 1:
            continue
        for sense in (1.0, -1.0):                         # a x <= uc   and   -a x <= -lc
            b0 = prob.uc[i] if sense > 0 else -prob.lc[i]
            if not np.isfinite(b0) or abs(sense * act[i] - b0) > 1e-3 * (1.0 + abs(b0)):
                continue                                  # only (nearly) tight rows
            row = {int(j): sense * v for j, v in zip(idx0, val0)}
            b = b0
            used = {i}
            best = None
            for level in range(max_aggr + 1):
                if any(integer[j] for j in row):
                    idx = np.fromiter(row.keys(), dtype=np.int64)
                    r = _mir_separate(idx, np.fromiter(row.values(), dtype=float), b, x, integer, lx, ux,
                                      max_divisors, vb)
                    if r is not None and (best is None or r[0] > best[0]):
                        best = r
                if level == max_aggr:
                    break
                # eliminate the continuous variable farthest from its bounds through an unused equality row
                pick = None
                for j, aj in row.items():
                    if integer[j] or abs(aj) < 1e-9:
                        continue
                    dist = min(x[j] - lx[j], ux[j] - x[j])
                    if dist > 1e-6 and (pick is None or dist > pick[0]):
                        pick = (dist, j)
                if pick is None:
                    break
                j = pick[1]
                k = None
                for q in range(AC.indptr[j], AC.indptr[j + 1]):
                    rr = AC.indices[q]
                    if eq[rr] and rr not in used and abs(AC.data[q]) > 1e-9:
                        k = rr
                        break
                if k is None:
                    break
                used.add(k)
                f = row[j] / A[k, j]
                for q in range(A.indptr[k], A.indptr[k + 1]):
                    jj = int(A.indices[q])
                    row[jj] = row.get(jj, 0.0) - f * A.data[q]
                    if abs(row[jj]) < 1e-12:
                        del row[jj]
                b -= f * prob.lc[k]
            if best is None:
                continue
            eff, pid, pi0 = best
            pi = np.zeros(prob.n)
            for j, v in pid.items():
                pi[j] += v
            cl = _clean(-pi, -pi0, lx, ux)                 # as -pi x >= -pi0
            if cl is None:
                continue
            pi, pi0 = -cl[0], -cl[1]
            nz = np.abs(pi[pi != 0])
            if nz.size == 0 or nz.max() / nz.min() > MAX_DYNAMISM:
                continue
            if pi @ x - pi0 <= MIN_VIOLATION * (1 + abs(pi0)):
                continue
            found.append((eff, -pi, -(pi0 + 1e-9 * (1 + abs(pi0)))))     # as P x >= rhs
    if not found:
        return None
    found.sort(key=lambda t: -t[0])
    return (sp.csr_matrix(np.array([f[1] for f in found[:max_cuts]])),
            np.array([f[2] for f in found[:max_cuts]]))
