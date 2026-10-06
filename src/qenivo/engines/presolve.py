"""Presolve for LP and MILP, with postsolve of the primal point.

Reductions, repeated until nothing changes (each is counted in `log`):
  * integer bounds rounded inwards (ceil lower, floor upper)
  * fixed columns (l = u) removed; their contribution moves into the row bounds and c0
  * empty rows removed (or the model is declared infeasible if 0 is outside the row range)
  * singleton rows turned into column bounds and removed
  * empty columns fixed at their cheapest finite bound
  * activity bound tightening: every row's minimum / maximum activity implies bounds on each of
    its integer columns (rounded); rows whose activity range already lies inside [lc, uc] are
    dropped as redundant
Continuous bounds are only changed by singleton rows, so LP duals of the reduced model stay
meaningful. The answer is always re-checked on the ORIGINAL model after postsolve.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..model import Problem

FEAS = 1e-9


@dataclass
class Presolved:
    prob: Problem                     # the reduced model
    cols: np.ndarray                  # original index of each kept column
    rows: np.ndarray                  # original index of each kept row
    fixed: dict                       # original column -> value
    status: str = "reduced"           # "reduced" | "infeasible"
    log: dict = field(default_factory=dict)

    def postsolve(self, x_red, n: int) -> np.ndarray:
        x = np.zeros(n)
        for j, v in self.fixed.items():
            x[j] = v
        if x_red is not None and len(self.cols):
            x[self.cols] = x_red
        return x


def _activity(A, lx, ux):
    """Per-row min/max activity (finite part) and the number of infinite contributions."""
    Ap, An = A.multiply(A > 0).tocsr(), A.multiply(A < 0).tocsr()
    lf, uf = np.where(np.isfinite(lx), lx, 0.0), np.where(np.isfinite(ux), ux, 0.0)
    li, ui = (~np.isfinite(lx)).astype(float), (~np.isfinite(ux)).astype(float)
    Bp, Bn = (Ap != 0).astype(float), (An != 0).astype(float)
    mn = Ap @ lf + An @ uf
    mx = Ap @ uf + An @ lf
    mn_inf = Bp @ li + Bn @ ui
    mx_inf = Bp @ ui + Bn @ li
    return mn, mx, mn_inf, mx_inf


def presolve(prob: Problem, max_passes: int = 20, coef_tighten: bool = True, dual_fix: bool = True) -> Presolved:
    n0, m0 = prob.n, prob.m
    A = prob.A.tocsr().copy()
    c, lc, uc, lx, ux = (v.astype(float).copy() for v in (prob.c, prob.lc, prob.uc, prob.lx, prob.ux))
    integer = prob.integer.copy() if prob.integer is not None else np.zeros(n0, bool)
    c0 = float(prob.c0)
    cols, rows = np.arange(n0), np.arange(m0)
    fixed: dict = {}
    log = {k: 0 for k in ("int_rounded", "fixed_cols", "empty_rows", "singleton_rows", "empty_cols",
                          "tightened", "redundant_rows", "passes")}

    def infeasible(why):
        log["reason"] = why
        return Presolved(prob, np.arange(n0), np.arange(m0), {}, "infeasible", log)

    for _ in range(max_passes):
        log["passes"] += 1
        changed = False
        # integer rounding
        lo_r, up_r = np.ceil(lx[integer] - 1e-9), np.floor(ux[integer] + 1e-9)
        cnt = int(np.sum(lo_r > lx[integer] + 1e-12) + np.sum(up_r < ux[integer] - 1e-12))
        if cnt:
            lx[integer], ux[integer] = lo_r, up_r
            log["int_rounded"] += cnt
            changed = True
        if np.any(lx > ux + FEAS):
            return infeasible("column bounds cross")

        # fixed columns
        fx = np.flatnonzero(np.abs(ux - lx) <= 1e-12)
        if fx.size:
            v = lx[fx]
            shift = A[:, fx] @ v
            lc, uc = lc - shift, uc - shift
            c0 += float(c[fx] @ v)
            for j, val in zip(cols[fx], v):
                fixed[int(j)] = float(val)
            keep = np.setdiff1d(np.arange(len(cols)), fx)
            A, c, lx, ux, integer, cols = A[:, keep], c[keep], lx[keep], ux[keep], integer[keep], cols[keep]
            log["fixed_cols"] += int(fx.size)
            changed = True

        nnz_row = np.diff(A.tocsr().indptr)
        # empty rows
        er = np.flatnonzero(nnz_row == 0)
        if er.size:
            if np.any(lc[er] > FEAS) or np.any(uc[er] < -FEAS):
                return infeasible("empty row with a nonzero requirement")
            log["empty_rows"] += int(er.size)
        # singleton rows -> bounds
        A = A.tocsr()
        sr = np.flatnonzero(nnz_row == 1)
        for i in sr:
            j = A.indices[A.indptr[i]]
            a = A.data[A.indptr[i]]
            lo, hi = (lc[i] / a, uc[i] / a) if a > 0 else (uc[i] / a, lc[i] / a)
            if integer[j]:
                lo, hi = np.ceil(lo - 1e-9), np.floor(hi + 1e-9)
            lx[j], ux[j] = max(lx[j], lo), min(ux[j], hi)
        if sr.size:
            log["singleton_rows"] += int(sr.size)
            if np.any(lx > ux + FEAS):
                return infeasible("singleton rows give crossing bounds")
            ux = np.maximum(ux, lx)
        # activity: redundant rows and integer bound tightening
        mn, mx, mn_inf, mx_inf = _activity(A, lx, ux)
        if np.any((mn_inf == 0) & (mn > uc + 1e-7 * (1 + np.abs(uc)))) or \
                np.any((mx_inf == 0) & (mx < lc - 1e-7 * (1 + np.abs(lc)))):
            return infeasible("row activity range misses the row bounds")
        # forcing rows: the row bound equals an extreme activity, so every column sits at that extreme
        with np.errstate(invalid="ignore"):
            f_lo = (mn_inf == 0) & np.isfinite(uc) & (mn >= uc - 1e-9 * (1 + np.abs(uc))) & (nnz_row > 0)
            f_hi = (mx_inf == 0) & np.isfinite(lc) & (mx <= lc + 1e-9 * (1 + np.abs(lc))) & (nnz_row > 0)
        if f_lo.any() or f_hi.any():
            Ar = A.tocsr()
            rr = np.repeat(np.arange(Ar.shape[0]), np.diff(Ar.indptr))
            jj, aa = Ar.indices, Ar.data
            to_low = (f_lo[rr] & (aa > 0)) | (f_hi[rr] & (aa < 0))
            to_up = (f_lo[rr] & (aa < 0)) | (f_hi[rr] & (aa > 0))
            nf = int(np.sum(np.abs(ux[jj[to_low]] - lx[jj[to_low]]) > 1e-12) +
                     np.sum(np.abs(ux[jj[to_up]] - lx[jj[to_up]]) > 1e-12))
            if nf:
                lo_t, up_t = lx.copy(), ux.copy()
                ux[jj[to_low]] = lo_t[jj[to_low]]
                lx[jj[to_up]] = up_t[jj[to_up]]
                if np.any(lx > ux + FEAS):
                    return infeasible("forcing rows fix a column to two values")
                log["forcing_rows"] = log.get("forcing_rows", 0) + int(f_lo.sum() + f_hi.sum())
                changed = True
                continue                                  # re-run the pass with the columns fixed
        redundant = ((mn_inf == 0) & (mn >= lc - FEAS) | ~np.isfinite(lc)) & \
                    ((mx_inf == 0) & (mx <= uc + FEAS) | ~np.isfinite(uc))
        redundant &= nnz_row > 0
        drop = np.zeros(len(rows), bool)
        drop[er] = True
        drop[sr] = True
        drop |= redundant
        log["redundant_rows"] += int(redundant.sum())
        tightened = _tighten_integers(A, lc, uc, lx, ux, integer, mn, mx, mn_inf, mx_inf)
        if tightened:
            log["tightened"] += tightened
            changed = True
            if np.any(lx > ux + FEAS):
                return infeasible("bound tightening crossed an integer range")
        if coef_tighten and integer.any() and not drop.all():
            lxi, uxi = implied_bounds(A, lc, uc, lx, ux, integer)
            # continuous columns whose implied range collapses onto a bound are fixed there
            with np.errstate(invalid="ignore"):
                cf_lo = ~integer & np.isfinite(lx) & (uxi <= lx + 1e-9 * (1 + np.abs(lx))) & (ux > lx + 1e-12)
                cf_up = ~integer & np.isfinite(ux) & (lxi >= ux - 1e-9 * (1 + np.abs(ux))) & (ux > lx + 1e-12) & ~cf_lo
            if cf_lo.any() or cf_up.any():
                ux[cf_lo] = lx[cf_lo]
                lx[cf_up] = ux[cf_up]
                log["implied_fixed"] = log.get("implied_fixed", 0) + int(cf_lo.sum() + cf_up.sum())
                changed = True
            A = A.tocsr()
            ct = tighten_coefficients(A, lc, uc, lx, ux, integer, lxi, uxi)
            if ct:
                A.eliminate_zeros()
                log["coef_tightened"] = log.get("coef_tightened", 0) + ct
                changed = True
        if drop.any():
            keep = np.flatnonzero(~drop)
            A, lc, uc, rows = A[keep], lc[keep], uc[keep], rows[keep]
            changed = True
        # dual fixing: a column whose decrease (increase) never hurts a row nor the objective sits at
        # its lower (upper) bound in some optimal solution
        if dual_fix:
            Ac = A.tocsc()
            cc = np.repeat(np.arange(Ac.shape[1]), np.diff(Ac.indptr))
            ri, av = Ac.indices, Ac.data
            blocks_down = np.zeros(Ac.shape[1], bool)   # lowering x_j could violate a row
            blocks_up = np.zeros(Ac.shape[1], bool)
            np.logical_or.at(blocks_down, cc, ((av > 0) & np.isfinite(lc[ri])) | ((av < 0) & np.isfinite(uc[ri])))
            np.logical_or.at(blocks_up, cc, ((av > 0) & np.isfinite(uc[ri])) | ((av < 0) & np.isfinite(lc[ri])))
            nz_col = np.diff(Ac.indptr) > 0
            dn = nz_col & ~blocks_down & (c >= 0) & np.isfinite(lx) & (ux > lx)
            upf = nz_col & ~blocks_up & (c <= 0) & np.isfinite(ux) & (ux > lx) & ~dn
            if dn.any() or upf.any():
                ux[dn] = lx[dn]
                lx[upf] = ux[upf]
                log["dual_fixed"] = log.get("dual_fixed", 0) + int(dn.sum() + upf.sum())
                changed = True
        # empty columns
        A = A.tocsc()
        ec = np.flatnonzero(np.diff(A.indptr) == 0)
        fixable = []
        for j in ec:
            if c[j] > 0 and np.isfinite(lx[j]):
                fixable.append((j, lx[j]))
            elif c[j] < 0 and np.isfinite(ux[j]):
                fixable.append((j, ux[j]))
            elif c[j] == 0:
                fixable.append((j, float(np.clip(0.0, lx[j], ux[j]))))
        if fixable:
            idx = np.array([j for j, _ in fixable])
            for j, v in fixable:
                fixed[int(cols[j])] = float(v)
                c0 += float(c[j] * v)
            keep = np.setdiff1d(np.arange(len(cols)), idx)
            A, c, lx, ux, integer, cols = A[:, keep], c[keep], lx[keep], ux[keep], integer[keep], cols[keep]
            log["empty_cols"] += len(fixable)
            changed = True
        A = A.tocsr()
        if not changed:
            break

    red = Problem(c=c, A=A, lc=lc, uc=uc, lx=lx, ux=ux, c0=c0, obj_sign=prob.obj_sign,
                  name=prob.name + " (presolved)",
                  row_names=[prob.row_names[i] for i in rows] if prob.row_names else None,
                  col_names=[prob.col_names[j] for j in cols] if prob.col_names else None,
                  integer=integer if prob.integer is not None else None)
    log.update(rows_before=m0, rows_after=red.m, cols_before=n0, cols_after=red.n)
    return Presolved(red, cols, rows, fixed, "reduced", log)


def _implied(A, lc, uc, lx, ux, mn, mx, mn_inf, mx_inf):
    """Per-column implied bounds (tightest over all rows) from each row's activity range."""
    A = A.tocsr()
    n = len(lx)
    r = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
    j, a = A.indices, A.data
    lxj, uxj = lx[j], ux[j]
    with np.errstate(invalid="ignore"):
        # contribution of column j to the row's min / max activity
        cmin = np.where(a > 0, lxj, uxj) * a
        cmax = np.where(a > 0, uxj, lxj) * a
        cmin_inf, cmax_inf = ~np.isfinite(cmin), ~np.isfinite(cmax)
        rest_min = np.where(cmin_inf, np.where(mn_inf[r] == 1, mn[r], np.nan),
                            np.where(mn_inf[r] == 0, mn[r] - cmin, np.nan))
        rest_max = np.where(cmax_inf, np.where(mx_inf[r] == 1, mx[r], np.nan),
                            np.where(mx_inf[r] == 0, mx[r] - cmax, np.nan))
        # a_ij x_j <= uc - rest_min   and   a_ij x_j >= lc - rest_max
        hi_act = (uc[r] - rest_min) / a
        lo_act = (lc[r] - rest_max) / a
    new_u = np.where(a > 0, hi_act, lo_act)
    new_l = np.where(a > 0, lo_act, hi_act)
    tu = np.full(n, np.inf)
    tl = np.full(n, -np.inf)
    np.fmin.at(tu, j, np.where(np.isfinite(new_u), new_u, np.inf))
    np.fmax.at(tl, j, np.where(np.isfinite(new_l), new_l, -np.inf))
    return tl, tu


def _tighten_integers(A, lc, uc, lx, ux, integer, mn, mx, mn_inf, mx_inf) -> int:
    """Implied bounds on integer columns from each row's activity range (vectorised)."""
    if not integer.any():
        return 0
    tl, tu = _implied(A, lc, uc, lx, ux, mn, mx, mn_inf, mx_inf)
    tu, tl = np.floor(tu + 1e-6), np.ceil(tl - 1e-6)
    up = integer & (tu < ux - 0.5)
    lo = integer & (tl > lx + 0.5)
    ux[up] = tu[up]
    lx[lo] = tl[lo]
    return int(up.sum() + lo.sum())


def implied_bounds(A, lc, uc, lx, ux, integer, rounds=25):
    """Bounds implied by the rows for every column (propagated `rounds` times), on copies: the model's
    own continuous bounds are not changed (they are used for coefficient tightening and bound
    substitution, which stay valid because the implied bounds hold at every feasible point)."""
    lx, ux = lx.astype(float).copy(), ux.astype(float).copy()
    for _ in range(rounds):
        mn, mx, mn_inf, mx_inf = _activity(A, lx, ux)
        tl, tu = _implied(A, lc, uc, lx, ux, mn, mx, mn_inf, mx_inf)
        tu = np.where(integer, np.floor(tu + 1e-6), tu)
        tl = np.where(integer, np.ceil(tl - 1e-6), tl)
        rng = np.where(np.isfinite(ux - lx), ux - lx, np.inf)
        thr = 1e-3 * np.minimum(1.0 + np.abs(np.where(np.isfinite(rng), rng, 0.0)), 1e6)
        up = (tu < ux - thr) & (np.abs(tu) < 1e9)
        lo = (tl > lx + thr) & (np.abs(tl) < 1e9)
        if not (up.any() or lo.any()):
            break
        ux[up] = tu[up]
        lx[lo] = tl[lo]
        if np.any(lx > ux + 1e-6):
            break                                   # infeasible: leave it to the solver to prove
    return lx, ux


def tighten_coefficients(A, lc, uc, lx, ux, integer, lxi, uxi) -> int:
    """Coefficient tightening on binaries (Savelsbergh 1994): in a one-sided row a x <= b whose maximum
    activity M (under the implied bounds lxi/uxi) satisfies
        a_k > 0 and M - a_k < b :  d = b - (M - a_k),  a_k -= d,  b -= d
        a_k < 0 and M + a_k < b :  a_k = b - M
    The integer points are unchanged and the LP relaxation gets tighter (big-M reduction). Edits A
    (CSR data), lc/uc in place; returns the number of coefficients changed."""
    binary = integer & (lx >= -1e-9) & (ux <= 1 + 1e-9)
    if not binary.any():
        return 0
    count = 0
    for i in range(A.shape[0]):
        lo_f, up_f = np.isfinite(lc[i]), np.isfinite(uc[i])
        if lo_f == up_f:
            continue                                # equality / ranged / free rows are left alone
        s, e = A.indptr[i], A.indptr[i + 1]
        idx = A.indices[s:e]
        if idx.size < 2 or not binary[idx].any():
            continue
        sg = 1.0 if up_f else -1.0
        a = sg * A.data[s:e]
        b = sg * (uc[i] if up_f else lc[i])
        hi = np.where(a > 0, uxi[idx], lxi[idx]) * a
        if not np.all(np.isfinite(hi)):
            continue
        M = float(hi.sum())
        changed = False
        for q in np.flatnonzero(binary[idx]):
            ak = a[q]
            tol = 1e-6 * (1 + abs(b) + abs(ak))
            if ak > 0 and M - ak < b - tol:
                d = b - (M - ak)
                a[q] = ak - d
                b -= d
                M -= d
                changed = True
            elif ak < 0 and M + ak < b - tol:
                a[q] = b - M
                changed = True
        if changed:
            a[np.abs(a) < 1e-9 * (1 + np.abs(a).max())] = 0.0
            A.data[s:e] = sg * a
            if up_f:
                uc[i] = b
            else:
                lc[i] = -b
            count += 1
    return count
