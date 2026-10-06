"""Diagonal preconditioning (host, once) — SPEC §2.1, order as in cuPDLPx
(src/preconditioner.cu, rescale_problem):

  1. geometric-mean scaling, 12 passes (row/col divided by sqrt(min*max) of |a_ij|)
  2. Ruiz equilibration, 10 passes (row/col divided by sqrt(max |a_ij|))
  3. Pock-Chambolle, alpha = 1 (row/col divided by sqrt(sum |a_ij|))
  4. bound/objective rescaling: bounds * 1/(||bhat||+1), objective * 1/(||c||+1)

Scaled problem: K = diag(row) A diag(col),  x = beta_b * col * x_s,  y = beta_c * row * y_s.
c, bounds may be 1-D or batched (n, S) / (m, S); row/col depend on A only, beta_b/beta_c
are per scenario (shape (S,) or scalar).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

GEO_MULT_MAX = 1e20


@dataclass
class Scaling:
    row: np.ndarray        # (m,)
    col: np.ndarray        # (n,)
    beta_b: np.ndarray     # () or (S,)  primal/bound scale
    beta_c: np.ndarray     # () or (S,)  objective/dual scale


class _Mat:
    """CSR values with a precomputed CSR->CSC permutation for column reductions."""

    def __init__(self, A: sp.csr_matrix):
        self.m, self.n = A.shape
        self.indptr = A.indptr
        self.cols = A.indices
        self.rows = np.repeat(np.arange(self.m, dtype=np.int64), np.diff(A.indptr))
        self.vals = A.data.astype(np.float64).copy()
        idx = sp.csr_matrix((np.arange(A.nnz, dtype=np.float64), A.indices, A.indptr), shape=A.shape)
        csc = idx.tocsc()
        self.perm = csc.data.astype(np.int64)
        self.cindptr = csc.indptr

    def _reduce(self, ufunc, v, indptr, count, empty):
        out = np.full(count, empty)
        if v.size == 0:
            return out
        nonempty = np.diff(indptr) > 0
        starts = indptr[:-1][nonempty]
        out[nonempty] = ufunc.reduceat(v, starts)
        return out

    def row_reduce(self, ufunc, v, empty=0.0):
        return self._reduce(ufunc, v, self.indptr, self.m, empty)

    def col_reduce(self, ufunc, v, empty=0.0):
        return self._reduce(ufunc, v[self.perm], self.cindptr, self.n, empty)

    def scale(self, r, s):
        """K <- diag(1/r) K diag(1/s)."""
        self.vals /= r[self.rows] * s[self.cols]


def _safe_sqrt(v):
    out = np.sqrt(v)
    out[~(out > 0) | ~np.isfinite(out)] = 1.0
    return out


def precondition(A, c, lc, uc, lx, ux, geo_iters=12, ruiz_iters=10, pc_alpha=1.0,
                 bound_obj=True):
    K, row, col = matrix_scaling(A, geo_iters, ruiz_iters, pc_alpha)
    return (K,) + scale_vectors(row, col, c, lc, uc, lx, ux, bound_obj)


def matrix_scaling(A, geo_iters=12, ruiz_iters=10, pc_alpha=1.0):
    """Steps 1-3 (they depend on A only): K = diag(row) A diag(col). Reusable across
    problems that share A, such as the correction problems of iterative refinement."""
    M = _Mat(sp.csr_matrix(A))
    div_r = np.ones(M.m)      # cumulative divisors: K = diag(1/div_r) A diag(1/div_c)
    div_c = np.ones(M.n)

    if geo_iters > 0 and M.vals.size:
        a = np.abs(M.vals)
        a[~(a > 0) | ~np.isfinite(a)] = np.nan
        mr, mc = np.ones(M.m), np.ones(M.n)
        for _ in range(geo_iters):
            t = a * mc[M.cols]
            lo = M.row_reduce(np.fmin, t, np.nan)
            hi = M.row_reduce(np.fmax, t, np.nan)
            ok = hi > 0
            mr[ok] = np.clip(1.0 / (np.sqrt(lo[ok]) * np.sqrt(hi[ok])), 1 / GEO_MULT_MAX, GEO_MULT_MAX)
            t = a * mr[M.rows]
            lo = M.col_reduce(np.fmin, t, np.nan)
            hi = M.col_reduce(np.fmax, t, np.nan)
            ok = hi > 0
            mc[ok] = np.clip(1.0 / (np.sqrt(lo[ok]) * np.sqrt(hi[ok])), 1 / GEO_MULT_MAX, GEO_MULT_MAX)
        r, s = 1.0 / mr, 1.0 / mc
        M.scale(r, s)
        div_r *= r
        div_c *= s

    for _ in range(ruiz_iters):
        a = np.abs(M.vals)
        r = _safe_sqrt(M.row_reduce(np.maximum, a))
        s = _safe_sqrt(M.col_reduce(np.maximum, a))
        M.scale(r, s)
        div_r *= r
        div_c *= s

    if pc_alpha is not None:
        a = np.abs(M.vals)
        r = _safe_sqrt(M.row_reduce(np.add, a ** pc_alpha))
        s = _safe_sqrt(M.col_reduce(np.add, a ** (2.0 - pc_alpha)))
        M.scale(r, s)
        div_r *= r
        div_c *= s

    K = sp.csr_matrix((M.vals, M.cols, M.indptr), shape=(M.m, M.n))
    return K, 1.0 / div_r, 1.0 / div_c


def scale_vectors(row, col, c, lc, uc, lx, ux, bound_obj=True):
    """Step 4 and the vector side of steps 1-3."""
    rb = row if np.ndim(lc) == 1 else row[:, None]
    cb = col if np.ndim(c) == 1 else col[:, None]
    c_s = c * cb
    lx_s, ux_s = lx / (col if np.ndim(lx) == 1 else col[:, None]), ux / (col if np.ndim(ux) == 1 else col[:, None])
    lc_s, uc_s = lc * rb, uc * rb

    if bound_obj:
        lf, uf = np.isfinite(lc_s), np.isfinite(uc_s)
        # cuPDLPx compute_bound_contrib_kernel: equality rows counted once
        acc = np.where(lf & (~uf | (np.abs(lc_s - uc_s) > 1e-12)), np.where(lf, lc_s, 0.0) ** 2, 0.0)
        acc = acc + np.where(uf, np.where(uf, uc_s, 0.0) ** 2, 0.0)
        beta_b = np.sqrt(np.sum(acc, axis=0)) + 1.0
        beta_c = np.linalg.norm(c_s, axis=0) + 1.0
    else:
        beta_b = np.ones(np.shape(c_s)[1:]) if np.ndim(c_s) == 2 else np.array(1.0)
        beta_c = beta_b.copy()
    lc_s, uc_s, lx_s, ux_s = lc_s / beta_b, uc_s / beta_b, lx_s / beta_b, ux_s / beta_b
    c_s = c_s / beta_c
    return c_s, lc_s, uc_s, lx_s, ux_s, Scaling(row, col, np.asarray(beta_b), np.asarray(beta_c))
