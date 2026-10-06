"""Feasibility Jump: an LP-free primal heuristic for MILP (Luteberget & Sartor, Math. Programming
Computation 2023; the heuristic NVIDIA cuOpt runs on the GPU, Cordduk et al., NeurIPS 2025).

State: a point x within its bounds (integers integral). Violation of row i:
    v_i(x) = max(lc_i - a_i x, a_i x - uc_i, 0),     weighted objective  W(x) = sum_i w_i v_i(x)
Move: pick a violated row; for each column j in it, the "jump" value is the value of x_j (integral
for integers, within bounds) that minimises W along that coordinate. The candidates are the points
where some row of column j becomes exactly satisfied (the breakpoints of a piecewise-linear 1-D
function), so each column is evaluated exactly over its own rows. The best-scoring move is taken;
if no move lowers W, the weights of the violated rows grow (the Lagrangian step of the paper).
A feasible point is returned for the caller to polish (fix the integers, solve the LP for the
continuous part) and to check on the original model.
"""
from __future__ import annotations

import time

import numpy as np


def _viol(act, lc, uc):
    return np.maximum(np.maximum(lc - act, act - uc), 0.0)


def feasibility_jump(prob, integer, lx, ux, x0=None, time_limit=5.0, max_steps=200_000, seed=0, tol=1e-6):
    """Return a point satisfying rows and bounds (integers integral) or None."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    A = prob.A.tocsr()
    AC = A.tocsc()
    m, n = A.shape
    lc, uc = prob.lc, prob.uc
    lo = np.where(np.isfinite(lx), lx, -1e9)
    hi = np.where(np.isfinite(ux), ux, 1e9)
    if x0 is None:
        x = np.where(np.isfinite(lx), lx, np.where(np.isfinite(ux), ux, 0.0))
    else:
        x = np.clip(np.asarray(x0, dtype=float), lo, hi)
    x[integer] = np.clip(np.round(x[integer]), np.ceil(lo[integer] - 1e-9), np.floor(hi[integer] + 1e-9))
    act = A @ x
    w = np.ones(m)
    scale = 1.0 + np.maximum(np.abs(np.where(np.isfinite(lc), lc, 0)), np.abs(np.where(np.isfinite(uc), uc, 0)))
    tolr = tol * scale
    for step in range(max_steps):
        v = _viol(act, lc, uc)
        bad = np.flatnonzero(v > tolr)
        if bad.size == 0:
            return x
        if (step & 63) == 0 and time.perf_counter() - t0 > time_limit:
            return None
        r = int(bad[rng.integers(bad.size)])
        cols = A.indices[A.indptr[r]:A.indptr[r + 1]]
        best = (0.0, -1, 0.0)
        for j in cols:
            s, e = AC.indptr[j], AC.indptr[j + 1]
            rows, a = AC.indices[s:e], AC.data[s:e]
            ar = act[rows]
            base = float(w[rows] @ _viol(ar, lc[rows], uc[rows]))
            # breakpoints: values of x_j at which a row of column j becomes exactly satisfied
            with np.errstate(divide="ignore", invalid="ignore"):
                cand = np.concatenate([x[j] + (lc[rows] - ar) / a, x[j] + (uc[rows] - ar) / a])
            cand = cand[np.isfinite(cand)]
            if integer[j]:
                cand = np.concatenate([np.floor(cand), np.ceil(cand)])
            cand = np.unique(np.clip(cand, lo[j], hi[j]))
            cand = cand[np.abs(cand - x[j]) > 1e-12]
            if cand.size == 0:
                continue
            d = cand - x[j]                                          # (k,)
            newact = ar[None, :] + d[:, None] * a[None, :]           # (k, rows)
            cost = _viol(newact, lc[rows][None, :], uc[rows][None, :]) @ w[rows]
            k = int(np.argmin(cost))
            gain = base - float(cost[k])
            if gain > best[0] + 1e-12:
                best = (gain, int(j), float(cand[k]))
        gain, j, val = best
        if j < 0:
            w[bad] += 1.0                                             # no improving move: reweight
            continue
        s, e = AC.indptr[j], AC.indptr[j + 1]
        act[AC.indices[s:e]] += (val - x[j]) * AC.data[s:e]
        x[j] = val
        if (step & 1023) == 1023:                                    # refresh the activities (drift)
            act = A @ x
    return None
