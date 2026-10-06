"""Indigenous Revised Primal/Dual Simplex (bounded-variable form).

Implements the revised simplex method for the internal LP form

    minimise    c'x            (obj_sign restores the original max/min sense)
    subject to  lc <= A x <= uc
                lx <=  x  <= ux

by working on the equivalent *equality* standard form obtained with one logical
(slack) variable per row:

    minimise    [c 0]'[x s]
    subject to  [A -I] [x s] = 0
                [lx lc] <= [x s] <= [ux uc]

The logical variable s_i equals the row activity (A x)_i, so bounding s_i inside
[lc_i, uc_i] enforces the original row.  This turns two-sided row ranges, equality
rows and free rows into ordinary bounded variables and makes the slack basis
B = -I a trivially invertible starting basis.

Algorithm (textbook bounded-variable revised simplex, cf. Maros, *Computational
Techniques of the Simplex Method*, 2003; Vanderbei, *Linear Programming*, 4th ed.):

  * Basis inverse maintained implicitly through an LU factorisation of B
    (scipy.linalg.lu_factor / lu_solve -- a numerical primitive, like LAPACK,
    not an optimisation routine).  B is refactorised after every basis change.
  * Phase I minimises the sum of primal-bound infeasibilities of the basic
    variables with a composite (+/-1) objective until a feasible basis is found.
  * Phase II runs Dantzig pricing with a Harris two-pass ratio test, bound
    flips for the entering variable, and Bland's rule as an anti-cycling
    fallback when consecutive degenerate pivots are detected.
  * Dual simplex (dual=True) is used when the starting slack basis is already
    dual feasible; otherwise the primal two-phase method is used.  Both return
    identical optima and a valid basis.

Row duals are recovered as y = c_B' B^{-1}: the reduced cost of the logical
variable s_i equals y_i, exactly the sign convention used by solver/terminate.py
and qenivo.certify.

Scope: the basis is refactorised densely after each pivot (O(m^3)), so this engine is the
exact vertex engine for small and medium models (up to a few thousand rows); large models go
to the first-order engines. A sparse LU with Forrest-Tomlin updates is the planned
replacement (docs/ROADMAP.md).

Imports: numpy, scipy.sparse, scipy.linalg only.  No highspy, no scipy.optimize.
"""
from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from ..model import Problem as LPProblem

# Variable status codes.
_BASIC = 0
_ATLOW = 1
_ATUPP = 2
_FREE = 3   # nonbasic free variable held at value 0

_STATUS_NAME = {_BASIC: "basic", _ATLOW: "at_lower", _ATUPP: "at_upper", _FREE: "free"}


class _BoundedSimplex:
    """Bounded-variable revised simplex on the [A -I] equality standard form."""

    def __init__(self, prob: LPProblem, tol: float, time_limit: float):
        self.tol = tol
        self.feas_tol = max(tol, 1e-9)
        self.time_limit = time_limit
        self.t0 = time.perf_counter()

        m, n = prob.m, prob.n
        self.m = m
        self.n = n
        self.N = n + m  # total structural + logical

        # Standard form matrix [A | -I] in CSC for fast column access.
        A = prob.A.tocsc()
        eyeneg = sp.csc_matrix(-sp.eye(m))
        self.Astd = sp.hstack([A, eyeneg], format="csc")
        self.Astd_csr = self.Astd.tocsr()
        self.AstdT = self.Astd.T.tocsr()

        self.cost = np.concatenate([prob.c.astype(float), np.zeros(m)])
        self.lo = np.concatenate([prob.lx.astype(float), prob.lc.astype(float)])
        self.hi = np.concatenate([prob.ux.astype(float), prob.uc.astype(float)])

        # Status of every variable, and the basis list.
        self.vstat = np.empty(self.N, dtype=np.int8)
        # Start: structural + logical variables all nonbasic; logical slacks basic.
        # Nonbasic structural go to a finite bound (lower if finite else upper else free@0).
        for j in range(self.N):
            self.vstat[j] = self._pick_nonbasic_bound(j)
        self.basis = np.arange(n, n + m)          # logical variables form initial basis
        self.vstat[self.basis] = _BASIC
        self.in_basis = np.zeros(self.N, dtype=bool)
        self.in_basis[self.basis] = True

        self.lu = None
        self.z = np.zeros(self.N)
        self.iters = 0
        self.max_iter = 2000 + 40 * (m + n)

    # ------------------------------------------------------------------ helpers
    def _pick_nonbasic_bound(self, j: int) -> int:
        lo, hi = self.lo[j], self.hi[j]
        if np.isfinite(lo):
            return _ATLOW
        if np.isfinite(hi):
            return _ATUPP
        return _FREE

    def _nb_value(self, j: int) -> float:
        s = self.vstat[j]
        if s == _ATLOW:
            return self.lo[j]
        if s == _ATUPP:
            return self.hi[j]
        return 0.0

    def _timed_out(self) -> bool:
        return time.perf_counter() - self.t0 > self.time_limit

    # Basis inverse in product form (Dantzig & Orchard-Hays): B_k^-1 = E_k^-1 ... E_1^-1 B_0^-1.
    # B_0 is factorised by LU; each pivot appends one eta column instead of refactorising, and
    # the basis is refactorised every REFACTOR pivots or when a pivot element is small.
    REFACTOR = 64

    def _factorize(self):
        """Factorise B = [A_S, -I_R] by eliminating the basic slacks exactly: only the square block
        K = A[N, S] (N = rows whose slack is not basic) is LU-factorised, not the whole m x m basis."""
        pos = np.arange(self.m)
        is_slack = self.basis >= self.n
        self.pS, self.pR = pos[~is_slack], pos[is_slack]              # basis positions
        self.cS = self.basis[~is_slack]                                 # structural columns
        self.rR = self.basis[is_slack] - self.n                         # rows with a basic slack
        mask = np.ones(self.m, dtype=bool)
        mask[self.rR] = False
        self.rN = np.flatnonzero(mask)                                  # rows without one
        self.lu = None
        if self.cS.size:
            A = self.Astd_csr[:, :self.n]
            K = A[self.rN][:, self.cS].toarray()
            self.ARS = A[self.rR][:, self.cS].tocsr()
            try:
                self.lu = la.lu_factor(K)
            except (la.LinAlgError, ValueError):
                self.lu = la.lu_factor(K + 1e-10 * np.eye(K.shape[0]))
        self.etas = []
        self.refactors = getattr(self, "refactors", 0) + 1
        self.factor_size = int(self.cS.size)

    def _base_ftran(self, b):
        """Solve B z = b (z by basis position)."""
        z = np.empty(self.m)
        if self.cS.size:
            zS = la.lu_solve(self.lu, b[self.rN])
            z[self.pS] = zS
            z[self.pR] = self.ARS @ zS - b[self.rR]
        else:
            z[self.pR] = -b[self.rR]
        return z

    def _base_btran(self, c):
        """Solve B' y = c (c by basis position, y by row)."""
        y = np.empty(self.m)
        yR = -c[self.pR]
        y[self.rR] = yR
        if self.cS.size:
            y[self.rN] = la.lu_solve(self.lu, c[self.pS] - self.ARS.T @ yR, trans=1)
        return y

    def _update(self, r: int, alpha: np.ndarray):
        """Basis change at position r with entering column alpha = B^-1 a_q: append an eta."""
        piv = alpha[r]
        if len(self.etas) >= self.REFACTOR or abs(piv) < 1e-9 * max(1.0, float(np.max(np.abs(alpha)))):
            self._factorize()
            return
        e = -alpha / piv
        e[r] = 1.0 / piv - 1.0                  # stored as (eta - e_r)
        self.etas.append((r, e))

    def _ftran(self, rhs: np.ndarray) -> np.ndarray:
        w = self._base_ftran(np.asarray(rhs, dtype=float))
        for r, e in self.etas:
            wr = w[r]
            if wr != 0.0:
                w = w + e * wr
        return w

    def _btran(self, rhs: np.ndarray) -> np.ndarray:
        v = np.array(rhs, dtype=float, copy=True)
        for r, e in reversed(self.etas):
            v[r] = v[r] + e @ v
        return self._base_btran(v)

    def _compute_basic_values(self):
        # z_nonbasic at their bound values; basic solved from B x_B = -N v_N.
        nb = ~self.in_basis
        v = np.where(nb & (self.vstat == _ATLOW), self.lo, 0.0)
        v = np.where(nb & (self.vstat == _ATUPP), self.hi, v)
        v[~np.isfinite(v)] = 0.0
        rhs = -(self.Astd @ v)  # since basic entries of v are 0
        xB = self._ftran(rhs)
        self.z = v
        self.z[self.basis] = xB

    def _column(self, j: int) -> np.ndarray:
        return self.Astd[:, j].toarray().ravel()

    # ------------------------------------------------------------------ pricing
    def _duals(self, cost: np.ndarray) -> np.ndarray:
        cB = cost[self.basis]
        return self._btran(cB)

    def _reduced_costs(self, cost: np.ndarray, pi: np.ndarray, cols: np.ndarray) -> np.ndarray:
        # d_j = cost_j - pi' Astd[:,j] for the given nonbasic columns.
        sub = self.Astd[:, cols]
        return cost[cols] - (sub.T @ pi)

    # ------------------------------------------------------------------ phase I
    def _phase1_costs(self) -> Tuple[np.ndarray, float]:
        """Composite objective gradient w over basic vars, and total infeasibility."""
        w = np.zeros(self.N)
        xB = self.z[self.basis]
        loB, hiB = self.lo[self.basis], self.hi[self.basis]
        below = xB < loB - self.feas_tol
        above = xB > hiB + self.feas_tol
        w[self.basis[below]] = -1.0   # want to increase these
        w[self.basis[above]] = 1.0    # want to decrease these
        infeas = float(np.sum(np.where(below, loB - xB, 0.0)) + np.sum(np.where(above, xB - hiB, 0.0)))
        return w, infeas

    def _run(self, cost: np.ndarray, phase1: bool, bland_after: int = 50) -> str:
        """One primal simplex loop with the given cost vector.

        Returns 'optimal' (no improving entering) / 'unbounded' / 'time_limit'.
        In phase1 the cost is recomputed each iteration from the infeasibility.
        """
        degenerate_run = 0
        while True:
            if self.iters >= self.max_iter:
                return "iteration_limit"
            if (self.iters & 15) == 0 and self._timed_out():
                return "time_limit"
            self.iters += 1

            if phase1:
                cost, infeas = self._phase1_costs()
                if infeas <= self.feas_tol * (1.0 + self.m):
                    return "feasible"

            pi = self._duals(cost)
            nb_mask = ~self.in_basis
            nb_cols = np.where(nb_mask)[0]
            if nb_cols.size == 0:
                return "optimal"
            d = self._reduced_costs(cost, pi, nb_cols)

            st = self.vstat[nb_cols]
            # improving if (can increase and d<0) or (can decrease and d>0)
            can_inc = (st == _ATLOW) | (st == _FREE)
            can_dec = (st == _ATUPP) | (st == _FREE)
            improve_inc = can_inc & (d < -self.tol)
            improve_dec = can_dec & (d > self.tol)
            improving = improve_inc | improve_dec
            if not np.any(improving):
                return "optimal"

            use_bland = degenerate_run >= bland_after
            if use_bland:
                cand = np.where(improving)[0]
                pick = cand[np.argmin(nb_cols[cand])]
            else:
                score = np.where(improve_inc, -d, 0.0) + np.where(improve_dec, d, 0.0)
                pick = int(np.argmax(score))
            q = int(nb_cols[pick])
            direction = 1.0 if (improve_inc[pick]) else -1.0

            alpha = self._ftran(self._column(q))
            gamma = alpha * direction  # dz_B/dt = -gamma ; xB_i(t) = xB_i - gamma_i t

            t, leave_pos, leave_to = self._ratio_test(q, direction, gamma, phase1)

            if t is None:
                return "unbounded"

            # Apply the step.
            xB = self.z[self.basis]
            self.z[self.basis] = xB - gamma * t
            self.z[q] = self._nb_value(q) + direction * t

            if leave_pos is None:
                # Bound flip of the entering variable: stays nonbasic at other bound.
                self.vstat[q] = _ATUPP if direction > 0 else _ATLOW
                degenerate_run = 0
                continue

            # Basis change: q enters, leaving variable exits to a bound.
            leaving = int(self.basis[leave_pos])
            self.basis[leave_pos] = q
            self.in_basis[q] = True
            self.in_basis[leaving] = False
            self.vstat[q] = _BASIC
            self.vstat[leaving] = leave_to
            # Snap the leaving variable exactly onto its bound.
            self.z[leaving] = self.lo[leaving] if leave_to == _ATLOW else self.hi[leaving]
            self._update(leave_pos, alpha)

            degenerate_run = degenerate_run + 1 if t <= self.feas_tol else 0

    def _ratio_test(self, q: int, direction: float, gamma: np.ndarray, phase1: bool):
        """Return (t, leave_pos, leave_status). leave_pos None => bound flip."""
        big = np.inf
        # Entering variable's own bound flip range.
        t_flip = big
        lo_q, hi_q = self.lo[q], self.hi[q]
        if np.isfinite(lo_q) and np.isfinite(hi_q):
            t_flip = hi_q - lo_q

        xB = self.z[self.basis]
        loB = self.lo[self.basis]
        hiB = self.hi[self.basis]

        # For each basic i: xB_i(t) = xB_i - gamma_i t.
        t_cand = np.full(self.m, big)
        to_status = np.empty(self.m, dtype=np.int8)

        eps = self.feas_tol
        if phase1:
            # Feasible basics limit at their nearest bound; infeasible basics
            # limit when they *reach* their violated bound (become feasible).
            below = xB < loB - eps
            above = xB > hiB + eps
            feas = ~(below | above)
            with np.errstate(divide="ignore", invalid="ignore"):
                # feasible & decreasing toward lower
                dec = feas & (gamma > eps) & np.isfinite(loB)
                t_cand = np.where(dec, (xB - loB) / gamma, t_cand)
                to_status = np.where(dec, _ATLOW, 0).astype(np.int8)
                # feasible & increasing toward upper
                inc = feas & (gamma < -eps) & np.isfinite(hiB)
                ti = (xB - hiB) / gamma
                t_cand = np.where(inc & (ti < t_cand), ti, t_cand)
                to_status = np.where(inc & (ti <= t_cand + 0), _ATUPP, to_status).astype(np.int8)
                # infeasible-below moving up (gamma<0) reaches lower bound
                bu = below & (gamma < -eps)
                tb = (xB - loB) / gamma  # positive
                t_cand = np.where(bu & (tb < t_cand), tb, t_cand)
                to_status = np.where(bu & (tb <= t_cand + 0), _ATLOW, to_status).astype(np.int8)
                # infeasible-above moving down (gamma>0) reaches upper bound
                ad = above & (gamma > eps)
                ta = (xB - hiB) / gamma  # positive
                t_cand = np.where(ad & (ta < t_cand), ta, t_cand)
                to_status = np.where(ad & (ta <= t_cand + 0), _ATUPP, to_status).astype(np.int8)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                dec = (gamma > eps) & np.isfinite(loB)
                t_cand = np.where(dec, (xB - loB) / gamma, t_cand)
                to_status = np.where(dec, _ATLOW, 0).astype(np.int8)
                inc = (gamma < -eps) & np.isfinite(hiB)
                ti = (xB - hiB) / gamma
                take = inc & (ti < t_cand)
                t_cand = np.where(take, ti, t_cand)
                to_status = np.where(take, _ATUPP, to_status).astype(np.int8)

        t_cand = np.where(t_cand < 0, 0.0, t_cand)
        t_basic_min = float(np.min(t_cand)) if self.m else big

        # Harris-style second pass: among rows within tol of the min ratio,
        # choose the largest |gamma| for numerical stability.
        t_min = min(t_flip, t_basic_min)
        if not np.isfinite(t_min):
            return None, None, None

        if t_flip <= t_basic_min + eps and np.isfinite(t_flip) and t_flip <= t_basic_min:
            # Entering variable hits its own opposite bound first -> bound flip.
            if t_flip < t_basic_min - eps:
                return t_flip, None, None

        # Candidate leaving rows near the minimum.
        near = np.where(t_cand <= t_basic_min + eps)[0]
        if near.size == 0 or (np.isfinite(t_flip) and t_flip < t_basic_min - eps):
            return t_flip, None, None
        # pick largest pivot magnitude among near-min rows
        best = near[np.argmax(np.abs(gamma[near]))]
        t = float(t_cand[best])
        return t, int(best), int(to_status[best])

    # ------------------------------------------------------------------ dual
    def _is_dual_feasible(self, cost: np.ndarray) -> bool:
        pi = self._duals(cost)
        nb_cols = np.where(~self.in_basis)[0]
        if nb_cols.size == 0:
            return True
        d = self._reduced_costs(cost, pi, nb_cols)
        st = self.vstat[nb_cols]
        at_low = (st == _ATLOW)
        at_upp = (st == _ATUPP)
        free = (st == _FREE)
        ok_low = ~at_low | (d >= -self.tol)
        ok_upp = ~at_upp | (d <= self.tol)
        ok_free = ~free | (np.abs(d) <= 1e-6)
        return bool(np.all(ok_low) and np.all(ok_upp) and np.all(ok_free))

    def _run_dual(self, cost: np.ndarray) -> str:
        """Dual simplex: keep dual feasibility, drive out primal infeasibility.

        Leaving row: largest bound violation. Entering column: Harris two-pass ratio test (the
        largest |alpha| among the ratios within a tolerance of the minimum), which avoids tiny,
        unstable pivots. Degenerate stalls are broken by a small cost perturbation that keeps
        dual feasibility; it is removed at the end and a primal pass cleans up if needed.
        """
        true_cost = cost
        cost = cost.copy()
        perturbed = False
        stall = 0
        last_inf = np.inf
        while True:
            if self.iters >= self.max_iter:
                return "iteration_limit"
            if (self.iters & 15) == 0 and self._timed_out():
                return "time_limit"
            self.iters += 1

            xB = self.z[self.basis]
            if not np.all(np.isfinite(xB)):
                return "iteration_limit"                         # singular basis: caller restarts cold
            loB, hiB = self.lo[self.basis], self.hi[self.basis]
            below = loB - xB
            above = xB - hiB
            infeas = np.maximum.reduce([below, above, np.zeros(self.m)])
            r = int(np.argmax(infeas))
            if infeas[r] <= self.feas_tol:
                break
            tot = float(infeas.sum())
            stall = stall + 1 if tot >= last_inf - 1e-12 * (1 + last_inf) else 0
            last_inf = min(last_inf, tot)
            if stall > 30 and not perturbed:
                cost = self._perturb(cost)
                perturbed = True
                stall = 0

            leaving_low = below[r] > above[r]  # violates lower -> must increase
            er = np.zeros(self.m)
            er[r] = 1.0
            rho = self._btran(er)                       # row r of B^{-1}
            nb_cols = np.flatnonzero(~self.in_basis)
            arow = (self.AstdT @ rho)[nb_cols]          # alpha_rj for nonbasic j
            pi = self._duals(cost)
            d = cost[nb_cols] - (self.AstdT @ pi)[nb_cols]

            st = self.vstat[nb_cols]
            big = np.abs(arow) > self.feas_tol
            lo_ok = st == _ATLOW
            up_ok = st == _ATUPP
            if leaving_low:
                elig = big & ((lo_ok & (arow < 0)) | (up_ok & (arow > 0)))
            else:
                elig = big & ((lo_ok & (arow > 0)) | (up_ok & (arow < 0)))
            if not elig.any():
                return "infeasible"
            aa = np.abs(arow[elig])
            dd = np.abs(d[elig])
            ratio = dd / aa
            fin = np.isfinite(ratio)
            if not fin.any():
                return "iteration_limit"                         # numerical trouble: caller refactors / restarts
            tmax = np.min(((dd + self.tol) / aa)[fin])          # Harris pass 1
            within = np.flatnonzero(fin & (ratio <= tmax))      # pass 2: largest pivot
            if within.size == 0:
                within = np.flatnonzero(fin)
            k = within[np.argmax(aa[within])]
            best = int(np.flatnonzero(elig)[k])

            q = int(nb_cols[best])
            alpha_q = self._ftran(self._column(q))
            leaving = int(self.basis[r])
            self.basis[r] = q
            self.in_basis[q] = True
            self.in_basis[leaving] = False
            self.vstat[q] = _BASIC
            self.vstat[leaving] = _ATLOW if leaving_low else _ATUPP
            self._update(r, alpha_q)
            self._compute_basic_values()
        if perturbed and not self._is_dual_feasible(true_cost):
            status = self._run(true_cost, phase1=False)      # primal feasible now: clean up
            return "optimal" if status == "optimal" else status
        return "optimal"

    def _perturb(self, cost: np.ndarray) -> np.ndarray:
        """Shift nonbasic costs away from zero reduced cost, keeping dual feasibility."""
        rng = np.random.default_rng(self.iters)
        pi = self._duals(cost)
        d = cost - self.AstdT @ pi
        eps = 1e-7 * (1.0 + np.abs(cost)) * (1.0 + rng.random(self.N))
        nb = ~self.in_basis
        out = cost.copy()
        out[nb & (self.vstat == _ATLOW)] += eps[nb & (self.vstat == _ATLOW)]
        out[nb & (self.vstat == _ATUPP)] -= eps[nb & (self.vstat == _ATUPP)]
        del d
        return out

    # ------------------------------------------------------------------ solve
    def set_start(self, col_statuses, row_statuses) -> bool:
        """Start from a given basis (statuses as returned by solve_simplex). False if unusable."""
        code = {v: k for k, v in _STATUS_NAME.items()}
        st = np.array([code.get(v, -1) for v in list(col_statuses) + list(row_statuses)], dtype=np.int8)
        if st.size != self.N or (st < 0).any() or int((st == _BASIC).sum()) != self.m:
            return False
        self.vstat = st
        self.basis = np.flatnonzero(st == _BASIC)
        self.in_basis = st == _BASIC
        return True

    def solve(self, dual: bool, warm: bool = False) -> Tuple[str, np.ndarray, np.ndarray, list, list, int]:
        self._factorize()
        self._compute_basic_values()

        status = None
        if warm:
            xB = self.z[self.basis]
            feasible = bool(np.all(xB >= self.lo[self.basis] - self.feas_tol)
                            and np.all(xB <= self.hi[self.basis] + self.feas_tol))
            if feasible:                       # e.g. only prices changed: phase II from the old vertex
                status = self._run(self.cost, phase1=False)
                if status not in ("optimal", "unbounded"):
                    status = None
                    self._reset_to_slack_basis()
            elif self._is_dual_feasible(self.cost):   # e.g. only right-hand sides changed
                status = self._run_dual(self.cost)
                # "infeasible" here is a dual ray: a row no entering column can repair (Farkas),
                # the common outcome of a branch; it is trusted. Only a stall falls back.
                if status == "iteration_limit":
                    status = None
                    self._reset_to_slack_basis()
        if status is None and dual and self._is_dual_feasible(self.cost):
            status = self._run_dual(self.cost)
            if status in ("iteration_limit", "infeasible"):
                # fall back to robust primal two-phase
                status = None
                self._reset_to_slack_basis()

        if status is None:
            p1 = self._run(self.cost, phase1=True)
            if p1 == "time_limit" or p1 == "iteration_limit":
                status = p1
            else:
                _, infeas = self._phase1_costs()
                if infeas > 1e-6 * (1.0 + self.m):
                    status = "infeasible"
                else:
                    status = self._run(self.cost, phase1=False)

        self._compute_basic_values()
        x = self.z[:self.n].copy()
        pi = self._duals(self.cost)
        y = pi.copy()
        col_statuses = [_STATUS_NAME[int(self.vstat[j])] for j in range(self.n)]
        row_statuses = [_STATUS_NAME[int(self.vstat[self.n + i])] for i in range(self.m)]
        if status == "feasible":
            status = "optimal"
        return status, x, y, col_statuses, row_statuses, self.iters

    def _reset_to_slack_basis(self):
        for j in range(self.N):
            self.vstat[j] = self._pick_nonbasic_bound(j)
        self.basis = np.arange(self.n, self.n + self.m)
        self.in_basis = np.zeros(self.N, dtype=bool)
        self.in_basis[self.basis] = True
        self.vstat[self.basis] = _BASIC
        self._factorize()
        self._compute_basic_values()


def solve_simplex(prob: LPProblem, tol: float = 1e-9, time_limit: float = 3600.0,
                  dual: bool = True, start: Optional[dict] = None) -> dict:
    """Bounded-variable revised simplex: the native C++ core (native/simplex_core.cpp) when it is
    available, else the Python implementation below. Same results contract either way; a native
    run that stops on a numerical problem or the iteration limit is redone in Python."""
    if prob.n and os.environ.get("QENIVO_NATIVE", "1") != "0":
        from .native_simplex import library, solve_simplex_native
        if library() is not None:
            r = solve_simplex_native(prob, tol=tol, time_limit=time_limit, dual=dual, start=start)
            if r["status"] in ("optimal", "infeasible", "unbounded", "time_limit"):
                return r
    out = solve_simplex_python(prob, tol=tol, time_limit=time_limit, dual=dual, start=start)
    out["engine_impl"] = "python"
    return out


def solve_simplex_python(prob: LPProblem, tol: float = 1e-9, time_limit: float = 3600.0,
                         dual: bool = True, start: Optional[dict] = None) -> dict:
    """Bounded-variable revised simplex. Returns a vertex solution with its basis.

    start: a previous result (its col_statuses / row_statuses) to warm-start from, for a model
    that differs only in costs or bounds."""
    t0 = time.perf_counter()
    if prob.n == 0:
        return {"status": "optimal", "x": np.zeros(0), "y": np.zeros(prob.m), "iterations": 0,
                "col_statuses": [], "row_statuses": ["basic"] * prob.m,
                "time": time.perf_counter() - t0}
    eng = _BoundedSimplex(prob, tol, time_limit)
    warm = bool(start) and eng.set_start(start["col_statuses"], start["row_statuses"])
    status, x, y, col_statuses, row_statuses, iters = eng.solve(dual, warm=warm)
    return {"status": status, "x": x, "y": y, "iterations": int(iters),
            "col_statuses": col_statuses, "row_statuses": row_statuses,
            "time": time.perf_counter() - t0}
