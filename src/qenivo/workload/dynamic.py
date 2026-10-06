"""Dynamic engine: a held optimal basis answers market ticks, Greeks, stalled-case hand-off, speculation.

Standard form (as engines/simplex.py): z = (x, s), [A -I] z = 0, lo <= z <= hi with lo = (lx, lc),
hi = (ux, uc). A basis B (m columns of [A -I]) with nonbasic values v_N at their bounds gives
    z_B = B^-1 (-N v_N),   y = B^-T c_B,   d = c_z - [A -I]' y.
The basis is optimal iff z_B lies inside its bounds and every nonbasic reduced cost has the sign its
bound allows (d >= 0 at lower, d <= 0 at upper, d = 0 free). Both conditions are linear: z_B is
affine in the bounds, (y, d) affine in the costs. The set of data for which B stays optimal is
therefore a polyhedron whose one-dimensional sections are the classical ranging intervals
(Gal, *Postoptimal Analyses, Parametric Programming and Related Topics*, 2nd ed. 1995, ch. 3-4;
Chvatal, *Linear Programming*, 1983, ch. 10; Vanderbei, *Linear Programming*, ch. 7).

PlanTicker. One factorisation of B answers every tick that stays in that polyhedron: a cost tick
needs one BTRAN, a bound tick one FTRAN, and the membership test above is exact (it is ranging along
the tick direction, `ray_interval`). The answer is then scored by the float64 KKT test on the new
data (certify.kkt via api._finish); only that check sets the verdict. A tick outside the region, or
an instant answer that fails its check, is re-solved by the simplex warm-started from the held basis.

Basis factor. B is permuted to block lower-triangular form by row-singleton and column-singleton
peeling (Orchard-Hays's triangular crash; Duff, Erisman and Reid, *Direct Methods for Sparse
Matrices*, 1986, ch. 8), the two triangular blocks are solved by level scheduling (Anderson and
Saad, Int. J. High Speed Computing 1(1), 1989) and only the remaining nucleus is factorised by dense
LU (LAPACK getrf through scipy.linalg.lu_factor).

plan_greeks. Along a data ray p(t) = p0 + t r the optimal value phi(t) is piecewise linear (concave
for cost rays, convex for right-hand-side rays). On the segment of one basis its slope is
r_c'x + d_N'r_v (envelope theorem; Gal ch. 4): dphi/dc = x, dphi/d(row limit) = y, dphi/d(column
bound) = reduced cost. At a breakpoint the derivative does not exist; the left and right slopes come
from the two adjacent bases and gamma is the jump between them (a point mass of the second
derivative). Breakpoints are located exactly by the ratio test of the held basis (Gass and Saaty,
Naval Res. Logist. Q. 2, 1955).

Stall hand-off. First-order lanes of a case stack that stop at 1e-2..1e-3 are finished by crossover
(engines/crossover.py) warm-started from the basis of the nearest case already holding an exact
vertex (nearest in data distance), in nearest-neighbour order.

Speculation. Likely next scenarios are solved ahead of time (background thread or idle-time queue)
and stored under a fingerprint of their data; a stored answer is reused only when the requested data
are bit-identical, and it is re-checked before it is returned.

Imports: NumPy, scipy.sparse arrays, scipy.linalg dense LU. No solver library.
"""
from __future__ import annotations

import copy
import hashlib
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from ..api import Solution, _finish
from ..engines.simplex import solve_simplex
from ..model import Problem

_CODE = {"basic": 0, "at_lower": 1, "at_upper": 2, "free": 3}
_NAME = {v: k for k, v in _CODE.items()}


# =============================================================================== basis factor
class SingularBasis(RuntimeError):
    pass


class _Tri:
    """Sparse lower-triangular L (CSR, nonzero diagonal) solved by level scheduling."""

    def __init__(self, L: sp.csr_matrix):
        k = L.shape[0]
        self.k = k
        self.diag = L.diagonal().astype(np.float64)
        if k and np.any(self.diag == 0.0):
            raise SingularBasis("zero pivot in a triangular block")
        S = sp.tril(L, k=-1, format="csr")
        S.eliminate_zeros()
        self.fwd = self._levels(S, reverse=False)
        self.bwd = self._levels(S.T.tocsr(), reverse=True)

    @staticmethod
    def _levels(S: sp.csr_matrix, reverse: bool):
        k = S.shape[0]
        lev = np.zeros(k, dtype=np.int64)
        ip, ix = S.indptr, S.indices
        rng = range(k - 1, -1, -1) if reverse else range(k)
        for i in rng:
            a, b = ip[i], ip[i + 1]
            if b > a:
                lev[i] = lev[ix[a:b]].max() + 1
        out = []
        if k:
            order = np.argsort(lev, kind="stable")
            cuts = np.flatnonzero(np.diff(lev[order])) + 1
            for idx in np.split(order, cuts):
                out.append((idx, S[idx]))
        return out

    def _run(self, levels, b):
        z = np.zeros(self.k)
        for idx, Sl in levels:
            z[idx] = (b[idx] - Sl @ z) / self.diag[idx]
        return z

    def solve(self, b):
        return self._run(self.fwd, b)

    def solve_t(self, c):
        return self._run(self.bwd, c)


class BasisFactor:
    """Factorisation of B = [A -I][:, basis] (columns by basis position) for repeated solves.

    P B Q = [[L1, 0, 0], [B21, K, 0], [B31, X, L3]] with L1 from row singletons, L3 from column
    singletons (both lower triangular) and K the nucleus (dense LU).
    """

    def __init__(self, Az: sp.csc_matrix, basis: np.ndarray, check: bool = True):
        t0 = time.perf_counter()
        m = Az.shape[0]
        basis = np.asarray(basis, dtype=np.int64)
        if basis.size != m:
            raise SingularBasis(f"basis has {basis.size} columns, expected {m}")
        self.m = m
        self.basis = basis
        B = Az[:, basis].tocsc()
        B.eliminate_zeros()
        p1, q1, p3, q3, nuc_r, nuc_c = self._peel(B)
        P = np.array(p1 + nuc_r + p3[::-1], dtype=np.int64)
        Q = np.array(q1 + nuc_c + q3[::-1], dtype=np.int64)
        self.P, self.Q = P, Q
        k1, k2 = len(p1), len(nuc_r)
        self.k1, self.k12 = k1, k1 + k2
        Bp = B.tocsr()[P][:, Q].tocsr()
        self.L1 = _Tri(Bp[:k1, :k1])
        self.B21 = Bp[k1:self.k12, :k1].tocsr()
        self.B21T = self.B21.T.tocsr()
        self.lu = None
        if k2:
            K = Bp[k1:self.k12, k1:self.k12].toarray()
            self.lu = la.lu_factor(K, check_finite=False)
            piv = np.abs(np.diag(self.lu[0]))
            if not np.all(np.isfinite(piv)) or piv.min() <= 1e-13 * max(1.0, piv.max()):
                raise SingularBasis("singular nucleus")
        self.B3 = Bp[self.k12:, :self.k12].tocsr()
        self.B3T = self.B3.T.tocsr()
        self.L3 = _Tri(Bp[self.k12:, self.k12:])
        self.nucleus = k2
        self.levels = (len(self.L1.fwd), len(self.L3.fwd))
        if check:
            rng = np.random.default_rng(0)
            b = rng.standard_normal(m)
            r = B @ self.ftran(b) - b
            if not np.all(np.isfinite(r)) or np.linalg.norm(r) > 1e-8 * (1.0 + np.linalg.norm(b)):
                raise SingularBasis("basis factor failed its residual check")
        self.time = time.perf_counter() - t0

    @staticmethod
    def _peel(B: sp.csc_matrix):
        m = B.shape[0]
        R = B.tocsr()
        Cp, Ci, Rp, Ri = B.indptr, B.indices, R.indptr, R.indices
        ra = np.ones(m, dtype=bool)
        ca = np.ones(m, dtype=bool)
        rc = np.diff(Rp).astype(np.int64)
        if np.any(rc == 0) or np.any(np.diff(Cp) == 0):
            raise SingularBasis("empty row or column in the basis")
        p1, q1 = [], []
        stack = list(np.flatnonzero(rc == 1))
        while stack:
            i = stack.pop()
            if not ra[i] or rc[i] != 1:
                continue
            cols = Ri[Rp[i]:Rp[i + 1]]
            j = int(cols[ca[cols]][0])
            p1.append(int(i))
            q1.append(j)
            ra[i] = False
            ca[j] = False
            for r in Ci[Cp[j]:Cp[j + 1]]:
                if ra[r]:
                    rc[r] -= 1
                    if rc[r] == 1:
                        stack.append(r)
                    elif rc[r] == 0:
                        raise SingularBasis("structurally singular basis")
        cc = np.zeros(m, dtype=np.int64)
        for j in np.flatnonzero(ca):
            cc[j] = int(ra[Ci[Cp[j]:Cp[j + 1]]].sum())
        p3, q3 = [], []
        stack = list(np.flatnonzero(ca & (cc == 1)))
        while stack:
            j = stack.pop()
            if not ca[j] or cc[j] != 1:
                continue
            rows = Ci[Cp[j]:Cp[j + 1]]
            i = int(rows[ra[rows]][0])
            p3.append(i)
            q3.append(int(j))
            ra[i] = False
            ca[j] = False
            for q in Ri[Rp[i]:Rp[i + 1]]:
                if ca[q]:
                    cc[q] -= 1
                    if cc[q] == 1:
                        stack.append(q)
        nuc_r, nuc_c = [int(v) for v in np.flatnonzero(ra)], [int(v) for v in np.flatnonzero(ca)]
        if len(nuc_r) != len(nuc_c):
            raise SingularBasis("structurally singular basis")
        return p1, q1, p3, q3, nuc_r, nuc_c

    def ftran(self, b: np.ndarray) -> np.ndarray:
        """Solve B z = b (b by row, z by basis position)."""
        bp = np.asarray(b, dtype=np.float64)[self.P]
        k1, k12 = self.k1, self.k12
        z1 = self.L1.solve(bp[:k1])
        z2 = bp[k1:k12] - self.B21 @ z1
        if self.lu is not None:
            z2 = la.lu_solve(self.lu, z2, check_finite=False)
        z12 = np.concatenate([z1, z2])
        z3 = self.L3.solve(bp[k12:] - self.B3 @ z12)
        out = np.empty(self.m)
        out[self.Q] = np.concatenate([z12, z3])
        return out

    def btran(self, c: np.ndarray) -> np.ndarray:
        """Solve B' y = c (c by basis position, y by row)."""
        cq = np.asarray(c, dtype=np.float64)[self.Q]
        k1, k12 = self.k1, self.k12
        y3 = self.L3.solve_t(cq[k12:])
        r12 = cq[:k12] - self.B3T @ y3
        y2 = r12[k1:]
        if self.lu is not None:
            y2 = la.lu_solve(self.lu, y2, trans=1, check_finite=False)
        y1 = self.L1.solve_t(r12[:k1] - self.B21T @ y2)
        out = np.empty(self.m)
        out[self.P] = np.concatenate([y1, y2, y3])
        return out


# =============================================================================== data and fingerprints
_DATA = ("c", "lc", "uc", "lx", "ux")
_A_TOKEN: dict = {}


def _matrix_token(A) -> bytes:
    key = id(A)
    hit = _A_TOKEN.get(key)
    if hit is not None and hit[0] is A:
        return hit[1]
    A = sp.csr_matrix(A)
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(A.shape, dtype=np.int64).tobytes())
    for arr in (A.indptr, A.indices, A.data):
        h.update(np.ascontiguousarray(arr).tobytes())
    if len(_A_TOKEN) > 16:
        _A_TOKEN.clear()
    _A_TOKEN[key] = (A, h.digest())
    return _A_TOKEN[key][1]


def fingerprint(prob: Problem) -> str:
    """Digest of everything that defines an LP's answer: A, costs, bounds, c0 and sense."""
    h = hashlib.blake2b(digest_size=20)
    h.update(_matrix_token(prob.A))
    for k in _DATA:
        h.update(np.ascontiguousarray(getattr(prob, k), dtype=np.float64).tobytes())
    h.update(np.array([prob.c0, prob.obj_sign], dtype=np.float64).tobytes())
    return h.hexdigest()


def _same_data(a: Problem, b: Problem) -> bool:
    return (a.A is b.A or _matrix_token(a.A) == _matrix_token(b.A)) and a.c0 == b.c0 \
        and a.obj_sign == b.obj_sign and all(np.array_equal(getattr(a, k), getattr(b, k)) for k in _DATA)


def with_data(prob: Problem, **vecs) -> Problem:
    """Shallow copy of `prob` (shares A) with some of c, lc, uc, lx, ux replaced."""
    p = copy.copy(prob)
    for k, v in vecs.items():
        if k not in _DATA:
            raise ValueError(f"unknown data vector {k!r}")
        if v is not None:
            setattr(p, k, np.ascontiguousarray(v, dtype=np.float64))
    p.notes = list(prob.notes)
    return p


def _apply_change(cur: np.ndarray, change) -> np.ndarray:
    """A change is a full vector, a dict {index: value} or a pair (indices, values)."""
    if change is None:
        return cur
    if isinstance(change, dict):
        out = cur.copy()
        for k, v in change.items():
            out[int(k)] = float(v)
        return out
    if isinstance(change, tuple) and len(change) == 2:
        out = cur.copy()
        out[np.asarray(change[0], dtype=np.int64)] = np.asarray(change[1], dtype=np.float64)
        return out
    arr = np.asarray(change, dtype=np.float64)
    if arr.shape != cur.shape:
        raise ValueError(f"change has shape {arr.shape}, expected {cur.shape}")
    return arr.copy()


# =============================================================================== held basis
class _Held:
    """An optimal basis with its factor and the vertex (z, y) on the data it was last checked on.

    The status codes, basic set and factor are derived on first use: a basis adopted after a warm
    solve whose next tick goes straight to another warm solve never needs them (and on refinery L3
    encoding 12k status strings one by one took about 8 ms per tick).
    """

    def __init__(self, prob: Problem, Az: sp.csc_matrix, col_statuses, row_statuses, factor=None):
        n, m = prob.n, prob.m
        col, row = list(col_statuses), list(row_statuses)
        if len(col) != n or len(row) != m:
            raise ValueError("statuses do not match the model")
        if col.count("basic") + row.count("basic") != m:
            raise SingularBasis(f"basis has {col.count('basic') + row.count('basic')} basic columns, expected {m}")
        self._statuses = {"col_statuses": col, "row_statuses": row}
        self._Az, self._factor, self.factor_seconds = Az, factor, 0.0
        self._vst = self._basis = None
        self.n, self.m = n, m

    @property
    def vst(self) -> np.ndarray:
        if self._vst is None:
            names = np.asarray(self._statuses["col_statuses"] + self._statuses["row_statuses"])
            vst = np.full(names.size, -1, dtype=np.int8)
            for name, code in _CODE.items():
                vst[names == name] = code
            if (vst < 0).any():
                raise ValueError(f"unknown basis status {names[vst < 0][0]!r}")
            self._vst = vst
        return self._vst

    @property
    def basis(self) -> np.ndarray:
        if self._basis is None:
            self._basis = np.flatnonzero(self.vst == 0)
        return self._basis

    @property
    def factor(self) -> "BasisFactor":
        """Factorised on first use: a basis reached by a warm solve may never be asked an instant question."""
        if self._factor is None:
            t0 = time.perf_counter()
            self._factor = BasisFactor(self._Az, self.basis)
            self.factor_seconds += time.perf_counter() - t0
        return self._factor

    def statuses(self) -> dict:
        """The basis as solve_simplex reports and accepts it; fresh lists, so a caller cannot alter the held one."""
        return {k: list(v) for k, v in self._statuses.items()}


class PlanTicker:
    """Hold an optimal basis; answer price / bound ticks instantly while it stays optimal.

    Every returned Solution has a verdict from the KKT check on the tick's own data. engine["path"]
    is "instant" (held basis), "speculative" (pre-solved, fingerprint match), "warm" (simplex from
    the held basis) or "cold" (fallback solve). tol is the certification tolerance; feas_tol the
    membership tolerance of the region test (relative to 1 + |bound| and 1 + max|c|).
    """

    PROBE_EVERY = 25          # ticks between tries of the mode not in use
    EWMA = 0.1                # weight of the newest tick in the per-mode latency averages

    def __init__(self, prob: Problem, tol: float = 1e-9, feas_tol: float = 1e-9,
                 time_limit: float = 600.0, start: Optional[dict] = None, speculator=None,
                 adaptive: bool = True):
        if prob.is_qp or prob.is_mip:
            raise ValueError("PlanTicker is for LPs")
        self.base = prob
        self.n, self.m = prob.n, prob.m
        self.tol, self.feas_tol, self.time_limit = float(tol), float(feas_tol), float(time_limit)
        self.Az = sp.hstack([prob.A, -sp.eye(prob.m)], format="csc")
        self.AT = prob.A.T.tocsr()
        self.prob = with_data(prob)
        self.held: Optional[_Held] = None
        self.z: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self._ref: Optional[Problem] = None                 # the data (z, y) were computed on
        self.log: list[dict] = []
        self.counts = {"instant": 0, "speculative": 0, "warm": 0, "cold": 0, "instant_rejected": 0,
                       "not_optimal": 0, "factor_time": 0.0}
        self.speculator = speculator
        self._region_hook: Optional[Callable] = None        # tests only: override the region test
        # Adaptive policy. An instant answer saves a warm solve but costs a region test, and after
        # every basis change a factorisation; when most ticks leave the region that costs more than
        # it saves. The ticker measures the mean tick latency of each mode and uses the cheaper one,
        # trying the other every PROBE_EVERY ticks. Either way every answer is KKT-checked.
        self.adaptive = bool(adaptive)
        self.mode = "instant"
        self._lat = {"instant": None, "warm": None}
        self._since_other = 0
        self.counts["mode_switches"] = 0
        t0 = time.perf_counter()
        sol = self._resolve(self.prob, start, path_if_warm="warm")
        self.first = sol
        self.first_time = time.perf_counter() - t0
        if sol.verdict != "optimal":
            raise ValueError(f"base model is not certified optimal ({sol.verdict}); nothing to hold")

    # ------------------------------------------------------------------ data
    def current(self) -> Problem:
        return self.prob

    def _bounds(self, p: Problem):
        return np.concatenate([p.lx, p.lc]), np.concatenate([p.ux, p.uc])

    # ------------------------------------------------------------------ adopt / solve
    def _adopt(self, p: Problem, statuses: dict, x, y, factor=None) -> None:
        self.held = _Held(p, self.Az, statuses["col_statuses"], statuses["row_statuses"], factor)
        self.z = np.concatenate([np.asarray(x, dtype=np.float64), p.A @ x])
        self.y = np.asarray(y, dtype=np.float64).copy()
        self._ref = p

    def _resolve(self, p: Problem, start, path_if_warm: str) -> Solution:
        """Warm simplex from `start` (or cold); falls back to the certified public solve."""
        t0 = time.perf_counter()
        r = solve_simplex(p, tol=min(self.tol, 1e-9), time_limit=self.time_limit, start=start)
        meta = {"engine": "simplex", "backend": "numpy", "iterations": r["iterations"],
                "path": path_if_warm if start is not None else "cold", "time": r.get("time"),
                "engine_impl": r.get("engine_impl")}
        sol = _finish(p, r["status"], r["x"], r["y"], self.tol, meta)
        if sol.verdict == "optimal":
            try:
                self._adopt(p, r, r["x"], r["y"])
            except SingularBasis:
                self.held = None
            sol.extra["basis"] = {"col_statuses": r["col_statuses"], "row_statuses": r["row_statuses"]}
        elif start is not None:
            from ..api import solve as _solve
            sol = _solve(p, engine="simplex", tol=self.tol, time_limit=self.time_limit)
            sol.engine = dict(sol.engine, path="cold", warm_status=r["status"])
            if sol.verdict == "optimal" and "basis" in sol.extra:
                try:
                    yi = p.obj_sign * np.asarray(sol.y)
                    self._adopt(p, sol.extra["basis"], sol.x, yi)
                except SingularBasis:
                    self.held = None
        sol.engine["solve_wall"] = time.perf_counter() - t0
        return sol

    # ------------------------------------------------------------------ region test
    def _changed(self, p: Problem) -> tuple[bool, bool]:
        """Whether costs / bounds of p differ from the data the held (z, y) belong to."""
        ref = self._ref
        if ref is None:
            return True, True
        return (not np.array_equal(p.c, ref.c),
                any(not np.array_equal(getattr(p, k), getattr(ref, k)) for k in ("lc", "uc", "lx", "ux")))

    def _instant(self, p: Problem, cost_changed: bool, bounds_changed: bool):
        """(x, y_internal, reason). x is None when the held basis is not optimal for p."""
        h = self.held
        if h is None:
            return None, None, "no basis held"
        lo, hi = self._bounds(p)
        vst = h.vst
        if bounds_changed or self.z is None:
            bad = ((vst == 1) & ~np.isfinite(lo)) | ((vst == 2) & ~np.isfinite(hi)) | \
                  ((vst == 3) & (np.isfinite(lo) | np.isfinite(hi)))
            if bad.any():
                return None, None, "a nonbasic bound became infinite"
            v = np.where(vst == 1, lo, np.where(vst == 2, hi, 0.0))
            v[h.basis] = 0.0
            rhs = -(p.A @ v[:self.n] - v[self.n:])
            z = v
            z[h.basis] = h.factor.ftran(rhs)
        else:
            z = self.z
        zB, loB, hiB = z[h.basis], lo[h.basis], hi[h.basis]
        ft = self.feas_tol
        with np.errstate(invalid="ignore"):
            low_ok = ~np.isfinite(loB) | (zB >= loB - ft * (1.0 + np.abs(loB)))
            up_ok = ~np.isfinite(hiB) | (zB <= hiB + ft * (1.0 + np.abs(hiB)))
        if self._region_hook is None and not (low_ok.all() and up_ok.all()):
            return None, None, "primal: a basic variable leaves its bounds"
        if cost_changed or self.y is None:
            cz = np.concatenate([p.c, np.zeros(self.m)])
            y = h.factor.btran(cz[h.basis])
        else:
            y = self.y
        d = np.concatenate([p.c - self.AT @ y, y])
        dt = ft * (1.0 + (np.abs(p.c).max() if self.n else 0.0))
        fixed = np.isfinite(lo) & (lo == hi)
        ok = (vst == 0) | fixed | ((vst == 1) & (d >= -dt)) | ((vst == 2) & (d <= dt)) | \
             ((vst == 3) & (np.abs(d) <= dt))
        inside = bool(ok.all())
        if self._region_hook is not None:
            inside = self._region_hook(inside)
        if not inside:
            return None, None, "dual: a reduced cost changes sign"
        return z, y, "inside"

    # ------------------------------------------------------------------ ticks
    def tick(self, c=None, lc=None, uc=None, lx=None, ux=None) -> Solution:
        """Apply changes (full vectors, {index: value} dicts or (indices, values) pairs) and answer."""
        cur = self.prob
        new = {k: _apply_change(getattr(cur, k), v) for k, v in
               (("c", c), ("lc", lc), ("uc", uc), ("lx", lx), ("ux", ux)) if v is not None}
        return self.tick_problem(with_data(cur, **new))

    def tick_problem(self, p: Problem) -> Solution:
        """Answer scenario `p` (same A as the base model)."""
        if p.A is not self.base.A and _matrix_token(p.A) != _matrix_token(self.base.A):
            raise ValueError("a tick may change costs and bounds only, not the matrix")
        t0 = time.perf_counter()
        self.prob = p
        sol = None
        mode = self._pick_mode()
        if mode == "instant":
            fac0 = self.held.factor_seconds if self.held is not None else 0.0
            z, y, reason = self._instant(p, *self._changed(p))
            if self.held is not None:
                self.counts["factor_time"] += max(0.0, self.held.factor_seconds - fac0)
        else:
            z, y, reason = None, None, "warm-only mode: instant answers were not paying off"
        if z is not None:
            x = z[:self.n].copy()
            cand = _finish(p, "optimal", x, y, self.tol,
                           {"engine": "plan_ticker", "backend": "numpy", "iterations": 0, "path": "instant",
                            "region": reason})
            if cand.verdict == "optimal":
                sol = cand
                self.z, self.y, self._ref = z, y, p
                sol.extra["basis"] = self.held.statuses()
            else:
                self.counts["instant_rejected"] += 1
                reason = "instant answer failed its KKT check: " + str(cand.engine.get("note"))
        if sol is None and self.speculator is not None:
            sol = self.speculator.take(self, p)
        if sol is None:
            start = self.held.statuses() if self.held is not None else None
            sol = self._resolve(p, start, path_if_warm="warm")
            sol.engine["region"] = reason
        path = sol.engine.get("path", "warm")
        self.counts[path] = self.counts.get(path, 0) + 1
        if sol.verdict != "optimal":
            self.counts["not_optimal"] += 1
        lat = time.perf_counter() - t0
        sol.engine["latency"] = lat
        sol.engine["mode"] = mode
        self._learn(mode, lat)
        self.log.append({"path": path, "latency": lat, "verdict": sol.verdict,
                         "iterations": sol.engine.get("iterations"), "max_rel": (sol.residuals or {}).get("max_rel")})
        if self.speculator is not None:
            self.speculator.after_tick(self, p)
        return sol

    def _pick_mode(self) -> str:
        if not self.adaptive or self.held is None:
            return "instant"
        other = "warm" if self.mode == "instant" else "instant"
        if self._lat[other] is None or self._since_other >= self.PROBE_EVERY:
            self._since_other = 0
            return other if self._lat[self.mode] is not None else self.mode
        self._since_other += 1
        return self.mode

    def _learn(self, mode: str, lat: float) -> None:
        if not self.adaptive:
            return
        old = self._lat[mode]
        self._lat[mode] = lat if old is None else (1 - self.EWMA) * old + self.EWMA * lat
        a, b = self._lat["instant"], self._lat["warm"]
        if a is not None and b is not None:
            best = "instant" if a <= b else "warm"
            if best != self.mode:
                self.mode = best
                self.counts["mode_switches"] += 1

    # ------------------------------------------------------------------ ranging along a ray
    def ray_interval(self, dc=None, dlc=None, duc=None, dlx=None, dux=None) -> tuple[float, float]:
        """Exact [t_lo, t_hi] (t_lo <= 0 <= t_hi) such that the held basis stays optimal for the
        current data + t * direction. Cost and bound parts are independent conditions; the result
        is their intersection. Returns (nan, nan) when the held basis is not optimal at t = 0."""
        h = self.held
        p = self.prob
        n, m = self.n, self.m
        z = self.z
        y = self.y
        if h is None or z is None or self._ref is not p:
            return float("nan"), float("nan")
        lo, hi = self._bounds(p)
        t_lo, t_hi = -np.inf, np.inf
        ft = self.feas_tol

        def clip(a, b, t_lo, t_hi):
            """Intersect with {t : a + t b >= 0} given a >= 0 (up to tolerance)."""
            a = np.maximum(a, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                neg = b < -1e-12 * (1.0 + np.abs(a))
                pos = b > 1e-12 * (1.0 + np.abs(a))
                if neg.any():
                    t_hi = min(t_hi, float(np.min(a[neg] / -b[neg])))
                if pos.any():
                    t_lo = max(t_lo, float(np.max(-a[pos] / b[pos])))
            return t_lo, t_hi

        if dc is not None:
            dcz = np.concatenate([np.asarray(dc, dtype=np.float64), np.zeros(m)])
            dy = h.factor.btran(dcz[h.basis])
            d0 = np.concatenate([p.c - self.AT @ y, y])
            dd = dcz - np.concatenate([self.AT @ dy, -dy])
            fixed = np.isfinite(lo) & (lo == hi)
            dt = ft * (1.0 + np.abs(p.c).max())
            sel = (h.vst == 1) & ~fixed
            if d0[sel].min(initial=0.0) < -dt:
                return float("nan"), float("nan")
            t_lo, t_hi = clip(d0[sel], dd[sel], t_lo, t_hi)
            sel = (h.vst == 2) & ~fixed
            if d0[sel].max(initial=0.0) > dt:
                return float("nan"), float("nan")
            t_lo, t_hi = clip(-d0[sel], -dd[sel], t_lo, t_hi)
            sel = h.vst == 3
            if np.any(np.abs(dd[sel]) > 1e-12):
                t_lo, t_hi = max(t_lo, 0.0), min(t_hi, 0.0)
        bdir = [np.zeros(k) if v is None else np.asarray(v, dtype=np.float64)
                for v, k in ((dlx, n), (dlc, m), (dux, n), (duc, m))]
        if any(np.any(v != 0) for v in bdir):
            dlo = np.concatenate([bdir[0], bdir[1]])
            dhi = np.concatenate([bdir[2], bdir[3]])
            dv = np.where(h.vst == 1, dlo, np.where(h.vst == 2, dhi, 0.0))
            dv[h.basis] = 0.0
            dz = dv.copy()
            dz[h.basis] = h.factor.ftran(-(p.A @ dv[:n] - dv[n:]))
            B = h.basis
            fl = np.isfinite(lo[B])
            t_lo, t_hi = clip((z[B] - lo[B])[fl], (dz[B] - dlo[B])[fl], t_lo, t_hi)
            fu = np.isfinite(hi[B])
            t_lo, t_hi = clip((hi[B] - z[B])[fu], (dhi[B] - dz[B])[fu], t_lo, t_hi)
        return float(min(t_lo, 0.0)), float(max(t_hi, 0.0))

    def ray_slope(self, dc=None, dlc=None, duc=None, dlx=None, dux=None) -> float:
        """d phi / dt of the held basis's affine piece along the ray (minimisation form)."""
        h, p, n, m = self.held, self.prob, self.n, self.m
        x, y = self.z[:n], self.y
        s = 0.0 if dc is None else float(np.asarray(dc) @ x)
        d = np.concatenate([p.c - self.AT @ y, y])
        z0 = lambda v, k: np.zeros(k) if v is None else np.asarray(v, dtype=np.float64)  # noqa: E731
        dlo = np.concatenate([z0(dlx, n), z0(dlc, m)])
        dhi = np.concatenate([z0(dux, n), z0(duc, m)])
        dv = np.where(h.vst == 1, dlo, np.where(h.vst == 2, dhi, 0.0))
        dv[h.basis] = 0.0
        return s + float(d @ dv)

    def cost_range(self, j: int) -> tuple[float, float]:
        """Range of c_j (minimisation form) over which the held basis stays optimal."""
        e = np.zeros(self.n)
        e[int(j)] = 1.0
        a, b = self.ray_interval(dc=e)
        return float(self.prob.c[j] + a), float(self.prob.c[j] + b)

    def row_limit_range(self, i: int, side: str = "upper") -> tuple[float, float]:
        """Range of one row limit over which the held basis stays optimal (primal feasible)."""
        e = np.zeros(self.m)
        e[int(i)] = 1.0
        a, b = self.ray_interval(**({"duc": e} if side == "upper" else {"dlc": e}))
        v = float((self.prob.uc if side == "upper" else self.prob.lc)[i])
        return v + a, v + b

    # ------------------------------------------------------------------ statistics
    def stats(self) -> dict:
        lat = np.array([r["latency"] for r in self.log]) if self.log else np.zeros(0)
        out = {"ticks": len(self.log), **{k: v for k, v in self.counts.items()}}
        out["instant_fraction"] = self.counts["instant"] / max(1, len(self.log))
        out["mode"] = self.mode
        out["mode_latency_ewma"] = dict(self._lat)
        for path in ("instant", "speculative", "warm", "cold"):
            v = np.array([r["latency"] for r in self.log if r["path"] == path])
            if v.size:
                out[f"{path}_latency_p50"] = float(np.median(v))
                out[f"{path}_latency_p95"] = float(np.quantile(v, 0.95))
                out[f"{path}_latency_max"] = float(v.max())
        if lat.size:
            out["latency_mean"] = float(lat.mean())
            out["latency_p50"] = float(np.median(lat))
            out["latency_p95"] = float(np.quantile(lat, 0.95))
            out["latency_total"] = float(lat.sum())
        return out


# =============================================================================== Greeks
def _ray_parts(prob: Problem, dc, dlc, duc, dlx, dux):
    z = lambda v, k: None if v is None else np.asarray(v, dtype=np.float64).reshape(k)  # noqa: E731
    return z(dc, prob.n), z(dlc, prob.m), z(duc, prob.m), z(dlx, prob.n), z(dux, prob.n)


def _at(prob, t, dc, dlc, duc, dlx, dux) -> Problem:
    vec = {}
    for k, d in (("c", dc), ("lc", dlc), ("uc", duc), ("lx", dlx), ("ux", dux)):
        if d is not None:
            vec[k] = getattr(prob, k) + t * d
    return with_data(prob, **vec)


def _slope_at(tk: PlanTicker, ray) -> tuple[float, float, float]:
    a, b = tk.ray_interval(*ray[:1], dlc=ray[1], duc=ray[2], dlx=ray[3], dux=ray[4])
    return a, b, tk.ray_slope(ray[0], dlc=ray[1], duc=ray[2], dlx=ray[3], dux=ray[4])


def plan_greeks(prob: Problem, ts, dc=None, dlc=None, duc=None, dlx=None, dux=None,
                tol: float = 1e-9, probe: float = 1e-4, max_probe_halvings: int = 30,
                max_segments: int = 2000) -> dict:
    """Deltas and gammas of the optimal value along the ray p(t) = prob + t * direction.

    Directions are in the model's internal (minimisation) storage, like CaseDelta values; the
    direction may be a pure cost ray (dc) or a pure bound ray (dlc/duc/dlx/dux) - mixing both makes
    phi piecewise quadratic and is rejected. Values and slopes are reported in the user's sense.

    Returns {"cases": [...], "breakpoints": [...], "segments": [...]}: per case t the certified
    value, delta_minus / delta_plus (one-sided derivatives; equal inside a segment), delta (None at
    a breakpoint), the per-coordinate gradients (dphi/dc = x, dphi/d row limit = y,
    dphi/d column bound = reduced cost, user's sense), and its basis interval; per breakpoint the
    location t*, left / right slopes and gamma = right slope - left slope.
    """
    ray = _ray_parts(prob, dc, dlc, duc, dlx, dux)
    has_c = ray[0] is not None and np.any(ray[0] != 0)
    has_b = any(v is not None and np.any(v != 0) for v in ray[1:])
    if has_c == has_b:
        raise ValueError("give either a cost direction (dc) or a bound direction (dlc/duc/dlx/dux)")
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size == 0 or np.any(np.diff(ts) <= 0):
        raise ValueError("ts must be strictly increasing")
    sgn = prob.obj_sign
    scale = 1.0 + float(np.max(np.abs(ts)))
    eps_t = 1e-9 * scale
    tk = PlanTicker(_at(prob, ts[0], *ray), tol=tol)
    notes: list[str] = []

    def state():
        a, b, s = _slope_at(tk, ray)
        return {"lo": a, "hi": b, "slope": s, "basis": tk.held.statuses()}

    def goto(t):
        return tk.tick_problem(_at(prob, t, *ray))

    def probe_side(t, side):
        """Basis valid on [t, t+eta] (side=+1) or [t-eta, t] (side=-1); its slope or None."""
        eta = probe * scale
        for _ in range(max_probe_halvings):
            tp = t + side * eta
            s = goto(tp)
            if s.verdict == "optimal":
                st = state()
                if (side > 0 and tp + st["lo"] <= t + eps_t) or (side < 0 and tp + st["hi"] >= t - eps_t):
                    return st, tp
            eta *= 0.5
        return None, None

    segments: list[dict] = []
    breakpoints: list[dict] = []
    # walk the segments from ts[0] to ts[-1] (Gass-Saaty: next breakpoint = ratio test of the basis)
    goto(float(ts[0]))
    st = state()
    segments.append({"t_lo": float(ts[0]) + st["lo"], "t_hi": float(ts[0]) + st["hi"], "slope": sgn * st["slope"]})
    while segments[-1]["t_hi"] < ts[-1] - eps_t and len(segments) < max_segments:
        t_b = segments[-1]["t_hi"]
        st_r, tp = probe_side(t_b, +1)
        if st_r is None:
            notes.append(f"could not find the basis right of t={t_b:.12g}")
            break
        slope = sgn * st_r["slope"]
        left = segments[-1]["slope"]
        if abs(slope - left) <= 1e-9 * (1.0 + abs(slope) + abs(left)):
            segments[-1]["t_hi"] = tp + st_r["hi"]               # basis change, same slope: one piece
            continue
        breakpoints.append({"t": t_b, "slope_left": left, "slope_right": slope, "gamma": slope - left})
        segments.append({"t_lo": t_b, "t_hi": tp + st_r["hi"], "slope": slope})
    if len(segments) >= max_segments:
        notes.append(f"hit max_segments={max_segments}")

    cases = []
    for t in ts:
        s = goto(float(t))
        entry = {"t": float(t), "verdict": s.verdict, "value": s.objective, "path": s.engine.get("path")}
        if s.verdict == "optimal":
            a, b, sl = _slope_at(tk, ray)
            entry.update(interval=(float(t) + a, float(t) + b), x=s.x.copy(), y=s.y.copy(),
                         reduced_costs=s.reduced_costs)
            dm = dp = sgn * sl
            at_bp = None
            for bp in breakpoints:
                if abs(bp["t"] - t) <= max(eps_t, 1e-9 * (1 + abs(t))):
                    at_bp = bp
            if at_bp is not None:
                dm, dp = at_bp["slope_left"], at_bp["slope_right"]
            entry.update(delta_minus=dm, delta_plus=dp, delta=None if at_bp is not None and dm != dp else dm,
                         gamma_kink=0.0 if at_bp is None else at_bp["gamma"], at_breakpoint=at_bp is not None)
        cases.append(entry)
    return {"cases": cases, "breakpoints": breakpoints, "segments": segments, "notes": notes,
            "kind": "cost" if has_c else "bounds", "ticker": tk.stats()}


def second_difference(greeks: dict, t: float, h: float) -> float:
    """(phi(t+h) - 2 phi(t) + phi(t-h)) / h^2 predicted from the breakpoints: phi is piecewise
    linear, so this equals sum_b gamma_b * max(0, h - |t - t_b|) / h^2 exactly."""
    return float(sum(b["gamma"] * max(0.0, h - abs(t - b["t"])) for b in greeks["breakpoints"]) / h ** 2)


# =============================================================================== stalled-lane hand-off
def _data_vector(p: Problem) -> np.ndarray:
    v = np.concatenate([getattr(p, k) for k in _DATA])
    return np.where(np.isfinite(v), v, 0.0)


def handoff_stalled(probs: list, X, Y, stalled, finished: Optional[dict] = None, tol: float = 1e-6,
                    stall_max: float = 1e-2, time_limit: float = 600.0) -> dict:
    """Finish stalled first-order lanes of a case stack exactly.

    probs: case Problems sharing one A. X (n, S), Y (m, S): the first-order points (internal sign,
    as pdhg.BatchResult). stalled: indices to finish. finished: {index: basis statuses} of cases
    that already hold an exact vertex (may be empty). Cases are taken in nearest-neighbour order;
    each one starts from the basis of the nearest case (data distance) that holds one. A point with
    relative KKT error <= stall_max goes through crossover (engines/crossover.py) from that basis;
    a worse point is re-solved by the simplex from that basis. Returns {index: Solution}, each with
    a verdict from the KKT check, and engine["warm_from"] naming the case whose basis was used.
    """
    from ..certify.kkt import kkt_residuals
    from ..engines.crossover import crossover

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    bases = dict(finished or {})
    S = len(probs)
    v0 = _data_vector(probs[0])
    mask = np.zeros(v0.size, dtype=bool)
    for p in probs[1:]:
        mask |= _data_vector(p) != v0
    V = np.stack([_data_vector(p)[mask] for p in probs]) if S else np.zeros((0, 0))
    best = np.full(S, np.inf)
    src_of = np.full(S, -1, dtype=np.int64)

    def offer(k):
        d = np.linalg.norm(V - V[k], axis=1)
        better = d < best
        best[better] = d[better]
        src_of[better] = k

    for k in bases:
        offer(int(k))
    todo = [int(i) for i in stalled]
    out: dict = {}
    while todo:
        a = int(np.argmin(best[todo]))
        i = todo.pop(a)
        src = int(src_of[i]) if src_of[i] >= 0 else None
        start = bases.get(src) if src is not None else None
        p = probs[i]
        t0 = time.perf_counter()
        k = kkt_residuals(p, X[:, i], Y[:, i])
        if k["max_rel"] <= stall_max:
            r = crossover(p, X[:, i], Y[:, i], time_limit=time_limit, tol=min(tol, 1e-9), start=start)
            how = "crossover"
        else:
            r = solve_simplex(p, tol=min(tol, 1e-9), time_limit=time_limit, start=start)
            how = "simplex"
        meta = {"engine": how, "backend": "numpy", "iterations": r.get("iterations"), "handoff": True,
                "warm_from": src, "first_order_max_rel": k["max_rel"], "time": time.perf_counter() - t0}
        sol = _finish(p, r.get("status", "numerical_error"), r.get("x"), r.get("y"), tol, meta)
        if sol.verdict != "optimal" and how == "crossover":
            r = solve_simplex(p, tol=min(tol, 1e-9), time_limit=time_limit, start=start)
            meta = dict(meta, engine="simplex", iterations=r["iterations"], note="crossover failed; simplex",
                        time=time.perf_counter() - t0)
            sol = _finish(p, r["status"], r["x"], r["y"], tol, meta)
        if sol.verdict == "optimal" and "col_statuses" in r:
            st = {"col_statuses": list(r["col_statuses"]), "row_statuses": list(r["row_statuses"])}
            bases[i] = st
            offer(i)
            sol.extra["basis"] = st
        out[i] = sol
    return out


# =============================================================================== speculation
@dataclass
class _Spec:
    prob: Problem
    fp: str
    result: Optional[dict] = None
    factor: Optional[BasisFactor] = None
    time: float = 0.0


@dataclass
class Speculator:
    """Pre-solve likely next scenarios; reuse an answer only on a bit-identical data match.

    predict(ticker, p) returns the candidate next scenarios after tick p (e.g. the neighbours of a
    price lattice). A candidate the held basis already answers is skipped. background=True runs a
    worker thread; otherwise call run_pending(budget) when the caller is idle.
    """
    predict: Callable
    background: bool = False
    max_cache: int = 256
    cache: dict = field(default_factory=dict)
    stats: dict = field(default_factory=lambda: {"scheduled": 0, "skipped_inside": 0, "solved": 0,
                                                 "hits": 0, "misses": 0, "rejected": 0, "work_time": 0.0})

    def __post_init__(self):
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._thread = None
        if self.background:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    # -------------------------------------------------------------- scheduling
    def after_tick(self, tk: PlanTicker, p: Problem) -> None:
        start = tk.held.statuses() if tk.held is not None else None
        for q in self.predict(tk, p) or []:
            fp = fingerprint(q)
            with self._lock:
                if fp in self.cache:
                    continue
            z, _, _ = _probe_instant(tk, q)
            if z is not None:
                self.stats["skipped_inside"] += 1
                continue
            with self._lock:
                self.cache[fp] = _Spec(prob=q, fp=fp)
                while len(self.cache) > self.max_cache:
                    self.cache.pop(next(iter(self.cache)))
            self.stats["scheduled"] += 1
            self._q.put((fp, start, tk))

    def _solve_one(self, item) -> None:
        fp, start, tk = item
        with self._lock:
            spec = self.cache.get(fp)
        if spec is None or spec.result is not None:
            return
        t0 = time.perf_counter()
        r = solve_simplex(spec.prob, tol=min(tk.tol, 1e-9), time_limit=tk.time_limit, start=start)
        factor = None
        if r["status"] == "optimal":
            try:
                st = np.array([_CODE[s] for s in list(r["col_statuses"]) + list(r["row_statuses"])])
                factor = BasisFactor(tk.Az, np.flatnonzero(st == 0))
            except SingularBasis:
                factor = None
        spec.time = time.perf_counter() - t0
        with self._lock:
            spec.result, spec.factor = r, factor
            self.stats["solved"] += 1
            self.stats["work_time"] += spec.time

    def run_pending(self, budget: float = float("inf")) -> int:
        """Solve queued candidates in the calling thread for up to `budget` seconds."""
        t0 = time.perf_counter()
        done = 0
        while time.perf_counter() - t0 < budget:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            try:
                self._solve_one(item)
            finally:
                self._q.task_done()
            done += 1
        return done

    def wait_idle(self, timeout: float = 60.0) -> bool:
        """Block until every scheduled candidate is solved (background mode); False on timeout."""
        t_end = time.perf_counter() + timeout
        while self._q.unfinished_tasks:
            if time.perf_counter() > t_end:
                return False
            time.sleep(0.001)
        return True

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            try:
                self._solve_one(item)
            finally:
                self._q.task_done()

    def close(self) -> None:
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=60)
            self._thread = None

    # -------------------------------------------------------------- reuse
    def take(self, tk: PlanTicker, p: Problem) -> Optional[Solution]:
        fp = fingerprint(p)
        with self._lock:
            spec = self.cache.get(fp)
        if spec is None or spec.result is None or not _same_data(spec.prob, p):
            self.stats["misses"] += 1
            return None
        r = spec.result
        meta = {"engine": "simplex", "backend": "numpy", "iterations": r["iterations"], "path": "speculative",
                "presolve_time": spec.time, "fingerprint": fp}
        sol = _finish(p, r["status"], r["x"], r["y"], tk.tol, meta)
        if sol.verdict != "optimal":
            self.stats["rejected"] += 1
            return None
        self.stats["hits"] += 1
        try:
            tk._adopt(p, r, r["x"], r["y"], factor=spec.factor)
        except SingularBasis:
            tk.held = None
        sol.extra["basis"] = {"col_statuses": r["col_statuses"], "row_statuses": r["row_statuses"]}
        return sol


def _probe_instant(tk: PlanTicker, q: Problem):
    """Region test of the held basis for q without changing the ticker's state."""
    saved = tk.z, tk.y, tk._ref
    try:
        return tk._instant(q, *tk._changed(q))
    finally:
        tk.z, tk.y, tk._ref = saved
