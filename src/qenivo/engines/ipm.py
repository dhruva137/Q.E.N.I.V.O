"""Bounded-variable Mehrotra predictor-corrector interior point method for LP (engine "ipm").

Formulation (no bound rows, no variable splitting):
    min c'v   s.t.  K v = b,   l <= v <= u
with v = [x (non-fixed columns); s (one slack per inequality or ranged row)] and
    K = [ A_eq   0 ]        b = [ b_eq ]        row  i in eq:  A_i x = b_i
        [ A_in  -I ]            [  0   ]        row  i in in:  A_i x - s_i = 0,  lc_i <= s_i <= uc_i
Fixed columns are substituted out; free rows are dropped (dual 0). Every finite bound gets a
slack t and a dual z >= 0 (t_l = v - l, t_u = u - v); free variables get a small primal
regularisation. The Newton system reduces to the normal equations of size (#rows):
    (K D K' + delta I) dy = r_b + K D r~,     D = (z_l/t_l + z_u/t_u + rho)^-1
factored ONCE per iteration and reused by the predictor and the corrector (Mehrotra 1992;
Wright 1997 ch. 10-11; the bounded-variable treatment of Lustig, Marsten & Shanno 1991).

Linear algebra: dense Cholesky (LAPACK via NumPy, or cuSOLVER via CuPy when a GPU is used)
up to `dense_limit` rows, preconditioned conjugate gradients beyond. A native sparse
supernodal Cholesky is the next milestone (docs/ROADMAP.md). No solver library is used.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from ..model import Problem

DENSE_LIMIT = 6000


class _Normal:
    """Factor K D K' + delta I once; solve many right-hand sides."""

    def __init__(self, K, KT, d, xp_gpu=None, dense_limit=DENSE_LIMIT):
        p = K.shape[0]
        self.p = p
        N = (K @ sp.diags(d) @ KT).tocsr()
        diag = N.diagonal()
        dmax = float(diag.max()) if p else 1.0
        self.delta = 0.0
        self.mode = "dense" if p <= dense_limit else "pcg"
        self.gpu = xp_gpu
        if self.mode == "dense":
            Nd = N.toarray()
            for rel in (1e-14, 1e-12, 1e-10, 1e-8, 1e-6):
                delta = rel * max(dmax, 1.0)
                try:
                    if self.gpu is not None:
                        cp = self.gpu
                        L = cp.linalg.cholesky(cp.asarray(Nd) + delta * cp.eye(p))
                        if not bool(cp.all(cp.isfinite(L))):
                            raise np.linalg.LinAlgError
                        self.L = L
                    else:
                        self.L = np.linalg.cholesky(Nd + delta * np.eye(p))
                    self.delta = delta
                    break
                except np.linalg.LinAlgError:
                    continue
            else:
                self.mode = "pcg"
        if self.mode == "pcg":
            self.delta = 1e-10 * max(dmax, 1.0)
            self.N = (N + self.delta * sp.eye(p)).tocsr()
            self.jac = 1.0 / np.maximum(self.N.diagonal(), 1e-300)
            self.last = np.zeros(p)

    def solve(self, r):
        if self.mode == "dense":
            if self.gpu is not None:
                cp = self.gpu
                from cupyx.scipy.linalg import solve_triangular as st
                rr = cp.asarray(r)
                t = st(self.L, rr, lower=True)
                return cp.asnumpy(st(self.L.T, t, lower=False))
            from scipy.linalg import solve_triangular
            t = solve_triangular(self.L, r, lower=True, check_finite=False)
            return solve_triangular(self.L.T, t, lower=False, check_finite=False)
        x = self.last.copy()
        N, Minv = self.N, self.jac
        rr = r - N @ x
        nb = np.linalg.norm(r) + 1e-300
        z = Minv * rr
        pdir = z.copy()
        rz = rr @ z
        for _ in range(3000):
            if np.linalg.norm(rr) <= 1e-13 * nb:
                break
            Np = N @ pdir
            a = rz / (pdir @ Np + 1e-300)
            x += a * pdir
            rr -= a * Np
            z = Minv * rr
            rz_new = rr @ z
            pdir = z + (rz_new / (rz + 1e-300)) * pdir
            rz = rz_new
        self.last = x
        return x


def _build(prob: Problem):
    lx, ux, lc, uc = prob.lx, prob.ux, prob.lc, prob.uc
    fixed = lx == ux
    keepc = ~fixed
    xfix = np.where(fixed, lx, 0.0)
    A = prob.A.tocsr()
    shift = A @ xfix                                  # contribution of fixed columns to rows
    fin_l, fin_u = np.isfinite(lc), np.isfinite(uc)
    free_row = ~fin_l & ~fin_u
    eq = fin_l & fin_u & (np.abs(uc - lc) <= 1e-12 * (1.0 + np.abs(uc)))
    ineq = ~eq & ~free_row
    rows_eq, rows_in = np.flatnonzero(eq), np.flatnonzero(ineq)
    Ac = A[:, keepc]
    k_in = len(rows_in)
    K = sp.vstack([sp.hstack([Ac[rows_eq], sp.csr_matrix((len(rows_eq), k_in))]),
                   sp.hstack([Ac[rows_in], -sp.eye(k_in, format="csr")])], format="csr")
    b = np.concatenate([uc[rows_eq] - shift[rows_eq], np.zeros(k_in)])
    c = np.concatenate([prob.c[keepc], np.zeros(k_in)])
    l = np.concatenate([lx[keepc], lc[rows_in] - shift[rows_in]])
    u = np.concatenate([ux[keepc], uc[rows_in] - shift[rows_in]])
    return {"K": K, "b": b, "c": c, "l": l, "u": u, "keepc": keepc, "xfix": xfix,
            "rows_eq": rows_eq, "rows_in": rows_in, "nx": int(keepc.sum()),
            "c0": prob.c0 + float(prob.c @ xfix)}


def _max_step(v, dv):
    neg = dv < 0
    if not np.any(neg):
        return 1.0
    return float(min(1.0, np.min(-v[neg] / dv[neg])))


def _mehrotra(F, tol, time_limit, t0, max_iter=300, use_gpu=False, verbose=False):
    K, b, c, l, u = F["K"], F["b"], F["c"], F["l"], F["u"]
    KT = K.T.tocsr()
    p, q = K.shape
    hl, hu = np.isfinite(l), np.isfinite(u)
    free = ~hl & ~hu
    xp_gpu = None
    if use_gpu:
        import cupy
        xp_gpu = cupy
    # Mehrotra starting point: least-norm primal / least-squares dual, then two shifts
    ns0 = _Normal(K, KT, np.ones(q), xp_gpu)
    v = KT @ ns0.solve(b) if p else np.zeros(q)
    y = ns0.solve(K @ c) if p else np.zeros(0)
    r = c - KT @ y
    tl = np.where(hl, v - l, 1.0)
    tu = np.where(hu, u - v, 1.0)
    both = hl & hu
    zl = np.where(hl & ~hu, r, 0.0) + np.where(both, np.maximum(r, 0.0), 0.0)
    zu = np.where(hu & ~hl, -r, 0.0) + np.where(both, np.maximum(-r, 0.0), 0.0)
    tmin = min(tl[hl].min(initial=np.inf), tu[hu].min(initial=np.inf))
    zmin = min(zl[hl].min(initial=np.inf), zu[hu].min(initial=np.inf))
    dp = max(-1.5 * tmin, 0.0) if np.isfinite(tmin) else 0.0
    dd = max(-1.5 * zmin, 0.0) if np.isfinite(zmin) else 0.0
    tl, tu = np.where(hl, tl + dp, 1.0), np.where(hu, tu + dp, 1.0)
    zl, zu = np.where(hl, zl + dd, 0.0), np.where(hu, zu + dd, 0.0)
    tz = float(tl[hl] @ zl[hl] + tu[hu] @ zu[hu])
    st, sz = float(tl[hl].sum() + tu[hu].sum()), float(zl[hl].sum() + zu[hu].sum())
    if st > 0 and sz > 0:
        tl, tu = np.where(hl, tl + 0.5 * tz / sz, 1.0), np.where(hu, tu + 0.5 * tz / sz, 1.0)
        zl, zu = np.where(hl, zl + 0.5 * tz / st, 0.0), np.where(hu, zu + 0.5 * tz / st, 0.0)
    tl, tu = np.where(hl, np.maximum(tl, 1e-8), 1.0), np.where(hu, np.maximum(tu, 1e-8), 1.0)
    zl, zu = np.where(hl, np.maximum(zl, 1e-8), 0.0), np.where(hu, np.maximum(zu, 1e-8), 0.0)
    bnorm, cnorm = 1.0 + np.linalg.norm(b), 1.0 + np.linalg.norm(c)
    rho = 1e-10
    ncomp = int(hl.sum() + hu.sum())
    lf, uf = np.where(hl, l, 0.0), np.where(hu, u, 0.0)
    info = {"regularisation": 0.0, "factor": None}
    best_err, hist = np.inf, []
    with np.errstate(all="ignore"):
        for it in range(max_iter):
            if time.perf_counter() - t0 > time_limit:
                return "time_limit", v, y, it, info
            rb = b - K @ v
            rc = c - KT @ y - zl + zu
            rl = np.where(hl, (v - l) - tl, 0.0)          # keep t_l = v - l
            ru = np.where(hu, (u - v) - tu, 0.0)
            mu = float((tl[hl] @ zl[hl] + tu[hu] @ zu[hu]) / max(ncomp, 1))
            pobj = float(c @ v)
            dobj = float(b @ y + lf @ zl - uf @ zu)
            gap = abs(pobj - dobj) / (1.0 + abs(pobj) + abs(dobj))
            rp = max(np.linalg.norm(rb), np.linalg.norm(rl), np.linalg.norm(ru)) / bnorm
            rd = np.linalg.norm(rc) / cnorm
            if verbose:
                print(f"{it:4d} pobj {pobj:+.8e} dobj {dobj:+.8e} rp {rp:.1e} rd {rd:.1e} gap {gap:.1e} mu {mu:.1e} "
                      f"|v| {np.abs(v).max(initial=0):.1e} |y| {np.abs(y).max(initial=0):.1e}")
            if rp <= tol and rd <= tol and gap <= tol:
                return "optimal", v, y, it, info
            err = max(rp, rd, gap)
            best_err = min(best_err, err)
            hist.append(best_err)
            if it >= 40 and hist[-31] <= 1.01 * best_err:        # no progress in 30 iterations
                return "stalled", v, y, it, info
            if not (np.all(np.isfinite(v)) and np.all(np.isfinite(y))):
                return "numerical_error", v, y, it, info
            if max(np.abs(v).max(initial=0), np.abs(y).max(initial=0)) > 1e15:
                return ("unbounded" if np.abs(v).max() > np.abs(y).max() else "infeasible"), v, y, it, info
            Dinv = np.where(hl, zl / tl, 0.0) + np.where(hu, zu / tu, 0.0) + rho + np.where(free, 1e-8, 0.0)
            D = 1.0 / Dinv
            ns = _Normal(K, KT, D, xp_gpu)
            info["regularisation"] = max(info["regularisation"], ns.delta)
            info["factor"] = ns.mode

            def direction(sl, su):
                # complementarity right-hand sides sl = target - t_l z_l (...), su likewise
                # z_l dt_l + t_l dz_l = sl,  dt_l = dv + rl;   z_u dt_u + t_u dz_u = su,  dt_u = -dv + ru
                rt = rc - np.where(hl, (sl - zl * rl) / tl, 0.0) + np.where(hu, (su - zu * ru) / tu, 0.0)
                dy = ns.solve(rb + K @ (D * rt))
                dv = D * (KT @ dy - rt)
                dtl = np.where(hl, dv + rl, 0.0)
                dtu = np.where(hu, -dv + ru, 0.0)
                dzl = np.where(hl, (sl - zl * dtl) / tl, 0.0)
                dzu = np.where(hu, (su - zu * dtu) / tu, 0.0)
                return dv, dy, dtl, dtu, dzl, dzu

            # predictor
            dv, dy, dtl, dtu, dzl, dzu = direction(-tl * zl * hl, -tu * zu * hu)
            ap = min(_max_step(tl[hl], dtl[hl]), _max_step(tu[hu], dtu[hu]))
            ad = min(_max_step(zl[hl], dzl[hl]), _max_step(zu[hu], dzu[hu]))
            mu_aff = float(((tl + ap * dtl)[hl] @ (zl + ad * dzl)[hl] + (tu + ap * dtu)[hu] @ (zu + ad * dzu)[hu])
                           / max(ncomp, 1))
            sigma = (mu_aff / mu) ** 3 if mu > 0 else 0.0
            # corrector
            sl = np.where(hl, sigma * mu - tl * zl - dtl * dzl, 0.0)
            su = np.where(hu, sigma * mu - tu * zu - dtu * dzu, 0.0)
            dv, dy, dtl, dtu, dzl, dzu = direction(sl, su)
            eta = 0.995 if mu > 1e-6 else 0.9995
            ap = min(1.0, eta * min(_max_step(tl[hl], dtl[hl]), _max_step(tu[hu], dtu[hu])))
            ad = min(1.0, eta * min(_max_step(zl[hl], dzl[hl]), _max_step(zu[hu], dzu[hu])))
            v = v + ap * dv
            tl = np.where(hl, np.maximum(tl + ap * dtl, 1e-300), 1.0)
            tu = np.where(hu, np.maximum(tu + ap * dtu, 1e-300), 1.0)
            y = y + ad * dy
            zl = np.where(hl, np.maximum(zl + ad * dzl, 1e-300), 0.0)
            zu = np.where(hu, np.maximum(zu + ad * dzu, 1e-300), 0.0)
    return "iteration_limit", v, y, max_iter, info


def solve_ipm(prob: Problem, tol: float = 1e-8, time_limit: float = 3600.0, use_gpu: bool = False,
              verbose: bool = False) -> dict:
    """Interior point solve. Returns x, row duals y (internal minimisation sign) and status."""
    t0 = time.perf_counter()
    if prob.is_qp:
        raise ValueError("ipm is an LP engine; use the pdqp engine for QP")
    if prob.n == 0:
        return {"status": "optimal", "x": np.zeros(0), "y": np.zeros(prob.m), "iterations": 0,
                "time": 0.0, "regularisation": 0.0}
    if np.any(prob.lx > prob.ux) or np.any(prob.lc > prob.uc):
        return {"status": "infeasible", "x": np.clip(np.zeros(prob.n), prob.lx, prob.ux), "y": np.zeros(prob.m),
                "iterations": 0, "time": time.perf_counter() - t0, "regularisation": 0.0}
    # equilibrate (geometric + Ruiz, as in the first-order engine), solve, unscale
    from .precondition import precondition
    K_s, c_s, lc_s, uc_s, lx_s, ux_s, sc = precondition(prob.A, prob.c, prob.lc, prob.uc, prob.lx, prob.ux,
                                                        geo_iters=12, ruiz_iters=10, pc_alpha=None,
                                                        bound_obj=False)
    scaled = Problem(c=c_s, A=K_s, lc=lc_s, uc=uc_s, lx=lx_s, ux=ux_s, c0=prob.c0, name=prob.name)
    F = _build(scaled)
    status, v, yk, iters, info = _mehrotra(F, tol, time_limit, t0, use_gpu=use_gpu, verbose=verbose)
    xs = F["xfix"].copy()
    xs[F["keepc"]] = v[:F["nx"]]
    ys = np.zeros(prob.m)
    ne = len(F["rows_eq"])
    ys[F["rows_eq"]] = yk[:ne]
    ys[F["rows_in"]] = yk[ne:]
    x = np.clip(sc.col * xs, prob.lx, prob.ux)
    y = sc.row * ys
    if status == "optimal":
        # judge on the ORIGINAL problem: scaled residuals can pass while unscaled ones do not
        from ..certify.kkt import kkt_residuals
        if kkt_residuals(prob, x, y)["max_rel"] > 10 * tol:
            status = "numerical_error"
    return {"status": status, "x": x, "y": y, "iterations": int(iters), "time": time.perf_counter() - t0,
            "regularisation": float(info["regularisation"]), "factor": info["factor"]}
