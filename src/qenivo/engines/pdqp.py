"""Convex QP by the PDQP first-order method (engine "pdqp").

    min 0.5 x'Qx + c'x + c0   s.t.  lc <= Ax <= uc,  lx <= x <= ux,   Q symmetric PSD

Algorithm: PDQP (Lu & Yang, "A practical and optimal first-order method for large-scale
convex quadratic programming", arXiv 2311.07710), re-implemented in NumPy/CuPy. Per iteration,
with k = iterations since the last restart, beta = 1 + k/2, theta = k/(k+1):
    x+ = clip(x - (eta/w) (c + (1-1/beta) Q xbar + (1/beta) Q x - A'y), lx, ux)
    y+ = sigma (clip(v, lc, uc) - v),  v = A(x+ + theta (x+ - x)) - y/sigma,  sigma = w eta
    xbar, ybar <- weighted averages with weight 1/beta
Restarts on a weighted relative KKT (candidate = the better of current and average,
constants 0.2 / 0.8 / 0.36), primal weight smoothed with 0.2. Same diagonal preconditioning
as the LP engine; termination on the unscaled KKT residuals.
"""
from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sps

from ..certify.kkt import is_optimal, kkt_from_products, rhs_norm
from ..kernels.backend import get_backend
from .pdhg import sigma_max
from .precondition import precondition


def solve_qp(prob, tol=1e-6, backend="numpy", max_iter=1_000_000, time_limit=3600.0,
             check_every=64, verbose=False, seed=0, warm_x=None, warm_y=None, warm_w=None,
             kernels="qenivo") -> dict:
    be = get_backend(backend, kernels)
    xp = be.xp
    t_setup = time.perf_counter()
    m, n = prob.m, prob.n
    K, cs, lcs, ucs, lxs, uxs, sc = precondition(prob.A, prob.c[:, None], prob.lc[:, None],
                                                 prob.uc[:, None], prob.lx[:, None], prob.ux[:, None])
    bb, bc = float(np.reshape(sc.beta_b, -1)[0]), float(np.reshape(sc.beta_c, -1)[0])
    Dc = sps.diags(sc.col)
    Qs = (bb / bc) * (Dc @ prob.Q @ Dc) if prob.is_qp else sps.csr_matrix((n, n))
    Kd, KTd, Qd = be.matrix(K), be.matrix(K.T.tocsr()), be.matrix(sps.csr_matrix(Qs))
    cs, lcs, ucs, lxs, uxs = (be.asarray(v[:, 0]) for v in (cs, lcs, ucs, lxs, uxs))
    row, col = be.asarray(sc.row), be.asarray(sc.col)
    c_o, lc_o, uc_o, lx_o, ux_o = (be.asarray(v) for v in (prob.c, prob.lc, prob.uc, prob.lx, prob.ux))
    bnorm, cnorm = rhs_norm(lc_o, uc_o, xp), xp.linalg.norm(c_o)
    LA = sigma_max(be, Kd, KTd, n, 5000, 1e-4, seed)
    LQ = sigma_max(be, Qd, Qd, n, 5000, 1e-4, seed + 1) if Qs.nnz else 0.0

    def step0(w):
        return 0.99 * 2 / (LQ / w + math.sqrt(4 * LA ** 2 + LQ ** 2 / w ** 2))

    def kkt(xh, yh, Qxh):
        xo, yo = bb * col * xh, bc * row * yh
        return kkt_from_products(xo, yo, bb * Kd.mv(xh) / row, bc * KTd.mv(yh) / col, c_o, prob.c0,
                                 lc_o, uc_o, lx_o, ux_o, bnorm, cnorm, Qx=bc * Qxh / col, xp=xp)

    def wkkt(k, w):
        return max(w * float(k["rel_primal"]), float(k["rel_dual"]) / w, float(k["rel_gap"]))

    w = float(warm_w) if warm_w is not None else 1.0
    x = (xp.clip(be.asarray(np.reshape(warm_x, -1)) / (bb * col), lxs, uxs) if warm_x is not None
         else xp.clip(xp.zeros(n), lxs, uxs))
    y = be.asarray(np.reshape(warm_y, -1)) / (bc * row) if warm_y is not None else xp.zeros(m)
    Ax, ATy, Qx = Kd.mv(x), KTd.mv(y), Qd.mv(x)
    xbar, ybar, Qxbar = x.copy(), y.copy(), Qx.copy()
    x_last, y_last = x.copy(), y.copy()
    eta = step0(w)
    k = total = restarts = 0
    kkt_last_restart, kkt_last_trial = None, math.inf
    status, out = "iteration_limit", None
    be.synchronize()
    setup_time = time.perf_counter() - t_setup
    t0 = time.perf_counter()
    elapsed = 0.0
    kc = None
    next_prog = 0.0
    best_err = math.inf
    last_improve = 0.0
    while total < max_iter:
        for _ in range(check_every):
            beta = 1.0 + k / 2.0
            x_new = xp.clip(x - (eta / w) * (cs + (1 - 1 / beta) * Qxbar + (1 / beta) * Qx - ATy), lxs, uxs)
            Ax_new, Qx_new = Kd.mv(x_new), Qd.mv(x_new)
            theta = k / (k + 1.0)
            sigma = w * eta
            v = Ax_new + theta * (Ax_new - Ax) - y / sigma
            y = sigma * (xp.clip(v, lcs, ucs) - v)
            ATy = KTd.mv(y)
            x, Ax, Qx = x_new, Ax_new, Qx_new
            wt = 1.0 / beta
            xbar += wt * (x - xbar)
            ybar += wt * (y - ybar)
            Qxbar += wt * (Qx - Qxbar)
            k += 1
            it = k + 2
            eta = min(0.99 * it / (LQ / w + math.sqrt(LA ** 2 * it ** 2 + LQ ** 2 / w ** 2)),
                      (it - 1) / (it - 2) * eta)
        total += check_every
        kc, ka = kkt(x, y, Qx), kkt(xbar, ybar, Qxbar)
        elapsed = time.perf_counter() - t0
        if verbose and (total == check_every or elapsed >= next_prog):
            print(
                f"PROGRESS pdqp it={total} t={elapsed:.1f}s "
                f"cur={wkkt(kc, 1):.2e} avg={wkkt(ka, 1):.2e} w={w:.3e}",
                flush=True,
            )
            next_prog = elapsed + 5.0
        for kk_, xx, yy in ((kc, x, y), (ka, xbar, ybar)):
            if bool(is_optimal(kk_, tol, xp)):
                status, out = "optimal", (kk_, xx, yy)
                break
        if status == "optimal" or elapsed > time_limit:
            break
        # Fail-fast: if KKT residual is flat and still far from tol, stop burning wall clock
        cur_err = float(min(wkkt(kc, 1), wkkt(ka, 1)))
        if cur_err < best_err * 0.99:
            best_err = cur_err
            last_improve = elapsed
        elif elapsed - last_improve >= 40.0 and cur_err > max(tol * 1e4, 1e-2):
            if verbose:
                print(
                    f"PROGRESS pdqp stalled cur={cur_err:.2e} no_improve={elapsed - last_improve:.0f}s",
                    flush=True,
                )
            status = "stalled"
            out = (kc, x, y)
            break
        use_avg = wkkt(ka, w) < wkkt(kc, w)
        cand = wkkt(ka, w) if use_avg else wkkt(kc, w)
        if kkt_last_restart is None:
            kkt_last_restart = wkkt(kkt(x_last, y_last, Qd.mv(x_last)), w)
        restart = (k >= 0.36 * total or cand <= 0.2 * kkt_last_restart
                   or (cand <= 0.8 * kkt_last_restart and cand > kkt_last_trial))
        kkt_last_trial = cand
        if restart:
            if use_avg:
                x, y, Qx = xbar.copy(), ybar.copy(), Qxbar.copy()
                Ax, ATy = Kd.mv(x), KTd.mv(y)
            dx, dy = float(xp.linalg.norm(x - x_last)), float(xp.linalg.norm(y - y_last))
            if dx > 1e-12 and dy > 1e-12:
                w = math.exp(0.2 * math.log(dy / dx) + 0.8 * math.log(w))
            x_last, y_last = x.copy(), y.copy()
            xbar, ybar, Qxbar = x.copy(), y.copy(), Qx.copy()
            k = 0
            eta = step0(w)
            restarts += 1
            kkt_last_restart, kkt_last_trial = cand, math.inf
    if out is None:
        out = (kc, x, y)
        if elapsed > time_limit:
            status = "time_limit"
    kk_, xx, yy = out
    xo, yo = be.to_host(bb * col * xx), be.to_host(bc * row * yy)
    be.synchronize()
    return {"status": status, "x": np.asarray(xo), "y": np.asarray(yo), "iterations": int(total),
            "restarts": int(restarts), "time": time.perf_counter() - t0 + setup_time,
            "primal_weight": w, "backend": be.name,
            "kkt": {k_: float(be.to_host(v)) for k_, v in kk_.items()} if kk_ else {}}
