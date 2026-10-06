"""Batched restarted Halpern PDHG with reflection, for LP (engine "pdhg").

Method: restarted Halpern PDHG with reflection (Lu & Yang, arXiv 2407.16144) in the
engineering form of cuPDLPx (Lu, Peng, Yang, arXiv 2507.14051, MIT licence): the
preconditioning order, restart constants and PID primal-weight rule follow that paper and its
public code. This is an independent re-implementation in NumPy/CuPy; no cuPDLPx code is used.

Batch axis: every iterate is (n, S) / (m, S). S = 1 solves one LP; S > 1 solves S LPs that
share A but differ in c, row bounds or column bounds (what-if cases) in one pass, so the
matrix is read once per iteration for all of them. Converged scenarios leave the active set.

One iteration, per scenario, on the scaled problem (tau = eta/w, sigma = eta*w):
    x_hat = clip(x - tau (c - K'y), lx, ux)                PDHG primal step
    x_bar = 2 x_hat - x                                    reflection
    v     = K x_bar - y/sigma
    y_hat = sigma (clip(v, lc, uc) - v)                    PDHG dual step
    y_bar = 2 y_hat - y
    z     = (k/(k+1)) (gamma z_bar + (1-gamma) z) + (1/(k+1)) z_anchor     Halpern
Every `check_every` iterations: KKT of (x_hat, y_hat) on the original problem, restart test,
and (new in QENIVO) infeasibility / unboundedness detection: the change of the dual iterate
between checks is scored as a Farkas ray and the change of the primal iterate as an unbounded
ray (Applegate, Diaz, Lu, Lubin, "Infeasibility detection with primal-dual hybrid gradient",
Math. Programming 2023). A ray is reported only after it passes the host-side check.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..certify.kkt import (
    check_farkas,
    check_ray,
    farkas_products,
    is_optimal,
    kkt_from_products,
    ray_products,
    rhs_norm,
)
from ..kernels.backend import get_backend
from ..kernels.byte_diet import ablation_flags, as_shared_or_batch, s_tile, take_active
from .precondition import precondition, scale_vectors


@dataclass
class PDHGOptions:
    tol: float = 1e-4
    max_iter: int = 1_000_000
    time_limit: float = 3600.0
    check_every: int = 64
    reflection: float = 1.0           # gamma
    step_safety: float = 0.998        # eta = safety / sigma_max(K)
    power_iters: int = 5000
    power_tol: float = 1e-4
    sufficient: float = 0.2
    necessary: float = 0.5
    artificial: float = 0.36
    k_p: float = 0.99
    k_i: float = 0.01
    k_d: float = 0.0
    i_smooth: float = 0.3
    geo_iters: int = 12
    ruiz_iters: int = 10
    pc_alpha: float | None = 1.0
    bound_obj: bool = True
    restarts: bool = True             # ablation switch
    halpern: bool = True              # False -> plain PDHG (ablation)
    fused: bool = True                # GPU only: one CUDA kernel per primal / dual update
    compact: bool = True              # batch: drop finished scenarios from the active set
    detect_infeasibility: bool = True
    infeas_tol: float = 1e-7          # ray violation / ray value to call a scenario infeasible
    infeas_min_iter: int = 512        # do not test rays before this many iterations
    infeas_every: int = 4             # test rays at every 4th check (the ray is the change over 4 checks)
    stall_checks: int = 0             # >0: stop a scenario whose KKT error has not halved in this many checks
    kernels: str = "qenivo"           # GPU sparse products: "qenivo" own kernels | "vendor"
    dtype: str = "float64"            # working precision; "float32" is used by engines/refine.py
    persistent: str = "auto"          # GPU: whole check blocks in one cooperative kernel ("auto", "on", "off")
    persistent_check_every: int = 256 # with the persistent kernel iterations are cheap and checks dominate
    # --- byte-diet (W04); default all-off keeps the f1dc50f baseline path for ablation ---
    diet_f32_cert: bool = False       # A1: iterate f32 at tol>=1e-4; host f64 certify; f64 warm re-solve on fail
    diet_shared_bounds: bool = False  # A2: keep shared lx/ux/lc/uc as (n,1)/(m,1) and S-broadcast in kernels
    diet_l2_tile: bool = False        # A3: process the stack in L2-sized case tiles
    diet_tile_size: int | None = None # A3 override (tests); None -> compute from device L2
    diet_retire_frac: float = 0.0     # A4: compact only when finished/active > this (0 = immediate)
    verbose: bool = False
    seed: int = 0


@dataclass
class BatchResult:
    x: np.ndarray                     # (n, S) primal, original space
    y: np.ndarray                     # (m, S) row duals, original space
    status: list                      # per scenario
    iterations: np.ndarray
    restarts: np.ndarray
    kkt: dict                         # name -> (S,)
    solve_time: float
    setup_time: float
    backend: str
    eta: float
    primal_weight: np.ndarray
    rays: dict = field(default_factory=dict)      # scenario -> {"kind", "vector", "check"}
    history: list = field(default_factory=list)

    def column(self, j: int) -> dict:
        return {"x": self.x[:, j], "y": self.y[:, j], "status": self.status[j],
                "iterations": int(self.iterations[j]), "restarts": int(self.restarts[j]),
                "kkt": {k: float(v[j]) for k, v in self.kkt.items()},
                "primal_weight": float(self.primal_weight[j]), "ray": self.rays.get(j)}


_K: dict = {}


def _build_fused(cp, dtype="float64"):
    """Fused elementwise updates (same arithmetic as the NumPy path, one launch each).
    The float32 pair uses float literals, so no step is promoted to float64."""
    if dtype in _K:
        return _K[dtype]
    T, C = ("double", "") if dtype == "float64" else ("float", "f")

    def f(src):
        return src.replace("{t}", dtype).replace("{T}", T).replace("{C}", C)
    _K[dtype] = {
        "primal": cp.ElementwiseKernel(
            f("{t} x, {t} aty, {t} c, {t} lx, {t} ux, {t} x0, {t} tau, {t} wk, {t} g"),
            f("{t} xh, {t} xbar, {t} xn"),
            f("""
            {T} p = fmin(fmax(x - tau * (c - aty), lx), ux);
            {T} r = 2.0{C} * p - x;
            xh = p;
            xbar = r;
            xn = wk * (g * r + (1.0{C} - g) * x) + (1.0{C} - wk) * x0;
            """), f"qenivo_pdhg_primal_{dtype}"),
        "dual": cp.ElementwiseKernel(
            f("{t} y, {t} kxbar, {t} lc, {t} uc, {t} y0, {t} sigma, {t} wk, {t} g"),
            f("{t} yh, {t} yn"),
            f("""
            {T} v = kxbar - y / sigma;
            {T} h = sigma * (fmin(fmax(v, lc), uc) - v);
            {T} r = 2.0{C} * h - y;
            yh = h;
            yn = wk * (g * r + (1.0{C} - g) * y) + (1.0{C} - wk) * y0;
            """), f"qenivo_pdhg_dual_{dtype}"),
    }
    return _K[dtype]


def _as2d(v, rows, S):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        v = v[:, None]
    return np.ascontiguousarray(np.broadcast_to(v, (rows, S)))

def _col(v, j):
    """Column j of a (rows,S) or shared (rows,1) slab."""
    a = np.asarray(v)
    if a.ndim == 1:
        return a
    return a[:, 0 if a.shape[1] == 1 else j]


def _host_certify_mask(A, C, LC, UC, LX, UX, c0, X, Y, tol, chunk: int = 16) -> np.ndarray:
    """Per-case float64 KKT optimality on the host (original A). True = certified.

    Certified in column chunks to keep peak RAM down on large (n,S) L4 stacks (OOM/TDR risk).
    """
    S = X.shape[1]
    out = np.zeros(S, dtype=bool)
    c0v = np.broadcast_to(np.asarray(c0, dtype=np.float64), (S,))
    for lo in range(0, S, max(1, int(chunk))):
        hi = min(S, lo + chunk)
        sl = slice(lo, hi)
        Ax = A @ X[:, sl]
        ATy = A.T @ Y[:, sl]
        bn = rhs_norm(LC[:, sl] if LC.shape[1] > 1 else LC, UC[:, sl] if UC.shape[1] > 1 else UC)
        cn = np.linalg.norm(C[:, sl], axis=0)
        k = kkt_from_products(X[:, sl], Y[:, sl], Ax, ATy, C[:, sl], c0v[sl],
                              LC[:, sl] if LC.shape[1] > 1 else LC,
                              UC[:, sl] if UC.shape[1] > 1 else UC,
                              LX[:, sl] if LX.shape[1] > 1 else LX,
                              UX[:, sl] if UX.shape[1] > 1 else UX,
                              bn, cn, xp=np)
        out[sl] = np.asarray(is_optimal(k, tol, xp=np)).reshape(-1)
    return out


def _merge_batch_results(parts: list, n: int, m: int, S: int) -> BatchResult:
    """Concatenate per-tile BatchResults into one (column order preserved)."""
    x = np.zeros((n, S))
    y = np.zeros((m, S))
    status = ["iteration_limit"] * S
    iterations = np.zeros(S, dtype=np.int64)
    restarts = np.zeros(S, dtype=np.int64)
    primal_weight = np.full(S, np.nan)
    rays: dict = {}
    history = []
    kkt_keys = set()
    for br in parts:
        kkt_keys.update((br.kkt or {}).keys())
    res_kkt = {k: np.full(S, np.nan) for k in kkt_keys}
    col = 0
    setup_time = 0.0
    solve_time = 0.0
    eta = parts[0].eta if parts else 0.0
    backend = parts[0].backend if parts else "numpy"
    for br in parts:
        w = br.x.shape[1]
        sl = slice(col, col + w)
        x[:, sl], y[:, sl] = br.x, br.y
        status[col:col + w] = list(br.status)
        iterations[sl], restarts[sl] = br.iterations, br.restarts
        primal_weight[sl] = br.primal_weight
        for k, v in (br.kkt or {}).items():
            res_kkt[k][sl] = v
        for j, ray in br.rays.items():
            rays[col + int(j)] = ray
        history.extend(br.history)
        setup_time += br.setup_time
        solve_time += br.solve_time
        col += w
    return BatchResult(x=x, y=y, status=status, iterations=iterations, restarts=restarts,
                       kkt=res_kkt, solve_time=solve_time, setup_time=setup_time,
                       backend=backend, eta=eta, primal_weight=primal_weight,
                       rays=rays, history=history)


def sigma_max(be, K, KT, n, iters=5000, tol=1e-4, seed=0):
    """Largest singular value of K by power iteration on K'K."""
    xp = be.xp
    if K.shape[0] == 0 or K.nnz == 0 or n == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    v = be.asarray(rng.standard_normal(n))
    v /= xp.linalg.norm(v)
    est = 0.0
    for _ in range(iters):
        w = KT.mv(K.mv(v))
        nw = float(xp.linalg.norm(w))
        if nw == 0.0:
            return 0.0
        new = math.sqrt(nw)
        v = w / nw
        if abs(new - est) <= tol * new:
            return new
        est = new
    return est


def solve_batch(A, c, lc, uc, lx, ux, c0=0.0, opts: PDHGOptions | None = None,
                backend: str = "numpy", warm_x=None, warm_y=None, warm_w=None,
                problem=None, norms=None, cache: dict | None = None) -> BatchResult:
    """Solve S LPs sharing A. c/lx/ux: (n,) or (n,S); lc/uc: (m,) or (m,S); c0 scalar or (S,).

    warm_x (n,S), warm_y (m,S): original-space starting points; warm_w (S,): primal weights.
    `problem` (optional) is used only to label rays with the host-side check.
    norms=(bnorm, cnorm), each scalar or (S,): replace ||b|| and ||c|| in the relative
    tolerances, so a correction problem is solved to an absolute accuracy (engines/refine.py).
    cache: a dict the caller keeps across solves that share A. It holds the matrix scaling, the
    device matrices and the step size. Used only with opts.bound_obj False, since that scaling
    then depends on A alone.

    Byte-diet flags (opts.diet_*): default off = baseline path. With diet_f32_cert, answers that
    look optimal in working precision are re-checked in float64 on the host; any failure is
    re-solved in float64 from the float32 warm point and never reported unproven-as-ok.
    """
    opts = opts or PDHGOptions()
    m, n = A.shape
    S = max(np.ndim(c) == 2 and np.shape(c)[1] or 1,
            *[np.ndim(v) == 2 and np.shape(v)[1] or 1 for v in (lc, uc, lx, ux)])

    # A1: f32 is a 1e-4 engine; tighter tolerances stay on (or fall back to) float64.
    if opts.diet_f32_cert and opts.tol < 1e-4 * (1.0 - 1e-12):
        opts = PDHGOptions(**{**opts.__dict__, "dtype": "float64", "diet_f32_cert": False})
    elif opts.diet_f32_cert and str(opts.dtype) in ("float64", "f64", "<f8"):
        opts = PDHGOptions(**{**opts.__dict__, "dtype": "float32"})

    # A3: outer L2 tiling — independent contiguous case tiles, then merge.
    if opts.diet_l2_tile and S > 1:
        st = int(opts.diet_tile_size) if opts.diet_tile_size else s_tile(
            n, m, opts.dtype, shared_bounds=opts.diet_shared_bounds)
        if S > st:
            C = _as2d(c, n, S)
            LX, UX = (_as2d(v, n, S) for v in (lx, ux))
            LC, UC = (_as2d(v, m, S) for v in (lc, uc))
            c0v = np.broadcast_to(np.asarray(c0, dtype=np.float64), (S,)).copy()
            tile_opts = PDHGOptions(**{**opts.__dict__, "diet_l2_tile": False, "diet_f32_cert": False})
            parts = []
            t_remain = opts.time_limit
            for lo in range(0, S, st):
                hi = min(S, lo + st)
                if t_remain <= 0:
                    br = BatchResult(
                        x=np.zeros((n, hi - lo)), y=np.zeros((m, hi - lo)),
                        status=["time_limit"] * (hi - lo),
                        iterations=np.zeros(hi - lo, dtype=np.int64),
                        restarts=np.zeros(hi - lo, dtype=np.int64), kkt={},
                        solve_time=0.0, setup_time=0.0, backend=backend, eta=0.0,
                        primal_weight=np.full(hi - lo, np.nan))
                else:
                    to = PDHGOptions(**{**tile_opts.__dict__, "time_limit": t_remain})
                    wx = None if warm_x is None else _as2d(warm_x, n, S)[:, lo:hi]
                    wy = None if warm_y is None else _as2d(warm_y, m, S)[:, lo:hi]
                    ww = None if warm_w is None else np.broadcast_to(
                        np.asarray(warm_w, dtype=np.float64), (S,))[lo:hi]
                    nrm = None
                    if norms is not None:
                        nrm = tuple(np.broadcast_to(np.asarray(v, dtype=np.float64), (S,))[lo:hi]
                                    for v in norms)
                    br = solve_batch(A, C[:, lo:hi], LC[:, lo:hi], UC[:, lo:hi],
                                     LX[:, lo:hi], UX[:, lo:hi], c0v[lo:hi], to, backend,
                                     warm_x=wx, warm_y=wy, warm_w=ww, problem=problem,
                                     norms=nrm, cache=cache)
                    t_remain -= br.setup_time + br.solve_time
                parts.append(br)
            out = _merge_batch_results(parts, n, m, S)
            if opts.diet_f32_cert:
                out = _certify_f32_with_f64_fallback(
                    A, C, LC, UC, LX, UX, c0v, out, opts, backend, problem=problem, cache=cache)
            return out

    return _solve_batch_core(A, c, lc, uc, lx, ux, c0, opts, backend, warm_x, warm_y, warm_w,
                             problem, norms, cache)


def _certify_f32_with_f64_fallback(A, C, LC, UC, LX, UX, c0, br: BatchResult, opts, backend,
                                   problem=None, cache=None) -> BatchResult:
    """Host f64 certify every 'optimal' case; re-solve failures in f64 from the f32 warm point."""
    import scipy.sparse as sps
    A = sps.csr_matrix(A, dtype=np.float64)
    S = br.x.shape[1]
    C = _as2d(C, A.shape[1], S)
    LX, UX = (_as2d(v, A.shape[1], S) for v in (LX, UX))
    LC, UC = (_as2d(v, A.shape[0], S) for v in (LC, UC))
    c0v = np.broadcast_to(np.asarray(c0, dtype=np.float64), (S,)).copy()
    opt = np.array([s == "optimal" for s in br.status])
    if not opt.any():
        return br
    ok = np.zeros(S, dtype=bool)
    ok[opt] = _host_certify_mask(A, C[:, opt], LC[:, opt], UC[:, opt], LX[:, opt], UX[:, opt],
                                 c0v[opt], br.x[:, opt], br.y[:, opt], opts.tol)
    fail = np.flatnonzero(opt & ~ok)
    for j in fail:
        br.status[j] = "iteration_limit"
    if fail.size == 0:
        return br
    o64 = PDHGOptions(**{**opts.__dict__, "dtype": "float64", "diet_f32_cert": False,
                         "diet_l2_tile": False})
    sub = _solve_batch_core(A, C[:, fail], LC[:, fail], UC[:, fail], LX[:, fail], UX[:, fail],
                            c0v[fail], o64, backend, warm_x=br.x[:, fail], warm_y=br.y[:, fail],
                            warm_w=br.primal_weight[fail], problem=problem, norms=None, cache=cache)
    br.x[:, fail], br.y[:, fail] = sub.x, sub.y
    for i, j in enumerate(fail):
        br.status[j] = sub.status[i]
        br.iterations[j] += int(sub.iterations[i])
        br.restarts[j] += int(sub.restarts[i])
        if sub.kkt:
            for k, v in sub.kkt.items():
                if k not in br.kkt:
                    br.kkt[k] = np.full(S, np.nan)
                br.kkt[k][j] = v[i]
        br.primal_weight[j] = sub.primal_weight[i]
        if i in sub.rays:
            br.rays[int(j)] = sub.rays[i]
    br.solve_time += sub.solve_time
    br.setup_time += sub.setup_time
    still = np.array([s == "optimal" for s in br.status])
    if still.any():
        certified = np.zeros(S, dtype=bool)
        certified[still] = _host_certify_mask(
            A, C[:, still], LC[:, still], UC[:, still], LX[:, still], UX[:, still],
            c0v[still], br.x[:, still], br.y[:, still], opts.tol)
        for j in np.flatnonzero(still & ~certified):
            br.status[j] = "iteration_limit"
    return br


def _solve_batch_core(A, c, lc, uc, lx, ux, c0=0.0, opts: PDHGOptions | None = None,
                      backend: str = "numpy", warm_x=None, warm_y=None, warm_w=None,
                      problem=None, norms=None, cache: dict | None = None) -> BatchResult:
    opts = opts or PDHGOptions()
    be = get_backend(backend, opts.kernels, opts.dtype)
    xp = be.xp
    dt = be.dtype
    t_setup = time.perf_counter()

    m, n = A.shape
    S = max(np.ndim(c) == 2 and np.shape(c)[1] or 1,
            *[np.ndim(v) == 2 and np.shape(v)[1] or 1 for v in (lc, uc, lx, ux)])
    prefer_shared = bool(opts.diet_shared_bounds)
    c = as_shared_or_batch(c, n, S, prefer_shared=False)
    if prefer_shared:
        lx = as_shared_or_batch(lx, n, S, True)
        ux = as_shared_or_batch(ux, n, S, True)
        lc = as_shared_or_batch(lc, m, S, True)
        uc = as_shared_or_batch(uc, m, S, True)
        if lx.shape[1] != 1 or ux.shape[1] != 1:
            lx, ux = _as2d(lx, n, S), _as2d(ux, n, S)
        if lc.shape[1] != 1 or uc.shape[1] != 1:
            lc, uc = _as2d(lc, m, S), _as2d(uc, m, S)
    else:
        lx, ux = _as2d(lx, n, S), _as2d(ux, n, S)
        lc, uc = _as2d(lc, m, S), _as2d(uc, m, S)
    if c.shape[1] == 1 and S > 1:
        c = _as2d(c, n, S)
    c0 = np.broadcast_to(np.asarray(c0, dtype=np.float64), (S,)).copy()

    use_cache = cache is not None and not opts.bound_obj
    low = dt != np.float64
    if use_cache and "K" in cache:
        K, Kd, KTd, KTd64 = cache["K"], cache["Kd"], cache["KTd"], cache["KTd64"]
        cs, lcs, ucs, lxs, uxs, sc = scale_vectors(cache["row"], cache["col"], c, lc, uc, lx, ux, False)
    else:
        K, cs, lcs, ucs, lxs, uxs, sc = precondition(A, c, lc, uc, lx, ux, opts.geo_iters,
                                                     opts.ruiz_iters, opts.pc_alpha, opts.bound_obj)
        KT = K.T.tocsr()
        Kd, KTd = be.matrix(K), be.matrix(KT)
        KTd64 = get_backend(backend, opts.kernels, np.float64).matrix(KT) if low else KTd
        if use_cache:
            cache.update(K=K, Kd=Kd, KTd=KTd, KTd64=KTd64, row=sc.row, col=sc.col)
    cs, lcs, ucs, lxs, uxs = (be.asarray(v) for v in (cs, lcs, ucs, lxs, uxs))
    row, col = be.asarray(sc.row)[:, None], be.asarray(sc.col)[:, None]
    bb = be.asarray(np.broadcast_to(sc.beta_b, (S,)))[None, :]
    bc = be.asarray(np.broadcast_to(sc.beta_c, (S,)))[None, :]
    c_o = be.asarray(c)
    lc_o, uc_o, lx_o, ux_o = (be.asarray(v) for v in (lc, uc, lx, ux))
    c0_d = be.asarray(c0)
    if norms is not None:
        bnorm, cnorm = (be.asarray(np.broadcast_to(np.asarray(v, dtype=np.float64), (S,)).copy())
                        for v in norms)
    else:
        lc_b = lc_o if (getattr(lc_o, "ndim", 1) == 1 or lc_o.shape[1] > 1) else xp.broadcast_to(lc_o, (m, S))
        uc_b = uc_o if (getattr(uc_o, "ndim", 1) == 1 or uc_o.shape[1] > 1) else xp.broadcast_to(uc_o, (m, S))
        bnorm, cnorm = rhs_norm(lc_b, uc_b, xp), xp.linalg.norm(c_o, axis=0)
    mv_K, mv_KT = Kd.mv, KTd.mv

    if use_cache and "smax" in cache:
        smax = cache["smax"]
    else:
        smax = sigma_max(be, Kd, KTd, n, opts.power_iters, opts.power_tol, opts.seed)
        if use_cache:
            cache["smax"] = smax
    eta = opts.step_safety / smax if smax > 0 else 1.0

    if warm_x is not None:
        x = xp.clip(be.asarray(_as2d(warm_x, n, S)) / (bb * col), lxs, uxs)
    else:
        x = xp.clip(xp.zeros((n, S), dtype=dt), lxs, uxs)
    y = be.asarray(_as2d(warm_y, m, S)) / (bc * row) if warm_y is not None else xp.zeros((m, S), dtype=dt)
    w = (be.asarray(np.broadcast_to(np.asarray(warm_w, dtype=np.float64), (S,)))[None, :]
         if warm_w is not None else xp.ones((1, S), dtype=dt))
    x0, y0 = x.copy(), y.copy()
    best_w = w.copy()
    best_gap = xp.full((1, S), xp.inf, dtype=dt)
    err_sum = xp.zeros((1, S), dtype=dt)
    last_err = xp.zeros((1, S), dtype=dt)
    kk = xp.ones((1, S), dtype=dt)
    inner = np.zeros(S, dtype=np.int64)
    fpe_init = xp.full((1, S), xp.inf, dtype=dt)
    fpe_last = xp.full((1, S), xp.inf, dtype=dt)
    need_init = np.zeros(S, dtype=bool)
    act = np.arange(S)
    n_restart = np.zeros(S, dtype=np.int64)
    done = np.zeros(S, dtype=bool)
    status = ["iteration_limit"] * S
    iters_done = np.zeros(S, dtype=np.int64)
    res_x, res_y = np.zeros((n, S)), np.zeros((m, S))
    res_w = np.full(S, np.nan)
    res_kkt = None
    best_err = np.full(S, np.inf)
    since_best = np.zeros(S, dtype=np.int64)
    rays: dict = {}
    history = []
    gamma = opts.reflection
    xh = yh = None
    prev = None

    fused = be.is_gpu and opts.fused and opts.halpern
    fkern = _build_fused(xp, dt.name) if fused else None
    # Persistent kernel supports shared bounds via shared_b (W04 A2).
    persist = (fused and opts.kernels != "vendor"
               and _use_persistent(opts.persistent, K.nnz, m, n, S))
    if persist:
        from ..kernels import persistent as _pk
        opts = PDHGOptions(**{**opts.__dict__, "check_every": max(opts.check_every, opts.persistent_check_every)})

    def fpe_of(xa, xha, ya, yha):
        if low:
            dx = xa.astype(np.float64) - xha.astype(np.float64)
            dy = ya.astype(np.float64) - yha.astype(np.float64)
            w64 = w.astype(np.float64)
            mov = w64 * xp.sum(dx * dx, axis=0) + xp.sum(dy * dy, axis=0) / w64
            inter = 2.0 * eta * xp.sum(KTd64.mv(dy) * dx, axis=0)
            return xp.sqrt(xp.maximum(mov + inter, 0.0)).astype(dt)
        dx, dy = xa - xha, ya - yha
        mov = w * xp.sum(dx * dx, axis=0) + xp.sum(dy * dy, axis=0) / w
        inter = 2.0 * eta * xp.sum(mv_KT(dy) * dx, axis=0)
        return xp.sqrt(xp.maximum(mov + inter, 0.0))
    be.synchronize()
    setup_time = time.perf_counter() - t_setup
    t0 = time.perf_counter()
    total = 0
    while total < opts.max_iter:
        tau, sigma = eta / w, eta * w
        fpe = None
        if persist:
            if need_init.any():
                xs, ys = x.copy(), y.copy()
            shared_b = prefer_shared and int(getattr(lxs, "shape", (0, 2))[1] == 1)
            xh, yh, xprev, yprev, xh0, yh0 = _pk.run_block(
                Kd, KTd, cs, lxs, uxs, lcs, ucs, x0, y0, x, y,
                tau, sigma, kk, gamma, opts.check_every, shared_bounds=bool(shared_b))
            if need_init.any():
                fpe_init = xp.where(xp.asarray(need_init)[None, :], fpe_of(xs, xh0, ys, yh0), fpe_init)
                need_init[:] = False
            fpe = fpe_of(xprev, xh, yprev, yh)
            kk = kk + float(opts.check_every)
        for it in range(0 if persist else opts.check_every):
            ATy = mv_KT(y)
            if fused:
                wk = kk / (kk + 1.0)
                xh, xbar, xn = fkern["primal"](x, ATy, cs, lxs, uxs, x0, tau, wk, dt.type(gamma))
                yh, yn = fkern["dual"](y, mv_K(xbar), lcs, ucs, y0, sigma, wk, dt.type(gamma))
            else:
                xh = xp.clip(x - tau * (cs - ATy), lxs, uxs)
                xbar = 2.0 * xh - x
                v = mv_K(xbar) - y / sigma
                yh = sigma * (xp.clip(v, lcs, ucs) - v)
                ybar = 2.0 * yh - y
            last = it == opts.check_every - 1
            if (it == 0 and need_init.any()) or last:
                f = fpe_of(x, xh, y, yh)
                if it == 0 and need_init.any():
                    fpe_init = xp.where(xp.asarray(need_init)[None, :], f, fpe_init)
                    need_init[:] = False
                if last:
                    fpe = f
            if fused:
                x, y = xn, yn
                kk = kk + 1.0
            elif opts.halpern:
                wk = kk / (kk + 1.0)
                x = wk * (gamma * xbar + (1.0 - gamma) * x) + (1.0 - wk) * x0
                y = wk * (gamma * ybar + (1.0 - gamma) * y) + (1.0 - wk) * y0
                kk = kk + 1.0
            else:
                x, y = xh, yh
        total += opts.check_every
        inner[act] += opts.check_every

        xo = bb * col * xh
        yo = bc * row * yh
        Ax = bb * mv_K(xh) / row
        ATyo = bc * mv_KT(yh) / col
        kkt = kkt_from_products(xo, yo, Ax, ATyo, c_o, c0_d, lc_o, uc_o, lx_o, ux_o, bnorm, cnorm, xp=xp)
        opt_a = be.to_host(is_optimal(kkt, opts.tol, xp)).reshape(-1)
        stall_a = np.zeros(len(act), dtype=bool)
        if opts.stall_checks > 0:
            e_a = be.to_host(xp.maximum(xp.maximum(kkt["rel_primal"], kkt["rel_dual"]), kkt["rel_gap"])).reshape(-1)
            better = e_a < 0.5 * best_err[act]
            best_err[act] = np.where(better, e_a, best_err[act])
            since_best[act] = np.where(better, 0, since_best[act] + 1)
            stall_a = since_best[act] > opts.stall_checks
        if res_kkt is None:
            res_kkt = {k: np.full(S, np.nan) for k in kkt}

        infeas_a = np.zeros(len(act), dtype=bool)
        unbnd_a = np.zeros(len(act), dtype=bool)
        ray_check = (total // opts.check_every) % opts.infeas_every == 0
        if opts.detect_infeasibility and prev is not None and total >= opts.infeas_min_iter and ray_check:
            pxo, pyo, pAx, pATy = prev
            dyr, dATy = yo - pyo, ATyo - pATy
            ny = xp.sqrt(xp.sum(dyr * dyr, axis=0))
            val, vio = farkas_products(dyr, dATy, lc_o, uc_o, lx_o, ux_o, xp=xp)
            fk = be.to_host((val > 0) & (vio <= opts.infeas_tol * val) & (ny > 0)).reshape(-1)
            dxr, dAx = xo - pxo, Ax - pAx
            nx = xp.sqrt(xp.sum(dxr * dxr, axis=0))
            dsc, vio2 = ray_products(dxr, dAx, c_o, lc_o, uc_o, lx_o, ux_o, xp=xp)
            rk = be.to_host((dsc > 0) & (vio2 <= opts.infeas_tol * dsc) & (nx > 0)).reshape(-1)
            not_opt = ~opt_a
            for a_ in np.flatnonzero((fk | rk) & not_opt):
                j = act[a_]
                if fk[a_]:
                    vec = be.to_host(dyr[:, a_])
                    chk = check_farkas(_ray_problem(problem, lc, uc, lx, ux, j, A), vec)
                    if chk["valid"]:
                        infeas_a[a_] = True
                        rays[j] = {"kind": "farkas", "vector": vec / np.linalg.norm(vec), "check": chk}
                        continue
                if rk[a_]:
                    vec = be.to_host(dxr[:, a_])
                    chk = check_ray(_ray_problem(problem, lc, uc, lx, ux, j, A, c=c), vec)
                    if chk["valid"]:
                        unbnd_a[a_] = True
                        rays[j] = {"kind": "unbounded_ray", "vector": vec / np.linalg.norm(vec), "check": chk}
        if ray_check or prev is None:
            prev = (xo, yo, Ax, ATyo)

        new_a = (opt_a | infeas_a | unbnd_a | stall_a) & ~done[act]
        elapsed = time.perf_counter() - t0
        final = total >= opts.max_iter or elapsed > opts.time_limit
        upd_a = new_a | (~done[act] & final)
        if upd_a.any():
            hx, hy = be.to_host(xo), be.to_host(yo)
            hk = {k: be.to_host(v).reshape(-1) for k, v in kkt.items()}
            cols = act[upd_a]
            res_x[:, cols], res_y[:, cols] = hx[:, upd_a], hy[:, upd_a]
            for k in hk:
                res_kkt[k][cols] = hk[k][upd_a]
            for a_ in np.flatnonzero(new_a):
                j = act[a_]
                status[j] = ("optimal" if opt_a[a_] else "infeasible" if infeas_a[a_]
                             else "unbounded" if unbnd_a[a_] else "stalled")
            if elapsed > opts.time_limit:
                for j in act[upd_a & ~new_a]:
                    status[j] = "time_limit"
            iters_done[cols] = total
            res_w[cols] = be.to_host(w).reshape(-1)[upd_a]
            done[act[new_a]] = True
        if opts.verbose:
            hk0 = {k: float(be.to_host(v).reshape(-1)[0]) for k, v in kkt.items()}
            print(f"{total:8d} {elapsed:8.2f}s  rp {hk0['rel_primal']:.2e} rd {hk0['rel_dual']:.2e} "
                  f"gap {hk0['rel_gap']:.2e}  obj {hk0['pobj']:.8e}  active {len(act)}")
        history.append({"iter": total, "time": elapsed, "cols": act.tolist(),
                        "rel_primal": be.to_host(kkt["rel_primal"]).reshape(-1).tolist(),
                        "rel_dual": be.to_host(kkt["rel_dual"]).reshape(-1).tolist(),
                        "rel_gap": be.to_host(kkt["rel_gap"]).reshape(-1).tolist()})
        if done.all() or final:
            break

        if opts.restarts:
            f_h, fi_h, fl_h = (be.to_host(a).reshape(-1) for a in (fpe, fpe_init, fpe_last))
            restart = np.zeros(len(act), dtype=bool)
            first_check = total == opts.check_every
            for a_, j in enumerate(act):
                if done[j]:
                    continue
                if first_check or f_h[a_] <= opts.sufficient * fi_h[a_]:
                    restart[a_] = True
                elif f_h[a_] <= opts.necessary * fi_h[a_] and f_h[a_] > fl_h[a_]:
                    restart[a_] = True
                elif inner[j] >= opts.artificial * total:
                    restart[a_] = True
            fpe_last = fpe
            if restart.any():
                r = xp.asarray(restart)[None, :]
                pdist = xp.sqrt(xp.sum((xh - x0) ** 2, axis=0))[None, :]
                ddist = xp.sqrt(xp.sum((yh - y0) ** 2, axis=0))[None, :]
                with np.errstate(all="ignore"):
                    ratio = kkt["rel_dual"][None, :] / kkt["rel_primal"][None, :]
                valid = ((pdist > 1e-16) & (ddist > 1e-16) & (pdist < 1e12) & (ddist < 1e12)
                         & (ratio > 1e-8) & (ratio < 1e8))
                with np.errstate(all="ignore"):
                    err = xp.log(xp.maximum(ddist, 1e-300)) - xp.log(xp.maximum(pdist, 1e-300)) - xp.log(w)
                esum_new = opts.i_smooth * err_sum + err
                w_pid = w * xp.exp(opts.k_p * err + opts.k_i * esum_new + opts.k_d * (err - last_err))
                w_new = xp.where(valid, w_pid, best_w)
                err_sum = xp.where(r, xp.where(valid, esum_new, 0.0), err_sum)
                last_err = xp.where(r, xp.where(valid, err, 0.0), last_err)
                w = xp.where(r, w_new, w)
                with np.errstate(all="ignore"):
                    bal = xp.abs(xp.log10(ratio))
                better = r & (bal < best_gap)
                best_gap = xp.where(better, bal, best_gap)
                best_w = xp.where(better, w, best_w)
                x0 = xp.where(r, xh, x0)
                y0 = xp.where(r, yh, y0)
                x = xp.where(r, xh, x)
                y = xp.where(r, yh, y)
                kk = xp.where(r, 1.0, kk)
                fpe_last = xp.where(r, xp.inf, fpe_last)
                inner[act[restart]] = 0
                need_init |= restart
                n_restart[act[restart]] += 1

        # A4: compact when finished/active > diet_retire_frac (0 => immediate, baseline).
        keep = ~done[act]
        n_fin = int((~keep).sum())
        frac_done = n_fin / max(len(act), 1)
        should_compact = (opts.compact and n_fin > 0
                          and (frac_done > opts.diet_retire_frac or keep.sum() == 0))
        if should_compact and not keep.all():
            kd = xp.asarray(np.flatnonzero(keep))
            x, y, x0, y0 = x[:, kd], y[:, kd], x0[:, kd], y0[:, kd]
            cs = take_active(cs, kd, xp)
            lcs, ucs = take_active(lcs, kd, xp), take_active(ucs, kd, xp)
            lxs, uxs = take_active(lxs, kd, xp), take_active(uxs, kd, xp)
            c_o = take_active(c_o, kd, xp)
            lc_o, uc_o = take_active(lc_o, kd, xp), take_active(uc_o, kd, xp)
            lx_o, ux_o = take_active(lx_o, kd, xp), take_active(ux_o, kd, xp)
            bb, bc, w, best_w, best_gap = bb[:, kd], bc[:, kd], w[:, kd], best_w[:, kd], best_gap[:, kd]
            err_sum, last_err, kk = err_sum[:, kd], last_err[:, kd], kk[:, kd]
            fpe_init, fpe_last = fpe_init[:, kd], fpe_last[:, kd]
            bnorm, cnorm, c0_d = bnorm[kd], cnorm[kd], c0_d[kd]
            if prev is not None:
                prev = tuple(take_active(p, kd, xp) for p in prev)
            need_init = need_init[keep]
            act = act[keep]

    be.synchronize()
    solve_time = time.perf_counter() - t0
    out = BatchResult(x=res_x, y=res_y, status=status, iterations=iters_done,
                      restarts=n_restart, kkt=res_kkt or {}, solve_time=solve_time,
                      setup_time=setup_time, backend=be.name, eta=eta,
                      primal_weight=res_w, rays=rays, history=history)
    if opts.diet_f32_cert:
        C_h = _as2d(c, n, S)
        LX_h, UX_h = (_as2d(v, n, S) for v in (lx, ux))
        LC_h, UC_h = (_as2d(v, m, S) for v in (lc, uc))
        out = _certify_f32_with_f64_fallback(A, C_h, LC_h, UC_h, LX_h, UX_h, c0, out, opts,
                                             backend, problem=problem, cache=cache)
    return out


PERSISTENT_MAX_WORK = 4_000_000       # (nnz + m + n) * S per iteration; above this, launches are cheap


def _use_persistent(mode, nnz, m, n, S) -> bool:
    if mode == "off":
        return False
    if mode == "on":
        return True
    return (nnz + m + n) * S <= PERSISTENT_MAX_WORK


class _RayView:
    """Minimal problem view for the host-side ray checks of scenario j."""

    def __init__(self, A, c, lc, uc, lx, ux):
        self.A, self.c, self.lc, self.uc, self.lx, self.ux = A, c, lc, uc, lx, ux
        self.Q = None
        self.is_qp = False


def _ray_problem(problem, lc, uc, lx, ux, j, A, c=None):
    return _RayView(A, None if c is None else _col(c, j),
                    _col(lc, j), _col(uc, j), _col(lx, j), _col(ux, j))


def solve(prob, tol=1e-4, backend="numpy", opts: PDHGOptions | None = None,
          warm_x=None, warm_y=None, warm_w=None, **kw) -> BatchResult:
    """Solve one Problem (S = 1)."""
    opts = opts or PDHGOptions()
    opts.tol = tol
    for k, v in kw.items():
        if hasattr(opts, k):
            setattr(opts, k, v)
    if prob.is_qp:
        raise ValueError("pdhg is an LP engine; use the pdqp engine for QP")
    return solve_batch(prob.A, prob.c, prob.lc, prob.uc, prob.lx, prob.ux, prob.c0, opts, backend,
                       warm_x=warm_x, warm_y=warm_y, warm_w=warm_w, problem=prob)


def make_ablation_options(name: str, **kw) -> PDHGOptions:
    """Build PDHGOptions for a named cumulative ablation (baseline / A1 / A2 / A3 / A4)."""
    flags = ablation_flags(name)
    flags.update(kw)
    return PDHGOptions(**{k: v for k, v in flags.items() if k in PDHGOptions.__dataclass_fields__})
