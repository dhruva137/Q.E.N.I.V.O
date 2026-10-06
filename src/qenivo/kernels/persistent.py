"""Persistent PDHG: a whole block of iterations in ONE kernel launch.

Measured on the GPU path: for refinery-sized LPs (10^3 - 10^5 nonzeros) an iteration costs
about 0.5 ms whatever the precision, because each iteration is four kernel launches driven
from Python; the GPU itself is idle most of that time. This kernel keeps the iterate on the GPU
and runs `iters` restarted-Halpern PDHG iterations for all S scenarios of a batch inside one
cooperative launch, separated by grid-wide barriers:

    for it in 0..iters-1:
        A  aty  = K' y                     (row of K' per thread, scenario axis contiguous)
        B  x    = Halpern(x, reflect(clip(x - tau (c - aty))))       and keep x_bar
        C  kxb  = K x_bar
        D  y    = Halpern(y, reflect(sigma (clip(v) - v))),  v = kxb - y / sigma

Arithmetic is the same as the fused kernels in engines/pdhg.py, in the same order, so the result
is identical to the launch-per-step path (checked by the tests). There are no atomics: every
output is written by one thread with a fixed summation order, so runs are bit-for-bit
repeatable. The kernel also records what the restart logic needs: the first and the last
iteration's T(z) and the iterate before the last step. Compiled once per device and precision
through NVRTC (no CUDA toolkit build step).
"""
from __future__ import annotations

import numpy as np

_TMPL = r"""
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__ void pdhg_block_{SUF}(
    const int m, const int n, const int S, const int iters, const int shared_b,
    const int* __restrict__ Kp, const int* __restrict__ Ki, const {T}* __restrict__ Kv,
    const int* __restrict__ Tp, const int* __restrict__ Ti, const {T}* __restrict__ Tv,
    const {T}* __restrict__ c, const {T}* __restrict__ lx, const {T}* __restrict__ ux,
    const {T}* __restrict__ lc, const {T}* __restrict__ uc,
    const {T}* __restrict__ x0, const {T}* __restrict__ y0,
    {T}* __restrict__ x, {T}* __restrict__ y,
    {T}* __restrict__ aty, {T}* __restrict__ xbar, {T}* __restrict__ kxb,
    {T}* __restrict__ xh, {T}* __restrict__ yh, {T}* __restrict__ xprev, {T}* __restrict__ yprev,
    {T}* __restrict__ xh0, {T}* __restrict__ yh0,
    const {T}* __restrict__ tau, const {T}* __restrict__ sigma, const {T}* __restrict__ kk0,
    const {T} g)
{
    cg::grid_group grid = cg::this_grid();
    const long N = (long)n * S, M = (long)m * S;
    const long start = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = (long)gridDim.x * blockDim.x;
    const {T} one = ({T})1, two = ({T})2;
    for (int it = 0; it < iters; ++it) {
        for (long idx = start; idx < N; idx += stride) {                 // A: aty = K' y
            const long j = idx / S; const int s = (int)(idx - j * S);
            {T} acc = ({T})0;
            for (int k = Tp[j]; k < Tp[j + 1]; ++k) acc += Tv[k] * y[(long)Ti[k] * S + s];
            aty[idx] = acc;
        }
        grid.sync();
        for (long idx = start; idx < N; idx += stride) {                 // B: primal step
            const long j = idx / S; const int s = (int)(idx - j * S);
            const {T} kk = kk0[s] + ({T})it;
            const {T} wk = kk / (kk + one);
            const {T} xo = x[idx];
            const long bi = shared_b ? j : idx;                          // shared lx/ux: (n,1)
            const {T} p = fmin(fmax(xo - tau[s] * (c[idx] - aty[idx]), lx[bi]), ux[bi]);
            const {T} r = two * p - xo;
            xbar[idx] = r;
            if (it == 0) xh0[idx] = p;
            if (it == iters - 1) { xh[idx] = p; xprev[idx] = xo; }
            x[idx] = wk * (g * r + (one - g) * xo) + (one - wk) * x0[idx];
        }
        grid.sync();
        for (long idx = start; idx < M; idx += stride) {                 // C: kxb = K x_bar
            const long i = idx / S; const int s = (int)(idx - i * S);
            {T} acc = ({T})0;
            for (int k = Kp[i]; k < Kp[i + 1]; ++k) acc += Kv[k] * xbar[(long)Ki[k] * S + s];
            kxb[idx] = acc;
        }
        grid.sync();
        for (long idx = start; idx < M; idx += stride) {                 // D: dual step
            const long i = idx / S; const int s = (int)(idx - i * S);
            const {T} kk = kk0[s] + ({T})it;
            const {T} wk = kk / (kk + one);
            const {T} yo = y[idx];
            const {T} v = kxb[idx] - yo / sigma[s];
            const long bi = shared_b ? i : idx;                          // shared lc/uc: (m,1)
            const {T} h = sigma[s] * (fmin(fmax(v, lc[bi]), uc[bi]) - v);
            const {T} r = two * h - yo;
            if (it == 0) yh0[idx] = h;
            if (it == iters - 1) { yh[idx] = h; yprev[idx] = yo; }
            y[idx] = wk * (g * r + (one - g) * yo) + (one - wk) * y0[idx];
        }
        grid.sync();
    }
}
"""

_CACHE: dict = {}


def _kernel(dtype):
    import cupy as cp
    dev = cp.cuda.Device().id
    key = (dev, np.dtype(dtype).name, "shared_b")
    if key not in _CACHE:
        T, suf = ("double", "f64") if np.dtype(dtype) == np.float64 else ("float", "f32")
        src = _TMPL.replace("{T}", T).replace("{SUF}", suf)
        k = None
        for std in ("-std=c++17", "-std=c++14"):          # cooperative_groups needs C++17 on new CUDA
            try:
                k = cp.RawKernel(src, f"pdhg_block_{suf}", options=(std,), enable_cooperative_groups=True)
                k.compile()
                break
            except Exception:  # noqa: BLE001
                k = None
        if k is None:
            raise RuntimeError("persistent PDHG kernel does not compile on this device")
        _CACHE[key] = {"k": k, "grid": None}
    return _CACHE[key]


def _grid(entry, threads):
    """Blocks for a cooperative launch: every block must be resident at once, or grid.sync() is undefined.

    Sized from the CUDA occupancy query for this kernel (registers decide it), because an oversized
    cooperative launch is not reported as an error by every CuPy/driver combination: it can run a
    partial grid silently.
    """
    import cupy as cp
    if entry["grid"] is None:
        sms = cp.cuda.Device().attributes["MultiProcessorCount"]
        try:
            per_sm = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(entry["k"].kernel.ptr, threads, 0)
        except Exception:  # noqa: BLE001
            per_sm = max(1, 65536 // max(1, entry["k"].num_regs * threads))   # register-file bound
        entry["grid"] = max(1, int(per_sm)) * sms
    return entry["grid"]


def available() -> bool:
    try:
        import cupy as cp
        _kernel(np.float64)
        a = cp.zeros((1, 1))
        return a is not None
    except Exception:  # noqa: BLE001
        return False


def run_block(Kd, KTd, c, lx, ux, lc, uc, x0, y0, x, y, tau, sigma, kk0, gamma, iters,
              shared_bounds: bool = False):
    """Run `iters` iterations in place on x, y. Returns (xh, yh, xprev, yprev, xh0, yh0).

    shared_bounds: lx/ux are (n,1) and lc/uc are (m,1); index by variable/row, not by (row,case).
    """
    import cupy as cp
    dt = x.dtype
    entry = _kernel(dt)
    n, S = x.shape
    m = y.shape[0]
    aty = cp.empty_like(x)
    xbar = cp.empty_like(x)
    kxb = cp.empty_like(y)
    xh, xprev, xh0 = cp.empty_like(x), cp.empty_like(x), cp.empty_like(x)
    yh, yprev, yh0 = cp.empty_like(y), cp.empty_like(y), cp.empty_like(y)
    threads = 256
    blocks = _grid(entry, threads)
    c_ = lambda a: cp.ascontiguousarray(a, dtype=dt)  # noqa: E731
    args = (np.int32(m), np.int32(n), np.int32(S), np.int32(iters), np.int32(1 if shared_bounds else 0),
            Kd.indptr, Kd.indices, Kd.vals, KTd.indptr, KTd.indices, KTd.vals,
            c_(c), c_(lx), c_(ux), c_(lc), c_(uc), c_(x0), c_(y0), x, y,
            aty, xbar, kxb, xh, yh, xprev, yprev, xh0, yh0,
            c_(tau).reshape(-1), c_(sigma).reshape(-1), c_(kk0).reshape(-1), dt.type(gamma))
    try:
        entry["k"]((blocks,), (threads,), args)
    except Exception:                                   # fewer resident blocks than assumed
        entry["grid"] = max(1, blocks // 2)
        entry["k"]((entry["grid"],), (threads,), args)
    return xh, yh, xprev, yprev, xh0, yh0
