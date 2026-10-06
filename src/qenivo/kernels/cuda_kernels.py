"""QENIVO's own CUDA kernels for the sparse products of the first-order engines.

Two kernels, both deterministic (no atomics, a fixed reduction order), so a GPU run is
bit-for-bit repeatable on the same card: a property an auditor can check.

  csr_spmv_warp      y = K x        one warp per row, lanes stride the row, shuffle reduction.
                                    Good for the single-scenario case and long rows.
  csr_spmm_rowmajor  Y = K X        X is (n, S) row-major: one block per row, threads stride
                                    the scenario axis, so neighbouring threads read
                                    neighbouring memory (coalesced). This is the batched-case
                                    kernel: S what-if scenarios share one matrix read.

Each kernel exists in float64 and float32 (suffix _f32). Consumer GPUs such as the T4 run
float32 about 32x faster than float64 and move half the bytes, so the refinement engine
(engines/refine.py) iterates in float32 and restores full accuracy in float64.

Compiled at first use through CuPy's NVRTC interface, once per device; nothing here links a
vendor sparse library. `use_vendor=True` switches to cuSPARSE for A/B timing only.
"""
from __future__ import annotations

import numpy as np

_TMPL = r"""
extern "C" __global__
void csr_spmv_warp{SUF}(const int m, const int* __restrict__ indptr, const int* __restrict__ indices,
                   const {T}* __restrict__ vals, const {T}* __restrict__ x,
                   {T}* __restrict__ y)
{{
    const long gtid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long row = gtid >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= m) return;                       // whole warp shares `row`, so it exits together
    {T} s = ({T})0;
    const int end = indptr[row + 1];
    for (int k = indptr[row] + lane; k < end; k += 32)
        s += vals[k] * x[indices[k]];
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffffu, s, off);
    if (lane == 0) y[row] = s;
}}

extern "C" __global__
void csr_spmm_rowmajor{SUF}(const int m, const int S, const int* __restrict__ indptr,
                       const int* __restrict__ indices, const {T}* __restrict__ vals,
                       const {T}* __restrict__ X, {T}* __restrict__ Y)
{{
    const int row = blockIdx.x;
    if (row >= m) return;
    const int start = indptr[row], end = indptr[row + 1];
    for (int s = threadIdx.x; s < S; s += blockDim.x) {{
        {T} acc = ({T})0;
        for (int k = start; k < end; ++k)
            acc += vals[k] * X[(long)indices[k] * S + s];
        Y[(long)row * S + s] = acc;
    }}
}}
"""
_SRC = _TMPL.format(T="double", SUF="") + _TMPL.format(T="float", SUF="_f32")

_MOD = {}


def _module():
    import cupy as cp
    dev = cp.cuda.Device().id                   # a module is bound to the device it was loaded on
    if dev not in _MOD:
        _MOD[dev] = cp.RawModule(code=_SRC, options=("-std=c++14",), name_expressions=None)
    return _MOD[dev]


class GPUMatrix:
    """CSR matrix resident on the GPU; `mv` accepts (n,) or (n, S) CuPy arrays."""

    def __init__(self, A, use_vendor: bool = False, dtype=np.float64):
        import cupy as cp
        import scipy.sparse as sps
        self.dtype = np.dtype(dtype)
        A = sps.csr_matrix(A, dtype=self.dtype)
        A.sort_indices()
        self.shape = A.shape
        self.nnz = A.nnz
        self.use_vendor = use_vendor
        self.indptr = cp.asarray(A.indptr.astype(np.int32))
        self.indices = cp.asarray(A.indices.astype(np.int32))
        self.vals = cp.asarray(A.data)
        if use_vendor:
            import cupyx.scipy.sparse as csp
            self._vendor = csp.csr_matrix(A)
        else:
            mod = _module()
            suf = "_f32" if self.dtype == np.float32 else ""
            self._spmv = mod.get_function("csr_spmv_warp" + suf)
            self._spmm = mod.get_function("csr_spmm_rowmajor" + suf)

    def mv(self, X):
        import cupy as cp
        if self.use_vendor:
            return self._vendor @ X
        m = self.shape[0]
        if X.ndim == 1 or X.shape[1] == 1:
            flat = cp.ascontiguousarray(X.reshape(-1), dtype=self.dtype)
            y = cp.empty(m, dtype=self.dtype)
            if m:
                threads = 256
                blocks = (m * 32 + threads - 1) // threads
                self._spmv((blocks,), (threads,), (np.int32(m), self.indptr, self.indices,
                                                   self.vals, flat, y))
            return y if X.ndim == 1 else y[:, None]
        X = cp.ascontiguousarray(X, dtype=self.dtype)
        S = X.shape[1]
        Y = cp.empty((m, S), dtype=self.dtype)
        if m:
            threads = min(256, max(32, ((S + 31) // 32) * 32))
            self._spmm((m,), (threads,), (np.int32(m), np.int32(S), self.indptr, self.indices,
                                         self.vals, X, Y))
        return Y

    __matmul__ = mv
