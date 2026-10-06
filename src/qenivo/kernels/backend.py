"""L0 kernel layer: one solver code path on the CPU (NumPy) or the GPU (CuPy arrays).

The first-order engines spend almost all their time in two products, K @ X and K' @ Y,
where X carries a batch axis (n, S): one column per scenario. On the GPU those products run
on QENIVO's own CUDA kernels (`cuda_kernels.py`), not on the vendor sparse library, so the
hot path of the GPU engine is code this project wrote and can audit. The vendor path is kept
behind `kernels="vendor"` only as an A/B baseline for benchmarking.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sps


class CPUMatrix:
    """CSR matrix on the host; `mv` accepts (n,) or (n, S)."""

    def __init__(self, A, dtype=np.float64):
        self.A = sps.csr_matrix(A, dtype=dtype)
        self.A.sort_indices()
        self.shape = self.A.shape
        self.nnz = self.A.nnz

    def mv(self, X):
        return self.A @ X

    __matmul__ = mv


@dataclass
class Backend:
    name: str          # "numpy" | "cupy"
    xp: Any            # numpy or cupy
    kernels: str = "qenivo"   # GPU only: "qenivo" (own CUDA kernels) | "vendor" (cuSPARSE baseline)
    dtype: Any = np.float64   # working precision of arrays and matrices made by this backend

    @property
    def is_gpu(self) -> bool:
        return self.name == "cupy"

    def asarray(self, a, dtype=None):
        return self.xp.asarray(a, dtype=dtype or self.dtype)

    def matrix(self, A):
        """Put a sparse matrix where this backend computes with it."""
        if not self.is_gpu:
            return CPUMatrix(A, self.dtype)
        from .cuda_kernels import GPUMatrix
        return GPUMatrix(A, use_vendor=(self.kernels == "vendor"), dtype=self.dtype)

    def to_host(self, a):
        if self.is_gpu and a is not None and not isinstance(a, (float, int, np.ndarray, np.generic)):
            return a.get()
        return a

    def synchronize(self):
        if self.is_gpu:
            self.xp.cuda.Device().synchronize()

    def now(self) -> float:
        self.synchronize()
        return time.perf_counter()


_GPU = {}


def gpu_available() -> bool:
    """True only if a CUDA device exists AND CuPy can compile and run a kernel on it (a CuPy without
    the CUDA headers sees the device but cannot build QENIVO's kernels). The reason is kept in
    gpu_status() and the answer is cached for the process."""
    if "ok" not in _GPU:
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() <= 0:
                raise RuntimeError("no CUDA device")
            k = cupy.ElementwiseKernel("float64 x", "float64 y", "y = 2.0 * x", "qenivo_probe")
            if float(k(cupy.arange(4, dtype=cupy.float64)).sum()) != 12.0:
                raise RuntimeError("probe kernel gave a wrong answer")
            _GPU.update(ok=True, reason="ok")
        except Exception as e:  # noqa: BLE001
            _GPU.update(ok=False, reason=f"{type(e).__name__}: {e}"[:300])
    return _GPU["ok"]


def gpu_status() -> str:
    gpu_available()
    return _GPU["reason"]


def get_backend(name: str = "numpy", kernels: str = "qenivo", dtype=np.float64) -> Backend:
    dtype = np.dtype(dtype)
    if name in ("numpy", "cpu"):
        return Backend("numpy", np, kernels, dtype)
    if name in ("cupy", "gpu", "cuda"):
        import cupy
        return Backend("cupy", cupy, kernels, dtype)
    if name == "auto":
        return get_backend("cupy" if gpu_available() else "numpy", kernels, dtype)
    raise ValueError(f"unknown backend {name!r} (use numpy, cupy or auto)")


def gpu_info() -> dict:
    """Name and memory of the current CUDA device, or {} when there is none."""
    try:
        import cupy
        dev = cupy.cuda.Device()
        props = cupy.cuda.runtime.getDeviceProperties(dev.id)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        return {"gpu": name, "gpu_mem_bytes": int(props["totalGlobalMem"]),
                "compute_capability": f"{props['major']}.{props['minor']}",
                "cuda_runtime": cupy.cuda.runtime.runtimeGetVersion(), "cupy": cupy.__version__}
    except Exception:
        return {}
