"""One batch of what-if LPs split across several GPUs (e.g. Kaggle's two T4s).

The S scenarios of a batch are independent, so the batch is cut into one slice per device and each
slice runs the ordinary batched engine (`pdhg` or `pdhg-rf`) on its own GPU in its own thread.
CuPy keeps one context, one memory pool and one compiled copy of QENIVO's kernels per device, and
the Python thread only launches work, so for batches large enough to keep a GPU busy the devices
run concurrently. Results are merged back in scenario order; each column is still certified by
the caller's float64 check.
"""
from __future__ import annotations

import threading
import time

import numpy as np


def device_count() -> int:
    """Number of CUDA devices QENIVO can actually use (0 if kernels cannot be compiled)."""
    from ..kernels.backend import gpu_available
    if not gpu_available():
        return 0
    import cupy
    return int(cupy.cuda.runtime.getDeviceCount())


def solve_batch_devices(A, C, lc, uc, lx, ux, c0=0.0, devices=None, engine="pdhg", tol=1e-6,
                        time_limit=3600.0, opts=None) -> dict:
    """Split the columns of C over `devices` (default: all GPUs). Returns x, y, status, iterations,
    per-device times and the wall time."""
    import cupy

    from . import pdhg, refine
    devices = list(range(device_count())) if devices is None else list(devices)
    if not devices:
        raise RuntimeError("no CUDA device")
    C = np.asarray(C, dtype=np.float64)
    n, S = C.shape
    m = A.shape[0]
    parts = [p for p in np.array_split(np.arange(S), len(devices)) if len(p)]
    X, Y = np.zeros((n, S)), np.zeros((m, S))
    status, iters = [None] * S, np.zeros(S, dtype=np.int64)
    times, errors = {}, {}

    def col(v, cols):
        v = np.asarray(v, dtype=np.float64)
        return v[:, cols] if v.ndim == 2 else v

    def work(dev, cols):
        t = time.perf_counter()
        try:
            with cupy.cuda.Device(dev):
                if engine == "pdhg-rf":
                    br = refine.solve_refined(A, C[:, cols], col(lc, cols), col(uc, cols), col(lx, cols),
                                              col(ux, cols), c0, tol=tol, backend="cupy", time_limit=time_limit)
                else:
                    o = opts or pdhg.PDHGOptions(tol=tol, time_limit=time_limit)
                    br = pdhg.solve_batch(A, C[:, cols], col(lc, cols), col(uc, cols), col(lx, cols),
                                          col(ux, cols), c0, o, "cupy")
                cupy.cuda.Device(dev).synchronize()
            X[:, cols], Y[:, cols] = br.x, br.y
            for k, j in enumerate(cols):
                status[j] = br.status[k]
                iters[j] = br.iterations[k]
        except Exception as e:  # noqa: BLE001
            errors[dev] = repr(e)[:300]
        times[dev] = time.perf_counter() - t

    t0 = time.perf_counter()
    threads = [threading.Thread(target=work, args=(d, c)) for d, c in zip(devices, parts)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return {"x": X, "y": Y, "status": status, "iterations": iters, "device_times": times,
            "wall": time.perf_counter() - t0, "devices": devices[:len(parts)], "errors": errors}
