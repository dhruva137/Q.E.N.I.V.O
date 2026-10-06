"""GPU byte-diet helpers: L2 tile sizing, shared-bounds detection, device L2 query.

These cut compulsory traffic of the batch PDHG path (research/2026-10-01/04_gpu_lab.md):
shared bounds (~1.25x), L2-sized case tiles (~1.3-1.5x), f32 storage (~2x at 1e-4).
"""
from __future__ import annotations

import numpy as np


def device_l2_bytes() -> int:
    """L2 cache size in bytes for the current CUDA device, or 32 MiB if unknown."""
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        n = int(props.get("l2CacheSize") or props.get("L2CacheSize") or 0)
        if n > 0:
            return n
    except Exception:  # noqa: BLE001
        pass
    return 32 << 20


def s_tile(n: int, m: int, dtype=np.float64, l2_bytes: int | None = None,
           shared_bounds: bool = True) -> int:
    """Largest tile width so the hot per-case vectors fit in L2.

    Matches the lab rule of thumb St ≈ L2 / (itemsize · n) for keeping X resident, then
    tightens by a small factor for the dual block and (when not shared) bound traffic.
    Always at least 1; never exceeds a soft cap of 1024.
    """
    l2 = int(l2_bytes if l2_bytes is not None else device_l2_bytes())
    vs = int(np.dtype(dtype).itemsize)
    n = max(1, int(n))
    m = max(0, int(m))
    # Primary: X (and a twin for the reflected bar) in L2 — lab St ≈ L2/(8n) on L4 f64.
    st = l2 // max(vs * n, 1)
    # Budget extra (n+m) words for y / temps; shared bounds avoid 2*(n+m) bound copies.
    extra = (n + m) * (1 if shared_bounds else 3)
    st2 = l2 // max(vs * (n + extra), 1)
    st = max(1, min(st, st2, 1024))
    return int(st)


def is_shared_columns(v, rows: int, S: int, rtol: float = 0.0, atol: float = 0.0) -> bool:
    """True if `v` is a single column (or every column is identical within tol)."""
    a = np.asarray(v)
    if a.ndim == 1:
        return a.shape[0] == rows
    if a.ndim != 2 or a.shape[0] != rows:
        return False
    if a.shape[1] == 1:
        return True
    if a.shape[1] != S:
        return False
    if rtol == 0.0 and atol == 0.0:
        return bool(np.array_equal(a, a[:, :1]))
    return bool(np.allclose(a, a[:, :1], rtol=rtol, atol=atol))


def as_shared_or_batch(v, rows: int, S: int, prefer_shared: bool) -> np.ndarray:
    """(rows, 1) when shared (and prefer_shared), else contiguous (rows, S)."""
    a = np.asarray(v, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(rows, 1)
    elif a.ndim == 2 and a.shape[1] == 1:
        a = np.ascontiguousarray(a.reshape(rows, 1))
    else:
        a = np.ascontiguousarray(np.broadcast_to(np.asarray(v, dtype=np.float64), (rows, S)))
    if prefer_shared and is_shared_columns(a, rows, S):
        return np.ascontiguousarray(a[:, :1])
    if a.shape[1] == 1 and not prefer_shared:
        return np.ascontiguousarray(np.broadcast_to(a, (rows, S)).copy())
    return a


def take_active(arr, kd, xp):
    """Column-gather for compaction; a shared (rows, 1) slab is left unchanged."""
    if arr is None:
        return arr
    if getattr(arr, "ndim", 1) == 1:
        return arr[kd]
    if arr.shape[1] == 1:
        return arr
    return arr[:, kd]


# Cumulative ablation presets used by the measurement harness.
ABLATIONS = ("baseline", "A1", "A2", "A3", "A4")


def ablation_flags(name: str) -> dict:
    """Map ablation name to PDHGOptions kwargs. Cumulative: A_k includes A_1..A_k."""
    name = name.lower().lstrip("+")
    if name in ("baseline", "off", "none"):
        return dict(diet_f32_cert=False, diet_shared_bounds=False, diet_l2_tile=False,
                    diet_retire_frac=0.0, dtype="float64")
    order = ["a1", "a2", "a3", "a4"]
    if name not in order:
        raise ValueError(f"unknown ablation {name!r}; use one of {ABLATIONS}")
    upto = order.index(name)
    return dict(
        diet_f32_cert=upto >= 0,
        diet_shared_bounds=upto >= 1,
        diet_l2_tile=upto >= 2,
        diet_retire_frac=0.25 if upto >= 3 else 0.0,
        dtype="float32" if upto >= 0 else "float64",
    )
