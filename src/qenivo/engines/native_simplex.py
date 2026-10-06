"""Native (C++) simplex: native/simplex_core.cpp, compiled on the target machine at first use.

Same contract as engines/simplex.solve_simplex (status, x, y, iterations, col/row statuses), so
the API, branch and bound, ranging and certificates use it unchanged. The shared library is
built with the machine's C++ compiler (g++ / clang++ / c++) into a per-user cache keyed by the
source hash; when no compiler is found the pure-Python engine is used instead. Nothing is
downloaded: the source ships with the package, so the build is auditable.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ._native_abi import native_abi_tag

SRC = Path(__file__).resolve().parents[1] / "native" / "simplex_core.cpp"
_LIB: dict = {}
STATUS = {0: "optimal", 1: "infeasible", 2: "unbounded", 3: "iteration_limit", 4: "time_limit", 5: "numerical_error"}
_NAME = {0: "basic", 1: "at_lower", 2: "at_upper", 3: "free"}
_CODE = {v: k for k, v in _NAME.items()}


def _cache_dir() -> Path:
    base = os.environ.get("QENIVO_CACHE") or (Path(os.environ.get("LOCALAPPDATA", Path.home())) / "qenivo"
                                              if os.name == "nt" else Path.home() / ".cache" / "qenivo")
    d = Path(base) / "native"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _compiler():
    for cxx in (os.environ.get("CXX"), "g++", "clang++", "c++"):
        if cxx and shutil.which(cxx):
            return cxx
    return None


def library():
    """The loaded native library, or None (no compiler, build failure, or QENIVO_NATIVE=0)."""
    if "lib" in _LIB:
        return _LIB["lib"]
    _LIB["lib"], _LIB["reason"] = None, "disabled"
    if os.environ.get("QENIVO_NATIVE", "1") == "0":
        _LIB["reason"] = "QENIVO_NATIVE=0; the Python engine is used"
        return None
    h = hashlib.sha256(SRC.read_bytes() + native_abi_tag()).hexdigest()[:16]
    ext = ".dll" if os.name == "nt" else (".dylib" if platform.system() == "Darwin" else ".so")
    out = _cache_dir() / f"qenivo_simplex_{h}{ext}"
    if not out.exists():                 # a prebuilt library (installer) needs no compiler
        cxx = _compiler()
        if cxx is None:
            _LIB["reason"] = "no C++ compiler found (set CXX); the Python engine is used"
            return None
        cmd = [cxx, "-O3", "-std=c++17", "-shared", "-o", str(out), str(SRC)]
        if os.name == "nt":
            cmd += ["-static"]              # no MinGW runtime DLLs (libstdc++, libgcc, winpthread) needed
        else:
            cmd += ["-fPIC"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            _LIB["reason"] = "build failed: " + r.stderr[-300:]
            return None
    try:
        lib = ctypes.CDLL(str(out))
    except OSError as e:                     # e.g. Windows Smart App Control blocks unsigned local builds
        _LIB["reason"] = f"native library could not be loaded ({e}); the Python engine is used"
        return None
    P = ctypes.POINTER
    lib.nr_simplex.restype = ctypes.c_int
    lib.nr_simplex.argtypes = [ctypes.c_int, ctypes.c_int, P(ctypes.c_int), P(ctypes.c_int), P(ctypes.c_double),
                               P(ctypes.c_double), P(ctypes.c_double), P(ctypes.c_double), P(ctypes.c_double),
                               P(ctypes.c_double), ctypes.c_double, ctypes.c_double, ctypes.c_long, P(ctypes.c_int),
                               P(ctypes.c_double), P(ctypes.c_double), P(ctypes.c_long), P(ctypes.c_int)]
    _LIB["lib"], _LIB["reason"], _LIB["path"] = lib, "ok", str(out)
    return lib


def status() -> str:
    library()
    return _LIB.get("reason", "unknown")


def _ptr(a, ct):
    return a.ctypes.data_as(ctypes.POINTER(ct))


_CSC: dict = {}          # id(A) -> (A, arrays): branch-and-bound nodes share one matrix


def _csc_arrays(A):
    # nodes rebuild the Problem, which re-wraps A, but the data buffer is shared: key on it
    key = (id(A.data), A.shape, A.nnz) if sp.issparse(A) else (id(A), None, None)
    hit = _CSC.get(key)
    if hit is not None and hit[0] is (A.data if sp.issparse(A) else A):
        return hit[1]
    C = sp.csc_matrix(A, dtype=np.float64)
    C.sort_indices()
    arrs = (np.ascontiguousarray(C.indptr, dtype=np.int32), np.ascontiguousarray(C.indices, dtype=np.int32),
            np.ascontiguousarray(C.data, dtype=np.float64))
    if len(_CSC) > 16:
        _CSC.clear()
    _CSC[key] = (A.data if sp.issparse(A) else A, arrs)   # kept alive so the id stays unique
    return arrs


def solve_simplex_native(prob, tol: float = 1e-9, time_limit: float = 3600.0, dual: bool = True, start=None) -> dict:
    lib = library()
    if lib is None:
        raise RuntimeError(status())
    t0 = time.perf_counter()
    m, n = prob.m, prob.n
    Ap, Ai, Ax = _csc_arrays(prob.A)
    f = lambda v: np.ascontiguousarray(v, dtype=np.float64)  # noqa: E731
    c, lx, ux, lc, uc = f(prob.c), f(prob.lx), f(prob.ux), f(prob.lc), f(prob.uc)
    st = np.full(n + m, -1, dtype=np.int32)
    if start and "col_statuses" in start and len(start["col_statuses"]) == n and len(start["row_statuses"]) == m:
        st[:] = [_CODE.get(v, 1) for v in list(start["col_statuses"]) + list(start["row_statuses"])]
    x = np.zeros(n)
    y = np.zeros(m)
    iters = ctypes.c_long(0)
    refac = ctypes.c_int(0)
    max_iter = 20000 + 50 * (m + n)
    code = lib.nr_simplex(m, n, _ptr(Ap, ctypes.c_int), _ptr(Ai, ctypes.c_int), _ptr(Ax, ctypes.c_double),
                          _ptr(c, ctypes.c_double), _ptr(lx, ctypes.c_double), _ptr(ux, ctypes.c_double),
                          _ptr(lc, ctypes.c_double), _ptr(uc, ctypes.c_double), float(tol), float(time_limit),
                          int(max_iter), _ptr(st, ctypes.c_int), _ptr(x, ctypes.c_double), _ptr(y, ctypes.c_double),
                          ctypes.byref(iters), ctypes.byref(refac))
    names = [_NAME.get(int(v), "at_lower") for v in st]
    stat = STATUS.get(code, "numerical_error")
    if stat == "optimal" and not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        stat = "numerical_error"                 # a near-singular basis must never be reported as optimal
    return {"status": stat, "x": x, "y": y, "iterations": int(iters.value),
            "col_statuses": names[:n], "row_statuses": names[n:], "time": time.perf_counter() - t0,
            "refactorizations": int(refac.value), "engine_impl": "native C++"}
