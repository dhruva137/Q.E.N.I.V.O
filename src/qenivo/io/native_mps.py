"""Compile-on-first-use loader for native/mps_reader.cpp (same pattern as engines/native_simplex)."""
from __future__ import annotations

import ctypes
import gzip
import hashlib
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..engines._native_abi import native_abi_tag
from ..model import Problem as LPProblem

SRC = Path(__file__).resolve().parents[1] / "native" / "mps_reader.cpp"
_LIB: dict = {}


class NrMpsProblem(ctypes.Structure):
    _fields_ = [
        ("m", ctypes.c_int),
        ("n", ctypes.c_int),
        ("nnz", ctypes.c_int),
        ("indptr", ctypes.POINTER(ctypes.c_int)),
        ("indices", ctypes.POINTER(ctypes.c_int)),
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("c", ctypes.POINTER(ctypes.c_double)),
        ("lc", ctypes.POINTER(ctypes.c_double)),
        ("uc", ctypes.POINTER(ctypes.c_double)),
        ("lx", ctypes.POINTER(ctypes.c_double)),
        ("ux", ctypes.POINTER(ctypes.c_double)),
        ("c0", ctypes.c_double),
        ("obj_sign", ctypes.c_double),
        ("is_integer", ctypes.POINTER(ctypes.c_int)),
        ("has_integer", ctypes.c_int),
        ("has_q", ctypes.c_int),
        ("q_nnz", ctypes.c_int),
        ("q_indptr", ctypes.POINTER(ctypes.c_int)),
        ("q_indices", ctypes.POINTER(ctypes.c_int)),
        ("q_data", ctypes.POINTER(ctypes.c_double)),
        ("name", ctypes.c_char_p),
        ("names_blob", ctypes.POINTER(ctypes.c_char)),
        ("names_blob_len", ctypes.c_int),
    ]


def _cache_dir() -> Path:
    base = os.environ.get("QENIVO_CACHE") or (
        Path(os.environ.get("LOCALAPPDATA", Path.home())) / "qenivo"
        if os.name == "nt"
        else Path.home() / ".cache" / "qenivo"
    )
    d = Path(base) / "native"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _compiler():
    for cxx in (os.environ.get("CXX"), "g++", "clang++", "c++"):
        if cxx and shutil.which(cxx):
            return cxx
    return None


def library():
    """Loaded native MPS library, or None on compile/load failure / QENIVO_NATIVE=0."""
    if "lib" in _LIB:
        return _LIB["lib"]
    _LIB["lib"], _LIB["reason"] = None, "disabled"
    if os.environ.get("QENIVO_NATIVE", "1") == "0":
        _LIB["reason"] = "QENIVO_NATIVE=0; the Python MPS reader is used"
        return None
    if not SRC.exists():
        _LIB["reason"] = f"missing source {SRC}"
        return None
    h = hashlib.sha256(SRC.read_bytes() + native_abi_tag()).hexdigest()[:16]
    ext = ".dll" if os.name == "nt" else (".dylib" if platform.system() == "Darwin" else ".so")
    out = _cache_dir() / f"qenivo_mps_{h}{ext}"
    if not out.exists():                 # a prebuilt library (installer) needs no compiler
        cxx = _compiler()
        if cxx is None:
            _LIB["reason"] = "no C++ compiler found (set CXX); the Python MPS reader is used"
            return None
        cmd = [cxx, "-O3", "-std=c++17", "-shared", "-o", str(out), str(SRC)]
        if os.name == "nt":
            cmd += ["-static"]
        else:
            cmd += ["-fPIC"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            _LIB["reason"] = "build failed: " + (r.stderr or r.stdout or "")[-300:]
            return None
    try:
        lib = ctypes.CDLL(str(out))
    except OSError as e:
        _LIB["reason"] = f"native library could not be loaded ({e}); the Python MPS reader is used"
        return None
    lib.nr_mps_parse.restype = ctypes.c_int
    lib.nr_mps_parse.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(NrMpsProblem), ctypes.c_char_p, ctypes.c_int,
    ]
    lib.nr_mps_free.restype = None
    lib.nr_mps_free.argtypes = [ctypes.POINTER(NrMpsProblem)]
    _LIB["lib"], _LIB["reason"], _LIB["path"] = lib, "ok", str(out)
    return lib


def status() -> str:
    library()
    return _LIB.get("reason", "unknown")


def _load_text_bytes(path) -> bytes:
    p = str(path)
    if p.endswith(".gz"):
        with gzip.open(p, "rb") as fh:
            return fh.read()
    if p.endswith(".bz2"):
        import bz2
        with bz2.open(p, "rb") as fh:
            return fh.read()
    return Path(p).read_bytes()


def _split_names(blob: bytes, m: int, n: int) -> tuple[list[str], list[str]]:
    parts = blob.split(b"\0")
    # trailing empty from final null
    while parts and parts[-1] == b"":
        parts.pop()
    if len(parts) < m + n:
        parts.extend([b""] * (m + n - len(parts)))
    row = [parts[i].decode("utf-8", errors="replace") for i in range(m)]
    col = [parts[m + j].decode("utf-8", errors="replace") for j in range(n)]
    return row, col


def _arr(ptr, n, dtype):
    if n <= 0:
        return np.zeros(0, dtype=dtype)
    ct = ctypes.c_double if dtype == np.float64 else ctypes.c_int
    return np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ct)), shape=(n,)).copy()


def read_mps_native(path, name: str | None = None) -> LPProblem:
    lib = library()
    if lib is None:
        raise RuntimeError(status())
    raw = _load_text_bytes(path)
    # Decode with replace then re-encode latin-1 so C++ sees the same byte stream as text.
    text = raw.decode("utf-8", errors="replace").encode("latin-1", errors="replace")
    stem = Path(str(path)).name.split(".")[0]
    default = (name or stem).encode("utf-8")
    name_locked = 1 if name is not None else 0
    out = NrMpsProblem()
    err = ctypes.create_string_buffer(512)
    code = lib.nr_mps_parse(text, len(text), default, name_locked, ctypes.byref(out), err, 512)
    if code != 0:
        msg = err.value.decode("utf-8", errors="replace") or f"native parse failed ({code})"
        raise RuntimeError(msg)
    try:
        m, n, nnz = int(out.m), int(out.n), int(out.nnz)
        indptr = _arr(out.indptr, m + 1, np.int32)
        indices = _arr(out.indices, nnz, np.int32)
        data = _arr(out.data, nnz, np.float64)
        A = sp.csr_matrix((data, indices, indptr), shape=(m, n))
        A.eliminate_zeros()
        c = _arr(out.c, n, np.float64)
        lc = _arr(out.lc, m, np.float64)
        uc = _arr(out.uc, m, np.float64)
        lx = _arr(out.lx, n, np.float64)
        ux = _arr(out.ux, n, np.float64)
        integer = None
        if out.has_integer:
            integer = _arr(out.is_integer, n, np.int32).astype(bool)
        Q = None
        if out.has_q:
            qnnz = int(out.q_nnz)
            q_indptr = _arr(out.q_indptr, n + 1, np.int32)
            q_indices = _arr(out.q_indices, qnnz, np.int32)
            q_data = _arr(out.q_data, qnnz, np.float64)
            Q = sp.csr_matrix((q_data, q_indices, q_indptr), shape=(n, n))
        blob = b""
        if out.names_blob and out.names_blob_len > 0:
            blob = ctypes.string_at(out.names_blob, out.names_blob_len)
        row_names, col_names = _split_names(blob, m, n)
        pname = (out.name or b"").decode("utf-8", errors="replace") if out.name else (name or stem)
        return LPProblem(
            c=c, A=A, lc=lc, uc=uc, lx=lx, ux=ux, c0=float(out.c0), obj_sign=float(out.obj_sign),
            name=pname, row_names=row_names, col_names=col_names, integer=integer, Q=Q,
        )
    finally:
        lib.nr_mps_free(ctypes.byref(out))
