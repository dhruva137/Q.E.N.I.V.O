"""Build and load the Qenivo C ABI the way the native simplex is built.

The shared library is compiled on this machine at first use, statically linked on
Windows and with -fPIC on Linux, and cached by a hash of the sources. The simplex
engine in that library calls native/simplex_core.cpp. Other engines go through
python_bridge.cpp, which embeds Python.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import threading
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_CORE = _HERE.parent / "src" / "qenivo" / "native" / "simplex_core.cpp"
_SOURCES = (_HERE / "qenivo_c.cpp", _HERE / "python_bridge.cpp", _CORE)
_LIB: dict = {}
_LOCK = threading.Lock()


class CapiError(RuntimeError):
    def __init__(self, code: int, message: str):
        self.code = int(code)
        super().__init__(f"{message} (code {code})")


def source_root() -> Path:
    """Directory that contains the qenivo package."""
    return _HERE.parent / "src"


def python_library() -> str:
    if os.name == "nt":
        ver = f"python{sys.version_info.major}{sys.version_info.minor}.dll"
        return str(Path(sys.executable).with_name(ver))
    libdir = sysconfig.get_config_var("LIBDIR") or ""
    name = sysconfig.get_config_var("LDLIBRARY") or ""
    return str(Path(libdir) / name)


def compiler() -> str | None:
    for cxx in (os.environ.get("CXX"), "g++", "clang++", "c++", "cl"):
        if cxx and shutil.which(cxx):
            return cxx
    return None


def _cache_dir() -> Path:
    base = os.environ.get("QENIVO_CACHE")
    if not base:
        if os.name == "nt":
            base = str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / "qenivo")
        else:
            base = str(Path.home() / ".cache" / "qenivo")
    path = Path(base) / "capi"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hash_key(cxx: str) -> str:
    h = hashlib.sha256()
    h.update(platform.platform().encode())
    h.update(cxx.encode())
    for src in _SOURCES:
        h.update(src.read_bytes())
    h.update(Path(_HERE / "qenivo.h").read_bytes())
    h.update(Path(_HERE / "python_bridge.h").read_bytes())
    return h.hexdigest()[:16]


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError("C ABI build failed:\n" + " ".join(cmd) + "\n" + tail)


def _compile(cxx: str, out: Path) -> None:
    objs = out.parent / (out.stem + "_obj")
    if objs.exists():
        shutil.rmtree(objs)
    objs.mkdir()
    core_obj = objs / "simplex_core.o"
    api_obj = objs / "qenivo_c.o"
    bridge_obj = objs / "python_bridge.o"
    if Path(cxx).stem.lower() == "cl" or cxx.lower().endswith("cl.exe"):
        _compile_msvc(out, core_obj, api_obj, bridge_obj)
        return
    pic = [] if os.name == "nt" else ["-fPIC"]
    base = [cxx, "-O3", "-std=c++17", *pic, "-c"]
    _run(base + ["-o", str(core_obj), str(_CORE)])
    ours = base + ["-Wall", "-Wextra", "-Werror", "-DQENIVO_BUILD", "-I", str(_HERE)]
    _run(ours + ["-o", str(api_obj), str(_HERE / "qenivo_c.cpp")])
    _run(ours + ["-o", str(bridge_obj), str(_HERE / "python_bridge.cpp")])
    link = [cxx, "-shared", "-o", str(out), str(api_obj), str(bridge_obj), str(core_obj)]
    if os.name == "nt":
        link.append("-static")
    else:
        link.append("-pthread")
    _run(link)


def _compile_msvc(out: Path, core_obj: Path, api_obj: Path, bridge_obj: Path) -> None:
    cl = compiler()
    assert cl is not None
    common = [cl, "/nologo", "/std:c++17", "/O2", "/EHsc", "/c"]
    _run(common + [str(_CORE), f"/Fo{core_obj}"])
    ours = common + ["/W4", "/WX", "/DQENIVO_BUILD", f"/I{str(_HERE)}"]
    _run(ours + [str(_HERE / "qenivo_c.cpp"), f"/Fo{api_obj}"])
    _run(ours + [str(_HERE / "python_bridge.cpp"), f"/Fo{bridge_obj}"])
    _run([cl, "/nologo", "/LD", str(api_obj), str(bridge_obj), str(core_obj), f"/Fe{out}"])


def library() -> ctypes.CDLL:
    """Loaded C ABI, compiling it on first use."""
    with _LOCK:
        if "lib" in _LIB:
            return _LIB["lib"]
        cxx = compiler()
        if cxx is None:
            raise RuntimeError("no C++ compiler found (set CXX); the C ABI was not built")
        ext = ".dll" if os.name == "nt" else (".dylib" if platform.system() == "Darwin" else ".so")
        out = _cache_dir() / f"qenivo_capi_{_hash_key(cxx)}{ext}"
        if not out.exists():
            _compile(cxx, out)
        os.environ["QENIVO_PYTHON_DLL"] = python_library()
        os.environ.setdefault("QENIVO_SRC", str(source_root()))
        lib = ctypes.CDLL(str(out))
        _bind(lib)
        _LIB["lib"] = lib
        _LIB["path"] = str(out)
        _LIB["compiler"] = cxx
        return lib


def library_path() -> str:
    library()
    return _LIB["path"]


def compiler_used() -> str:
    library()
    return _LIB["compiler"]


_PROGRESS = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int64, ctypes.c_double, ctypes.c_int32)


def _bind(lib: ctypes.CDLL) -> None:
    P = ctypes.POINTER
    i32 = ctypes.c_int32
    f64 = ctypes.c_double
    lib.qenivo_abi_version.restype = i32
    lib.qenivo_error_name.restype = ctypes.c_char_p
    lib.qenivo_error_name.argtypes = [i32]
    lib.qenivo_create_error_message.restype = ctypes.c_char_p
    lib.qenivo_model_create.restype = ctypes.c_void_p
    lib.qenivo_model_create.argtypes = [
        i32, i32, P(i32), P(i32), P(f64), P(f64), P(f64), P(f64), P(f64), P(f64), P(i32), P(i32), P(i32), P(f64), P(i32)
    ]
    lib.qenivo_model_free.argtypes = [ctypes.c_void_p]
    lib.qenivo_set_tolerance.restype = i32
    lib.qenivo_set_tolerance.argtypes = [ctypes.c_void_p, f64]
    lib.qenivo_set_time_limit.restype = i32
    lib.qenivo_set_time_limit.argtypes = [ctypes.c_void_p, f64]
    lib.qenivo_set_name.restype = i32
    lib.qenivo_set_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.qenivo_set_progress.restype = i32
    lib.qenivo_set_progress.argtypes = [ctypes.c_void_p, _PROGRESS, ctypes.c_void_p]
    lib.qenivo_solve.restype = i32
    lib.qenivo_solve.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.qenivo_last_error.restype = i32
    lib.qenivo_last_error.argtypes = [ctypes.c_void_p]
    lib.qenivo_last_error_message.restype = ctypes.c_char_p
    lib.qenivo_last_error_message.argtypes = [ctypes.c_void_p]
    lib.qenivo_result_status.restype = i32
    lib.qenivo_result_status.argtypes = [ctypes.c_void_p]
    lib.qenivo_result_status_name.restype = ctypes.c_char_p
    lib.qenivo_result_status_name.argtypes = [ctypes.c_void_p]
    lib.qenivo_result_objective.restype = i32
    lib.qenivo_result_objective.argtypes = [ctypes.c_void_p, P(f64)]
    lib.qenivo_result_iterations.restype = ctypes.c_int64
    lib.qenivo_result_iterations.argtypes = [ctypes.c_void_p]
    lib.qenivo_result_has_duals.restype = i32
    lib.qenivo_result_has_duals.argtypes = [ctypes.c_void_p]
    lib.qenivo_result_has_basis.restype = i32
    lib.qenivo_result_has_basis.argtypes = [ctypes.c_void_p]
    lib.qenivo_result_has_reduced_costs.restype = i32
    lib.qenivo_result_has_reduced_costs.argtypes = [ctypes.c_void_p]
    lib.qenivo_result_x.restype = i32
    lib.qenivo_result_x.argtypes = [ctypes.c_void_p, P(f64), i32]
    lib.qenivo_result_y.restype = i32
    lib.qenivo_result_y.argtypes = [ctypes.c_void_p, P(f64), i32]
    lib.qenivo_result_reduced_costs.restype = i32
    lib.qenivo_result_reduced_costs.argtypes = [ctypes.c_void_p, P(f64), i32]
    lib.qenivo_result_basis.restype = i32
    lib.qenivo_result_basis.argtypes = [ctypes.c_void_p, P(i32), i32, P(i32), i32]
    lib.qenivo_result_certificate_json.restype = ctypes.c_char_p
    lib.qenivo_result_certificate_json.argtypes = [ctypes.c_void_p]
    lib.qenivo_model_rows.restype = i32
    lib.qenivo_model_rows.argtypes = [ctypes.c_void_p]
    lib.qenivo_model_cols.restype = i32
    lib.qenivo_model_cols.argtypes = [ctypes.c_void_p]


def _ptr(arr, ctype):
    if arr is None:
        return None
    return np.ascontiguousarray(arr).ctypes.data_as(ctypes.POINTER(ctype))


def _check(lib, handle, code: int) -> None:
    if code == 0:
        return
    name = lib.qenivo_error_name(code)
    detail = lib.qenivo_last_error_message(handle) if handle else lib.qenivo_create_error_message()
    text = (detail or name or b"").decode()
    raise CapiError(code, text or "C ABI call failed")


class SolveResult:
    def __init__(self, status, status_name, objective, x, y, reduced_costs, column_basis, row_basis, iterations, certificate):
        self.status = status
        self.status_name = status_name
        self.objective = objective
        self.x = x
        self.y = y
        self.reduced_costs = reduced_costs
        self.column_basis = column_basis
        self.row_basis = row_basis
        self.iterations = iterations
        self.certificate = certificate


def solve_csr(cost, a_ptr, a_idx, a_val, row_lower, row_upper, col_lower, col_upper, *, integrality=None,
              q_ptr=None, q_idx=None, q_val=None, engine="simplex", tol=1e-8, time_limit=60.0, name="capi",
              progress=None) -> SolveResult:
    """Solve one CSR model through the C ABI. Arrays are copied by the library."""
    lib = library()
    cost = np.ascontiguousarray(cost, dtype=np.float64).reshape(-1)
    a_ptr = np.ascontiguousarray(a_ptr, dtype=np.int32).reshape(-1)
    cols = int(cost.shape[0])
    rows = int(a_ptr.shape[0] - 1)
    a_idx_a = None if a_idx is None else np.ascontiguousarray(a_idx, dtype=np.int32)
    a_val_a = None if a_val is None else np.ascontiguousarray(a_val, dtype=np.float64)
    rl = np.ascontiguousarray(row_lower, dtype=np.float64).reshape(-1) if rows else None
    ru = np.ascontiguousarray(row_upper, dtype=np.float64).reshape(-1) if rows else None
    cl = np.ascontiguousarray(col_lower, dtype=np.float64).reshape(-1)
    cu = np.ascontiguousarray(col_upper, dtype=np.float64).reshape(-1)
    integ = None if integrality is None else np.ascontiguousarray(integrality, dtype=np.int32)
    qp = None if q_ptr is None else np.ascontiguousarray(q_ptr, dtype=np.int32)
    qi = None if q_idx is None else np.ascontiguousarray(q_idx, dtype=np.int32)
    qv = None if q_val is None else np.ascontiguousarray(q_val, dtype=np.float64)
    err = ctypes.c_int32(0)
    handle = lib.qenivo_model_create(
        rows, cols, _ptr(a_ptr, ctypes.c_int32), _ptr(a_idx_a, ctypes.c_int32), _ptr(a_val_a, ctypes.c_double),
        _ptr(cost, ctypes.c_double), _ptr(rl, ctypes.c_double), _ptr(ru, ctypes.c_double),
        _ptr(cl, ctypes.c_double), _ptr(cu, ctypes.c_double), _ptr(integ, ctypes.c_int32),
        _ptr(qp, ctypes.c_int32), _ptr(qi, ctypes.c_int32), _ptr(qv, ctypes.c_double), ctypes.byref(err))
    if not handle:
        _check(lib, None, int(err.value))
    cb_ref = None
    try:
        _check(lib, handle, lib.qenivo_set_name(handle, name.encode()))
        _check(lib, handle, lib.qenivo_set_tolerance(handle, float(tol)))
        _check(lib, handle, lib.qenivo_set_time_limit(handle, float(time_limit)))
        if progress is not None:
            cb_ref = _PROGRESS(progress)
            _check(lib, handle, lib.qenivo_set_progress(handle, cb_ref, None))
        _check(lib, handle, lib.qenivo_solve(handle, engine.encode()))
        obj = ctypes.c_double()
        _check(lib, handle, lib.qenivo_result_objective(handle, ctypes.byref(obj)))
        x = np.zeros(cols)
        _check(lib, handle, lib.qenivo_result_x(handle, _ptr(x, ctypes.c_double), cols))
        y = None
        if lib.qenivo_result_has_duals(handle):
            y = np.zeros(rows)
            _check(lib, handle, lib.qenivo_result_y(handle, _ptr(y, ctypes.c_double), rows))
        reduced = None
        if lib.qenivo_result_has_reduced_costs(handle):
            reduced = np.zeros(cols)
            _check(lib, handle, lib.qenivo_result_reduced_costs(handle, _ptr(reduced, ctypes.c_double), cols))
        col_basis = row_basis = None
        if lib.qenivo_result_has_basis(handle):
            col_basis = np.zeros(cols, dtype=np.int32)
            row_basis = np.zeros(rows, dtype=np.int32)
            _check(lib, handle, lib.qenivo_result_basis(
                handle, _ptr(col_basis, ctypes.c_int32), cols, _ptr(row_basis, ctypes.c_int32), rows))
        cert = lib.qenivo_result_certificate_json(handle) or b""
        status = int(lib.qenivo_result_status(handle))
        status_name = (lib.qenivo_result_status_name(handle) or b"").decode()
        return SolveResult(status, status_name, float(obj.value), x, y, reduced, col_basis, row_basis,
                           int(lib.qenivo_result_iterations(handle)), cert.decode())
    finally:
        lib.qenivo_model_free(handle)
        del cb_ref
