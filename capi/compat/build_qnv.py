"""Build and run the QNV C test program (capi/compat) with the GCC-family toolchain.

    python capi/compat/build_qnv.py          build (cached by source hash) and run the test program

The program links qnv_compat.c, the QENIVO C ABI (capi/qenivo_c.cpp, capi/python_bridge.cpp)
and the native simplex core statically into one executable. The embedded-Python checks need the
interpreter's shared library, home and package path, which `run_env()` provides.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import site
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPI = HERE.parent
ROOT = CAPI.parent
CORE = ROOT / "src" / "qenivo" / "native" / "simplex_core.cpp"
SOURCES = (HERE / "qnv_compat.c", HERE / "qnv_compat.h", HERE / "test_qnv_compat.c", CAPI / "qenivo.h",
           CAPI / "qenivo_c.cpp", CAPI / "python_bridge.cpp", CAPI / "python_bridge.h", CORE)


def toolchain() -> tuple[str, str] | None:
    """(C compiler, C++ compiler) of the GCC family, or None."""
    for cc, cxx in (("gcc", "g++"), ("clang", "clang++")):
        if shutil.which(cc) and shutil.which(cxx):
            return cc, cxx
    return None


def _cache() -> Path:
    base = os.environ.get("QENIVO_CACHE") or str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / "qenivo")
    d = Path(base) / "qnv_compat"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run(cmd) -> None:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError("build failed: " + " ".join(map(str, cmd)) + "\n" + (p.stderr or p.stdout)[-3000:])


def build() -> Path:
    tc = toolchain()
    if tc is None:
        raise RuntimeError("no gcc/g++ or clang/clang++ on PATH")
    cc, cxx = tc
    h = hashlib.sha256((cc + cxx + sys.platform).encode())
    for s in SOURCES:
        h.update(s.read_bytes())
    exe = _cache() / f"test_qnv_compat_{h.hexdigest()[:16]}{'.exe' if os.name == 'nt' else ''}"
    if exe.exists():
        return exe
    obj = exe.with_suffix("")
    obj = obj.parent / (obj.name + "_obj")
    obj.mkdir(exist_ok=True)
    pic = [] if os.name == "nt" else ["-fPIC"]
    inc = ["-DQENIVO_BUILD", "-I", CAPI, "-I", HERE]
    strict = ["-Wall", "-Wextra", "-Werror"]
    _run([cxx, "-O2", "-std=c++17", *pic, "-c", "-o", obj / "simplex_core.o", CORE])
    _run([cxx, "-O2", "-std=c++17", *pic, *strict, *inc, "-c", "-o", obj / "qenivo_c.o", CAPI / "qenivo_c.cpp"])
    _run([cxx, "-O2", "-std=c++17", *pic, *strict, *inc, "-c", "-o", obj / "bridge.o", CAPI / "python_bridge.cpp"])
    _run([cc, "-O2", "-std=c11", "-pedantic", *strict, *inc, "-c", "-o", obj / "qnv.o", HERE / "qnv_compat.c"])
    _run([cc, "-O2", "-std=c11", *strict, *inc, "-c", "-o", obj / "test.o", HERE / "test_qnv_compat.c"])
    link = [cxx, "-o", exe, obj / "test.o", obj / "qnv.o", obj / "qenivo_c.o", obj / "bridge.o", obj / "simplex_core.o"]
    link += ["-static"] if os.name == "nt" else ["-pthread", "-ldl"]
    _run(link)
    return exe


def python_library() -> str:
    if os.name == "nt":
        return str(Path(sys.base_prefix) / f"python{sys.version_info.major}{sys.version_info.minor}.dll")
    return str(Path(sysconfig.get_config_var("LIBDIR") or "") / (sysconfig.get_config_var("LDLIBRARY") or ""))


def run_env() -> dict:
    """Environment for the embedded interpreter: its DLL, home and every package directory."""
    env = dict(os.environ)
    paths = [str(ROOT / "src"), *site.getsitepackages(), site.getusersitepackages()]
    env.update(QENIVO_PYTHON_DLL=python_library(), QENIVO_SRC=str(ROOT / "src"), PYTHONHOME=sys.base_prefix,
               PYTHONPATH=os.pathsep.join(p for p in paths if p))
    return env


def run(timeout: float = 300.0) -> subprocess.CompletedProcess:
    return subprocess.run([str(build())], capture_output=True, text=True, env=run_env(), timeout=timeout)


if __name__ == "__main__":
    r = run()
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    sys.exit(r.returncode)
