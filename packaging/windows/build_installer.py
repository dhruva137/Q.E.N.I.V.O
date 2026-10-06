"""Build QENIVO-Setup-<version>.exe: native libraries, PyInstaller folder, Inno Setup installer.

    python packaging/windows/build_installer.py            all three steps
    python packaging/windows/build_installer.py --native   only compile and check the native libraries

Needs a C++ compiler (MinGW g++) on PATH for step 1, `pip install pyinstaller pywebview` for step 2
and Inno Setup 6 (ISCC.exe) for step 3. Each step stops with a plain message when its tool is missing.
The native libraries are compiled once here and shipped inside the bundle, so the planner's machine
needs no compiler (their cache key ignores the Windows release; see engines/_native_abi.py).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "build" / "qenivo_native"
LOADERS = ["qenivo.engines.native_simplex", "qenivo.engines.native_ipm", "qenivo.engines.exact_lp",
           "qenivo.io.native_mps"]


def stage_native() -> list[Path]:
    """Compile every native library into STAGE and confirm each one loads."""
    if shutil.which(os.environ.get("CXX") or "g++") is None:
        raise SystemExit("step 1: no C++ compiler on PATH (install MinGW-w64, e.g. "
                         "winget install BrechtSanders.WinLibs.POSIX.UCRT)")
    if STAGE.exists():
        shutil.rmtree(STAGE)
    env = {**os.environ, "QENIVO_CACHE": str(STAGE), "QENIVO_NATIVE": "1", "PYTHONPATH": str(ROOT / "src")}
    check = ("import importlib, sys\n"
             "bad = [m for m in sys.argv[1:] if importlib.import_module(m).library() is None]\n"
             "print('not built:', bad) if bad else print('all native libraries load')\n"
             "sys.exit(1 if bad else 0)\n")
    r = subprocess.run([sys.executable, "-c", check, *LOADERS], env=env, capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip()[-2000:], sep="\n")
    if r.returncode != 0:
        raise SystemExit("step 1: a native library failed to build or load (see above)")
    libs = sorted((STAGE / "native").glob("qenivo_*.dll"))
    for p in libs:
        print(f"  staged {p.name}  {p.stat().st_size / 1024:.0f} KiB")
    return libs


def freeze() -> Path:
    if importlib.util.find_spec("PyInstaller") is None:
        raise SystemExit("step 2: PyInstaller is not installed (pip install pyinstaller pywebview)")
    if importlib.util.find_spec("webview") is None:
        print("warning: pywebview is not installed; the app will open the default browser instead of a window")
    env = {**os.environ, "QENIVO_STAGED_NATIVE": str(STAGE)}
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                    "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build" / "pyinstaller"),
                    str(ROOT / "packaging" / "windows" / "qenivo_desktop.spec")], check=True, env=env, cwd=ROOT)
    exe = ROOT / "dist" / "QENIVO" / "QENIVO.exe"
    if not exe.exists():
        raise SystemExit("step 2: PyInstaller finished but dist/QENIVO/QENIVO.exe is missing")
    return exe


def installer() -> Path:
    iscc = shutil.which("ISCC") or next((str(p) for p in (
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe") if p.exists()), None)
    if iscc is None:
        raise SystemExit("step 3: Inno Setup 6 is not installed (winget install JRSoftware.InnoSetup)")
    subprocess.run([iscc, str(ROOT / "packaging" / "windows" / "qenivo.iss")], check=True)
    from qenivo import __version__
    out = ROOT / "dist" / f"QENIVO-Setup-{__version__}.exe"
    if not out.exists():
        raise SystemExit(f"step 3: ISCC finished but {out.name} is missing")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--native", action="store_true", help="only compile and check the native libraries")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    stage_native()
    if a.native:
        return 0
    print("frozen app:", freeze())
    print("installer:", installer())
    return 0


if __name__ == "__main__":
    sys.exit(main())
