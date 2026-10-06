# PyInstaller spec for the QENIVO desktop app (one-folder build). Run through build_installer.py,
# which first compiles the native libraries into build/qenivo_native so the planner's machine
# needs no compiler. GPU (CuPy) and benchmark-only comparators are left out of the bundle.
# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parents[1]
NATIVE = Path(os.environ["QENIVO_STAGED_NATIVE"])          # set by build_installer.py

datas = collect_data_files("qenivo", includes=["**/*.html", "**/*.cpp", "**/*.hpp", "**/*.h",
                                               "**/CMakeLists.txt"])
datas += [(str(NATIVE / "native"), "qenivo_native/native")]

a = Analysis(
    [str(ROOT / "packaging" / "windows" / "desktop_entry.py")],
    pathex=[str(ROOT / "src")],
    datas=datas,
    hiddenimports=collect_submodules("qenivo") + ["webview"],
    excludes=["cupy", "cupyx", "highspy", "pulp", "pyomo", "cvxpy", "ortools", "matplotlib", "tkinter",
              "IPython", "pytest", "hypothesis"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="QENIVO", console=False,
          icon=None, version=None)
coll = COLLECT(exe, a.binaries, a.datas, name="QENIVO")
