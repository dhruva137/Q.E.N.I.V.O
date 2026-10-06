"""Build a public-only source tree and gate it for open-core honesty.

Usage::

    python packaging/export_public.py [--out DIR] [--pytest] [--skip-pytest]

Steps:
  1. Copy the repo into ``DIR`` (default ``dist/public_tree``), excluding
     ``PRIVATE_PATHS`` and usual build/cache noise.
  2. Fail if any remaining file text matches a private-import marker, or if a
     private path somehow remains.
  3. Optionally run the offline public pytest suite inside the tree when free
     RAM is at least 1.5 GB (skipped otherwise; never starts heavy/network jobs).

Exit code 0 = gate passed.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_paths import (  # noqa: E402
    PRIVATE_IMPORT_MARKERS,
    PRIVATE_PATHS,
    is_private_rel,
)

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".hypothesis",
    ".ruff_cache", ".mypy_cache", "dist", "build", "runs", "egg-info",
    "node_modules",
}
SKIP_SUFFIXES = {".pyc", ".pyo", ".so", ".pyd", ".dll", ".egg"}
TEXT_SUFFIXES = {
    ".py", ".md", ".toml", ".cfg", ".ini", ".txt", ".yml", ".yaml", ".json",
    ".jsonl", ".html", ".css", ".js", ".cpp", ".hpp", ".h", ".c", ".cu",
    ".bat", ".sh", ".cmake", ".rst", ".cff", "",
}
MIN_FREE_RAM_GB = 1.5


def _free_ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return None
            return stat.ullAvailPhys / (1024 ** 3)
        # Linux
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:  # noqa: BLE001
        return None
    return None


def _should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if is_private_rel(rel):
        return True
    parts = path.parts
    for p in parts:
        if p in SKIP_DIRS or p.endswith(".egg-info"):
            return True
    if path.suffix in SKIP_SUFFIXES:
        return True
    # Local benchmark downloads / results stay out of the public export artifact
    # (published results under git may still be copied if present and not private).
    if rel.startswith("bench/instances/"):
        return True
    return False


def copy_public_tree(out: Path) -> int:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    n = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        d = Path(dirpath)
        # prune
        kept = []
        for name in dirnames:
            child = d / name
            if _should_skip(child):
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            src = d / name
            if _should_skip(src):
                continue
            rel = src.relative_to(ROOT)
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    return n


# Files that intentionally document or load the Pro entry-point group.
_ALLOW_PRIVATE_MARKERS = {
    "packaging/private_paths.py",
    "packaging/export_public.py",
    "packaging/build_pro.py",
    "docs/EDITIONS.md",
    ".gitattributes",
    "src/qenivo/edition.py",
    "src/qenivo/engines/registry.py",
}


def _marker_hits(text: str, marker: str) -> bool:
    """True if ``marker`` appears as a real reference, not a longer identifier.

    Avoids false positives such as ``qenivo_probe`` matching ``qenivo_pro``.
    """
    if marker not in text:
        return False
    if re.search(r"[A-Za-z0-9_]", marker):
        return re.search(r"(?<![A-Za-z0-9_])" + re.escape(marker) + r"(?![A-Za-z0-9_])", text) is not None
    return True


def check_no_private_residue(tree: Path) -> list[str]:
    errors: list[str] = []
    for p in PRIVATE_PATHS:
        if (tree / p).exists():
            errors.append(f"private path present in export: {p}")
    for path in tree.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(tree).as_posix()
        if is_private_rel(rel):
            errors.append(f"private path present in export: {rel}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.suffix:
            continue
        if rel in _ALLOW_PRIVATE_MARKERS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for marker in PRIVATE_IMPORT_MARKERS:
            if _marker_hits(text, marker):
                errors.append(f"{rel}: contains private marker {marker!r}")
    return errors


def run_public_pytest(tree: Path) -> tuple[bool, str]:
    free = _free_ram_gb()
    if free is not None and free < MIN_FREE_RAM_GB:
        return True, f"pytest skipped: free RAM {free:.2f} GB < {MIN_FREE_RAM_GB} GB"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tree / "src") + os.pathsep + env.get("PYTHONPATH", "")
    # Short offline subset: provenance + plugin registry (no network, no GPU, no nightly).
    cmd = [
        sys.executable, "-m", "pytest", "-q",
        "tests/test_extensions.py",
        "tests/test_security.py",
        "-m", "not network and not nightly and not gpu and not slow",
        "--maxfail=3",
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=str(tree), env=env, capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        return False, "pytest timed out after 180s"
    ok = proc.returncode == 0
    detail = (proc.stdout or "")[-2000:] + "\n" + (proc.stderr or "")[-1000:]
    if free is not None:
        detail = f"free_ram_gb={free:.2f}\n" + detail
    return ok, detail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "public_tree")
    ap.add_argument("--pytest", action="store_true", default=True,
                    help="run a short public pytest subset (default on)")
    ap.add_argument("--skip-pytest", action="store_true",
                    help="build + import grep only")
    args = ap.parse_args(argv)
    out = args.out if args.out.is_absolute() else ROOT / args.out

    print(f"exporting public tree -> {out}")
    n = copy_public_tree(out)
    print(f"copied {n} files")

    errors = check_no_private_residue(out)
    if errors:
        print("GATE FAIL: private residue")
        for e in errors:
            print(" ", e)
        return 1
    print("private-path / import grep: clean")

    do_pytest = args.pytest and not args.skip_pytest
    if do_pytest:
        ok, detail = run_public_pytest(out)
        print(detail.strip())
        if not ok:
            print("GATE FAIL: public pytest")
            return 1
        if "pytest skipped" in detail:
            print("GATE PASS (pytest skipped for RAM)")
        else:
            print("GATE PASS (tree clean + short pytest)")
    else:
        print("GATE PASS (tree clean; pytest not requested)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
