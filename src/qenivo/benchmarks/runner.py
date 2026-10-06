"""Thin wrapper around ``bench/run.py`` (import preferred, subprocess fallback)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# qenivo package root = parents[3] from src/qenivo/benchmarks/runner.py
#   runner.py -> benchmarks -> qenivo -> src -> <package root>
_PKG_ROOT = Path(__file__).resolve().parents[3]
_BENCH = _PKG_ROOT / "bench"


def available_sets() -> list[str]:
    """Return declared benchmark set names from ``bench/sets.py``."""
    sets_path = _BENCH / "sets.py"
    if not sets_path.is_file():
        return []
    spec = importlib.util.spec_from_file_location("qenivo_bench_sets", sets_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return sorted(mod.SETS.keys())


def _scrub_comparator_imports() -> None:
    """Drop HiGHS/highspy modules that bench/run.py may have imported for metadata.

    The solve path must stay provenance-clean; comparator imports belong only in
    the bench subprocess, never in the long-lived qenivo process / test suite.
    """
    for name in list(sys.modules):
        top = name.split(".", 1)[0]
        if top == "highspy" or name.startswith("highspy."):
            del sys.modules[name]


def _load_run_module():
    run_path = _BENCH / "run.py"
    if not run_path.is_file():
        raise FileNotFoundError(f"benchmark runner not found at {run_path}")
    # Ensure bench/ imports (fetch, sets) resolve.
    bench_str = str(_BENCH)
    src_str = str(_PKG_ROOT / "src")
    for p in (bench_str, src_str):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("qenivo_bench_run", run_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def run_benchmark(
    set_name: str,
    *,
    engine: str = "auto",
    time_limit: float = 120.0,
    out: str | Path | None = None,
    backend: str = "numpy",
    jobs: int = 1,
    no_ref: bool = True,
    tol: float = 1e-6,
    via_subprocess: bool = True,
) -> dict:
    """Run a declared benchmark set and return ``{"out": path, "summary": ..., "meta": ...}``.

    Defaults to ``no_ref=True`` so HiGHS is not required for CLI smoke runs.
    Defaults to ``via_subprocess=True`` so ``bench/run.py`` comparator imports
    (highspy) never pollute the calling process.
    """
    if set_name not in available_sets():
        known = ", ".join(available_sets()) or "(none)"
        raise ValueError(f"unknown benchmark set {set_name!r}; known: {known}")

    out_path = Path(out) if out else (_BENCH / "results" / f"{set_name}_cli.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        "--set", set_name,
        "--engine", engine,
        "--backend", backend,
        "--tol", str(tol),
        "--time-limit", str(time_limit),
        "--jobs", str(jobs),
        "--out-file", str(out_path),
    ]
    if no_ref:
        argv.append("--no-ref")

    if via_subprocess:
        cmd = [sys.executable, str(_BENCH / "run.py"), *argv]
        proc = subprocess.run(cmd, cwd=str(_PKG_ROOT), capture_output=True, text=True)
        text = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0 and not out_path.exists():
            raise RuntimeError(f"bench/run.py failed ({proc.returncode}): {text[-500:]}")
    else:
        try:
            mod = _load_run_module()
            mod.main(argv)
        finally:
            _scrub_comparator_imports()

    payload = {}
    if out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    summary = ""
    if payload.get("runs") is not None:
        try:
            mod = _load_run_module()
            summary = mod.summarise(payload["runs"], set_name)
        except Exception:  # noqa: BLE001
            summary = f"{set_name}: {len(payload.get('runs', []))} runs -> {out_path}"
        finally:
            _scrub_comparator_imports()
    return {
        "out": str(out_path),
        "summary": summary,
        "meta": payload.get("meta", {}),
        "n_runs": len(payload.get("runs", [])),
        "set": set_name,
        "engine": engine,
        "time_limit": time_limit,
    }


def smoke_namespace(**kwargs) -> SimpleNamespace:
    """Build an argparse-like namespace matching ``bench/run.py`` expectations."""
    return SimpleNamespace(
        set=kwargs.get("set"),
        instances=kwargs.get("instances"),
        engine=kwargs.get("engine", "auto"),
        backend=kwargs.get("backend", "numpy"),
        tol=kwargs.get("tol", 1e-6),
        time_limit=kwargs.get("time_limit", 5.0),
        jobs=kwargs.get("jobs", 1),
        no_ref=kwargs.get("no_ref", True),
        keep_certs=False,
        out_file=kwargs.get("out_file"),
    )
