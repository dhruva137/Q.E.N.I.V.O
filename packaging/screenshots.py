"""Capture the deck screenshots (demo/screens/README.md) at 1600x900 from the live console.

    python packaging/screenshots.py            all shots
    python packaging/screenshots.py S03 S05    some of them

Starts the console on a free loopback port, then opens each scripted view (index.html ?shot=...)
in headless Microsoft Edge with a throwaway profile and saves a PNG. Every number in a shot comes
from a live solve on this machine; nothing is typed in. Shots of results files (S07, S08) and of the
operating system (S11 audit file, S12 netstat) are not UI views and are not taken here.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "screens"
SHOTS = {
    "S01_problem_chrome": "shot=home",
    "S02_model_loaded": "shot=solve&model=sample",
    "S03_solve_certified": "shot=solve&model=refinery_l3&engine=simplex&tol=1e-8&run=1",
    "S04_shadow_prices": "shot=solve&model=refinery_l3&engine=simplex&tol=1e-8&run=1&scroll=limits",
    "S05_crude_valuation": "shot=value&model=refinery_l3&run=1",
    "S06_case_stack_badges": "shot=cases&model=refinery_l3&run=1",
    "S09_verify": "shot=verify&model=refinery_l3&engine=simplex&run=1",
    "S10_provenance": "shot=about",
    "S13_sensitivity": "shot=range&model=williams&run=1",
}


def _edge() -> str:
    for p in (Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft/Edge/Application/msedge.exe",
              Path(os.environ.get("ProgramFiles", "")) / "Microsoft/Edge/Application/msedge.exe"):
        if p.exists():
            return str(p)
    found = shutil.which("msedge") or shutil.which("chrome") or shutil.which("chromium")
    if not found:
        raise SystemExit("no Microsoft Edge / Chromium found for headless capture")
    return found


def main(argv=None) -> int:
    want = [a.upper() for a in (argv if argv is not None else sys.argv[1:])]
    shots = {k: v for k, v in SHOTS.items() if not want or k.split("_")[0] in want}
    sys.path.insert(0, str(ROOT / "src"))
    from qenivo.server import make_server
    httpd = make_server("127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    OUT.mkdir(parents=True, exist_ok=True)
    edge = _edge()
    profile = tempfile.mkdtemp(prefix="qenivo_shots_")
    try:
        for name, query in shots.items():
            png = OUT / f"{name}.png"
            url = f"http://127.0.0.1:{httpd.server_port}/?{query}&theme=dark"
            t0 = time.perf_counter()
            subprocess.run([edge, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                            "--disable-extensions", f"--user-data-dir={profile}", "--window-size=1600,900",
                            "--virtual-time-budget=180000", f"--screenshot={png}", url],
                           check=False, capture_output=True, timeout=600)
            ok = png.exists() and png.stat().st_size > 10_000
            print(f"{name:24s} {'ok' if ok else 'FAILED'}  {png.stat().st_size // 1024 if png.exists() else 0} KiB"
                  f"  {time.perf_counter() - t0:5.1f}s")
    finally:
        httpd.shutdown()
        httpd.server_close()
        shutil.rmtree(profile, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
