"""Generate the Kaggle notebook: architecture campaign sized for 2 x T4 (about 4.5 hours).

    python notebooks/build_kaggle.py   ->  notebooks/kaggle_t4.ipynb

Kaggle specifics handled here: the code arrives as a Kaggle Dataset under /kaggle/input (Kaggle
unzips an uploaded zip, so both a folder and a zip are accepted); results go to
/kaggle/working/qenivo_results/<RUN_NAME>, which Kaggle keeps as the notebook's output after
"Save Version -> Save & Run All" (runs in the background for up to 12 hours, no browser needed).
"""
import json
from pathlib import Path

from build_notebook import CUPDLPX, code, md

HERE = Path(__file__).resolve().parent
RUN_NAME = "kaggle_t4_run1"
PARTS = "a1,a2,a3,a4,a5,a7,a8,m1"
BUDGET = {"a1": 900, "a2": 4500, "a3": 1800, "a4": 1800, "a5": 2400, "a7": 2400, "a8": 1500, "m1": 1500}
EXTRA_PIP = ["ortools", "pyscipopt"]

TITLE = r"""
# QENIVO · Kaggle notebook · **Architecture, precision and world** on 2 x T4 (about 4.5 hours)

| Part | What it measures | Budget |
|---|---|---|
| a1 | preflight: download every benchmark set (retries + mirrors), check each file reads; manifest | 15 min |
| a2 | coverage through the full pipeline (route, solve, certify, verify): Netlib, Netlib infeasible, Kennington, Maros-Meszaros QP, large Mittelmann LPs; HiGHS alongside | 75 min |
| a3 | engine map: every engine on every LP up to 40k rows; router thresholds fitted from data | 30 min |
| a4 | PDHG design ablation on the T4 (restarts, Halpern, reflection, scaling, check interval, own vs vendor kernels) | 30 min |
| a5 | world: HiGHS, OR-Tools GLOP + PDLP, SCIP, cuPDLPx (GPU) on the same machine | 40 min |
| a7 | **precision:** float64 PDHG vs float32 PDHG + float64 refinement at 1e-6 and 1e-8 (T4 float32 is ~32x its float64) | 40 min |
| a8 | **two GPUs:** one refinery what-if batch on 1 T4 vs both T4s vs HiGHS on all CPU cores | 25 min |
| m1 | MIPLIB 2017 subset: our branch and cut vs HiGHS, official optima | 25 min |
| a6 | `PARITY.md`, `ARCHITECTURE_DECISIONS.md`, `WORLD.md`, `T4.md` | 1 min |

## How to run (once)

1. **Add the code:** right panel → **Add Input** → **Upload** → choose `qenivo.zip` → name it
   `qenivo` → **Create**. (Kaggle unzips it; that is fine.)
2. **Settings** (right panel): **Accelerator → GPU T4 x2**, **Internet → On** (needs a phone-verified
   Kaggle account; the benchmark sets and comparator solvers download from their official sites).
3. **Save Version** (top right) → **Save & Run All (Commit)** → Save. It runs in the background: you can
   close the browser or sleep. When it finishes, open the version → **Output** → download
   `kaggle_t4_run1_results.zip`.

Every result row is written the moment it exists, each part has a hard time budget, and the results zip is
rebuilt after every part, so even a cut-short run keeps everything finished. With Internet off, the run
still completes on the built-in models, and the downloads are recorded as missing.
"""

CONFIG = r"""
# ---- settings (the defaults are the full run) -------------------------------------------------
RUN_NAME  = "{run}"
PARTS     = "{parts}"
BUDGET    = {budget}   # seconds per part, hard stop (saved rows are kept)
SELF_TEST = True       # short end-to-end check of the whole pipeline first
"""

SETUP = r"""
# ---- setup: GPUs, code from /kaggle/input, fresh install, comparators in their own environment -----
import os, subprocess, sys, shutil, glob, json, time, zipfile, urllib.request
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True)
print("GPUs:", gpu.stdout.strip() or "none (Settings > Accelerator > GPU T4 x2). The run continues on the CPU.")
RUN_DIR = f"/kaggle/working/qenivo_results/{RUN_NAME}"
os.makedirs(RUN_DIR, exist_ok=True)

def project_root(folder):
    for d, dirs, fs in os.walk(folder):
        if "pyproject.toml" in fs and os.path.isdir(os.path.join(d, "src", "qenivo")):
            return d
    return None

root = project_root("/kaggle/input")
if root is None:                                   # a zip that was not unpacked
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        try:
            with zipfile.ZipFile(z) as f:
                if any(n.endswith("src/qenivo/__init__.py") for n in f.namelist()):
                    shutil.rmtree("/tmp/code_x", ignore_errors=True)
                    f.extractall("/tmp/code_x")
                    root = project_root("/tmp/code_x")
                    break
        except zipfile.BadZipFile:
            pass
assert root, "Code not found: Add Input > Upload > qenivo.zip (see the first cell)"
shutil.rmtree("/tmp/qenivo", ignore_errors=True)
shutil.copytree(root, "/tmp/qenivo")
os.chdir("/tmp/qenivo")
print("code:", root)

def online():
    try:
        urllib.request.urlopen("https://pypi.org/simple/highspy/", timeout=10)
        return True
    except Exception:
        return False
ONLINE = online()
print("internet:", "on" if ONLINE else "OFF: benchmark downloads and comparators will be recorded as missing "
      "(Settings > Internet > On for the full run)")
os.environ["CUPY_CACHE_DIR"] = f"/tmp/cupy_cache_{int(time.time())}"
os.environ["PYTHONUNBUFFERED"] = "1"

def pip(*args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *args], capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or r.stdout)[-1500:]
def importable(mod):
    return subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True).returncode == 0
ok, err = pip("-e", ".", "highspy", "matplotlib") if ONLINE else (False, "offline")
if not ok:
    ok, err = pip("--no-deps", "--no-build-isolation", "-e", ".")
if not importable("qenivo"):                      # last resort: the source folder, no install
    os.environ["PYTHONPATH"] = os.path.abspath("src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, os.path.abspath("src"))
GPU_PROBE = "from qenivo.kernels.backend import gpu_available, gpu_status; print(gpu_available(), gpu_status())"
def gpu_ok():
    r = subprocess.run([sys.executable, "-c", GPU_PROBE], capture_output=True, text=True)
    return r.stdout.strip().startswith("True"), (r.stdout.strip() or r.stderr[-300:])
if not importable("cupy"):
    pip("cupy-cuda12x")
ok_gpu, why = gpu_ok()
if not ok_gpu:                                    # device seen but kernels cannot compile: add the CUDA headers
    pip("cupy-cuda12x[ctk]")
    ok_gpu, why = gpu_ok()
print("GPU kernels:", "ok" if ok_gpu else f"UNAVAILABLE, the run continues on the CPU ({why})")
chk = subprocess.run([sys.executable, "-c", "import qenivo, numpy, scipy; print('qenivo', qenivo.__version__)"],
                     capture_output=True, text=True)
assert chk.returncode == 0, "qenivo cannot be imported: " + chk.stderr[-1500:]
print("install: ok |", chk.stdout.strip(), "| highspy", "ok" if importable("highspy") else "missing",
      "| cupy", "ok" if importable("cupy") else "missing")

if ONLINE and EXTRA_PIP:   # comparators in their own environment, so they can never break the solver's packages
    envd = "/tmp/cmpenv"
    shutil.rmtree(envd, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], capture_output=True)
    made = subprocess.run(["uv", "venv", "-q", "--python", sys.executable, envd], capture_output=True).returncode == 0
    if not made:
        made = subprocess.run([sys.executable, "-m", "venv", envd], capture_output=True).returncode == 0
    if made:
        py = f"{envd}/bin/python"
        for pkg in ["numpy scipy highspy"] + EXTRA_PIP:
            cmd = (["uv", "pip", "install", "-q", "--python", py] if shutil.which("uv") else [py, "-m", "pip", "install", "-q"]) + pkg.split()
            good = subprocess.run(cmd, capture_output=True, text=True).returncode == 0
            print(("comparator ready    " if good else "comparator skipped  ") + pkg.split()[0])
        os.environ["CMP_PYTHON"] = py
"""

RUN = r"""
# ---- self-test, then every part with its budget; the results zip is rebuilt after each part ----
import threading
def stream(cmd, budget=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    t = threading.Timer(budget, p.kill) if budget else None
    if t: t.start()
    for line in p.stdout:
        print(line, end="")
    p.wait()
    if t: t.cancel()
    return p.returncode

def save_results():
    return shutil.make_archive(f"/kaggle/working/{RUN_NAME}_results", "zip", RUN_DIR)

if SELF_TEST:
    t0 = time.time()
    shutil.rmtree("/tmp/selftest", ignore_errors=True)
    rc = stream([sys.executable, "-u", "bench/arch_campaign.py", "--run-dir", "/tmp/selftest", "--quick",
                 "--only", "a1,a2,a7,a8,m1"], 900)
    print(f"\n=== SELF-TEST {'PASSED' if rc == 0 else 'REPORTED A PROBLEM (see above); the long run continues'} in {time.time()-t0:.0f}s ===\n")

for part in PARTS.split(","):
    t0 = time.time()
    rc = stream([sys.executable, "-u", "bench/arch_campaign.py", "--run-dir", RUN_DIR, "--only", part], BUDGET.get(part))
    hit = time.time() - t0 >= BUDGET.get(part, 1e12) - 2
    print(f"--- {part}: {time.time()-t0:.0f}s" + ("  (budget reached: saved rows kept)" if hit else ""))
    save_results()
"""

SHOW = r"""
# ---- the tables for the slides ----
from IPython.display import Markdown, display
subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0,'bench'); import arch_campaign as a; a.a6_report('{RUN_DIR}')"],
               capture_output=True)
for f in ("PARITY.md", "T4.md", "ARCHITECTURE_DECISIONS.md", "WORLD.md"):
    if os.path.exists(f"{RUN_DIR}/{f}"): display(Markdown(open(f"{RUN_DIR}/{f}").read()))
print("results zip:", save_results(), "(Output tab after the run)")
"""


def main():
    cells = [md(TITLE),
             code(CONFIG.replace("{run}", RUN_NAME).replace("{parts}", PARTS).replace("{budget}", repr(BUDGET))),
             code(SETUP.replace("EXTRA_PIP", repr(EXTRA_PIP))),
             code(CUPDLPX.replace("print(\"cuPDLPx:\"", "print(\"cuPDLPx:\"")),
             code(RUN), code(SHOW)]
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True,
                                  "language": "python", "sourceType": "notebook"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "cells": cells}
    out = HERE / "kaggle_t4.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
