"""Generate the one-hour A100 Round-2 notebook.

    python notebooks/build_round2.py
      -> notebooks/a100_round2.ipynb

Code arrives via CODE_SOURCE:
  * "upload"  — qenivo.zip (Colab files.upload / session storage), or
  * "github"  — clone https://github.com/dhruva137/Q.E.N.I.V.O.git
                branch main, then copy the qenivo/ subfolder to /content/qenivo.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZIP = "qenivo.zip"
REPO = "https://github.com/dhruva137/Q.E.N.I.V.O.git"
BRANCH = "main"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


TITLE = rf"""
# QENIVO · A100 notebook · **Round 2** (about 1 hour)

Measures only what the first evidence run did not cover. Total intended Colab budget ≤ 60 min.

| Part | What it measures | Budget |
|---|---|---|
| stamp | GPU, driver, commit, `gpu_status()`, native simplex status | 2 min |
| tight | `Linf_520c`, `rmine15`, `cont1` at 1e-6 (60 s/trial): PDHG persistent off/on, refine float32+refinement, PDHG→1e-3 then crossover; host KKT; compare with cuPDLPx rows already in `e2_large_lp.jsonl` (not re-run) | 25 min |
| l5 | refinery level 5 (45,032 rows), tol 1e-4, batch 8 and 32, certified; HiGHS all cores with 8 min hard cap | 12 min |
| refinery | case 1 T4 homotopy on CPU: objective, max violation, feasible | 20 min |
| headline | `HEADLINE_ROUND2.md` from recorded rows only | 1 min |

## How to run

1. **Runtime → Change runtime type → A100 GPU** (High-RAM if offered) → **Save**.
2. **Give it the code** (pick one):
   * set `CODE_SOURCE = "upload"` and upload **`{ZIP}`**, or
   * set `CODE_SOURCE = "github"` to clone `{REPO}` branch `{BRANCH}` (copies the `qenivo/` subfolder).
3. **Runtime → Run all.** Allow multiple downloads when asked.

Crash safety: every result row is written the moment it exists; after every part a results zip is
downloaded. Re-run with the same `RUN_NAME` to resume.
"""

CONFIG = r"""
# ---- settings ---------------------------------------------------------------------------------
RUN_NAME      = "round2_run1"
PARTS         = "stamp,tight,l5,refinery,headline"
BUDGET        = {"stamp": 120, "tight": 1500, "l5": 720, "refinery": 1200, "headline": 60}
SELF_TEST     = True       # short --quick end-to-end check before the long run
AUTO_DOWNLOAD = True
CODE_SOURCE   = "upload"   # "upload" (zip) or "github" (clone main)
GITHUB_REPO   = "https://github.com/dhruva137/Q.E.N.I.V.O.git"
GITHUB_BRANCH = "main"
ZIP_NAME      = "qenivo.zip"
"""

SETUP = r"""
# ---- setup: GPU probe, code from zip upload OR github clone, install ladder --------------------
import os, subprocess, sys, shutil, glob, json, time, zipfile
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True)
assert gpu.returncode == 0, "No GPU: Runtime > Change runtime type > A100 GPU, then Run all again"
print("GPU:", gpu.stdout.strip())
RUN_DIR = f"/content/qenivo_results/{RUN_NAME}"
os.makedirs(RUN_DIR, exist_ok=True)

def project_root(folder):
    for d, dirs, fs in os.walk(folder):
        if "pyproject.toml" in fs and os.path.isdir(os.path.join(d, "src", "qenivo")):
            return d
    return None

def find_zip():
    hits = [p for p in glob.glob("/content/**/*.zip", recursive=True)
            if "qenivo_results" not in p and os.path.getsize(p) > 100_000]
    good = []
    for p in hits:
        try:
            with zipfile.ZipFile(p) as z:
                if any(n.endswith("src/qenivo/__init__.py") for n in z.namelist()):
                    good.append(p)
        except zipfile.BadZipFile:
            pass
    return sorted(good, key=lambda p: (os.path.basename(p) != ZIP_NAME, -os.path.getmtime(p)))

def install_from_root(root, label):
    shutil.rmtree("/content/qenivo", ignore_errors=True)
    shutil.copytree(root, "/content/qenivo")
    os.chdir("/content/qenivo")
    print(f"code: {label}")

src = (CODE_SOURCE or "upload").strip().lower()
if src == "github":
    dest = "/tmp/qenivo_src"
    shutil.rmtree(dest, ignore_errors=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", GITHUB_BRANCH,
                    GITHUB_REPO, dest], check=True, timeout=600)
    # Repo layout: <repo>/qenivo/ is the installable package (pyproject + src/qenivo).
    cand = os.path.join(dest, "qenivo")
    root = project_root(cand) or project_root(dest)
    assert root, f"qenivo package not found under {dest} after cloning {GITHUB_BRANCH}"
    install_from_root(root, f"git {GITHUB_REPO}@{GITHUB_BRANCH}")
elif src == "upload":
    zips = find_zip()
    if not zips:
        from google.colab import files
        print(f"Choose {ZIP_NAME}:")
        files.upload()
        zips = find_zip()
    assert zips, f"Upload {ZIP_NAME} (must contain src/qenivo/__init__.py)"
    shutil.rmtree("/tmp/code_x", ignore_errors=True)
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall("/tmp/code_x")
    root = project_root("/tmp/code_x")
    assert root, "zip is not a qenivo project"
    install_from_root(root, f"{zips[0]} ({os.path.getsize(zips[0])/1e6:.1f} MB)")
else:
    raise SystemExit(f"CODE_SOURCE must be 'upload' or 'github', got {CODE_SOURCE!r}")

os.environ["CUPY_CACHE_DIR"] = f"/tmp/cupy_cache_{int(time.time())}"
os.environ["PYTHONUNBUFFERED"] = "1"

def pip(*args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *args],
                       capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or r.stdout)[-1500:]

def importable(mod):
    return subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True).returncode == 0

ok, err = pip("-e", ".", "highspy", "matplotlib")
if not ok:
    print("full install failed, trying step by step. pip said:", err, sep=chr(10))
    for extra in ("numpy", "scipy", "highspy", "matplotlib"):
        if not importable(extra):
            print("  ", extra, "ok" if pip(extra)[0] else "FAILED")
    ok, err = pip("--no-deps", "--no-build-isolation", "-e", ".")
    if not ok:
        ok, err = pip("--no-deps", ".")
if not importable("qenivo"):
    srcp = os.path.abspath("src")
    os.environ["PYTHONPATH"] = srcp + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, srcp)
    print("using the source folder directly:", srcp)

GPU_PROBE = "from qenivo.kernels.backend import gpu_available, gpu_status; print(gpu_available(), gpu_status())"

def gpu_ok():
    r = subprocess.run([sys.executable, "-c", GPU_PROBE], capture_output=True, text=True)
    return r.stdout.strip().startswith("True"), (r.stdout.strip() or r.stderr[-300:])

if not importable("cupy"):
    pip("cupy-cuda12x")
ok_gpu, why = gpu_ok()
if not ok_gpu:
    pip("cupy-cuda12x[ctk]")
    ok_gpu, why = gpu_ok()
print("GPU kernels:", "ok" if ok_gpu else f"UNAVAILABLE, the run continues on the CPU ({why})")
chk = subprocess.run([sys.executable, "-c",
                      "import qenivo, numpy, scipy; print('qenivo', qenivo.__version__, '| numpy', numpy.__version__)"],
                     capture_output=True, text=True)
assert chk.returncode == 0, "qenivo cannot be imported: " + chk.stderr[-1500:]
print("install: ok |", chk.stdout.strip(), "| highspy", "ok" if importable("highspy") else "missing",
      "| cupy", "ok" if importable("cupy") else "missing")
"""

RUN = r"""
# ---- self-test, then every part with its budget; results zipped + downloaded after each part ----
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

def save_results(final=False):
    z = shutil.make_archive(f"/content/{RUN_NAME}_results", "zip", RUN_DIR)
    if AUTO_DOWNLOAD or final:
        try:
            from google.colab import files
            files.download(z)
        except Exception as e:
            print("download not started (results stay in", RUN_DIR, "):", e)
    return z

SCRIPT = "bench/round2_campaign.py"
if SELF_TEST:
    t0 = time.time()
    shutil.rmtree("/tmp/r2_selftest", ignore_errors=True)
    rc = stream([sys.executable, "-u", SCRIPT, "--run-dir", "/tmp/r2_selftest", "--quick",
                 "--only", "stamp,tight,l5,refinery,headline"], 900)
    print(f"\n=== SELF-TEST {'PASSED' if rc == 0 else 'REPORTED A PROBLEM (see above); long run continues'} "
          f"in {time.time()-t0:.0f}s ===\n")

for part in PARTS.split(","):
    part = part.strip()
    if not part:
        continue
    t0 = time.time()
    rc = stream([sys.executable, "-u", SCRIPT, "--run-dir", RUN_DIR, "--only", part], BUDGET.get(part))
    hit = time.time() - t0 >= BUDGET.get(part, 1e12) - 2
    print(f"--- {part}: {time.time()-t0:.0f}s" + ("  (budget reached: saved rows kept)" if hit else ""))
    save_results()
"""

SHOW = r"""
# ---- the round-2 headline (from recorded rows only) ----
from IPython.display import Markdown, display
subprocess.run([sys.executable, "-u", "bench/round2_campaign.py", "--run-dir", RUN_DIR, "--only", "headline"],
               capture_output=True)
hp = f"{RUN_DIR}/HEADLINE_ROUND2.md"
if os.path.exists(hp):
    display(Markdown(open(hp).read()))
else:
    print("HEADLINE_ROUND2.md not written yet")
"""

FINAL = r"""
# ---- final results zip ----
print("results zip:", save_results(final=True))
"""


def build():
    cells = [md(TITLE), code(CONFIG), code(SETUP), code(RUN), code(SHOW), code(FINAL)]
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU",
                       "colab": {"name": "a100_round2.ipynb", "gpuType": "A100",
                                 "machine_shape": "hm"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "cells": cells}
    out = HERE / "a100_round2.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    build()
