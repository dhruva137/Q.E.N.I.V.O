"""Generate the two A100 notebooks. Both use ONE code file: qenivo.zip.

    python notebooks/build_notebook.py
      -> notebooks/a100_evidence.ipynb
      -> notebooks/a100_architecture.ipynb
Everything runs inside Colab: no Google Drive, no GitHub, no tokens. Results are written row by row to
/content/qenivo_results/<RUN_NAME>/ and, after every part, zipped and downloaded to your computer.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZIP = "qenivo.zip"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


HOWTO = rf"""
## How to run (3 steps, nothing else)

1. **Runtime → Change runtime type → A100 GPU** (turn on **High-RAM** if offered) → **Save**.
2. **Give it the code:** click the **folder icon** on the left → **Upload to session storage** (the page-with-arrow
   icon) → choose **`{ZIP}`**. (Or skip this: the notebook shows a **Choose Files** button when it needs the file.)
3. **Runtime → Run all.** Keep the tab open. If the browser asks *"allow multiple downloads?"*, click **Allow**.

**Datasets download by themselves** from the official public sources (Netlib at netlib.org and its coin-or mirror,
Kennington at netlib.org, the Mittelmann LP set at plato.asu.edu, the Maros-Meszaros QP set with its published
optima, the open refinery-petrochemical benchmark on GitHub). No account or token is needed. Each file is tried 3
times across mirrors; a file that still fails is recorded as missing and skipped, so the run never stops for it.

**If Colab crashes:** every result row is written to `/content/qenivo_results/<RUN_NAME>/` the moment it
exists, and after every part the notebook **downloads a zip of all results so far to your computer**
(`<RUN_NAME>_results.zip`). If the session survives, run all again with the same `RUN_NAME` and it continues
where it stopped. If the machine was lost, the zips already on your computer hold every finished part.
"""

CONFIG = r"""
# ---- settings (the defaults are the full run) -------------------------------------------------
RUN_NAME      = "{run}"
PARTS         = "{parts}"
BUDGET        = {budget}   # seconds per part, hard stop (saved rows are kept)
SELF_TEST     = True       # 2-minute end-to-end check of the whole pipeline before the long run
AUTO_DOWNLOAD = True       # download the results zip to your computer after every part
"""

SETUP = r"""
# ---- setup: GPU check, code from the zip, fresh install --------------------------------------------
import os, subprocess, sys, shutil, glob, json, time, zipfile
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True)
assert gpu.returncode == 0, "No GPU: Runtime > Change runtime type > A100 GPU, then Run all again"
print("GPU:", gpu.stdout.strip())
ZIP_NAME = "{zip}"
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

zips = find_zip()
if not zips:
    from google.colab import files
    print(f"Choose {ZIP_NAME} (about 0.3 MB):")
    up = files.upload()
    zips = find_zip()
    if not zips:
        raise SystemExit(f"That file is not the project zip. Upload {ZIP_NAME} as you received it.")
shutil.rmtree("/tmp/code_x", ignore_errors=True)
with zipfile.ZipFile(zips[0]) as z:
    z.extractall("/tmp/code_x")
root = project_root("/tmp/code_x")
shutil.rmtree("/content/qenivo", ignore_errors=True)
shutil.copytree(root, "/content/qenivo")
os.chdir("/content/qenivo")
print(f"code: {zips[0]} ({os.path.getsize(zips[0])/1e6:.1f} MB)")
os.environ["CUPY_CACHE_DIR"] = f"/tmp/cupy_cache_{int(time.time())}"     # GPU kernels compiled fresh
os.environ["PYTHONUNBUFFERED"] = "1"
def pip(*args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *args], capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or r.stdout)[-1500:]
def importable(mod):
    return subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True).returncode == 0
# install ladder: each step is tried only if the one before failed, so the run never stops here
ok, err = pip("-e", ".", "highspy", "matplotlib")
if not ok:
    print("full install failed, trying step by step. pip said:", err, sep=chr(10))
    for extra in ("numpy", "scipy", "highspy", "matplotlib"):
        if not importable(extra):
            print("  ", extra, "ok" if pip(extra)[0] else "FAILED")
    ok, err = pip("--no-deps", "--no-build-isolation", "-e", ".")
    if not ok:
        ok, err = pip("--no-deps", ".")
if not importable("qenivo"):   # last resort: use the source folder directly, no install at all
    src = os.path.abspath("src")
    os.environ["PYTHONPATH"] = src + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, src)
    print("using the source folder directly:", src)
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
chk = subprocess.run([sys.executable, "-c", "import qenivo, numpy, scipy; print('qenivo', qenivo.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__)"],
                     capture_output=True, text=True)
assert chk.returncode == 0, "qenivo cannot be imported: " + chk.stderr[-1500:]
print("install: ok |", chk.stdout.strip(), "| highspy", "ok" if importable("highspy") else "missing",
      "| cupy", "ok" if importable("cupy") else "missing")

if EXTRA_PIP:   # comparators get their own environment, so they can never break the solver's packages
    envd = "/content/cmpenv"
    shutil.rmtree(envd, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], capture_output=True)
    made = subprocess.run(["uv", "venv", "-q", "--python", sys.executable, envd], capture_output=True).returncode == 0
    if not made:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "virtualenv"], capture_output=True)
        made = subprocess.run([sys.executable, "-m", "virtualenv", "-q", envd], capture_output=True).returncode == 0
    if made:
        py = f"{envd}/bin/python"
        for pkg in ["numpy scipy highspy"] + EXTRA_PIP:
            cmd = (["uv", "pip", "install", "-q", "--python", py] if shutil.which("uv") else [py, "-m", "pip", "install", "-q"]) + pkg.split()
            ok = subprocess.run(cmd, capture_output=True, text=True).returncode == 0
            print(("comparator ready    " if ok else "comparator skipped  ") + pkg.split()[0])
        os.environ["CMP_PYTHON"] = py
    else:
        print("comparator environment unavailable: only HiGHS will be compared")
"""

CUPDLPX = r"""
# ---- comparator: cuPDLPx (MIT) built from source on this GPU (recorded as unavailable if the build fails) ----
os.environ["CUPDLPX_BIN"] = ""
try:
    shutil.rmtree("/content/cuPDLPx", ignore_errors=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/MIT-Lu-Lab/cuPDLPx", "/content/cuPDLPx"],
                   check=True, timeout=300)
    subprocess.run(["cmake", "-S", "/content/cuPDLPx", "-B", "/content/cuPDLPx/build", "-DCMAKE_BUILD_TYPE=Release"],
                   check=True, capture_output=True, timeout=300)
    subprocess.run(["cmake", "--build", "/content/cuPDLPx/build", "-j", "8"], check=True, capture_output=True, timeout=900)
    exe = [p for p in glob.glob("/content/cuPDLPx/build/**/cupdlpx", recursive=True) if os.access(p, os.X_OK)]
    os.environ["CUPDLPX_BIN"] = exe[0] if exe else ""
except Exception as e:
    print("cuPDLPx build failed:", type(e).__name__)
print("cuPDLPx:", os.environ["CUPDLPX_BIN"] or "unavailable")
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

if SELF_TEST:
    t0 = time.time()
    shutil.rmtree("/tmp/selftest", ignore_errors=True)
    rc = stream([sys.executable, "-u", "{script}", "--run-dir", "/tmp/selftest", "--quick", "--only", "{selftest}"], 900)
    print(f"\n=== SELF-TEST {'PASSED' if rc == 0 else 'REPORTED A PROBLEM (see above); the long run continues'} in {time.time()-t0:.0f}s ===\n")

for part in PARTS.split(","):
    t0 = time.time()
    rc = stream([sys.executable, "-u", "{script}", "--run-dir", RUN_DIR, "--only", part], BUDGET.get(part))
    hit = time.time() - t0 >= BUDGET.get(part, 1e12) - 2
    print(f"--- {part}: {time.time()-t0:.0f}s" + ("  (budget reached: saved rows kept)" if hit else ""))
    save_results()
"""

SHOW1 = r"""
# ---- the slide numbers ----
from IPython.display import Markdown, Image, display
subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0,'bench'); import gpu_campaign as g; from pathlib import Path; g.e7_headline(Path('{RUN_DIR}'))"],
               capture_output=True)
for f in ("HEADLINE.md",):
    if os.path.exists(f"{RUN_DIR}/{f}"): display(Markdown(open(f"{RUN_DIR}/{f}").read()))
if os.path.exists(f"{RUN_DIR}/speedup.png"): display(Image(f"{RUN_DIR}/speedup.png"))
"""

SHOW2 = r"""
# ---- the decisions and the coverage tables ----
from IPython.display import Markdown, display
subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0,'bench'); import arch_campaign as a; a.a6_report('{RUN_DIR}')"],
               capture_output=True)
for f in ("PARITY.md", "ARCHITECTURE_DECISIONS.md", "WORLD.md"):
    if os.path.exists(f"{RUN_DIR}/{f}"): display(Markdown(open(f"{RUN_DIR}/{f}").read()))
"""

FINAL = r"""
# ---- final results zip (also downloaded after every part) ----
print("results zip:", save_results(final=True))
"""

NB1_TITLE = r"""
# QENIVO · A100 notebook 1 of 2 · **Evidence** (about 4 hours)

| Part | What it measures | Budget |
|---|---|---|
| e0 | environment, GPU, driver, versions; the full test suite on this machine | 15 min |
| e1 | our own CUDA sparse kernels vs the vendor library: agreement, bit-for-bit repeatability, GB/s | 15 min |
| e2 | large Mittelmann LPs at 1e-4 and 1e-6, repeated, host-checked; cuPDLPx on the same GPU; HiGHS on all cores | 60 min |
| e3 | refinery what-if batches (1 to 512 cases) vs HiGHS on all CPU cores; every case certified | 75 min |
| e4 | the open refinery-petrochemical benchmark (Du et al. 2025) vs BARON's published 18,000 s runs | 75 min |
| e5 | distributive recursion: warm first-order vs cold vs interior point | 30 min |
| e6 | learned warm starts: a network trained on GPU-solved cases; every answer still certified | 25 min |
| e7 | `HEADLINE.md` + `speedup.png` for the slides, from the recorded rows only | 1 min |
"""

NB2_TITLE = r"""
# QENIVO · A100 notebook 2 of 2 · **Architecture & World** (about 4.5 hours)

| Part | What it measures | Budget |
|---|---|---|
| a1 | preflight: download every benchmark set (retries + mirrors), check each file reads; manifest | 20 min |
| a2 | coverage through the full pipeline (route, solve, certify, independent verify): Netlib 98, Netlib infeasible 29, Kennington 16, Maros-Meszaros QP 138, large Mittelmann LPs; HiGHS alongside | 100 min |
| a3 | engine map: every engine on every LP up to 40k rows; router thresholds fitted from data | 50 min |
| a4 | PDHG design ablation on the GPU (restarts, Halpern, reflection, scaling, check interval, own vs vendor kernels) | 40 min |
| a5 | world: HiGHS (UK), OR-Tools GLOP + PDLP (Google), SCIP (ZIB Germany), cuPDLPx (MIT, GPU), NVIDIA cuOpt (GPU) | 60 min |
| a6 | `PARITY.md`, `ARCHITECTURE_DECISIONS.md`, `WORLD.md` | 1 min |
"""


def build(name, title, run, parts, script, selftest, show, extra_pip, budget):
    cells = [md(title + HOWTO),
             code(CONFIG.replace("{run}", run).replace("{parts}", parts).replace("{budget}", repr(budget))),
             code(SETUP.replace("{zip}", ZIP).replace("EXTRA_PIP", repr(extra_pip))),
             code(CUPDLPX),
             code(RUN.replace("{script}", script).replace("{selftest}", selftest)),
             code(show), code(FINAL)]
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"name": name, "gpuType": "A100", "machine_shape": "hm"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
          "cells": cells}
    (HERE / name).write_text(json.dumps(nb, indent=1))
    print("wrote", HERE / name)


build("a100_evidence.ipynb", NB1_TITLE, "evidence_run1", "e0,e1,e2,e3,e4,e5,e6",
      "bench/gpu_campaign.py", "e1,e2,e3,e5", SHOW1, [],
      {"e0": 900, "e1": 900, "e2": 3600, "e3": 4500, "e4": 4500, "e5": 1800, "e6": 1500})
build("a100_architecture.ipynb", NB2_TITLE, "architecture_run1", "a1,a2,a3,a4,a5",
      "bench/arch_campaign.py", "a1,a2,a3,a4,a5", SHOW2,
      ["ortools", "pyscipopt", "cuopt-cu12 --extra-index-url=https://pypi.nvidia.com"],
      {"a1": 1200, "a2": 6000, "a3": 3000, "a4": 2400, "a5": 3600})
