"""Build qenivo.zip: the project as uploaded to Colab or Kaggle (one top folder 'qenivo/').

    python packaging/make_upload_zip.py [output.zip]

Excluded: git data, caches, build output, downloaded benchmark instances and local results (the
notebooks download the benchmarks from their official sources). The zip is checked after writing:
it must contain src/qenivo/__init__.py and both campaign scripts.
"""
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "runs", "dist", "__pycache__", ".pytest_cache", ".venv", "build"}
SKIP_PATHS = {ROOT / "bench" / "instances", ROOT / "bench" / "results"}


def main(out=None):
    out = Path(out or ROOT.parent / "qenivo.zip")
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d, dirs, files in os.walk(ROOT):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.endswith(".egg-info")
                       and Path(d, x) not in SKIP_PATHS]
            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    continue
                p = Path(d, f)
                z.write(p, "qenivo/" + p.relative_to(ROOT).as_posix())
                n += 1
    names = set(zipfile.ZipFile(out).namelist())
    for need in ("qenivo/src/qenivo/__init__.py", "qenivo/bench/arch_campaign.py", "qenivo/bench/gpu_campaign.py"):
        assert need in names, f"{need} missing from {out}"
    print(f"{out}: {n} files, {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
