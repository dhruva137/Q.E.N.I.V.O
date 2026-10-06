"""Build the air-gapped installation bundle: wheel + every dependency wheel + checksums + SBOM.

    python packaging/build_offline_bundle.py            -> dist/qenivo-offline-<version>-<platform>/
Install on a machine with NO internet:
    pip install --no-index --find-links qenivo-offline-<...>/wheels qenivo
Verify integrity first:
    python qenivo-offline-<...>/verify_bundle.py        (checks every SHA-256 in SHA256SUMS)

The SBOM (sbom.spdx.json, SPDX 2.3) lists every package in the bundle with its version and
SHA-256, so a security team can check exactly what is being installed, and that no optimisation
solver is among them.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qenivo import __version__  # noqa: E402
from qenivo.provenance import FORBIDDEN_PACKAGES  # noqa: E402

VERIFY = '''"""Check every file of this bundle against SHA256SUMS."""
import hashlib, pathlib, sys
here = pathlib.Path(__file__).parent
bad = 0
for line in (here / "SHA256SUMS").read_text().splitlines():
    digest, name = line.split(maxsplit=1)
    ok = hashlib.sha256((here / name).read_bytes()).hexdigest() == digest
    bad += not ok
    print(("OK   " if ok else "FAIL ") + name)
sys.exit(1 if bad else 0)
'''


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    tag = f"{platform.system().lower()}-{platform.machine().lower()}-py{sys.version_info.major}{sys.version_info.minor}"
    out = ROOT / "dist" / f"qenivo-offline-{__version__}-{tag}"
    wheels = out / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "wheel", str(ROOT), "-w", str(wheels), "--quiet"], check=True)
    pkgs = []
    for w in sorted(wheels.glob("*.whl")):
        name, ver = w.name.split("-")[:2]
        with zipfile.ZipFile(w) as z:
            lic = next((n for n in z.namelist() if n.upper().endswith(("LICENSE", "LICENSE.TXT", "COPYING"))), None)
        pkgs.append({"name": name, "version": ver, "file": f"wheels/{w.name}", "sha256": sha256(w),
                     "license_file": lic})
    forbidden = [p for p in pkgs if p["name"].lower().replace("_", "-") in {f.split(".")[0] for f in FORBIDDEN_PACKAGES}]
    if forbidden:
        sys.exit(f"refusing to build: solver packages in the bundle: {forbidden}")
    sbom = {"spdxVersion": "SPDX-2.3", "dataLicense": "CC0-1.0", "SPDXID": "SPDXRef-DOCUMENT",
            "name": f"qenivo-offline-{__version__}-{tag}",
            "documentNamespace": f"https://qenivo.local/spdx/{__version__}/{tag}",
            "creationInfo": {"creators": ["Tool: packaging/build_offline_bundle.py"]},
            "packages": [{"SPDXID": f"SPDXRef-{i}", "name": p["name"], "versionInfo": p["version"],
                          "downloadLocation": "NOASSERTION", "filesAnalyzed": False,
                          "checksums": [{"algorithm": "SHA256", "checksumValue": p["sha256"]}]}
                         for i, p in enumerate(pkgs)]}
    (out / "sbom.spdx.json").write_text(json.dumps(sbom, indent=1))
    (out / "verify_bundle.py").write_text(VERIFY)
    (out / "INSTALL.txt").write_text(
        "Air-gapped installation\n=======================\n"
        "1. python verify_bundle.py                         (every checksum must be OK)\n"
        "2. pip install --no-index --find-links wheels qenivo\n"
        "3. qenivo info                                      (provenance guard must read CLEAN)\n"
        "4. qenivo demo                                      (two-minute tour, no network needed)\n"
        "5. qenivo serve                                     (planner console on http://127.0.0.1:8765)\n")
    files = [f for f in sorted(out.rglob("*")) if f.is_file() and f.name != "SHA256SUMS"]
    (out / "SHA256SUMS").write_text("".join(f"{sha256(f)}  {f.relative_to(out).as_posix()}\n" for f in files))
    print(f"bundle: {out}  ({len(pkgs)} wheels)")
    for p in pkgs:
        print(f"  {p['name']:12s} {p['version']:10s} {p['sha256'][:16]}")


if __name__ == "__main__":
    main()
