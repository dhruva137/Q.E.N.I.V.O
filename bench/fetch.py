"""Download, decompress and cache benchmark instances as plain MPS.

    python -m bench.fetch afiro qap15          # by name
    python bench/fetch.py --set netlib

Sources were checked against the live directory listings on 2026-09-27:
  Netlib LP        coin-or-tools/Data-Netlib (plain MPS, gzip)
  Kennington       netlib.org/lp/data/kennington (emps, gzip)
  Mittelmann LP    plato.asu.edu/ftp/lptestset (bzip2; files without .mps are emps)
Payloads in netlib's compressed format are expanded by bench.emps (checksummed port of emps.c).
"""
from __future__ import annotations

import bz2
import gzip
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

from emps import decode_emps, is_emps

NETLIB = "https://raw.githubusercontent.com/coin-or-tools/Data-Netlib/master/{}.mps.gz"
NETLIB_EMPS = "https://www.netlib.org/lp/data/{}"
KENN = "https://www.netlib.org/lp/data/kennington/{}"
PLATO = "https://plato.asu.edu/ftp/lptestset/"

MITTELMANN = {
    "Linf_520c": "Linf_520c.bz2", "L1_sixm250obs": "L1_sixm250obs.bz2", "bdry2": "bdry2.bz2",
    "a2864": "a2864.mps.bz2", "dlr1": "dlr1.mps.bz2", "savsched1": "savsched1.mps.bz2",
    "rmine15": "rmine15.mps.bz2", "square41": "square41.mps.bz2", "s82": "s82.mps.bz2",
    "s100": "s100.mps.bz2", "s250r10": "s250r10.mps.bz2", "ex10": "ex10.mps.bz2",
    "irish-e": "irish-electricity.mps.bz2", "datt256": "datt256_lp.mps.bz2",
    "Dual2_5000": "Dual2_5000.mps.bz2", "Primal2_1000": "Primal2_1000.mps.bz2",
    "thk_63": "thk_63.mps.bz2", "L2CTA3D": "L2CTA3D.mps.bz2", "woodlands09": "woodlands09.mps.bz2",
    "set-cover": "set-cover-model.mps.bz2", "scpm1": "scpm1.mps.bz2",
    "pds-100": "pds/pds-100.bz2", "rail4284": "rail/rail4284.bz2", "rail02": "rail/rail582.bz2",
    "nug08-3rd": "nug/nug08-3rd.bz2", "fome13": "fome/fome13.bz2",
    "cont1": "misc/cont1.bz2", "cont11": "misc/cont11.bz2", "neos3": "misc/neos3.bz2",
    "ns1687037": "misc/ns1687037.bz2", "ns1688926": "misc/ns1688926.bz2",
    "storm_1000": "misc/stormG2_1000.bz2", "qap15": "qap15.mps.bz2",
}
KENNINGTON = {"ken-07", "ken-11", "ken-13", "ken-18", "osa-07", "osa-14", "osa-30", "osa-60",
              "pds-02", "pds-06", "pds-10", "pds-20", "cre-a", "cre-b", "cre-c", "cre-d"}
# qap15 is not in the coin-or netlib mirror; the plato copy is used.


MAROS_URL = "https://raw.githubusercontent.com/YimingYAN/QP-Test-Problems/master/QPS_Files/{}.QPS"
MAROS_README = "https://raw.githubusercontent.com/YimingYAN/QP-Test-Problems/master/QPS_Files/00README.QP"


def _is_maros(name: str) -> bool:
    from sets import MAROS
    return name in MAROS


def maros_references() -> dict:
    """Published optimal objective values of the Maros-Meszaros set (from its 00README.QP)."""
    import re
    cache = Path(__file__).parent / "instances" / "maros_00README.QP"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        _download(MAROS_README, cache)
    ref = {}
    for line in cache.read_text(errors="replace").splitlines():
        m = re.match(r"^(\S+)\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+([-+]?\d\.\d+[eE][+-]\d+)", line)
        if m:
            ref[m.group(1).lower().replace("_", "").replace("-", "")] = float(m.group(2))
    return ref


def maros_key(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def source_url(name: str) -> str:
    if name.endswith("-infeas"):              # infeasible variant that shares a feasible model's name
        return f"https://www.netlib.org/lp/infeas/{name[:-len('-infeas')].lower()}"
    if _is_maros(name):
        return MAROS_URL.format(name)
    if name in MITTELMANN:
        return PLATO + MITTELMANN[name]
    if name in KENNINGTON:
        return KENN.format(name)
    return NETLIB.format(name.lower())


def _download(url: str, dest: Path, retries: int = 3, timeout: int = 120) -> None:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _decompress(src: Path, dest: Path) -> None:
    with open(src, "rb") as f:
        magic = f.read(3)
    if magic[:2] == b"\x1f\x8b":
        opener = gzip.open
    elif magic == b"BZh":
        opener = bz2.open
    else:
        shutil.copyfile(src, dest)
        return
    with opener(src, "rb") as fi, open(dest, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)


def fetch(name: str, dest_dir: str | os.PathLike = Path(__file__).parent / "instances", verbose: bool = True) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{name}.mps"
    if out.exists() and out.stat().st_size > 0:
        return out
    if name.startswith("refinery_L"):          # generated: refinery_L<level>[_s<seed>]
        from qenivo.io.mps import write_mps
        from qenivo.models.refinery import refinery_level
        parts = name[len("refinery_L"):].split("_s")
        prob = refinery_level(int(parts[0]), seed=int(parts[1]) if len(parts) > 1 else 0)
        prob.row_names = [f"r{i}" for i in range(prob.m)]
        prob.col_names = [f"x{j}" for j in range(prob.n)]
        tmp = dest_dir / f"{name}.mps.tmp"
        write_mps(prob, tmp, name=name)
        os.replace(tmp, out)
        if verbose:
            print(f"generated {name}: {prob.size_str()}")
        return out
    url = source_url(name)
    t0 = time.time()
    raw, plain = dest_dir / f"{name}.download", dest_dir / f"{name}.raw"
    try:
        _download(url, raw)
    except Exception:
        if url != NETLIB.format(name.lower()):
            raise
        try:
            _download(NETLIB_EMPS.format(name.lower()), raw)     # not in the coin-or mirror
        except Exception:
            try:
                _download(f"https://www.netlib.org/lp/infeas/{name.lower()}", raw)
            except Exception:
                _download(f"https://miplib.zib.de/WebData/instances/{name}.mps.gz", raw)
    _decompress(raw, plain)
    raw.unlink()
    with open(plain, "r", encoding="latin-1") as f:
        head = f.read(4096)
    if is_emps(head):
        text = plain.read_text(encoding="latin-1")
        tmp = dest_dir / f"{name}.mps.tmp"
        tmp.write_text(decode_emps(text), encoding="latin-1")
        plain.unlink()
        os.replace(tmp, out)
        kind = "emps"
    else:
        os.replace(plain, out)
        kind = "mps"
    if verbose:
        print(f"fetched {name} ({kind}, {out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")
    return out


def load_set(set_name: str) -> list[str]:
    from sets import SETS
    return list(SETS[set_name])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    names = load_set(argv[1]) if argv[:1] == ["--set"] else argv
    failed = []
    for n in names:
        try:
            fetch(n)
        except Exception as e:
            print(f"FAILED {n}: {e!r}")
            failed.append(n)
    if failed:
        sys.exit(f"failed: {failed}")


if __name__ == "__main__":
    main()
