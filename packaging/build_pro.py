"""Outline: build the private ``qenivo-pro`` wheel (not for the public repo).

This script is a stub. The Pro wheel ships only the competitive modules listed
in ``private_paths.PRIVATE_PATHS`` / ``docs/EDITIONS.md``, registers them through
the ``qenivo_pro`` entry-point group, and never vendors Apache-2.0 third-party
solver code into those modules (R2).

Intended layout (separate private repository or ``src/qenivo_pro/``)::

    src/qenivo_pro/
      __init__.py          # register() loads engines / policies
      crude_valuation.py   # W02
      byte_diet.py         # W04 kernels
      stack_simplex.py     # W05
      mixed_precision.py   # W06
      router_thresholds.py # W03 calibrated policy
      mrpl/                # MRPL-shaped models + data loaders

    pyproject.toml
      [project]
      name = "qenivo-pro"
      # license: all rights reserved (not Apache-2.0)
      dependencies = ["qenivo>=0.1"]

      [project.entry-points."qenivo_pro"]
      register = "qenivo_pro:register"

``register()`` must call only ``qenivo.register_engine(...)`` (and optional
policy hooks). Public ``qenivo`` code must never ``import qenivo_pro``.

Build steps (when the private tree exists)::

    1. Assert each private module carries an "All rights reserved" header.
    2. Assert no private file matches forbidden R1/R2 solver origins.
    3. python -m build --wheel
    4. Install wheel alongside public qenivo; ``qenivo.edition.detect_edition()``
       returns ``"pro"``; short smoke tests for registered engines.

Until those modules exist, this script exits with a clear status.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_paths import PRIVATE_PATHS  # noqa: E402


def main(argv=None) -> int:
    del argv  # outline only
    present = [p for p in PRIVATE_PATHS if (ROOT / p).exists()]
    print("qenivo-pro build outline")
    print(f"  repo root: {ROOT}")
    print(f"  private paths listed: {len(PRIVATE_PATHS)}")
    print(f"  present in this worktree: {len(present)}")
    for p in present:
        print(f"    - {p}")
    if not present:
        print("  no private modules checked in yet; wheel build deferred")
        print("  when ready: implement register() + `python -m build` as above")
        return 0
    print("  TODO: package present private modules into qenivo-pro wheel")
    print("  TODO: verify All-rights-reserved headers; run Pro smoke tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
