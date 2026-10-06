"""Canonical private-path and import-marker lists for the open-core split.

Shared by ``export_public.py`` and ``build_pro.py``. Paths are relative to the
repository root, POSIX-style. A path that does not exist yet is still listed so
export stays honest when those workstreams land.
"""
from __future__ import annotations

# Directory or file paths excluded from the public tree.
PRIVATE_PATHS: tuple[str, ...] = (
    # Separate Pro package (preferred layout when split into its own repo/wheel).
    "src/qenivo_pro",
    "pro",
    # W02 — parametric crude-valuation engine and breakpoint reports.
    "src/qenivo/workload/parametric.py",
    "src/qenivo/workload/breakpoints.py",
    "src/qenivo/workload/crude_value.py",
    "tests/test_parametric.py",
    "tests/test_breakpoints.py",
    "tests/test_crude_value.py",
    "src/qenivo/engines/parametric_simplex.py",
    # W04 byte-diet helpers stay public: generic layout helpers, imported by engines/pdhg.py.
    # W05 — Stack Simplex (prototype then engine).
    "research_prototypes/stack_simplex",
    "src/qenivo/engines/stack_simplex.py",
    # W06 — mixed-precision backbone prototype / engine.
    "research_prototypes/mixed_precision",
    "src/qenivo/engines/mixed_precision.py",
    "src/qenivo/native/mixed_precision",
    # W03 — proprietary calibrated router thresholds (public keeps published defaults).
    "src/qenivo/workload/router_thresholds_pro.py",
    "src/qenivo/workload/router_policy_pro.json",
    # MRPL-shaped models and data (not the public Williams / industry generators).
    "src/qenivo/models/mrpl",
    "data/mrpl",
    # X5 public-data twin: derived from ExxonMobil assays whose site terms restrict copying.
    "data/twin",
    "tests/test_mrpl_twin.py",
    "bench/twin_replay.py",
    "bench/crude_value_bench.py",       # W02 bench imports the private crude-value engine
    "bench/results/crude_value*",       # W02 curve results stay with the private engine
    "bench/results/twin_replay*",       # X5 twin replay stays with the private model
    "bench/instances/mrpl",
    # Internal manuals, audits, deck sources, worklogs, and unfinished work
    # stay out of a public export.
    "wip",
)

# Substrings that must not appear as imports (or as package references) in the
# public tree after export. Matched case-sensitively against file text.
PRIVATE_IMPORT_MARKERS: tuple[str, ...] = (
    "qenivo_pro",
    "from qenivo.workload.parametric",
    "import qenivo.workload.parametric",
    "from qenivo.workload.breakpoints",
    "import qenivo.workload.breakpoints",
    "from qenivo.workload.crude_value",
    "from qenivo.engines.stack_simplex",
    "import qenivo.engines.stack_simplex",
    "from qenivo.engines.mixed_precision",
    "import qenivo.engines.mixed_precision",
    "from qenivo.engines.parametric_simplex",
    "import qenivo.engines.parametric_simplex",
    "from qenivo.workload.router_thresholds_pro",
    "import qenivo.workload.router_thresholds_pro",
    "from qenivo.models.mrpl",
    "import qenivo.models.mrpl",
    "research_prototypes.stack_simplex",
    "research_prototypes.mixed_precision",
)

# Filename / dirname tokens treated as private when scanning the tree.
PRIVATE_NAME_TOKENS: tuple[str, ...] = (
    "qenivo_pro",
    "stack_simplex",
    "mixed_precision",
    "parametric.py",
    "breakpoints.py",
    "parametric_simplex",
    "router_thresholds_pro",
    "router_policy_pro",
)


def is_private_rel(rel: str) -> bool:
    """True if ``rel`` (POSIX, relative to repo root) is under a private path.

    A private path ending in ``*`` matches that prefix (``bench/results/crude_value*``).
    """
    rel = rel.replace("\\", "/").lstrip("./")
    for p in PRIVATE_PATHS:
        p = p.rstrip("/")
        if p.endswith("*"):
            if rel.startswith(p[:-1]):
                return True
        elif rel == p or rel.startswith(p + "/"):
            return True
    return False
