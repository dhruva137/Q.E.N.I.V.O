"""Open-core split stays importable: the public archive must never ship a broken `import qenivo`.

Two lists decide what leaves the repo. `.gitattributes` (export-ignore) drives `git archive`, which is
also what GitHub's "Download ZIP" runs; `packaging/private_paths.py` drives `export_public.py`. When
they drifted, the ZIP stripped `kernels/byte_diet.py` (meant to stay public) and shipped
`workload/crude_value.py` without its private dependencies, so pdhg/pdqp/refine failed to import and
seven test files failed to collect. These checks are static and offline.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT / "packaging"))

from private_paths import PRIVATE_PATHS, is_private_rel  # noqa: E402


def _export_ignored() -> set[str]:
    out = set()
    for line in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "export-ignore" in line.split()[1:]:
            out.add(line.split()[0].rstrip("/"))
    return out


def test_gitattributes_matches_private_paths():
    """Every private path is stripped from `git archive`, and nothing else is."""
    attrs, priv = _export_ignored(), {p.rstrip("/") for p in PRIVATE_PATHS}
    assert not attrs - priv, f"export-ignored but public in private_paths.py: {sorted(attrs - priv)}"
    assert not priv - attrs, f"private in private_paths.py but shipped by git archive: {sorted(priv - attrs)}"


def _module_rel(mod: str) -> list[str]:
    """Repo-relative candidates for a dotted module: the .py file and the package directory."""
    base = "src/" + mod.replace(".", "/")
    return [base + ".py", base]


def _resolve(node: ast.ImportFrom | ast.Import, pkg: str) -> list[str]:
    """Absolute module names an import statement loads, relative imports resolved against `pkg`."""
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if node.level:
        parts = pkg.split(".")
        anchor = ".".join(parts[: len(parts) - node.level + 1])
        base = f"{anchor}.{node.module}" if node.module else anchor
    else:
        base = node.module or ""
    # `from pkg import name` may load the submodule pkg.name
    return [base] + [f"{base}.{a.name}" for a in node.names if a.name != "*"]


def _guarded(node: ast.AST, parents: dict) -> bool:
    """Inside a `try:` body whose handlers catch ImportError (or a superclass)."""
    child, cur = node, parents.get(node)
    while cur is not None:
        if isinstance(cur, ast.Try) and child in cur.body:
            for h in cur.handlers:
                names = [h.type] if not isinstance(h.type, ast.Tuple) else list(h.type.elts)
                if h.type is None or any(isinstance(n, ast.Name) and n.id in
                                         ("ImportError", "ModuleNotFoundError", "Exception", "BaseException")
                                         for n in names):
                    return True
        child, cur = cur, parents.get(cur)
    return False


def test_public_code_never_hard_imports_private_modules():
    """A public module may import a private one only under `except ImportError` (absolute or relative)."""
    bad = []
    for path in sorted((SRC / "qenivo").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if is_private_rel(rel):
            continue
        pkg = ".".join(path.relative_to(SRC).parent.parts)   # package that holds this file
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for target in _resolve(node, pkg):
                if any(is_private_rel(c) for c in _module_rel(target)) and not _guarded(node, parents):
                    bad.append(f"{rel}:{node.lineno} imports private {target}")
    assert not bad, "public code hard-imports private modules:\n  " + "\n  ".join(bad)


def test_guard_detector_catches_relative_imports():
    """The checker itself: an unguarded relative import of a private module is flagged, a guarded one is not."""
    src = ("from .workload.crude_value import crude_value\n"
           "try:\n    from .workload.parametric import x\nexcept ImportError:\n    x = None\n")
    tree = ast.parse(src)
    parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    hits = [(n.lineno, _guarded(n, parents)) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    targets = [t for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for t in _resolve(n, "qenivo")]
    assert "qenivo.workload.crude_value" in targets and "qenivo.workload.parametric" in targets
    assert sorted(hits) == [(1, False), (3, True)]


def test_tests_and_benches_of_private_modules_are_private():
    """A test or bench that imports a private module must not ship in the public export either,
    or the export carries a file that cannot even be collected."""
    bad = []
    for top in ("tests", "bench"):
        for path in sorted((ROOT / top).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if is_private_rel(rel):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) or (isinstance(node, ast.ImportFrom) and not node.level):
                    for target in _resolve(node, ""):
                        private = target.startswith("qenivo") and any(is_private_rel(c) for c in _module_rel(target))
                        if private and not _guarded(node, parents):
                            bad.append(f"{rel}:{node.lineno} imports private {target}")
    assert not bad, "public tests/benches import private modules:\n  " + "\n  ".join(bad)
