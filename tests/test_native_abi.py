"""Native libraries are keyed on source, OS family and CPU, not on the OS release.

With the release in the key, a library built on one Windows machine was never found on another,
so an installer's prebuilt libraries were silently ignored and the Python engines ran instead.
"""
import platform
from pathlib import Path

from qenivo.engines import _native_abi

SRC = Path(__file__).resolve().parents[1] / "src" / "qenivo"
LOADERS = ["engines/native_simplex.py", "engines/native_ipm.py", "engines/exact_lp.py", "io/native_mps.py"]


def test_tag_ignores_the_os_release(monkeypatch):
    tag = _native_abi.native_abi_tag()
    monkeypatch.setattr(platform, "platform", lambda *a, **k: "Windows-10-10.0.19045-SP0")
    monkeypatch.setattr(platform, "release", lambda: "10")
    monkeypatch.setattr(platform, "version", lambda: "10.0.19045")
    assert _native_abi.native_abi_tag() == tag


def test_tag_still_separates_os_and_cpu(monkeypatch):
    tag = _native_abi.native_abi_tag()
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")
    assert _native_abi.native_abi_tag() != tag


def test_every_loader_uses_the_tag():
    for rel in LOADERS:
        text = (SRC / rel).read_text(encoding="utf-8")
        assert "native_abi_tag()" in text and "platform.platform()" not in text, rel


def test_a_prebuilt_library_loads_without_a_compiler(monkeypatch):
    """The installer ships built libraries; the planner's machine has no compiler and must still load them.

    The loaders used to look for a compiler before looking in the cache, so they never loaded a
    library they had not just built.
    """
    import importlib

    import pytest
    loaders = [importlib.import_module(m) for m in ("qenivo.engines.native_simplex", "qenivo.engines.native_ipm",
                                                     "qenivo.engines.exact_lp", "qenivo.io.native_mps")]
    if any(m.library() is None for m in loaders):
        pytest.skip("native libraries are not built on this machine")
    for m in loaders:
        saved = dict(m._LIB)
        m._LIB.clear()
        monkeypatch.setattr(m, "_compiler", lambda: None)
        try:
            assert m.library() is not None, (m.__name__, m._LIB.get("reason"))
        finally:
            m._LIB.clear()
            m._LIB.update(saved)
