"""Edition tag for results honesty (public vs pro).

Every results JSON / JSONL row (or its file-level ``meta``) must record
``edition: "public" | "pro"``. Public docs must not claim numbers that require
the Pro plugins. See ``docs/EDITIONS.md``.
"""
from __future__ import annotations

from typing import Literal

Edition = Literal["public", "pro"]


def detect_edition() -> Edition:
    """Return ``"pro"`` if any ``qenivo_pro`` entry point is installed, else ``"public"``.

    Detection does not import private modules: it only inspects packaging metadata
    so the public core never depends on Pro code.
    """
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        group = eps.select(group="qenivo_pro") if hasattr(eps, "select") else eps.get("qenivo_pro", [])
        if list(group):
            return "pro"
    except Exception:  # noqa: BLE001 — missing metadata must not break solves
        pass
    return "public"


def edition_meta() -> dict:
    """Small dict to merge into results ``meta`` blocks."""
    return {"edition": detect_edition()}
