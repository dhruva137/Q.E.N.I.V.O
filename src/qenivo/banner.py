"""Opening mark for a bare ``qenivo`` launch.

Shown only when the process is attached to a terminal. Pipes, ``NO_COLOR``, and
``QENIVO_NO_BANNER`` get the same command card with no animation and no colour.
"""
from __future__ import annotations

import math
import os
import sys
import time

_GAP = 2
# '#' is ink. Every glyph is 6 rows by 7 columns.
_GLYPHS = {
    "Q": (
        " ##### ",
        "#     #",
        "#     #",
        "#  #  #",
        " ##### ",
        "     # ",
    ),
    "E": (
        "#######",
        "#      ",
        "#####  ",
        "#      ",
        "#      ",
        "#######",
    ),
    "N": (
        "##   ##",
        "###  ##",
        "#### ##",
        "## ####",
        "##  ###",
        "##   ##",
    ),
    "I": (
        "#######",
        "   #   ",
        "   #   ",
        "   #   ",
        "   #   ",
        "#######",
    ),
    "V": (
        "#     #",
        "#     #",
        "#     #",
        " #   # ",
        "  # #  ",
        "   #   ",
    ),
    "O": (
        " ##### ",
        "#     #",
        "#     #",
        "#     #",
        "#     #",
        " ##### ",
    ),
}

_STOPS = (
    (0.00, (186, 132, 72)),
    (0.38, (244, 228, 196)),
    (0.68, (118, 196, 188)),
    (1.00, (28, 108, 106)),
)

_COMMANDS = (
    ("solve", "an MPS model, with a certificate"),
    ("verify", "check that certificate, float or exact"),
    ("explain", "why the plan cannot be met"),
    ("range", "how far a price or a limit can move"),
    ("crude-value", "the cargo price curve"),
    ("cases", "a stack of what-ifs"),
    ("serve", "the planner console"),
    ("info", "version, GPU, provenance"),
)

_TAG = "every answer certified"


def word_rows() -> list[str]:
    """Six rows of the QENIVO wordmark. '#' marks ink."""
    rows = [""] * 6
    for i, ch in enumerate("QENIVO"):
        glyph = _GLYPHS[ch]
        gap = " " * _GAP if i else ""
        for r in range(6):
            rows[r] += gap + glyph[r]
    return rows


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("qenivo")
    except Exception:
        return "0.1.0"


def _gradient(u: float) -> tuple[int, int, int]:
    u = min(1.0, max(0.0, u))
    for (u0, c0), (u1, c1) in zip(_STOPS, _STOPS[1:]):
        if u <= u1:
            span = u1 - u0
            t = 0.0 if span <= 0 else (u - u0) / span
            return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))
    return _STOPS[-1][1]


def _shade(rgb: tuple[int, int, int], u: float, phase: float | None) -> tuple[int, int, int]:
    if phase is None:
        return rgb
    w = math.exp(-((u - phase) ** 2) / 0.006)
    return tuple(min(255, int(c + (255 - c) * 0.78 * w)) for c in rgb)


def _supports(stream, char: str) -> bool:
    enc = getattr(stream, "encoding", None) or "utf-8"
    try:
        char.encode(enc)
        return True
    except Exception:
        return False


def render_mark(rows: list[str], *, reveal: int, phase: float | None, ink: str, color: bool) -> list[str]:
    """One frame. ``reveal`` is how many columns of ink are visible."""
    width = len(rows[0])
    out = []
    for row in rows:
        parts: list[str] = []
        open_color = None
        for x, ch in enumerate(row):
            if ch != "#" or x >= reveal:
                if open_color:
                    parts.append("\x1b[0m")
                    open_color = None
                parts.append(" ")
                continue
            if not color:
                parts.append(ink)
                continue
            rgb = _shade(_gradient(x / max(width - 1, 1)), x / max(width - 1, 1), phase)
            code = f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
            if code != open_color:
                parts.append(code)
                open_color = code
            parts.append(ink)
        if color and open_color:
            parts.append("\x1b[0m")
        line = "  " + "".join(parts)
        if color:
            line += "\x1b[K"
        out.append(line)
    return out


def _enable_windows_vt() -> bool:
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        kernel = ctypes.windll.kernel32
        handle = kernel.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        if not kernel.SetConsoleMode(handle, mode.value | 0x0004):
            return False
        kernel.SetConsoleOutputCP(65001)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        return True
    except Exception:
        return False


def _animate(stream, rows: list[str]) -> bool:
    """Play the mark. False means this stream cannot host it."""
    flag = os.environ.get("QENIVO_BANNER", "").strip().lower()
    if flag in {"0", "off", "plain"} or os.environ.get("QENIVO_NO_BANNER"):
        return False
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    forced = flag == "animate"
    is_tty = bool(getattr(stream, "isatty", lambda: False)())
    if not forced and not is_tty:
        return False
    if sys.platform == "win32" and stream is sys.stdout and is_tty and not _enable_windows_vt():
        return False
    ink = "█" if _supports(stream, "█") else "#"
    color = ink == "█"
    width = len(rows[0])
    hide = "\x1b[?25l" if color else ""
    show = "\x1b[?25h" if color else ""

    def paint(reveal: int, phase: float | None) -> None:
        frame = render_mark(rows, reveal=reveal, phase=phase, ink=ink, color=color)
        stream.write("\n".join(frame) + "\n")
        stream.flush()

    stream.write("\n" + hide)
    try:
        paint(0, None)
        step = 2
        for cols in range(step, width + step, step):
            stream.write(f"\x1b[{len(rows)}A")
            paint(min(cols, width), None)
            time.sleep(0.012)
        sweeps = 14
        for i in range(sweeps):
            phase = -0.12 + (1.24 * i / (sweeps - 1))
            stream.write(f"\x1b[{len(rows)}A")
            paint(width, phase)
            time.sleep(0.028)
        stream.write(f"\x1b[{len(rows)}A")
        paint(width, None)
    finally:
        if show:
            stream.write(show)
            stream.flush()
    return True


def _card(version: str, *, color: bool) -> str:
    name = "\x1b[1m\x1b[38;2;244;228;196m" if color else ""
    dim = "\x1b[38;2;128;148;146m" if color else ""
    cmd = "\x1b[38;2;236;226;206m" if color else ""
    off = "\x1b[0m" if color else ""
    lines = [
        f"  {name}qenivo {version}{off}",
        f"  {dim}{_TAG}{off}",
        "",
    ]
    width = max(len(c) for c, _ in _COMMANDS)
    for command, blurb in _COMMANDS:
        lines.append(f"  {cmd}{command:<{width}}{off}  {dim}{blurb}{off}")
    lines.append("")
    lines.append(f"  {dim}qenivo <command> --help{off}")
    lines.append("")
    return "\n".join(lines)


def _write(stream, text: str) -> None:
    enc = getattr(stream, "encoding", None) or "utf-8"
    try:
        text.encode(enc)
    except Exception:
        text = text.encode(enc, errors="replace").decode(enc, errors="replace")
    stream.write(text)
    stream.flush()


def launch(stream=None) -> None:
    """Paint the wordmark when a terminal is watching, then the command card."""
    stream = stream or sys.stdout
    version = _version()
    rows = word_rows()
    played = False
    try:
        played = _animate(stream, rows)
    except Exception:
        played = False
    color = played and _supports(stream, "█")
    _write(stream, _card(version, color=color))
