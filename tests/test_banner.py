"""Bare `qenivo` opens with a command card, and the wordmark stays encodable."""
from __future__ import annotations

import io

from qenivo.banner import render_mark, word_rows
from qenivo.cli import main


def test_wordmark_spells_qenivo_on_one_grid():
    rows = word_rows()
    assert len(rows) == 6
    assert len({len(r) for r in rows}) == 1
    assert all(set(r) <= set("# ") for r in rows)
    # Top row keeps the six letterforms in order: Q's bowl, then E, N, I, V, O.
    assert rows[0].count("#") > 20


def test_bare_launch_is_a_plain_card_when_not_a_terminal(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "qenivo 0.1.0" in out
    assert "every answer certified" in out
    for name in ("solve", "verify", "explain", "range", "crude-value", "cases", "serve", "info"):
        assert name in out
    assert "\x1b" not in out


def test_coloured_frame_uses_blocks_and_resets():
    frame = render_mark(word_rows(), reveal=10_000, phase=None, ink="█", color=True)
    assert len(frame) == 6
    assert all(line.startswith("  ") for line in frame)
    assert "█" in frame[0]
    assert "\x1b[0m" in frame[0]
    assert "\x1b[38;2;" in frame[0]


def test_animation_finishes_and_restores_the_cursor(monkeypatch):
    monkeypatch.setenv("QENIVO_BANNER", "animate")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("qenivo.banner.time.sleep", lambda _s: None)

    class Tty(io.StringIO):
        encoding = "utf-8"

        def isatty(self):
            return True

    buf = Tty()
    from qenivo.banner import launch

    launch(buf)
    text = buf.getvalue()
    assert "█" in text
    assert "\x1b[?25h" in text
    assert "every answer certified" in text
    assert "qenivo <command> --help" in text
