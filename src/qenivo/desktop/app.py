"""QENIVO desktop: the planner console in its own window, offline.

    qenivo desktop                 open the window (pywebview, Edge WebView2 on Windows)
    qenivo desktop --browser       use the default browser instead

The console server starts on a free loopback port with a fresh random token. The token travels in
the URL fragment, which browsers never send to a server or put in a Referer, and every API request
must carry it (AccessPolicy.loopback_requires_token), so other programs and other users on the
machine cannot drive the session. The server stops when the window closes. Without pywebview the
default browser is used and the console serves until Ctrl+C.

In an installed build (PyInstaller) the prebuilt native libraries ship inside the bundle; the
launcher points QENIVO_CACHE at them so no compiler is needed on the planner's machine.
"""
from __future__ import annotations

import os
import secrets
import sys
import threading
import webbrowser
from pathlib import Path

TITLE = "QENIVO — certified optimisation workstation"


def _use_bundled_native() -> None:
    """In a frozen build, find the native libraries that were built and shipped with it."""
    base = getattr(sys, "_MEIPASS", None)
    if base and not os.environ.get("QENIVO_CACHE"):
        bundled = Path(base) / "qenivo_native"
        if (bundled / "native").is_dir():
            os.environ["QENIVO_CACHE"] = str(bundled)


def start(port: int = 0, audit_dir: str | None = None):
    """Start the console in a background thread; return (server, url_with_token)."""
    from ..security import AccessPolicy
    from ..server import make_server
    token = secrets.token_urlsafe(32)
    policy = AccessPolicy(token=token, loopback_requires_token=True,
                          audit_dir=Path(audit_dir) if audit_dir else None)
    httpd = make_server("127.0.0.1", port, policy)
    threading.Thread(target=httpd.serve_forever, name="qenivo-console", daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_port}/#t={token}"


def run(port: int = 0, window: bool = True, audit_dir: str | None = None) -> int:
    _use_bundled_native()
    httpd, url = start(port, audit_dir)
    try:
        webview = None
        if window:
            try:
                import webview  # type: ignore[import-not-found]
            except ImportError:
                print("pywebview is not installed; opening the default browser instead "
                      "(pip install pywebview for the desktop window).")
        if webview is not None:
            webview.create_window(TITLE, url, width=1600, height=900, min_size=(1100, 700))
            webview.start()                       # returns when the window is closed
            return 0
        print(f"QENIVO console on http://127.0.0.1:{httpd.server_port} (local only). Ctrl+C to stop.")
        webbrowser.open(url)
        threading.Event().wait()                  # serve until interrupted
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        httpd.shutdown()
        httpd.server_close()


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="qenivo desktop", description="QENIVO planner console in its own window")
    ap.add_argument("--browser", action="store_true", help="use the default browser instead of a window")
    ap.add_argument("--port", type=int, default=0, help="loopback port (default: a free one)")
    ap.add_argument("--audit-dir", help="write the request audit trail here (one JSON line per request)")
    a = ap.parse_args(argv)
    return run(a.port, window=not a.browser, audit_dir=a.audit_dir)


if __name__ == "__main__":
    sys.exit(main())
