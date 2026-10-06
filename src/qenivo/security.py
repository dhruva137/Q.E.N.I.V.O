"""Access control and audit trail for `qenivo serve` in a plant or head-office network.

Designed for how Indian refinery IT runs (docs/GROUND_REALITY.md): planning desktops on the
corporate network, OT isolated from the internet, sign-in through the company directory, and
CERT-In's 2022 direction to keep logs for 180 days.

  * local use (the default)   binds to 127.0.0.1; the signed-in Windows / Linux user is recorded
  * shared server             `--token-file`: every request must carry  Authorization: Bearer <token>
  * company sign-in (SSO)     put the server behind the site's reverse proxy (IIS with Windows
                              Authentication, or Apache/Nginx with Kerberos); with
                              `--proxy-user-header X-Remote-User --trusted-proxy 10.0.0.5` the user
                              name the proxy authenticated is taken from that header, and only
                              from that proxy's address. No password ever reaches this process.
  * audit                     one JSON line per request (who, from where, which model by SHA-256,
                              engine, verdict, objective, time), fsync'ed, one file per month,
                              files older than the retention period (180 days by default) removed
  * refusal                   a non-loopback bind without a token or a trusted proxy is refused
  * desktop app               a random per-launch token that even loopback clients must present
                              (`loopback_requires_token`), so other local processes and other users
                              of the machine cannot drive the planner's session
Nothing here opens an outbound connection.
"""
from __future__ import annotations

import datetime as dt
import getpass
import hashlib
import hmac
import ipaddress
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

_AUDIT_LOCK = threading.Lock()


def is_loopback(addr: str) -> bool:
    try:
        return ipaddress.ip_address(addr).is_loopback
    except ValueError:
        return addr in ("localhost",)


@dataclass
class AccessPolicy:
    token: str | None = None
    proxy_user_header: str | None = None
    trusted_proxies: set = field(default_factory=set)
    audit_dir: Path | None = None
    retention_days: int = 180
    loopback_requires_token: bool = False

    @classmethod
    def from_args(cls, token_file=None, proxy_user_header=None, trusted_proxies=(), audit_dir=None,
                  retention_days=180):
        token = os.environ.get("QENIVO_TOKEN") or None
        if token_file:
            token = Path(token_file).read_text(encoding="utf-8").strip() or None
        return cls(token=token, proxy_user_header=proxy_user_header, trusted_proxies=set(trusted_proxies or ()),
                   audit_dir=Path(audit_dir) if audit_dir else None, retention_days=int(retention_days))

    def check_bind(self, host: str):
        if not is_loopback(host) and not self.token and not (self.proxy_user_header and self.trusted_proxies):
            raise SystemExit(f"refusing to listen on {host} without --token-file or a trusted proxy "
                             "(--proxy-user-header with --trusted-proxy); see docs/DEPLOYMENT.md")

    def authenticate(self, client: str, headers) -> tuple[str | None, str]:
        """Return (user, how) or (None, reason)."""
        if self.proxy_user_header and client in self.trusted_proxies:
            user = headers.get(self.proxy_user_header)
            if user:
                return user.strip(), "proxy"
            return None, f"trusted proxy sent no {self.proxy_user_header}"
        if self.token:
            got = headers.get("Authorization", "")
            if got.startswith("Bearer ") and hmac.compare_digest(got[7:].strip(), self.token):
                return headers.get("X-QENIVO-User", "token-holder"), "token"
            if not is_loopback(client) or self.loopback_requires_token:
                return None, "missing or wrong token"
        if is_loopback(client):
            try:
                return f"local:{getpass.getuser()}", "local"
            except Exception:  # noqa: BLE001
                return "local", "local"
        return None, "not allowed"

    # ------------------------------------------------------------------ audit
    def audit(self, record: dict):
        if self.audit_dir is None:
            return
        now = dt.datetime.now(dt.timezone.utc)
        rec = {"utc": now.isoformat(timespec="seconds"), **record}
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        path = self.audit_dir / f"audit-{now:%Y-%m}.jsonl"
        line = json.dumps(rec, default=str) + "\n"
        with _AUDIT_LOCK, open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def purge_old(self) -> list:
        """Remove monthly audit files entirely older than the retention period."""
        if self.audit_dir is None or not self.audit_dir.exists():
            return []
        cutoff = dt.date.today() - dt.timedelta(days=self.retention_days)
        gone = []
        for f in self.audit_dir.glob("audit-*.jsonl"):
            try:
                y, m = map(int, f.stem.split("-")[1:3])
            except ValueError:
                continue
            last_day = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1))
            if last_day < cutoff:
                f.unlink()
                gone.append(f.name)
        return gone


def model_digest(text: str | bytes | None) -> str | None:
    if text is None:
        return None
    b = text.encode("utf-8", "replace") if isinstance(text, str) else text
    return hashlib.sha256(b).hexdigest()
