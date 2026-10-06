"""Server access policy and audit trail (docs/DEPLOYMENT.md)."""
import datetime as dt
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from qenivo import server
from qenivo.security import AccessPolicy


def test_token_required_off_the_machine():
    pol = AccessPolicy(token="s3cret")
    assert pol.authenticate("10.1.2.3", {})[0] is None
    assert pol.authenticate("10.1.2.3", {"Authorization": "Bearer wrong"})[0] is None
    assert pol.authenticate("10.1.2.3", {"Authorization": "Bearer s3cret"})[0] is not None


def test_proxy_user_is_trusted_only_from_the_proxy():
    pol = AccessPolicy(proxy_user_header="X-Remote-User", trusted_proxies={"10.0.0.5"})
    assert pol.authenticate("10.0.0.5", {"X-Remote-User": r"MRPL\planner1"}) == (r"MRPL\planner1", "proxy")
    assert pol.authenticate("10.0.0.9", {"X-Remote-User": r"MRPL\intruder"})[0] is None


def test_refuses_an_open_network_bind():
    with pytest.raises(SystemExit):
        AccessPolicy().check_bind("0.0.0.0")
    AccessPolicy(token="t").check_bind("0.0.0.0")
    AccessPolicy().check_bind("127.0.0.1")


def test_audit_trail_and_retention(tmp_path):
    pol = AccessPolicy(audit_dir=tmp_path, retention_days=180)
    old = tmp_path / "audit-2020-01.jsonl"
    old.write_text("{}\n")
    this = tmp_path / f"audit-{dt.date.today():%Y-%m}.jsonl"
    assert "audit-2020-01.jsonl" in pol.purge_old()
    server.POLICY = pol
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        model = "NAME t\nROWS\n N obj\n L c1\nCOLUMNS\n x obj -1 c1 1\nRHS\n rhs c1 4\nENDATA\n"
        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}/api/solve",
                                     data=json.dumps({"model": model}).encode(), method="POST")
        out = json.loads(urllib.request.urlopen(req, timeout=60).read())
        assert out["verdict"] == "optimal"
    finally:
        httpd.shutdown()
        server.POLICY = None
    rec = [json.loads(line) for line in this.read_text().splitlines()]
    assert rec[-1]["verdict"] == "optimal" and rec[-1]["user"].startswith("local")
    assert len(rec[-1]["model_sha256"]) == 64
