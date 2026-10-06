"""The console API (`qenivo serve`, also behind the desktop app) over real HTTP.

Covers built-in models, planner plan tables, the crude curve in planner units, xlsx case stacks,
strict JSON, temp-file hygiene, and the browser-facing defences (Host / Origin / desktop token).
"""
from __future__ import annotations

import base64
import http.client
import json
import tempfile
import threading

import pytest

from qenivo import server, solve
from qenivo.desktop.catalog import builtin, objective_view
from qenivo.engines import native_simplex as ns
from qenivo.security import AccessPolicy


@pytest.fixture
def live():
    """A server on a free loopback port; yields a request(method, path, body, headers) helper."""
    servers = []

    def start(policy=None):
        httpd = server.make_server("127.0.0.1", 0, policy or AccessPolicy())
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        port = httpd.server_port

        def request(method, path, body=None, headers=None):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
            hdrs = {"Host": f"127.0.0.1:{port}", **(headers or {})}
            data = None if body is None else json.dumps(body).encode()
            conn.request(method, path, body=data, headers=hdrs)
            r = conn.getresponse()
            raw = r.read()
            conn.close()
            return r.status, raw, dict(r.getheaders())
        request.port = port
        return request

    yield start
    for h in servers:
        h.shutdown()
        h.server_close()


def strict_json(raw: bytes):
    """Parse as a browser's JSON.parse would: NaN / Infinity are not JSON."""
    def refuse(token):
        raise ValueError(f"non-JSON token {token}")
    return json.loads(raw, parse_constant=refuse)


def post(req, path, body, **headers):
    code, raw, _ = req("POST", path, body, headers)
    return code, strict_json(raw)


# ------------------------------------------------------------------ JSON, temp files, robustness
def test_ranging_response_is_strict_json(live):
    """Ranging bounds are often infinite; the old server sent `Infinity`, which browsers reject."""
    req = live()
    code, out = post(req, "/api/range", {"builtin": "williams"})
    assert code == 200
    assert any(d["cost_max"] is None or d["cost_min"] is None for d in out["cost"])   # unbounded side -> null


def test_requests_leave_no_temp_files(live, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    req = live()
    mps = "NAME t\nROWS\n N obj\n L c1\nCOLUMNS\n x obj -1 c1 1\nRHS\n rhs c1 4\nENDATA\n"
    code, out = post(req, "/api/solve", {"model": mps})
    assert code == 200 and out["verdict"] == "optimal"
    code, rep = post(req, "/api/verify", {"model": mps, "certificate": out["certificate"]})
    assert code == 200 and rep["passed"]
    assert post(req, "/api/verify", {"builtin": "williams",
                                     "certificate": post(req, "/api/solve", {"builtin": "williams"})[1]["certificate"]}
                )[1]["passed"]
    assert list(tmp_path.iterdir()) == []


def test_bad_content_length_is_an_error_not_a_crash(live):
    req = live()
    code, raw, _ = req("POST", "/api/solve", None, {"Content-Length": "lots"})
    assert code == 400 and "Content-Length" in strict_json(raw)["error"]
    code, raw, _ = req("POST", "/api/solve", None, {"Content-Length": "-5"})
    assert code == 400
    code, raw, _ = req("POST", "/api/solve", [1, 2])                    # JSON, but not an object
    assert code == 400 and "JSON object" in strict_json(raw)["error"]


def test_unknown_builtin_is_a_clear_error(live):
    code, out = post(live(), "/api/solve", {"builtin": "nope"})
    assert code == 400 and "unknown built-in" in out["error"]


# ------------------------------------------------------------------ catalogue and planner tables
def test_catalog_and_crude_columns(live):
    req = live()
    code, raw, _ = req("GET", "/api/catalog")
    keys = [m["key"] for m in strict_json(raw)["models"]]
    assert {"refinery_l3", "williams"} <= set(keys)
    code, cols = post(req, "/api/crude-columns", {"builtin": "williams"})
    assert [c["column"] for c in cols["columns"]] == ["crude1", "crude2"]
    assert all(c["relative_ok"] is False for c in cols["columns"])          # costless supply


def _fd_shadow_prices(key, n=3):
    """(reported, left FD, right FD) planner shadow prices for the top binding limits.

    The optimal value is piecewise linear in a limit. Off a kink both one-sided differences equal
    the dual; at a degenerate vertex they differ and the dual is a subgradient between them.
    """
    p = builtin(key)
    s = solve(p, engine="simplex", tol=1e-9)
    shown = objective_view(p, s.objective)["value"]
    req_out = server.api_solve({"builtin": key, "engine": "simplex", "tol": 1e-9})
    pairs = []
    rn = p.names()[0]
    for lim in [d for d in req_out["planner"]["limits"] if d["side"] != "-"][:n]:
        i = rn.index(lim["row"])
        h = 1e-4 * max(1.0, abs(lim["limit"]))
        fds = []
        for step in (-h, h):
            q = p.copy()
            q.lc, q.uc = q.lc.copy(), q.uc.copy()
            if lim["side"] in ("fixed", "upper"):
                q.uc[i] += step
            if lim["side"] in ("fixed", "lower"):
                q.lc[i] += step
            sq = solve(q, engine="simplex", tol=1e-9)
            fds.append(None if sq.verdict != "optimal" else (objective_view(q, sq.objective)["value"] - shown) / step)
        pairs.append((lim["shadow_price"], *fds))
    return pairs


def _assert_subgradient(pairs):
    assert pairs
    for rep, left, right in pairs:
        sides = [v for v in (left, right) if v is not None]     # a step may make the plan infeasible
        assert sides
        tol = 1e-5 * max(1.0, abs(rep))
        assert min(sides) - tol <= rep <= max(sides) + tol, (rep, left, right)


@pytest.mark.parametrize("key", ["williams", "crude_blending"])    # a max model and a min model
def test_planner_shadow_prices_match_finite_differences(key):
    _assert_subgradient(_fd_shadow_prices(key))


@pytest.mark.skipif(ns.library() is None, reason="refinery L2 takes ~20 s per solve without the native simplex")
def test_planner_shadow_prices_refinery_margin_sign():
    """Refinery models report net margin = -objective: the third sign combination."""
    _assert_subgradient(_fd_shadow_prices("refinery_l2", n=6))


def test_solve_builtin_returns_plan_tables(live):
    code, out = post(live(), "/api/solve", {"builtin": "williams", "engine": "simplex"})
    assert code == 200 and out["verdict"] == "optimal"
    pl = out["planner"]
    assert pl["objective"]["label"] == "Daily profit"
    assert pl["objective"]["value"] == pytest.approx(211365.13, rel=1e-7)     # published optimum
    assert pl["plan"]["crude_slate"] and pl["limits"]


def test_pdhg_solve_carries_a_trace(live):
    code, out = post(live(), "/api/solve", {"builtin": "williams", "engine": "pdhg", "backend": "numpy"})
    assert code == 200 and out["history"]
    assert {"iter", "rel_primal", "rel_dual", "rel_gap"} <= set(out["history"][0])


# ------------------------------------------------------------------ crude valuation
def test_crude_value_display_matches_the_solved_objective(live):
    """The curve's value at today's price is the plan's displayed objective (planner units, with c0)."""
    req = live()
    key = "crude_blending"
    col = "c_0"
    code, cv = post(req, "/api/crude-value", {"builtin": key, "column": col, "t_lo": -0.2, "t_hi": 0.2})
    assert code == 200 and cv["cert_valid"] is True, cv.get("summary")
    seg = next(s for s in cv["display"]["segments"] if s["t_lo"] <= 0.0 <= s["t_hi"])
    k = (0.0 - seg["t_lo"]) / ((seg["t_hi"] - seg["t_lo"]) or 1.0)
    at_today = seg["value_lo"] + k * (seg["value_hi"] - seg["value_lo"])
    shown = post(req, "/api/solve", {"builtin": key, "engine": "simplex", "tol": 1e-9})[1]["planner"]["objective"]["value"]
    assert at_today == pytest.approx(shown, rel=1e-7, abs=1e-6)
    p = builtin(key)
    assert cv["display"]["base_price"] == pytest.approx(float(p.c[p.names()[1].index(col)]))


@pytest.mark.parametrize("key,col", [("crude_blending", "c_3"), ("williams", "crude2")])
def test_curve_slope_is_the_cargo_volume(live, key, col):
    """Envelope theorem: d(objective)/d(price of j) = x_j on each segment, in the displayed units.

    Ties the planner-unit transform, the price axis and the plan-change volumes to one identity.
    """
    p = builtin(key)
    rel = float(p.c[p.names()[1].index(col)]) != 0.0
    body = {"builtin": key, "column": col, "relative": rel, "t_lo": -0.3 if rel else 0.0, "t_hi": 0.3 if rel else 30.0}
    code, cv = post(live(), "/api/crude-value", body)
    assert code == 200 and cv["cert_valid"] is True
    k = float(p.meta["planner"]["objective_sign"]) * float(p.obj_sign)
    for s in cv["display"]["segments"]:
        if s["price_hi"] - s["price_lo"] < 1e-9:
            continue
        slope = (s["value_hi"] - s["value_lo"]) / (s["price_hi"] - s["price_lo"])
        assert slope == pytest.approx(k * s["volume"], rel=1e-7, abs=1e-6)
    for ch in cv["display"]["changes"]:
        assert ch["starts"] or ch["stops"]          # every plan change names what moved


def test_costless_cargo_needs_an_absolute_ray(live):
    req = live()
    code, out = post(req, "/api/crude-value", {"builtin": "williams", "column": "crude1"})
    assert code == 200 and "absolute" in out["error"]
    code, cv = post(req, "/api/crude-value", {"builtin": "williams", "column": "crude1",
                                              "relative": False, "t_lo": 0.0, "t_hi": 20.0})
    assert code == 200 and cv["cert_valid"] is True
    assert cv["display"]["segments"][0]["price_lo"] == pytest.approx(0.0)


# ------------------------------------------------------------------ case stacks with xlsx in and out
def test_cases_xlsx_round_trip(live, tmp_path):
    from qenivo.io.sheets import read_rows, write_rows
    src = tmp_path / "cases.xlsx"
    write_rows(src, [["case", "kind", "name", "value"],
                     ["base", "cost", "crude1", 0.0],
                     ["tight_cdu", "row_hi", "distillation_cap", 40000.0]])
    code, out = post(live(), "/api/cases", {"builtin": "williams", "want_xlsx": True,
                                            "cases_xlsx": base64.b64encode(src.read_bytes()).decode()})
    assert code == 200
    assert [t["case"] for t in out["table"]] == ["base", "tight_cdu"]
    assert all(t["verdict"] == "optimal" for t in out["table"])
    back = tmp_path / "report.xlsx"
    back.write_bytes(base64.b64decode(out["report_xlsx"]))
    assert len(read_rows(back)) >= 3


# ------------------------------------------------------------------ browser-facing defences
def test_dns_rebinding_host_is_refused(live):
    req = live()
    code, raw, _ = req("GET", "/api/info", headers={"Host": f"evil.example:{req.port}"})
    assert code == 403
    code, raw, _ = req("POST", "/api/solve", {"builtin": "williams"}, {"Host": f"evil.example:{req.port}"})
    assert code == 403


def test_cross_origin_post_is_refused_but_scripts_and_same_origin_pass(live):
    req = live()
    body = {"builtin": "williams", "engine": "simplex"}
    assert req("POST", "/api/solve", body, {"Origin": "https://evil.example"})[0] == 403
    assert req("POST", "/api/solve", body, {"Origin": "null"})[0] == 403
    assert req("POST", "/api/solve", body, {"Origin": f"http://127.0.0.1:{req.port}"})[0] == 200
    assert req("POST", "/api/solve", body)[0] == 200                       # no Origin: a script


def test_console_page_and_stylesheet(live):
    req = live()
    code, raw, hdrs = req("GET", "/")
    assert code == 200 and b"Try QENIVO" in raw and b"QENIVO" in raw
    assert "text/html" in hdrs.get("Content-Type", "")
    code, css, hdrs = req("GET", "/tokens.css")
    assert code == 200 and b"--accent" in css and b".lab" in css
    assert hdrs.get("Content-Type", "").startswith("text/css")


def test_desktop_token_mode(live):
    req = live(AccessPolicy(token="launch-token", loopback_requires_token=True))
    code, raw, hdrs = req("GET", "/")
    assert code == 200 and b"QENIVO" in raw                               # page loads to read its token
    assert "connect-src 'self'" in hdrs.get("Content-Security-Policy", "")
    assert req("GET", "/api/info")[0] == 401
    assert req("POST", "/api/solve", {"builtin": "williams"})[0] == 401
    ok = {"Authorization": "Bearer launch-token"}
    assert req("GET", "/api/info", headers=ok)[0] == 200
    assert req("POST", "/api/solve", {"builtin": "williams", "engine": "simplex"}, ok)[0] == 200
    assert req("GET", "/api/info", headers={"Authorization": "Bearer wrong"})[0] == 401


def test_info_reports_edition_and_native_status(live):
    code, raw, _ = live()("GET", "/api/info")
    info = strict_json(raw)
    assert info["edition"] in ("public", "pro") and isinstance(info["native"]["simplex"], bool)
    assert info["crude_valuation"] is True
    assert info["version"] and info["provenance"]["tripwires_installed"]


# ------------------------------------------------------------------ desktop launcher and CLI entry
def test_desktop_start_enforces_its_launch_token():
    from urllib.parse import urlparse

    from qenivo.desktop.app import start
    httpd, url = start()
    try:
        u = urlparse(url)
        token = u.fragment.split("t=", 1)[1]
        assert u.hostname == "127.0.0.1" and len(token) >= 32

        def get(path, auth=None):
            conn = http.client.HTTPConnection("127.0.0.1", u.port, timeout=60)
            conn.request("GET", path, headers={"Host": u.netloc, **({"Authorization": f"Bearer {auth}"} if auth else {})})
            r = conn.getresponse()
            r.read()
            conn.close()
            return r.status
        assert get("/") == 200
        assert get("/api/info") == 401
        assert get("/api/info", token) == 200
        assert start()[1] != url                     # a fresh token and port every launch
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_cli_hands_io_its_own_arguments(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    mps = Path(__file__).resolve().parent / "data"
    model = next(mps.glob("*.mps"))
    script = tmp_path / "cmds.txt"
    script.write_text(f"read {model}\noptimize\ndisplay solution objective\nquit\n", encoding="utf-8")
    src = Path(__file__).resolve().parents[1] / "src"
    r = subprocess.run([sys.executable, "-m", "qenivo.cli", "io", "-f", str(script)], capture_output=True,
                       text=True, timeout=300, env={**__import__("os").environ, "PYTHONPATH": str(src)})
    assert r.returncode == 0, r.stderr[-2000:]
    assert "objective" in r.stdout.lower()


# ------------------------------------------------------------------ case patterns and scaling
def test_case_patterns_expand_like_explicit_names(live):
    req = live()
    p = builtin("crude_blending")
    cn = p.names()[1]
    explicit = {"name": "explicit", "col_hi": {n: 0 for n in cn if n.startswith("c_1")}}   # c_1, c_10, c_11
    pattern = {"name": "pattern", "col_hi": {"c_1*": 0}}
    code, out = post(req, "/api/cases", {"builtin": "crude_blending", "cases": [explicit, pattern]})
    assert code == 200
    a, b = out["table"]
    assert a["verdict"] == b["verdict"] and a["objective"] == pytest.approx(b["objective"], rel=1e-12)


def test_cost_scale_keeps_the_sign_on_a_max_model(live):
    """Williams maximises profit: +10% on the premium petrol price must raise profit, exactly as typed."""
    req = live()
    p = builtin("williams")
    j = p.names()[1].index("PMF")
    price = float(p.obj_sign * p.c[j])              # the model's own sense: the selling price, positive
    assert price > 0
    cases = [{"name": "base"}, {"name": "typed", "cost": {"PMF": price * 1.1}},
             {"name": "scaled", "cost_scale": {"PMF": 1.1}}]
    code, out = post(req, "/api/cases", {"builtin": "williams", "cases": cases})
    assert code == 200
    base, typed, scaled = out["table"]
    assert scaled["objective"] == pytest.approx(typed["objective"], rel=1e-12)
    assert scaled["value"] > base["value"]          # dearer product, more profit


def test_a_pattern_that_matches_nothing_is_an_error(live):
    code, out = post(live(), "/api/cases", {"builtin": "williams", "cases": [{"name": "typo", "col_hi": {"BUY_*": 0}}]})
    assert code == 400 and "matches no name" in out["error"]
