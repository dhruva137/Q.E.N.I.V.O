"""`qenivo serve`: a local planner console and JSON API (standard library only, works air-gapped).

    qenivo serve --port 8765          then open http://127.0.0.1:8765

Endpoints (JSON in, JSON out). A model is sent as text (`model`, MPS / GAMS) or named from the
built-in catalogue (`builtin`, see GET /api/catalog), so a large model never has to travel:
    GET  /api/info                     version, edition, native cores, GPU, provenance guard
    GET  /api/catalog                  built-in planner models
    POST /api/solve        {model|builtin, format?, engine?, tol?, backend?}
                                       -> verdict, objective, residuals, routing, top values, marginals,
                                          certificate; plus plan tables when the model has planner names
    POST /api/explain      {model|builtin}                  -> infeasibility diagnosis + Farkas proof
    POST /api/verify       {model|builtin, certificate, exact?}  -> independent re-check
    POST /api/cases        {model|builtin, cases | cases_xlsx (base64), want_xlsx?}
    POST /api/range        {model|builtin}                  -> exact cost / RHS ranging
    POST /api/crude-columns {model|builtin}                 -> cargo columns grouped by crude grade
    POST /api/crude-value  {model|builtin, column, t_lo?, t_hi?, relative?}
                                       -> exact break-even curve, segment certificates, plan changes
JSON is strict (RFC 8259): an infinite or undefined number is sent as null (an absent bound).
Binds to 127.0.0.1 by default; nothing leaves the machine. When bound to loopback the Host header
must name the loopback address (DNS-rebinding defence), and a browser request from another origin
is refused. For a shared server, company sign-in through the site's reverse proxy, and the 180-day
audit trail, see qenivo/security.py.
"""
from __future__ import annotations

import base64
import json
import math
import os
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

UI = Path(__file__).resolve().parent / "ui" / "index.html"
TOKENS = Path(__file__).resolve().parent / "ui" / "tokens.css"
UI_PATHS = ("/", "/index.html")
MAX_BODY = 512 * 1024 * 1024
_LOCK = threading.Lock()
# The page may talk only to this server: no CDN, font, image or analytics request can leave the machine.
CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
       "img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")


# --------------------------------------------------------------------------- model sources
_BUILTINS: dict = {}


def _builtin(key: str):
    """A catalogue model; generated once, then handed out as a copy so no request can alter another's."""
    from .desktop.catalog import BUILTINS, builtin
    if key not in BUILTINS:
        raise ValueError(f"unknown built-in model {key!r}; see GET /api/catalog")
    if key not in _BUILTINS:
        _BUILTINS[key] = builtin(key)
    return _BUILTINS[key].copy()


def _parse_text(text: str, fmt: str | None, name: str | None):
    from .api import read
    fmt = (fmt or ("gams" if "Solve " in text and ".." in text else "mps")).lower()
    fd, path = tempfile.mkstemp(suffix=".gms" if fmt == "gams" else ".mps")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        if fmt == "gams":
            from .io.gams import read_gams
            return read_gams(path), "gams"
        return read(path, name=name), "mps"
    finally:
        os.unlink(path)


def _load(body):
    """(problem, kind) from a request: kind is "builtin", "mps" or "gams"."""
    if body.get("builtin"):
        return _builtin(str(body["builtin"])), "builtin"
    text = body.get("model")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("send a model as text ('model') or name a built-in one ('builtin')")
    return _parse_text(text, body.get("format"), body.get("name"))


def _model_file(body, prob) -> str:
    """A model file the verifier can re-read with its own reader; the caller deletes it."""
    fd, path = tempfile.mkstemp(suffix=".mps")
    os.close(fd)
    if body.get("builtin"):
        from .io.mps import write_mps
        write_mps(prob, path, name=prob.name)
    else:
        Path(path).write_text(body["model"], encoding="utf-8")
    return path


def _pairs(names, vals, top):
    import numpy as np
    v = np.asarray(vals, float)
    idx = np.argsort(-np.abs(v))[:top]
    return [[names[i], float(v[i])] for i in idx if abs(v[i]) > 0]


def _planner_sign(prob) -> float:
    return float((prob.meta.get("planner") or {}).get("objective_sign", 1.0))


# --------------------------------------------------------------------------- endpoints
def api_solve(body):
    import numpy as np

    from .api import solve
    prob, kind = _load(body)
    if kind == "gams":
        from .workload.slp import run_slp
        r = run_slp(prob, engine=body.get("engine", "auto"), max_iter=int(body.get("max_iter", 60)))
        return {"kind": "bilinear", "size": prob.size_str(), "objective": r.objective, "status": r.status,
                "feasible": r.feasible, "max_violation": r.max_violation, "iterations": r.iterations,
                "time": r.time, "history": r.history}
    sol = solve(prob, engine=body.get("engine", "auto"), tol=float(body.get("tol", 1e-6)),
                backend=body.get("backend", "auto"))
    rn, cn = prob.names()
    out = {"kind": "lp", "model": prob.name, "size": prob.size_str(), "verdict": sol.verdict,
           "objective": sol.objective, "residuals": sol.residuals, "engine": sol.engine,
           "top_values": _pairs(cn, sol.x, 25) if sol.x is not None else [],
           "marginals": _pairs(rn, sol.y, 25) if sol.y is not None else [],
           "history": sol.history,
           "certificate": sol.certificate(include_solution=True)}
    if kind == "builtin":
        out["builtin"] = body["builtin"]
    if prob.meta.get("planner") and sol.x is not None and sol.verdict == "optimal":
        from .desktop.catalog import binding_limits, objective_view, plan_tables
        # Planner shadow price: the change in the displayed objective per unit rise of the limit.
        # sol.y is in the model's own sense; the planner's sign maps it (finite-difference checked
        # in tests/test_server.py on a max and a min model).
        y = None if sol.y is None else _planner_sign(prob) * np.asarray(sol.y, dtype=float)
        out["planner"] = {"objective": objective_view(prob, sol.objective),
                          "plan": plan_tables(prob, sol.x, y),
                          "limits": [] if y is None else binding_limits(prob, sol.x, y)}
    if sol.verdict == "infeasible":
        from .workload.explain import explain_infeasibility
        out["explanation"] = explain_infeasibility(prob)
    return out


def api_explain(body):
    from .workload.explain import explain_infeasibility, format_explanation
    prob, _ = _load(body)
    rep = explain_infeasibility(prob)
    rep["text"] = format_explanation(rep)
    return rep


def api_verify(body):
    from .certify.verify import format_report, verify
    cert = body["certificate"] if isinstance(body["certificate"], dict) else json.loads(body["certificate"])
    prob = _builtin(str(body["builtin"])) if body.get("builtin") else None
    path = _model_file(body, prob)
    try:
        rep = verify(path, cert, exact=bool(body.get("exact")))
    finally:
        os.unlink(path)
    rep["text"] = format_report(rep)
    rep["checks"] = [list(c) for c in rep["checks"]]
    return rep


def api_cases(body):
    from .workload.cases import Case, solve_cases
    prob, _ = _load(body)
    if body.get("cases_xlsx"):
        from .io.sheets import read_case_table
        fd, path = tempfile.mkstemp(suffix=".xlsx")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(base64.b64decode(body["cases_xlsx"]))
            cases = read_case_table(path)
        finally:
            os.unlink(path)
    else:
        cases = [Case(name=c["name"], cost=c.get("cost", {}), col_lo=c.get("col_lo", {}), col_hi=c.get("col_hi", {}),
                      row_lo=c.get("row_lo", {}), row_hi=c.get("row_hi", {})) for c in body["cases"]]
    if not cases:
        raise ValueError("no cases given")
    if not body.get("cases_xlsx"):
        cases = [_expand_case(prob, c, raw) for c, raw in zip(cases, body["cases"])]
    rep = solve_cases(prob, cases, tol=float(body.get("tol", 1e-6)), backend=body.get("backend", "auto"))
    out = {"engine": rep.engine, "wall_time": rep.wall_time, "table": rep.table()}
    if prob.meta.get("planner"):
        from .desktop.catalog import objective_view
        ps = _planner_sign(prob)
        out["objective_label"] = objective_view(prob, None)["label"]
        for row in out["table"]:                     # the planner's objective (e.g. net margin, not -margin)
            row["value"] = None if row.get("objective") is None else ps * row["objective"]
            row["value_delta"] = None if row.get("delta_vs_first") is None else ps * row["delta_vs_first"]
    if body.get("want_xlsx"):
        from .io.sheets import write_case_report
        fd, path = tempfile.mkstemp(suffix=".xlsx")
        os.close(fd)
        try:
            write_case_report(path, rep)
            out["report_xlsx"] = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        finally:
            os.unlink(path)
    return out


def _expand_case(prob, case, raw: dict):
    """Names may be shell-style patterns, and cost_scale multiplies current costs.

        {"name": "Basrah +10%", "cost_scale": {"BUY_Basrah_Light_S*": 1.1}}
        {"name": "no Arab Light", "col_hi": {"BUY_Arab_Light_S*": 0}}

    A planner thinks "every Basrah cargo", not 48 column names. Case costs are in the model's own
    sense (cases.py applies c = obj_sign * value), so a scaled cost is obj_sign * c[j] * factor, right
    for min and max models alike. A pattern that matches nothing is an error, not a silent no-op.
    """
    from fnmatch import fnmatchcase

    from .workload.cases import Case
    rn, cn = prob.names()

    def names(pattern: str, pool: list) -> list:
        if not any(ch in pattern for ch in "*?["):
            return [pattern]
        hit = [n for n in pool if fnmatchcase(n, pattern)]
        if not hit:
            raise ValueError(f"case {case.name!r}: pattern {pattern!r} matches no name")
        return hit

    def expand(d: dict, pool: list) -> dict:
        return {n: v for k, v in d.items() for n in names(k, pool)}

    cost = expand(case.cost, cn)
    ci = {n: j for j, n in enumerate(cn)}
    for k, f in (raw.get("cost_scale") or {}).items():
        for n in names(k, cn):
            if n not in ci:
                raise ValueError(f"case {case.name!r}: no column named {n!r}")
            cost[n] = float(prob.obj_sign) * float(prob.c[ci[n]]) * float(f)
    return Case(name=case.name, cost=cost, col_lo=expand(case.col_lo, cn), col_hi=expand(case.col_hi, cn),
                row_lo=expand(case.row_lo, rn), row_hi=expand(case.row_hi, rn))


def api_range(body):
    from .api import solve
    from .workload.ranging import cost_ranging, rhs_ranging
    prob, _ = _load(body)
    sol = solve(prob, engine="simplex", tol=1e-9)
    if sol.verdict != "optimal":
        return {"error": f"model is {sol.verdict}; ranging needs an optimal basis"}
    return {"objective": sol.objective, "cost": cost_ranging(prob, sol),
            "rhs": [d for d in rhs_ranging(prob, sol) if d["binding"]]}


def api_info(_body=None):
    from . import __version__
    from .certify.certificate import environment
    from .edition import detect_edition
    from .engines import native_simplex
    from .provenance import guard
    try:
        from .workload import crude_value  # noqa: F401
        valuation = True
    except ImportError:
        valuation = False
    return {"version": __version__, "edition": detect_edition(), "environment": environment(),
            "provenance": guard(), "crude_valuation": valuation,
            "native": {"simplex": native_simplex.library() is not None, "status": native_simplex.status()}}


def api_catalog(_body=None):
    from .desktop.catalog import BUILTINS
    return {"models": [{"key": k, "title": v["title"], "detail": v["detail"]} for k, v in BUILTINS.items()]}


def api_crude_columns(body):
    from .desktop.catalog import crude_columns
    prob, _ = _load(body)
    return crude_columns(prob)


def api_crude_value(body):
    """Exact break-even curve along one cargo's price ray, with a certificate per segment.

    relative (default True): t is a fractional move of the cargo's cost (-0.2 = -20%). A cargo that
    costs nothing in the model has no price to scale, so it takes an absolute ray (t in price units).
    The engine ships in the full edition; the public edition answers honestly that it is absent.
    """
    try:
        from .workload.crude_value import crude_value
    except ImportError:
        return {"status": "full_edition_only", "note": "crude valuation ships in the full edition",
                "detail": "This build does not include qenivo.workload.crude_value (see docs/EDITIONS.md).",
                "request": {"column": body.get("column"), "t_lo": body.get("t_lo", -0.2),
                            "t_hi": body.get("t_hi", 0.2)},
                "segments": [], "breakpoints": [], "explain": ""}
    prob, _ = _load(body)
    column = body.get("column") or body.get("cargo")
    if not column:
        return {"error": "column (or cargo) is required"}
    _, cn = prob.names()
    if column not in cn:
        return {"error": f"no column named {column!r} in this model"}
    j = cn.index(column)
    relative = bool(body.get("relative", True))
    base = float(prob.c[j])
    if relative and base == 0.0:
        return {"error": f"{column} costs nothing in this model, so a percentage move is zero; "
                         "use an absolute price range (relative: false)"}
    t_lo, t_hi = float(body.get("t_lo", -0.2)), float(body.get("t_hi", 0.2))
    if not t_lo < t_hi:
        return {"error": "the price range must have t_lo < t_hi"}
    result = crude_value(prob, column, (t_lo, t_hi), relative=relative, unit=str(body.get("unit", "bbl")))
    payload = result.to_json()
    payload.setdefault("status", "ok")
    payload.setdefault("note", "")
    payload["display"] = _curve_display(prob, j, relative, payload)
    return payload


def _curve_display(prob, j: int, relative: bool, payload: dict) -> dict:
    """The curve in the planner's units: cargo price on x, displayed objective on y.

    phi(t) = intercept + slope*t is the minimised c(t)'x without the constant c0, so the model's own
    objective is obj_sign*(phi + c0) and the planner's is that times the planner sign.
    """
    from .desktop.catalog import objective_view
    base = float(prob.c[j])
    k = _planner_sign(prob) * float(prob.obj_sign)
    c0 = float(prob.c0)

    def price(t: float) -> float:
        return base * (1.0 + t) if relative else base + t

    segs = []
    for s in payload.get("segments") or []:
        a, b = float(s.get("intercept", 0.0)), float(s["slope"])
        segs.append({"t_lo": s["t_lo"], "t_hi": s["t_hi"], "price_lo": price(s["t_lo"]), "price_hi": price(s["t_hi"]),
                     "value_lo": k * (a + b * s["t_lo"] + c0), "value_hi": k * (a + b * s["t_hi"] + c0),
                     "volume": s.get("volume"), "cert_valid": s.get("cert_valid")})
    # Plan changes in planner words. On this ray d = c_j e_j (relative) or e_j (absolute), so the curve's
    # slope d'x is the cargo's own volume times d_j.
    from .desktop.catalog import label
    dj = base if relative else 1.0
    changes = []
    for ch in payload.get("changes") or []:
        changes.append({"t": ch["t"], "price": price(ch["t"]),
                        "volume_before": ch["slope_before"] / dj if dj else None,
                        "volume_after": ch["slope_after"] / dj if dj else None,
                        "starts": [label(n) for n in ch.get("entering") or []],
                        "stops": [label(n) for n in ch.get("leaving") or []], "detail": ch.get("text", "")})
    return {"label": objective_view(prob, None)["label"], "base_price": base, "relative": relative,
            "segments": segs, "changes": changes,
            "price_breakpoints": [price(t) for t in payload.get("breakpoints") or []]}


ROUTES = {"/api/solve": api_solve, "/api/explain": api_explain, "/api/verify": api_verify,
          "/api/cases": api_cases, "/api/range": api_range, "/api/crude-value": api_crude_value,
          "/api/crude-columns": api_crude_columns}
GET_ROUTES = {"/api/info": api_info, "/api/catalog": api_catalog}


POLICY = None          # security.AccessPolicy, set by serve(); a server made by make_server() carries its own


# --------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "qenivo"

    def _policy(self):
        from .security import AccessPolicy
        return getattr(self.server, "policy", None) or POLICY or AccessPolicy()

    def _auth(self):
        pol = self._policy()
        user, how = pol.authenticate(self.client_address[0], self.headers)
        if user is None:
            pol.audit({"event": "denied", "path": self.path, "client": self.client_address[0], "reason": how})
            self._send(401, {"error": f"not authorised: {how}"})
        return user, how

    def _origin_problem(self, post: bool) -> str | None:
        """Why this request must be refused before authentication, or None.

        On a loopback bind the Host header must name the loopback address and this port, so a page on
        another site cannot reach the server by re-pointing its own DNS name at 127.0.0.1. A browser
        always sends Origin on a cross-site POST; it must name this same host.
        """
        from .security import is_loopback
        host = (self.headers.get("Host") or "").strip().lower()
        bound, port = self.server.server_address[:2]
        if is_loopback(str(bound)) and host not in {f"{n}:{port}" for n in LOOPBACK_NAMES}:
            return "unexpected Host header"
        origin = self.headers.get("Origin")
        if post and origin is not None and urlparse(origin).netloc.lower() != host:
            return "cross-origin request"
        return None

    def _send(self, code, payload, ctype="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(
            _strict(payload), default=_default, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if ctype.startswith("text/html"):
            self.send_header("Content-Security-Policy", CSP)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")

    def do_GET(self):
        why = self._origin_problem(post=False)
        if why:
            return self._send(403, {"error": f"refused: {why}"})
        path = urlparse(self.path).path
        # Fixed path only. A <link> cannot send the desktop bearer token, and this file holds no
        # model data. Anything other than this exact path falls through to auth and the 404.
        if path == "/tokens.css":
            return self._send(200, TOKENS.read_bytes(), "text/css; charset=utf-8")
        if path in UI_PATHS and self._policy().loopback_requires_token:
            # Desktop mode: the page holds no data and must load before it can present its token.
            return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
        user, _ = self._auth()
        if user is None:
            return
        if path in UI_PATHS:
            return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
        if path in GET_ROUTES:
            return self._send(200, GET_ROUTES[path]())
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        from .security import is_loopback, model_digest
        why = self._origin_problem(post=True)
        if why:
            return self._send(403, {"error": f"refused: {why}"})
        user, how = self._auth()
        if user is None:
            return
        pol = self._policy()
        fn = ROUTES.get(urlparse(self.path).path)
        if fn is None:
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return self._send(400, {"error": "bad Content-Length"})
        if n < 0:
            return self._send(400, {"error": "bad Content-Length"})
        if n > MAX_BODY:
            return self._send(413, {"error": "model too large for the console; use the CLI"})
        t0 = time.perf_counter()
        rec = {"event": "request", "path": self.path, "user": user, "auth": how, "client": self.client_address[0]}
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("the request body must be a JSON object")
            rec["model_sha256"] = model_digest(body.get("model"))
            rec["builtin"] = body.get("builtin")
            rec["engine_requested"] = body.get("engine")
            with _LOCK:                      # one solve at a time: predictable timings, bounded memory
                out = fn(body)
            rec.update(status=200, verdict=out.get("verdict") if isinstance(out, dict) else None,
                       objective=out.get("objective") if isinstance(out, dict) else None,
                       engine=(out.get("engine") or {}).get("engine") if isinstance(out, dict) and
                       isinstance(out.get("engine"), dict) else None)
            self._send(200, out)
        except Exception as e:  # noqa: BLE001
            rec.update(status=400, error=f"{type(e).__name__}: {e}")
            err = {"error": f"{type(e).__name__}: {e}"}
            if is_loopback(self.client_address[0]):       # stack traces only for the local user
                err["trace"] = traceback.format_exc()[-2000:]
            self._send(400, err)
        finally:
            rec["seconds"] = round(time.perf_counter() - t0, 4)
            pol.audit(_strict(rec))


def _strict(o):
    """Replace inf / nan (not representable in JSON) by None, recursively."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _strict(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_strict(v) for v in o]
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return o
    if isinstance(o, np.ndarray):
        return _strict(o.tolist())
    if isinstance(o, np.generic):
        return _strict(o.item())
    return o


def _default(o):
    return str(o)


def make_server(host: str = "127.0.0.1", port: int = 8765, policy=None) -> ThreadingHTTPServer:
    """A configured, bound server that the caller runs (serve_forever) and stops (shutdown)."""
    from .provenance import install_tripwires
    from .security import AccessPolicy
    install_tripwires()
    pol = policy or AccessPolicy()
    pol.check_bind(host)
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.policy = pol
    httpd.daemon_threads = True
    return httpd


def serve(host: str = "127.0.0.1", port: int = 8765, policy=None):
    global POLICY
    httpd = make_server(host, port, policy)
    POLICY = httpd.policy
    removed = POLICY.purge_old()
    mode = ("token" if POLICY.token else "") + (" proxy-sso" if POLICY.proxy_user_header else "") or "local only"
    print(f"QENIVO console on http://{host}:{httpd.server_port}  (access: {mode}; audit: {POLICY.audit_dir or 'off'}"
          f"{f'; purged {len(removed)} old audit files' if removed else ''})  Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
