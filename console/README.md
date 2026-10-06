# QENIVO planner console

Offline TypeScript console for the JSON API started by `qenivo serve`. Scripts, styles and fonts are files in this folder. Nothing is loaded from a CDN.

The pages are case library, run history, plan comparison, crude valuation, infeasibility and certificate. The case library is the `LIBRARY` dict in `qenivo/src/qenivo/models/industry.py`. History is the monthly audit file the server writes. Comparison, explain, verify and solve call the live API. Crude valuation calls `POST /api/crude-value` (stub returns `needs w02 merge` until branch `w02-crude-value` is merged).


## Solver

Flags below are the `serve` arguments in `qenivo/src/qenivo/cli.py`:

```
qenivo serve [--port 8765]                      local planner console (browser) and JSON API
```

```
sv.add_argument("--host", default="127.0.0.1")
sv.add_argument("--port", type=int, default=8765)
sv.add_argument("--token-file", help="shared server: requests must send 'Authorization: Bearer <token>'")
sv.add_argument("--proxy-user-header", help="company sign-in via reverse proxy, e.g. X-Remote-User")
sv.add_argument("--trusted-proxy", action="append", default=[], help="address of that proxy (repeatable)")
sv.add_argument("--audit-dir", help="write the audit trail here (one JSONL file per month)")
sv.add_argument("--retention-days", type=int, default=180, help="audit retention (CERT-In: 180 days)")
```

Start the API on the loopback defaults:

```
qenivo serve --host 127.0.0.1 --port 8765
```

History needs the audit directory from the same argument list:

```
qenivo serve --host 127.0.0.1 --port 8765 --audit-dir audit
```

Shared or proxied use, still only those flags:

```
qenivo serve --host 127.0.0.1 --port 8765 --token-file token.txt --proxy-user-header X-Remote-User --trusted-proxy 127.0.0.1 --audit-dir audit --retention-days 180
```

`http://127.0.0.1:8765/` is the HTML page shipped inside the Python package. This console is the process below. The browser calls this host, and the host forwards `/api` to the solver so the page and the API share one origin.

If `--port` is not 8765, set `QENIVO_ORIGIN` to the same origin (default `http://127.0.0.1:8765`). Put the bearer token in the token field when `--token-file` is set. The user field is sent as `X-QENIVO-User`.

## Console

From `qenivo/console`, with Node.js 22.18 or newer:

```
npm install
npm test
npm start
```

**Bundle rebuild:** `static/js/` is gitignored. After pulling source changes under `console/src/`, run `npm install` in `console/` (installs local `typescript`), then `npm run build` or `npm start`. Do not use a bare `npx tsc` — it can pull the wrong package. Demo recording for W12 uses the package UI from `qenivo serve` (`src/qenivo/ui/index.html`); rebuilding this TypeScript console is optional for the script.

Open `http://127.0.0.1:8770`. `CONSOLE_PORT` changes that port. `npm test` compiles TypeScript and runs `node --test`. `npm run build` compiles without tests. `npm start` serves `static/` and refuses to start until `static/js/browser/main.js` exists.

A built-in model is written with the `model` command from `cli.py` (`--param` / `-p`, `--out` / `-o`), for example:

```
qenivo model crude_blending -p crudes=12 -p seed=0 -o crude_blending.mps
```

The library view shows the command for every `LIBRARY` name. Open the MPS in the workspace. Two small MPS samples are included so explain, compare and certificates can be tried before a generated model exists.

## What each view reads

| View | Source |
| --- | --- |
| Case library | Local manifest generated from `industry.py`. Server block is `GET /api/info`: `version`, `environment`, `provenance`. |
| Run history | `audit-YYYY-MM.jsonl` from `AccessPolicy.audit` in `security.py`. Fields filled in `server.Handler`: `utc`, `event`, `path`, `user`, `auth`, `client`, `model_sha256`, `engine_requested`, `status`, `verdict`, `objective`, `engine`, `seconds`, `error`, `reason`. |
| Plan comparison | Two `POST /api/solve` bodies. Tables are `top_values` and `marginals`. Objectives and the largest `marginals` differences are computed in the page. |
| Crude valuation | `POST /api/crude-value` with `column`, `t_lo`, `t_hi`. Stub: `status=needs_w02_merge` / note `needs w02 merge` until W02 merges. |
| Infeasibility | `POST /api/explain`: `feasible`, `kind`, `items`, `total_relaxation`, `conflict_rows`, `farkas_check`, `proof`, `text`. |
| Certificate | `certificate` on the solve response, or a JSON file. Summary is verdict, residuals, engine and the other schema fields. Pretty-printed JSON is the full object. `POST /api/verify` with `model`, `certificate`, `exact`. |

Solve sends `model`, `engine`, `tol`, `backend`, and `format` when format is not auto. Explain and verify send `model` and `format` the same way.
