# Deploying QENIVO at a refinery site

Three ways to run it, from one planner's desktop to a shared planning server behind the company
sign-in. None needs internet access. Background and sources: `docs/GROUND_REALITY.md`.

## 1. Install without internet

On a machine that has internet (once):

```
python packaging/build_offline_bundle.py
```

This writes `dist/qenivo-offline-<version>-<platform>/` with every wheel, `SHA256SUMS` and an SPDX
SBOM (`sbom.spdx.json`) that the security team can review. Copy that folder to the planning
network, then:

```
python verify_bundle.py                                          # checks every SHA-256
pip install --no-index --find-links wheels qenivo
qenivo info                                                      # version, engines, provenance guard
```

Build the bundle on the same OS and Python version as the target machine (wheels are platform
specific). Docker sites can use `packaging/Dockerfile` instead.

## 2. One planner's desktop (Windows)

```
qenivo serve                     (or double-click packaging\windows\qenivo-console.bat)
```

The console opens on `http://127.0.0.1:8765` and only that machine can reach it. The signed-in
Windows user is recorded against each solve. Case stacks come straight from Excel:

```
qenivo cases plan.mps cases.xlsx --out results.xlsx
```

`cases.xlsx` has one sheet with columns `case, kind, name, value` (kind: cost, col_lo, col_hi,
row_lo, row_hi), the same idea as a PIMS CASE table.

## 3. Shared planning server

### 3a. With a token

```
qenivo serve --host 0.0.0.0 --port 8765 --token-file C:\qenivo\token.txt --audit-dir C:\qenivo\audit
```

Every request from another machine must send `Authorization: Bearer <token>`. The server refuses
to listen on a network address with neither a token nor a trusted proxy.

### 3b. With the company sign-in (recommended)

Let the site's web server do the sign-in; QENIVO never sees a password.

1. Run QENIVO on the loopback address of the server: `qenivo serve --port 8765
   --proxy-user-header X-Remote-User --trusted-proxy 127.0.0.1 --audit-dir D:\qenivo\audit`
2. In IIS: install URL Rewrite and Application Request Routing, enable Windows Authentication on
   the site (and disable Anonymous), and use `packaging/iis/web.config`. It forwards requests to
   QENIVO and passes the authenticated user (`LOGON_USER`, e.g. `MRPL\planner1`) in `X-Remote-User`.
   Allow the server variable `HTTP_X_REMOTE_USER` in URL Rewrite's "View Server Variables".
3. Apache or Nginx with Kerberos (`mod_auth_gssapi` / `spnego`) work the same way: authenticate,
   then set `X-Remote-User` to the authenticated principal and proxy to `127.0.0.1:8765`.
4. A Domino directory (MRPL's mail runs on HCL Domino): Apache `mod_authnz_ldap` pointed at the
   Domino LDAP service authenticates the user; then `RequestHeader set X-Remote-User
   "%{AUTHENTICATE_UID}e"` and `ProxyPass / http://127.0.0.1:8765/`.

The header is trusted only from the addresses given with `--trusted-proxy`; the same header from
any other client is rejected.

## 4. Audit trail and retention

With `--audit-dir`, every request adds one line to `audit-YYYY-MM.jsonl`: time (UTC), user, how
they were authenticated, client address, endpoint, the SHA-256 of the model text, the engine,
the verdict, the objective and the duration. Denied requests are logged too. Monthly files older
than `--retention-days` (default 180, the CERT-In requirement) are removed when the server starts.
Point a SIEM file collector at the folder if the site centralises logs.

## 5. What leaves the machine

Nothing. The code opens no outbound connection; benchmark downloads exist only in `bench/`, which
is not part of the installed package. Stack traces in error replies are sent only to local
clients.

## 6. Checking an answer independently

Every solve can write a certificate (`--cert plan.cert.json`). Anyone can re-check it, on another
machine, with the verifier alone:

```
qenivo verify plan.mps plan.cert.json            (add --exact for exact rational arithmetic)
```
