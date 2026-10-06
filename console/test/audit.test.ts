import assert from "node:assert/strict";
import { test } from "node:test";
import { parseAuditLog } from "../static/js/lib/audit.js";

const sha = "a".repeat(64);
const text = [
  `{"utc":"2026-09-30T09:00:00+00:00","event":"request","path":"/api/solve","user":"local:planner","auth":"local","client":"127.0.0.1","model_sha256":"${sha}","engine_requested":"auto","status":200,"verdict":"optimal","objective":1.5,"engine":"simplex","seconds":0.02}`,
  "",
  `{"utc":"2026-09-30T09:00:01+00:00","event":"denied","path":"/api/solve","client":"10.1.2.3","reason":"missing or wrong token"}`,
  "not json",
].join("\n");

test("parseAuditLog reads the server audit line and counts bad lines", () => {
  const log = parseAuditLog(text);
  assert.equal(log.skipped, 1);
  assert.equal(log.records.length, 2);
  const request = log.records[0];
  assert.equal(request.event, "request");
  assert.equal(request.user, "local:planner");
  assert.equal(request.auth, "local");
  assert.equal(request.verdict, "optimal");
  assert.equal(request.objective, 1.5);
  assert.equal(request.engine, "simplex");
  assert.equal(request.model_sha256, sha);
  assert.equal(request.seconds, 0.02);
  assert.equal(request.status, 200);
  const denied = log.records[1];
  assert.equal(denied.event, "denied");
  assert.equal(denied.reason, "missing or wrong token");
  assert.equal(denied.verdict, null);
  assert.equal(denied.objective, null);
});

test("an error response keeps the error string", () => {
  const log = parseAuditLog('{"utc":"2026-09-30T09:00:02+00:00","event":"request","path":"/api/explain","status":400,"error":"ValueError: bad model","seconds":0.01}\n');
  assert.equal(log.records[0].error, "ValueError: bad model");
  assert.equal(log.records[0].status, 400);
  assert.equal(log.skipped, 0);
});
