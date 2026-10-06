import assert from "node:assert/strict";
import { test } from "node:test";
import { startConsole } from "../src/host.ts";

test("the console serves its pages and reports an unreachable solver", async () => {
  const server = await startConsole({ port: 0, upstream: "http://127.0.0.1:9" });
  try {
    const addr = server.address();
    assert.ok(addr && typeof addr === "object");
    const base = `http://127.0.0.1:${addr.port}`;
    const page = await fetch(`${base}/`);
    assert.equal(page.status, 200);
    const html = await page.text();
    assert.match(html, /Case library/);
    assert.match(html, /qenivo serve --host 127\.0\.0\.1 --port 8765/);
    const css = await fetch(`${base}/styles.css`);
    assert.equal(css.status, 200);
    assert.match(await css.text(), /--paper/);
    const script = await fetch(`${base}/js/browser/main.js`);
    assert.equal(script.status, 200);
    assert.match(await script.text(), /\/api\/solve/);
    const status = await fetch(`${base}/console/status`);
    const body = await status.json() as { upstream: string };
    assert.equal(body.upstream, "http://127.0.0.1:9");
    const blocked = await fetch(`${base}/%2e%2e/package.json`);
    assert.equal(blocked.status, 404);
    assert.doesNotMatch(await blocked.text(), /qenivo-console/);
    const api = await fetch(`${base}/api/info`);
    assert.equal(api.status, 502);
    const error = await api.json() as { error: string };
    assert.match(error.error, /unreachable/);
  } finally {
    server.closeAllConnections();
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
  }
});
