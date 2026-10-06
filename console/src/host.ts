import fs from "node:fs";
import http from "node:http";
import https from "node:https";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const staticRoot = path.resolve(here, "../static");

const types: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".txt": "text/plain; charset=utf-8",
};

interface HostOptions {
  port?: number;
  upstream?: string;
}

function fileFor(root: string, pathname: string): string | null {
  let decoded: string;
  try {
    decoded = decodeURIComponent(pathname);
  } catch {
    return null;
  }
  if (decoded.includes("\0")) return null;
  const abs = path.resolve(root, `.${decoded}`);
  const rootWithSep = root.endsWith(path.sep) ? root : root + path.sep;
  if (abs !== root && !abs.startsWith(rootWithSep)) return null;
  return abs === root ? path.join(root, "index.html") : abs;
}

function send(res: http.ServerResponse, status: number, body: string, contentType = "text/plain; charset=utf-8"): void {
  res.writeHead(status, { "content-type": contentType, "cache-control": "no-store" });
  res.end(body);
}

function proxy(req: http.IncomingMessage, res: http.ServerResponse, upstream: URL): void {
  const dest = new URL(req.url ?? "/", upstream);
  const lib = dest.protocol === "https:" ? https : http;
  const headers: http.OutgoingHttpHeaders = {};
  if (req.headers["content-type"]) headers["content-type"] = req.headers["content-type"];
  if (req.headers.authorization) headers.authorization = req.headers.authorization;
  if (req.headers["x-qenivo-user"]) headers["x-qenivo-user"] = req.headers["x-qenivo-user"];
  if (req.headers["content-length"]) headers["content-length"] = req.headers["content-length"];
  const preq = lib.request(dest, { method: req.method, headers }, (pres) => {
    res.writeHead(pres.statusCode ?? 502, {
      "content-type": pres.headers["content-type"] ?? "application/json",
      "cache-control": "no-store",
    });
    pres.pipe(res);
  });
  preq.on("error", (err) => {
    if (res.headersSent) return;
    const payload = JSON.stringify({
      error: `qenivo serve unreachable at ${upstream.origin}: ${err.message}`,
    });
    res.writeHead(502, {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      "content-length": Buffer.byteLength(payload),
    });
    res.end(payload);
  });
  req.pipe(preq);
}

async function handle(req: http.IncomingMessage, res: http.ServerResponse, upstream: URL): Promise<void> {
  const pathname = new URL(req.url ?? "/", "http://127.0.0.1").pathname;
  if (pathname === "/console/status" && req.method === "GET") {
    send(res, 200, JSON.stringify({ upstream: upstream.origin }), "application/json; charset=utf-8");
    return;
  }
  if (pathname.startsWith("/api/")) {
    if (req.method !== "GET" && req.method !== "POST") {
      send(res, 405, "method not allowed");
      return;
    }
    proxy(req, res, upstream);
    return;
  }
  if (req.method !== "GET" && req.method !== "HEAD") {
    send(res, 405, "method not allowed");
    return;
  }
  const abs = fileFor(staticRoot, pathname);
  if (!abs) {
    send(res, 404, "not found");
    return;
  }
  try {
    const stat = await fs.promises.stat(abs);
    if (!stat.isFile()) {
      send(res, 404, "not found");
      return;
    }
    const contentType = types[path.extname(abs).toLowerCase()] ?? "application/octet-stream";
    res.writeHead(200, { "content-type": contentType, "cache-control": "no-store" });
    if (req.method === "HEAD") {
      res.end();
      return;
    }
    fs.createReadStream(abs).pipe(res);
  } catch {
    send(res, 404, "not found");
  }
}

export function startConsole(opts: HostOptions = {}): Promise<http.Server> {
  let upstream: URL;
  const rawUpstream = opts.upstream ?? process.env.QENIVO_ORIGIN ?? "http://127.0.0.1:8765";
  try {
    upstream = new URL(rawUpstream);
  } catch {
    return Promise.reject(new Error(`QENIVO_ORIGIN is not a URL: ${rawUpstream}`));
  }
  const port = opts.port ?? Number(process.env.CONSOLE_PORT ?? 8770);
  const server = http.createServer((req, res) => {
    void handle(req, res, upstream).catch((err: unknown) => {
      if (!res.headersSent) send(res, 500, err instanceof Error ? err.message : String(err));
    });
  });
  server.requestTimeout = 0;
  server.headersTimeout = 60_000;
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => resolve(server));
  });
}

function isDirectRun(): boolean {
  const entry = process.argv[1];
  if (!entry) return false;
  try {
    return fs.realpathSync(entry) === fs.realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
}

if (isDirectRun()) {
  const bundle = path.join(staticRoot, "js", "browser", "main.js");
  if (!fs.existsSync(bundle)) {
    console.error("Missing static/js/browser/main.js. Run npm run build in qenivo/console.");
    process.exit(1);
  }
  const server = await startConsole();
  const addr = server.address();
  const port = addr && typeof addr === "object" ? addr.port : 0;
  const upstream = process.env.QENIVO_ORIGIN ?? "http://127.0.0.1:8765";
  console.log(`QENIVO planner console on http://127.0.0.1:${port}`);
  console.log(`Proxying /api to ${upstream}`);
}
