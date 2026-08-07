// TENRYU Studio dev bridge — Linux development stand-in for the Tauri shell.
// Node stdlib only. Binds 127.0.0.1 only. NOT for production use.
import { spawn } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";

const PORT = Number(process.env.TENRYU_GUI_BRIDGE_PORT || 5175);
const HOST = "127.0.0.1";
const CONF_DIR =
  process.env.TENRYU_GUI_BRIDGE_DIR || path.join(os.homedir(), ".config", "tenryu-studio-dev");
const MOCK_BIN = process.env.TENRYU_GUI_MOCK_BIN || "";
const MAX_BODY = 8 * 1024 * 1024;
const MAX_CAPTURE = 4 * 1024 * 1024;

fs.mkdirSync(CONF_DIR, { recursive: true });

function expandHome(p) {
  if (p === "~") return os.homedir();
  if (p.startsWith("~/")) return path.join(os.homedir(), p.slice(2));
  return p;
}

function readJsonFile(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return fallback;
  }
}

function originAllowed(req) {
  const origin = req.headers.origin;
  if (!origin) return true;
  return /^http:\/\/(localhost|127\.0\.0\.1):\d+$/.test(origin);
}

function sendJson(res, status, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(status, { "content-type": "application/json" });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on("data", (c) => {
      size += c.length;
      if (size > MAX_BODY) {
        reject(new Error("body too large"));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function handleSpawn(body, res) {
  const { argv, stdin, timeoutMs } = body;
  if (!Array.isArray(argv) || argv.length === 0 || !argv.every((a) => typeof a === "string")) {
    sendJson(res, 400, { error: "argv must be a non-empty string array" });
    return;
  }
  const env = { ...process.env };
  if (MOCK_BIN) env.PATH = `${MOCK_BIN}:${env.PATH ?? ""}`;
  const child = spawn(argv[0], argv.slice(1), { env, stdio: ["pipe", "pipe", "pipe"] });
  let stdout = "";
  let stderr = "";
  let timedOut = false;
  let settled = false;
  const limit = Math.min(Math.max(Number(timeoutMs) || 120000, 1000), 600000);
  const timer = setTimeout(() => {
    timedOut = true;
    child.kill("SIGKILL");
  }, limit);
  child.stdout.on("data", (c) => {
    if (stdout.length < MAX_CAPTURE) stdout += c.toString("utf8");
  });
  child.stderr.on("data", (c) => {
    if (stderr.length < MAX_CAPTURE) stderr += c.toString("utf8");
  });
  child.on("error", (err) => {
    if (settled) return;
    settled = true;
    clearTimeout(timer);
    sendJson(res, 200, { code: null, stdout, stderr: String(err), timedOut });
  });
  child.on("close", (code) => {
    if (settled) return;
    settled = true;
    clearTimeout(timer);
    sendJson(res, 200, { code, stdout, stderr, timedOut });
  });
  if (typeof stdin === "string" && stdin.length > 0) {
    child.stdin.write(stdin);
  }
  child.stdin.end();
}

const server = http.createServer(async (req, res) => {
  if (!originAllowed(req)) {
    sendJson(res, 403, { error: "origin not allowed" });
    return;
  }
  const url = new URL(req.url ?? "/", `http://${HOST}:${PORT}`);
  const route = `${req.method} ${url.pathname}`;
  console.log(`[bridge] ${route}`);
  try {
    if (route === "GET /api/health") {
      sendJson(res, 200, { ok: true, version: 1 });
      return;
    }
    if (route === "GET /api/profiles") {
      sendJson(res, 200, readJsonFile(path.join(CONF_DIR, "profiles.json"), { profiles: [] }));
      return;
    }
    if (route === "PUT /api/profiles") {
      const body = JSON.parse(await readBody(req));
      if (!Array.isArray(body.profiles)) {
        sendJson(res, 400, { error: "profiles must be an array" });
        return;
      }
      fs.writeFileSync(
        path.join(CONF_DIR, "profiles.json"),
        JSON.stringify({ profiles: body.profiles }, null, 2),
      );
      sendJson(res, 200, { ok: true });
      return;
    }
    if (route === "GET /api/settings") {
      sendJson(res, 200, readJsonFile(path.join(CONF_DIR, "settings.json"), { settings: {} }));
      return;
    }
    if (route === "PUT /api/settings") {
      const body = JSON.parse(await readBody(req));
      fs.writeFileSync(
        path.join(CONF_DIR, "settings.json"),
        JSON.stringify({ settings: body.settings ?? {} }, null, 2),
      );
      sendJson(res, 200, { ok: true });
      return;
    }
    if (route === "POST /api/spawn") {
      const body = JSON.parse(await readBody(req));
      handleSpawn(body, res);
      return;
    }
    if (route === "POST /api/write-file") {
      const body = JSON.parse(await readBody(req));
      if (typeof body.path !== "string" || typeof body.content !== "string") {
        sendJson(res, 400, { error: "path and content are required" });
        return;
      }
      const p = expandHome(body.path);
      if (!path.isAbsolute(p)) {
        sendJson(res, 400, { error: "path must be absolute (or ~/...)" });
        return;
      }
      fs.mkdirSync(path.dirname(p), { recursive: true });
      fs.writeFileSync(p, body.content);
      sendJson(res, 200, { ok: true });
      return;
    }
    if (route === "POST /api/read-file") {
      const body = JSON.parse(await readBody(req));
      if (typeof body.path !== "string") {
        sendJson(res, 400, { error: "path is required" });
        return;
      }
      const p = expandHome(body.path);
      if (!path.isAbsolute(p)) {
        sendJson(res, 400, { error: "path must be absolute (or ~/...)" });
        return;
      }
      const maxBytes = Math.min(Math.max(Number(body.maxBytes) || 262144, 1), MAX_CAPTURE);
      const stat = fs.statSync(p);
      const start = Math.max(0, stat.size - maxBytes);
      const fd = fs.openSync(p, "r");
      const buf = Buffer.alloc(Math.min(maxBytes, stat.size));
      fs.readSync(fd, buf, 0, buf.length, start);
      fs.closeSync(fd);
      sendJson(res, 200, { content: buf.toString("utf8"), size: stat.size });
      return;
    }
    sendJson(res, 404, { error: `no route: ${route}` });
  } catch (err) {
    sendJson(res, 500, { error: String(err) });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`[bridge] listening on http://${HOST}:${PORT} (conf: ${CONF_DIR})`);
  if (MOCK_BIN) console.log(`[bridge] MOCK BIN PATH: ${MOCK_BIN}`);
});
