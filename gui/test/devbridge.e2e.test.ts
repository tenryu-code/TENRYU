import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { ServerProfile } from "@tenryu-common/core/profiles";
import { parseValidateOutput } from "../src/core/validateParse";
import { DevBridgeBackend } from "@tenryu-common/backend/devbridge";

const PORT = 5199;
const BASE = `http://127.0.0.1:${PORT}`;
const GUI_DIR = path.resolve(__dirname, "..");
const REPO_DIR = path.resolve(GUI_DIR, "..");

let bridge: ChildProcess;
let tmpDir: string;
let mockLog: string;

const backend = new DevBridgeBackend(BASE);

const sshProfile: ServerProfile = {
  id: "t-ssh",
  name: "mock ssh server",
  transport: "ssh",
  host: "mockhost",
  user: "mockuser",
  port: undefined,
  tenryuBin: "/opt/tenryu/build/tenryu",
  runDir: "~/tenryu_gui_runs",
};

async function waitHealth(timeoutMs: number): Promise<void> {
  const t0 = Date.now();
  for (;;) {
    try {
      const res = await fetch(`${BASE}/api/health`);
      if (res.ok) return;
    } catch {
      /* retry */
    }
    if (Date.now() - t0 > timeoutMs) throw new Error("bridge did not come up");
    await new Promise((r) => setTimeout(r, 100));
  }
}

beforeAll(async () => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "tenryu-gui-test-"));
  mockLog = path.join(tmpDir, "mock-ssh.log");
  bridge = spawn("node", [path.join(GUI_DIR, "dev-bridge", "server.mjs")], {
    env: {
      ...process.env,
      TENRYU_GUI_BRIDGE_PORT: String(PORT),
      TENRYU_GUI_BRIDGE_DIR: path.join(tmpDir, "conf"),
      TENRYU_GUI_MOCK_BIN: path.join(GUI_DIR, "dev-bridge", "mock-bin"),
      TENRYU_GUI_MOCK_LOG: mockLog,
      TENRYU_GUI_MOCK_FIXTURES: path.join(GUI_DIR, "test", "fixtures"),
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  await waitHealth(5000);
}, 20000);

afterAll(() => {
  bridge?.kill("SIGKILL");
});

describe("dev bridge", () => {
  it("profiles roundtrip", async () => {
    await backend.saveProfiles([sshProfile]);
    const got = await backend.listProfiles();
    expect(got).toEqual([sshProfile]);
  });

  it("settings roundtrip", async () => {
    await backend.saveSettings({ lastProfileId: "t-ssh" });
    const got = await backend.getSettings();
    expect(got).toEqual({ lastProfileId: "t-ssh" });
  });

  it("mock ssh validate ok end-to-end (planExec -> spawn -> parse)", async () => {
    const r = await backend.exec(sshProfile, [sshProfile.tenryuBin, "validate", "/tmp/deck.py"]);
    const parsed = parseValidateOutput(r.stdout, r.stderr, r.code);
    expect(parsed.ok).toBe(true);
    expect(parsed.summary.length).toBe(10);
    const logged = fs
      .readFileSync(mockLog, "utf8")
      .trim()
      .split("\n")
      .map((l) => JSON.parse(l));
    const last = logged[logged.length - 1];
    expect(last.argv).toContain("BatchMode=yes");
    expect(last.argv).toContain("mockuser@mockhost");
    expect(last.argv[last.argv.length - 1]).toBe(
      "/opt/tenryu/build/tenryu validate /tmp/deck.py",
    );
  });

  it("mock ssh validate failure parses TENRYU ERROR", async () => {
    const r = await backend.exec(sshProfile, [sshProfile.tenryuBin, "validate", "/tmp/bad_deck.py"]);
    expect(r.code).toBe(1);
    const parsed = parseValidateOutput(r.stdout, r.stderr, r.code);
    expect(parsed.ok).toBe(false);
    expect(parsed.errors[0]).toContain("not a supported key");
  });

  it("local transport write/read roundtrip", async () => {
    const localProfile: ServerProfile = {
      ...sshProfile,
      id: "t-local",
      transport: "local",
      tenryuBin: "/bin/true",
    };
    const p = path.join(tmpDir, "up", "deck.py");
    const content = "line1\nline2 'quoted'\n";
    await backend.uploadText(localProfile, p, content);
    const got = await backend.readText(localProfile, p);
    expect(got).toBe(content);
  });

  it("real tenryu validate via local transport (skipped when TENRYU_BIN unset)", async () => {
    const bin =
      process.env.TENRYU_BIN && fs.existsSync(process.env.TENRYU_BIN)
        ? process.env.TENRYU_BIN
        : "";
    if (!bin) {
      console.warn("[e2e] TENRYU_BIN not set or missing — skipping real validate");
      return;
    }
    const localProfile: ServerProfile = {
      ...sshProfile,
      id: "t-real",
      transport: "local",
      tenryuBin: bin,
    };
    const deck = fs.readFileSync(
      path.join(REPO_DIR, "examples", "templates", "template_1d_slab_radiation.py"),
      "utf8",
    );
    const deckPath = path.join(tmpDir, "real", "deck.py");
    await backend.uploadText(localProfile, deckPath, deck);
    const r = await backend.exec(localProfile, [bin, "validate", deckPath], { timeoutMs: 120000 });
    const parsed = parseValidateOutput(r.stdout, r.stderr, r.code);
    expect(parsed.ok).toBe(true);
    expect(parsed.summary.length).toBeGreaterThanOrEqual(8);
    expect(parsed.summary[0].label).toBe("main");
  }, 180000);
});
