import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { parseLastProgress } from "../src/core/progressParse";
import { parseStatusFile } from "@tenryu-common/core/runstate";

const GUI_DIR = path.resolve(__dirname, "..");
const SCRIPT = path.join(GUI_DIR, "src", "core", "assets", "run_detached.sh");
const MOCK_TENRYU = path.join(GUI_DIR, "dev-bridge", "mock-bin", "tenryu");

function launch(runDir: string, env: NodeJS.ProcessEnv): string {
  const out = execFileSync("bash", [SCRIPT, runDir, MOCK_TENRYU, path.join(runDir, "deck.py")], {
    encoding: "utf8",
    env: { ...process.env, ...env },
    timeout: 20000,
  });
  return out.trim();
}

async function waitTerminal(statusPath: string, timeoutMs: number) {
  const t0 = Date.now();
  for (;;) {
    const st = parseStatusFile(fs.readFileSync(statusPath, "utf8"));
    if (st && st.state !== "running") return st;
    if (Date.now() - t0 > timeoutMs) throw new Error("run did not finish in time");
    await new Promise((r) => setTimeout(r, 100));
  }
}

function mkRun(namePrefix: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), namePrefix));
  fs.writeFileSync(path.join(dir, "deck.py"), "# mock deck\n");
  return dir;
}

describe("run_detached.sh contract (local bash + mock tenryu)", () => {
  it("finished path: running -> finished with progress in log", async () => {
    const dir = mkRun("tenryu-rd-ok-");
    const statusPath = launch(dir, { TENRYU_MOCK_STEPS: "3", TENRYU_MOCK_SLEEP_MS: "50" });
    expect(statusPath).toBe(path.join(dir, "status.json"));
    const first = parseStatusFile(fs.readFileSync(statusPath, "utf8"));
    expect(first?.state).toBe("running");
    expect(first && first.pid > 0).toBe(true);
    const fin = await waitTerminal(statusPath, 15000);
    expect(fin.state).toBe("finished");
    expect(fin.exit_code).toBe(0);
    expect(fin.end_epoch).not.toBeNull();
    const log = fs.readFileSync(path.join(dir, "run.log"), "utf8");
    const prog = parseLastProgress(log);
    expect(prog?.pct).toBeCloseTo(100.0, 3);
  }, 30000);

  it("failure path: nonzero exit -> failed with exit_code", async () => {
    const dir = mkRun("tenryu-rd-fail-");
    const statusPath = launch(dir, {
      TENRYU_MOCK_STEPS: "4",
      TENRYU_MOCK_SLEEP_MS: "30",
      TENRYU_MOCK_FAIL: "1",
    });
    const fin = await waitTerminal(statusPath, 15000);
    expect(fin.state).toBe("failed");
    expect(fin.exit_code).toBe(9);
    const log = fs.readFileSync(path.join(dir, "run.log"), "utf8");
    expect(log).toContain("TENRYU ERROR [mock]: injected failure");
  }, 30000);

  it("stop path: kill -TERM -- -pid ends as failed(143) via the child trap", async () => {
    const dir = mkRun("tenryu-rd-stop-");
    const statusPath = launch(dir, { TENRYU_MOCK_STEPS: "100", TENRYU_MOCK_SLEEP_MS: "200" });
    const st = parseStatusFile(fs.readFileSync(statusPath, "utf8"));
    expect(st?.state).toBe("running");
    await new Promise((r) => setTimeout(r, 500));
    const kill = spawnSync("kill", ["-TERM", "--", `-${st!.pid}`], { encoding: "utf8" });
    expect(kill.status).toBe(0);
    const fin = await waitTerminal(statusPath, 15000);
    expect(fin.state).toBe("failed");
    expect(fin.exit_code).toBe(143);
  }, 30000);
});
