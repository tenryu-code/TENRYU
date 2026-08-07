import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { describe, expect, it } from "vitest";
import type { ServerProfile } from "@tenryu-common/core/profiles";
import { planExec, planUploadText, shQuotePath } from "@tenryu-common/core/ssh";

const HOST = process.env.TENRYU_SSH_E2E ?? "";
const execFileAsync = promisify(execFile);

const profile: ServerProfile = {
  id: "e2e",
  name: "e2e",
  transport: "ssh",
  host: HOST,
  user: undefined,
  port: undefined,
  tenryuBin: "/bin/true",
  runDir: "~/gui_e2e_tilde",
};

async function runPlan(
  plan: ReturnType<typeof planExec>,
): Promise<{ code: number; stdout: string; stderr: string }> {
  try {
    const { stdout, stderr } = await execFileAsync(plan.argv[0], plan.argv.slice(1), {
      timeout: 20000,
    });
    return { code: 0, stdout: String(stdout), stderr: String(stderr) };
  } catch (err) {
    const e = err as Error & {
      code?: number;
      stdout?: string | Buffer;
      stderr?: string | Buffer;
    };
    return {
      code: e.code ?? 1,
      stdout: String(e.stdout ?? ""),
      stderr: String(e.stderr ?? ""),
    };
  }
}

describe.skipIf(HOST === "")("real ssh tilde expansion", () => {
  it("tilde runDir round-trip stays in the expanded home", async () => {
    const setup = await runPlan(
      planExec(profile, [
        "bash",
        "-lc",
        "rm -rf " +
          shQuotePath("~/gui_e2e_tilde") +
          " && mkdir -p " +
          shQuotePath("~/gui_e2e_tilde/results"),
      ]),
    );
    expect(setup.code).toBe(0);

    const upload = planUploadText(profile, "~/gui_e2e_tilde/results/marker.h5", "x");
    expect(upload.kind).toBe("exec");
    if (upload.kind !== "exec") return;
    const uploaded = await runPlan(upload);
    expect(uploaded.code).toBe(0);

    const found = await runPlan(
      planExec(profile, [
        "bash",
        "-lc",
        "ls -t " +
          shQuotePath("~/gui_e2e_tilde" + "/results") +
          "/*.h5 2>/dev/null | head -5",
      ]),
    );
    expect(found.code).toBe(0);
    const first = found.stdout.split("\n")[0];
    expect(first.endsWith("/gui_e2e_tilde/results/marker.h5")).toBe(true);
    expect(first.startsWith("/")).toBe(true);
    expect(first).not.toContain("~");

    const literal = await runPlan(
      planExec(profile, [
        "bash",
        "-lc",
        "test -e './~/gui_e2e_tilde' && echo LITERAL || echo CLEAN",
      ]),
    );
    expect(literal.code).toBe(0);
    expect(literal.stdout).toContain("CLEAN");

    const cleanup = await runPlan(
      planExec(profile, ["bash", "-lc", "rm -rf " + shQuotePath("~/gui_e2e_tilde")]),
    );
    expect(cleanup.code).toBe(0);
  }, 20000);

  it("argv tilde args expand remotely", async () => {
    const result = await runPlan(planExec(profile, ["ls", "-d", "~/."]));
    expect(result.code).toBe(0);
    expect(result.stdout.split("\n")[0].startsWith("/")).toBe(true);
  }, 20000);
});
