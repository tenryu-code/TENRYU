import { beforeEach, describe, expect, it } from "vitest";
import type { Backend, ExecResult } from "@tenryu-common/backend/types";
import { defaultFormState } from "../src/core/deck/formState";
import type { ServerProfile } from "@tenryu-common/core/profiles";
import { __setBackendForTest, useApp } from "../src/store";

const PROG = (step: number, pct: number) =>
  `[x] [info] [TENRYU] [progress] step=${step}/500000 t=1.000e-09/2.000e-09 dt=1.000e-12 (${pct}.0%) elapsed=0:00:01`;

function statusJson(state: string, exitCode: number | null): string {
  return JSON.stringify({
    schema: 1,
    state,
    pid: 4242,
    exit_code: exitCode,
    start_epoch: 1000,
    end_epoch: exitCode === null ? null : 1010,
    host: "fake",
    deck: "/home/fake/x/deck.py",
    log: "run.log",
  });
}

class FakeRunBackend implements Backend {
  readonly kind = "devbridge" as const;
  profiles: ServerProfile[] = [];
  settings: Record<string, unknown> = {};
  uploads: Array<{ path: string; content: string }> = [];
  saveCalls: Array<{ suggestedName: string; content: string }> = [];
  writeCalls: Array<{ path: string; content: string }> = [];
  execLog: string[][] = [];
  statusText = statusJson("running", null);
  logText = PROG(100, 20);
  launchCode = 0;
  localTextFile: { name: string; path: string | null; content: string } | null = null;

  async listProfiles() {
    return this.profiles;
  }
  async saveProfiles(p: ServerProfile[]) {
    this.profiles = p;
  }
  async getSettings() {
    return this.settings as { lastProfileId?: string };
  }
  async saveSettings(s: object) {
    this.settings = { ...s };
  }
  async saveLocalTextFile(suggestedName: string, content: string): Promise<string> {
    this.saveCalls.push({ suggestedName, content });
    return `/fake/dir/${suggestedName}`;
  }
  async writeLocalTextFile(path: string, content: string): Promise<void> {
    this.writeCalls.push({ path, content });
  }
  async openLocalTextFile(): Promise<{ name: string; path: string | null; content: string } | null> {
    return this.localTextFile;
  }
  async readBinary(): Promise<Uint8Array> {
    return new Uint8Array();
  }
  async exec(_p: ServerProfile, argv: string[]): Promise<ExecResult> {
    if (argv[0] === "bash" && argv[1] === "-lc" && argv[2].includes("mesh_planner")) return { code: 0, stdout: "", stderr: "", timedOut: false };
    this.execLog.push(argv);
    if (argv[0] === "printenv") {
      return { code: 0, stdout: "/home/fake\n", stderr: "", timedOut: false };
    }
    if (argv[0] === "bash") {
      return {
        code: this.launchCode,
        stdout: this.launchCode === 0 ? `${argv[2]}/status.json\n` : "",
        stderr: this.launchCode === 0 ? "" : "boom",
        timedOut: false,
      };
    }
    if (argv[0] === "kill") {
      return { code: 0, stdout: "", stderr: "", timedOut: false };
    }
    return { code: 0, stdout: "", stderr: "", timedOut: false };
  }
  async uploadText(_p: ServerProfile, path: string, content: string) {
    this.uploads.push({ path, content });
  }
  async readText(_p: ServerProfile, path: string): Promise<string> {
    if (path.endsWith("status.json")) return this.statusText;
    return this.logText;
  }
}

const profile: ServerProfile = {
  id: "p1",
  name: "lab",
  transport: "ssh",
  host: "example-host",
  user: "u",
  port: undefined,
  tenryuBin: "/opt/tenryu",
  runDir: "~/tenryu_gui_runs",
};

let fake: FakeRunBackend;

beforeEach(async () => {
  fake = new FakeRunBackend();
  fake.profiles = [profile];
  __setBackendForTest(fake);
  useApp.setState({ runs: [], runLogs: {}, runRates: {}, starting: false });
  await useApp.getState().loadInitial();
  useApp.getState().loadForm(defaultFormState());
});

describe("run store lifecycle", () => {
  it("startRun uploads deck+wrapper into a home-expanded runDir and launches bash", async () => {
    await useApp.getState().startRun();
    const rec = useApp.getState().runs[0];
    expect(rec.state).toBe("running");
    expect(rec.runDir.startsWith("/home/fake/tenryu_gui_runs/my_run_")).toBe(true);
    expect(fake.uploads.length).toBe(2);
    expect(fake.uploads[0].path).toBe(`${rec.runDir}/deck.py`);
    expect(fake.uploads[0].content).toContain("# TENRYU-GUI-STATE: ");
    expect(fake.uploads[1].path).toBe(`${rec.runDir}/run_detached.sh`);
    expect(fake.uploads[1].content).toContain("TENRYU Studio run wrapper");
    const bash = fake.execLog.find((a) => a[0] === "bash");
    expect(bash).toEqual(["bash", `${rec.runDir}/run_detached.sh`, rec.runDir, "/opt/tenryu", `${rec.runDir}/deck.py`]);
    expect(rec.pid).toBe(4242);
    expect(rec.lastProgress?.step).toBe(100);
  });

  it("poll transitions to finished and persists", async () => {
    await useApp.getState().startRun();
    const id = useApp.getState().runs[0].id;
    fake.statusText = statusJson("finished", 0);
    fake.logText = PROG(500, 100);
    await useApp.getState().pollRunOnce(id);
    const rec = useApp.getState().runs[0];
    expect(rec.state).toBe("finished");
    expect(rec.exitCode).toBe(0);
    expect(rec.lastProgress?.step).toBe(500);
    expect((fake.settings as { runs?: unknown[] }).runs?.length).toBe(1);
  });

  it("stopRun sends group TERM and maps 143 to stopped", async () => {
    await useApp.getState().startRun();
    const id = useApp.getState().runs[0].id;
    await useApp.getState().stopRun(id);
    expect(useApp.getState().runs[0].state).toBe("stopping");
    const kill = fake.execLog.find((a) => a[0] === "kill");
    expect(kill).toEqual(["kill", "-TERM", "--", "-4242"]);
    fake.statusText = statusJson("failed", 143);
    await useApp.getState().pollRunOnce(id);
    expect(useApp.getState().runs[0].state).toBe("stopped");
    expect(useApp.getState().runs[0].exitCode).toBe(143);
  });

  it("launch failure records failed + launchError", async () => {
    fake.launchCode = 2;
    await useApp.getState().startRun();
    const rec = useApp.getState().runs[0];
    expect(rec.state).toBe("failed");
    expect(rec.launchError).toContain("boom");
  });
});
