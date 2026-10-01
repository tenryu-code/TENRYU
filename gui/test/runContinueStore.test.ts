import { beforeEach, describe, expect, it } from "vitest";
import type { AppSettings, Backend, ExecResult } from "@tenryu-common/backend/types";
import type { ServerProfile } from "@tenryu-common/core/profiles";
import type { RunRecord } from "@tenryu-common/core/runstate";
import { defaultFormState } from "../src/core/deck/formState";
import { t } from "../src/i18n";
import { __setBackendForTest, useApp } from "../src/store";

// The run history's "Continue": restart from the latest checkpoint with the unchanged deck, passing a later end
// time or step limit as `tenryu run --t-end / --max-steps` through run_detached.sh.

const RUN = "/home/fake/tenryu_gui_runs/case_20260930";
const ok = (stdout = ""): ExecResult => ({ code: 0, stdout, stderr: "", timedOut: false });

class FakeBackend implements Backend {
  readonly kind = "devbridge" as const;
  profiles: ServerProfile[] = [];
  settings: Record<string, unknown> = {};
  uploads: Array<{ path: string; content: string }> = [];
  execLog: string[][] = [];
  checkpointListing = "";
  launchCode = 0;

  async listProfiles() {
    return this.profiles;
  }
  async saveProfiles(p: ServerProfile[]) {
    this.profiles = p;
  }
  async getSettings() {
    return this.settings as AppSettings;
  }
  async saveSettings(s: object) {
    this.settings = { ...s };
  }
  async saveLocalTextFile() {
    return null;
  }
  async writeLocalTextFile() {}
  async openLocalTextFile() {
    return null;
  }
  async readBinary() {
    return new Uint8Array();
  }
  async execLocal() {
    return ok();
  }
  async readLocalText(): Promise<string> {
    throw new Error("no local files");
  }
  async writeLocalText() {}
  async appConfigDir() {
    return "/home/fake/appconfig";
  }
  async assistHarnessDir() {
    return "/app/resources/tools/assist";
  }
  async exec(_p: ServerProfile, argv: string[]): Promise<ExecResult> {
    this.execLog.push(argv);
    if (argv[0] === "printenv") return ok("/home/fake\n");
    if (argv[0] === "bash" && argv[1] === "-lc" && argv[2].includes("_ckpt_")) return ok(this.checkpointListing);
    if (argv[0] === "bash" && argv[1].endsWith("run_detached.sh")) {
      return this.launchCode === 0
        ? ok(`${argv[2]}/status.json\n`)
        : { code: this.launchCode, stdout: "", stderr: "boom", timedOut: false };
    }
    return ok();
  }
  async uploadText(_p: ServerProfile, path: string, content: string) {
    this.uploads.push({ path, content });
  }
  async readText(_p: ServerProfile, path: string): Promise<string> {
    if (path.endsWith("status.json")) {
      return JSON.stringify({
        schema: 1,
        state: "running",
        pid: 77,
        exit_code: null,
        start_epoch: 1020,
        end_epoch: null,
        deck: `${RUN}/deck.py`,
        log: "run.log",
      });
    }
    return "";
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

function finishedRun(patch: Partial<RunRecord> = {}): RunRecord {
  return {
    id: "r1",
    profileId: profile.id,
    profileName: profile.name,
    name: "case_20260930",
    runDir: RUN,
    statusPath: `${RUN}/status.json`,
    tEnd: 2e-9,
    maxSteps: 500000,
    createdAtIso: "2026-09-30T00:00:00.000Z",
    state: "finished",
    pid: 4242,
    exitCode: 0,
    stopRequested: false,
    lastProgress: { step: 1100, t: 2e-9, pct: 100 },
    startEpoch: 1000,
    endEpoch: 1010,
    launchError: null,
    ...patch,
  };
}

let fake: FakeBackend;

const launches = () => fake.execLog.filter((argv) => argv[0] === "bash" && argv[1].endsWith("run_detached.sh"));
const run = () => useApp.getState().runs[0];

beforeEach(async () => {
  fake = new FakeBackend();
  fake.profiles = [profile];
  __setBackendForTest(fake);
  useApp.setState({ runs: [], runLogs: {}, runRates: {}, starting: false });
  await useApp.getState().loadInitial();
  useApp.getState().loadForm(defaultFormState());
  useApp.setState({ runs: [finishedRun()], histories: {}, profileLists: {}, profileSnaps: {} });
});

describe("continuing a finished run from the run history", () => {
  it("restarts from the latest directory's last checkpoint and passes the new end time and step limit", async () => {
    fake.checkpointListing = [
      `${RUN}/outputs/case/checkpoints/case_ckpt_0004.h5`,
      `${RUN}/outputs/case_001/checkpoints/case_ckpt_0001.h5`,
      `${RUN}/outputs/case_001/checkpoints/case_ckpt_0000.h5`,
      "",
    ].join("\n");
    useApp.setState({
      histories: { r1: { status: "ready" } },
      profileLists: { r1: { status: "ready", paths: [] } },
      profileSnaps: { "r1:0": { status: "ready" }, "r2:0": { status: "ready" } },
    });
    await useApp.getState().continueRun("r1", { tEndS: 4e-9, maxSteps: 2000 });

    // The wrapper is uploaded again: the one of the original launch may not pass tenryu run arguments.
    const wrapper = fake.uploads.find((u) => u.path === `${RUN}/run_detached.sh`);
    expect(wrapper?.content).toContain("<tenryu run args>");
    expect(launches()).toEqual([
      [
        "bash",
        `${RUN}/run_detached.sh`,
        RUN,
        "/opt/tenryu",
        `${RUN}/deck.py`,
        `${RUN}/outputs/case_001/checkpoints/case_ckpt_0001`,
        "--t-end",
        "4e-9",
        "--max-steps",
        "2000",
      ],
    ]);
    expect(run()).toMatchObject({ state: "running", tEnd: 4e-9, maxSteps: 2000, launchError: null, pid: 77 });
    // The history and snapshot lists are read again, including the continuation's directory.
    const state = useApp.getState();
    expect(state.histories.r1).toBeUndefined();
    expect(state.profileLists.r1).toBeUndefined();
    expect(state.profileSnaps["r1:0"]).toBeUndefined();
    expect(state.profileSnaps["r2:0"]).toBeDefined();
  });

  it("restartRun restarts with the deck's values", async () => {
    fake.checkpointListing = `${RUN}/outputs/case/checkpoints/case_ckpt_0000.h5\n`;
    await useApp.getState().restartRun("r1");
    expect(launches()).toEqual([
      [
        "bash",
        `${RUN}/run_detached.sh`,
        RUN,
        "/opt/tenryu",
        `${RUN}/deck.py`,
        `${RUN}/outputs/case/checkpoints/case_ckpt_0000`,
      ],
    ]);
    expect(run()).toMatchObject({ state: "running", tEnd: 2e-9, maxSteps: 500000 });
  });

  it("a run without checkpoints keeps its state and says why", async () => {
    fake.checkpointListing = "";
    await useApp.getState().continueRun("r1", { tEndS: 4e-9, maxSteps: null });
    expect(launches()).toEqual([]);
    expect(run()).toMatchObject({ state: "finished", tEnd: 2e-9, launchError: t().run.noCheckpoints });
  });

  it("refuses an end time the run has already reached before contacting the server", async () => {
    await useApp.getState().continueRun("r1", { tEndS: 1e-9, maxSteps: null });
    expect(fake.execLog).toEqual([]);
    expect(run().state).toBe("finished");
    expect(run().launchError).toBe(t().run.continueIssues.tEndNotLater);
  });

  it("a failed launch keeps the run's state", async () => {
    fake.checkpointListing = `${RUN}/outputs/case/checkpoints/case_ckpt_0000.h5\n`;
    fake.launchCode = 2;
    await useApp.getState().continueRun("r1", { tEndS: 4e-9, maxSteps: null });
    expect(run()).toMatchObject({ state: "finished", tEnd: 2e-9, launchError: "boom" });
  });

  it("does nothing for a run that is still running", async () => {
    useApp.setState({ runs: [finishedRun({ state: "running", endEpoch: null })] });
    await useApp.getState().continueRun("r1", { tEndS: 4e-9, maxSteps: null });
    expect(fake.execLog).toEqual([]);
    expect(run().state).toBe("running");
  });
});
