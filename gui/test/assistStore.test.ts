import { beforeEach, describe, expect, it } from "vitest";
import type { AppSettings, Backend, ExecResult } from "@tenryu-common/backend/types";
import type { ServerProfile } from "@tenryu-common/core/profiles";
import type { RunRecord } from "@tenryu-common/core/runstate";
import { defaultFormState } from "../src/core/deck/formState";
import { __setBackendForTest, useApp } from "../src/store";

const SSH_PROFILE: ServerProfile = {
  id: "ssh",
  name: "GPU",
  transport: "ssh",
  host: "gpu",
  tenryuBin: "~/repo/build/tenryu",
  runDir: "~/gui_runs",
};

const LOCAL_PROFILE: ServerProfile = {
  id: "local",
  name: "Local",
  transport: "local",
  host: "",
  tenryuBin: "/opt/tenryu/build/tenryu",
  runDir: "/tmp/runs",
};

const LINT_JSON = JSON.stringify({
  schema: "tenryu.assist.decklint.v0",
  validate: { exit_code: 0, ok: true, stderr_tail: "" },
  lints: [{ id: "node-monotonic", severity: "hard", ok: true }],
});

const DIGEST_JSON = JSON.stringify({
  run: { terminated_normally: true },
  frozen_config: { dimension: "1D_SPH" },
  derived: { bang_time_proxy_s: 1e-9 },
  history: { n_samples: 2, missing: [], series: {} },
  notes: [],
});

function result(code: number, stdout = "", stderr = ""): ExecResult {
  return { code, stdout, stderr, timedOut: false };
}

function statusJson(): string {
  return JSON.stringify({
    schema: 1,
    state: "running",
    pid: 4242,
    exit_code: null,
    start_epoch: 1000,
    end_epoch: null,
    deck: "/tmp/deck.py",
    log: "run.log",
  });
}

class FakeAssistBackend implements Backend {
  readonly kind = "devbridge" as const;
  profiles: ServerProfile[] = [];
  settings: AppSettings = {};
  uploads: Array<{ path: string; content: string }> = [];
  localWrites: Array<{ path: string; content: string }> = [];
  execLog: string[][] = [];
  localExecLog: string[][] = [];
  localTextFiles = new Map<string, string>();
  remoteRoot = "/srv/tenryu";
  remoteProbeOk = true;
  localProbeOk = true;
  lintStdout = LINT_JSON;
  lintCode = 0;
  latestDirStdout = "/home/remote/gui_runs/x/outputs/20260902/\n";
  digestStdout = DIGEST_JSON;
  statusStdout = "{}";
  generateResults: Array<{ code: number; stdout: string; stderr?: string }> = [];
  hangGenerate = false;

  async listProfiles(): Promise<ServerProfile[]> {
    return this.profiles;
  }

  async saveProfiles(profiles: ServerProfile[]): Promise<void> {
    this.profiles = profiles;
  }

  async getSettings(): Promise<AppSettings> {
    return this.settings;
  }

  async saveSettings(settings: AppSettings): Promise<void> {
    this.settings = { ...settings };
  }

  async exec(_profile: ServerProfile, argv: string[]): Promise<ExecResult> {
    this.execLog.push(argv);
    const script = argv[2] ?? "";
    if (argv[0] === "printenv" && argv[1] === "HOME") {
      return result(0, "/home/remote\n");
    }
    if (script.includes("mesh_planner")) {
      return result(0, `${this.remoteRoot}\n`);
    }
    if (script.includes("ASSIST_OK")) {
      return result(0, this.remoteProbeOk ? "ASSIST_OK\n" : "");
    }
    if (script.includes("lint-deck")) {
      return result(this.lintCode, this.lintStdout, this.lintCode === 0 ? "" : "lint failed");
    }
    if (script.includes("ls -td")) {
      return result(0, this.latestDirStdout);
    }
    if (script.includes("digest")) {
      return result(0, this.digestStdout);
    }
    if (script.includes("mkdir -p") && script.includes("printf")) {
      return result(0);
    }
    const wrapperIndex = argv.findIndex((arg) => arg.endsWith("run_detached.sh"));
    if (wrapperIndex >= 0) {
      return result(0, `${argv[wrapperIndex + 1]}/status.json\n`);
    }
    return result(0);
  }

  async uploadText(_profile: ServerProfile, path: string, content: string): Promise<void> {
    this.uploads.push({ path, content });
  }

  async readText(_profile: ServerProfile, path: string): Promise<string> {
    if (path.endsWith("status.json")) return statusJson();
    return "";
  }

  async saveLocalTextFile(suggestedName: string): Promise<string> {
    return `/tmp/${suggestedName}`;
  }

  async writeLocalTextFile(): Promise<void> {}

  async openLocalTextFile(): Promise<null> {
    return null;
  }

  async readBinary(): Promise<Uint8Array> {
    return new Uint8Array();
  }

  async execLocal(argv: string[]): Promise<ExecResult> {
    this.localExecLog.push(argv);
    const script = argv[2] ?? "";
    if (script.includes('echo "$HOME"')) {
      return result(0, "/home/fake\n");
    }
    if (script.includes("ASSIST_OK")) {
      return result(0, this.localProbeOk ? "ASSIST_OK\n" : "");
    }
    if (script.includes("assist.py status")) {
      return result(0, this.statusStdout);
    }
    if (script.includes("generate-deck")) {
      const workdir = /--workdir (\S+)/.exec(script)?.[1];
      if (workdir === undefined) throw new Error("generate script has no workdir");
      this.localTextFiles.set(`${workdir}/out_deck.py`, "# generated\nt_end = 1.0e-9\n");
      if (this.hangGenerate) return new Promise<ExecResult>(() => {});
      const next = this.generateResults.shift() ?? {
        code: 0,
        stdout: JSON.stringify({ status: "accepted", iterations: 1 }),
      };
      return result(next.code, next.stdout, next.stderr ?? "");
    }
    if (script.includes("pkill")) {
      return result(0);
    }
    return result(0);
  }

  async readLocalText(path: string): Promise<string> {
    const text = this.localTextFiles.get(path);
    if (text === undefined) throw new Error(`missing local file: ${path}`);
    return text;
  }

  async writeLocalText(path: string, content: string): Promise<void> {
    this.localTextFiles.set(path, content);
    this.localWrites.push({ path, content });
  }
}

function terminalRun(id = "run-1"): RunRecord {
  return {
    id,
    profileId: SSH_PROFILE.id,
    profileName: SSH_PROFILE.name,
    name: "x",
    runDir: "/home/remote/gui_runs/x",
    statusPath: "/home/remote/gui_runs/x/status.json",
    tEnd: 1e-9,
    maxSteps: 10,
    createdAtIso: "2026-09-02T00:00:00.000Z",
    state: "finished",
    pid: 1,
    exitCode: 0,
    stopRequested: false,
    lastProgress: null,
    startEpoch: 1,
    endEpoch: 2,
    launchError: null,
  };
}

let fake: FakeAssistBackend;

async function resetStore(profile: ServerProfile = SSH_PROFILE): Promise<void> {
  fake = new FakeAssistBackend();
  fake.profiles = [profile];
  fake.settings = { assistLocalRepo: "/local/tenryu" };
  __setBackendForTest(fake);
  useApp.setState({
    profiles: [profile],
    profilesLoaded: true,
    loadError: null,
    currentProfileId: profile.id,
    runs: [],
    runLogs: {},
    runRates: {},
    starting: false,
    assistLocalRepo: "",
    assistStatus: { status: "idle" },
    assistLint: { status: "idle" },
    assistSpec: "",
    assistUseTemplate: true,
    assistMaxIters: 10,
    assistIntentJson: "",
    assistGen: {
      phase: "idle",
      workdir: null,
      iterations: 0,
      lastKind: null,
      question: null,
      deckText: null,
      deckName: null,
      errorCode: null,
      errorDetail: null,
      resultRaw: null,
      lint: null,
    },
    assistDiag: {},
    assistDeckValidate: { status: "idle" },
  });
  await useApp.getState().loadInitial();
  useApp.getState().loadForm(defaultFormState());
}

beforeEach(async () => {
  await resetStore();
});

describe("fetchAssistStatus", () => {
  it("reports NO_LOCAL_REPO for an empty setting", async () => {
    useApp.setState({ assistLocalRepo: "" });
    await useApp.getState().fetchAssistStatus();
    expect(useApp.getState().assistStatus).toEqual({
      status: "error",
      error: "NO_LOCAL_REPO",
    });
  });

  it("reports NO_LOCAL_ASSIST when the local probe fails", async () => {
    fake.localProbeOk = false;
    await useApp.getState().fetchAssistStatus();
    expect(useApp.getState().assistStatus.error).toBe("NO_LOCAL_ASSIST");
  });

  it("parses a resolved status payload", async () => {
    fake.statusStdout = JSON.stringify({
      enabled: false,
      disabled_by: null,
      providers: { codex: { model: "gpt-5.6" } },
      roles: { deck_design: "codex" },
    });
    await useApp.getState().fetchAssistStatus();
    expect(useApp.getState().assistStatus.status).toBe("ready");
    expect(useApp.getState().assistStatus.view?.enabled).toBe(false);
    expect(useApp.getState().assistStatus.view?.roles).toHaveLength(1);
  });
});

describe("setAssistLocalRepo", () => {
  it("persists the local repository path", async () => {
    await useApp.getState().setAssistLocalRepo("/new/tenryu");
    expect(fake.settings.assistLocalRepo).toBe("/new/tenryu");
  });
});

describe("runAssistLint", () => {
  it("uploads an absolute deck path and parses lint output", async () => {
    await useApp.getState().runAssistLint();
    const lint = useApp.getState().assistLint;
    expect(lint.status).toBe("ready");
    expect(lint.exitOk).toBe(true);
    expect(lint.view?.lints).toHaveLength(1);
    expect(
      fake.execLog.some(
        (argv) => argv[0] === "bash" && argv[1] === "-lc" &&
          argv[2].includes("lint-deck") && argv[2].includes("--tenryu"),
      ),
    ).toBe(true);
    expect(fake.uploads[0].path.startsWith("/home/remote/")).toBe(true);
  });

  it("reports NO_ASSIST when the remote probe fails", async () => {
    fake.remoteProbeOk = false;
    await useApp.getState().runAssistLint();
    expect(useApp.getState().assistLint.error).toBe("NO_ASSIST");
  });
});

describe("runAssistDiag digest", () => {
  it("looks up the latest output before running digest", async () => {
    const rec = terminalRun();
    useApp.setState({ runs: [rec] });
    await useApp.getState().runAssistDiag(rec.id, "digest");
    expect(useApp.getState().assistDiag[rec.id].digest?.status).toBe("ready");
    const latestIndex = fake.execLog.findIndex((argv) => (argv[2] ?? "").includes("ls -td"));
    const digestIndex = fake.execLog.findIndex((argv) => (argv[2] ?? "").includes("assist.py digest"));
    expect(latestIndex).toBeGreaterThanOrEqual(0);
    expect(digestIndex).toBeGreaterThan(latestIndex);
  });

  it("reports NO_OUTPUTS when the latest output lookup is empty", async () => {
    const rec = terminalRun();
    useApp.setState({ runs: [rec] });
    fake.latestDirStdout = "";
    await useApp.getState().runAssistDiag(rec.id, "digest");
    expect(useApp.getState().assistDiag[rec.id].digest?.error).toBe("NO_OUTPUTS");
  });
});

describe("generateAssistDeck", () => {
  it("completes an accepted lifecycle with a local profile", async () => {
    await resetStore(LOCAL_PROFILE);
    useApp.setState({ assistSpec: "Create a small 1D deck." });
    fake.generateResults.push({
      code: 0,
      stdout: JSON.stringify({ status: "accepted", iterations: 2 }),
    });
    await useApp.getState().generateAssistDeck();
    const gen = useApp.getState().assistGen;
    expect(gen.phase).toBe("accepted");
    expect(gen.deckText).toBe("# generated\nt_end = 1.0e-9\n");
    expect(gen.deckName?.startsWith("assist_")).toBe(true);
    const script = fake.localExecLog.find((argv) => (argv[2] ?? "").includes("generate-deck"))?.[2] ?? "";
    expect(script).toContain("--tenryu /opt/tenryu/build/tenryu");
    expect(script).not.toContain("&& env ");
    expect(fake.localWrites.find((write) => write.path.endsWith("/spec.md"))?.content)
      .toBe("Create a small 1D deck.");
  });

  it("composes the remote-wrapper environment for an ssh profile", async () => {
    useApp.setState({ assistSpec: "Create a remote 1D deck." });
    await useApp.getState().generateAssistDeck();
    const script = fake.localExecLog.find((argv) => (argv[2] ?? "").includes("generate-deck"))?.[2] ?? "";
    expect(script).toContain("TENRYU_REMOTE_HOST=gpu");
    expect(script).toContain("TENRYU_REMOTE_BIN=/home/remote/repo/build/tenryu");
    expect(script).toContain("--tenryu tools/assist/tenryu_remote.sh");
  });

  it("appends clarification and relaunches after an uncertain result", async () => {
    await resetStore(LOCAL_PROFILE);
    useApp.setState({ assistSpec: "Create a deck." });
    fake.generateResults.push(
      {
        code: 0,
        stdout: JSON.stringify({
          status: "uncertain",
          question: "What radius?",
          iterations: 1,
        }),
      },
      {
        code: 0,
        stdout: JSON.stringify({ status: "accepted", iterations: 2 }),
      },
    );
    await useApp.getState().generateAssistDeck();
    expect(useApp.getState().assistGen.phase).toBe("uncertain");
    await useApp.getState().answerAssistClarification("1 mm");
    expect(useApp.getState().assistSpec).toContain("== CLARIFICATION ==");
    expect(useApp.getState().assistGen.phase).toBe("accepted");
    expect(
      fake.localExecLog.filter((argv) => (argv[2] ?? "").includes("generate-deck")),
    ).toHaveLength(2);
  });

  it("cancels a running generation and invokes pkill", async () => {
    await resetStore(LOCAL_PROFILE);
    useApp.setState({ assistSpec: "Create a deck." });
    fake.hangGenerate = true;
    void useApp.getState().generateAssistDeck();
    for (let i = 0; i < 100 && useApp.getState().assistGen.phase !== "running"; i += 1) {
      await new Promise((resolve) => setTimeout(resolve, 1));
    }
    expect(useApp.getState().assistGen.phase).toBe("running");
    await useApp.getState().cancelAssistGeneration();
    expect(useApp.getState().assistGen.phase).toBe("error");
    expect(useApp.getState().assistGen.errorCode).toBe("CANCELLED");
    expect(fake.localExecLog.some((argv) => (argv[2] ?? "").includes("pkill"))).toBe(true);
  });

  it("rejects invalid intent JSON before launching generate-deck", async () => {
    await resetStore(LOCAL_PROFILE);
    useApp.setState({ assistSpec: "Create a deck.", assistIntentJson: "{" });
    await useApp.getState().generateAssistDeck();
    expect(useApp.getState().assistGen.errorCode).toBe("INTENT_JSON_INVALID");
    expect(fake.localExecLog.some((argv) => (argv[2] ?? "").includes("generate-deck"))).toBe(false);
  });
});

describe("startRun override", () => {
  it("uploads the override deck and derives its name and t_end", async () => {
    const deck = "# assistant deck\nt_end = 5.0e-9\n";
    await useApp.getState().startRun({ deck, name: "assistant_case" });
    const rec = useApp.getState().runs[0];
    expect(fake.uploads.find((upload) => upload.path.endsWith("/deck.py"))?.content).toBe(deck);
    expect(rec.name.startsWith("assistant_case_")).toBe(true);
    expect(rec.tEnd).toBe(5e-9);
  });
});
