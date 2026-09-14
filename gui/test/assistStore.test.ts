import { beforeEach, describe, expect, it } from "vitest";
import type { AppSettings, Backend, ExecResult } from "@tenryu-common/backend/types";
import type { ServerProfile } from "@tenryu-common/core/profiles";
import type { RunRecord } from "@tenryu-common/core/runstate";
import { buildAssistantToml } from "../src/core/assist/config";
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
  appConfigDirValue = "/home/fake/appconfig";
  assistHarnessDirValue = "/app/resources/tools/assist";
  remoteRoot = "/srv/tenryu";
  localRoot = "/opt/tenryu";
  remoteProbeOk = true;
  localProbeOk = true;
  mirrorPresent = true;
  mirrorSyncs: string[] = [];
  mirrorSyncCode = 0;
  mirrorSyncStderr = "";
  lintStdout = LINT_JSON;
  lintCode = 0;
  latestDirStdout = "/home/remote/gui_runs/x/outputs/20260902/\n";
  digestStdout = DIGEST_JSON;
  statusStdout = "{}";
  generateResults: Array<{ code: number; stdout: string; stderr?: string }> = [];
  askResults: Array<{ code: number; stdout: string; stderr?: string }> = [];
  askInvocations: Array<{ workdir: string; questionPath: string }> = [];
  hangGenerate = false;
  hangAsk = false;

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
    if (script.includes("tools/assist/assist.py && echo MIRROR_OK")) {
      return result(0, this.mirrorPresent ? "MIRROR_OK\n" : "");
    }
    if (script.includes("tar -xzf")) {
      this.mirrorSyncs.push(script);
      if (this.mirrorSyncCode === 0) this.mirrorPresent = true;
      const stdout = this.mirrorSyncCode === 0
        ? "MIRROR_OK\n"
        : this.mirrorSyncCode === 3
          ? "MISSING: docs/SPECIFICATION.md\n"
          : this.mirrorSyncCode === 4
            ? "MISSING: bundled tools/assist/assist.py\n"
            : "";
      return result(
        this.mirrorSyncCode,
        stdout,
        this.mirrorSyncStderr,
      );
    }
    if (script.includes("mesh_planner")) {
      return result(0, `${this.localRoot}\n`);
    }
    if (script.includes("ASSIST_OK")) {
      return result(0, this.localProbeOk ? "ASSIST_OK\n" : "");
    }
    if (script.includes("assist.py status")) {
      return result(0, this.statusStdout);
    }
    if (script.includes("assist.py ask")) {
      const workdirMatch = /--workdir (?:'([^']*)'|(\S+))/.exec(script);
      const questionPathMatch = /--question-file (?:'([^']*)'|(\S+))/.exec(script);
      const workdir = workdirMatch?.[1] ?? workdirMatch?.[2];
      const questionPath = questionPathMatch?.[1] ?? questionPathMatch?.[2];
      if (workdir === undefined) throw new Error("ask script has no workdir");
      if (questionPath === undefined) throw new Error("ask script has no question file");
      this.askInvocations.push({ workdir, questionPath });
      if (this.hangAsk) return new Promise<ExecResult>(() => {});
      const next = this.askResults.shift() ?? {
        code: 2,
        stdout: "",
        stderr: "no ask result queued",
      };
      return result(next.code, next.stdout, next.stderr ?? "");
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
    if (script.includes("mkdir -p")) {
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

  async appConfigDir(): Promise<string> {
    return this.appConfigDirValue;
  }

  async assistHarnessDir(): Promise<string> {
    return this.assistHarnessDirValue;
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
  fake.settings = {};
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
    assistMirror: {
      status: "idle",
      root: null,
      profileId: null,
      lastSyncIso: null,
      errorCode: null,
      errorDetail: null,
    },
    assistConfig: {
      form: null,
      path: null,
      status: "idle",
      savedPath: null,
      error: null,
    },
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
    assistQuestion: "",
    assistAsk: {
      phase: "idle",
      workdir: null,
      turns: [],
      pendingQuestion: null,
      lastKind: null,
      errorCode: null,
      errorDetail: null,
      resultRaw: null,
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

describe("assistant configuration", () => {
  it("seeds the two read-only presets from an idle status", () => {
    useApp.getState().initAssistConfigForm();
    const config = useApp.getState().assistConfig;
    expect(config.form?.providers.map((provider) => provider.name)).toEqual([
      "claude_readonly",
      "codex_readonly",
    ]);
    expect(config.form?.roles.question_answering).toBe("claude_readonly");
  });

  it("saves app settings and refreshes assistant status", async () => {
    fake.statusStdout = JSON.stringify({
      enabled: false,
      disabled_by: null,
      config_source: "/home/fake/appconfig/assistant.toml",
      providers: {},
      roles: {},
      budget: {
        max_interventions_per_run: 3,
        max_tokens_per_decision: 30000,
      },
      warnings: [],
    });
    useApp.getState().initAssistConfigForm();
    const form = useApp.getState().assistConfig.form;
    if (form === null) throw new Error("assistant config form was not initialized");

    await useApp.getState().saveAssistConfig();

    expect(
      fake.localExecLog.some(
        (argv) => argv[2] === "mkdir -p /home/fake/appconfig",
      ),
    ).toBe(true);
    expect(fake.localWrites).toContainEqual({
      path: "/home/fake/appconfig/assistant.toml",
      content: buildAssistantToml(form),
    });
    expect(useApp.getState().assistConfig).toMatchObject({
      status: "saved",
      savedPath: "/home/fake/appconfig/assistant.toml",
      path: "/home/fake/appconfig/assistant.toml",
      error: null,
    });
    expect(
      fake.localExecLog.some((argv) =>
        (argv[2] ?? "").includes(
          "assist.py status --config-or-defaults /home/fake/appconfig/assistant.toml",
        ),
      ),
    ).toBe(true);
  });

  it("rejects an empty model without writing", async () => {
    useApp.getState().initAssistConfigForm();
    const form = useApp.getState().assistConfig.form;
    if (form === null) throw new Error("assistant config form was not initialized");
    useApp.getState().setAssistConfigForm({
      ...form,
      providers: form.providers.map((provider, index) =>
        index === 0 ? { ...provider, model: "" } : provider,
      ),
    });

    await useApp.getState().saveAssistConfig();

    expect(useApp.getState().assistConfig.status).toBe("error");
    expect(useApp.getState().assistConfig.error).toContain("MODEL_EMPTY");
    expect(fake.localWrites).toEqual([]);
  });

  it("fills deck_design when adding the codex preset", () => {
    useApp.getState().initAssistConfigForm();
    useApp.getState().addAssistProviderPreset("codex");
    expect(useApp.getState().assistConfig.form?.roles.deck_design).toBe(
      "codex",
    );
  });
});

describe("fetchAssistStatus", () => {
  it("reports NO_SOURCE when no profile is selected", async () => {
    useApp.setState({ profiles: [], currentProfileId: null, assistLocalRepo: "" });
    await useApp.getState().fetchAssistStatus();
    expect(useApp.getState().assistStatus).toEqual({
      status: "error",
      error: "NO_SOURCE",
    });
    expect(useApp.getState().assistMirror.errorCode).toBe("NO_PROFILE");
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
    expect(useApp.getState().assistConfig.path).toBe(
      "/home/fake/appconfig/assistant.toml",
    );
    expect(
      fake.localExecLog.some((argv) =>
        (argv[2] ?? "").includes(
          "assist.py status --config-or-defaults /home/fake/appconfig/assistant.toml",
        ),
      ),
    ).toBe(true);
  });

  it("passes --config-or-defaults while the configuration file does not exist", async () => {
    await useApp.getState().fetchAssistStatus();

    const script = fake.localExecLog.find((argv) =>
      (argv[2] ?? "").includes("assist.py status")
    )?.[2] ?? "";
    expect(script).toContain(
      "--config-or-defaults /home/fake/appconfig/assistant.toml",
    );
    expect(script).not.toContain("--config ");
  });

  it("syncs a missing mirror before running status", async () => {
    await resetStore({ ...SSH_PROFILE, host: "parma" });
    fake.mirrorPresent = false;

    await useApp.getState().fetchAssistStatus();

    expect(fake.mirrorSyncs).toHaveLength(1);
    expect(fake.mirrorSyncs[0]).toContain("ssh -o BatchMode=yes");
    expect(fake.mirrorSyncs[0]).toContain(" parma ");
    expect(fake.mirrorSyncs[0]).toContain("cd /srv/tenryu && tar -czf -");
    expect(fake.mirrorSyncs[0]).toContain("docs/site");
    expect(fake.mirrorSyncs[0]).toContain("cp -R /app/resources/tools/assist");
    expect(fake.mirrorSyncs[0]).not.toContain(" tools/assist ");
    const metadataPath = "/home/fake/appconfig/mirror/ssh/mirror.json";
    expect(fake.localTextFiles.has(metadataPath)).toBe(true);
    expect(
      fake.localExecLog.some((argv) =>
        (argv[2] ?? "").includes(
          "cd /home/fake/appconfig/mirror/ssh && python3 tools/assist/assist.py status",
        ),
      ),
    ).toBe(true);
    expect(useApp.getState().assistMirror.status).toBe("ready");
    expect(useApp.getState().assistMirror.lastSyncIso).not.toBeNull();
  });

  it("uses an existing mirror and reads its last sync time", async () => {
    const lastSyncIso = "2026-09-01T02:03:04.000Z";
    fake.mirrorPresent = true;
    fake.localTextFiles.set(
      "/home/fake/appconfig/mirror/ssh/mirror.json",
      JSON.stringify({ lastSyncIso }),
    );

    await useApp.getState().fetchAssistStatus();

    expect(fake.mirrorSyncs).toEqual([]);
    expect(
      fake.localExecLog.some((argv) =>
        (argv[2] ?? "").includes(
          "cd /home/fake/appconfig/mirror/ssh && python3 tools/assist/assist.py status",
        ),
      ),
    ).toBe(true);
    expect(useApp.getState().assistMirror.lastSyncIso).toBe(lastSyncIso);
  });

  it("reports a mirror sync failure as NO_SOURCE", async () => {
    fake.mirrorPresent = false;
    fake.mirrorSyncCode = 1;
    fake.mirrorSyncStderr = "sync failed";

    await useApp.getState().fetchAssistStatus();

    expect(useApp.getState().assistMirror.errorCode).toBe("MIRROR_SYNC_FAILED");
    expect(useApp.getState().assistStatus.error?.startsWith("NO_SOURCE")).toBe(true);
  });

  it("reports missing server documents", async () => {
    fake.mirrorPresent = false;
    fake.mirrorSyncCode = 3;

    await useApp.getState().fetchAssistStatus();

    expect(useApp.getState().assistMirror.errorCode).toBe("MIRROR_INCOMPLETE");
    expect(useApp.getState().assistMirror.errorDetail).toContain("docs/SPECIFICATION.md");
    expect(useApp.getState().assistMirror.errorDetail).toContain("(server: /srv/tenryu)");
  });

  it("reports a missing bundled assistant harness", async () => {
    fake.mirrorPresent = false;
    fake.mirrorSyncCode = 4;

    await useApp.getState().fetchAssistStatus();

    expect(useApp.getState().assistMirror.errorCode).toBe("MIRROR_HARNESS_MISSING");
  });

  it("uses the developer override without probing or syncing the mirror", async () => {
    fake.mirrorPresent = false;
    useApp.setState({ assistLocalRepo: "/local/tenryu" });

    await useApp.getState().fetchAssistStatus();

    expect(fake.mirrorSyncs).toEqual([]);
    expect(
      fake.localExecLog.some((argv) => (argv[2] ?? "").includes("MIRROR_OK")),
    ).toBe(false);
    expect(
      fake.localExecLog.some((argv) =>
        (argv[2] ?? "").includes(
          "cd /local/tenryu && python3 tools/assist/assist.py status",
        ),
      ),
    ).toBe(true);
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

describe("askAssistQuestion", () => {
  it("syncs a missing mirror before asking", async () => {
    fake.mirrorPresent = false;
    useApp.setState({ assistQuestion: "What does this mean?" });
    fake.askResults.push({
      code: 0,
      stdout: JSON.stringify({ status: "answered", answer: "An answer", turn: 1 }),
    });

    await useApp.getState().askAssistQuestion();

    expect(fake.mirrorSyncs).toHaveLength(1);
    expect(
      fake.localExecLog.some((argv) =>
        (argv[2] ?? "").includes("cd /home/fake/appconfig/mirror/ssh") &&
        (argv[2] ?? "").includes("assist.py ask"),
      ),
    ).toBe(true);
    expect(useApp.getState().assistAsk.phase).toBe("answered");
  });

  it("completes an answered lifecycle and records checks", async () => {
    useApp.setState({ assistQuestion: "What does Numerics.dt.initial_s mean?" });
    fake.askResults.push({
      code: 0,
      stdout: JSON.stringify({
        status: "answered",
        answer: "It is the initial time step.",
        turn: 1,
        workdir: "/home/fake/appconfig/ask/20260903010203",
        checks: {
          citations: {
            verified: ["SPECIFICATION.md:42"],
            unverified: [
              { citation: "missing.md:1", reason: "path not found" },
            ],
          },
          keys: {
            known: ["Numerics.dt.initial_s"],
            unknown: ["Numerics.dt.typo"],
          },
        },
      }),
    });

    await useApp.getState().askAssistQuestion();

    const ask = useApp.getState().assistAsk;
    expect(ask.phase).toBe("answered");
    expect(ask.turns).toEqual([
      {
        turn: 1,
        question: "What does Numerics.dt.initial_s mean?",
        answer: "It is the initial time step.",
        checks: {
          citationsVerified: ["SPECIFICATION.md:42"],
          citationsUnverified: [
            { citation: "missing.md:1", reason: "path not found" },
          ],
          keysKnown: ["Numerics.dt.initial_s"],
          keysUnknown: ["Numerics.dt.typo"],
        },
      },
    ]);
    expect(useApp.getState().assistQuestion).toBe("");
    expect(ask.workdir).toMatch(/^\/home\/fake\/appconfig\/ask\/\d{14}$/);
    const questionWrite = fake.localWrites.find((write) =>
      write.path.endsWith("/question_1.md")
    );
    expect(questionWrite?.path).toBe(`${ask.workdir}/question_1.md`);
    expect(questionWrite?.content).toBe(
      "What does Numerics.dt.initial_s mean?\n",
    );
    expect(
      fake.localExecLog.some((argv) =>
        (argv[2] ?? "").includes(
          "--config-or-defaults /home/fake/appconfig/assistant.toml",
        ) && (argv[2] ?? "").includes("assist.py ask"),
      ),
    ).toBe(true);
  });

  it("reuses the same workdir for a second question", async () => {
    fake.askResults.push(
      {
        code: 0,
        stdout: JSON.stringify({ status: "answered", answer: "First", turn: 1 }),
      },
      {
        code: 0,
        stdout: JSON.stringify({ status: "answered", answer: "Second", turn: 2 }),
      },
    );
    useApp.setState({ assistQuestion: "First question" });
    await useApp.getState().askAssistQuestion();
    const firstWorkdir = useApp.getState().assistAsk.workdir;

    useApp.setState({ assistQuestion: "Second question" });
    await useApp.getState().askAssistQuestion();

    expect(fake.askInvocations).toHaveLength(2);
    expect(fake.askInvocations[1].workdir).toBe(firstWorkdir);
    expect(fake.askInvocations[1].questionPath).toBe(
      `${firstWorkdir}/question_2.md`,
    );
    expect(useApp.getState().assistAsk.turns).toHaveLength(2);
    expect(
      fake.localWrites.find((write) => write.path.endsWith("/question_2.md"))
        ?.content,
    ).toBe("Second question\n");
  });

  it("reports NO_QUESTION for a blank question", async () => {
    useApp.setState({ assistQuestion: "   " });
    await useApp.getState().askAssistQuestion();
    expect(useApp.getState().assistAsk.errorCode).toBe("NO_QUESTION");
  });

  it("reports NO_SOURCE when no profile or override is available", async () => {
    useApp.setState({
      assistQuestion: "What does this mean?",
      assistLocalRepo: "",
      profiles: [],
      currentProfileId: null,
    });
    await useApp.getState().askAssistQuestion();
    expect(useApp.getState().assistAsk.errorCode).toBe("NO_SOURCE");
    expect(useApp.getState().assistMirror.errorCode).toBe("NO_PROFILE");
  });

  it("keeps existing turns when the backend returns an error result", async () => {
    fake.askResults.push(
      {
        code: 0,
        stdout: JSON.stringify({ status: "answered", answer: "First", turn: 1 }),
      },
      {
        code: 2,
        stdout: JSON.stringify({ status: "error", error: "provider failed" }),
        stderr: "provider exited",
      },
    );
    useApp.setState({ assistQuestion: "First question" });
    await useApp.getState().askAssistQuestion();
    useApp.setState({ assistQuestion: "Follow-up question" });
    await useApp.getState().askAssistQuestion();

    const ask = useApp.getState().assistAsk;
    expect(ask.phase).toBe("error");
    expect(ask.errorCode).toBe("ASK_FAILED");
    expect(ask.errorDetail).toBe("provider failed");
    expect(ask.turns).toHaveLength(1);
    expect(useApp.getState().assistQuestion).toBe("Follow-up question");
  });

  it("reports RESULT_PARSE for non-JSON stdout", async () => {
    useApp.setState({ assistQuestion: "What does this mean?" });
    fake.askResults.push({ code: 2, stdout: "not json", stderr: "bad output" });
    await useApp.getState().askAssistQuestion();
    const ask = useApp.getState().assistAsk;
    expect(ask.errorCode).toBe("RESULT_PARSE");
    expect(ask.resultRaw).toBe("not json");
  });

  it("cancels a hanging question and invokes pkill", async () => {
    useApp.setState({ assistQuestion: "What does this mean?" });
    fake.hangAsk = true;
    void useApp.getState().askAssistQuestion();
    for (let i = 0; i < 100 && useApp.getState().assistAsk.phase !== "running"; i += 1) {
      await new Promise((resolve) => setTimeout(resolve, 1));
    }
    expect(useApp.getState().assistAsk.phase).toBe("running");
    await useApp.getState().cancelAssistQuestion();
    expect(useApp.getState().assistAsk.phase).toBe("error");
    expect(useApp.getState().assistAsk.errorCode).toBe("CANCELLED");
    expect(useApp.getState().assistQuestion).toBe("What does this mean?");
    expect(fake.localExecLog.some((argv) => (argv[2] ?? "").includes("pkill"))).toBe(true);
  });

  it("resets the conversation turns and workdir", () => {
    useApp.setState({
      assistAsk: {
        phase: "answered",
        workdir: "/tmp/ask",
        turns: [
          {
            turn: 1,
            question: "Question",
            answer: "Answer",
            checks: null,
          },
        ],
        pendingQuestion: null,
        lastKind: "answer_checked",
        errorCode: null,
        errorDetail: null,
        resultRaw: "{}",
      },
    });
    useApp.getState().resetAssistConversation();
    expect(useApp.getState().assistAsk.turns).toEqual([]);
    expect(useApp.getState().assistAsk.workdir).toBeNull();
    expect(useApp.getState().assistAsk.pendingQuestion).toBeNull();
  });

  it("moves the draft into pendingQuestion once the ask is running", async () => {
    useApp.setState({ assistQuestion: "What does this mean?" });
    fake.hangAsk = true;
    void useApp.getState().askAssistQuestion();
    for (let i = 0; i < 100 && useApp.getState().assistAsk.phase !== "running"; i += 1) {
      await new Promise((resolve) => setTimeout(resolve, 1));
    }
    expect(useApp.getState().assistAsk.pendingQuestion).toBe("What does this mean?");
    expect(useApp.getState().assistQuestion).toBe("");
    await useApp.getState().cancelAssistQuestion();
    expect(useApp.getState().assistAsk.pendingQuestion).toBe("What does this mean?");
  });

  it("clears pendingQuestion once the answer arrives", async () => {
    useApp.setState({ assistQuestion: "First question" });
    fake.askResults.push({
      code: 0,
      stdout: JSON.stringify({ status: "answered", answer: "First", turn: 1 }),
    });
    await useApp.getState().askAssistQuestion();
    expect(useApp.getState().assistAsk.phase).toBe("answered");
    expect(useApp.getState().assistAsk.pendingQuestion).toBeNull();
  });

  it("keeps a newly typed draft instead of restoring the cancelled question", async () => {
    useApp.setState({
      assistAsk: {
        phase: "running",
        workdir: "/tmp/ask",
        turns: [],
        pendingQuestion: "q1",
        lastKind: null,
        errorCode: null,
        errorDetail: null,
        resultRaw: null,
      },
      assistQuestion: "new draft",
    });
    await useApp.getState().cancelAssistQuestion();
    expect(useApp.getState().assistAsk.errorCode).toBe("CANCELLED");
    expect(useApp.getState().assistQuestion).toBe("new draft");
  });

  it("keeps pendingQuestion when the ask fails", async () => {
    useApp.setState({ assistQuestion: "First question" });
    fake.askResults.push({
      code: 2,
      stdout: JSON.stringify({ status: "error", error: "provider failed" }),
      stderr: "provider exited",
    });
    await useApp.getState().askAssistQuestion();
    expect(useApp.getState().assistAsk.phase).toBe("error");
    expect(useApp.getState().assistAsk.pendingQuestion).toBe("First question");
  });
});

describe("chat dock visibility", () => {
  it("toggles and clears chatOpen", () => {
    expect(useApp.getState().chatOpen).toBe(false);
    useApp.getState().toggleChat();
    expect(useApp.getState().chatOpen).toBe(true);
    useApp.getState().toggleChat();
    expect(useApp.getState().chatOpen).toBe(false);
    useApp.getState().setChatOpen(true);
    expect(useApp.getState().chatOpen).toBe(true);
    useApp.getState().setChatOpen(false);
    expect(useApp.getState().chatOpen).toBe(false);
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
    expect(gen.workdir).toMatch(/^\/home\/fake\/appconfig\/generate\/\d{14}$/);
    const script = fake.localExecLog.find((argv) => (argv[2] ?? "").includes("generate-deck"))?.[2] ?? "";
    expect(script).toContain("--tenryu /opt/tenryu/build/tenryu");
    expect(script).toContain("--config-or-defaults /home/fake/appconfig/assistant.toml");
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
