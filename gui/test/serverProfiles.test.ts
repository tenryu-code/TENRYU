import { beforeEach, describe, expect, it } from "vitest";
import type { Backend, ExecResult } from "@tenryu-common/backend/types";
import { defaultFormState, ensureBackgroundGas } from "../src/core/deck/formState";
import { defaultShape2D } from "../src/core/geometry2d";
import {
  newProfile,
  parsePodSsh,
  profileBinMissing,
  validateProfile,
  type ServerProfile,
} from "@tenryu-common/core/profiles";
import { t } from "../src/i18n";
import { __setBackendForTest, useApp } from "../src/store";
import { q } from "../src/core/units";

const OK: ExecResult = { code: 0, stdout: "", stderr: "", timedOut: false };

class FakeBackend implements Backend {
  readonly kind = "devbridge" as const;
  profiles: ServerProfile[] = [];
  settings = {};
  savedProfiles: ServerProfile[][] = [];
  execs: string[][] = [];
  uploads: Array<{ path: string; content: string }> = [];
  saveCalls: Array<{ suggestedName: string; content: string }> = [];
  writeCalls: Array<{ path: string; content: string }> = [];
  repoRootStdout = "";
  localTextFile: { name: string; path: string | null; content: string } | null = null;
  localExecLog: string[][] = [];
  localTextFiles: Record<string, string> = {};

  async listProfiles() {
    return this.profiles;
  }
  async saveProfiles(profiles: ServerProfile[]) {
    this.savedProfiles.push(profiles);
    this.profiles = profiles;
  }
  async getSettings() {
    return this.settings;
  }
  async saveSettings() {}
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
  async execLocal(argv: string[]): Promise<ExecResult> {
    this.localExecLog.push(argv);
    return { code: 0, stdout: "", stderr: "", timedOut: false };
  }
  async readLocalText(path: string): Promise<string> {
    const c = this.localTextFiles[path];
    if (c === undefined) throw new Error(`fake readLocalText: no file ${path}`);
    return c;
  }
  async writeLocalText(path: string, content: string): Promise<void> {
    this.localTextFiles[path] = content;
  }
  async exec(_profile: ServerProfile, argv: string[]): Promise<ExecResult> {
    this.execs.push(argv);
    if (argv[0] === "bash" && argv[1] === "-lc" && argv[2].includes("mesh_planner")) {
      return { ...OK, stdout: this.repoRootStdout };
    }
    if (argv[0] === "bash" && argv[1] === "-lc" && argv[2].startsWith("cd ")) {
      return { ...OK, stdout: "sub/\nx.h5\n" };
    }
    if (argv[0] === "printenv" && argv[1] === "HOME") {
      return { ...OK, stdout: "/home/fake\n" };
    }
    if (argv[0] === "nvidia-smi") return { ...OK, stdout: "NVIDIA RTX 4090\n" };
    return OK;
  }
  async uploadText(_profile: ServerProfile, path: string, content: string) {
    this.uploads.push({ path, content });
  }
  async readText() {
    return "";
  }
}

const binlessProfile: ServerProfile = {
  ...newProfile(),
  id: "p-binless",
  name: "example-host",
  host: "example-host",
  tenryuBin: "",
  runDir: "~/x",
};

let fake: FakeBackend;

beforeEach(async () => {
  fake = new FakeBackend();
  fake.profiles = [binlessProfile];
  __setBackendForTest(fake);
  const form = defaultFormState();
  useApp.setState({
    profiles: [],
    profilesLoaded: false,
    loadError: null,
    currentProfileId: null,
    form,
    formErrors: [],
    deck: "",
    validating: false,
    validateResult: null,
    validateSentTo: null,
    meshSnapshot: null,
    meshSnapshotBusy: false,
    meshSnapshotError: null,
    connTesting: false,
    connResult: null,
  });
  useApp.getState().loadForm(form);
  await useApp.getState().loadInitial();
});

describe("server profiles", () => {
  it("parses RunPod ssh state", () => {
    expect(parsePodSsh("1.2.3.4 12345\n")).toEqual({ host: "1.2.3.4", port: 12345 });
    expect(parsePodSsh("garbage")).toBeNull();
    expect(parsePodSsh("1.2.3.4 70000")).toBeNull();
    expect(parsePodSsh("1.2.3.4 22 extra")).toBeNull();
  });

  it("validates a profile without requiring a binary path", () => {
    const valid = {
      ...newProfile(),
      name: "example-host",
      host: "example-host",
      tenryuBin: "",
      runDir: "~/x",
    };
    expect(validateProfile(valid)).toEqual([]);
    expect(validateProfile({ ...valid, name: "" }).length).toBeGreaterThan(0);
    expect(validateProfile({ ...valid, host: "", transport: "ssh" }).length).toBeGreaterThan(0);
    expect(validateProfile({ ...valid, tenryuBin: "/opt/tenryu\n" }).length).toBeGreaterThan(0);
  });

  it("detects missing binary paths", () => {
    expect(profileBinMissing({ ...binlessProfile, tenryuBin: "" })).toBe(true);
    expect(profileBinMissing({ ...binlessProfile, tenryuBin: "  " })).toBe(true);
    expect(profileBinMissing({ ...binlessProfile, tenryuBin: "/opt/tenryu" })).toBe(false);
  });

  it("guards validation before upload or execution", async () => {
    await useApp.getState().runValidate();
    expect(useApp.getState().validateResult?.errors).toContain("NO_BIN");
    expect(fake.execs).toEqual([]);
    expect(fake.uploads).toEqual([]);
  });

  it("guards mesh snapshot fetching before execution", async () => {
    await useApp.getState().fetchMeshSnapshot();
    expect(useApp.getState().meshSnapshotError).toBe("NO_BIN");
    expect(fake.execs).toEqual([]);
  });

  it("skips binary probes but runs the other connection probes", async () => {
    await useApp.getState().testConnection(binlessProfile);
    const result = useApp.getState().connResult;
    expect(result?.binOk).toBe(false);
    expect(result?.binRuns).toBe(false);
    expect(result?.detail).toContain(t().server.binSkippedDetail);
    expect(fake.execs.some((argv) => argv[0] === "test" && argv[1] === "-x")).toBe(false);
    expect(fake.execs.some((argv) => argv.includes("--help"))).toBe(false);
    expect(fake.execs).toContainEqual(["echo", "tenryu-studio-ping"]);
    expect(fake.execs).toContainEqual([
      "nvidia-smi",
      "--query-gpu=name",
      "--format=csv,noheader",
    ]);
    expect(fake.execs.some((argv) => argv[0] === "bash" && argv[1] === "-lc")).toBe(true);
  });

  it("persists a valid bin-less profile through the store action", async () => {
    await useApp.getState().upsertProfile(binlessProfile);
    expect(fake.savedProfiles).toHaveLength(1);
    expect(fake.savedProfiles[0]).toContainEqual(binlessProfile);
  });

  it("round-trips cloud ssh fields through the store action", async () => {
    const cloudProfile = {
      ...binlessProfile,
      id: "p-cloud",
      identityFile: "~/.ssh/runpod_ed25519",
      ephemeralHostKey: true,
    };
    await useApp.getState().upsertProfile(cloudProfile);
    expect(fake.savedProfiles).toHaveLength(1);
    expect(fake.savedProfiles[0]).toContainEqual(cloudProfile);
  });

  it("rejects remote directory listing without a current profile", async () => {
    useApp.setState({ currentProfileId: null });

    const result = await useApp.getState().listRemoteDir("");

    expect(result.ok).toBe(false);
    expect(result.error).toBe("NO_PROFILE");
    expect(fake.execs).toEqual([]);
  });

  it("lists a remote directory from the resolved home", async () => {
    const result = await useApp.getState().listRemoteDir("");

    expect(result).toEqual({
      ok: true,
      path: "/home/fake",
      entries: [
        { name: "sub", dir: true },
        { name: "x.h5", dir: false },
      ],
      error: "",
    });
    const listing = fake.execs.find(
      (argv) => argv[0] === "bash" && argv[1] === "-lc" && argv[2].startsWith("cd "),
    );
    expect(listing?.[2].startsWith("cd /home/fake && ls -1paL")).toBe(true);
  });

  it("resolves and caches the repo root", async () => {
    const profile = { ...binlessProfile, id: "p-repo", tenryuBin: "/repo/build/tenryu" };
    fake.repoRootStdout = "/repo\n";
    useApp.setState({ profiles: [profile], currentProfileId: profile.id });

    await useApp.getState().runValidate();
    await useApp.getState().runValidate();

    const probes = fake.execs.filter(
      (argv) => argv[0] === "bash" && argv[1] === "-lc" && argv[2].includes("mesh_planner"),
    );
    expect(probes).toHaveLength(1);
    const validates = fake.execs.filter((argv) => argv[0] === "env");
    expect(validates).toHaveLength(2);
    expect(validates[0].slice(0, 4)).toEqual([
      "env",
      "TENRYU_REPO=/repo",
      "/repo/build/tenryu",
      "validate",
    ]);
  });

  it("pib without tools/ on the server yields NO_TOOLS", async () => {
    const profile = { ...binlessProfile, id: "p-pib", tenryuBin: "/repo/build/tenryu" };
    useApp.setState({ profiles: [profile], currentProfileId: profile.id });
    useApp.getState().updateForm((form) => {
      form.main.dimension = "2D_RZ";
      form.mesh.meshMode2d = "polar_in_box";
      ensureBackgroundGas(form);
      const sphere = defaultShape2D("solidSphere");
      sphere.materialName = "gas";
      sphere.rho = 1.0e-4;
      sphere.z0 = q(0, "µm");
      sphere.radius = q(200, "µm");
      form.geometry.regions = [];
      form.geometry.shapes2d = [sphere];
    });
    expect(useApp.getState().formErrors).toEqual([]);

    await useApp.getState().runValidate();

    expect(useApp.getState().validateResult?.errors).toContain("NO_TOOLS");
    expect(fake.uploads).toEqual([]);
  });
});
