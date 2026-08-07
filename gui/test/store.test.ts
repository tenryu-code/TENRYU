import { beforeEach, describe, expect, it } from "vitest";
import type { Backend, ExecResult } from "@tenryu-common/backend/types";
import { defaultFormState } from "../src/core/deck/formState";
import type { ServerProfile } from "@tenryu-common/core/profiles";
import { __setBackendForTest, currentProfile, useApp } from "../src/store";
import { q } from "../src/core/units";

const OK_STDOUT = [
  "[x] [info] [TENRYU] Configuration validated successfully.",
  "[x] [info] [TENRYU] ---- pre-flight summary ----",
  "[x] [info] [TENRYU] main      : name=t  dimension=1D_SPH  geometry=spherical  t_end=2e-09 s",
  "[x] [info] [TENRYU] ----------------------------",
].join("\n");

class FakeBackend implements Backend {
  readonly kind = "devbridge" as const;
  profiles: ServerProfile[] = [];
  settings: { lastProfileId?: string } = {};
  uploads: Array<{ path: string; content: string }> = [];
  saveCalls: Array<{ suggestedName: string; content: string }> = [];
  writeCalls: Array<{ path: string; content: string }> = [];
  execs: string[][] = [];
  execResult: ExecResult = { code: 0, stdout: OK_STDOUT, stderr: "", timedOut: false };
  localTextFile: { name: string; path: string | null; content: string } | null = null;

  async listProfiles() {
    return this.profiles;
  }
  async saveProfiles(p: ServerProfile[]) {
    this.profiles = p;
  }
  async getSettings() {
    return this.settings;
  }
  async saveSettings(s: { lastProfileId?: string }) {
    this.settings = s;
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
  async exec(_p: ServerProfile, argv: string[]) {
    this.execs.push(argv);
    return this.execResult;
  }
  async uploadText(_p: ServerProfile, path: string, content: string) {
    this.uploads.push({ path, content });
  }
  async readText() {
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

let fake: FakeBackend;

beforeEach(() => {
  fake = new FakeBackend();
  __setBackendForTest(fake);
  const form = defaultFormState();
  useApp.setState({
    profiles: [],
    profilesLoaded: false,
    loadError: null,
    currentProfileId: null,
    view: "form",
    section: "basic",
    form,
    formErrors: [],
    deck: "",
    validating: false,
    validateResult: null,
    validateSentTo: null,
    connTesting: false,
    connResult: null,
  });
  useApp.getState().loadForm(form);
});

describe("store (form-centric)", () => {
  it("updateForm regenerates deck and embeds GUI state", () => {
    useApp.getState().updateForm((f) => {
      f.main.name = "form_case";
    });
    const s = useApp.getState();
    expect(s.formErrors).toEqual([]);
    expect(s.deck).toContain('name="form_case"');
    expect(s.deck).toContain("# TENRYU-GUI-STATE: ");
  });

  it("invalid form yields errors and empty deck", () => {
    useApp.getState().updateForm((f) => {
      f.mesh.rMax = q(-1, "cm");
    });
    const s = useApp.getState();
    expect(s.formErrors.length).toBeGreaterThan(0);
    expect(s.deck).toBe("");
  });

  it("runValidate uploads generated deck and parses result", async () => {
    fake.profiles = [profile];
    await useApp.getState().loadInitial();
    useApp.getState().updateForm((f) => {
      f.main.name = "sent_case";
    });
    await useApp.getState().runValidate();
    expect(fake.uploads.length).toBe(1);
    expect(fake.uploads[0].path).toBe("~/tenryu_gui_runs/validate_scratch/sent_case.py");
    expect(fake.uploads[0].content).toContain("# TENRYU-GUI-STATE: ");
    const validateExec = fake.execs.find((argv) => argv.includes("validate"));
    expect(validateExec).toEqual(["/opt/tenryu", "validate", "~/tenryu_gui_runs/validate_scratch/sent_case.py"]);
    expect(fake.execs[0][0]).toBe("bash");
    expect(fake.execs[0][2]).toContain("mesh_planner");
    expect(useApp.getState().validateResult?.ok).toBe(true);
    expect(currentProfile(useApp.getState())?.id).toBe("p1");
  });

  it("runValidate with invalid form reports FORM_INVALID", async () => {
    fake.profiles = [profile];
    await useApp.getState().loadInitial();
    useApp.getState().updateForm((f) => {
      f.geometry.regions[0].rho = -5;
    });
    await useApp.getState().runValidate();
    const errs = useApp.getState().validateResult?.errors ?? [];
    expect(errs[0]).toBe("FORM_INVALID");
    expect(errs.length).toBeGreaterThan(1);
  });
});
