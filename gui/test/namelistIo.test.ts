import { beforeEach, describe, expect, it } from "vitest";
import type { Backend, ExecResult } from "@tenryu-common/backend/types";
import { defaultFormState } from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";
import type { ServerProfile } from "@tenryu-common/core/profiles";
import { t } from "../src/i18n";
import { __setBackendForTest, useApp } from "../src/store";

class FakeBackend implements Backend {
  readonly kind = "devbridge" as const;
  openResult: { name: string; path: string | null; content: string } | null = null;
  saveCalls: Array<{ suggestedName: string; content: string }> = [];
  writeCalls: Array<{ path: string; content: string }> = [];

  async listProfiles(): Promise<ServerProfile[]> {
    return [];
  }
  async saveProfiles(): Promise<void> {}
  async getSettings() {
    return {};
  }
  async saveSettings(): Promise<void> {}
  async exec(): Promise<ExecResult> {
    return { code: 0, stdout: "", stderr: "", timedOut: false };
  }
  async uploadText(): Promise<void> {}
  async readText(): Promise<string> {
    return "";
  }
  async saveLocalTextFile(suggestedName: string, content: string): Promise<string> {
    this.saveCalls.push({ suggestedName, content });
    return `/fake/dir/${suggestedName}`;
  }
  async writeLocalTextFile(path: string, content: string): Promise<void> {
    this.writeCalls.push({ path, content });
  }
  async openLocalTextFile(): Promise<{ name: string; path: string | null; content: string } | null> {
    return this.openResult;
  }
  async readBinary(): Promise<Uint8Array> {
    return new Uint8Array();
  }
}

let fake: FakeBackend;

beforeEach(() => {
  fake = new FakeBackend();
  __setBackendForTest(fake);
  useApp.getState().loadForm(defaultFormState());
  useApp.setState({ deckIoStatus: null });
});

describe("namelist IO", () => {
  it("load round-trips a generated deck", async () => {
    const form = defaultFormState();
    form.main.name = "rt_case";
    fake.openResult = { name: "x.py", path: "/tmp/rt_case.py", content: generateDeck(form) };

    await useApp.getState().loadNamelist();

    expect(useApp.getState().form.main.name).toBe("rt_case");
    expect(useApp.getState().deckIoStatus?.kind).toBe("loaded");
    expect(useApp.getState().namelistPath).toBe("/tmp/rt_case.py");
  });

  it("save-as records the path, save overwrites without a dialog", async () => {
    await useApp.getState().saveNamelistAs();

    const path = `/fake/dir/${useApp.getState().form.main.name}.py`;
    expect(useApp.getState().namelistPath).toBe(path);
    expect(fake.saveCalls).toHaveLength(1);

    await useApp.getState().saveNamelist();

    expect(fake.writeCalls).toEqual([{ path, content: useApp.getState().deck }]);
    expect(fake.saveCalls).toHaveLength(1);
  });

  it("first save without a path falls back to save-as", async () => {
    expect(useApp.getState().namelistPath).toBeNull();

    await useApp.getState().saveNamelist();

    expect(fake.saveCalls).toHaveLength(1);
    expect(fake.writeCalls).toEqual([]);
  });

  it("loadForm clears the path", async () => {
    await useApp.getState().saveNamelistAs();
    expect(useApp.getState().namelistPath).not.toBeNull();

    useApp.getState().loadForm(defaultFormState());

    expect(useApp.getState().namelistPath).toBeNull();
  });

  it("load reports a non-GUI deck", async () => {
    const before = structuredClone(useApp.getState().form);
    fake.openResult = { name: "x.py", path: "/tmp/x.py", content: "print(1)" };

    await useApp.getState().loadNamelist();

    expect(useApp.getState().deckIoStatus).toEqual({
      kind: "error",
      detail: t().deck.loadErrNoMarker,
    });
    expect(useApp.getState().form).toEqual(before);
  });

  it("cancel leaves state untouched", async () => {
    fake.openResult = null;

    await useApp.getState().loadNamelist();

    expect(useApp.getState().deckIoStatus).toBeNull();
  });

  it("saves a WIP namelist when the form is invalid", async () => {
    useApp.getState().updateForm((form) => {
      form.mesh.rMax.value = -1;
    });

    expect(useApp.getState().formErrors.length).toBeGreaterThan(0);
    expect(useApp.getState().deck).toBe("");

    await useApp.getState().saveNamelistAs();

    expect(fake.saveCalls).toHaveLength(1);
    expect(fake.saveCalls[0].content).toContain("# TENRYU-GUI-STATE: ");
    expect(fake.saveCalls[0].content).toContain("WARNING: saved with form validation errors");
    expect(useApp.getState().deckIoStatus?.kind).toBe("saved");
    expect(useApp.getState().deckIoStatus?.detail).toContain(t().deckIo.savedWithErrors);
  });

  it("a WIP save round-trips through load", async () => {
    const corruptedRMax = -1;
    useApp.getState().updateForm((form) => {
      form.mesh.rMax.value = corruptedRMax;
    });

    await useApp.getState().saveNamelistAs();
    const content = fake.saveCalls[0].content;

    useApp.getState().loadForm(defaultFormState());
    fake.openResult = { name: "wip.py", path: "/tmp/wip.py", content };
    await useApp.getState().loadNamelist();

    expect(useApp.getState().form.mesh.rMax.value).toBe(corruptedRMax);
    expect(useApp.getState().deckIoStatus?.kind).toBe("loaded");
  });
});
