import { beforeEach, describe, expect, it } from "vitest";
import type { Backend, ExecResult } from "@tenryu-common/backend/types";
import { defaultFormState } from "../src/core/deck/formState";
import type { ServerProfile } from "@tenryu-common/core/profiles";
import { parseTmatGroupBounds } from "@tenryu-common/core/results/tmatBounds";
import { t } from "../src/i18n";
import { __setBackendForTest, useApp } from "../src/store";

class FakeBackend implements Backend {
  readonly kind = "devbridge" as const;
  readBinaryCalls: Array<{ path: string; maxBytes?: number }> = [];
  localExecLog: string[][] = [];
  localTextFiles: Record<string, string> = {};

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
  async saveLocalTextFile(): Promise<string | null> {
    return null;
  }
  async writeLocalTextFile(): Promise<void> {}
  async openLocalTextFile(): Promise<null> {
    return null;
  }
  async readBinary(
    _profile: ServerProfile,
    remotePath: string,
    maxBytes?: number,
  ): Promise<Uint8Array> {
    this.readBinaryCalls.push({ path: remotePath, maxBytes });
    return new Uint8Array(8);
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
    currentProfileId: null,
    tmatBoundsBusy: false,
    tmatBoundsError: null,
  });
  useApp.getState().loadForm(form);
});

describe("parseTmatGroupBounds", () => {
  it("returns null for an invalid HDF5 buffer", async () => {
    await expect(parseTmatGroupBounds(new ArrayBuffer(8))).resolves.toBeNull();
  });
});

describe("fetchTmatGroupBounds", () => {
  it("requires a current profile", async () => {
    await useApp.getState().fetchTmatGroupBounds();

    expect(useApp.getState().tmatBoundsError).toBe("NO_PROFILE");
    expect(fake.readBinaryCalls).toEqual([]);
  });

  it("requires a material with a TMAT opacity table", async () => {
    useApp.setState({ profiles: [profile], currentProfileId: profile.id });

    await useApp.getState().fetchTmatGroupBounds();

    expect(useApp.getState().tmatBoundsError).toBe(t().tmatBounds.noTmatMaterial);
    expect(fake.readBinaryCalls).toEqual([]);
  });

  it("leaves radiation bounds unchanged when the TMAT file cannot be parsed", async () => {
    useApp.setState({ profiles: [profile], currentProfileId: profile.id });
    useApp.getState().updateForm((form) => {
      form.materials[0].opacityModel = "tmat";
      form.materials[0].opacityFile = "/data/material.tmat.h5";
    });
    const before = structuredClone(useApp.getState().form.radiation);

    await useApp.getState().fetchTmatGroupBounds();

    expect(useApp.getState().tmatBoundsError).toBe(t().tmatBounds.parseFailed);
    expect(useApp.getState().form.radiation.groups).toBe(before.groups);
    expect(useApp.getState().form.radiation.groupBoundsEV).toEqual(before.groupBoundsEV);
    expect(fake.readBinaryCalls).toEqual([
      { path: "/data/material.tmat.h5", maxBytes: 33554432 },
    ]);
  });
});
