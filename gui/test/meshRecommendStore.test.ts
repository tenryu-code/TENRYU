import { beforeEach, describe, expect, it } from "vitest";
import type { Backend, ExecResult } from "@tenryu-common/backend/types";
import { newProfile, type ServerProfile } from "@tenryu-common/core/profiles";
import { defaultFormState, type FormState } from "../src/core/deck/formState";
import { recommendationConditionsKey } from "../src/core/deck/mesh1d";
import { __setBackendForTest, useApp } from "../src/store";
import { q } from "../src/core/units";
import type { RunRecord } from "@tenryu-common/core/runstate";

const OK: ExecResult = { code: 0, stdout: "", stderr: "", timedOut: false };

/** Minimal server: a checkout above the binary, tools/assist present, recommend-mesh writing a
 *  JSON result that readText returns. */
class FakeServer implements Partial<Backend> {
  readonly kind = "devbridge" as const;
  execs: string[][] = [];
  uploads: Array<{ path: string; content: string }> = [];
  files: Record<string, string> = {};
  recommendation = "";
  onRecommend: (() => void) | null = null;
  /** stdout of other bash scripts, by a pattern of the script. */
  scriptOutputs: Array<[RegExp, string]> = [];
  async listProfiles() {
    return [];
  }
  async saveProfiles() {}
  async getSettings() {
    return {};
  }
  async saveSettings() {}
  async exec(_profile: ServerProfile, argv: string[]): Promise<ExecResult> {
    this.execs.push(argv);
    const script = argv[0] === "bash" && argv[1] === "-lc" ? argv[2] : "";
    if (script.includes("mesh_planner")) return { ...OK, stdout: "/repo\n" };
    if (script.includes("ASSIST_OK")) return { ...OK, stdout: "ASSIST_OK\n" };
    if (script.includes("recommend-mesh")) {
      const out = argv[2].match(/-o (\S+)/)?.[1] ?? "";
      this.files[out] = this.recommendation;
      this.onRecommend?.();
      return OK;
    }
    for (const [pattern, stdout] of this.scriptOutputs) {
      if (pattern.test(script)) return { ...OK, stdout };
    }
    return OK;
  }
  async uploadText(_profile: ServerProfile, path: string, content: string) {
    this.uploads.push({ path, content });
  }
  async readText(_profile: ServerProfile, path: string) {
    if (!(path in this.files)) throw new Error(`no file ${path}`);
    return this.files[path];
  }
}

const profile: ServerProfile = {
  ...newProfile(),
  id: "p1",
  name: "server",
  host: "server",
  tenryuBin: "/repo/build/tenryu",
  runDir: "/runs",
};

function laserForm(): FormState {
  const f = defaultFormState();
  f.main.geometry1d = "planar";
  f.mesh.rMax = q(125, "µm");
  f.geometry.regions = [{ materialName: f.materials[0].name, rOuter: q(25, "µm"), rho: 1.05, Te: q(0.025, "eV"), Ti: q(0.025, "eV") }];
  f.geometry.vacuumOutside1d = true;
  f.laser.enabled = true;
  f.laser.ghostCorona.enabled = true;
  f.laser.powerW = q(100, "TW");
  f.laser.pulseDuration = q(1, "ns");
  f.radiation.enabled = false;
  return f;
}

function recommendation(status = "validated"): string {
  return JSON.stringify({
    recommendation: { surface_areal_mass_g_cm2: 8.9e-7, mode: "apriori_fallback" },
    evidence: [],
    flags: ["extrapolation"],
    warnings: ["material outside CD"],
    confidence: "outside campaign; convergence pair required",
    validation: { status, attempts: [], achieved_surface_areal_mass_g_cm2: 8.0e-7 },
    mesh: {
      r_min: 0.0,
      r_max: 0.0125,
      geometry_1d: "planar",
      zoning_intent: {
        n_cells: 1460,
        measure: "areal_mass",
        density_regions: [
          { r_end: 0.0025, rho: 1.05 },
          { r_end: 0.0125, rho: 1e-9 },
        ],
        pins: [{ r: 0.0025, ratio_jump_allowed: true }],
        profile: [{ r: 0.0, w: 1e-6 }],
        bands: [{ measure_frac_begin: 0.5, measure_frac_end: 0.9999, cell_measure_max: 8.4e-7 }],
        dr_min: 4.0e-8,
        preferred_ratio: 1.3,
        ratio_hard_max: 1.3,
        min_cells_per_segment: 40,
      },
      resolution_requirement: { apply: "enforce" },
    },
  });
}

let server: FakeServer;

beforeEach(() => {
  server = new FakeServer();
  __setBackendForTest(server as unknown as Backend);
  useApp.setState({ profiles: [profile], currentProfileId: profile.id, meshRecommend: { status: "idle" } });
  useApp.getState().loadForm(laserForm());
});

describe("recommend-mesh from Studio", () => {
  it("runs the recommender on the server from the checkout and applies its mesh", async () => {
    server.recommendation = recommendation();
    await useApp.getState().recommendMesh();
    expect(useApp.getState().meshRecommend).toEqual({ status: "done" });
    const form = useApp.getState().form;
    expect(form.mesh.grid1d).toBe("recommended");
    expect(form.mesh.zoningIntent.nCells).toBe(1460);
    expect(form.mesh.recommendation?.conditionsKey).toBe(recommendationConditionsKey(form));
    expect(form.mesh.recommendation?.binary).toBe("/repo/build/tenryu");
    expect(useApp.getState().formErrors).toEqual([]);
    const run = server.execs.find((argv) => argv[2]?.includes("recommend-mesh"))!;
    expect(run[2]).toMatch(/^cd \/repo && python3 tools\/assist\/assist.py recommend-mesh --deck \S+\.py --deck-out \S+_out\.py -o \S+\.json --tenryu \/repo\/build\/tenryu$/);
    // The deck sent is the form with a placeholder mesh whose nodes include the interface.
    const deck = server.uploads.find((u) => u.path.endsWith("_recommend_mesh.py"))!.content;
    expect(deck).toContain("explicit_nodes=MESH_NODES");
    expect(deck).toMatch(/\b0\.0025,/);
  });

  it("does not apply a recommendation that failed validation", async () => {
    server.recommendation = recommendation("failed");
    await useApp.getState().recommendMesh();
    expect(useApp.getState().meshRecommend.status).toBe("error");
    expect(useApp.getState().meshRecommend.error).toBe("NOT_VALIDATED");
    expect(useApp.getState().form.mesh.grid1d).toBe("uniform");
  });

  it("drops the result when the form changed while the recommender ran", async () => {
    server.recommendation = recommendation();
    server.onRecommend = () => useApp.getState().updateForm((f) => { f.laser.powerW = q(200, "TW"); });
    await useApp.getState().recommendMesh();
    expect(useApp.getState().meshRecommend.error).toBe("FORM_CHANGED");
    expect(useApp.getState().form.mesh.grid1d).toBe("uniform");
  });

  it("refuses decks without a laser and decks with a material corona ramp", async () => {
    useApp.getState().updateForm((f) => { f.laser.enabled = false; });
    await useApp.getState().recommendMesh();
    expect(useApp.getState().meshRecommend.error).toBe("NEEDS_LASER");
    useApp.getState().updateForm((f) => {
      f.laser.enabled = true;
      f.geometry.coronaRamp1d.enabled = true;
    });
    await useApp.getState().recommendMesh();
    expect(useApp.getState().meshRecommend.error).toBe("CORONA_RAMP");
    expect(server.execs.some((argv) => argv[2]?.includes("recommend-mesh"))).toBe(false);
  });
});

describe("the mesh of a run", () => {
  it("reads the nodes from the frozen configuration the solver writes in config/", async () => {
    const run = {
      id: "r1",
      profileId: profile.id,
      profileName: profile.name,
      name: "laser_foil",
      runDir: "/runs/r1",
      statusPath: "/runs/r1/status.json",
      tEnd: 1.5e-9,
      maxSteps: 1000,
      createdAtIso: "",
      state: "finished",
      pid: null,
      exitCode: 0,
      stopRequested: false,
      lastProgress: null,
      startEpoch: null,
      endEpoch: null,
      launchError: null,
    } as RunRecord;
    useApp.setState({ runs: [run] });
    const frozen = "/runs/r1/outputs/laser_foil//config/laser_foil_frozen.json";
    server.scriptOutputs = [
      [/^ls -td /, "/runs/r1/outputs/laser_foil/\n"],
      [/config\/\*_frozen\.json/, `${frozen}\n`],
    ];
    server.files[frozen] = JSON.stringify({ mesh: { explicit_nodes: [0, 0.001, 0.0025, 0.0125] } });
    const result = await useApp.getState().readRunNodes("r1");
    expect(result).toEqual({ nodes: [0, 0.001, 0.0025, 0.0125], source: "laser_foil: laser_foil_frozen.json" });
    // No solver call was needed.
    expect(server.execs.some((argv) => argv.join(" ").includes("--mesh-preview"))).toBe(false);
  });
});

describe("inRepoRoot", () => {
  it("runs from the checkout and keeps relative paths relative to the login directory", async () => {
    const { mkdtempSync, mkdirSync, writeFileSync, chmodSync, realpathSync } = await import("node:fs");
    const { join } = await import("node:path");
    const { tmpdir } = await import("node:os");
    const { spawnSync } = await import("node:child_process");
    const { inRepoRoot } = await import("../src/store");
    const base = realpathSync(mkdtempSync(join(tmpdir(), "inreporoot-")));
    const login = join(base, "home");
    const root = join(base, "repo with space");
    mkdirSync(join(login, "bin"), { recursive: true });
    mkdirSync(join(login, "runs"), { recursive: true });
    mkdirSync(root, { recursive: true });
    const fake = join(login, "bin", "fake");
    writeFileSync(fake, '#!/bin/bash\necho "pwd=$PWD"\necho "repo=$TENRYU_REPO"\nfor a in "$@"; do echo "arg=$a"; done\n');
    chmodSync(fake, 0o755);
    const argv = inRepoRoot(root, ["bin/fake", "validate", "runs/x.py", "--mesh-preview"]);
    expect(argv.slice(0, 2)).toEqual(["bash", "-lc"]);
    const r = spawnSync("bash", ["-c", argv[2]], { cwd: login, encoding: "utf8" });
    expect(r.status).toBe(0);
    expect(r.stdout.split("\n").filter((line) => line.length > 0)).toEqual([
      `pwd=${root}`,
      `repo=${root}`,
      "arg=validate",
      `arg=${join(login, "runs/x.py")}`,
      "arg=--mesh-preview",
    ]);
    expect(inRepoRoot(null, ["a/b", "c"])).toEqual(["a/b", "c"]);
  });
});
