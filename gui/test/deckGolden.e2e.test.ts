import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { defaultFormState, type FormState } from "../src/core/deck/formState";
import { presetIndirectTr, presetLaserSphere, presetSlabRadiation } from "../src/core/presets";
import { generateDeck } from "../src/core/deck/generate";
import { parseValidateOutput } from "../src/core/validateParse";
import { q } from "../src/core/units";

const BIN = process.env.TENRYU_BIN && fs.existsSync(process.env.TENRYU_BIN) ? process.env.TENRYU_BIN : "";

interface GoldenCase {
  name: string;
  form: FormState;
  expectSummary: Array<[string, string]>;
}

function buildCases(): GoldenCase[] {
  const cases: GoldenCase[] = [];

  {
    const f = defaultFormState();
    f.main.name = "golden_slab_fld_marshak";
    f.main.geometry1d = "planar";
    f.main.temperatureModel = "1T";
    f.mesh.rMax = q(0.06, "cm");
    f.geometry.regions[0].rOuter = q(0.06, "cm");
    f.geometry.regions[0].rho = 0.2;
    f.geometry.radiationField = "zero";
    f.materials[0].cvEOverride = 8.68e11;
    f.radiation.outerR = "marshak";
    f.radiation.marshakTrEV = 120.0;
    f.hydro.enabled = false;
    f.conduction.enabled = false;
    f.numerics.floors.TeFloorEV = 1.0;
    f.numerics.floors.TiFloorEV = 1.0;
    cases.push({
      name: "slab_fld_marshak",
      form: f,
      expectSummary: [
        ["main", "geometry=planar"],
        ["mesh", "nr=300"],
        ["radiation", "mode=multigroup_diffusion"],
        ["radiation", "outer=marshak"],
        ["laser", "enabled=false"],
        ["hydro", "enabled=false"],
      ],
    });
  }

  {
    const f = defaultFormState();
    f.main.name = "golden_laser_sphere";
    f.laser.enabled = true;
    f.conduction.fLim = 0.06;
    cases.push({
      name: "laser_sphere",
      form: f,
      expectSummary: [
        ["main", "dimension=1D_SPH"],
        ["laser", "enabled=true"],
        ["conduction", "enabled=true"],
      ],
    });
  }

  {
    const f = defaultFormState();
    f.main.name = "golden_two_material_tr";
    f.materials = [
      { name: "fuel", A: 2.5, Z: 1.0, eosModel: "ideal_gas", gamma: 5 / 3, eosFile: "", opacityModel: "constant", kappaA: 1.0, kappaS: 0.0, opacityFile: "" },
      { name: "ablator", A: 6.5, Z: 3.5, eosModel: "ideal_gas", gamma: 5 / 3, eosFile: "", opacityModel: "constant", kappaA: 200.0, kappaS: 0.0, opacityFile: "" },
    ];
    f.mesh.rMax = q(0.033, "cm");
    f.mesh.nr = 200;
    f.geometry.regions = [
      { materialName: "fuel", rOuter: q(300, "µm"), rho: 0.01, Te: q(1e-3, "eV"), Ti: q(1e-3, "eV") },
      { materialName: "ablator", rOuter: q(0.033, "cm"), rho: 1.05, Te: q(1e-3, "eV"), Ti: q(1e-3, "eV") },
    ];
    f.geometry.radiationField = "zero";
    f.radiation.outerR = "marshak";
    f.radiation.marshakTrEV = 160.0;
    cases.push({
      name: "two_material_tr",
      form: f,
      expectSummary: [
        ["mesh", "nr=200"],
        ["radiation", "outer=marshak"],
      ],
    });
  }

  {
    const f = defaultFormState();
    f.main.name = "golden_sn_vacuum";
    f.radiation.mode = "sn_transport";
    f.radiation.snNAngles = 8;
    cases.push({
      name: "sn_vacuum",
      form: f,
      expectSummary: [["radiation", "mode=sn_transport"]],
    });
  }

  {
    const f = defaultFormState();
    f.main.name = "golden_sn_marshak";
    f.radiation.mode = "sn_transport";
    f.radiation.outerR = "marshak";
    f.radiation.marshakTrEV = 120.0;
    cases.push({
      name: "sn_marshak",
      form: f,
      expectSummary: [
        ["radiation", "mode=sn_transport"],
        ["radiation", "outer=marshak"],
      ],
    });
  }

  {
    const f = defaultFormState();
    f.main.name = "golden_multigroup4";
    f.radiation.groups = 4;
    f.radiation.groupBoundsEV = [0.1, 10.0, 100.0, 1000.0, 100000.0];
    cases.push({
      name: "multigroup4",
      form: f,
      expectSummary: [["radiation", "groups=4"]],
    });
  }

  {
    const f = defaultFormState();
    f.main.name = "golden_2d_rz_fld";
    f.main.dimension = "2D_RZ";
    f.mesh.nr = 16;
    f.mesh.nz = 32;
    f.hydro.enabled = false;
    f.conduction.enabled = false;
    cases.push({
      name: "rz_fld",
      form: f,
      expectSummary: [
        ["main", "dimension=2D_RZ"],
        ["radiation", "mode=multigroup_diffusion"],
      ],
    });
  }

  {
    const f = defaultFormState();
    f.main.name = "golden_rz_laser";
    f.main.dimension = "2D_RZ";
    f.mesh.nr = 16;
    f.mesh.nz = 32;
    f.conduction.enabled = false;
    f.laser.enabled = true;
    f.laser.beams.push({ ...f.laser.beams[0], name: "beam_01", axialDirection: "plus_z" });
    cases.push({
      name: "rz_laser_axial",
      form: f,
      expectSummary: [
        ["main", "dimension=2D_RZ"],
        ["laser", "enabled=true"],
      ],
    });
  }

  {
    const f = defaultFormState();
    f.main.name = "golden_graded_custom";
    f.mesh.grid1d = "graded";
    f.mesh.segments = [
      { rEnd: q(300, "µm"), nr: 120 },
      { rEnd: q(500, "µm"), nr: 100 },
    ];
    f.customPythonBlock = 'X_CUSTOM_NOTE = "golden custom block"';
    cases.push({
      name: "graded_custom",
      form: f,
      expectSummary: [["mesh", "nr=220"]],
    });
  }

  {
    const f = presetSlabRadiation();
    f.main.name = "golden_preset_slab";
    cases.push({
      name: "preset_slab",
      form: f,
      expectSummary: [
        ["main", "geometry=planar"],
        ["radiation", "outer=marshak"],
      ],
    });
  }

  {
    const f = presetLaserSphere();
    f.main.name = "golden_preset_laser_sphere";
    cases.push({
      name: "preset_laser_sphere",
      form: f,
      expectSummary: [["laser", "enabled=true"]],
    });
  }

  {
    const f = presetIndirectTr();
    f.main.name = "golden_preset_indirect_tr";
    cases.push({
      name: "preset_indirect_tr",
      form: f,
      expectSummary: [
        ["radiation", "outer=marshak"],
        ["mesh", "nr=200"],
      ],
    });
  }

  {
    const f = presetLaserSphere();
    f.main.name = "golden_laser_table";
    f.laser.waveformMode = "table";
    f.laser.waveformPoints = [
      { t: 0, v: 0.2 },
      { t: 0.5, v: 1.0 },
      { t: 2, v: 0.0 },
    ];
    cases.push({
      name: "laser_table",
      form: f,
      expectSummary: [["laser", "enabled=true"]],
    });
  }

  {
    const f = defaultFormState();
    f.main.name = "golden_sn_marshak_table";
    f.radiation.mode = "sn_transport";
    f.radiation.outerR = "marshak";
    f.radiation.marshakMode = "table";
    cases.push({
      name: "sn_marshak_table",
      form: f,
      expectSummary: [["radiation", "mode=sn_transport"]],
    });
  }

  {
    const f = presetLaserSphere();
    f.main.name = "golden_pvisc";
    f.hydro.plasmaVisc.enabled = true;
    f.hydro.plasmaVisc.species = "both";
    cases.push({
      name: "pvisc_laser_sphere",
      form: f,
      expectSummary: [
        ["hydro", "enabled=true"],
        ["laser", "enabled=true"],
      ],
    });
  }

  {
    const f = defaultFormState();
    f.main.name = "golden_burn";
    f.burn.enabled = true;
    f.burn.fuelMaterials = "CH";
    f.burn.neutronHeating = true;
    cases.push({
      name: "burn_sphere",
      form: f,
      expectSummary: [["main", "dimension=1D_SPH"]],
    });
  }

  {
    const f = presetLaserSphere();
    f.main.name = "golden_hote";
    f.laser.hotE.enabled = true;
    f.laser.hotE.etaHot = 0.02;
    cases.push({
      name: "hote_laser_sphere",
      form: f,
      expectSummary: [["laser", "enabled=true"]],
    });
  }

  {
    const f = defaultFormState();
    f.main.name = "golden_snb";
    f.conduction.nonlocalModel = "snb";
    cases.push({
      name: "snb_conduction",
      form: f,
      expectSummary: [["conduction", "enabled=true"]],
    });
  }

  {
    const f = presetLaserSphere();
    f.main.name = "golden_cbet";
    f.laser.mode = "raytrace_2d";
    f.laser.cbet.enabled = true;
    f.laser.beams[0].deltaLambdaNm = 0.5;
    cases.push({
      name: "cbet_raytrace",
      form: f,
      expectSummary: [["laser", "enabled=true"]],
    });
  }

  {
    const f = presetLaserSphere();
    f.main.name = "golden_table_profile";
    f.laser.beams[0].profileModel = "table";
    f.laser.beams[0].profilePoints = [
      { t: 0, v: 1.0 },
      { t: 150, v: 0.6 },
      { t: 300, v: 0.0 },
    ];
    cases.push({
      name: "table_profile_laser",
      form: f,
      expectSummary: [["laser", "enabled=true"]],
    });
  }

  return cases;
}

describe.skipIf(!BIN)("golden gate: generated decks pass real `tenryu validate`", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "tenryu-golden-"));

  for (const c of buildCases()) {
    it(
      `${c.name} validates PASS`,
      () => {
        const deck = generateDeck(c.form);
        const dir = path.join(tmp, c.name);
        fs.mkdirSync(dir, { recursive: true });
        const deckPath = path.join(dir, "deck.py");
        fs.writeFileSync(deckPath, deck);
        const r = spawnSync(BIN, ["validate", deckPath], {
          encoding: "utf8",
          timeout: 120000,
          cwd: dir,
        });
        const parsed = parseValidateOutput(r.stdout ?? "", r.stderr ?? "", r.status);
        if (!parsed.ok) {
          console.error(`--- deck (${c.name}) ---\n${deck}\n--- stderr ---\n${r.stderr}`);
        }
        expect(parsed.ok).toBe(true);
        const byLabel = new Map<string, string>();
        for (const row of parsed.summary) {
          byLabel.set(row.label, (byLabel.get(row.label) ?? "") + row.text + " ");
        }
        for (const [label, substr] of c.expectSummary) {
          const text = byLabel.get(label) ?? "";
          expect(text, `summary[${label}] should contain "${substr}"`).toContain(substr);
        }
      },
      180000,
    );
  }
});

describe("golden gate presence", () => {
  it("warns when TENRYU_BIN is unset", () => {
    if (!BIN) {
      console.warn("[golden] TENRYU_BIN unset or missing — golden validate gate SKIPPED");
    }
    expect(true).toBe(true);
  });
});
