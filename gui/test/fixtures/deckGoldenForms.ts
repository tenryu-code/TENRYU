import { defaultFormState, type FormState } from "../../src/core/deck/formState";
import { q } from "../../src/core/units";

// The 1D presets of Studio before 2026-10-01 (uniform meshes), kept as fixed forms for the
// generated-deck golden cases below.
function logspaceBounds(loEV: number, hiEV: number, nGroups: number): number[] {
  const out: number[] = [];
  const a = Math.log10(loEV);
  const b = Math.log10(hiEV);
  for (let i = 0; i <= nGroups; i++) out.push(10 ** (a + ((b - a) * i) / nGroups));
  return out;
}

/** 平面 1D 放射スラブ: 灰色 FLD + Marshak 120 eV、流体 OFF。
 *  κ=2000 cm²/g・cv_e_override=3e11 erg/(cm³·eV)（体積あたり）で 1 ns に壁 ~104 eV・前線 ~35 µm・前方冷域が立つ実証済み構成。 */
function presetSlabRadiation(): FormState {
  const f = defaultFormState();
  f.main.name = "slab_radiation";
  f.main.geometry1d = "planar";
  f.main.temperatureModel = "1T";
  f.main.tEnd = q(1.0, "ns");
  f.mesh.rMax = q(0.01, "cm");
  f.mesh.nr = 200;
  f.materials[0].name = "ch_slab";
  f.materials[0].cvEOverride = 3.0e11;
  f.materials[0].kappaA = 2000.0;
  f.geometry.regions[0].materialName = "ch_slab";
  f.geometry.regions[0].rOuter = q(0.01, "cm");
  f.geometry.regions[0].rho = 0.2;
  f.geometry.radiationField = "zero";
  f.radiation.outerR = "marshak";
  f.radiation.marshakMode = "constant";
  f.radiation.marshakTrEV = 120.0;
  f.hydro.enabled = false;
  f.conduction.enabled = false;
  f.numerics.floors.TeFloorEV = 1.0;
  f.numerics.floors.TiFloorEV = 1.0;
  return f;
}

/** 球 1D 直接照射カプセル (GXII 検証レジーム): DT ガス + CH シェル + コロナ ramp + VOID 外側、
 *  raytrace 10 TW × 1 ns。gxii_1d_fld_regression と同一の物理条件系。 */
function presetLaserSphere(): FormState {
  const f = defaultFormState();
  f.main.name = "laser_capsule";
  f.main.maxSteps = 2_000_000;
  f.mesh.nr = 200;
  f.materials = [
    { name: "CH", A: 6.5, Z: 3.5, eosModel: "ideal_gas", gamma: 5 / 3, cvEOverride: undefined, eosFile: "", opacityModel: "constant", kappaA: 100.0, kappaS: 0.0, opacityFile: "" },
    { name: "DT", A: 2.5, Z: 1.0, eosModel: "ideal_gas", gamma: 5 / 3, cvEOverride: undefined, eosFile: "", opacityModel: "constant", kappaA: 100.0, kappaS: 0.0, opacityFile: "" },
  ];
  f.geometry.regions = [
    { materialName: "DT", rOuter: q(230, "µm"), rho: 0.010, Te: q(0.025, "eV"), Ti: q(0.025, "eV") },
    { materialName: "CH", rOuter: q(250, "µm"), rho: 1.05, Te: q(0.025, "eV"), Ti: q(0.025, "eV") },
  ];
  f.geometry.vacuumOutside1d = true;
  f.geometry.coronaRamp1d = { enabled: true, scaleUm: 2.0, extentUm: 10.0, rho0: 0.05, rhoMin: 3.0e-4 };
  f.radiation.groups = 20;
  f.radiation.groupBoundsEV = logspaceBounds(0.01, 100.0, 20);
  f.laser.enabled = true;
  f.laser.mode = "raytrace_2d";
  f.laser.powerW = q(10.0, "TW");
  f.laser.pulseDuration = q(1.0, "ns");
  f.laser.riseTime = q(10, "ps");
  f.laser.fallTime = q(10, "ps");
  f.laser.beams[0].fNumber = 3.0;
  f.laser.beams[0].w0Um = 200.0;
  f.numerics.dtInitial = q(1e-15, "s");
  f.numerics.dtMax = q(1e-11, "s");
  return f;
}

/** 球 1D 間接照射 (テンプレ③ 相当): Tr(t) 折れ線 Marshak 駆動、fuel+ablator。 */
function presetIndirectTr(): FormState {
  const f = defaultFormState();
  f.main.name = "indirect_tr";
  f.main.tEnd = q(5.0, "ns");
  f.mesh.rMax = q(0.033, "cm");
  f.mesh.nr = 200;
  f.materials = [
    { name: "fuel", A: 2.5, Z: 1.0, eosModel: "ideal_gas", gamma: 5 / 3, cvEOverride: undefined, eosFile: "", opacityModel: "constant", kappaA: 1.0, kappaS: 0.0, opacityFile: "" },
    { name: "ablator", A: 6.5, Z: 3.5, eosModel: "ideal_gas", gamma: 5 / 3, cvEOverride: undefined, eosFile: "", opacityModel: "constant", kappaA: 2000.0, kappaS: 0.0, opacityFile: "" },
  ];
  f.geometry.regions = [
    { materialName: "fuel", rOuter: q(300, "µm"), rho: 0.01, Te: q(1e-3, "eV"), Ti: q(1e-3, "eV") },
    { materialName: "ablator", rOuter: q(0.033, "cm"), rho: 1.05, Te: q(1e-3, "eV"), Ti: q(1e-3, "eV") },
  ];
  f.geometry.radiationField = "zero";
  f.radiation.outerR = "marshak";
  f.radiation.marshakMode = "table";
  f.radiation.marshakPoints = [
    { t: 0, v: 120 },
    { t: 0.5, v: 120 },
    { t: 1, v: 200 },
    { t: 5, v: 200 },
  ];
  f.numerics.dtInitial = q(1e-15, "s");
  f.numerics.floors.TeFloorEV = 1e-3;
  f.numerics.floors.TiFloorEV = 1e-3;
  return f;
}

interface GoldenCase {
  name: string;
  form: FormState;
  expectSummary: Array<[string, string]>;
}

export function buildCases(): GoldenCase[] {
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
    f.geometry.background2d.materialName = f.materials[0].name;
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
    f.geometry.background2d.materialName = f.materials[0].name;
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
    f.laser.cbet.detuneSplitNm = 1.0;
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

