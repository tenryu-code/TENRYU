import { describe, expect, it } from "vitest";
import { defaultFormState, makeHotEChannel, validateFormState } from "../src/core/deck/formState";
import { generateDeck, pyList, pyNum, pyStr } from "../src/core/deck/generate";
import { extractGuiState } from "../src/core/deck/roundtrip";
import { defaultShape2D } from "../src/core/geometry2d";
import { q } from "../src/core/units";

describe("py emitters", () => {
  it("pyNum", () => {
    expect(pyNum(0.06)).toBe("0.06");
    expect(pyNum(1e-9)).toBe("1e-9");
    expect(pyNum(300)).toBe("300");
    expect(() => pyNum(Number.NaN)).toThrow();
  });
  it("pyStr", () => {
    expect(pyStr("1D_SPH")).toBe('"1D_SPH"');
  });
  it("pyList", () => {
    expect(pyList([0.1, 1e5])).toBe("[0.1, 100000]");
  });
});

describe("generateDeck", () => {
  it("default form validates clean and generates deterministically", () => {
    const f = defaultFormState();
    expect(validateFormState(f)).toEqual([]);
    const a = generateDeck(f);
    const b = generateDeck(f);
    expect(a).toBe(b);
    expect(a).toContain('from tenryu_namelist import *');
    expect(a).toContain('dimension="1D_SPH"');
    expect(a).toContain('geometry_1d="spherical"');
    expect(a).toContain("r_max=0.05,  # 500 µm");
    expect(a).toContain('verbosity="normal"');
    expect(a).toContain("T_start_eV=0");
    expect(a).toContain("Burn(enabled=False)");
  });

  it("round-trips the exact form state", () => {
    const f = defaultFormState();
    f.main.name = "roundtrip_case";
    f.geometry.regions[0].Te = q(150, "eV");
    const deck = generateDeck(f);
    const r = extractGuiState(deck);
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.state).toEqual(f);
  });

  it("emits piecewise regions in cm", () => {
    const f = defaultFormState();
    f.materials.push({ name: "fuel", A: 2.5, Z: 1.0, eosModel: "ideal_gas", gamma: 5 / 3, eosFile: "", opacityModel: "constant", kappaA: 1.0, kappaS: 0.0, opacityFile: "" });
    f.geometry.regions = [
      { materialName: "fuel", rOuter: q(300, "µm"), rho: 0.01, Te: q(1e-3, "keV"), Ti: q(1, "eV") },
      { materialName: "CH", rOuter: q(500, "µm"), rho: 1.05, Te: q(1, "eV"), Ti: q(1, "eV") },
    ];
    const deck = generateDeck(f);
    expect(deck).toContain("def gui_rho(r_cm):");
    expect(deck).toContain("    if r_cm < 0.03: return 0.01");
    expect(deck).toContain("    return 1.05");
    expect(deck).toContain("def gui_vf_fuel(r_cm):");
    expect(deck).toContain("def gui_vf_CH(r_cm):");
    expect(deck).toContain("        fuel=gui_vf_fuel,");
    expect(deck).toContain("Te=gui_Te");
  });

  it("rejects region/mesh mismatch", () => {
    const f = defaultFormState();
    f.geometry.regions[0].rOuter = q(400, "µm");
    expect(validateFormState(f).length).toBeGreaterThan(0);
    expect(() => generateDeck(f)).toThrow();
  });

  it("2D form emits z keys and two-argument functions", () => {
    const f = defaultFormState();
    f.main.dimension = "2D_RZ";
    f.mesh.nr = 16;
    f.mesh.nz = 32;
    f.hydro.enabled = false;
    const shape = defaultShape2D("solidSphere");
    shape.materialName = "CH";
    shape.radius = q(500, "µm");
    f.geometry.regions = [];
    f.geometry.shapes2d = [shape];
    f.geometry.background2d = {
      materialName: "CH",
      rho: 1.0e-4,
      Te: q(1, "eV"),
      Ti: q(1, "eV"),
    };
    const deck = generateDeck(f);
    expect(deck).toContain("z_min=-0.05,  # -500 µm");
    expect(deck).toContain('motion="lagrangian"');
    expect(deck).toContain("_GUI_SHAPES = [");
    expect(deck).toContain("def gui_vf_CH(r_cm, z_cm):");
    expect(deck).toContain('z="vacuum"');
    expect(deck).not.toContain("geometry_1d");
    expect(deck).not.toContain("boundary_1d");
  });

  it("custom python block is appended verbatim and round-trips", () => {
    const f = defaultFormState();
    f.customPythonBlock = 'X_CUSTOM_NOTE = "gui custom block"';
    const deck = generateDeck(f);
    expect(deck).toContain('X_CUSTOM_NOTE = "gui custom block"');
    const r = extractGuiState(deck);
    expect(r.ok && r.state.customPythonBlock === f.customPythonBlock).toBe(true);
  });

  it("extractGuiState failure modes", () => {
    expect(extractGuiState("print(1)").ok).toBe(false);
    const bad = "# TENRYU-GUI-STATE: {not json}";
    const r = extractGuiState(bad);
    expect(r.ok).toBe(false);
  });

  it("gaussian waveform emits math.exp and import math", () => {
    const f = defaultFormState();
    f.main.dimension = "1D_SPH";
    f.laser.enabled = true;
    f.laser.waveformMode = "gaussian";
    f.laser.gaussianSpec = "peak";
    f.laser.gaussianPeakW = q(2, "TW");
    f.laser.gaussianFwhm = q(1, "ns");
    f.laser.gaussianCenter = q(2, "ns");
    const deck = generateDeck(f);
    expect(deck).toContain("import math");
    expect(deck).toContain("math.exp(-2.772588722239781");
    expect(deck).toContain("def gui_beam_power");
    expect(deck).toContain("math.exp");
  });

  it("square with rise/fall emits a trapezoid pwl", () => {
    const f = defaultFormState();
    f.main.dimension = "1D_SPH";
    f.laser.enabled = true;
    f.laser.riseTime = q(0.1, "ns");
    f.laser.fallTime = q(0.2, "ns");
    f.laser.pulseDuration = q(1, "ns");
    f.laser.powerW = q(1, "TW");
    const deck = generateDeck(f);
    expect(deck).toContain("_gui_pwl");
    expect(deck).toContain("trapezoid");
  });

  it("emits a constant external pressure drive", () => {
    const f = defaultFormState();
    f.main.dimension = "1D_SPH";
    f.hydro.boundary1d = "pressure";
    f.hydro.boundaryPressure = {
      mode: "constant",
      value: { value: 2, unit: "Mbar" },
      points: [],
    };
    const deck = generateDeck(f);
    expect(deck).toContain('boundary_1d="pressure"');
    expect(deck).toContain("boundary_pressure=(lambda t_s: 2000000000000)");
  });

  it("emits a piecewise external pressure drive table", () => {
    const f = defaultFormState();
    f.main.dimension = "1D_SPH";
    f.hydro.boundary1d = "pressure";
    f.hydro.boundaryPressure = {
      mode: "table",
      value: { value: 1, unit: "Mbar" },
      points: [{ t: 0, v: 0.5 }, { t: 2, v: 3 }],
    };
    const deck = generateDeck(f);
    expect(deck).toContain("def gui_boundary_pressure(t_s):");
    expect(deck).toContain("boundary_pressure=gui_boundary_pressure");
    expect(deck).toContain("_gui_pwl");
  });

  it("f_number and focus are emitted when set", () => {
    const f = defaultFormState();
    f.main.dimension = "1D_SPH";
    f.laser.enabled = true;
    f.laser.beams[0].fNumber = 6.7;
    f.laser.beams[0].focusZUm = -2000;
    const deck = generateDeck(f);
    expect(deck).toContain("f_number=6.7");
    expect(deck).toContain("focus=(0.0, 0.0, -0.2)");
  });

  it("tmat material emits table eos/opacity", () => {
    const f = defaultFormState();
    f.materials[0].eosModel = "tmat";
    f.materials[0].eosFile = "/data/ch.tmat.h5";
    f.materials[0].opacityModel = "tmat";
    f.materials[0].opacityFile = "/data/ch.tmat.h5";
    const deck = generateDeck(f);
    expect(deck).toContain(`eos=dict(model="tmat", file="/data/ch.tmat.h5")`);
    expect(deck).toContain(`opacity=dict(model="tmat", file="/data/ch.tmat.h5")`);
    expect(deck).not.toContain("ideal_gas=dict(gamma=");
  });

  it("regions-source graded mesh derives segments from material regions", () => {
    const f = defaultFormState();
    f.main.geometry1d = "planar";
    f.mesh.rMax = q(400, "µm");
    f.mesh.grid1d = "graded";
    f.mesh.segmentSource = "regions";
    f.mesh.nr = 100;
    f.materials.push({
      name: "Au",
      A: 197,
      Z: 79,
      eosModel: "ideal_gas",
      gamma: 5 / 3,
      cvEOverride: undefined,
      eosFile: "",
      opacityModel: "constant",
      kappaA: 1.0,
      kappaS: 0.0,
      opacityFile: "",
    });
    f.geometry.regions = [
      { materialName: "CH", rOuter: q(200, "µm"), rho: 1.0, Te: q(1, "eV"), Ti: q(1, "eV") },
      { materialName: "Au", rOuter: q(400, "µm"), rho: 3.0, Te: q(1, "eV"), Ti: q(1, "eV") },
    ];
    const deck = generateDeck(f);
    expect(deck).toContain('"r_end": 0.02');
    expect(deck).toContain('"r_end": 0.04');
    expect(deck).toContain('"nr": 25');
    expect(deck).toContain('"nr": 75');
    expect(deck).toContain("nr=100,");
    expect(deck).toContain("auto-derived from material regions");
  });

  it("plasma viscosity emits the hydro sub-dict when enabled", () => {
    const f = defaultFormState();
    f.hydro.plasmaVisc.enabled = true;
    f.hydro.plasmaVisc.species = "both";
    const deck = generateDeck(f);
    expect(deck).toContain(`plasma_viscosity=dict(enabled=True, model="braginskii", species="both"`);
    expect(deck).toContain(`dt_safety=0.3`);
  });

  it("plasma viscosity absent when disabled", () => {
    const f = defaultFormState();
    expect(generateDeck(f)).not.toContain("plasma_viscosity");
  });

  it("burn emits the full block when enabled", () => {
    const f = defaultFormState();
    f.burn.enabled = true;
    f.burn.fuelMaterials = "CH";
    const deck = generateDeck(f);
    expect(deck).toContain('fuels=["DT", "DD"]');
    expect(deck).toContain('scheme="fraley"');
    expect(deck).toContain('fuel_materials=["CH"]');
    expect(deck).toContain("neutron_heating=False");
    expect(deck).not.toContain("Burn(enabled=False)");
  });

  it("burn stays a one-liner when disabled", () => {
    const f = defaultFormState();
    const deck = generateDeck(f);
    expect(deck).toContain("Burn(enabled=False)");
  });

  it("burn diffusion scheme emits group knobs", () => {
    const f = defaultFormState();
    f.burn.enabled = true;
    f.burn.fuelMaterials = "CH";
    f.burn.scheme = "diffusion";
    const deck = generateDeck(f);
    expect(deck).toContain("diffusion_groups=30");
    expect(deck).toContain("diffusion_E_min_keV=20");
  });

  it("hot electrons emit the laser sub-dict when enabled", () => {
    const f = defaultFormState();
    f.laser.enabled = true;
    f.laser.hotE.enabled = true;
    f.laser.hotE.etaHot = 0.02;
    const deck = generateDeck(f);
    expect(deck).toContain("hot_electron=dict(");
    expect(deck).toContain("eta_hot=0.02");
    expect(deck).toContain('angular_model="cone"');
    expect(deck).toContain("theta_div_deg=60");
    expect(deck).not.toContain("inner_bc=");
  });

  it("hot electrons radial model emits inner_bc and no theta", () => {
    const f = defaultFormState();
    f.laser.enabled = true;
    f.laser.hotE.enabled = true;
    f.laser.hotE.angularModel = "radial";
    const deck = generateDeck(f);
    expect(deck).toContain('inner_bc="deposit_residual"');
    expect(deck).not.toContain("theta_div_deg");
  });

  it("hot electrons eta table emits the pwl helper", () => {
    const f = defaultFormState();
    f.laser.enabled = true;
    f.laser.hotE.enabled = true;
    f.laser.hotE.etaMode = "table";
    f.laser.hotE.etaPoints = [{ t: 0, v: 0 }, { t: 1, v: 0.03 }];
    const deck = generateDeck(f);
    expect(deck).toContain("def gui_hote_eta");
    expect(deck).toContain("eta_hot_table=gui_hote_eta");
    expect(deck).not.toContain("eta_hot=0");
  });

  it("hot-electron channels + eta model deck", () => {
    const f = defaultFormState();
    f.laser.enabled = true;
    f.laser.hotE.enabled = true;
    f.laser.hotE.useChannels = true;
    f.laser.hotE.etaEvolution = "model";
    f.laser.hotE.subtractFromLaser = true;
    f.laser.hotE.channels = [makeHotEChannel("srs"), makeHotEChannel("tpd")];
    const deck = generateDeck(f);
    expect(deck).toContain('eta_mode="model"');
    expect(deck).toContain("eta_model=dict(ln_filter_tau_s=5e-12, eta_total_cap=0.08)");
    expect(deck).toContain("sources=[");
    expect(deck).toContain('mechanism="srs"');
    expect(deck).toContain("eval_nc_fraction=0.18");
    expect(deck).toContain("tpd_theta_deg=45");
    expect(deck).not.toContain("source_nc_fraction");
    expect(deck).not.toContain("eta_hot");
    expect(deck).not.toMatch(/^\s+eta=/m);
  });

  it("hot-electron legacy channels with per-channel table", () => {
    const f = defaultFormState();
    f.laser.enabled = true;
    f.laser.hotE.enabled = true;
    f.laser.hotE.useChannels = true;
    f.laser.hotE.etaEvolution = "legacy";
    f.laser.hotE.subtractFromLaser = true;
    const srs = makeHotEChannel("srs");
    srs.etaMode = "table";
    srs.etaPoints = [{ t: 0, v: 0 }, { t: 1, v: 0.02 }];
    f.laser.hotE.channels = [srs, makeHotEChannel("tpd")];
    const deck = generateDeck(f);
    expect(deck).toContain("def gui_hote_eta_ch0");
    expect(deck).toContain("eta_table=gui_hote_eta_ch0");
    expect(deck).toContain("eta=0");
  });

  it("snb conduction emits the nonlocal knobs", () => {
    const f = defaultFormState();
    f.conduction.nonlocalModel = "snb";
    const deck = generateDeck(f);
    expect(deck).toContain('nonlocal_model="snb"');
    expect(deck).toContain("snb_n_groups=24");
    expect(deck).toContain('snb_mfp="geometric_r2"');
  });

  it("ion conduction emits only when on", () => {
    const f = defaultFormState();
    f.conduction.ionConduction = true;
    const deck = generateDeck(f);
    expect(deck).toContain("ion_conduction=True");
    expect(generateDeck(defaultFormState())).not.toContain("ion_conduction");
  });

  it("raytrace mode emits rays_per_beam", () => {
    const f = defaultFormState();
    f.laser.enabled = true;
    f.laser.mode = "raytrace_2d";
    const deck = generateDeck(f);
    expect(deck).toContain('mode="raytrace_2d"');
    expect(deck).toContain("rays_per_beam=1000");
  });

  it("emits ray trajectory keys only when requested and clamps the count", () => {
    const f = defaultFormState();
    f.laser.enabled = true;
    f.laser.mode = "raytrace_2d";
    f.laser.rayOutputTrajectory = true;
    f.laser.rayOutputCount = 1200;
    const flaggedDeck = generateDeck(f);
    expect(flaggedDeck).toContain("ray_output_trajectory=True");
    expect(flaggedDeck).toContain("ray_output_count=1000");

    f.laser.rayOutputTrajectory = false;
    const unflaggedDeck = generateDeck(f);
    expect(unflaggedDeck).not.toContain("ray_output_trajectory");
    expect(unflaggedDeck).not.toContain("ray_output_count");
  });

  it("cbet emits port_section laser sub-dict without per-beam delta lambda", () => {
    const f = defaultFormState();
    f.laser.enabled = true;
    f.laser.mode = "raytrace_2d";
    f.laser.cbet.enabled = true;
    f.laser.beams[0].deltaLambdaNm = 0;
    const deck = generateDeck(f);
    const beamsStart = deck.indexOf("    beams=[");
    const beamsEnd = deck.indexOf("\n    ],", beamsStart);
    const beamsBlock = deck.slice(beamsStart, beamsEnd + "\n    ],".length);
    expect(deck).toContain('geometry_mode="port_section"');
    expect(deck).toContain("port_configuration");
    expect(beamsBlock).not.toContain("delta_lambda_nm=");
  });

  it("cbet with radial mode is a validation error", () => {
    const f = defaultFormState();
    f.laser.enabled = true;
    f.laser.cbet.enabled = true;
    expect(() => generateDeck(f)).toThrow();
  });

  it("huge nr is rejected by validation, not a crash", () => {
    const f = defaultFormState();
    f.mesh.nr = 100_000_000;
    expect(() => generateDeck(f)).toThrow();
    expect(validateFormState(f).length).toBeGreaterThan(0);
  });

  it("two beams emit fraction-scaled powers", () => {
    const f = defaultFormState();
    f.main.dimension = "2D_RZ";
    f.mesh.nr = 16;
    f.mesh.nz = 32;
    f.hydro.enabled = false;
    f.conduction.enabled = false;
    f.laser.enabled = true;
    const shape = defaultShape2D("solidSphere");
    shape.materialName = "CH";
    shape.radius = q(500, "µm");
    f.geometry.regions = [];
    f.geometry.shapes2d = [shape];
    f.geometry.background2d = {
      materialName: "CH",
      rho: 1.0e-4,
      Te: q(1, "eV"),
      Ti: q(1, "eV"),
    };
    f.laser.beams.push({ ...f.laser.beams[0], name: "beam_01", powerFraction: 0.5 });
    const deck = generateDeck(f);
    expect(deck).toContain('name="beam_00"');
    expect(deck).toContain('name="beam_01"');
    expect(deck).toContain("lambda t_s, _f=0.5: _f * gui_beam_power(t_s)");
  });

  it("table beam profile emits r_um/I_rel lists", () => {
    const f = defaultFormState();
    f.laser.enabled = true;
    f.laser.beams[0].profileModel = "table";
    f.laser.beams[0].profilePoints = [{ t: 0, v: 1 }, { t: 100, v: 0.5 }, { t: 200, v: 0 }];
    const deck = generateDeck(f);
    expect(deck).toContain('profile=dict(model="table"');
    expect(deck).toContain("r_um=[0, 100, 200]");
    expect(deck).toContain("I_rel=[1, 0.5, 0]");
    expect(deck).not.toContain("super_gaussian");
  });

  it("2D laser emits raytrace_3d with axial beams", () => {
    const f = defaultFormState();
    f.main.dimension = "2D_RZ";
    f.mesh.nr = 16;
    f.mesh.nz = 32;
    f.hydro.enabled = false;
    f.conduction.enabled = false;
    f.laser.enabled = true;
    const shape = defaultShape2D("solidSphere");
    shape.materialName = "CH";
    shape.radius = q(500, "µm");
    f.geometry.regions = [];
    f.geometry.shapes2d = [shape];
    f.geometry.background2d = {
      materialName: "CH",
      rho: 1.0e-4,
      Te: q(1, "eV"),
      Ti: q(1, "eV"),
    };
    f.laser.beams.push({
      ...f.laser.beams[0],
      name: "beam_01",
      axialDirection: "plus_z",
      dirX: 0,
      dirY: 0,
      dirZ: 1,
    });
    const deck = generateDeck(f);
    expect(deck).toContain("_GUI_SHAPES = [");
    expect(deck).toContain("def gui_vf_CH(r_cm, z_cm):");
    expect(deck).toContain('mode="raytrace_3d"');
    expect(deck).toContain("direction=(0, 0, -1)");
    expect(deck).toContain("direction=(0, 0, 1)");
    expect(deck).not.toContain("hot_electron=");
    expect(deck).not.toContain("cbet=");
  });

  it("1D keeps the legacy axial default", () => {
    const f = defaultFormState();
    f.laser.enabled = true;
    const deck = generateDeck(f);
    expect(deck).toContain("direction=(0.0, 0.0, -1");
  });

  it("checkpoint interval emits checkpoint_every_s", () => {
    const f = defaultFormState();
    f.output.checkpointEveryS = q(0.5, "ns");
    expect(generateDeck(f)).toContain("checkpoint_every_s=5e-10");
    expect(generateDeck(defaultFormState())).toContain("checkpoint_every_s=-1.0");
  });
});
