import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import * as presets from "../src/core/presets";
import { defaultFormState, makeHotEChannel, migrateFormState, setInactiveCellMode } from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";
import { extractGuiState } from "../src/core/deck/roundtrip";
import { q } from "../src/core/units";
import { coldEquilibriumForm } from "./coldEquilibriumSupport";
import { createImportTestSupport, root } from "./deckImportSupport";

const { tmp, evaluate, imported, checkForm } = createImportTestSupport("forms");

describe("headerless GUI decks: binary-free equivalence", () => {
  it("default form maps scalar settings and region boundary",()=>{
    const f = defaultFormState();
    const r = checkForm("default",f);
    expect(r.main.name).toBe(f.main.name);
    expect(r.geometry.regions[0].rho).toBe(1);
    expect(r.geometry.regions[0].rOuter).toEqual({value:0.05,unit:"cm"});
    expect(r.deckImport!.rules.filter(x=>x.kind==="passthrough")).toEqual([]);
  },120000);
  it("polar-in-box planner output and geometry survive import",()=>{
    const f = presets.preset2dPolarSphere();
    f.mesh.meshMode2d = "polar_in_box";
    f.mesh.pibNRadial = 24;
    f.mesh.pibNTheta = 24;
    f.mesh.pibTailRings = 3;
    const r = checkForm("polar_in_box",f);
    expect(r.mesh.meshMode2d).toBe("polar_in_box");
    expect(r.mesh.pibNRadial).toBe(24);
    expect(r.mesh.pibTailRings).toBe(3);
    expect(r.deckImport!.rules.filter(x=>x.kind==="passthrough")).toEqual([]);
  },120000);
  for (const gaussianSpec of ["peak","energy"] as const) it(`Gaussian ${gaussianSpec} and pressure table`,()=>{
    const f = defaultFormState();
    f.laser.enabled = true;
    f.laser.waveformMode = "gaussian";
    f.laser.gaussianSpec = gaussianSpec;
    f.hydro.boundary1d = "pressure";
    f.hydro.boundaryPressure.mode = "table";
    f.hydro.boundaryPressure.points = [{t:0,v:0.1},{t:0.25,v:2},{t:2,v:0.5}];
    const r = checkForm(`gaussian_${gaussianSpec}`,f);
    expect(r.laser.waveformMode).toBe("gaussian");
    expect(r.hydro.boundaryPressure.mode).toBe("table");
  },120000);
  it("hot-electron channels with tables",()=>{
    const f = defaultFormState();
    f.laser.enabled = true;
    f.laser.hotE.enabled = true;
    f.laser.hotE.etaMode = "table";
    f.laser.hotE.etaPoints = [{t:0,v:0.01},{t:0.4,v:0.02},{t:2,v:0.005}];
    checkForm("hote_table",f);
    f.laser.hotE.useChannels = true;
    f.laser.hotE.channels = [makeHotEChannel("tpd"),makeHotEChannel("srs")];
    f.laser.hotE.channels[0].etaMode = "table";
    f.laser.hotE.channels[0].etaPoints = [{t:0,v:0.01},{t:0.3,v:0.02},{t:2,v:0.005}];
    checkForm("hote_channel_tables",f);
  },120000);
});

describe("handwritten decks and edit fidelity",()=>{
  for (const fixture of ["handwritten.py","waveforms.py","polar_in_box.py"]) it(fixture,()=>{
    const filename = path.join(root,"tests/tools/fixtures/deck_import",fixture);
    const form = imported(fs.readFileSync(filename,"utf8"),filename);
    expect(form.deckImport!.rules.some(r=>r.kind==="passthrough")).toBe(true);
    if (fixture==="handwritten.py") {
      expect(form.geometry.regions).toHaveLength(3);
      expect(form.geometry.regions[0].rOuter.value).toBeCloseTo(0.0123456789,14);
      const restored = extractGuiState(generateDeck(form));
      expect(restored.ok).toBe(true);
      if (restored.ok) {
        expect(migrateFormState(restored.state)).toEqual(form);
        const strip = (s: string) => s.replace(/^# TENRYU-GUI-STATE: .+\n/m, "");
        expect(strip(generateDeck(migrateFormState(restored.state)))).toBe(strip(generateDeck(form)));
      }
      const edited = structuredClone(form);
      edited.main.name = "edited";
      const value = evaluate(generateDeck(edited),filename);
      expect(value.ok,value.error).toBe(true);
      expect(value.blocks.Main.name).toBe("edited");
      expect(value.blocks.Main.seed).toEqual({_type:"integer",value:"9007199254740993"});
      expect(value.blocks.Main).not.toHaveProperty("temperature_model");
      edited.main.seed = 10;
      expect(()=>generateDeck(edited)).toThrow(/Main.seed/);
    }
  },120000);
  it("editing an omitted field adds only that field",()=>{
    const filename = path.join(tmp,"minimal.py");
    const form = imported('from tenryu_namelist import *\nMain(name="minimal",t_end=1e-9)\nLaser(enabled=False)\n',filename);
    form.main.seed = 77;
    const recorded = evaluate(generateDeck(form),filename);
    expect(recorded.ok,recorded.error).toBe(true);
    expect(recorded.blocks.Main).toEqual({name:"minimal",t_end:1e-9,seed:77});
    expect(recorded.blocks).not.toHaveProperty("Mesh");
  },120000);
  it("repeated nested setters retain their original order",()=>{
    const filename = path.join(tmp,"repeated.py");
    const source = 'from tenryu_namelist import *\nMain(t_end=1e-9)\nNumerics(hydro=dict(enabled=False))\nNumerics(hydro=dict(boundary_1d="reflect"))\n';
    const form = imported(source,filename);
    expect(form.deckImport!.rules).toContainEqual(expect.objectContaining({path:["Numerics"],kind:"passthrough"}));
    form.hydro.enabled = false;
    expect(()=>generateDeck(form)).toThrow(/Numerics/);
  },120000);
  it("editing a field in an absent block adds only that nested field",()=>{
    const filename = path.join(tmp,"absent_block.py");
    const form = imported('from tenryu_namelist import *\nMain(t_end=1e-9)\n',filename);
    form.numerics.dtInitial = {value:7e-14,unit:"s"};
    const recorded = evaluate(generateDeck(form),filename);
    expect(recorded.ok,recorded.error).toBe(true);
    expect(recorded.blocks).toEqual({Main:{t_end:1e-9},Numerics:{dt:{initial_s:7e-14}}});
  },120000);
  it("constant callable Marshak tables remain callable and editable",()=>{
    const f = defaultFormState();
    f.radiation.outerR = "marshak";
    f.radiation.marshakMode = "table";
    f.radiation.marshakPoints = [{t:0,v:155},{t:2,v:155}];
    const form = checkForm("constant_marshak",f);
    expect(form.radiation.marshakMode).toBe("table");
    expect(form.radiation.marshakPoints[0].v).toBe(155);
    expect(form.deckImport!.rules.filter(r=>r.kind==="passthrough")).toEqual([]);
  },120000);
});

describe("inactive-cell treatment and cold-equilibrium reference states",()=>{
  it("a cold_equilibrium form is recovered from its headerless deck",()=>{
    const r = checkForm("cold_equilibrium",coldEquilibriumForm());
    expect(r.hydro.inactiveCells).toBe("cold_equilibrium");
    expect(r.hydro.qeiHeatCapacity).toBe("table");
    expect(r.materials[1].coldReference!.K0).toEqual({value:5.8e10,unit:"dyn/cm²"});
    expect(r.deckImport!.rules.filter(x=>x.kind==="passthrough")).toEqual([]);
  },120000);
  it("a mode the form cannot represent keeps the source settings",()=>{
    const filename = path.join(tmp,"cold_sesame.py");
    const source = [
      "from tenryu_namelist import *",
      'Main(name="cold_sesame", dimension="1D_SPH", temperature_model="2T", t_end=1e-9)',
      "Mesh(r_min=0.0, r_max=0.1, nr=100)",
      'Materials(materials=[Material(name="D2", A=2.014, Z=1.0, eos=dict(model="sesame", file="sesame/5263", sesame_material_id=5263, cold_reference=dict(rho_gcc=0.17, Te0_eV=0.001, Ti0_eV=0.001, P0_dyn_cm2=0.0, bulk_modulus_dyn_cm2=1.2e9)), opacity=dict(model="constant", kappa_a=100.0, kappa_s=0.0, units="cm2_per_g"))])',
      'Geometry(rho=lambda r: 0.17, Te=lambda r: 0.001, Ti=lambda r: 0.001, volfrac={"D2": lambda r: 1.0})',
      'Numerics(hydro=dict(enabled=True, T_start_eV=2.0, T_start_inactive_cells="cold_equilibrium", qei_heat_capacity="table"))',
      "",
    ].join("\n");
    const form = imported(source,filename);
    // The SESAME EOS has no form representation, so the mode is shown as the default.
    expect(form.hydro.inactiveCells).toBe("passive_fill");
    expect(form.hydro.tStartEV).toBe(2);
    expect(form.deckImport!.rules).toContainEqual(expect.objectContaining({path:["Numerics","hydro","T_start_inactive_cells"],kind:"passthrough"}));
    const recorded = evaluate(generateDeck(form),filename);
    expect(recorded.ok,recorded.error).toBe(true);
    expect(recorded.blocks.Numerics.hydro.T_start_inactive_cells).toBe("cold_equilibrium");
    expect(recorded.blocks.Numerics.hydro.qei_heat_capacity).toBe("table");
    expect(recorded.blocks.Materials.materials[0].eos.cold_reference).toEqual({rho_gcc:0.17,Te0_eV:0.001,Ti0_eV:0.001,P0_dyn_cm2:0,bulk_modulus_dyn_cm2:1.2e9});
  },120000);
  it("switching an imported handwritten deck to cold_equilibrium adds only the mode, its parameters and the reference state",()=>{
    const filename = path.join(tmp,"switch_to_cold.py");
    const source = [
      "from tenryu_namelist import *",
      'Main(name="switch_to_cold", dimension="1D_SPH", temperature_model="2T", t_end=1e-9)',
      "Mesh(r_min=0.0, r_max=0.1, nr=100)",
      'Materials(materials=[Material(name="D2", A=2.014, Z=1.0, eos=dict(model="tmat", file="TMAT-H5/D2.tmat.h5"), opacity=dict(model="constant", kappa_a=100.0, kappa_s=0.0, units="cm2_per_g"))])',
      'Geometry(rho=lambda r: 0.17, Te=lambda r: 0.001, Ti=lambda r: 0.001, volfrac={"D2": lambda r: 1.0})',
      "Numerics(hydro=dict(enabled=True, T_start_eV=2.0))",
      "",
    ].join("\n");
    const form = imported(source,filename);
    // Keys the form emits but the source lacks are omitted, so the regenerated
    // deck replays the source and adds only the paths the edit activates.
    expect(form.deckImport!.rules.some(r=>r.kind==="omitted")).toBe(true);
    expect(form.hydro.inactiveCells).toBe("passive_fill");
    setInactiveCellMode(form,"cold_equilibrium");
    form.materials[0].coldReference!.K0 = q(0.12,"GPa");
    const recorded = evaluate(generateDeck(form),filename);
    expect(recorded.ok,recorded.error).toBe(true);
    expect(recorded.blocks.Numerics).toEqual({hydro:{
      enabled:true,
      T_start_eV:2,
      T_start_inactive_cells:"cold_equilibrium",
      cold_equilibrium:{transition_begin_fraction:0.5,density_core_ratio:1.1,density_outer_ratio:1.5,inverse_max_iterations:80},
      qei_heat_capacity:"table",
    }});
    expect(recorded.blocks.Materials.materials[0].eos).toEqual({model:"tmat",file:"TMAT-H5/D2.tmat.h5",cold_reference:{rho_gcc:0.17,Te0_eV:0.001,Ti0_eV:0.001,P0_dyn_cm2:0,bulk_modulus_dyn_cm2:1.2e9}});
  },120000);
  it("explicitly written defaults stay in the regenerated deck and can still be changed in the form",()=>{
    const filename = path.join(tmp,"explicit_default.py");
    const source = [
      "from tenryu_namelist import *",
      'Main(name="explicit_default", dimension="1D_SPH", temperature_model="2T", t_end=1e-9)',
      "Mesh(r_min=0.0, r_max=0.1, nr=100)",
      'Materials(materials=[Material(name="D2", A=2.014, Z=1.0, eos=dict(model="tmat", file="TMAT-H5/D2.tmat.h5"), opacity=dict(model="constant", kappa_a=100.0, kappa_s=0.0, units="cm2_per_g"))])',
      'Geometry(rho=lambda r: 0.17, Te=lambda r: 0.001, Ti=lambda r: 0.001, volfrac={"D2": lambda r: 1.0})',
      'Numerics(hydro=dict(enabled=True, T_start_eV=2.0, T_start_inactive_cells="passive_fill", qei_heat_capacity="ideal_gas"))',
      "",
    ].join("\n");
    const form = imported(source,filename);
    for (const key of ["T_start_inactive_cells","qei_heat_capacity"]) {
      expect(form.deckImport!.rules).toContainEqual(expect.objectContaining({path:["Numerics","hydro",key],kind:"passthrough",sourceOnly:true}));
    }
    const hydroOf = () => {
      const recorded = evaluate(generateDeck(form),filename);
      expect(recorded.ok,recorded.error).toBe(true);
      return recorded.blocks.Numerics.hydro;
    };
    expect(hydroOf()).toMatchObject({T_start_inactive_cells:"passive_fill",qei_heat_capacity:"ideal_gas"});
    setInactiveCellMode(form,"rigid_wall");
    expect(hydroOf()).toMatchObject({T_start_inactive_cells:"rigid_wall",qei_heat_capacity:"ideal_gas"});
    setInactiveCellMode(form,"cold_equilibrium");
    form.materials[0].coldReference!.K0 = q(0.12,"GPa");
    const cold = hydroOf();
    expect(cold).toMatchObject({T_start_inactive_cells:"cold_equilibrium",qei_heat_capacity:"table"});
    expect(cold.cold_equilibrium).toBeDefined();
    // Back at the imported values the source text is kept.
    setInactiveCellMode(form,"passive_fill");
    form.hydro.qeiHeatCapacity = "ideal_gas";
    expect(hydroOf()).toEqual({enabled:true,T_start_eV:2,T_start_inactive_cells:"passive_fill",qei_heat_capacity:"ideal_gas"});
  },120000);
});
