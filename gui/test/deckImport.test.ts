import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import * as presets from "../src/core/presets";
import { defaultFormState, makeHotEChannel, migrateFormState } from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";
import { extractGuiState } from "../src/core/deck/roundtrip";
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
