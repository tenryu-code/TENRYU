import { afterAll, expect } from "vitest";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { defaultFormState, type FormState } from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";
import { mapRecordedDeck, type DeckRecord, type ImportRule } from "../src/core/deck/deckImport";
import { UNIT_CHOICES, toCanonical, type UnitKind } from "../src/core/units";

export const root = path.resolve("..");

// Each test file owns its subprocess requests, temporary files and report.
export function createImportTestSupport(suite: string) {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "tenryu-import-test-"));
  const results: object[] = [];
  afterAll(() => {
    if (process.env.TENRYU_IMPORT_REPORT) {
      const report = path.parse(process.env.TENRYU_IMPORT_REPORT);
      fs.writeFileSync(path.join(report.dir, `${report.name}.${suite}${report.ext || ".json"}`), JSON.stringify(results, null, 2));
    }
    fs.rmSync(tmp, {recursive:true,force:true});
  });
  function evaluate(source: string, filename: string, operation="record", candidate?: string): DeckRecord & {failures?: ImportRule[]} {
    const request = path.join(tmp, "request.json");
    fs.writeFileSync(request, JSON.stringify({source,filename,operation,candidate}));
    const r = spawnSync(process.env.TENRYU_PYTHON || "python3", [path.join(root,"tools/assist/assist.py"),"import-deck","--request",request,"--repo-root",root,"--timeout","90"], {encoding:"utf8",timeout:100000,maxBuffer:64*1024*1024});
    if (r.error) throw r.error;
    try { return JSON.parse(r.stdout); }
    catch { throw new Error(r.stderr || r.stdout); }
  }
  function imported(source: string, filename: string) {
    const record = evaluate(source,filename);
    expect(record.ok, record.error).toBe(true);
    const mapped = mapRecordedDeck(record);
    const candidate = generateDeck(mapped.form);
    const verified = evaluate(source,filename,"verify",candidate);
    expect(verified.ok,verified.error).toBe(true);
    const form = mapped.form;
    form.deckImport = {source, filename, baseline:structuredClone(form), bindings:mapped.bindings, rules:verified.rules!, notes:mapped.notes};
    const compared = evaluate(source,filename,"compare",generateDeck(form));
    expect(compared.ok,JSON.stringify(compared.failures ?? compared.error)).toBe(true);
    return form;
  }
  function checkForm(name: string, form: FormState) {
    const source = generateDeck(form).replace(/^# TENRYU-GUI-STATE: .+\n/m, "");
    const filename = path.join(tmp,`${name}.py`);
    fs.writeFileSync(filename,source);
    const recovered = imported(source,filename);
    if (form.mesh.meshMode2d !== "polar_in_box" && !form.geometry.vacuumOutside1d && form.geometry.shapes2d.length===0) {
      expect(comparableForm(recovered)).toEqual(comparableForm(form));
    }
    results.push({deck:name,result:"equivalent",rules:recovered.deckImport!.rules,notes:recovered.deckImport!.notes});
    return recovered;
  }
  return { tmp, results, evaluate, imported, checkForm };
}

/** Non-recoverable controls are listed in docs/gui/DECK_IMPORT.md. */
function comparableForm(form: FormState): unknown {
  const f = structuredClone(form);
  delete f.deckImport;
  f.output.directory ||= `outputs/${f.main.name}`;
  f.customPythonBlock = "";
  // These controls do not contribute to plain radial decks.
  f.geometry.background2d = defaultFormState().geometry.background2d;
  if (f.main.dimension === "2D_RZ") {
    f.main.geometry1d = "spherical";
    f.hydro.boundary1d = "free";
    f.hydro.plasmaVisc = defaultFormState().hydro.plasmaVisc;
    f.laser.beams.forEach(beam=>{beam.axialDirection="minus_z";});
  } else {
    f.mesh.zMin = defaultFormState().mesh.zMin;
    f.mesh.zMax = defaultFormState().mesh.zMax;
    f.mesh.nz = defaultFormState().mesh.nz;
  }
  if (f.mesh.grid1d === "graded") f.mesh.nr = f.mesh.segments.reduce((n,s)=>n+s.nr,0);
  // Disabled branches carry no recoverable parameters beyond their enable bit.
  if (!f.laser.enabled) f.laser = defaultFormState().laser;
  if (!f.radiation.enabled) f.radiation = {...defaultFormState().radiation,enabled:false};
  if (f.laser.waveformMode !== "gaussian") {
    const d = defaultFormState().laser;
    Object.assign(f.laser,{gaussianSpec:d.gaussianSpec,gaussianPeakW:d.gaussianPeakW,gaussianEnergyJ:d.gaussianEnergyJ,gaussianFwhm:d.gaussianFwhm,gaussianCenter:d.gaussianCenter});
  } else {
    if (f.laser.gaussianSpec === "energy") f.laser.gaussianPeakW = {value:f.laser.gaussianEnergyJ*0.9394372786996513/toCanonical(f.laser.gaussianFwhm,"time"),unit:"W"};
    f.laser.gaussianSpec = "peak";
    f.laser.gaussianEnergyJ = 0;
  }
  if (f.laser.waveformMode !== "table") f.laser.waveformPoints = defaultFormState().laser.waveformPoints;
  if (f.radiation.marshakMode !== "table") f.radiation.marshakPoints = defaultFormState().radiation.marshakPoints;
  if (f.laser.hotE.useChannels) {
    const d = defaultFormState().laser.hotE;
    for (const key of ["sourceNcFraction","etaHot","etaMode","etaPoints","THotEV","nEnergyGroups","EMinOverTh","EMaxOverTh","thetaDivDeg","nMu","nPhi"] as const) (f.laser.hotE as any)[key] = d[key];
  }
  const walk = (value: any): any => {
    if (typeof value==="number") return Number(value.toPrecision(12));
    if (Array.isArray(value)) return value.map(walk);
    if (value && typeof value==="object") {
      if (typeof value.value==="number" && typeof value.unit==="string") {
        const kind = (Object.keys(UNIT_CHOICES) as UnitKind[]).find(k=>UNIT_CHOICES[k].includes(value.unit))!;
        return walk(toCanonical(value,kind));
      }
      return Object.fromEntries(Object.entries(value).filter(([,v])=>v!==undefined).map(([k,v])=>[k,walk(v)]));
    }
    return value;
  };
  return walk(f);
}

