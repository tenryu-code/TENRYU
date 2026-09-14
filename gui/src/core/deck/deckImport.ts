import { defaultFormState, makeHotEChannel, validateFormState, type FormState } from "./formState";
import { defaultShape2D } from "../geometry2d";
import { q, toCanonical, type Q } from "../units";
import { BEAM_PRESETS, expandedPairCount, PAIR_CAP } from "./beamPresets";
import IMPORT_RUNTIME from "../../../../tools/assist/deck_import_runtime.py?raw";
import { t } from "../../i18n";
import { computeShapeRadialRegions, computeShapeZSegments } from "./meshAuto";

export type DeckPath = Array<string | number>;
export interface ImportRule {
  path: DeckPath;
  kind: "mapped" | "approximated" | "passthrough" | "omitted";
  reason: string;
}
export interface ImportBinding { formPath: string; deckPath: DeckPath }
export interface ImportEvaluationSettings {
  venue: "local" | "server";
  workingDirectory: string;
  environment: Record<string, string>;
  filename: string;
  repoRoot?: string | null;
  profile?: { id: string; name: string; host: string };
}
export interface DeckImportState {
  source: string;
  filename: string;
  baseline: Omit<FormState, "deckImport">;
  bindings: ImportBinding[];
  rules: ImportRule[];
  notes: string[];
  unverified?: ImportRule[];
  evaluation?: ImportEvaluationSettings;
}

// The recorder is a tagged JSON protocol. Object access is deliberately kept
// inside this adapter; only checked scalar values reach the typed form.
type Recorded = Record<string, any>;
export interface DeckRecord {
  ok: boolean;
  error?: string;
  workingDirectory?: string;
  filename?: string;
  blocks: Recorded;
  regions?: Array<{ materialName: string; rOuter: number; rho: number; Te: number; Ti: number }> | null;
  shapes?: any[];
  background?: any[];
  corona?: {start:number; scaleUm:number; extentUm:number; rho0:number; rhoMin:number};
  profileReason?: string;
  meshSampling?: string;
  planner?: {radialCount:number; tailRings:number; tailRatio:number};
  rules?: ImportRule[];
  failures?: ImportRule[];
  stdout?: string;
  stderr?: string;
}

function getAt(value: any, path: Array<string | number>): any {
  return path.reduce((v, key) => v?.[key], value);
}

export function mapRecordedDeck(record: DeckRecord): { form: FormState; bindings: ImportBinding[]; notes: string[] } {
  if (!record.ok) throw new Error(record.error || "Deck recording failed");
  const f = defaultFormState();
  const bindings: ImportBinding[] = [];
  const notes: string[] = [];
  const b = record.blocks;
  function bind(formPath: string, deckPath: DeckPath) { bindings.push({ formPath, deckPath }); }
  function put(formPath: string, deckPath: DeckPath, convert: (v: any) => any = (v) => v) {
    bind(formPath, deckPath);
    const raw = getAt(b, deckPath);
    if (raw === undefined || (raw && typeof raw === "object" && raw._type)) return;
    const keys = formPath.split(".");
    const target = getAt(f, keys.slice(0, -1));
    if (!target) return;
    const old = target[keys[keys.length-1]];
    const value = convert(raw);
    if (value === undefined || (old !== undefined && old !== null && typeof old !== typeof value)) return;
    target[keys[keys.length-1]] = value;
  }
  const quantity = (unit: Q["unit"]) => (v: unknown) => typeof v === "number" ? q(v, unit) : undefined;
  const choice = (...values: string[]) => (v: unknown) => typeof v === "string" && values.includes(v) ? v : undefined;
  function fields(form: string, deck: DeckPath, mapping: Record<string, string>) {
    for (const [key, source] of Object.entries(mapping)) put(`${form}.${key}`, [...deck, source]);
  }
  fields("main", ["Main"], { name: "name", seed: "seed", maxSteps: "max_steps" });
  put("main.dimension", ["Main", "dimension"], choice("1D_SPH", "2D_RZ"));
  put("main.temperatureModel", ["Main", "temperature_model"], choice("1T", "2T"));
  put("main.tEnd", ["Main", "t_end"], quantity("s"));
  put("main.geometry1d", ["Mesh", "geometry_1d"], choice("spherical", "cylindrical", "planar"));
  if (b.Main?.dimension === "1D_CYL") f.main.geometry1d = "cylindrical";
  for (const [key, source] of Object.entries({ rMin: "r_min", rMax: "r_max", zMin: "z_min", zMax: "z_max" })) {
    put(`mesh.${key}`, ["Mesh", source], quantity("cm"));
  }
  fields("mesh", ["Mesh"], { nr: "nr", nz: "nz" });
  const mesh = b.Mesh ?? {};
  if (mesh.grid?.type === "graded" && Array.isArray(mesh.grid.segments) && f.main.dimension === "1D_SPH") {
    f.mesh.grid1d = "graded";
    f.mesh.segments = mesh.grid.segments.map((s: Recorded) => ({ rEnd: q(s.r_end, "cm"), nr: s.nr }));
    bind("mesh.segments", ["Mesh", "grid", "segments"]);
    fields("mesh.grading", ["Mesh", "grid", "grading"], { edgeRatio: "edge_ratio", sgOrder: "sg_order", sgSigma: "sg_sigma" });
  }
  if (Array.isArray(mesh.auto_regions) && f.main.dimension === "2D_RZ") {
    f.mesh.nr = mesh.auto_regions.reduce((n: number, r: Recorded) => n + r.nz, 0);
    f.mesh.radialZoning2d = "regions";
    bind("mesh.nr", ["Mesh", "auto_regions"]);
  }
  const materials = b.Materials?.materials;
  if (Array.isArray(materials) && materials.length) {
    f.materials = materials.map(() => ({ ...defaultFormState().materials[0] }));
    materials.forEach((_m: Recorded, i: number) => {
      const dest = `materials.${i}`, src: DeckPath = ["Materials", "materials", i];
      fields(dest, src, { name: "name", A: "A", Z: "Z" });
      put(`${dest}.eosModel`, [...src, "eos", "model"], choice("ideal_gas", "tmat"));
      fields(dest, [...src, "eos"], { eosFile: "file", cvEOverride: "cv_e_override" });
      put(`${dest}.gamma`, [...src, "eos", "ideal_gas", "gamma"]);
      put(`${dest}.opacityModel`, [...src, "opacity", "model"], choice("constant", "tmat"));
      fields(dest, [...src, "opacity"], { opacityFile: "file", kappaA: "kappa_a", kappaS: "kappa_s" });
    });
    bind("materials.length", ["Materials", "materials"]);
  }
  if (b.Materials?.zbar?.model === "fixed") put("zbarFixedValue", ["Materials", "zbar", "fixed_value"]);
  put("geometry.radiationField", ["Geometry", "radiation_field"], choice("equilibrium", "zero"));
  if (record.regions?.length) {
    f.geometry.regions = record.regions.map((r) => ({ materialName: r.materialName, rOuter: q(r.rOuter, "cm"), rho: r.rho, Te: q(r.Te, "eV"), Ti: q(r.Ti, "eV") }));
    if (f.main.dimension === "1D_SPH" && f.geometry.regions.at(-1)?.materialName === "VOID" && materials?.at(-1)?.is_void) {
      f.geometry.vacuumOutside1d = true;
      f.geometry.regions.pop();
      f.materials.pop();
      if (record.corona) f.geometry.coronaRamp1d = {enabled:true, scaleUm:record.corona.scaleUm, extentUm:record.corona.extentUm, rho0:record.corona.rho0, rhoMin:record.corona.rhoMin};
    }
  } else {
    f.geometry.regions = [{ ...f.geometry.regions[0], materialName: f.materials[0].name, rOuter: { ...f.mesh.rMax } }];
  }
  f.geometry.background2d.materialName = f.materials[f.materials.length-1].name;
  if (record.shapes && record.background && f.main.dimension === "2D_RZ") {
    f.geometry.shapes2d = record.shapes.map(([kind, materialName, rho, te, ti, params]) => {
      const shape = defaultShape2D(kind);
      Object.assign(shape, { materialName, rho, Te: q(te, "eV"), Ti: q(ti, "eV") });
      for (const [key, value] of Object.entries(params)) {
        if (key === "vertices") shape.vertices = (value as number[][]).map(([r, z]) => ({ r: q(r, "cm"), z: q(z, "cm") }));
        else if (key in shape) (shape as any)[key] = q(value as number, "cm");
      }
      return shape;
    });
    const [materialName, rho, te, ti] = record.background;
    f.geometry.background2d = { materialName, rho, Te: q(te, "eV"), Ti: q(ti, "eV") };
    f.geometry.regions = [];
  }
  if (mesh.logical_mesh_2d === "polar_in_box" && record.shapes?.length) {
    f.mesh.meshMode2d = "polar_in_box";
    for (const [key,source] of Object.entries({rMax:"box_r_max",zMin:"box_z_min",zMax:"box_z_max"})) put(`mesh.${key}`,["Mesh",source],quantity("cm"));
    fields("mesh",["Mesh"],{pibNTheta:"nz",pibMorphRings:"morph_rings",pibCollarRings:"collar_rings",pibMorphGrowthMax:"morph_growth_max"});
    if (record.planner) Object.assign(f.mesh,{pibNRadial:record.planner.radialCount,pibTailRings:record.planner.tailRings,pibTailRatio:record.planner.tailRatio});
    for (const field of ["pibNRadial","pibTailRings","pibTailRatio"]) bind(`mesh.${field}`,["Mesh","explicit_nodes"]);
  }
  fields("radiation", ["Radiation"], { enabled: "enabled", groups: "groups", groupBoundsEV: "group_bounds_eV" });
  put("radiation.mode", ["Radiation", "mode"], choice("multigroup_diffusion", "sn_transport"));
  const radMode = f.radiation.mode;
  put("radiation.outerR", ["Radiation", radMode, "boundary", "outer_r"], choice("vacuum", "reflect", "marshak"));
  put("radiation.zBc", ["Radiation", radMode, "boundary", "z"], choice("vacuum", "reflect"));
  put("radiation.outerR", ["Radiation", "boundary", "outer_r"], choice("vacuum", "reflect", "marshak"));
  put("radiation.marshakTrEV", ["Radiation", "boundary", "marshak_Tr_eV"]);
  put("radiation.snNAngles", ["Radiation", "sn_transport", "n_angles"]);
  function table(source: DeckPath, mode: string, points: string, factor=1) {
    const c = getAt(b, source)?.curve;
    if (c?.kind === "table" || (c?.kind === "constant" && typeof c.value === "number")) {
      const keys = points.split(".");
      const vertices = c.kind === "table" ? c.points : [[0,c.value],[toCanonical(f.main.tEnd,"time"),c.value]];
      getAt(f, keys.slice(0,-1))[keys[keys.length-1]] = vertices.map(([x,y]: number[]) => ({ t:x*1e9, v:y*factor }));
      const mk = mode.split(".");
      getAt(f, mk.slice(0,-1))[mk[mk.length-1]] = "table";
      bind(points, source); bind(mode, source);
      return true;
    }
    return false;
  }
  table(["Radiation", "boundary", "marshak_Tr"], "radiation.marshakMode", "radiation.marshakPoints");
  fields("numerics", ["Numerics", "dt"], { growthFactor: "growth_factor" });
  for (const [key, src] of Object.entries({ dtInitial: "initial_s", dtMax: "max_s", dtMin: "min_s" })) put(`numerics.${key}`, ["Numerics", "dt", src], quantity("s"));
  fields("numerics.floors", ["Numerics", "floors"], { rhoFloorGcc: "rho_floor_gcc", TeFloorEV: "Te_floor_eV", TiFloorEV: "Ti_floor_eV" });
  fields("hydro", ["Numerics", "hydro"], { enabled: "enabled", tStartEV: "T_start_eV" });
  put("hydro.boundary1d", ["Numerics", "hydro", "boundary_1d"], choice("free", "reflect", "pressure"));
  const pressure = b.Numerics?.hydro?.boundary_pressure?.curve;
  if (pressure?.kind === "constant") f.hydro.boundaryPressure.value = q(pressure.value*1e-12, "Mbar");
  table(["Numerics", "hydro", "boundary_pressure"], "hydro.boundaryPressure.mode", "hydro.boundaryPressure.points", 1e-12);
  bind("hydro.boundaryPressure", ["Numerics", "hydro", "boundary_pressure"]);
  fields("hydro.plasmaVisc", ["Numerics", "hydro", "plasma_viscosity"], { enabled: "enabled", model: "model", species: "species", etaConst: "eta_const", eta0Scale: "eta0_scale", mfpCapCells: "mfp_cap_cells", lnLambdaFixed: "lnlambda_fixed", dtSafety: "dt_safety" });
  fields("conduction", ["Numerics", "conduction"], { enabled: "enabled", fLim: "f_lim", ionConduction: "ion_conduction", nonlocalModel: "nonlocal_model", snbNGroups: "snb_n_groups", snbEMaxOverTe: "snb_E_max_over_Te", snbMfp: "snb_mfp", snbEfield: "snb_efield", snbPicardMaxIters: "snb_picard_max_iters", snbPicardRtol: "snb_picard_rtol" });
  fields("laser", ["Laser"], { enabled: "enabled", wavelengthNm: "wavelength_nm", raysPerBeam: "rays_per_beam", rayOutputTrajectory: "ray_output_trajectory", rayOutputCount: "ray_output_count" });
  put("laser.mode", ["Laser", "mode"], choice("radial_absorption_1d", "raytrace_2d"));
  const beams = b.Laser?.beams;
  if (Array.isArray(beams) && beams.length) {
    f.laser.beams = beams.map(() => ({ ...defaultFormState().laser.beams[0] }));
    beams.forEach((beam: Recorded, i: number) => {
      const dest = `laser.beams.${i}`, src: DeckPath = ["Laser", "beams", i];
      fields(dest, src, { name: "name", fNumber: "f_number" });
      if (Array.isArray(beam.direction)) {
        [f.laser.beams[i].dirX, f.laser.beams[i].dirY, f.laser.beams[i].dirZ] = beam.direction;
        f.laser.beams[i].axialDirection = beam.direction[2] > 0 ? "plus_z" : "minus_z";
      }
      if (Array.isArray(beam.focus)) f.laser.beams[i].focusZUm = beam.focus[2]*1e4;
      if (beam.profile?.model === "table") {
        f.laser.beams[i].profileModel = "table";
        f.laser.beams[i].profilePoints = beam.profile.r_um.map((x: number, j: number) => ({ t:x, v:beam.profile.I_rel[j] }));
      } else fields(dest, [...src,"profile"], { w0Um:"w0_um", superGaussianM:"m" });
    });
    const c = beams[0].power?.curve;
    if (c?.kind === "gaussian") {
      f.laser.waveformMode = "gaussian";
      f.laser.gaussianPeakW = q(c.peak, "W");
      f.laser.gaussianFwhm = q(c.fwhm, "s");
      f.laser.gaussianCenter = q(c.center, "s");
    } else if (c?.kind === "square" || c?.kind === "constant") {
      f.laser.powerW = q(c.power ?? c.value, "W");
      f.laser.pulseDuration = q(c.duration ?? toCanonical(f.main.tEnd,"time"), "s");
    } else if (c?.kind === "table") {
      table(["Laser","beams",0,"power"], "laser.waveformMode", "laser.waveformPoints", 1e-12);
      const p = c.points as number[][];
      if (p.length >= 3 && p.length <= 5 && p[0][0] === 0) {
        const peak = Math.max(...p.map(v=>v[1]));
        const first = p.findIndex(v=>v[1]===peak), last = p.length-1-[...p].reverse().findIndex(v=>v[1]===peak);
        const zeroTail = p[p.length-1][1]===0;
        if (first <= 1 && last > first && last-first <= 1 && zeroTail && p.length-last<=3 && (first===0 || p[0][1]===0)) {
          f.laser.waveformMode = "square";
          f.laser.powerW = q(peak, "W");
          f.laser.riseTime = q(p[first][0], "s");
          f.laser.pulseDuration = q(p[last][0]-p[first][0], "s");
          f.laser.fallTime = q(p[last+1][0]-p[last][0], "s");
        }
      }
    }
    if (f.main.dimension === "2D_RZ" && c) {
      const amplitude = (v: any) => v?.kind === "gaussian" ? v.peak : v?.kind === "constant" ? v.value : v?.kind === "square" ? v.power : v?.kind === "table" ? Math.max(...v.points.map((p: number[])=>p[1])) : NaN;
      const p0 = amplitude(c);
      beams.forEach((beam: Recorded, i: number) => { if (p0>0 && Number.isFinite(amplitude(beam.power?.curve))) f.laser.beams[i].powerFraction = amplitude(beam.power.curve)/p0; });
    }
    bind("laser.beams.length", ["Laser", "beams"]);
    beams.forEach((_beam: Recorded, i: number) => {
      const src: DeckPath = ["Laser","beams",i];
      for (const field of ["axialDirection","dirX","dirY","dirZ"]) bind(`laser.beams.${i}.${field}`,[...src,"direction"]);
      bind(`laser.beams.${i}.focusZUm`,[...src,"focus"]);
      for (const field of ["profileModel","profilePoints","w0Um","superGaussianM"]) bind(`laser.beams.${i}.${field}`,[...src,"profile"]);
      for (const field of ["waveformMode","powerW","pulseDuration","riseTime","fallTime","gaussianSpec","gaussianPeakW","gaussianEnergyJ","gaussianFwhm","gaussianCenter","waveformPoints"]) bind(`laser.${field}`,[...src,"power"]);
      bind(`laser.beams.${i}.powerFraction`,[...src,"power"]);
    });
  }
  const hePath = ["Laser", "hot_electron"];
  fields("laser.hotE", hePath, { enabled:"enable", sourceNcFraction:"source_nc_fraction", etaHot:"eta_hot", THotEV:"T_hot_eV", nEnergyGroups:"n_energy_groups", EMinOverTh:"E_min_over_Th", EMaxOverTh:"E_max_over_Th", angularModel:"angular_model", thetaDivDeg:"theta_div_deg", nMu:"n_mu", nPhi:"n_phi", subtractFromLaser:"subtract_from_laser", innerBc:"inner_bc", explicitSourceLimit:"explicit_source_limit" });
  table([...hePath,"eta_hot_table"], "laser.hotE.etaMode", "laser.hotE.etaPoints");
  const he = b.Laser?.hot_electron;
  if (Array.isArray(he?.sources)) {
    f.laser.hotE.useChannels = true;
    f.laser.hotE.etaEvolution = he.eta_mode === "model" ? "model" : "legacy";
    fields("laser.hotE", [...hePath,"eta_model"], { lnFilterTauS:"ln_filter_tau_s", etaTotalCap:"eta_total_cap" });
    f.laser.hotE.channels = he.sources.map((ch: Recorded) => makeHotEChannel(ch.mechanism));
    he.sources.forEach((_ch: Recorded, i: number) => {
      const dest = `laser.hotE.channels.${i}`, src: DeckPath = [...hePath,"sources",i];
      fields(dest, src, { mechanism:"mechanism", captureNcFraction:"capture_nc_fraction", THotEV:"T_hot_eV", nEnergyGroups:"n_energy_groups", EMinOverTh:"E_min_over_Th", EMaxOverTh:"E_max_over_Th", nMu:"n_mu", nPhi:"n_phi", thetaDivDeg:"theta_div_deg", tpdThetaDeg:"tpd_theta_deg", tpdDeltaDeg:"tpd_delta_deg", etaHot:"eta", evalNcFraction:"eval_nc_fraction", thresholdMultiplier:"threshold_multiplier", etaInf:"eta_inf", etaHardCap:"eta_hard_cap", shapeCoefficient:"shape_coefficient", relaxationModel:"relaxation_model", relaxationTauS:"relaxation_tau_s", relaxationTauMinS:"relaxation_tau_min_s", relaxationTauMaxS:"relaxation_tau_max_s" });
      table([...src,"eta_table"], `${dest}.etaMode`, `${dest}.etaPoints`);
    });
  }
  fields("laser.cbet", ["Laser","cbet"], { enabled:"enable", nSectionPhi:"n_section_phi", fCbet:"f_cbet", alphaIaw:"alpha_iaw", thetaCap:"theta_cap", tol:"tol", maxIters:"max_iters", nImpactBins:"n_impact_bins", neFracCutoff:"ne_frac_cutoff", kAFloor:"k_a_floor" });
  const ports = b.Laser?.port_configuration?.ports;
  if (Array.isArray(ports)) {
    for (const [key, preset] of Object.entries(BEAM_PRESETS)) {
      if (ports.length === preset.ports.length && ports.every((p: Recorded, i: number) => p.direction.every((v: number,j: number)=>Math.abs(v-preset.ports[i].dir[j])<1e-12))) f.laser.cbet.portPreset = key as typeof f.laser.cbet.portPreset;
    }
    f.laser.cbet.detuneSplitNm = Math.max(0,...ports.map((p: Recorded)=>Math.abs(p.delta_lambda_nm ?? 0)*2));
  }
  fields("burn", ["Burn"], { enabled:"enabled", scheme:"scheme", screening:"screening", xD:"x_D", xT:"x_T", xHe3:"x_He3", TFloorKeV:"T_floor_keV", neutronHeating:"neutron_heating", neutronHeatingNMu:"neutron_heating_n_mu", mcParticlesPerCell:"mc_particles_per_cell", diffusionGroups:"diffusion_groups", diffusionEMinKeV:"diffusion_E_min_keV", partition:"partition", explicitSourceLimit:"explicit_source_limit", epsDeplete:"eps_deplete", subcycleMax:"subcycle_max", vfThreshold:"vf_threshold" });
  if (Array.isArray(b.Burn?.fuels)) for (const fuel of ["DT","DD","D3He"] as const) f.burn.fuels[fuel] = b.Burn.fuels.includes(fuel);
  put("burn.fuelMaterials", ["Burn","fuel_materials"], v=>Array.isArray(v)?v.join(","):undefined);
  put("output.directory", ["Output","directory"]);
  for (const [key, src] of Object.entries({ plotEveryS:"plot_every_s", historyEveryS:"history_every_s", checkpointEveryS:"checkpoint_every_s" })) {
    put(`output.${key}`, ["Output",src], v=>typeof v==="number" ? v>0?q(v,"s"):null : undefined);
  }
  // Explicitly enabling a previously absent section adds its generated keys.
  for (const [formPath, deckPath] of Object.entries({ "radiation.enabled":["Radiation"], "laser.enabled":["Laser"], "burn.enabled":["Burn"], "laser.hotE.enabled":["Laser","hot_electron"], "laser.hotE.useChannels":["Laser","hot_electron"], "laser.hotE.channels.length":["Laser","hot_electron","sources"], "laser.cbet.enabled":["Laser","cbet"], "hydro.plasmaVisc.enabled":["Numerics","hydro","plasma_viscosity"], "mesh.grid1d":["Mesh","grid"], "mesh.segments":["Mesh","grid","segments"], "mesh.radialZoning2d":["Mesh","auto_regions"] })) bind(formPath,deckPath);
  for (const field of ["regions","shapes2d","background2d","vacuumOutside1d","coronaRamp1d"]) for (const key of ["rho","Te","Ti","volfrac"]) bind(`geometry.${field}`,["Geometry",key]);
  for (const key of ["Mesh","Geometry","Laser","Radiation"]) bind("main.dimension",[key]);
  bind("burn.fuels",["Burn","fuels"]);
  bind("laser.cbet.portPreset",["Laser","port_configuration"]);
  bind("laser.cbet.detuneSplitNm",["Laser","port_configuration"]);
  // The solver accepts some values outside the editor's supported ranges.
  // Placeholders only make the UI operable; verification marks each differing
  // source value as passthrough, so these values never enter the saved physics.
  function placeholder(formPath: string, value: unknown) {
    const keys = formPath.split(".");
    const target = getAt(f,keys.slice(0,-1));
    target[keys[keys.length-1]] = value;
    notes.push(t().deck.importPlaceholder(formPath, JSON.stringify(value)));
  }
  const d = defaultFormState();
  if (!/^[A-Za-z0-9_-]+$/.test(f.main.name)) placeholder("main.name", "imported_deck");
  if (f.main.dimension === "2D_RZ" && f.mesh.rMin.value !== 0) placeholder("mesh.rMin", q(0,"cm"));
  if (!(f.mesh.nr>=4)) placeholder("mesh.nr",4);
  if (!(f.mesh.nz>=4)) placeholder("mesh.nz",4);
  if (f.mesh.radialZoning2d === "regions" && (!record.shapes || computeShapeRadialRegions(f)===null || computeShapeZSegments(f)===null)) placeholder("mesh.radialZoning2d","uniform");
  f.materials.forEach((m,i)=>{ if (!(m.Z>0)) placeholder(`materials.${i}.Z`,1); });
  if (f.radiation.enabled) {
    if (f.radiation.groupBoundsEV.some(v=>v<=0)) placeholder("radiation.groupBoundsEV",f.radiation.groupBoundsEV.map(v=>v<=0?1e-8:v));
    if (f.radiation.groups !== f.radiation.groupBoundsEV.length-1) placeholder("radiation.groups",f.radiation.groupBoundsEV.length-1);
    if (f.radiation.marshakMode === "table" && f.radiation.marshakPoints.some(p=>p.v<=0)) placeholder("radiation.marshakMode","constant");
  }
  if (f.main.dimension !== "2D_RZ" && f.laser.beams.length>1) placeholder("laser.beams",f.laser.beams.slice(0,1));
  if (f.main.dimension === "2D_RZ") {
    if (f.laser.hotE.enabled) placeholder("laser.hotE.enabled",false);
    if (f.hydro.plasmaVisc.enabled) placeholder("hydro.plasmaVisc.enabled",false);
    if (f.laser.cbet.enabled) placeholder("laser.cbet.enabled",false);
  }
  if (f.burn.enabled && (f.main.dimension!=="1D_SPH" || f.main.geometry1d!=="spherical")) placeholder("burn.enabled",false);
  if (f.laser.cbet.enabled && expandedPairCount(BEAM_PRESETS[f.laser.cbet.portPreset].ports.length,f.laser.cbet.nImpactBins)>PAIR_CAP) placeholder("laser.cbet.nImpactBins",d.laser.cbet.nImpactBins);
  if (!(f.numerics.growthFactor>1 && f.numerics.growthFactor<=2)) placeholder("numerics.growthFactor",d.numerics.growthFactor);
  const initial = toCanonical(f.numerics.dtInitial,"time"), max = toCanonical(f.numerics.dtMax,"time");
  if (!(max>=initial)) placeholder("numerics.dtMax",q(initial,"s"));
  if (!(toCanonical(f.numerics.dtMin,"time")>0 && toCanonical(f.numerics.dtMin,"time")<toCanonical(f.numerics.dtMax,"time"))) placeholder("numerics.dtMin",q(toCanonical(f.numerics.dtMax,"time")*1e-10,"s"));
  const errors = validateFormState(f);
  if (errors.length) throw new Error("The deck was evaluated but its mapped form is not supported:\n"+errors.join("\n"));
  notes.push(t().deck.importNonRecoverable);
  if (record.meshSampling) notes.push(record.meshSampling);
  if (record.stdout?.trim()) notes.push("Deck stdout: " + record.stdout.trim());
  if (record.stderr?.trim()) notes.push("Deck stderr: " + record.stderr.trim());
  return { form:f, bindings, notes };
}

export function importActivePaths(f: FormState): DeckPath[] {
  const state = f.deckImport;
  if (!state) return [];
  return state.bindings.filter(({formPath}) => JSON.stringify(getAt(f,formPath.split("."))) !== JSON.stringify(getAt(state.baseline,formPath.split(".")))).map(b=>b.deckPath);
}

/** Retain source-only leaves and preserve repeated structured calls in order. */
export function generateImportedDeck(f: FormState, plain: string): string {
  const state = f.deckImport!;
  const active = importActivePaths(f);
  const conflicts = state.rules.filter(rule=>rule.kind==="passthrough" && active.some(p=>p.every((v,i)=>rule.path[i]===v) || rule.path.every((v,i)=>p[i]===v)));
  if (conflicts.length) throw new Error(t().deck.importConflict+"\n"+conflicts.map(r=>r.path.join(".")).join("\n"));
  const header = `# TENRYU-GUI-STATE: ${JSON.stringify(f)}`;
  if (!state.rules.some(rule=>rule.kind==="passthrough" || rule.kind==="omitted")) {
    return plain.replace(/^# TENRYU-GUI-STATE: .+$/m,()=>header);
  }
  const body = plain.replace(/^# TENRYU-GUI-STATE: .+\n/m, "");
  const prepare = `globals().update(_studio_import_prepare(${JSON.stringify(state.source)}, ${JSON.stringify(state.filename)}, _studio_import_json.loads(${JSON.stringify(JSON.stringify(state.rules))}), _studio_import_json.loads(${JSON.stringify(JSON.stringify(active))}), globals().get("__file__")))`;
  return ["# TENRYU Studio imported deck", header, "import json as _studio_import_json", IMPORT_RUNTIME,
    body.replace("from tenryu_namelist import *", prepare), "_studio_import_finish()", ""].join("\n");
}
