import type { Q } from "../units";
import { q } from "../units";
import { defaultShape2D, type Shape2D } from "../geometry2d";
import { t } from "../../i18n";
import { computeRegionSegments2d, computeShapeRadialRegions } from "./meshAuto";
import { BEAM_PRESETS, expandedPairCount, PAIR_CAP } from "./beamPresets";

// Flux-limiter default per electron-transport model. 0.06 is the tuned local
// (Spitzer-Harm) limiter. Under SNB the cap is a safety guard only (NUMERICS
// §4.4): a tuned value masks the nonlocal physics (design-doc A1.2 measured
// θ_min≈0.75 at f=0.06 on steep profiles), so the default loosens to 0.5 —
// transparent across the kinetically plausible band (q ≲ 0.15 q_fs) while
// still bounding cold-side range-mfp preheat at half local free streaming.
export function conductionFLimDefault(model: "none" | "snb"): number {
  return model === "snb" ? 0.5 : 0.06;
}

export const GUI_SCHEMA_VERSION = 1 as const;

export type Dimension = "1D_SPH" | "2D_RZ";
export type Geometry1d = "spherical" | "cylindrical" | "planar";
export type RadMode = "multigroup_diffusion" | "sn_transport";
export type OuterRadBc = "vacuum" | "reflect" | "marshak";
export type ZRadBc = "vacuum" | "reflect";
export type HydroBc1d = "free" | "reflect" | "pressure";

export interface MaterialForm {
  name: string;
  A: number;
  Z: number;
  eosModel: "ideal_gas" | "tmat";
  gamma: number;
  cvEOverride?: number;
  eosFile: string;
  opacityModel: "constant" | "tmat";
  kappaA: number;
  kappaS: number;
  opacityFile: string;
}

export interface RegionForm {
  materialName: string;
  rOuter: Q;
  rho: number;
  Te: Q;
  Ti: Q;
}

export interface MeshSegmentForm {
  rEnd: Q;
  nr: number;
}

/** One waveform vertex: t in ns, v in the editor's unit (TW or eV). */
export interface WaveformPoint {
  t: number;
  v: number;
}

export interface BeamForm {
  name: string;
  axialDirection: "minus_z" | "plus_z";
  dirX: number;
  dirY: number;
  dirZ: number;
  powerFraction: number;
  fNumber: number;
  focusZUm: number;
  w0Um: number;
  superGaussianM: number;
  profileModel: "super_gaussian" | "table";
  profilePoints: WaveformPoint[];
  deltaLambdaNm: number;
}

export type HotEChannelForm = {
  mechanism: "cone" | "tpd" | "srs";
  captureNcFraction: number;
  THotEV: number;
  // legacy eta prescription (etaEvolution === "legacy")
  etaMode: "constant" | "table";
  etaHot: number;
  etaPoints: WaveformPoint[];
  // angular knobs
  thetaDivDeg: number;    // cone/srs
  tpdThetaDeg: number;    // tpd
  tpdDeltaDeg: number;    // tpd
  // spectrum/quadrature
  nEnergyGroups: number;
  EMinOverTh: number;
  EMaxOverTh: number;
  nMu: number;
  nPhi: number;
  // eta_mode="model" per-channel knobs
  evalNcFraction: number;
  thresholdMultiplier: number;
  etaInf: number;
  etaHardCap: number;
  shapeCoefficient: number;
  relaxationModel: "vu2012" | "fixed";
  relaxationTauS: number;
  relaxationTauMinS: number;
  relaxationTauMaxS: number;
};

export function makeHotEChannel(mechanism: "cone" | "tpd" | "srs"): HotEChannelForm {
  const captureNcFraction = mechanism === "srs" ? 0.18 : 0.25;
  return {
    mechanism,
    captureNcFraction,
    THotEV: mechanism === "tpd" ? 6.0e4 : mechanism === "srs" ? 4.5e4 : 5.0e4,
    etaMode: "constant",
    etaHot: 0,
    etaPoints: [],
    thetaDivDeg: mechanism === "srs" ? 20 : 60,
    tpdThetaDeg: 45,
    tpdDeltaDeg: 10,
    nEnergyGroups: 30,
    EMinOverTh: 0.2,
    EMaxOverTh: 8,
    nMu: 6,
    nPhi: 8,
    evalNcFraction: captureNcFraction,
    thresholdMultiplier: mechanism === "srs" ? 8.0 : 1.0,
    etaInf: mechanism === "srs" ? 0.08 : 0.01,
    etaHardCap: mechanism === "srs" ? 0.08 : 0.03,
    shapeCoefficient: 1.0,
    relaxationModel: mechanism === "srs" ? "fixed" : "vu2012",
    relaxationTauS: 6.0e-12,
    relaxationTauMinS: 3.0e-12,
    relaxationTauMaxS: 1.0e-11,
  };
}

export interface FormState {
  guiSchemaVersion: typeof GUI_SCHEMA_VERSION;
  main: {
    name: string;
    dimension: Dimension;
    geometry1d: Geometry1d;
    tEnd: Q;
    temperatureModel: "1T" | "2T";
    seed: number;
    maxSteps: number;
  };
  mesh: {
    rMin: Q;
    rMax: Q;
    nr: number;
    zMin: Q;
    zMax: Q;
    nz: number;
    logicalMesh2d: "rectangular_rz" | "spherical_polar_halfplane";
    meshMode2d: "rect" | "polar_in_box";
    radialZoning2d: "uniform" | "regions";
    pibNTheta: number;
    pibNRadial: number;
    pibMorphRings: number;
    pibCollarRings: number;
    pibMorphGrowthMax: number;
    pibTailRings: number;
    pibTailRatio: number;
    polarCenterTreatment: "annular" | "tri_fan";
    polarKappa: number;
    grid1d: "uniform" | "graded";
    segmentSource: "manual" | "regions";
    segments: MeshSegmentForm[];
    regionNrOverrides: Array<number | null>;
    grading: { edgeRatio: number; sgOrder: number; sgSigma: number };
  };
  materials: MaterialForm[];
  zbarFixedValue: number | null;
  geometry: {
    regions: RegionForm[];
    vacuumOutside1d: boolean;
    coronaRamp1d: {
      enabled: boolean;
      scaleUm: number;
      extentUm: number;
      rho0: number;
      rhoMin: number;
    };
    shapes2d: Shape2D[];
    background2d: { materialName: string; rho: number; Te: Q; Ti: Q };
    radiationField: "equilibrium" | "zero";
  };
  radiation: {
    enabled: boolean;
    mode: RadMode;
    groups: number;
    groupBoundsEV: number[];
    outerR: OuterRadBc;
    zBc: ZRadBc;
    marshakTrEV: number;
    marshakMode: "constant" | "table";
    marshakPoints: WaveformPoint[];
    snNAngles: number;
  };
  laser: {
    enabled: boolean;
    wavelengthNm: number;
    mode: "radial_absorption_1d" | "raytrace_2d";
    raysPerBeam: number;
    rayOutputTrajectory: boolean;
    rayOutputCount: number;
    waveformMode: "square" | "gaussian" | "table";
    powerW: Q;
    pulseDuration: Q;
    riseTime: Q;
    fallTime: Q;
    gaussianSpec: "peak" | "energy";
    gaussianPeakW: Q;
    gaussianEnergyJ: number;
    gaussianFwhm: Q;
    gaussianCenter: Q;
    waveformPoints: WaveformPoint[];
    beams: BeamForm[];
    hotE: {
      enabled: boolean;
      useChannels: boolean;
      etaEvolution: "legacy" | "model";
      lnFilterTauS: number;
      etaTotalCap: number;
      channels: HotEChannelForm[];
      sourceNcFraction: number;
      etaMode: "constant" | "table";
      etaHot: number;
      etaPoints: WaveformPoint[];
      THotEV: number;
      nEnergyGroups: number;
      EMinOverTh: number;
      EMaxOverTh: number;
      angularModel: "cone" | "radial";
      thetaDivDeg: number;
      nMu: number;
      nPhi: number;
      subtractFromLaser: boolean;
      innerBc: "deposit_residual" | "escape";
      explicitSourceLimit: number;
    };
    cbet: {
      enabled: boolean;
      fCbet: number;
      alphaIaw: number;
      thetaCap: number;
      tol: number;
      maxIters: number;
      nImpactBins: number;
      nSectionPhi: number;
      portPreset: "gxii" | "omega" | "nif";
      detuneSplitNm: number;
      nPhi: number;
      neFracCutoff: number;
      kAFloor: number;
    };
  };
  conduction: {
    enabled: boolean;
    fLim: number;
    ionConduction: boolean;
    nonlocalModel: "none" | "snb";
    snbNGroups: number;
    snbEMaxOverTe: number;
    snbMfp: "geometric_r2" | "original";
    snbEfield: "none" | "local";
    snbPicardMaxIters: number;
    snbPicardRtol: number;
  };
  hydro: {
    enabled: boolean;
    boundary1d: HydroBc1d;
    boundaryPressure: {
      mode: "constant" | "table";
      value: Q;
      points: WaveformPoint[];
    };
    tStartEV: number;
    plasmaVisc: {
      enabled: boolean;
      model: "braginskii" | "constant";
      species: "ion" | "electron" | "both";
      etaConst: number;
      eta0Scale: number;
      mfpCapCells: number;
      lnLambdaFixed: number;
      dtSafety: number;
    };
  };
  burn: {
    enabled: boolean;
    fuels: { DT: boolean; DD: boolean; D3He: boolean };
    scheme: "fraley" | "diffusion" | "mc";
    screening: "none" | "salpeter" | "chugunov_dewitt";
    fuelMaterials: string;
    xD: number;
    xT: number;
    xHe3: number;
    TFloorKeV: number;
    neutronHeating: boolean;
    neutronHeatingNMu: number;
    mcParticlesPerCell: number;
    diffusionGroups: number;
    diffusionEMinKeV: number;
    partition: "li_petrasso" | "fraley";
    explicitSourceLimit: number;
    epsDeplete: number;
    subcycleMax: number;
    vfThreshold: number;
  };
  numerics: {
    dtInitial: Q;
    dtMax: Q;
    dtMin: Q;
    growthFactor: number;
    floors: { rhoFloorGcc: number; TeFloorEV: number; TiFloorEV: number };
  };
  output: {
    directory: string;
    plotEveryS: Q | null;
    historyEveryS: Q | null;
    checkpointEveryS: Q | null;
  };
  customPythonBlock: string;
}

/** Ensure a background-gas material named "gas" exists and is assigned as the 2D
 *  background. No-op when background2d.materialName already names an existing material. */
export function ensureBackgroundGas(f: FormState): void {
  if (f.materials.some((material) => material.name === f.geometry.background2d.materialName)) return;
  if (!f.materials.some((material) => material.name === "gas")) {
    f.materials.push({
      name: "gas",
      A: 1,
      Z: 1,
      eosModel: "ideal_gas",
      gamma: 5 / 3,
      eosFile: "",
      opacityModel: "constant",
      kappaA: 1.0,
      kappaS: 0.0,
      opacityFile: "",
    });
  }
  f.geometry.background2d.materialName = "gas";
}

export function defaultFormState(): FormState {
  return {
    guiSchemaVersion: GUI_SCHEMA_VERSION,
    main: {
      name: "my_run",
      dimension: "1D_SPH",
      geometry1d: "spherical",
      tEnd: q(2.0, "ns"),
      temperatureModel: "2T",
      seed: 12345,
      maxSteps: 500000,
    },
    mesh: {
      rMin: q(0, "cm"),
      rMax: q(500, "µm"),
      nr: 300,
      zMin: q(-500, "µm"),
      zMax: q(500, "µm"),
      nz: 64,
      logicalMesh2d: "rectangular_rz",
      meshMode2d: "rect",
      radialZoning2d: "uniform",
      pibNTheta: 48,
      pibNRadial: 48,
      pibMorphRings: 36,
      pibCollarRings: 5,
      pibMorphGrowthMax: 1.30,
      pibTailRings: 5,
      pibTailRatio: 1.15,
      polarCenterTreatment: "annular",
      polarKappa: 0.5,
      grid1d: "uniform",
      segmentSource: "manual",
      segments: [],
      regionNrOverrides: [],
      grading: { edgeRatio: 0.1, sgOrder: 4, sgSigma: 0.7 },
    },
    materials: [
      {
        name: "CH",
        A: 6.5,
        Z: 3.5,
        eosModel: "ideal_gas",
        gamma: 5 / 3,
        cvEOverride: undefined,
        eosFile: "",
        opacityModel: "constant",
        kappaA: 100.0,
        kappaS: 0.0,
        opacityFile: "",
      },
    ],
    zbarFixedValue: null,
    geometry: {
      regions: [
        { materialName: "CH", rOuter: q(500, "µm"), rho: 1.0, Te: q(1, "eV"), Ti: q(1, "eV") },
      ],
      vacuumOutside1d: false,
      coronaRamp1d: { enabled: false, scaleUm: 2.0, extentUm: 10.0, rho0: 0.05, rhoMin: 3.0e-4 },
      shapes2d: [],
      background2d: { materialName: "", rho: 1.0e-4, Te: q(1, "eV"), Ti: q(1, "eV") },
      radiationField: "equilibrium",
    },
    radiation: {
      enabled: true,
      mode: "multigroup_diffusion",
      groups: 1,
      groupBoundsEV: [0.1, 1.0e5],
      outerR: "vacuum",
      zBc: "vacuum",
      marshakTrEV: 120.0,
      marshakMode: "constant",
      marshakPoints: [
        { t: 0, v: 120 },
        { t: 1, v: 120 },
        { t: 2, v: 200 },
        { t: 3, v: 200 },
      ],
      snNAngles: 16,
    },
    laser: {
      enabled: false,
      wavelengthNm: 351.0,
      mode: "radial_absorption_1d",
      raysPerBeam: 1000,
      rayOutputTrajectory: false,
      rayOutputCount: 200,
      waveformMode: "square",
      powerW: q(1.0, "TW"),
      pulseDuration: q(2.0, "ns"),
      riseTime: q(0, "ns"),
      fallTime: q(0, "ns"),
      gaussianSpec: "peak",
      gaussianPeakW: q(1.0, "TW"),
      gaussianEnergyJ: 1000.0,
      gaussianFwhm: q(1.0, "ns"),
      gaussianCenter: q(2.0, "ns"),
      waveformPoints: [
        { t: 0, v: 1 },
        { t: 2, v: 1 },
      ],
      beams: [
        {
          name: "beam_00",
          axialDirection: "minus_z",
          dirX: 0,
          dirY: 0,
          dirZ: -1,
          powerFraction: 1.0,
          fNumber: 6.7,
          focusZUm: 0.0,
          w0Um: 500.0,
          superGaussianM: 4,
          profileModel: "super_gaussian",
          profilePoints: [],
          deltaLambdaNm: 0.0,
        },
      ],
      hotE: {
        enabled: false,
        useChannels: false,
        etaEvolution: "legacy",
        lnFilterTauS: 5.0e-12,
        etaTotalCap: 0.08,
        channels: [],
        sourceNcFraction: 0.25,
        etaMode: "constant",
        etaHot: 0.0,
        etaPoints: [],
        THotEV: 5.0e4,
        nEnergyGroups: 30,
        EMinOverTh: 0.2,
        EMaxOverTh: 8.0,
        angularModel: "cone",
        thetaDivDeg: 60.0,
        nMu: 6,
        nPhi: 8,
        subtractFromLaser: true,
        innerBc: "deposit_residual",
        explicitSourceLimit: 0.2,
      },
      cbet: {
        enabled: false,
        fCbet: 1.0,
        alphaIaw: 0.2,
        thetaCap: 0.3,
        tol: 1.0e-3,
        maxIters: 50,
        nImpactBins: 4,
        nSectionPhi: 8,
        portPreset: "gxii",
        detuneSplitNm: 0,
        nPhi: 8,
        neFracCutoff: 0.95,
        kAFloor: 1.0e-6,
      },
    },
    conduction: {
      enabled: true,
      fLim: 0.06,
      ionConduction: false,
      nonlocalModel: "none",
      snbNGroups: 24,
      snbEMaxOverTe: 20.0,
      snbMfp: "geometric_r2",
      snbEfield: "none",
      snbPicardMaxIters: 8,
      snbPicardRtol: 0.01,
    },
    hydro: {
      enabled: true,
      boundary1d: "free",
      boundaryPressure: {
        mode: "constant",
        value: { value: 1, unit: "Mbar" },
        points: [
          { t: 0, v: 1 },
          { t: 2, v: 1 },
        ],
      },
      tStartEV: 0.0,
      plasmaVisc: {
        enabled: false,
        model: "braginskii",
        species: "ion",
        etaConst: 0.0,
        eta0Scale: 1.0,
        mfpCapCells: 20.0,
        lnLambdaFixed: 0.0,
        dtSafety: 0.3,
      },
    },
    burn: {
      enabled: false,
      fuels: { DT: true, DD: true, D3He: false },
      scheme: "fraley",
      screening: "none",
      fuelMaterials: "",
      xD: 0.5,
      xT: 0.5,
      xHe3: 0.0,
      TFloorKeV: 0.2,
      neutronHeating: false,
      neutronHeatingNMu: 16,
      mcParticlesPerCell: 16,
      diffusionGroups: 30,
      diffusionEMinKeV: 20.0,
      partition: "li_petrasso",
      explicitSourceLimit: 0.2,
      epsDeplete: 0.1,
      subcycleMax: 64,
      vfThreshold: 1.0e-3,
    },
    numerics: {
      dtInitial: q(1e-14, "s"),
      dtMax: q(2e-12, "s"),
      dtMin: q(1e-22, "s"),
      growthFactor: 1.2,
      floors: { rhoFloorGcc: 1e-10, TeFloorEV: 1e-3, TiFloorEV: 1e-3 },
    },
    output: {
      directory: "",
      plotEveryS: q(50, "ps"),
      historyEveryS: null,
      checkpointEveryS: null,
    },
    customPythonBlock: "",
  };
}

/** Fill fields missing from older GUI-STATE JSON with current defaults (schema v1-compatible). */
export function migrateFormState(raw: FormState): FormState {
  const d = defaultFormState();
  const legacyLaser = raw.laser as typeof raw.laser & {
    fNumber?: number;
    focusZUm?: number;
    w0Um?: number;
    superGaussianM?: number;
    cbet: typeof raw.laser.cbet & { deltaLambdaNm?: number };
  };
  const merged: FormState = {
    ...d,
    ...raw,
    materials: (raw.materials ?? d.materials).map((mat) => {
      const partial = mat as Partial<MaterialForm> & Pick<MaterialForm, "name" | "A" | "Z" | "gamma">;
      return {
        eosModel: "ideal_gas" as const,
        eosFile: "",
        opacityModel: "constant" as const,
        kappaA: 0.0,
        kappaS: 0.0,
        opacityFile: "",
        ...partial,
      } as MaterialForm;
    }),
    main: { ...d.main, ...raw.main },
    mesh: {
      ...d.mesh,
      ...raw.mesh,
      grading: { ...d.mesh.grading, ...raw.mesh?.grading },
      segmentSource: raw.mesh?.segmentSource === "regions" ? ("regions" as const) : d.mesh.segmentSource,
      regionNrOverrides: Array.isArray(raw.mesh?.regionNrOverrides)
        ? raw.mesh.regionNrOverrides.map((x: unknown) => (typeof x === "number" ? x : null))
        : d.mesh.regionNrOverrides,
    },
    geometry: {
      ...d.geometry,
      ...raw.geometry,
      vacuumOutside1d: raw.geometry?.vacuumOutside1d ?? d.geometry.vacuumOutside1d,
      coronaRamp1d: { ...d.geometry.coronaRamp1d, ...raw.geometry?.coronaRamp1d },
      shapes2d: Array.isArray(raw.geometry?.shapes2d) ? raw.geometry.shapes2d : d.geometry.shapes2d,
      background2d: { ...d.geometry.background2d, ...raw.geometry?.background2d },
    },
    radiation: { ...d.radiation, ...raw.radiation },
    laser: {
      ...d.laser,
      ...raw.laser,
      beams: Array.isArray(raw.laser?.beams) && raw.laser.beams.length > 0
        ? raw.laser.beams.map((rawBeam) => {
            const beam = { ...d.laser.beams[0], ...rawBeam };
            if (typeof rawBeam?.dirX !== "number") beam.dirX = 0;
            if (typeof rawBeam?.dirY !== "number") beam.dirY = 0;
            if (typeof rawBeam?.dirZ !== "number") {
              beam.dirZ = beam.axialDirection === "plus_z" ? 1 : -1;
            }
            return beam;
          })
        : [
            {
              ...d.laser.beams[0],
              fNumber:
                legacyLaser && typeof legacyLaser.fNumber === "number"
                  ? legacyLaser.fNumber
                  : d.laser.beams[0].fNumber,
              focusZUm:
                legacyLaser && typeof legacyLaser.focusZUm === "number"
                  ? legacyLaser.focusZUm
                  : d.laser.beams[0].focusZUm,
              w0Um:
                legacyLaser && typeof legacyLaser.w0Um === "number"
                  ? legacyLaser.w0Um
                  : d.laser.beams[0].w0Um,
              superGaussianM:
                legacyLaser && typeof legacyLaser.superGaussianM === "number"
                  ? legacyLaser.superGaussianM
                  : d.laser.beams[0].superGaussianM,
              deltaLambdaNm:
                legacyLaser?.cbet && typeof legacyLaser.cbet.deltaLambdaNm === "number"
                  ? legacyLaser.cbet.deltaLambdaNm
                  : 0.0,
            },
          ],
      hotE: {
        ...d.laser.hotE,
        ...raw.laser?.hotE,
        etaPoints: Array.isArray(raw.laser?.hotE?.etaPoints)
          ? raw.laser.hotE.etaPoints
          : d.laser.hotE.etaPoints,
        channels: Array.isArray(raw.laser?.hotE?.channels)
          ? raw.laser.hotE.channels.map((ch) => ({
              ...makeHotEChannel(ch.mechanism ?? "cone"),
              ...ch,
            }))
          : d.laser.hotE.channels,
      },
      cbet: { ...d.laser.cbet, ...raw.laser?.cbet },
    },
    conduction: {
      ...d.conduction,
      ...raw.conduction,
      fLim:
        raw.conduction && typeof raw.conduction.fLim === "number"
          ? raw.conduction.fLim
          : d.conduction.fLim,
    },
    hydro: {
      ...d.hydro,
      ...raw.hydro,
      boundaryPressure: raw.hydro?.boundaryPressure ?? d.hydro.boundaryPressure,
      tStartEV:
        raw.hydro && typeof raw.hydro.tStartEV === "number" ? raw.hydro.tStartEV : d.hydro.tStartEV,
      plasmaVisc: { ...d.hydro.plasmaVisc, ...raw.hydro?.plasmaVisc },
    },
    burn: {
      ...d.burn,
      ...raw.burn,
      fuels: { ...d.burn.fuels, ...raw.burn?.fuels },
    },
    numerics: { ...d.numerics, ...raw.numerics, floors: { ...d.numerics.floors, ...raw.numerics?.floors } },
    output: { ...d.output, ...raw.output },
  };
  if (raw.mesh?.logicalMesh2d === "spherical_polar_halfplane") {
    merged.mesh.logicalMesh2d = "rectangular_rz";
    merged.mesh.zMin = { ...merged.mesh.rMax, value: -merged.mesh.rMax.value };
    merged.mesh.zMax = { ...merged.mesh.rMax };
  }
  if (
    merged.main.dimension === "2D_RZ" &&
    (!Array.isArray(raw.geometry?.shapes2d) || raw.geometry.shapes2d.length === 0) &&
    Array.isArray(raw.geometry?.regions) &&
    raw.geometry.regions.length > 0
  ) {
    let previousOuter = q(0, "µm");
    merged.geometry.shapes2d = raw.geometry.regions.map((region) => {
      const shape = defaultShape2D("shell");
      shape.z0 = q(0, "µm");
      shape.rIn = { ...previousOuter };
      shape.radius = { ...region.rOuter };
      shape.materialName = region.materialName;
      shape.rho = region.rho;
      shape.Te = { ...region.Te };
      shape.Ti = { ...region.Ti };
      previousOuter = region.rOuter;
      return shape;
    });
    const lastRegion = raw.geometry.regions[raw.geometry.regions.length - 1];
    merged.geometry.background2d = {
      materialName: lastRegion.materialName,
      rho: lastRegion.rho,
      Te: { ...lastRegion.Te },
      Ti: { ...lastRegion.Ti },
    };
  }
  return merged;
}

export type DriveMode = "laser" | "radiation_tr" | "pressure" | "none";

/** Derived drive mode over the existing fields (no separate state). */
export function driveMode(f: FormState): DriveMode {
  if (f.laser.enabled) return "laser";
  if (f.hydro.boundary1d === "pressure") return "pressure";
  if (f.radiation.enabled && f.radiation.outerR === "marshak") return "radiation_tr";
  return "none";
}

/** Set the drive mode by mutating the underlying fields (immer-style draft). */
export function setDriveMode(f: FormState, mode: DriveMode): void {
  if (mode !== "pressure" && f.hydro.boundary1d === "pressure") {
    f.hydro.boundary1d = "free";
  }
  if (mode === "laser") {
    f.laser.enabled = true;
    if (f.radiation.outerR === "marshak") f.radiation.outerR = "vacuum";
  } else if (mode === "radiation_tr") {
    f.laser.enabled = false;
    f.radiation.enabled = true;
    f.radiation.outerR = "marshak";
  } else if (mode === "pressure") {
    f.laser.enabled = false;
    if (f.radiation.outerR === "marshak") f.radiation.outerR = "vacuum";
    f.hydro.boundary1d = "pressure";
  } else {
    f.laser.enabled = false;
    if (f.radiation.outerR === "marshak") f.radiation.outerR = "vacuum";
  }
}

export function effectiveLaserRaysPerBeam(f: FormState): number {
  const is2d = f.main.dimension === "2D_RZ";
  const raysPerBeamIsEmitted =
    (!is2d && f.laser.mode === "raytrace_2d") ||
    (is2d && f.laser.raysPerBeam !== 1000);
  if (raysPerBeamIsEmitted) return f.laser.raysPerBeam;
  return is2d ? 128 : 1000;
}

const IDENT_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
const NAME_RE = /^[A-Za-z0-9_-]+$/;

/** Form validation in the current UI language. Empty result = generatable. */
export function validateFormState(f: FormState): string[] {
  const v = t().validation;
  const errs: string[] = [];
  const { toCm, toS, toEV } = conv(f);

  if (!NAME_RE.test(f.main.name)) errs.push(v.caseName);
  if (!(toS(f.main.tEnd) > 0)) errs.push(v.tEndPositive);
  if (!(Number.isInteger(f.main.maxSteps) && f.main.maxSteps >= 1)) errs.push(v.maxStepsInt);

  const rMin = toCm(f.mesh.rMin);
  const rMax = toCm(f.mesh.rMax);
  if (!(rMin >= 0)) errs.push(v.rMinNonNeg);
  if (!(rMax > rMin)) errs.push(v.rMaxGtRMin);
  if (
    !(f.main.dimension === "2D_RZ" && f.mesh.meshMode2d === "polar_in_box") &&
    !(Number.isInteger(f.mesh.nr) && f.mesh.nr >= 4 && f.mesh.nr <= 2_000_000)
  ) {
    errs.push(v.nrInt);
  }
  if (f.main.dimension === "2D_RZ") {
    if (!(toCm(f.mesh.zMax) > toCm(f.mesh.zMin))) errs.push(v.zMaxGtZMin);
    if (
      f.mesh.meshMode2d === "rect" &&
      !(Number.isInteger(f.mesh.nz) && f.mesh.nz >= 4)
    ) {
      errs.push(v.nzInt);
    }
    if (rMin !== 0) errs.push(v.rz2dRMinZero);
    if (f.mesh.grid1d === "graded") errs.push(v.gradedIs1d);
    const materialNames = new Set(f.materials.map((material) => material.name));
    for (const [i, shape] of f.geometry.shapes2d.entries()) {
      if (!materialNames.has(shape.materialName)) errs.push(v.shapeMaterialUnknown(i + 1));
      if (!(shape.rho > 0)) errs.push(v.shapeRhoPositive(i + 1));
      let paramsValid = true;
      if (shape.kind === "solidSphere") {
        paramsValid = toCm(shape.radius) > 0;
      } else if (shape.kind === "shell") {
        const rIn = toCm(shape.rIn);
        const radius = toCm(shape.radius);
        paramsValid = radius > 0 && rIn >= 0 && rIn < radius;
      } else if (shape.kind === "block") {
        const r0 = toCm(shape.r0);
        paramsValid = r0 >= 0 && toCm(shape.r1) > r0 && toCm(shape.z1) !== toCm(shape.z0);
      } else if (shape.kind === "cone") {
        paramsValid = toCm(shape.baseRadius) > 0 && toCm(shape.zBase) !== toCm(shape.zApex);
      } else if (shape.kind === "polygon") {
        paramsValid = shape.vertices.length >= 3 && shape.vertices.every((vertex) => toCm(vertex.r) >= 0);
      }
      if (!paramsValid) errs.push(v.shapeParamInvalid(i + 1));
    }
    if (!(f.geometry.background2d.rho > 0)) errs.push(v.backgroundRhoPositive);
    if (!materialNames.has(f.geometry.background2d.materialName)) {
      errs.push(v.backgroundMaterialUnknown);
    }
    if (f.mesh.meshMode2d === "polar_in_box") {
      const originFamily = f.geometry.shapes2d
        .filter(
          (shape) =>
            (shape.kind === "solidSphere" || shape.kind === "shell") &&
            Math.abs(toCm(shape.z0)) <= 1e-12,
        )
        .map((shape) => ({
          inner: shape.kind === "solidSphere" ? 0 : toCm(shape.rIn),
          outer: toCm(shape.radius),
        }))
        .sort((a, b) => a.inner - b.inner || a.outer - b.outer);
      if (originFamily.length === 0) {
        errs.push(v.pibNeedsOriginSphere);
      } else {
        const sTarget = Math.max(...originFamily.map((interval) => interval.outer));
        const tolerance = 1e-9 * Math.abs(sTarget);
        let expectedInner = 0;
        for (const interval of originFamily) {
          if (Math.abs(interval.inner - expectedInner) > tolerance) {
            errs.push(v.pibFamilyNotContiguous);
            break;
          }
          expectedInner = interval.outer;
        }
        if (!(sTarget < Math.min(rMax, Math.abs(toCm(f.mesh.zMin)), toCm(f.mesh.zMax)))) {
          errs.push(v.pibTargetExceedsBox);
        }
      }
      if (
        !(f.mesh.pibNTheta >= 8) ||
        !(f.mesh.pibNRadial >= 8) ||
        !(f.mesh.pibMorphRings >= 4) ||
        !(f.mesh.pibCollarRings >= 2 && f.mesh.pibCollarRings <= 32) ||
        !(f.mesh.pibMorphGrowthMax > 1.0 && f.mesh.pibMorphGrowthMax <= 2.0) ||
        !(f.mesh.pibTailRings >= 0) ||
        !(f.mesh.pibTailRatio >= 1.0 && f.mesh.pibTailRatio <= 2.0)
      ) {
        errs.push(v.pibParamInvalid);
      }
    }
    if (f.mesh.meshMode2d === "rect" && f.mesh.radialZoning2d === "regions") {
      if (f.geometry.shapes2d.length > 0) {
        if (computeShapeRadialRegions(f) === null) errs.push(v.meshAutoUnresolvable);
      } else if (computeRegionSegments2d(f) === null) {
        errs.push(v.meshAutoUnresolvable);
        const bounds = [toCanonical(f.mesh.rMin, "length")];
        let ascending = f.geometry.regions.length > 0;
        for (const region of f.geometry.regions) {
          const bound = toCanonical(region.rOuter, "length");
          if (!(bound > bounds[bounds.length - 1])) ascending = false;
          bounds.push(bound);
        }
        const meshRMax = toCanonical(f.mesh.rMax, "length");
        if (
          ascending &&
          Math.abs(bounds[bounds.length - 1] - meshRMax) > 1e-12 * Math.abs(meshRMax)
        ) {
          errs.push(v.regionsMustSpanRMax);
        }
      }
    }
  }
  if (f.mesh.grid1d === "graded") {
    if (f.mesh.segmentSource === "manual") {
      if (f.mesh.segments.length === 0) errs.push(v.gradedNeedsSegment);
      let prev = rMin;
      for (const [i, seg] of f.mesh.segments.entries()) {
        const e = toCm(seg.rEnd);
        if (!(e > prev)) errs.push(v.gradedSegEnd(i + 1));
        if (!(Number.isInteger(seg.nr) && seg.nr >= 1 && seg.nr <= 2_000_000)) errs.push(v.gradedSegNr(i + 1));
        prev = e;
      }
      if (f.mesh.segments.length > 0) {
        const last = toCm(f.mesh.segments[f.mesh.segments.length - 1].rEnd);
        if (Math.abs(last - rMax) > 1e-14 * Math.max(1, rMax)) {
          errs.push(v.gradedLastEnd);
        }
      }
    } else {
      let fixedSum = 0;
      let freeCount = 0;
      for (let i = 0; i < f.geometry.regions.length; i++) {
        const o = f.mesh.regionNrOverrides[i];
        if (o === null || o === undefined) {
          freeCount += 1;
        } else if (!(Number.isInteger(o) && o >= 1 && o <= 2_000_000)) {
          errs.push(v.autoOverrideInt(i + 1));
        } else {
          fixedSum += o;
        }
      }
      if (freeCount > 0 && f.mesh.nr - fixedSum < freeCount) errs.push(v.autoNrTooSmall);
    }
    const g = f.mesh.grading;
    if (!(g.edgeRatio > 0 && g.edgeRatio < 1)) errs.push(v.edgeRatioRange);
    if (!(Number.isInteger(g.sgOrder) && g.sgOrder >= 2 && g.sgOrder % 2 === 0)) errs.push(v.sgOrderEven);
    if (!(g.sgSigma > 0 && g.sgSigma < 1)) errs.push(v.sgSigmaRange);
  }

  if (f.materials.length === 0) errs.push(v.needMaterial);
  const seen = new Set<string>();
  for (const mat of f.materials) {
    if (!IDENT_RE.test(mat.name)) errs.push(v.matNameIdent(mat.name));
    if (seen.has(mat.name)) errs.push(v.matNameDup(mat.name));
    seen.add(mat.name);
    if (!(mat.A > 0)) errs.push(v.matAPositive(mat.name));
    if (!(mat.Z > 0)) errs.push(v.matZPositive(mat.name));
    if (mat.eosModel === "ideal_gas" && !(mat.gamma > 1 && mat.gamma <= 3)) errs.push(v.matGammaRange(mat.name));
    if (mat.opacityModel === "constant" && !(mat.kappaA >= 0)) errs.push(v.matKappaANonNeg(mat.name));
    if (mat.opacityModel === "constant" && !(mat.kappaS >= 0)) errs.push(v.matKappaSNonNeg(mat.name));
    if (mat.eosModel === "tmat" && mat.eosFile.trim().length === 0) {
      errs.push(t().validation.matEosFileRequired(mat.name));
    }
    if (mat.opacityModel === "tmat" && mat.opacityFile.trim().length === 0) {
      errs.push(t().validation.matOpacityFileRequired(mat.name));
    }
  }

  if (
    f.main.dimension !== "2D_RZ" ||
    (f.geometry.shapes2d.length === 0 && f.geometry.regions.length > 0)
  ) {
    if (f.geometry.regions.length === 0) errs.push(v.needRegion);
    let prevR = rMin;
    for (const [i, reg] of f.geometry.regions.entries()) {
      if (!seen.has(reg.materialName)) errs.push(v.regionUnknownMaterial(i + 1, reg.materialName));
      const ro = toCm(reg.rOuter);
      if (!(ro > prevR)) errs.push(v.regionOuterMonotone(i + 1));
      prevR = ro;
      if (!(reg.rho > 0)) errs.push(v.regionRhoPositive(i + 1));
      if (!(toEV(reg.Te) > 0)) errs.push(v.regionTePositive(i + 1));
      if (!(toEV(reg.Ti) > 0)) errs.push(v.regionTiPositive(i + 1));
    }
    if (f.geometry.regions.length > 0) {
      const lastR = toCm(f.geometry.regions[f.geometry.regions.length - 1].rOuter);
      if (f.main.dimension !== "2D_RZ" && f.geometry.vacuumOutside1d) {
        if (!(lastR < rMax)) errs.push(v.vacuumNeedsRoom);
        const corona = f.geometry.coronaRamp1d;
        if (corona.enabled) {
          if (!(lastR + corona.extentUm * 1.0e-4 <= rMax)) errs.push(v.coronaExceedsRMax);
          if (
            !(corona.scaleUm > 0) ||
            !(corona.extentUm > 0) ||
            !(corona.rho0 > 0) ||
            !(corona.rhoMin > 0) ||
            !(corona.rhoMin < corona.rho0)
          ) {
            errs.push(v.coronaParamInvalid);
          }
        }
        if (seen.has("VOID")) errs.push(v.voidNameReserved);
      } else if (Math.abs(lastR - rMax) > 1e-12 * Math.max(1, rMax)) {
        errs.push(v.lastRegionOuter);
      }
    }
    if (
      f.main.dimension !== "2D_RZ" &&
      f.geometry.coronaRamp1d.enabled &&
      !f.geometry.vacuumOutside1d
    ) {
      errs.push(v.coronaNeedsVacuum);
    }
  }

  if (f.radiation.enabled) {
    const g = f.radiation.groups;
    if (!(Number.isInteger(g) && g >= 1 && g <= 80)) errs.push(v.groupsRange);
    const b = f.radiation.groupBoundsEV;
    if (b.length !== g + 1) errs.push(v.groupBoundsCount(g + 1, b.length));
    for (let i = 0; i < b.length; i++) {
      if (!(b[i] > 0)) errs.push(v.groupBoundsPositive);
      if (i > 0 && !(b[i] > b[i - 1])) errs.push(v.groupBoundsMonotone);
    }
    if (f.radiation.mode === "sn_transport" && ![2, 4, 8, 16, 32].includes(f.radiation.snNAngles)) {
      errs.push(v.snAngles);
    }
    if (f.radiation.outerR === "marshak") {
      if (f.radiation.marshakMode === "constant" && !(f.radiation.marshakTrEV > 0)) {
        errs.push(v.marshakTrPositive);
      }
      if (f.radiation.marshakMode === "table") {
        errs.push(...validateWaveform(f.radiation.marshakPoints, v.wfLabelTr, true));
      }
    }
    if (f.main.dimension === "1D_SPH" && f.main.geometry1d === "cylindrical" && f.radiation.mode === "sn_transport") {
      errs.push(v.cyl1dSnUnsupported);
    }
  }

  if (f.laser.enabled) {
    if (f.main.dimension === "2D_RZ") {
      if (f.laser.cbet.enabled) errs.push(v.cbet2dUnsupported);
      if (f.laser.mode === "raytrace_2d") errs.push(v.raytrace2dIs1d);
    }
    if (!(f.laser.wavelengthNm > 0)) errs.push(v.wavelengthPositive);
    if (f.laser.waveformMode === "square") {
      if (!(f.laser.powerW.value > 0)) errs.push(v.laserPowerPositive);
      if (!(toS(f.laser.pulseDuration) > 0)) errs.push(v.pulsePositive);
      if (!(toS(f.laser.riseTime) >= 0)) errs.push(v.riseNonNeg);
      if (!(toS(f.laser.fallTime) >= 0)) errs.push(v.fallNonNeg);
    } else if (f.laser.waveformMode === "gaussian") {
      if (!(toS(f.laser.gaussianFwhm) > 0)) errs.push(v.gaussFwhmPositive);
      if (!(toS(f.laser.gaussianCenter) >= 0)) errs.push(v.gaussCenterNonNeg);
      if (f.laser.gaussianSpec === "peak") {
        if (!(f.laser.gaussianPeakW.value > 0)) errs.push(v.gaussPeakPositive);
      } else if (!(f.laser.gaussianEnergyJ > 0)) {
        errs.push(v.gaussEnergyPositive);
      }
    } else {
      errs.push(...validateWaveform(f.laser.waveformPoints, v.wfLabelLaser, false));
    }
    if (f.laser.beams.length === 0) errs.push(v.beamsNeedOne);
    for (const [i, b] of f.laser.beams.entries()) {
      if (f.main.dimension === "2D_RZ" && !(Math.hypot(b.dirX, b.dirY, b.dirZ) > 0)) {
        errs.push(v.beamDirZero(i + 1));
      }
      if (!(b.powerFraction > 0)) errs.push(v.beamFractionPositive(i + 1));
      if (!(Number.isFinite(b.fNumber) && b.fNumber > 0)) errs.push(v.beamFNumberPositive(i + 1));
      if (!Number.isFinite(b.focusZUm)) errs.push(v.beamFocusZFinite(i + 1));
      if (b.profileModel === "super_gaussian") {
        if (!(b.w0Um > 0)) errs.push(v.beamW0Positive(i + 1));
        if (!(Number.isInteger(b.superGaussianM) && b.superGaussianM >= 2)) errs.push(v.beamMInt(i + 1));
      } else {
        const pts = b.profilePoints;
        if (pts.length < 2) errs.push(v.beamProfileMinPoints(i + 1));
        let prev = -1;
        let imax = 0;
        for (const p of pts) {
          if (!(Number.isFinite(p.t) && p.t >= 0 && p.t > prev)) errs.push(v.beamProfileRAscending(i + 1));
          prev = p.t;
          if (!(Number.isFinite(p.v) && p.v >= 0)) errs.push(v.beamProfileINonNeg(i + 1));
          imax = Math.max(imax, p.v);
        }
        if (!(imax > 0)) errs.push(v.beamProfileINotZero(i + 1));
      }
      if (!Number.isFinite(b.deltaLambdaNm)) errs.push(v.beamDeltaLambdaFinite(i + 1));
    }
  }

  if (!(Number.isFinite(f.hydro.tStartEV) && f.hydro.tStartEV >= 0)) {
    errs.push(t().validation.hydroTStartNonNeg);
  }
  const pv = f.hydro.plasmaVisc;
  if (pv.enabled) {
    if (f.main.dimension !== "1D_SPH") errs.push(v.pvisc1dOnly);
    if (!f.hydro.enabled) errs.push(v.pviscNeedsHydro);
    if (pv.model === "constant" && !(pv.etaConst >= 0)) errs.push(v.pviscEtaConstNonNeg);
    if (!(pv.eta0Scale > 0)) errs.push(v.pviscEta0ScalePositive);
    if (!(pv.mfpCapCells > 0)) errs.push(v.pviscMfpCapPositive);
    if (!(pv.lnLambdaFixed >= 0)) errs.push(v.pviscLnLambdaNonNeg);
    if (!(pv.dtSafety > 0 && pv.dtSafety <= 1)) errs.push(v.pviscDtSafetyRange);
  }
  const bu = f.burn;
  if (bu.enabled) {
    if (!(f.main.dimension === "1D_SPH" && f.main.geometry1d === "spherical")) {
      errs.push(v.burnSphericalOnly);
    }
    if (!bu.fuels.DT && !bu.fuels.DD && !bu.fuels.D3He) errs.push(v.burnNeedsFuel);
    if (bu.partition === "fraley" && (bu.fuels.DD || bu.fuels.D3He)) {
      errs.push(v.burnFraleyDtOnly);
    }
    const names = bu.fuelMaterials.split(",").map((s) => s.trim()).filter((s) => s.length > 0);
    if (names.length === 0) errs.push(v.burnNeedsFuelMaterial);
    for (const name of names) {
      if (!seen.has(name)) errs.push(v.burnUnknownMaterial(name));
    }
    if (!(bu.xD >= 0 && bu.xT >= 0 && bu.xHe3 >= 0)) errs.push(v.burnFractionsNonNeg);
    if (Math.abs(bu.xD + bu.xT + bu.xHe3 - 1.0) > 1.0e-6) errs.push(v.burnFractionsSum);
    if (!(bu.TFloorKeV > 0)) errs.push(v.burnTFloorPositive);
    if (bu.neutronHeating && !(Number.isInteger(bu.neutronHeatingNMu) && bu.neutronHeatingNMu >= 2 && bu.neutronHeatingNMu <= 64 && bu.neutronHeatingNMu % 2 === 0)) {
      errs.push(v.burnNMuEven);
    }
    if (bu.scheme === "mc" && !(Number.isInteger(bu.mcParticlesPerCell) && bu.mcParticlesPerCell >= 1 && bu.mcParticlesPerCell <= 4096)) {
      errs.push(v.burnMcParticlesRange);
    }
    if (bu.scheme === "diffusion") {
      if (!(Number.isInteger(bu.diffusionGroups) && bu.diffusionGroups >= 4 && bu.diffusionGroups <= 512)) errs.push(v.burnDiffGroupsRange);
      if (!(bu.diffusionEMinKeV > 1 && bu.diffusionEMinKeV <= 100)) errs.push(v.burnDiffEMinRange);
    }
    if (!(bu.explicitSourceLimit > 0 && bu.explicitSourceLimit <= 1)) errs.push(v.burnSourceLimitRange);
    if (!(bu.epsDeplete > 0 && bu.epsDeplete <= 0.5)) errs.push(v.burnEpsDepleteRange);
    if (!(Number.isInteger(bu.subcycleMax) && bu.subcycleMax >= 1 && bu.subcycleMax <= 4096)) errs.push(v.burnSubcycleRange);
    if (!(bu.vfThreshold > 0 && bu.vfThreshold < 1)) errs.push(v.burnVfRange);
  }
  const he = f.laser.hotE;
  if (he.enabled) {
    if (!f.laser.enabled) errs.push(v.hoteNeedsLaser);
    if (f.main.dimension !== "1D_SPH") errs.push(v.hote1dOnly);
    if (he.angularModel === "cone" && f.main.geometry1d === "cylindrical") {
      errs.push(v.hoteConeNotCylindrical);
    }
    if (!(he.sourceNcFraction > 0 && he.sourceNcFraction <= 1)) errs.push(v.hoteNcFractionRange);
    if (he.etaMode === "constant" && !(he.etaHot >= 0 && he.etaHot <= 1)) errs.push(v.hoteEtaRange);
    if (he.etaMode === "table") {
      errs.push(...validateWaveform(he.etaPoints, v.wfLabelEta, false));
    }
    if (!(he.THotEV > 0)) errs.push(v.hoteTHotPositive);
    if (!(Number.isInteger(he.nEnergyGroups) && he.nEnergyGroups >= 2)) errs.push(v.hoteGroupsInt);
    if (!(he.EMinOverTh > 0 && he.EMaxOverTh > he.EMinOverTh)) errs.push(v.hoteERange);
    if (he.angularModel === "cone" && !(he.thetaDivDeg > 0 && he.thetaDivDeg <= 90)) {
      errs.push(v.hoteThetaRange);
    }
    if (!(Number.isInteger(he.nMu) && he.nMu >= 2)) errs.push(v.hoteNMuInt);
    if (!(Number.isInteger(he.nPhi) && he.nPhi >= 1)) errs.push(v.hoteNPhiInt);
    if (!(he.explicitSourceLimit > 0 && he.explicitSourceLimit <= 1)) errs.push(v.hoteSourceLimitRange);
    if (f.main.dimension !== "2D_RZ") {
      if (he.useChannels && he.channels.length === 0) errs.push(v.hoteChannelsEmpty);
      if (he.channels.length > 4) errs.push(v.hoteChannelsMax);
      if (he.etaEvolution === "model" && !he.useChannels) errs.push(v.hoteModelNeedsChannels);
      if (he.etaEvolution === "model" && he.channels.some((ch) => ch.mechanism === "cone")) {
        errs.push(v.hoteModelNoCone);
      }
      if (he.etaEvolution === "model" && !he.subtractFromLaser) {
        errs.push(v.hoteModelNeedsSubtract);
      }
      if (he.etaEvolution === "model" && !(he.etaTotalCap > 0 && he.etaTotalCap <= 0.5)) {
        errs.push(v.hoteEtaTotalCapRange);
      }
      if (he.etaEvolution === "model" && !(he.lnFilterTauS > 0)) {
        errs.push(v.hoteLnFilterTauRange);
      }
      for (const ch of he.channels) {
        if (!(ch.captureNcFraction > 0 && ch.captureNcFraction <= 1)) {
          errs.push(v.hoteCaptureNcRange);
        }
        if (
          he.etaEvolution === "legacy" &&
          ch.etaMode === "constant" &&
          !(ch.etaHot >= 0 && ch.etaHot <= 0.95)
        ) {
          errs.push(v.hoteEtaRange);
        }
        if (
          he.etaEvolution === "model" &&
          !(ch.evalNcFraction > 0 && ch.evalNcFraction < 1)
        ) {
          errs.push(v.hoteEvalNcRange);
        }
      }
    }
  }
  const cn = f.conduction;
  if (cn.nonlocalModel === "snb") {
    if (!cn.enabled) errs.push(v.snbNeedsConduction);
    if (f.main.dimension !== "1D_SPH") errs.push(v.snb1dOnly);
    if (f.main.temperatureModel !== "2T") errs.push(v.snbNeeds2T);
    if (!(Number.isInteger(cn.snbNGroups) && cn.snbNGroups >= 2)) errs.push(v.snbGroupsInt);
    if (!(cn.snbEMaxOverTe > 1)) errs.push(v.snbEMaxRange);
    if (!(Number.isInteger(cn.snbPicardMaxIters) && cn.snbPicardMaxIters >= 2)) {
      errs.push(v.snbPicardItersInt);
    }
    if (!(cn.snbPicardRtol > 0)) errs.push(v.snbPicardRtolPositive);
  }
  if (
    f.laser.rayOutputTrajectory &&
    !(Number.isInteger(f.laser.rayOutputCount) && f.laser.rayOutputCount >= 1 && f.laser.rayOutputCount <= 100_000)
  ) {
    errs.push(v.rayOutputCountRange);
  }
  if (f.laser.enabled && f.laser.mode === "raytrace_2d") {
    if (f.main.geometry1d !== "spherical") errs.push(v.raytraceSphericalOnly);
    if (!(Number.isInteger(f.laser.raysPerBeam) && f.laser.raysPerBeam >= 10)) {
      errs.push(v.raysPerBeamRange);
    }
  }
  const cb = f.laser.cbet;
  if (f.main.dimension !== "2D_RZ" && f.laser.beams.length > 1) {
    errs.push(v.cbetSingleRefBeam);
  }
  if (cb.enabled) {
    if (!f.laser.enabled) errs.push(v.cbetNeedsLaser);
    if (f.laser.mode !== "raytrace_2d") errs.push(v.cbetNeedsRaytrace);
    if (f.laser.beams[0] && f.laser.beams[0].deltaLambdaNm !== 0) {
      errs.push(v.cbetBeamDetuneConflict);
    }
    if (f.laser.hotE.enabled && f.laser.hotE.etaEvolution !== "model") {
      errs.push(v.cbetHoteExclusive);
    }
    if (!(cb.fCbet > 0)) errs.push(v.cbetFCbetPositive);
    if (!(cb.alphaIaw > 0)) errs.push(v.cbetAlphaPositive);
    if (!(cb.thetaCap > 0)) errs.push(v.cbetThetaCapPositive);
    if (!(cb.tol > 0)) errs.push(v.cbetTolPositive);
    if (!(Number.isInteger(cb.maxIters) && cb.maxIters >= 1)) errs.push(v.cbetMaxItersInt);
    if (!(Number.isInteger(cb.nImpactBins) && cb.nImpactBins >= 1)) errs.push(v.cbetBinsInt);
    if (!(Number.isInteger(cb.nSectionPhi) && cb.nSectionPhi >= 1)) {
      errs.push(v.cbetNSectionPhiInt);
    }
    if (
      expandedPairCount(BEAM_PRESETS[cb.portPreset].ports.length, cb.nImpactBins) > PAIR_CAP
    ) {
      errs.push(v.cbetPairCap);
    }
    if (!(cb.detuneSplitNm >= 0)) errs.push(v.cbetDetuneNonneg);
    if (!(Number.isInteger(cb.nPhi) && cb.nPhi >= 1)) errs.push(v.cbetNPhiInt);
    if (!(cb.neFracCutoff > 0 && cb.neFracCutoff <= 1)) errs.push(v.cbetNeCutoffRange);
    if (!(cb.kAFloor > 0)) errs.push(v.cbetKaFloorPositive);
  }
  if (!(Number.isFinite(f.conduction.fLim) && f.conduction.fLim > 0)) {
    errs.push(t().validation.fLimPositive);
  }
  const dtI = toS(f.numerics.dtInitial);
  const dtX = toS(f.numerics.dtMax);
  const dtN = toS(f.numerics.dtMin);
  if (!(dtI > 0)) errs.push(v.dtInitialPositive);
  if (!(dtX >= dtI)) errs.push(v.dtMaxGeInitial);
  if (!(dtN > 0 && dtN < dtX)) errs.push(v.dtMinRange);
  if (!(f.numerics.growthFactor > 1 && f.numerics.growthFactor <= 2)) errs.push(v.growthFactorRange);
  if (!(f.numerics.floors.rhoFloorGcc > 0)) errs.push(v.rhoFloorPositive);
  if (!(f.numerics.floors.TeFloorEV > 0)) errs.push(v.teFloorPositive);
  if (!(f.numerics.floors.TiFloorEV > 0)) errs.push(v.tiFloorPositive);

  if (f.output.plotEveryS && !(toS(f.output.plotEveryS) > 0)) errs.push(v.plotEveryPositive);
  if (f.output.historyEveryS && !(toS(f.output.historyEveryS) > 0)) errs.push(v.historyEveryPositive);
  if (f.output.checkpointEveryS && !(toS(f.output.checkpointEveryS) > 0)) {
    errs.push(v.checkpointEveryPositive);
  }

  return errs;
}

function validateWaveform(points: WaveformPoint[], label: string, strictlyPositiveV: boolean): string[] {
  const v = t().validation;
  const errs: string[] = [];
  if (points.length < 2) {
    errs.push(v.wfMinPoints(label));
    return errs;
  }
  for (const [i, p] of points.entries()) {
    if (!Number.isFinite(p.t) || !Number.isFinite(p.v)) errs.push(v.wfBadNumber(label, i + 1));
    if (i === 0 && !(p.t >= 0)) errs.push(v.wfTStartNonNeg(label));
    if (i > 0 && !(p.t > points[i - 1].t)) errs.push(v.wfTMonotone(label));
    if (strictlyPositiveV ? !(p.v > 0) : !(p.v >= 0)) {
      errs.push(strictlyPositiveV ? v.wfVPositive(label, i + 1) : v.wfVNonNeg(label, i + 1));
    }
  }
  return errs;
}

function conv(_f: FormState) {
  return {
    toCm: (qq: Q) => require_(qq, "length"),
    toS: (qq: Q) => require_(qq, "time"),
    toEV: (qq: Q) => require_(qq, "temperature"),
  };
}

import { toCanonical, type UnitKind } from "../units";
function require_(qq: Q, kind: UnitKind): number {
  return toCanonical(qq, kind);
}
