// 1D mesh methods of TENRYU Studio beyond the uniform and graded forms: the mesh recommended by
// tools/assist (recommend-mesh, a zoning_intent block with the resolution requirement), a per-layer
// table whose nodes Studio computes, imported node coordinates, and a directly edited zoning_intent.
// Studio writes the per-layer and imported meshes as Mesh.explicit_nodes (SPECIFICATION 6.4.2), so
// the mesh it previews is the mesh the solver uses.
import type { FormState } from "./formState";
import { toCanonical } from "../units";
import { computeRegionSegments } from "./meshAuto";
import { computeZoningIntentNodes, type ZoningIntentConfig } from "./zoningIntent";
import { t } from "../../i18n";

export type MeshMethod1d = "uniform" | "graded" | "recommended" | "layers" | "explicit" | "zoning_intent";

/** Methods whose nodes the solver computes from the deck's zoning_intent (Studio draws them with
 *  its copy of the solver's zoning; validate --mesh-preview shows the solver's own). */
export function solverZonedMethod(method: MeshMethod1d): boolean {
  return method === "recommended" || method === "zoning_intent";
}

// ---------------------------------------------------------------------------------------------
// Form types

export type LayerSpacing = "width" | "mass" | "ratio";
/** How a geometric (equal-ratio) spacing is fixed: the ratio itself, the width of the smallest
 *  cell, or the mass of the neighbouring layer's adjacent cell. */
export type LayerRatioSpec = "ratio" | "first_width" | "match_mass";
/** Side of the layer that holds the smallest cells. */
export type LayerFineSide = "outer" | "inner" | "both";

export interface LayerZoningForm {
  cells: number;
  spacing: LayerSpacing;
  ratioSpec: LayerRatioSpec;
  /** Width ratio of adjacent cells, growing away from the fine side (1 = equal widths). */
  ratio: number;
  /** Width of the cell at the fine side [cm] (ratioSpec "first_width"). */
  firstWidthCm: number;
  fineSide: LayerFineSide;
}

export function defaultLayerZoning(): LayerZoningForm {
  return { cells: 40, spacing: "mass", ratioSpec: "ratio", ratio: 1.1, firstWidthCm: 1.0e-5, fineSide: "outer" };
}

export interface ExplicitNodesForm {
  /** Node radii [cm], strictly increasing, from r_min to r_max. */
  nodesCm: number[];
  /** Where the nodes came from (shown in the mesh section). */
  source: string;
}

export type ZoningMeasure = "width" | "areal_mass" | "cylindrical_line_mass" | "spherical_cell_mass";

export interface ZoningIntentForm {
  nCells: number;
  measure: ZoningMeasure;
  /** Piecewise-constant zoning density; empty = taken from the initial regions. */
  densityRegions: Array<{ rEndCm: number; rho: number }>;
  /** Add a pin (ratio jump allowed) at every material interface. */
  pinInterfaces: boolean;
  pins: Array<{ rCm: number; ratioJumpAllowed: boolean }>;
  profile: Array<{ rCm: number; w: number }>;
  anchors: Array<{ rCm: number; halfWidthCm: number; logAmplitude: number }>;
  bands: Array<{ fracBegin: number; fracEnd: number; cellMeasureMin: number | null; cellMeasureMax: number | null }>;
  extraEventsCm: number[];
  drMinCm: number | null;
  cellMeasureMin: number | null;
  cellMeasureMax: number | null;
  preferredRatio: number | null;
  ratioHardMax: number | null;
  minCellsPerSegment: number | null;
}

export function defaultZoningIntent(): ZoningIntentForm {
  return {
    nCells: 400,
    measure: "width",
    densityRegions: [],
    pinInterfaces: true,
    pins: [],
    profile: [],
    anchors: [],
    bands: [],
    extraEventsCm: [],
    drMinCm: null,
    cellMeasureMin: null,
    cellMeasureMax: null,
    preferredRatio: null,
    ratioHardMax: null,
    minCellsPerSegment: null,
  };
}

export interface EmpiricalForm {
  referenceSha256: string;
  caseIds: string[];
  surfaceCeilingGcm2: number;
  referenceAprioriGcm2: number;
}

/** Mesh.resolution_requirement: "default" writes no block (the solver reports only). */
export interface ResolutionRequirementForm {
  apply: "default" | "report" | "enforce";
  empirical: EmpiricalForm | null;
}

export function defaultResolutionRequirement(): ResolutionRequirementForm {
  return { apply: "default", empirical: null };
}

/** What recommend-mesh returned, kept with the zoning_intent it produced. */
export interface MeshRecommendationMeta {
  /** recommendationConditionsKey(form) at the time of the recommendation. */
  conditionsKey: string;
  createdAt: string;
  /** validation.status of recommend-mesh: validated | unvalidated | failed. */
  status: string;
  flags: string[];
  warnings: string[];
  caseIds: string[];
  confidence: string;
  mode: string;
  surfaceCeilingGcm2: number | null;
  achievedSurfaceGcm2: number | null;
  nCells: number;
  /** Server binary the recommendation was validated with ("" when none). */
  binary: string;
  /** The zoning_intent was edited after the recommendation. */
  edited: boolean;
}

// ---------------------------------------------------------------------------------------------
// Layers of the 1D target as the deck writes them

/** Density of the void material that Studio writes (Materials.void_config rho). */
export const GUI_VOID_RHO = 1.0e-10;

/** The value the deck writes for a length (generate.ts pyNum: 15 significant digits). */
export function deckRound(x: number): number {
  return Number(x.toPrecision(15));
}

export interface MeshLayer1d {
  kind: "region" | "corona" | "void";
  /** Material name ("VOID" for the void padding). */
  materialName: string;
  regionIndex: number | null;
  rLoCm: number;
  rHiCm: number;
  /** Constant density of the layer, or null when it varies (the corona ramp). */
  rhoConst: number | null;
  rhoAt: (rCm: number) => number;
  /** Radii inside the layer where the density has a kink (quadrature split points). */
  kinks: number[];
}

export type Geometry1d = "spherical" | "cylindrical" | "planar";

/** The initial regions, the corona ramp and the void padding of a 1D form, inner to outer, with
 *  the radii the deck writes; null when they do not form a valid sequence. */
export function meshLayers1d(f: FormState): MeshLayer1d[] | null {
  if (f.main.dimension !== "1D_SPH") return null;
  const regions = f.geometry.regions;
  if (regions.length === 0) return null;
  const rMin = deckRound(toCanonical(f.mesh.rMin, "length"));
  const rMax = deckRound(toCanonical(f.mesh.rMax, "length"));
  if (!(Number.isFinite(rMin) && Number.isFinite(rMax) && rMax > rMin)) return null;
  const out: MeshLayer1d[] = [];
  let lo = rMin;
  for (const [i, region] of regions.entries()) {
    const hi = deckRound(toCanonical(region.rOuter, "length"));
    const rho = region.rho;
    if (!(hi > lo) || !(rho > 0) || !Number.isFinite(rho)) return null;
    out.push({
      kind: "region",
      materialName: region.materialName,
      regionIndex: i,
      rLoCm: lo,
      rHiCm: hi,
      rhoConst: rho,
      rhoAt: () => rho,
      kinks: [],
    });
    lo = hi;
  }
  if (f.geometry.vacuumOutside1d) {
    if (!(lo < rMax)) return null;
    const corona = f.geometry.coronaRamp1d;
    if (corona.enabled) {
      const end = deckRound(lo + corona.extentUm * 1.0e-4);
      const scale = corona.scaleUm * 1.0e-4;
      if (!(end > lo) || end > rMax || !(scale > 0) || !(corona.rho0 > corona.rhoMin) || !(corona.rhoMin > 0)) {
        return null;
      }
      const surface = lo;
      const kink = surface + scale * Math.log(corona.rho0 / corona.rhoMin);
      out.push({
        kind: "corona",
        materialName: regions[regions.length - 1].materialName,
        regionIndex: null,
        rLoCm: lo,
        rHiCm: end,
        rhoConst: null,
        rhoAt: (r) => Math.max(corona.rho0 * Math.exp(-(r - surface) / scale), corona.rhoMin),
        kinks: kink > lo && kink < end ? [kink] : [],
      });
      lo = end;
    }
    if (lo < rMax) {
      out.push({
        kind: "void",
        materialName: "VOID",
        regionIndex: null,
        rLoCm: lo,
        rHiCm: rMax,
        rhoConst: GUI_VOID_RHO,
        rhoAt: () => GUI_VOID_RHO,
        kinks: [],
      });
    }
  } else if (Math.abs(lo - rMax) > 1e-12 * Math.max(1, Math.abs(rMax))) {
    return null;
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Measures

/** Volume of the shell [a, b] per unit area (planar), per unit length (cylindrical) or total. */
export function shellVolume(geometry: Geometry1d, a: number, b: number): number {
  if (geometry === "spherical") return ((4 * Math.PI) / 3) * (b * b * b - a * a * a);
  if (geometry === "cylindrical") return Math.PI * (b * b - a * a);
  return b - a;
}

/** Area through which the laser-side surface at radius R is driven (SPECIFICATION 6.4.2
 *  resolution_requirement; recommend-mesh reference_area): planar 1, cylindrical 2πR, spherical 4πR². */
export function referenceArea(geometry: Geometry1d, radius: number): number {
  if (geometry === "spherical") return 4 * Math.PI * radius * radius;
  if (geometry === "cylindrical") return 2 * Math.PI * radius;
  return 1;
}

function geometryDensityWeight(geometry: Geometry1d, r: number): number {
  if (geometry === "spherical") return 4 * Math.PI * r * r;
  if (geometry === "cylindrical") return 2 * Math.PI * r;
  return 1;
}

/** Mass of [a, b] inside one layer (Simpson's rule between density kinks when it varies). */
export function layerMass(layer: MeshLayer1d, geometry: Geometry1d, a: number, b: number): number {
  if (!(b > a)) return 0;
  if (layer.rhoConst !== null) return layer.rhoConst * shellVolume(geometry, a, b);
  const cuts = [a, ...layer.kinks.filter((k) => k > a && k < b), b];
  let total = 0;
  for (let s = 0; s + 1 < cuts.length; s++) {
    const lo = cuts[s];
    const hi = cuts[s + 1];
    const n = 64;
    const h = (hi - lo) / n;
    let sum = 0;
    for (let i = 0; i <= n; i++) {
      const r = lo + i * h;
      const w = i === 0 || i === n ? 1 : i % 2 === 1 ? 4 : 2;
      sum += w * layer.rhoAt(r) * geometryDensityWeight(geometry, r);
    }
    total += (sum * h) / 3;
  }
  return total;
}

// ---------------------------------------------------------------------------------------------
// Per-layer node construction

export interface LayerNodesResult {
  nodes: number[];
  error: string | null;
}

/** Exponents of the geometric widths w_k = w0 q^e_k, cells ordered inner to outer. */
function ratioExponents(n: number, fineSide: LayerFineSide): number[] {
  const out: number[] = [];
  for (let k = 0; k < n; k++) {
    if (fineSide === "inner") out.push(k);
    else if (fineSide === "outer") out.push(n - 1 - k);
    else out.push(Math.min(k, n - 1 - k));
  }
  return out;
}

function exponentSum(exponents: number[], q: number): number {
  let s = 0;
  for (const e of exponents) s += Math.pow(q, e);
  return s;
}

/** Ratio q > 0 with sum_k q^e_k = target (the sum increases with q); null when impossible. */
export function solveRatio(exponents: number[], target: number): number | null {
  const n = exponents.length;
  if (!(Number.isFinite(target) && target > 0) || n === 0) return null;
  if (Math.abs(target - n) <= 1e-12 * n) return 1;
  // With every exponent zero the sum is n for any q; otherwise it runs from the number of zero
  // exponents (q -> 0) to infinity.
  if (exponents.every((e) => e === 0)) return null;
  const minimum = exponents.filter((e) => e === 0).length;
  if (!(target > minimum)) return null;
  let lo = Math.log(1e-6);
  let hi = Math.log(1e6);
  if (exponentSum(exponents, Math.exp(hi)) < target || exponentSum(exponents, Math.exp(lo)) > target) return null;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (exponentSum(exponents, Math.exp(mid)) < target) lo = mid;
    else hi = mid;
  }
  return Math.exp(0.5 * (lo + hi));
}

function nodesFromWidths(a: number, b: number, widths: number[]): number[] {
  const total = widths.reduce((s, w) => s + w, 0);
  const scale = (b - a) / total;
  const nodes = [a];
  let acc = 0;
  for (let k = 0; k + 1 < widths.length; k++) {
    acc += widths[k] * scale;
    nodes.push(a + acc);
  }
  nodes.push(b);
  return nodes;
}

function equalMassNodes(layer: MeshLayer1d, geometry: Geometry1d, n: number): number[] {
  const a = layer.rLoCm;
  const b = layer.rHiCm;
  if (layer.rhoConst !== null) {
    const p = geometry === "spherical" ? 3 : geometry === "cylindrical" ? 2 : 1;
    const ap = Math.pow(a, p);
    const bp = Math.pow(b, p);
    const nodes = [a];
    for (let j = 1; j < n; j++) nodes.push(Math.pow(ap + (j / n) * (bp - ap), 1 / p));
    nodes.push(b);
    return nodes;
  }
  // Varying density: cumulative mass on a fine grid (kinks included), inverted piecewise linearly.
  const k = Math.max(8192, 16 * n);
  const grid: number[] = [];
  for (let i = 0; i <= k; i++) grid.push(a + ((b - a) * i) / k);
  for (const kink of layer.kinks) grid.push(kink);
  grid.sort((x, y) => x - y);
  const cumulative = [0];
  for (let i = 1; i < grid.length; i++) {
    const lo = grid[i - 1];
    const hi = grid[i];
    const mid = 0.5 * (lo + hi);
    const piece = layer.rhoAt(mid) * shellVolume(geometry, lo, hi);
    cumulative.push(cumulative[i - 1] + piece);
  }
  const total = cumulative[cumulative.length - 1];
  const nodes = [a];
  let i = 1;
  for (let j = 1; j < n; j++) {
    const target = (j / n) * total;
    while (i < cumulative.length - 1 && cumulative[i] < target) i++;
    const m0 = cumulative[i - 1];
    const m1 = cumulative[i];
    const t = m1 > m0 ? (target - m0) / (m1 - m0) : 0;
    nodes.push(grid[i - 1] + t * (grid[i] - grid[i - 1]));
  }
  nodes.push(b);
  return nodes;
}

/** Width w0 of the cell at the given side whose mass equals `mass` (bisection; monotone). */
function widthForMass(layer: MeshLayer1d, geometry: Geometry1d, side: "inner" | "outer", mass: number): number | null {
  const a = layer.rLoCm;
  const b = layer.rHiCm;
  const thickness = b - a;
  const massOfWidth = (w: number) =>
    side === "inner" ? layerMass(layer, geometry, a, a + w) : layerMass(layer, geometry, b - w, b);
  if (!(mass > 0) || !(massOfWidth(thickness) > mass)) return null;
  let lo = 0;
  let hi = thickness;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (massOfWidth(mid) < mass) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/** Mass of the cell next to the given side of a layer whose nodes are known. */
function edgeCellMass(layer: MeshLayer1d, geometry: Geometry1d, nodes: number[], side: "inner" | "outer"): number {
  return side === "inner"
    ? layerMass(layer, geometry, nodes[0], nodes[1])
    : layerMass(layer, geometry, nodes[nodes.length - 2], nodes[nodes.length - 1]);
}

/** Specs of all derived layers: the stored table, padded with the default for new layers. */
export function resolvedLayerZoning(f: FormState, layers: MeshLayer1d[]): LayerZoningForm[] {
  return layers.map((_, i) => ({ ...defaultLayerZoning(), ...(f.mesh.layerZoning[i] ?? {}) }));
}

/** Index of the neighbouring layer whose adjacent cell a "match_mass" layer copies, or null. */
export function matchNeighbour(spec: LayerZoningForm, index: number, count: number): number | null {
  if (spec.spacing !== "ratio" || spec.ratioSpec !== "match_mass") return null;
  if (spec.fineSide === "outer") return index + 1 < count ? index + 1 : null;
  if (spec.fineSide === "inner") return index > 0 ? index - 1 : null;
  return null;
}

export interface LayersNodesResult {
  edges: number[] | null;
  /** Per-layer node lists (null where the layer failed). */
  perLayer: Array<number[] | null>;
  /** Per-layer error messages (keys of t().mesh1d.layerErrors). */
  errors: Array<{ layer: number; code: LayerErrorCode }>;
}

export type LayerErrorCode =
  | "cells"
  | "ratio"
  | "firstWidth"
  | "matchBothSides"
  | "matchNoNeighbour"
  | "matchCycle"
  | "matchTooHeavy"
  | "notIncreasing";

/** Nodes of the per-layer table (Studio's "layers" mesh); interfaces are nodes by construction. */
export function computeLayerNodes(f: FormState, layers?: MeshLayer1d[] | null): LayersNodesResult {
  const derived = layers === undefined ? meshLayers1d(f) : layers;
  if (derived === null || derived.length === 0) return { edges: null, perLayer: [], errors: [] };
  const geometry = f.main.geometry1d;
  const specs = resolvedLayerZoning(f, derived);
  const perLayer: Array<number[] | null> = derived.map(() => null);
  const errors: LayersNodesResult["errors"] = [];
  const failed = new Set<number>();
  const fail = (layer: number, code: LayerErrorCode) => {
    if (!failed.has(layer)) errors.push({ layer, code });
    failed.add(layer);
  };
  for (const [i, spec] of specs.entries()) {
    if (!(Number.isInteger(spec.cells) && spec.cells >= 1 && spec.cells <= 2_000_000)) fail(i, "cells");
    if (spec.spacing === "ratio") {
      if (spec.ratioSpec === "ratio" && !(Number.isFinite(spec.ratio) && spec.ratio > 0)) fail(i, "ratio");
      // A progression whose end cell is prescribed needs a second cell to take up the rest.
      if (spec.ratioSpec !== "ratio" && spec.cells < 2) fail(i, "cells");
      if (spec.ratioSpec === "first_width") {
        const thickness = derived[i].rHiCm - derived[i].rLoCm;
        if (!(Number.isFinite(spec.firstWidthCm) && spec.firstWidthCm > 0 && spec.firstWidthCm < thickness)) {
          fail(i, "firstWidth");
        }
      }
      if (spec.ratioSpec === "match_mass") {
        if (spec.fineSide === "both") fail(i, "matchBothSides");
        else if (matchNeighbour(spec, i, derived.length) === null) fail(i, "matchNoNeighbour");
      }
    }
  }
  // Layers that copy a neighbour's cell mass are built after that neighbour; a dependency cycle
  // (two neighbours copying each other) is an error.
  const pending = new Set(specs.map((_, i) => i).filter((i) => !failed.has(i)));
  let progress = true;
  while (pending.size > 0 && progress) {
    progress = false;
    for (const i of [...pending]) {
      const spec = specs[i];
      const layer = derived[i];
      const neighbour = matchNeighbour(spec, i, derived.length);
      if (neighbour !== null && perLayer[neighbour] === null) {
        if (failed.has(neighbour)) {
          pending.delete(i);
          fail(i, "matchNoNeighbour");
          progress = true;
        }
        continue;
      }
      pending.delete(i);
      progress = true;
      const n = spec.cells;
      let nodes: number[];
      if (spec.spacing === "width") {
        nodes = nodesFromWidths(layer.rLoCm, layer.rHiCm, new Array<number>(n).fill(1));
      } else if (spec.spacing === "mass") {
        nodes = equalMassNodes(layer, geometry, n);
      } else {
        const exponents = ratioExponents(n, spec.fineSide);
        let q: number | null = spec.ratio;
        if (spec.ratioSpec !== "ratio") {
          let w0: number | null = spec.firstWidthCm;
          if (spec.ratioSpec === "match_mass" && neighbour !== null) {
            const neighbourNodes = perLayer[neighbour] as number[];
            const side = neighbour > i ? "inner" : "outer";
            const target = edgeCellMass(derived[neighbour], geometry, neighbourNodes, side);
            w0 = widthForMass(layer, geometry, spec.fineSide === "outer" ? "outer" : "inner", target);
            if (w0 === null) {
              fail(i, "matchTooHeavy");
              continue;
            }
          }
          q = solveRatio(exponents, (layer.rHiCm - layer.rLoCm) / (w0 as number));
          if (q === null) {
            fail(i, spec.ratioSpec === "first_width" ? "firstWidth" : "matchTooHeavy");
            continue;
          }
        }
        nodes = nodesFromWidths(layer.rLoCm, layer.rHiCm, exponents.map((e) => Math.pow(q as number, e)));
      }
      for (let k = 1; k < nodes.length; k++) {
        if (!(nodes[k] > nodes[k - 1])) {
          fail(i, "notIncreasing");
          break;
        }
      }
      if (!failed.has(i)) perLayer[i] = nodes;
    }
  }
  for (const i of pending) fail(i, "matchCycle");
  if (errors.length > 0 || perLayer.some((nodes) => nodes === null)) {
    return { edges: null, perLayer, errors };
  }
  const edges: number[] = [];
  for (const nodes of perLayer as number[][]) {
    if (edges.length === 0) edges.push(...nodes);
    else edges.push(...nodes.slice(1));
  }
  return { edges, perLayer, errors };
}

// ---------------------------------------------------------------------------------------------
// Graded form (mirror of src/mesh/mesh.cu build_graded_nodes)

export interface GradedSegment {
  rStart: number;
  rEnd: number;
  nr: number;
}

export interface GradingParams {
  edgeRatio: number;
  sgOrder: number;
  sgSigma: number;
}

/** Exact TS mirror of src/mesh/mesh.cu build_graded_nodes (as-built 2026-07-15):
 *  super-Gaussian mass weighting q = w(xi)/(r_est^2 + r_ref^2), per-segment
 *  normalization, then geometric-mean boundary matching between segments. */
export function computeGradedWidths(segments: GradedSegment[], grading: GradingParams): number[][] | null {
  const raw: number[][] = [];
  for (const seg of segments) {
    const length = seg.rEnd - seg.rStart;
    const n = seg.nr;
    if (!(length > 0) || !(Number.isInteger(n) && n >= 1)) return null;
    const w = new Array<number>(n);
    if (1.0 - grading.edgeRatio <= 1.0e-12) {
      w.fill(length / n);
      raw.push(w);
      continue;
    }
    let qSum = 0.0;
    for (let k = 0; k < n; k++) {
      const xi = (k + 0.5) / n;
      const u = Math.abs(2.0 * xi - 1.0);
      const exponent = Math.pow(u / grading.sgSigma, grading.sgOrder);
      const weight = grading.edgeRatio + (1.0 - grading.edgeRatio) * Math.exp(-exponent);
      const rEst = seg.rStart + length * xi;
      const rRef = seg.rStart < 1.0e-12 ? length / Math.sqrt(n) : 0.0;
      const q = weight / (rEst * rEst + rRef * rRef);
      w[k] = q;
      qSum += q;
    }
    if (!(Number.isFinite(qSum) && qSum > 0)) return null;
    const scale = length / qSum;
    for (let k = 0; k < n; k++) w[k] *= scale;
    raw.push(w);
  }
  // Collect ALL junction targets from the raw widths first (C++ order), then apply.
  const firstTarget: Array<number | null> = segments.map(() => null);
  const lastTarget: Array<number | null> = segments.map(() => null);
  for (let s = 0; s + 1 < segments.length; s++) {
    const drL = raw[s][raw[s].length - 1];
    const drR = raw[s + 1][0];
    const m = Math.sqrt(drL * drR);
    lastTarget[s] = m;
    firstTarget[s + 1] = m;
  }
  for (let s = 0; s < segments.length; s++) {
    const length = segments[s].rEnd - segments[s].rStart;
    if (!applyBoundaryTargets(raw[s], length, firstTarget[s], lastTarget[s])) return null;
  }
  return raw;
}

function applyBoundaryTargets(widths: number[], length: number, left: number | null, right: number | null): boolean {
  const n = widths.length;
  const tol = 1.0e-12 * Math.max(1.0, Math.abs(length));
  if (left === null && right === null) return true;
  if (n === 1) {
    const target = left !== null ? left : (right as number);
    if (left !== null && right !== null && Math.abs(left - right) > tol) return false;
    if (Math.abs(length - target) > tol) return false;
    widths[0] = length;
    return true;
  }
  if (left !== null && right !== null) {
    if (n === 2) {
      if (Math.abs(left + right - length) > tol) return false;
      widths[0] = left;
      widths[1] = length - left;
      return true;
    }
    const interiorOld = length - widths[0] - widths[n - 1];
    const interiorNew = length - left - right;
    if (!(interiorOld > 0 && interiorNew > 0)) return false;
    const scale = interiorNew / interiorOld;
    for (let k = 1; k < n - 1; k++) widths[k] *= scale;
    widths[0] = left;
    widths[n - 1] = right;
    return true;
  }
  if (left !== null) {
    const tailOld = length - widths[0];
    const tailNew = length - left;
    if (!(tailOld > 0 && tailNew > 0)) return false;
    const scale = tailNew / tailOld;
    for (let k = 1; k < n; k++) widths[k] *= scale;
    widths[0] = left;
    return true;
  }
  const headOld = length - widths[n - 1];
  const headNew = length - (right as number);
  if (!(headOld > 0 && headNew > 0)) return false;
  const scale = headNew / headOld;
  for (let k = 0; k < n - 1; k++) widths[k] *= scale;
  widths[n - 1] = right as number;
  return true;
}

// ---------------------------------------------------------------------------------------------
// zoning_intent (recommended and directly edited): the deck carries the intent, the solver zones it

/** Zoning density of a zoning_intent: the form's list, or the initial regions (a varying corona
 *  ramp enters with its mean density, the void padding with the void density). */
export function zoningDensityRegions(f: FormState, z: ZoningIntentForm): Array<{ rEndCm: number; rho: number }> | null {
  if (z.densityRegions.length > 0) return z.densityRegions;
  const layers = meshLayers1d(f);
  if (layers === null) return null;
  return layers.map((layer) => ({
    rEndCm: layer.rHiCm,
    rho:
      layer.rhoConst !== null
        ? layer.rhoConst
        : layerMass(layer, f.main.geometry1d, layer.rLoCm, layer.rHiCm) /
          shellVolume(f.main.geometry1d, layer.rLoCm, layer.rHiCm),
  }));
}

/** Pins of a zoning_intent: the form's pins plus, when asked, one at every interior layer
 *  boundary (material interfaces, the target surface, the end of the corona ramp). */
export function zoningPins(f: FormState, z: ZoningIntentForm): Array<{ rCm: number; ratioJumpAllowed: boolean }> {
  const pins = z.pins.map((pin) => ({ ...pin }));
  if (z.pinInterfaces) {
    const layers = meshLayers1d(f) ?? [];
    const rMax = deckRound(toCanonical(f.mesh.rMax, "length"));
    const span = Math.max(rMax - deckRound(toCanonical(f.mesh.rMin, "length")), 1e-300);
    for (const layer of layers) {
      const r = layer.rHiCm;
      if (r >= rMax) continue;
      if (pins.some((pin) => Math.abs(pin.rCm - r) <= 1e-12 * span)) continue;
      pins.push({ rCm: r, ratioJumpAllowed: true });
    }
  }
  return pins.sort((a, b) => a.rCm - b.rCm);
}

/** Nodes the solver zones from the form's zoning_intent, computed with Studio's copy of the
 *  solver's zoning (zoningIntent.ts), or the solver's error. The input is the intent the deck
 *  writes (meshEmit.ts emitZoningIntentMesh) as the namelist builder hands it to the zoning: the
 *  domain and the numbers as written, the zoning density as a piecewise-constant rho0 whose region
 *  ends inside the domain are quadrature events, and the solver's defaults where the form leaves a
 *  bound empty. Not included: the bands that resolution_requirement apply="enforce" adds before
 *  zoning (they tighten the mesh only where the intent's own bands are looser; the server check
 *  shows the solver's final mesh). */
export function zoningIntentNodes1d(f: FormState): { nodes: number[] } | { error: string } {
  const z = f.mesh.zoningIntent;
  const geometry = f.main.geometry1d;
  const required = z.measure === "spherical_cell_mass" ? "spherical" : z.measure === "cylindrical_line_mass" ? "cylindrical" : null;
  if (required !== null && geometry !== required) {
    return {
      error: `Mesh.zoning_intent.measure='${z.measure}' requires Geometry '${required}', got '${geometry}'; 'width' and 'areal_mass' are geometry-independent`,
    };
  }
  if (!(Number.isInteger(z.nCells) && z.nCells >= 1)) return { error: "Mesh.zoning_intent.n_cells must be >= 1" };
  const rMin = deckRound(toCanonical(f.mesh.rMin, "length"));
  const rMax = deckRound(toCanonical(f.mesh.rMax, "length"));
  if (!Number.isFinite(rMin) || !Number.isFinite(rMax)) return { error: "Mesh.zoning_intent requires r_min and r_max" };
  const extraEvents = [...z.extraEventsCm];
  let rho0: ((r: number) => number) | null = null;
  if (isMassMeasure(z.measure)) {
    const density = zoningDensityRegions(f, z);
    if (density === null || density.length === 0) return { error: `Mesh.zoning_intent.${z.measure} requires density_regions` };
    let previous = -Infinity;
    for (const region of density) {
      if (!(region.rEndCm > previous)) return { error: "Mesh.zoning_intent.density_regions r_end values must be strictly increasing" };
      if (!(region.rho >= 0)) return { error: "Mesh.zoning_intent.density_regions[k].rho must be >= 0" };
      previous = region.rEndCm;
    }
    const rLast = density[density.length - 1].rEndCm;
    if (Math.abs(rLast - rMax) > 1.0e-12 * Math.max(1.0, Math.abs(rMax))) {
      return { error: "Mesh.zoning_intent.density_regions last r_end must equal the outer boundary" };
    }
    const rEnds = density.map((region) => region.rEndCm);
    const rhos = density.map((region) => region.rho);
    for (const r of rEnds) if (r > rMin && r < rMax) extraEvents.push(r);
    rho0 = (r) => {
      // std::upper_bound over the region ends, the last region beyond them.
      let lo = 0;
      let hi = rEnds.length;
      while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (r < rEnds[mid]) hi = mid;
        else lo = mid + 1;
      }
      return rhos[Math.min(lo, rhos.length - 1)];
    };
  }
  const cfg: ZoningIntentConfig = {
    nCells: z.nCells,
    measure: z.measure,
    pins: zoningPins(f, z).map((pin) => ({ r: pin.rCm, ratioJumpAllowed: pin.ratioJumpAllowed })),
    profile: z.profile.map((point) => ({ r: point.rCm, w: point.w })),
    anchors: z.anchors.map((anchor) => ({ r: anchor.rCm, halfWidth: anchor.halfWidthCm, logAmplitude: anchor.logAmplitude })),
    bands: z.bands.map((band) => ({
      measureFracBegin: band.fracBegin,
      measureFracEnd: band.fracEnd,
      cellMeasureMin: band.cellMeasureMin ?? 0.0,
      cellMeasureMax: band.cellMeasureMax ?? 0.0,
    })),
    extraEvents,
    drMin: z.drMinCm ?? 0.0,
    cellMeasureMin: z.cellMeasureMin ?? 0.0,
    cellMeasureMax: z.cellMeasureMax ?? 0.0,
    preferredRatio: z.preferredRatio ?? 1.3,
    ratioHardMax: z.ratioHardMax ?? 2.0,
    minCellsPerSegment: z.minCellsPerSegment ?? 1,
  };
  const result = computeZoningIntentNodes(rMin, rMax, cfg, rho0);
  if (!result.ok) return { error: `[mesh-zoning-intent] ${result.diag.code}: ${result.diag.message}` };
  return { nodes: result.nodes };
}

// ---------------------------------------------------------------------------------------------
// Nodes of the current 1D mesh as Studio can compute them

/** Node radii [cm] of the form's 1D mesh, or null when they cannot be computed (an invalid form, or
 *  a zoning_intent the solver's zoning refuses: zoningIntentNodes1d gives the reason). */
export function computeMeshNodes1d(f: FormState): number[] | null {
  if (f.main.dimension !== "1D_SPH") return null;
  const rMin = toCanonical(f.mesh.rMin, "length");
  const rMax = toCanonical(f.mesh.rMax, "length");
  const method = f.mesh.grid1d;
  if (method === "uniform") {
    const nr = f.mesh.nr;
    if (!(Number.isInteger(nr) && nr >= 1 && nr <= 2_000_000) || !(rMax > rMin)) return null;
    const nodes: number[] = [];
    for (let i = 0; i <= nr; i++) nodes.push(rMin + (i * (rMax - rMin)) / nr);
    return nodes;
  }
  if (method === "graded") {
    let segs: GradedSegment[] | null;
    if (f.mesh.segmentSource === "regions") {
      const auto = computeRegionSegments(f);
      segs = auto === null ? null : auto.map((s) => ({ rStart: s.rStartCm, rEnd: s.rEndCm, nr: s.nr }));
    } else {
      segs = [];
      let start = rMin;
      for (const segment of f.mesh.segments) {
        const end = toCanonical(segment.rEnd, "length");
        segs.push({ rStart: start, rEnd: end, nr: segment.nr });
        start = end;
      }
    }
    if (segs === null || segs.length === 0) return null;
    const widths = computeGradedWidths(segs, f.mesh.grading);
    if (widths === null) return null;
    const nodes = [rMin];
    let r = rMin;
    for (const w of widths) {
      for (const dw of w) {
        r += dw;
        nodes.push(r);
      }
    }
    return nodes;
  }
  if (method === "layers") return computeLayerNodes(f).edges;
  if (method === "explicit") {
    const nodes = f.mesh.explicitNodes.nodesCm;
    return nodes.length >= 2 ? [...nodes] : null;
  }
  if (method === "recommended" || method === "zoning_intent") {
    const zoned = zoningIntentNodes1d(f);
    return "nodes" in zoned ? zoned.nodes : null;
  }
  return null;
}

// ---------------------------------------------------------------------------------------------
// Cell metrics for the diagnostics

export interface MeshCellMetrics {
  edges: number[];
  widths: number[];
  /** Initial density of each cell: the deck's density at the cell centre (as the solver samples it). */
  rho: number[];
  /** rho * volume (g; g/cm for cylinders; g/cm² for planar). */
  masses: number[];
  /** rho * width [g/cm²]. */
  localArealMass: number[];
  /** Cell mass over the reference area of the target surface (the resolution requirement's measure). */
  referenceArealMass: number[];
  materials: string[];
  /** Outer radius of the target (outer edge of the last region). */
  targetOuterCm: number;
}

export function cellMetrics(f: FormState, edges: number[]): MeshCellMetrics | null {
  const layers = meshLayers1d(f);
  if (layers === null || edges.length < 2) return null;
  const geometry = f.main.geometry1d;
  const regions = layers.filter((layer) => layer.kind === "region");
  const targetOuter = regions[regions.length - 1].rHiCm;
  const area = referenceArea(geometry, targetOuter);
  const out: MeshCellMetrics = {
    edges,
    widths: [],
    rho: [],
    masses: [],
    localArealMass: [],
    referenceArealMass: [],
    materials: [],
    targetOuterCm: targetOuter,
  };
  let li = 0;
  for (let i = 0; i + 1 < edges.length; i++) {
    const a = edges[i];
    const b = edges[i + 1];
    if (!(b > a)) return null;
    const centre = 0.5 * (a + b);
    while (li + 1 < layers.length && centre >= layers[li].rHiCm) li++;
    while (li > 0 && centre < layers[li].rLoCm) li--;
    const layer = layers[li];
    const rho = layer.rhoAt(centre);
    const mass = rho * shellVolume(geometry, a, b);
    out.widths.push(b - a);
    out.rho.push(rho);
    out.masses.push(mass);
    out.localArealMass.push(rho * (b - a));
    out.referenceArealMass.push(mass / area);
    out.materials.push(layer.materialName);
  }
  return out;
}

export interface MeshSummary {
  nCells: number;
  minWidthCm: number;
  /** Largest ratio of adjacent cell masses anywhere (void cells excluded). */
  maxAdjacentMassRatio: number;
  interfaceRatios: Array<{ left: string; right: string; ratio: number; rCm: number }>;
  /** Outermost non-void cell (the laser-side surface of the target). */
  surfaceWidthCm: number | null;
  surfaceLocalArealMass: number | null;
  surfaceReferenceArealMass: number | null;
}

export function summarizeMesh(metrics: MeshCellMetrics): MeshSummary {
  const n = metrics.masses.length;
  let minWidth = Infinity;
  let maxRatio = 1;
  const interfaceRatios: MeshSummary["interfaceRatios"] = [];
  let surface = -1;
  for (let i = 0; i < n; i++) {
    minWidth = Math.min(minWidth, metrics.widths[i]);
    if (metrics.materials[i] !== "VOID") surface = i;
    if (i + 1 < n && metrics.materials[i] !== "VOID" && metrics.materials[i + 1] !== "VOID") {
      const a = metrics.masses[i];
      const b = metrics.masses[i + 1];
      const ratio = Math.max(a, b) / Math.min(a, b);
      if (Number.isFinite(ratio)) maxRatio = Math.max(maxRatio, ratio);
      if (metrics.materials[i] !== metrics.materials[i + 1]) {
        interfaceRatios.push({
          left: metrics.materials[i],
          right: metrics.materials[i + 1],
          ratio,
          rCm: metrics.edges[i + 1],
        });
      }
    }
  }
  return {
    nCells: n,
    minWidthCm: minWidth,
    maxAdjacentMassRatio: maxRatio,
    interfaceRatios,
    surfaceWidthCm: surface >= 0 ? metrics.widths[surface] : null,
    surfaceLocalArealMass: surface >= 0 ? metrics.localArealMass[surface] : null,
    surfaceReferenceArealMass: surface >= 0 ? metrics.referenceArealMass[surface] : null,
  };
}

// ---------------------------------------------------------------------------------------------
// Imported node lists

export interface ParsedNodes {
  nodes: number[];
  columns: number;
  rows: number;
}

/** Numbers of a pasted node list: one value per line, several per line (comma, space, tab or
 *  semicolon separated), or a table whose `column` (0-based) holds the radii. Lines that start with
 *  `#` or hold no number (headers) are skipped. Values are multiplied by `toCm`. */
export function parseNodeText(text: string, toCm: number, column = 0): ParsedNodes | { error: "empty" | "column" | "number" } {
  const rows: number[][] = [];
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (trimmed.length === 0 || trimmed.startsWith("#")) continue;
    const tokens = trimmed.split(/[\s,;]+/).filter((token) => token.length > 0);
    const values = tokens.map(Number);
    if (values.every((v) => Number.isNaN(v))) continue;
    if (values.some((v) => !Number.isFinite(v))) return { error: "number" };
    rows.push(values);
  }
  if (rows.length === 0) return { error: "empty" };
  const columns = Math.max(...rows.map((row) => row.length));
  let values: number[];
  if (rows.length === 1 || rows.every((row) => row.length === 1)) {
    values = rows.flat();
  } else {
    if (!(column >= 0 && column < columns) || rows.some((row) => row.length <= column)) return { error: "column" };
    values = rows.map((row) => row[column]);
  }
  return { nodes: values.map((v) => v * toCm), columns: rows.length === 1 ? 1 : columns, rows: rows.length };
}

/** Node radii stored in a run's frozen configuration, config/<case>_frozen.json (Mesh.explicit_nodes,
 *  which a zoning_intent or auto_regions mesh fills with its computed nodes); null when the mesh form
 *  does not store them. */
export function nodesFromFrozenConfig(json: unknown): number[] | null {
  if (typeof json !== "object" || json === null) return null;
  const root = json as Record<string, unknown>;
  const candidates = [root.mesh, root.Mesh, (root.config as Record<string, unknown> | undefined)?.mesh];
  for (const mesh of candidates) {
    if (typeof mesh !== "object" || mesh === null) continue;
    const nodes = (mesh as Record<string, unknown>).explicit_nodes;
    if (Array.isArray(nodes) && nodes.length >= 2 && nodes.every((v) => typeof v === "number" && Number.isFinite(v))) {
      return nodes as number[];
    }
  }
  return null;
}

// ---------------------------------------------------------------------------------------------
// Conditions that determine a recommended mesh

/** Canonical description of the form fields that enter recommend-mesh's conditions and the
 *  solver's resolution requirement (target layers, drive, duration, physics switches). A
 *  recommended mesh whose stored key differs from the form's key was made for other conditions. */
export function recommendationConditionsKey(f: FormState): string {
  const len = (x: { value: number; unit: string }) => deckRound(toCanonical(x, "length"));
  const time = (x: { value: number; unit: string }) => toCanonical(x, "time");
  const power = (x: { value: number; unit: string }) => toCanonical(x, "power");
  const laser = f.laser;
  const waveform =
    laser.waveformMode === "square"
      ? { square: [power(laser.powerW), time(laser.pulseDuration), time(laser.riseTime), time(laser.fallTime)] }
      : laser.waveformMode === "gaussian"
        ? {
            gaussian: [
              laser.gaussianSpec,
              laser.gaussianSpec === "peak" ? power(laser.gaussianPeakW) : laser.gaussianEnergyJ,
              time(laser.gaussianFwhm),
              time(laser.gaussianCenter),
            ],
          }
        : { table: laser.waveformPoints.map((p) => [p.t, p.v]) };
  const used = new Set(f.geometry.regions.map((r) => r.materialName));
  const key = {
    geometry: f.main.geometry1d,
    tEnd: time(f.main.tEnd),
    temperatureModel: f.main.temperatureModel,
    rMin: len(f.mesh.rMin),
    rMax: len(f.mesh.rMax),
    regions: f.geometry.regions.map((r) => [r.materialName, len(r.rOuter), r.rho]),
    vacuum: f.geometry.vacuumOutside1d,
    corona: f.geometry.coronaRamp1d.enabled && f.geometry.vacuumOutside1d ? f.geometry.coronaRamp1d : null,
    materials: f.materials
      .filter((m) => used.has(m.name))
      .map((m) => [m.name, m.A, m.Z, m.eosModel, m.eosModel === "tmat" ? m.eosFile.trim() : ""]),
    zbar: f.zbarFixedValue,
    radiation: f.radiation.enabled,
    laser: laser.enabled ? { wavelength: laser.wavelengthNm, waveform } : null,
    conduction: [f.conduction.enabled, f.conduction.solver],
  };
  return JSON.stringify(key);
}

// ---------------------------------------------------------------------------------------------
// Form validation of the 1D mesh methods (mirrors the solver's checks; validate is the authority)

const MASS_MEASURES: ZoningMeasure[] = ["areal_mass", "cylindrical_line_mass", "spherical_cell_mass"];

export function isMassMeasure(measure: ZoningMeasure): boolean {
  return MASS_MEASURES.includes(measure);
}

/** The mass measure that matches a 1D geometry (the measure recommend-mesh uses). */
export function massMeasureFor(geometry: Geometry1d): ZoningMeasure {
  if (geometry === "spherical") return "spherical_cell_mass";
  if (geometry === "cylindrical") return "cylindrical_line_mass";
  return "areal_mass";
}

/** Errors of the 1D mesh method fields (empty for uniform and graded, which formState checks). */
export function mesh1dErrors(f: FormState): string[] {
  const v = t().mesh1d.errors;
  const errs: string[] = [];
  if (f.main.dimension !== "1D_SPH") return errs;
  const method = f.mesh.grid1d;
  const rMin = toCanonical(f.mesh.rMin, "length");
  const rMax = toCanonical(f.mesh.rMax, "length");
  const span = Math.max(Math.abs(rMax - rMin), 1e-300);
  const inside = (r: number) => Number.isFinite(r) && r > rMin && r < rMax;
  if (method === "layers") {
    const layers = meshLayers1d(f);
    if (layers === null) {
      errs.push(v.layersUnresolved);
    } else {
      const result = computeLayerNodes(f, layers);
      for (const error of result.errors) errs.push(v.layer(error.layer + 1, t().mesh1d.layerErrors[error.code]));
    }
  }
  if (method === "explicit") {
    const nodes = f.mesh.explicitNodes.nodesCm;
    if (nodes.length < 2) {
      errs.push(v.explicitTooFew);
    } else {
      if (nodes.some((r) => !Number.isFinite(r))) errs.push(v.explicitNotFinite);
      for (let i = 1; i < nodes.length; i++) {
        if (!(nodes[i] > nodes[i - 1])) {
          errs.push(v.explicitNotIncreasing(i + 1));
          break;
        }
      }
      const tolerance = 1e-12 * span;
      if (Math.abs(nodes[0] - rMin) > tolerance || Math.abs(nodes[nodes.length - 1] - rMax) > tolerance) {
        errs.push(v.explicitEndpoints);
      }
    }
  }
  if (method === "zoning_intent" || method === "recommended") {
    const z = f.mesh.zoningIntent;
    if (!(Number.isInteger(z.nCells) && z.nCells >= 1 && z.nCells <= 2_000_000)) errs.push(v.intentCells);
    if (z.measure === "spherical_cell_mass" && f.main.geometry1d !== "spherical") errs.push(v.intentMeasureGeometry);
    if (z.measure === "cylindrical_line_mass" && f.main.geometry1d !== "cylindrical") errs.push(v.intentMeasureGeometry);
    if (z.densityRegions.length > 0) {
      let previous = rMin;
      for (const [i, region] of z.densityRegions.entries()) {
        if (!(region.rEndCm > previous) || !(region.rho > 0) || !Number.isFinite(region.rho)) {
          errs.push(v.intentDensityRegion(i + 1));
        }
        previous = region.rEndCm;
      }
      if (Math.abs(z.densityRegions[z.densityRegions.length - 1].rEndCm - rMax) > 1e-12 * span) {
        errs.push(v.intentDensityEnd);
      }
    } else if (isMassMeasure(z.measure) && meshLayers1d(f) === null) {
      errs.push(v.layersUnresolved);
    }
    z.pins.forEach((pin, i) => {
      if (!inside(pin.rCm)) errs.push(v.intentPin(i + 1));
    });
    z.profile.forEach((point, i) => {
      if (!Number.isFinite(point.rCm) || !(point.w > 0) || !Number.isFinite(point.w)) errs.push(v.intentProfile(i + 1));
    });
    z.anchors.forEach((anchor, i) => {
      if (
        !Number.isFinite(anchor.rCm) ||
        !(anchor.halfWidthCm > 0) ||
        !(Math.abs(anchor.logAmplitude) <= Math.log(1e4))
      ) {
        errs.push(v.intentAnchor(i + 1));
      }
    });
    z.bands.forEach((band, i) => {
      const okRange = band.fracBegin >= 0 && band.fracEnd <= 1 && band.fracBegin < band.fracEnd;
      const okMin = band.cellMeasureMin === null || band.cellMeasureMin >= 0;
      const okMax = band.cellMeasureMax === null || band.cellMeasureMax > 0;
      const okOrder =
        band.cellMeasureMin === null || band.cellMeasureMax === null || band.cellMeasureMax >= band.cellMeasureMin;
      const okAny = band.cellMeasureMin !== null || band.cellMeasureMax !== null;
      if (!(okRange && okMin && okMax && okOrder && okAny)) errs.push(v.intentBand(i + 1));
    });
    z.extraEventsCm.forEach((r, i) => {
      if (!inside(r)) errs.push(v.intentEvent(i + 1));
    });
    if (z.drMinCm !== null && !(z.drMinCm >= 0)) errs.push(v.intentDrMin);
    if (z.cellMeasureMin !== null && !(z.cellMeasureMin >= 0)) errs.push(v.intentMeasureBounds);
    if (z.cellMeasureMax !== null && !(z.cellMeasureMax > 0)) errs.push(v.intentMeasureBounds);
    if (z.preferredRatio !== null && !(z.preferredRatio > 1)) errs.push(v.intentPreferredRatio);
    if (z.ratioHardMax !== null && !(z.ratioHardMax > 1 && z.ratioHardMax <= 2)) errs.push(v.intentHardRatio);
    if (z.minCellsPerSegment !== null && !(Number.isInteger(z.minCellsPerSegment) && z.minCellsPerSegment >= 1)) {
      errs.push(v.intentMinCells);
    }
  }
  if (method === "recommended") {
    const meta = f.mesh.recommendation;
    if (meta === null) errs.push(v.recommendationMissing);
    else if (meta.conditionsKey !== recommendationConditionsKey(f)) errs.push(v.recommendationStale);
    if (f.mesh.resolutionRequirement.apply !== "enforce") errs.push(v.recommendationEnforce);
  }
  const rr = f.mesh.resolutionRequirement;
  if (rr.empirical !== null && method !== "uniform" && method !== "graded") {
    if (!f.laser.enabled) errs.push(v.empiricalNeedsLaser);
    if (rr.apply !== "enforce") errs.push(v.empiricalNeedsEnforce);
    if (!(method === "zoning_intent" || method === "recommended") || !isMassMeasure(f.mesh.zoningIntent.measure)) {
      errs.push(v.empiricalNeedsMassIntent);
    }
  }
  return errs;
}

/** Advisories of the 1D mesh (shown in the mesh section; they do not block the deck). */
export function mesh1dWarnings(f: FormState): string[] {
  const w = t().mesh1d.warnings;
  const out: string[] = [];
  if (f.main.dimension !== "1D_SPH") return out;
  const method = f.mesh.grid1d;
  if (method === "explicit") {
    const layers = meshLayers1d(f);
    const nodes = f.mesh.explicitNodes.nodesCm;
    if (layers !== null && nodes.length >= 2) {
      for (const layer of layers.slice(0, -1)) {
        const r = layer.rHiCm;
        let nearest = Infinity;
        for (const node of nodes) nearest = Math.min(nearest, Math.abs(node - r));
        if (nearest > 1e-9 * Math.max(Math.abs(r), 1e-300)) out.push(w.interfaceOffNode(r * 1.0e4, nearest * 1.0e4));
      }
    }
  }
  if (method === "recommended" && f.mesh.recommendation !== null) {
    const meta = f.mesh.recommendation;
    if (meta.flags.includes("extrapolation")) out.push(w.extrapolation);
    if (meta.flags.includes("unconverged_reference")) out.push(w.unconverged);
    if (meta.status !== "validated") out.push(w.notValidated(meta.status));
  }
  if (method === "zoning_intent" && f.mesh.recommendation !== null && f.mesh.recommendation.edited) {
    out.push(w.editedRecommendation);
  }
  // A zoning density with its own regions must change where the target does: a region boundary
  // off a material interface puts that interface inside a cell (a recommendation made from a deck
  // whose placeholder mesh had no node there reads the interface at a node of that mesh).
  if ((method === "zoning_intent" || method === "recommended") && f.mesh.zoningIntent.densityRegions.length > 0) {
    const layers = meshLayers1d(f);
    if (layers !== null) {
      const span = Math.max(layers[layers.length - 1].rHiCm - layers[0].rLoCm, 1e-300);
      const ends = f.mesh.zoningIntent.densityRegions.map((region) => region.rEndCm);
      for (const layer of layers.slice(0, -1)) {
        if (layer.kind === "corona" || (layer.kind === "void" && layer.rHiCm >= layers[layers.length - 1].rHiCm)) continue;
        const r = layer.rHiCm;
        if (!ends.some((end) => Math.abs(end - r) <= 1e-9 * span)) out.push(w.densityOffInterface(r * 1.0e4));
      }
    }
  }
  if (f.laser.enabled && (method === "layers" || method === "explicit") && f.mesh.resolutionRequirement.apply === "default") {
    out.push(w.laserRequirementReportOnly);
  }
  return out;
}

/** Switch the 1D mesh method (immer-style draft). The new method starts from the current mesh
 *  where it can: an imported node list takes the nodes of the method left behind; a declarative
 *  zoning made from scratch uses the geometry's mass measure; leaving the recommended mesh for its
 *  zoning_intent marks the recommendation as edited. */
export function switchMesh1dMethod(f: FormState, method: MeshMethod1d): void {
  const previous = f.mesh.grid1d;
  if (previous === method) return;
  if (method === "explicit" && f.mesh.explicitNodes.nodesCm.length < 2) {
    const nodes = computeMeshNodes1d(f);
    if (nodes !== null) f.mesh.explicitNodes = { nodesCm: nodes, source: t().mesh1d.ui.methods[previous] };
  }
  if (method === "zoning_intent") {
    if (previous === "recommended" && f.mesh.recommendation !== null) {
      f.mesh.recommendation = { ...f.mesh.recommendation, edited: true };
    } else if (f.mesh.recommendation === null && JSON.stringify(f.mesh.zoningIntent) === JSON.stringify(defaultZoningIntent())) {
      f.mesh.zoningIntent = { ...defaultZoningIntent(), measure: massMeasureFor(f.main.geometry1d) };
    }
  }
  f.mesh.grid1d = method;
}
