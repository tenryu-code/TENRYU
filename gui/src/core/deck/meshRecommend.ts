// recommend-mesh output (tools/assist/recommend_mesh.py, schema tenryu.assist.mesh_recommendation.v1)
// into the form, and the finer companion deck of a convergence pair.
import type { FormState } from "./formState";
import { toCanonical } from "../units";
import {
  defaultLayerZoning,
  meshLayers1d,
  recommendationConditionsKey,
  type EmpiricalForm,
  type MeshRecommendationMeta,
  type ZoningIntentForm,
  type ZoningMeasure,
} from "./mesh1d";

export interface ParsedRecommendation {
  zoningIntent: ZoningIntentForm;
  empirical: EmpiricalForm | null;
  apply: "report" | "enforce";
  rMinCm: number;
  rMaxCm: number;
  geometry: string;
  status: string;
  flags: string[];
  warnings: string[];
  caseIds: string[];
  confidence: string;
  mode: string;
  surfaceCeilingGcm2: number | null;
  achievedSurfaceGcm2: number | null;
  /** Errors of the validation attempts (validation.attempts[].error), for a failed recommendation. */
  attemptErrors: string[];
}

type Json = Record<string, unknown>;

function isObject(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function num(value: unknown, what: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`recommend-mesh: ${what} is not a finite number`);
  return value;
}

function optNum(value: unknown, what: string): number | null {
  return value === undefined || value === null ? null : num(value, what);
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

const MEASURES: ZoningMeasure[] = ["width", "areal_mass", "cylindrical_line_mass", "spherical_cell_mass"];

/** Mesh.zoning_intent written by the solver's vocabulary (SPECIFICATION 6.4.2) into the form. */
export function zoningIntentFromDict(intent: Json): ZoningIntentForm {
  const measure = intent.measure ?? "width";
  if (typeof measure !== "string" || !MEASURES.includes(measure as ZoningMeasure)) {
    throw new Error(`recommend-mesh: unknown zoning measure ${String(measure)}`);
  }
  const list = (key: string): Json[] => {
    const value = intent[key];
    if (value === undefined) return [];
    if (!Array.isArray(value) || !value.every(isObject)) throw new Error(`recommend-mesh: zoning_intent.${key} is not a list of dicts`);
    return value;
  };
  const known = new Set([
    "n_cells",
    "measure",
    "density_regions",
    "pins",
    "profile",
    "anchors",
    "bands",
    "extra_events",
    "dr_min",
    "cell_measure_min",
    "cell_measure_max",
    "preferred_ratio",
    "ratio_hard_max",
    "min_cells_per_segment",
  ]);
  for (const key of Object.keys(intent)) {
    if (!known.has(key)) throw new Error(`recommend-mesh: zoning_intent key ${key} is not handled by Studio`);
  }
  const events = intent.extra_events;
  if (events !== undefined && !(Array.isArray(events) && events.every((v) => typeof v === "number"))) {
    throw new Error("recommend-mesh: zoning_intent.extra_events is not a list of numbers");
  }
  return {
    nCells: num(intent.n_cells, "zoning_intent.n_cells"),
    measure: measure as ZoningMeasure,
    densityRegions: list("density_regions").map((d) => ({ rEndCm: num(d.r_end, "density r_end"), rho: num(d.rho, "density rho") })),
    // The recommender lists every pin it wants, so none is added for the interfaces.
    pinInterfaces: false,
    pins: list("pins").map((p) => ({ rCm: num(p.r, "pin r"), ratioJumpAllowed: p.ratio_jump_allowed === true })),
    profile: list("profile").map((p) => ({ rCm: num(p.r, "profile r"), w: num(p.w, "profile w") })),
    anchors: list("anchors").map((a) => ({
      rCm: num(a.r, "anchor r"),
      halfWidthCm: num(a.half_width, "anchor half_width"),
      logAmplitude: num(a.log_amplitude, "anchor log_amplitude"),
    })),
    bands: list("bands").map((b) => ({
      fracBegin: num(b.measure_frac_begin, "band begin"),
      fracEnd: num(b.measure_frac_end, "band end"),
      cellMeasureMin: optNum(b.cell_measure_min, "band cell_measure_min"),
      cellMeasureMax: optNum(b.cell_measure_max, "band cell_measure_max"),
    })),
    extraEventsCm: (events as number[] | undefined) ?? [],
    drMinCm: optNum(intent.dr_min, "dr_min"),
    cellMeasureMin: optNum(intent.cell_measure_min, "cell_measure_min"),
    cellMeasureMax: optNum(intent.cell_measure_max, "cell_measure_max"),
    preferredRatio: optNum(intent.preferred_ratio, "preferred_ratio"),
    ratioHardMax: optNum(intent.ratio_hard_max, "ratio_hard_max"),
    minCellsPerSegment: optNum(intent.min_cells_per_segment, "min_cells_per_segment"),
  };
}

export function empiricalFromDict(value: unknown): EmpiricalForm | null {
  if (value === undefined || value === null) return null;
  if (!isObject(value)) throw new Error("recommend-mesh: empirical is not a dict");
  if (typeof value.reference_sha256 !== "string") throw new Error("recommend-mesh: empirical.reference_sha256 is missing");
  return {
    referenceSha256: value.reference_sha256,
    caseIds: strings(value.case_ids),
    surfaceCeilingGcm2: num(value.surface_ceiling_g_cm2, "empirical.surface_ceiling_g_cm2"),
    referenceAprioriGcm2: num(value.reference_apriori_g_cm2, "empirical.reference_apriori_g_cm2"),
  };
}

/** The JSON that recommend-mesh writes with -o (one object). */
export function parseRecommendation(text: string): ParsedRecommendation {
  let payload: unknown;
  try {
    payload = JSON.parse(text);
  } catch (err) {
    throw new Error(`recommend-mesh: the output is not JSON (${String(err)})`);
  }
  if (!isObject(payload)) throw new Error("recommend-mesh: the output is not a JSON object");
  const mesh = payload.mesh;
  if (!isObject(mesh) || !isObject(mesh.zoning_intent)) throw new Error("recommend-mesh: the output has no mesh.zoning_intent");
  const rr = isObject(mesh.resolution_requirement) ? mesh.resolution_requirement : {};
  const recommendation = isObject(payload.recommendation) ? payload.recommendation : {};
  const validation = isObject(payload.validation) ? payload.validation : {};
  const evidence = Array.isArray(payload.evidence) ? payload.evidence.filter(isObject) : [];
  const attempts = Array.isArray(validation.attempts) ? validation.attempts.filter(isObject) : [];
  return {
    zoningIntent: zoningIntentFromDict(mesh.zoning_intent),
    empirical: empiricalFromDict(rr.empirical),
    apply: rr.apply === "report" ? "report" : "enforce",
    rMinCm: num(mesh.r_min, "mesh.r_min"),
    rMaxCm: num(mesh.r_max, "mesh.r_max"),
    geometry: typeof mesh.geometry_1d === "string" ? mesh.geometry_1d : "",
    status: typeof validation.status === "string" ? validation.status : "unvalidated",
    flags: strings(payload.flags),
    warnings: strings(payload.warnings),
    caseIds: evidence.map((item) => item.id).filter((id): id is string => typeof id === "string"),
    confidence: typeof payload.confidence === "string" ? payload.confidence : "",
    mode: typeof recommendation.mode === "string" ? recommendation.mode : "",
    surfaceCeilingGcm2: optNum(recommendation.surface_areal_mass_g_cm2, "surface_areal_mass_g_cm2"),
    achievedSurfaceGcm2: optNum(validation.achieved_surface_areal_mass_g_cm2, "achieved_surface_areal_mass_g_cm2"),
    attemptErrors: attempts.map((a) => a.error).filter((e): e is string => typeof e === "string"),
  };
}

export class RecommendationMismatch extends Error {}

/** Put a recommendation into the form (immer-style draft): the recommended method, its
 *  zoning_intent and resolution requirement, and the conditions key it was made for. The
 *  recommendation must belong to this form: same geometry and domain. */
export function applyRecommendation(
  f: FormState,
  rec: ParsedRecommendation,
  conditionsKey: string,
  binary: string,
  createdAt: string,
): void {
  const rMin = toCanonical(f.mesh.rMin, "length");
  const rMax = toCanonical(f.mesh.rMax, "length");
  const tolerance = 1e-9 * Math.max(Math.abs(rMax - rMin), 1e-300);
  if (rec.geometry !== "" && rec.geometry !== f.main.geometry1d) {
    throw new RecommendationMismatch(`geometry ${rec.geometry} differs from the form's ${f.main.geometry1d}`);
  }
  if (Math.abs(rec.rMinCm - rMin) > tolerance || Math.abs(rec.rMaxCm - rMax) > tolerance) {
    throw new RecommendationMismatch("the recommended mesh spans another domain than the form's r_min..r_max");
  }
  f.mesh.grid1d = "recommended";
  f.mesh.zoningIntent = rec.zoningIntent;
  f.mesh.resolutionRequirement = { apply: rec.apply, empirical: rec.empirical };
  const meta: MeshRecommendationMeta = {
    conditionsKey,
    createdAt,
    status: rec.status,
    flags: rec.flags,
    warnings: rec.warnings,
    caseIds: rec.caseIds,
    confidence: rec.confidence,
    mode: rec.mode,
    surfaceCeilingGcm2: rec.surfaceCeilingGcm2,
    achievedSurfaceGcm2: rec.achievedSurfaceGcm2,
    nCells: rec.zoningIntent.nCells,
    binary,
    edited: false,
  };
  f.mesh.recommendation = meta;
}

/** Deck for recommend-mesh's --deck input: the form with a placeholder mesh of equal-width cells
 *  in every layer. The recommender replaces the Mesh block and takes the target's layers from the
 *  solver's preview, which samples the deck's density at the placeholder's cell centres: a material
 *  interface between two nodes would move to a node of the placeholder (243 µm read as 244 µm on a
 *  uniform 2 µm mesh), so every interface must be a node. */
export function placeholderForRecommendation(f: FormState): FormState {
  const copy = structuredClone(f);
  const layers = meshLayers1d(copy);
  if (layers === null) {
    copy.mesh.grid1d = "uniform";
    copy.mesh.nr = 200;
  } else {
    copy.mesh.grid1d = "layers";
    copy.mesh.layerZoning = layers.map(() => ({ ...defaultLayerZoning(), cells: 40, spacing: "width" as const }));
  }
  copy.mesh.resolutionRequirement = { apply: "default", empirical: null };
  copy.mesh.recommendation = null;
  delete copy.deckImport;
  return copy;
}

/** The key the recommendation will be checked against (exported for the store). */
export function conditionsKeyForRecommendation(f: FormState): string {
  return recommendationConditionsKey(f);
}

// ---------------------------------------------------------------------------------------------
// Convergence pair

/** A copy of the form whose mesh carries half the cell mass at the laser-side surface (and
 *  everywhere else), the finer member of a convergence pair (tenryu-mesh-1d: "compare two
 *  otherwise identical decks with the surface cell mass halved"). The case name and the output
 *  directory get the suffix "_fine". */
export function refinedForConvergence(f: FormState): FormState {
  const g = structuredClone(f);
  delete g.deckImport;
  g.main.name = `${f.main.name}_fine`;
  if (g.output.directory.trim().length > 0) g.output.directory = `${g.output.directory.trim()}_fine`;
  const m = g.mesh;
  switch (m.grid1d) {
    case "uniform":
      m.nr = 2 * m.nr;
      break;
    case "graded":
      if (m.segmentSource === "manual") {
        m.segments = m.segments.map((s) => ({ ...s, nr: 2 * s.nr }));
      } else {
        m.nr = 2 * m.nr;
        m.regionNrOverrides = m.regionNrOverrides.map((n) => (n === null ? null : 2 * n));
      }
      break;
    case "layers":
      // Twice the cells; a geometric progression keeps its shape (sqrt of the ratio, half the
      // end cell); a matched layer keeps matching its (refined) neighbour.
      m.layerZoning = m.layerZoning.map((spec) => ({
        ...spec,
        cells: 2 * spec.cells,
        ratio: Math.sqrt(spec.ratio),
        firstWidthCm: spec.firstWidthCm / 2,
      }));
      break;
    case "explicit": {
      // Split every cell into two of equal mass (equal volume inside one cell).
      const nodes = m.explicitNodes.nodesCm;
      const out: number[] = [nodes[0]];
      const p = f.main.geometry1d === "spherical" ? 3 : f.main.geometry1d === "cylindrical" ? 2 : 1;
      for (let i = 1; i < nodes.length; i++) {
        const a = nodes[i - 1];
        const b = nodes[i];
        out.push(Math.pow(0.5 * (Math.pow(a, p) + Math.pow(b, p)), 1 / p), b);
      }
      m.explicitNodes = { nodesCm: out, source: `${m.explicitNodes.source} (each cell split in two)`.trim() };
      break;
    }
    case "recommended":
    case "zoning_intent": {
      // Twice the cells and half of every cell-measure bound; the recommender's empirical
      // ceiling stays (its integrity lint checks it against the conditions) and the halved bands
      // are the stricter constraint.
      const z = m.zoningIntent;
      m.zoningIntent = {
        ...z,
        nCells: 2 * z.nCells,
        bands: z.bands.map((b) => ({
          ...b,
          cellMeasureMin: b.cellMeasureMin === null ? null : b.cellMeasureMin / 2,
          cellMeasureMax: b.cellMeasureMax === null ? null : b.cellMeasureMax / 2,
        })),
        drMinCm: z.drMinCm === null ? null : z.drMinCm / 2,
        cellMeasureMin: z.cellMeasureMin === null ? null : z.cellMeasureMin / 2,
        cellMeasureMax: z.cellMeasureMax === null ? null : z.cellMeasureMax / 2,
        minCellsPerSegment: z.minCellsPerSegment === null ? null : 2 * z.minCellsPerSegment,
      };
      // The finer deck is a hand-made variant of the recommendation, not a recommendation.
      if (m.grid1d === "recommended") {
        m.grid1d = "zoning_intent";
        if (m.recommendation !== null) m.recommendation = { ...m.recommendation, edited: true };
      }
      break;
    }
  }
  return g;
}
