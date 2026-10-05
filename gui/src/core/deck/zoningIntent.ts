// The solver's 1D zoning of a Mesh.zoning_intent (src/core/zoning_intent.cpp,
// compute_zoning_intent_nodes; SPECIFICATION 6.4.2) in TypeScript: the same panel quadrature,
// monitor equidistribution, integer allocation over the pin-delimited segments, cell-measure
// projection, band coverage reconciliation and post-checks, in the same order and with the same
// constants, so Studio draws the mesh the solver builds from a deck's zoning_intent without running
// the solver. Math.exp/log/cos may differ from the C library in the last bit, so nodes agree to
// round-off, not bitwise. Keep this file in step with the C++: test/zoningIntent.test.ts compares
// it with the C++ on fixed intents (scripts/zoningIntentGolden.cpp) and with the solver's mesh
// preview of the laser presets and of edited forms, and holds a digest of the C++ source.
import type { ZoningMeasure } from "./mesh1d";

const kPi = 3.141592653589793;
const kPositionToleranceScale = 1.0e-12;
const kMuFloor = 1.0e-300;
const kIntegralTolerance = 1.0e-12;
const kBinTolerance = 1.0e-10;
const kCapacityRoundoff = 1.0e-12;
const kEnvelopeLogGuard = 100.0;
const kProjectionCheckTolerance = 1.0e-12;
const kFeasibilityWindowTolerance = 1.0e-12;
const kWidthCheckTolerance = 1.0e-10;
const kRatioCheckTolerance = 1.0e-6;
const kBandOverlapTolerance = 1.0e-9;
const kClosureCheckTolerance = 1.0e-8;
const kMaxAnchorLogAmplitude = 9.210340371976184; // ln(1e4)
const kMaxAnchorLogAggregate = 13.815510557964274; // ln(1e6)
const kInitialQuadratureBins = 8;
const kMaximumQuadratureBins = 16384;
const kNewtonIterations = 3;
const kBisectionIterations = 200;
const kIntMax = 2147483647;

export interface ZoningPin {
  r: number;
  /** true: the hard ratio cap is not applied across the two cells next to this pin. */
  ratioJumpAllowed: boolean;
}

export interface ZoningProfilePoint {
  r: number;
  w: number;
}

export interface ZoningAnchor {
  r: number;
  halfWidth: number;
  logAmplitude: number;
}

export interface ZoningBand {
  measureFracBegin: number;
  measureFracEnd: number;
  /** 0 disables. */
  cellMeasureMin: number;
  /** 0 disables. */
  cellMeasureMax: number;
}

/** ZoningIntentConfig of zoning_intent.hpp (the C++ defaults are the namelist's). */
export interface ZoningIntentConfig {
  nCells: number;
  measure: ZoningMeasure;
  pins: ZoningPin[];
  profile: ZoningProfilePoint[];
  anchors: ZoningAnchor[];
  bands: ZoningBand[];
  extraEvents: number[];
  drMin: number;
  cellMeasureMin: number;
  cellMeasureMax: number;
  preferredRatio: number;
  ratioHardMax: number;
  minCellsPerSegment: number;
}

export type ZoningStatus = "ok" | "invalid_input" | "infeasible" | "numerical_failure";

export interface ZoningDiagnostics {
  status: ZoningStatus;
  /** Stable machine-readable code of a failure, e.g. "MESH_PIN_OUT_OF_DOMAIN". */
  code: string;
  message: string;
  ratioMaxAchieved: number;
  ratioMeanAchieved: number;
  nRatioSoftExceed: number;
  widthMinAchieved: number;
  cellMeasureMinAchieved: number;
  cellMeasureMaxAchieved: number;
  bandCellMeasureMaxAchieved: number[];
  bandCellMeasureMinAchieved: number[];
  quadratureRelResidual: number;
  cellsPerSegment: number[];
  warnings: string[];
}

export interface ZoningResult {
  ok: boolean;
  /** nCells + 1 node radii [cm] when ok. */
  nodes: number[];
  diag: ZoningDiagnostics;
}

// ---------------------------------------------------------------------------------------------
// C++ library semantics

/** std::max (returns the first argument unless it is less than the second). */
function cmax(a: number, b: number): number {
  return a < b ? b : a;
}

/** std::min (returns the first argument unless the second is less). */
function cmin(a: number, b: number): number {
  return b < a ? b : a;
}

/** std::clamp. */
function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : hi < v ? hi : v;
}

const f64 = new Float64Array(1);
const u64 = new BigUint64Array(f64.buffer);

function bitsOf(x: number): bigint {
  f64[0] = x;
  return u64[0];
}

function bitwiseEqual(a: number, b: number): boolean {
  return bitsOf(a) === bitsOf(b);
}

/** std::nextafter. */
function nextAfter(x: number, toward: number): number {
  if (Number.isNaN(x) || Number.isNaN(toward)) return NaN;
  if (x === toward) return toward;
  if (x === 0) return toward > 0 ? Number.MIN_VALUE : -Number.MIN_VALUE;
  f64[0] = x;
  if (toward > x === x > 0) u64[0] += 1n;
  else u64[0] -= 1n;
  return f64[0];
}

/** std::upper_bound over a[lo, hi): first index whose element is greater than value. */
function upperBound(a: number[], value: number, lo = 0, hi = a.length): number {
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (value < a[mid]) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

/** std::lower_bound over a[lo, hi): first index whose element is not less than value. */
function lowerBound(a: number[], value: number, lo = 0, hi = a.length): number {
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (a[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** std::accumulate from 0.0, in order. */
function accumulate(values: number[]): number {
  let sum = 0.0;
  for (const v of values) sum += v;
  return sum;
}

/** The solver's format_double: printf %.17g. */
export function formatDouble(value: number): string {
  if (Number.isNaN(value)) return "nan";
  if (value === Infinity) return "inf";
  if (value === -Infinity) return "-inf";
  if (value === 0) return Object.is(value, -0) ? "-0" : "0";
  const [mantissa, exponentText] = value.toExponential(16).split("e");
  const exponent = Number(exponentText);
  if (exponent < -4 || exponent >= 17) {
    const trimmed = mantissa.includes(".") ? mantissa.replace(/\.?0+$/, "") : mantissa;
    const magnitude = Math.abs(exponent);
    return `${trimmed}e${exponent < 0 ? "-" : "+"}${magnitude < 10 ? "0" : ""}${magnitude}`;
  }
  const fixed = value.toFixed(16 - exponent);
  return fixed.includes(".") ? fixed.replace(/\.?0+$/, "") : fixed;
}

// ---------------------------------------------------------------------------------------------
// Quadrature tables

interface SamplePair {
  measure: number;
  monitor: number;
}

interface IntegralTable {
  values: number[];
  midValues: number[];
  cumulative: number[];
  total: number;
  maxSimpsonTrapezoidDifference: number;
}

interface PanelTable {
  rBegin: number;
  rEnd: number;
  nBins: number;
  coordinates: number[];
  measure: IntegralTable;
  monitor: IntegralTable;
}

interface SegmentData {
  rBegin: number;
  rEnd: number;
  panels: PanelTable[];
  measureTotal: number;
  monitorTotal: number;
  nMin: number;
  nMax: number;
  nCells: number;
}

type IntegrandKind = "measure" | "monitor";

function emptyTable(): IntegralTable {
  return { values: [], midValues: [], cumulative: [], total: 0.0, maxSimpsonTrapezoidDifference: 0.0 };
}

function emptySegment(rBegin: number, rEnd: number): SegmentData {
  return { rBegin, rEnd, panels: [], measureTotal: 0.0, monitorTotal: 0.0, nMin: 0, nMax: 0, nCells: 0 };
}

class IntegrandEvaluator {
  densityInvalid = false;
  invalidCoordinate = 0.0;
  invalidDensity = 0.0;
  private readonly profileR: number[];
  private readonly profileLogW: number[];

  constructor(
    readonly measure: ZoningMeasure,
    readonly profile: ZoningProfilePoint[],
    readonly anchors: ZoningAnchor[],
    readonly rho0: ((r: number) => number) | null,
  ) {
    this.profileR = profile.map((point) => point.r);
    this.profileLogW = profile.map((point) => Math.log(point.w));
  }

  preferredWeight(r: number): number {
    let logWeight = 0.0;
    const n = this.profile.length;
    if (n > 0) {
      if (r <= this.profileR[0]) {
        logWeight = this.profileLogW[0];
      } else if (r >= this.profileR[n - 1]) {
        logWeight = this.profileLogW[n - 1];
      } else {
        const upper = upperBound(this.profileR, r);
        const lower = upper - 1;
        const fraction = (r - this.profileR[lower]) / (this.profileR[upper] - this.profileR[lower]);
        logWeight = this.profileLogW[lower] + fraction * (this.profileLogW[upper] - this.profileLogW[lower]);
      }
    }
    for (const anchor of this.anchors) {
      const u = (r - anchor.r) / anchor.halfWidth;
      if (Math.abs(u) < 1.0) {
        logWeight += anchor.logAmplitude * 0.5 * (1.0 + Math.cos(kPi * u));
      }
    }
    return Math.exp(logWeight);
  }

  measureDensity(r: number): number {
    if (this.measure === "width") return 1.0;
    const density = (this.rho0 as (r: number) => number)(r);
    if (!Number.isFinite(density) || density < 0.0) {
      if (!this.densityInvalid) {
        this.densityInvalid = true;
        this.invalidCoordinate = r;
        this.invalidDensity = density;
      }
      return 0.0;
    }
    switch (this.measure) {
      case "areal_mass":
        return density;
      case "cylindrical_line_mass":
        return density * 2.0 * kPi * r;
      case "spherical_cell_mass":
        return density * 4.0 * kPi * r * r;
    }
    return 0.0;
  }

  sample(r: number): SamplePair {
    const measure = this.measureDensity(r);
    return { measure, monitor: measure / this.preferredWeight(r) };
  }
}

function defaultDiagnostics(): ZoningDiagnostics {
  return {
    status: "ok",
    code: "",
    message: "",
    ratioMaxAchieved: 1.0,
    ratioMeanAchieved: 1.0,
    nRatioSoftExceed: 0,
    widthMinAchieved: 0.0,
    cellMeasureMinAchieved: 0.0,
    cellMeasureMaxAchieved: 0.0,
    bandCellMeasureMaxAchieved: [],
    bandCellMeasureMinAchieved: [],
    quadratureRelResidual: 0.0,
    cellsPerSegment: [],
    warnings: [],
  };
}

function failureResult(status: ZoningStatus, code: string, message: string, warnings: string[] = []): ZoningResult {
  const diag = defaultDiagnostics();
  diag.status = status;
  diag.code = code;
  diag.message = message;
  diag.warnings = [...warnings];
  return { ok: false, nodes: [], diag };
}

function densityFailure(evaluator: IntegrandEvaluator, warnings: string[]): ZoningResult {
  return failureResult(
    "invalid_input",
    "MESH_DENSITY_INVALID",
    "rho0 sample at r=" +
      formatDouble(evaluator.invalidCoordinate) +
      " is non-finite or negative: " +
      formatDouble(evaluator.invalidDensity),
    warnings,
  );
}

function integrateSamples(coordinates: number[], values: number[], midValues: number[]): IntegralTable {
  const cumulative = new Array<number>(coordinates.length).fill(0.0);
  let maxDifference = 0.0;
  for (let i = 0; i + 1 < coordinates.length; i++) {
    const h = coordinates[i + 1] - coordinates[i];
    const simpson = (h / 6.0) * (values[i] + 4.0 * midValues[i] + values[i + 1]);
    const trapezoid = (h / 2.0) * (values[i] + values[i + 1]);
    cumulative[i + 1] = cumulative[i] + simpson;
    maxDifference = cmax(maxDifference, Math.abs(simpson - trapezoid));
  }
  return {
    values,
    midValues,
    cumulative,
    total: cumulative[cumulative.length - 1],
    maxSimpsonTrapezoidDifference: maxDifference,
  };
}

function buildPanel(rBegin: number, rEnd: number, nBins: number, evaluator: IntegrandEvaluator): PanelTable {
  const coordinates = new Array<number>(nBins + 1);
  const measureValues = new Array<number>(nBins + 1);
  const monitorValues = new Array<number>(nBins + 1);
  const measureMidValues = new Array<number>(nBins);
  const monitorMidValues = new Array<number>(nBins);
  const h = (rEnd - rBegin) / nBins;
  for (let k = 0; k <= nBins; k++) {
    let coordinate = rBegin + k * h;
    if (k === 0) coordinate = rBegin;
    else if (k === nBins) coordinate = rEnd;
    coordinates[k] = coordinate;
    let sampleCoordinate = coordinate;
    if (k === 0) sampleCoordinate = nextAfter(rBegin, rEnd);
    else if (k === nBins) sampleCoordinate = nextAfter(rEnd, rBegin);
    const pair = evaluator.sample(sampleCoordinate);
    measureValues[k] = pair.measure;
    monitorValues[k] = pair.monitor;
  }
  for (let k = 0; k < nBins; k++) {
    const midpoint = (coordinates[k] + coordinates[k + 1]) / 2.0;
    const pair = evaluator.sample(midpoint);
    measureMidValues[k] = pair.measure;
    monitorMidValues[k] = pair.monitor;
  }
  return {
    rBegin,
    rEnd,
    nBins,
    coordinates,
    measure: integrateSamples(coordinates, measureValues, measureMidValues),
    monitor: integrateSamples(coordinates, monitorValues, monitorMidValues),
  };
}

function integralConverged(current: IntegralTable, previous: IntegralTable): boolean {
  const totalConverged =
    Math.abs(current.total - previous.total) <= kIntegralTolerance * cmax(Math.abs(current.total), kMuFloor);
  const binsConverged = current.maxSimpsonTrapezoidDifference <= kBinTolerance * cmax(current.total, kMuFloor);
  return totalConverged && binsConverged;
}

/** build_refined_panel: the converged panel, or null (quadrature not converged or invalid density). */
function buildRefinedPanel(rBegin: number, rEnd: number, evaluator: IntegrandEvaluator): PanelTable | null {
  let previous = buildPanel(rBegin, rEnd, kInitialQuadratureBins / 2, evaluator);
  if (evaluator.densityInvalid) return null;
  for (let nBins = kInitialQuadratureBins; nBins <= kMaximumQuadratureBins; nBins *= 2) {
    const current = buildPanel(rBegin, rEnd, nBins, evaluator);
    if (evaluator.densityInvalid) return null;
    if (integralConverged(current.measure, previous.measure) && integralConverged(current.monitor, previous.monitor)) {
      return current;
    }
    if (nBins === kMaximumQuadratureBins) return null;
    previous = current;
  }
  return null;
}

function quadraticValue(table: IntegralTable, bin: number, fraction: number): number {
  const f0 = table.values[bin];
  const fm = table.midValues[bin];
  const f1 = table.values[bin + 1];
  const coefficientA = 2.0 * (f1 + f0 - 2.0 * fm);
  const coefficientB = 4.0 * fm - 3.0 * f0 - f1;
  return (coefficientA * fraction + coefficientB) * fraction + f0;
}

function quadraticIntegral(table: IntegralTable, bin: number, fraction: number, width: number): number {
  const f0 = table.values[bin];
  const fm = table.midValues[bin];
  const f1 = table.values[bin + 1];
  const coefficientA = 2.0 * (f1 + f0 - 2.0 * fm);
  const coefficientB = 4.0 * fm - 3.0 * f0 - f1;
  return (
    width *
    (f0 * fraction + 0.5 * coefficientB * fraction * fraction + (coefficientA / 3.0) * fraction * fraction * fraction)
  );
}

function selectTable(panel: PanelTable, kind: IntegrandKind): IntegralTable {
  return kind === "measure" ? panel.measure : panel.monitor;
}

function segmentTotal(segment: SegmentData, kind: IntegrandKind): number {
  return kind === "measure" ? segment.measureTotal : segment.monitorTotal;
}

function evaluateCumulative(segment: SegmentData, r: number, kind: IntegrandKind): number {
  if (r <= segment.rBegin) return 0.0;
  if (r >= segment.rEnd) return segmentTotal(segment, kind);
  let offset = 0.0;
  for (const panel of segment.panels) {
    const table = selectTable(panel, kind);
    if (r >= panel.rEnd) {
      offset += table.total;
      continue;
    }
    if (r <= panel.rBegin) return offset;
    const bin = upperBound(panel.coordinates, r) - 1;
    const binBegin = panel.coordinates[bin];
    const width = panel.coordinates[bin + 1] - binBegin;
    const fraction = (r - binBegin) / width;
    return offset + table.cumulative[bin] + quadraticIntegral(table, bin, fraction, width);
  }
  return segmentTotal(segment, kind);
}

function invertCumulative(segment: SegmentData, target: number, kind: IntegrandKind): number {
  if (target <= 0.0) return segment.rBegin;
  if (target >= segmentTotal(segment, kind)) return segment.rEnd;
  const panelCumulative: number[] = [];
  let runningTotal = 0.0;
  for (const panel of segment.panels) {
    runningTotal += selectTable(panel, kind).total;
    panelCumulative.push(runningTotal);
  }
  const panelIndex = lowerBound(panelCumulative, target);
  const panel = segment.panels[panelIndex];
  const table = selectTable(panel, kind);
  const offset = panelIndex === 0 ? 0.0 : panelCumulative[panelIndex - 1];
  const localTarget = target - offset;
  const bin = lowerBound(table.cumulative, localTarget, 1) - 1;
  const binIntegral = table.cumulative[bin + 1] - table.cumulative[bin];
  const linearFraction = clamp((localTarget - table.cumulative[bin]) / binIntegral, 0.0, 1.0);
  const binBegin = panel.coordinates[bin];
  const binEnd = panel.coordinates[bin + 1];
  const width = binEnd - binBegin;
  const linearEstimate = binBegin + linearFraction * width;
  let estimate = linearEstimate;
  for (let iteration = 0; iteration < kNewtonIterations; iteration++) {
    const fraction = (estimate - binBegin) / width;
    const derivative = cmax(quadraticValue(table, bin, fraction), 0.0);
    if (derivative < kMuFloor) return linearEstimate;
    const residual = table.cumulative[bin] + quadraticIntegral(table, bin, fraction, width) - localTarget;
    estimate = clamp(estimate - residual / derivative, binBegin, binEnd);
  }
  return estimate;
}

type ProjectionOutcome =
  | { kind: "ok"; projected: number[] }
  | { kind: "infeasible"; message: string }
  | { kind: "numerical_failure"; message: string };

function projectCellMeasures(
  segment: SegmentData,
  cfg: ZoningIntentConfig,
  loBounds: number[],
  hiBounds: number[],
  preferred: number[],
): ProjectionOutcome {
  for (let i = 0; i < preferred.length; i++) {
    if (!(preferred[i] > 0.0) || !Number.isFinite(preferred[i])) {
      return {
        kind: "numerical_failure",
        message: "cell " + String(i) + " has non-positive measure " + formatDouble(preferred[i]),
      };
    }
  }
  const n = preferred.length;
  const totalMeasure = segment.measureTotal;
  const scale = totalMeasure / n;
  const ratioLimit = Math.log(cfg.ratioHardMax);
  const x0 = new Array<number>(n).fill(0.0);
  const envelopeLo = new Array<number>(n).fill(0.0);
  const envelopeHi = new Array<number>(n).fill(0.0);
  for (let i = 0; i < n; i++) {
    x0[i] = Math.log(preferred[i] / scale);
    envelopeLo[i] = loBounds[i] > 0.0 ? Math.log(loBounds[i] / scale) : -kEnvelopeLogGuard;
    envelopeHi[i] = hiBounds[i] > 0.0 ? Math.log(hiBounds[i] / scale) : kEnvelopeLogGuard;
  }
  for (let i = 1; i < n; i++) {
    envelopeLo[i] = cmax(envelopeLo[i], envelopeLo[i - 1] - ratioLimit);
    envelopeHi[i] = cmin(envelopeHi[i], envelopeHi[i - 1] + ratioLimit);
  }
  for (let i = n - 2; i >= 0; i--) {
    envelopeLo[i] = cmax(envelopeLo[i], envelopeLo[i + 1] - ratioLimit);
    envelopeHi[i] = cmin(envelopeHi[i], envelopeHi[i + 1] + ratioLimit);
  }
  for (let i = 0; i < n; i++) {
    if (envelopeLo[i] > envelopeHi[i]) {
      return {
        kind: "infeasible",
        message: "cell " + String(i) + ": the ratio chain and the cell-measure bounds admit no value (envelope empty)",
      };
    }
  }
  let minimumSum = 0.0;
  let maximumSum = 0.0;
  for (let i = 0; i < n; i++) {
    minimumSum += Math.exp(envelopeLo[i]);
    maximumSum += Math.exp(envelopeHi[i]);
  }
  minimumSum *= scale;
  maximumSum *= scale;
  if (
    totalMeasure < minimumSum * (1.0 - kFeasibilityWindowTolerance) ||
    totalMeasure > maximumSum * (1.0 + kFeasibilityWindowTolerance)
  ) {
    return {
      kind: "infeasible",
      message:
        "total measure " +
        formatDouble(totalMeasure) +
        " lies outside the feasible window [" +
        formatDouble(minimumSum) +
        ", " +
        formatDouble(maximumSum) +
        "] of the ratio chain and cell-measure bounds",
    };
  }

  const x = new Array<number>(n).fill(0.0);
  const shiftedSum = (sigma: number): number => {
    let sum = 0.0;
    for (let i = 0; i < n; i++) {
      let intervalLo = envelopeLo[i];
      let intervalHi = envelopeHi[i];
      if (i > 0) {
        intervalLo = cmax(intervalLo, x[i - 1] - ratioLimit);
        intervalHi = cmin(intervalHi, x[i - 1] + ratioLimit);
      }
      x[i] = clamp(x0[i] + sigma, intervalLo, intervalHi);
      sum += Math.exp(x[i]);
    }
    return sum * scale;
  };
  let sigmaLo = envelopeLo[0] - x0[0];
  let sigmaHi = envelopeHi[0] - x0[0];
  for (let i = 1; i < n; i++) {
    sigmaLo = cmin(sigmaLo, envelopeLo[i] - x0[i]);
    sigmaHi = cmax(sigmaHi, envelopeHi[i] - x0[i]);
  }
  for (let iteration = 0; iteration < kBisectionIterations; iteration++) {
    const sigmaMid = 0.5 * (sigmaLo + sigmaHi);
    if (shiftedSum(sigmaMid) < totalMeasure) sigmaLo = sigmaMid;
    else sigmaHi = sigmaMid;
  }
  shiftedSum(0.5 * (sigmaLo + sigmaHi));
  const projected = new Array<number>(n);
  for (let i = 0; i < n; i++) projected[i] = Math.exp(x[i]) * scale;

  let worstViolation = 0.0;
  let worstDetail = "none";
  for (let i = 0; i + 1 < n; i++) {
    const ratio = cmax(projected[i], projected[i + 1]) / cmin(projected[i], projected[i + 1]);
    const violation = ratio / cfg.ratioHardMax - 1.0;
    if (!Number.isFinite(ratio) || violation > worstViolation) {
      worstViolation = Number.isFinite(violation) ? violation : Infinity;
      worstDetail = "adjacent ratio at edge " + String(i + 1) + " is " + formatDouble(ratio);
    }
  }
  for (let i = 0; i < n; i++) {
    if (loBounds[i] > 0.0) {
      const violation = loBounds[i] / projected[i] - 1.0;
      if (!Number.isFinite(projected[i]) || violation > worstViolation) {
        worstViolation = Number.isFinite(violation) ? violation : Infinity;
        worstDetail = "cell-measure lower-bound value in cell " + String(i) + " is " + formatDouble(projected[i]);
      }
    }
  }
  for (let i = 0; i < n; i++) {
    if (hiBounds[i] > 0.0) {
      const violation = projected[i] / hiBounds[i] - 1.0;
      if (!Number.isFinite(projected[i]) || violation > worstViolation) {
        worstViolation = Number.isFinite(violation) ? violation : Infinity;
        worstDetail = "cell-measure upper-bound value in cell " + String(i) + " is " + formatDouble(projected[i]);
      }
    }
  }
  const projectedSum = accumulate(projected);
  const sumViolation = Math.abs(projectedSum - segment.measureTotal) / segment.measureTotal;
  if (!Number.isFinite(sumViolation) || sumViolation > worstViolation) {
    worstViolation = Number.isFinite(sumViolation) ? sumViolation : Infinity;
    worstDetail = "relative measure-sum residual is " + formatDouble(sumViolation);
  }
  if (worstViolation > kProjectionCheckTolerance) {
    return {
      kind: "numerical_failure",
      message: "projection constraint verification failed; worst violation: " + worstDetail,
    };
  }
  return { kind: "ok", projected };
}

function buildVerificationPanel(source: PanelTable, evaluator: IntegrandEvaluator): PanelTable {
  const nBins = 2 * source.nBins;
  const coordinates = new Array<number>(nBins + 1);
  const values = new Array<number>(nBins + 1);
  const midValues = new Array<number>(nBins);
  const h = (source.rEnd - source.rBegin) / nBins;
  for (let k = 0; k <= nBins; k++) {
    let coordinate = source.rBegin + k * h;
    if (k === 0) coordinate = source.rBegin;
    else if (k === nBins) coordinate = source.rEnd;
    coordinates[k] = coordinate;
    let sampleCoordinate = coordinate;
    if (k === 0) sampleCoordinate = nextAfter(source.rBegin, source.rEnd);
    else if (k === nBins) sampleCoordinate = nextAfter(source.rEnd, source.rBegin);
    values[k] = evaluator.measureDensity(sampleCoordinate);
  }
  for (let k = 0; k < nBins; k++) {
    const midpoint = (coordinates[k] + coordinates[k + 1]) / 2.0;
    midValues[k] = evaluator.measureDensity(midpoint);
  }
  return {
    rBegin: source.rBegin,
    rEnd: source.rEnd,
    nBins,
    coordinates,
    measure: integrateSamples(coordinates, values, midValues),
    monitor: emptyTable(),
  };
}

// ---------------------------------------------------------------------------------------------
// compute_zoning_intent_nodes

/** Node radii of a zoning intent on [rMin, rMax] (compute_zoning_intent_nodes). rho0 [g/cc] is
 *  required for the three mass measures and must be smooth inside every quadrature panel. */
export function computeZoningIntentNodes(
  rMin: number,
  rMax: number,
  cfg: ZoningIntentConfig,
  rho0: ((r: number) => number) | null,
): ZoningResult {
  if (!Number.isFinite(rMin) || !Number.isFinite(rMax) || !(rMax > rMin)) {
    return failureResult(
      "invalid_input",
      "MESH_DOMAIN_EMPTY",
      "domain requires finite r_min < r_max; got r_min=" + formatDouble(rMin) + ", r_max=" + formatDouble(rMax),
    );
  }
  const domainLength = rMax - rMin;
  const positionTolerance = kPositionToleranceScale * domainLength;
  if (cfg.nCells < 1) {
    return failureResult("invalid_input", "MESH_BUDGET_NONPOSITIVE", "n_cells must be >= 1; got " + String(cfg.nCells));
  }
  if (cfg.minCellsPerSegment < 1) {
    return failureResult(
      "invalid_input",
      "MESH_MIN_CELLS_NONPOSITIVE",
      "min_cells_per_segment must be >= 1; got " + String(cfg.minCellsPerSegment),
    );
  }
  if (!Number.isFinite(cfg.drMin) || cfg.drMin < 0.0) {
    return failureResult("invalid_input", "MESH_DR_MIN_NEGATIVE", "dr_min must be finite and >= 0; got " + formatDouble(cfg.drMin));
  }
  if (
    !Number.isFinite(cfg.cellMeasureMin) ||
    cfg.cellMeasureMin < 0.0 ||
    !Number.isFinite(cfg.cellMeasureMax) ||
    cfg.cellMeasureMax < 0.0 ||
    (cfg.cellMeasureMin > 0.0 && cfg.cellMeasureMax > 0.0 && cfg.cellMeasureMin > cfg.cellMeasureMax)
  ) {
    return failureResult(
      "invalid_input",
      "MESH_CELL_MEASURE_BOX_INVALID",
      "cell_measure_min and cell_measure_max must be finite and >= 0, and " +
        "cell_measure_min must not exceed cell_measure_max when both are enabled; " +
        "got cell_measure_min=" +
        formatDouble(cfg.cellMeasureMin) +
        ", cell_measure_max=" +
        formatDouble(cfg.cellMeasureMax),
    );
  }
  if (!Number.isFinite(cfg.ratioHardMax) || !(cfg.ratioHardMax > 1.0) || cfg.ratioHardMax > 2.0) {
    return failureResult(
      "invalid_input",
      "MESH_RATIO_CAP_OUT_OF_RANGE",
      "ratio_hard_max must be in (1.0, 2.0]; 2.0 is the immutable solver policy ceiling; got " +
        formatDouble(cfg.ratioHardMax),
    );
  }

  for (let i = 0; i < cfg.pins.length; i++) {
    const r = cfg.pins[i].r;
    if (!Number.isFinite(r) || !(r > rMin + positionTolerance) || !(r < rMax - positionTolerance)) {
      return failureResult(
        "invalid_input",
        "MESH_PIN_OUT_OF_DOMAIN",
        "pin " +
          String(i) +
          " at r=" +
          formatDouble(r) +
          " must lie strictly inside (" +
          formatDouble(rMin + positionTolerance) +
          ", " +
          formatDouble(rMax - positionTolerance) +
          ")",
      );
    }
  }
  for (let i = 1; i < cfg.pins.length; i++) {
    if (!(cfg.pins[i].r > cfg.pins[i - 1].r)) {
      return failureResult(
        "invalid_input",
        "MESH_PIN_NOT_SORTED",
        "pins must be strictly increasing; pin " +
          String(i - 1) +
          " is " +
          formatDouble(cfg.pins[i - 1].r) +
          ", pin " +
          String(i) +
          " is " +
          formatDouble(cfg.pins[i].r),
      );
    }
  }
  for (let i = 1; i < cfg.pins.length; i++) {
    const separation = cfg.pins[i].r - cfg.pins[i - 1].r;
    if (separation < positionTolerance) {
      return failureResult(
        "invalid_input",
        "MESH_PIN_DUPLICATE",
        "adjacent pins " +
          String(i - 1) +
          " and " +
          String(i) +
          " are separated by " +
          formatDouble(separation) +
          ", below tol_r=" +
          formatDouble(positionTolerance),
      );
    }
  }

  for (let i = 0; i < cfg.profile.length; i++) {
    const r = cfg.profile[i].r;
    if (!Number.isFinite(r) || r < rMin || r > rMax) {
      return failureResult(
        "invalid_input",
        "MESH_PROFILE_OUT_OF_DOMAIN",
        "profile point " + String(i) + " at r=" + formatDouble(r) + " must lie in [" + formatDouble(rMin) + ", " + formatDouble(rMax) + "]",
      );
    }
  }
  for (let i = 1; i < cfg.profile.length; i++) {
    if (!(cfg.profile[i].r > cfg.profile[i - 1].r)) {
      return failureResult(
        "invalid_input",
        "MESH_PROFILE_COORD_NOT_SORTED",
        "profile coordinates must be strictly increasing; point " +
          String(i - 1) +
          " is " +
          formatDouble(cfg.profile[i - 1].r) +
          ", point " +
          String(i) +
          " is " +
          formatDouble(cfg.profile[i].r),
      );
    }
  }
  for (let i = 1; i < cfg.profile.length; i++) {
    const separation = cfg.profile[i].r - cfg.profile[i - 1].r;
    if (separation <= positionTolerance) {
      return failureResult(
        "invalid_input",
        "MESH_PROFILE_COORD_DUPLICATE",
        "profile coordinates " +
          String(i - 1) +
          " and " +
          String(i) +
          " are separated by " +
          formatDouble(separation) +
          ", within tol_r=" +
          formatDouble(positionTolerance),
      );
    }
  }
  for (let i = 0; i < cfg.profile.length; i++) {
    if (!Number.isFinite(cfg.profile[i].w) || !(cfg.profile[i].w > 0.0)) {
      return failureResult(
        "invalid_input",
        "MESH_PROFILE_VALUE_NONPOSITIVE",
        "profile point " + String(i) + " has w=" + formatDouble(cfg.profile[i].w) + "; w must be finite and > 0",
      );
    }
  }
  for (let i = 0; i < cfg.anchors.length; i++) {
    const anchor = cfg.anchors[i];
    if (!Number.isFinite(anchor.r) || anchor.r < rMin || anchor.r > rMax) {
      return failureResult(
        "invalid_input",
        "MESH_ANCHOR_OUT_OF_DOMAIN",
        "anchor " + String(i) + " at r=" + formatDouble(anchor.r) + " must lie in [" + formatDouble(rMin) + ", " + formatDouble(rMax) + "]",
      );
    }
    if (!Number.isFinite(anchor.halfWidth) || !(anchor.halfWidth > 0.0)) {
      return failureResult(
        "invalid_input",
        "MESH_ANCHOR_WIDTH_NONPOSITIVE",
        "anchor " + String(i) + " has half_width=" + formatDouble(anchor.halfWidth) + "; half_width must be finite and > 0",
      );
    }
    if (!Number.isFinite(anchor.logAmplitude) || Math.abs(anchor.logAmplitude) > kMaxAnchorLogAmplitude) {
      return failureResult(
        "invalid_input",
        "MESH_ANCHOR_AMPLITUDE_INVALID",
        "anchor " +
          String(i) +
          " has log_amplitude=" +
          formatDouble(anchor.logAmplitude) +
          "; |log_amplitude| must be <= " +
          formatDouble(kMaxAnchorLogAmplitude),
      );
    }
  }
  if (cfg.anchors.length > 0) {
    const anchorCheckPoints: number[] = [];
    for (const anchor of cfg.anchors) {
      anchorCheckPoints.push(clamp(anchor.r - anchor.halfWidth, rMin, rMax));
      anchorCheckPoints.push(anchor.r);
      anchorCheckPoints.push(clamp(anchor.r + anchor.halfWidth, rMin, rMax));
    }
    let worstCoordinate = anchorCheckPoints[0];
    let worstAbsoluteSum = 0.0;
    for (const checkPoint of anchorCheckPoints) {
      let sum = 0.0;
      for (const anchor of cfg.anchors) {
        const u = (checkPoint - anchor.r) / anchor.halfWidth;
        if (Math.abs(u) < 1.0) sum += anchor.logAmplitude * 0.5 * (1.0 + Math.cos(kPi * u));
      }
      const absoluteSum = Math.abs(sum);
      if (absoluteSum > worstAbsoluteSum) {
        worstAbsoluteSum = absoluteSum;
        worstCoordinate = checkPoint;
      }
    }
    if (worstAbsoluteSum > kMaxAnchorLogAggregate) {
      return failureResult(
        "invalid_input",
        "MESH_ANCHOR_AGGREGATE_EXCESSIVE",
        "anchor aggregate at r=" +
          formatDouble(worstCoordinate) +
          " has |S|=" +
          formatDouble(worstAbsoluteSum) +
          ", above the limit " +
          formatDouble(kMaxAnchorLogAggregate),
      );
    }
  }
  for (let i = 0; i < cfg.bands.length; i++) {
    const band = cfg.bands[i];
    if (
      !Number.isFinite(band.measureFracBegin) ||
      !Number.isFinite(band.measureFracEnd) ||
      !Number.isFinite(band.cellMeasureMin) ||
      !Number.isFinite(band.cellMeasureMax) ||
      band.measureFracBegin < 0.0 ||
      !(band.measureFracBegin < band.measureFracEnd) ||
      band.measureFracEnd > 1.0
    ) {
      return failureResult(
        "invalid_input",
        "MESH_BAND_RANGE_INVALID",
        "band " +
          String(i) +
          " has measure_frac_begin=" +
          formatDouble(band.measureFracBegin) +
          ", measure_frac_end=" +
          formatDouble(band.measureFracEnd),
      );
    }
    if (
      band.cellMeasureMin < 0.0 ||
      band.cellMeasureMax < 0.0 ||
      (band.cellMeasureMin > 0.0 && band.cellMeasureMax > 0.0 && band.cellMeasureMin > band.cellMeasureMax)
    ) {
      return failureResult(
        "invalid_input",
        "MESH_BAND_BOUNDS_INVALID",
        "band " +
          String(i) +
          " has cell_measure_min=" +
          formatDouble(band.cellMeasureMin) +
          ", cell_measure_max=" +
          formatDouble(band.cellMeasureMax),
      );
    }
  }
  if (cfg.measure !== "width" && rho0 === null) {
    return failureResult("invalid_input", "MESH_MEASURE_NEEDS_DENSITY", "the selected mass measure requires a rho0(r) function");
  }

  let loMeasure = cfg.cellMeasureMin;
  const hiMeasure = cfg.cellMeasureMax;
  if (cfg.measure === "width" && cfg.drMin > 0.0) loMeasure = cmax(loMeasure, cfg.drMin);

  const warnings: string[] = [];
  const segmentEdges: number[] = [rMin, ...cfg.pins.map((pin) => pin.r), rMax];
  const usableEvents: number[] = [];
  for (const event of cfg.extraEvents) {
    if (!Number.isFinite(event) || event < rMin || event > rMax) {
      warnings.push("extra_event outside domain ignored: " + formatDouble(event));
    } else if (event > rMin && event < rMax) {
      usableEvents.push(event);
    }
  }

  const evaluator = new IntegrandEvaluator(cfg.measure, cfg.profile, cfg.anchors, rho0);
  const segments: SegmentData[] = [];
  for (let segmentIndex = 0; segmentIndex + 1 < segmentEdges.length; segmentIndex++) {
    const segment = emptySegment(segmentEdges[segmentIndex], segmentEdges[segmentIndex + 1]);
    const panelEdges: number[] = [segment.rBegin, segment.rEnd];
    for (const point of cfg.profile) {
      if (point.r > segment.rBegin && point.r < segment.rEnd) panelEdges.push(point.r);
    }
    for (const anchor of cfg.anchors) {
      const supportBegin = anchor.r - anchor.halfWidth;
      if (supportBegin > segment.rBegin && supportBegin < segment.rEnd) panelEdges.push(supportBegin);
      if (anchor.r > segment.rBegin && anchor.r < segment.rEnd) panelEdges.push(anchor.r);
      const supportEnd = anchor.r + anchor.halfWidth;
      if (supportEnd > segment.rBegin && supportEnd < segment.rEnd) panelEdges.push(supportEnd);
    }
    for (const event of usableEvents) {
      if (event > segment.rBegin && event < segment.rEnd) panelEdges.push(event);
    }
    panelEdges.sort((a, b) => a - b);
    const deduplicatedEdges: number[] = [];
    for (const edge of panelEdges) {
      if (deduplicatedEdges.length === 0 || edge - deduplicatedEdges[deduplicatedEdges.length - 1] > positionTolerance) {
        deduplicatedEdges.push(edge);
      }
    }
    if (!bitwiseEqual(deduplicatedEdges[deduplicatedEdges.length - 1], segment.rEnd)) {
      deduplicatedEdges[deduplicatedEdges.length - 1] = segment.rEnd;
    }
    for (let panelIndex = 0; panelIndex + 1 < deduplicatedEdges.length; panelIndex++) {
      const panelBegin = deduplicatedEdges[panelIndex];
      const panelEnd = deduplicatedEdges[panelIndex + 1];
      const panel = buildRefinedPanel(panelBegin, panelEnd, evaluator);
      if (panel === null) {
        if (evaluator.densityInvalid) return densityFailure(evaluator, warnings);
        return failureResult(
          "numerical_failure",
          "MESH_QUADRATURE_NOT_CONVERGED",
          "quadrature did not converge on panel [" +
            formatDouble(panelBegin) +
            ", " +
            formatDouble(panelEnd) +
            "]; undeclared discontinuities must be declared via pins/extra_events",
          warnings,
        );
      }
      segment.measureTotal += panel.measure.total;
      segment.monitorTotal += panel.monitor.total;
      segment.panels.push(panel);
    }
    if (cfg.measure !== "width" && !(segment.monitorTotal > 0.0)) {
      return failureResult(
        "invalid_input",
        "MESH_DENSITY_INVALID",
        "segment [" +
          formatDouble(segment.rBegin) +
          ", " +
          formatDouble(segment.rEnd) +
          "] has zero measure; zero-density (void) regions are not representable in " +
          "this measure — use kWidth or declare the void boundary as a pin",
        warnings,
      );
    }
    segments.push(segment);
  }

  let minimumSum = 0;
  let maximumSum = 0;
  let minimumSumWithoutBox = 0;
  let maximumSumWithoutBox = 0;
  let widthCapacitySum = 0.0;
  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];
    const segmentLength = segment.rEnd - segment.rBegin;
    segment.nMin = cfg.minCellsPerSegment;
    if (cfg.drMin > 0.0) {
      const rawCapacity = Math.floor((segmentLength / cfg.drMin) * (1.0 + kCapacityRoundoff));
      segment.nMax = rawCapacity >= kIntMax ? kIntMax : Math.trunc(rawCapacity);
      widthCapacitySum += segmentLength / cfg.drMin;
    } else {
      segment.nMax = kIntMax;
    }
    minimumSumWithoutBox += segment.nMin;
    maximumSumWithoutBox = Math.min(kIntMax, maximumSumWithoutBox + segment.nMax);

    let nMinBoxBinding = false;
    let nMaxBoxBinding = false;
    if (hiMeasure > 0.0) {
      const rawMinimum = Math.ceil((segment.measureTotal / hiMeasure) * (1.0 - kCapacityRoundoff));
      const nMinBox = rawMinimum >= kIntMax ? kIntMax : Math.max(1, Math.trunc(rawMinimum));
      nMinBoxBinding = nMinBox > segment.nMin;
      segment.nMin = Math.max(segment.nMin, nMinBox);
    }
    if (loMeasure > 0.0) {
      const rawMaximum = Math.floor((segment.measureTotal / loMeasure) * (1.0 + kCapacityRoundoff));
      const nMaxBox = rawMaximum >= kIntMax ? kIntMax : Math.trunc(rawMaximum);
      const explicitLowerBoxIsStricter =
        cfg.cellMeasureMin > 0.0 && (cfg.measure !== "width" || cfg.cellMeasureMin > cfg.drMin);
      nMaxBoxBinding = explicitLowerBoxIsStricter && nMaxBox < segment.nMax;
      segment.nMax = Math.min(segment.nMax, nMaxBox);
    }
    if (segment.nMin > segment.nMax) {
      if (nMinBoxBinding || nMaxBoxBinding) {
        return failureResult(
          "infeasible",
          "MESH_CELL_MEASURE_BOX_INFEASIBLE",
          "segment " +
            String(i) +
            " has Q_s=" +
            formatDouble(segment.measureTotal) +
            ", lo_measure=" +
            formatDouble(loMeasure) +
            ", hi_measure=" +
            formatDouble(hiMeasure) +
            ", N_min_s=" +
            String(segment.nMin) +
            ", N_max_s=" +
            String(segment.nMax),
          warnings,
        );
      }
      return failureResult(
        "infeasible",
        "MESH_SEGMENT_BUDGET_CONFLICT",
        "segment " +
          String(i) +
          " has L_s=" +
          formatDouble(segmentLength) +
          ", dr_min=" +
          formatDouble(cfg.drMin) +
          ", N_min_s=" +
          String(segment.nMin) +
          ", N_max_s=" +
          String(segment.nMax),
        warnings,
      );
    }
    minimumSum += segment.nMin;
    maximumSum = Math.min(kIntMax, maximumSum + segment.nMax);
  }

  if (minimumSum > cfg.nCells) {
    if (minimumSumWithoutBox <= cfg.nCells) {
      return failureResult(
        "infeasible",
        "MESH_CELL_MEASURE_BOX_INFEASIBLE",
        "sum of box-constrained N_min_s=" +
          String(minimumSum) +
          " exceeds n_cells=" +
          String(cfg.nCells) +
          "; lo_measure=" +
          formatDouble(loMeasure) +
          ", hi_measure=" +
          formatDouble(hiMeasure),
        warnings,
      );
    }
    return failureResult(
      "infeasible",
      "MESH_SEGMENT_MIN_COUNT_INFEASIBLE",
      "sum(N_min_s)=" + String(minimumSum) + " exceeds n_cells=" + String(cfg.nCells),
      warnings,
    );
  }
  if (maximumSum < cfg.nCells) {
    if (maximumSumWithoutBox >= cfg.nCells) {
      return failureResult(
        "infeasible",
        "MESH_CELL_MEASURE_BOX_INFEASIBLE",
        "sum of box-constrained N_max_s=" +
          String(maximumSum) +
          " is below n_cells=" +
          String(cfg.nCells) +
          "; lo_measure=" +
          formatDouble(loMeasure) +
          ", hi_measure=" +
          formatDouble(hiMeasure),
        warnings,
      );
    }
    return failureResult(
      "infeasible",
      "MESH_DR_MIN_COUNT_INFEASIBLE",
      "sum(L_s/dr_min) capacity=" + formatDouble(widthCapacitySum) + " is below n_cells=" + String(cfg.nCells),
      warnings,
    );
  }

  let totalMeasure = 0.0;
  let totalMonitor = 0.0;
  for (const segment of segments) {
    totalMeasure += segment.measureTotal;
    totalMonitor += segment.monitorTotal;
  }
  const idealShares = new Array<number>(segments.length).fill(0.0);
  let allocatedCells = 0;
  for (let i = 0; i < segments.length; i++) {
    idealShares[i] = cfg.nCells * (segments[i].monitorTotal / totalMonitor);
    segments[i].nCells = segments[i].nMin;
    allocatedCells += segments[i].nCells;
  }
  while (allocatedCells < cfg.nCells) {
    let selected = 0;
    let largestDeficit = -Infinity;
    for (let i = 0; i < segments.length; i++) {
      if (segments[i].nCells >= segments[i].nMax) continue;
      const deficit = idealShares[i] - segments[i].nCells;
      if (deficit > largestDeficit) {
        largestDeficit = deficit;
        selected = i;
      }
    }
    segments[selected].nCells++;
    allocatedCells++;
  }

  const nodes: number[] = [];
  let segmentMeasureOffset = 0.0;
  for (let segmentIndex = 0; segmentIndex < segments.length; segmentIndex++) {
    const segment = segments[segmentIndex];
    const nCells = segment.nCells;
    const preferredNodes = new Array<number>(nCells + 1);
    preferredNodes[0] = segment.rBegin;
    preferredNodes[nCells] = segment.rEnd;
    for (let j = 1; j < nCells; j++) {
      const target = (j / nCells) * segment.monitorTotal;
      preferredNodes[j] = invertCumulative(segment, target, "monitor");
    }
    const preferredMeasures = new Array<number>(nCells);
    for (let i = 0; i < nCells; i++) {
      preferredMeasures[i] =
        evaluateCumulative(segment, preferredNodes[i + 1], "measure") - evaluateCumulative(segment, preferredNodes[i], "measure");
    }

    const bandCoverage = (candidate: number[]): boolean[][] => {
      const coverage = cfg.bands.map(() => new Array<boolean>(nCells).fill(false));
      let prefix = segmentMeasureOffset;
      for (let i = 0; i < nCells; i++) {
        const spanBegin = prefix / totalMeasure;
        prefix += candidate[i];
        const spanEnd = prefix / totalMeasure;
        for (let k = 0; k < cfg.bands.length; k++) {
          const band = cfg.bands[k];
          coverage[k][i] = spanBegin < band.measureFracEnd && spanEnd > band.measureFracBegin;
        }
      }
      return coverage;
    };

    const coverageAcc = bandCoverage(preferredMeasures);
    const maximumReconciliationRounds = nCells * cfg.bands.length + 1;
    let reconciliationRounds = 0;
    let coverageReconciled = false;
    let projectedMeasures: number[] = [];
    // Coverage only ever grows: a cell once inside a band stays bounded even if its final span
    // leaves the band. Conservative, and guarantees termination. Bands bound cell measures only;
    // they do not steer the integer allocation.
    while (reconciliationRounds < maximumReconciliationRounds) {
      reconciliationRounds++;
      const loBounds = new Array<number>(nCells).fill(loMeasure);
      const hiBounds = new Array<number>(nCells).fill(hiMeasure);
      for (let k = 0; k < cfg.bands.length; k++) {
        const band = cfg.bands[k];
        for (let cell = 0; cell < nCells; cell++) {
          if (!coverageAcc[k][cell]) continue;
          if (band.cellMeasureMin > 0.0) loBounds[cell] = cmax(loBounds[cell], band.cellMeasureMin);
          if (band.cellMeasureMax > 0.0) {
            hiBounds[cell] = hiBounds[cell] > 0.0 ? cmin(hiBounds[cell], band.cellMeasureMax) : band.cellMeasureMax;
          }
        }
      }

      if (cfg.bands.length > 0) {
        let lowerSum = 0.0;
        let upperSum = 0.0;
        let everyUpperBoundEnabled = true;
        for (let cell = 0; cell < nCells; cell++) {
          if (loBounds[cell] > 0.0 && hiBounds[cell] > 0.0 && loBounds[cell] > hiBounds[cell]) {
            return failureResult(
              "infeasible",
              "MESH_BAND_BOX_INFEASIBLE",
              "segment " +
                String(segmentIndex) +
                ", cell " +
                String(cell) +
                " has lo_i=" +
                formatDouble(loBounds[cell]) +
                ", hi_i=" +
                formatDouble(hiBounds[cell]),
              warnings,
            );
          }
          if (loBounds[cell] > 0.0) lowerSum += loBounds[cell];
          if (hiBounds[cell] > 0.0) upperSum += hiBounds[cell];
          else everyUpperBoundEnabled = false;
        }
        if (lowerSum > segment.measureTotal) {
          return failureResult(
            "infeasible",
            "MESH_BAND_BOX_INFEASIBLE",
            "segment " +
              String(segmentIndex) +
              " has sum(enabled lo_i)=" +
              formatDouble(lowerSum) +
              ", above measure_total=" +
              formatDouble(segment.measureTotal),
            warnings,
          );
        }
        if (everyUpperBoundEnabled && upperSum < segment.measureTotal) {
          return failureResult(
            "infeasible",
            "MESH_BAND_BOX_INFEASIBLE",
            "segment " +
              String(segmentIndex) +
              " has sum(enabled hi_i)=" +
              formatDouble(upperSum) +
              ", below measure_total=" +
              formatDouble(segment.measureTotal),
            warnings,
          );
        }
      }

      const outcome = projectCellMeasures(segment, cfg, loBounds, hiBounds, preferredMeasures);
      if (outcome.kind === "infeasible") {
        return failureResult("infeasible", "MESH_CHAIN_SUM_INFEASIBLE", "segment " + String(segmentIndex) + ": " + outcome.message, warnings);
      }
      if (outcome.kind === "numerical_failure") {
        return failureResult(
          "numerical_failure",
          "MESH_PROJECTION_STAGNATED",
          "segment " + String(segmentIndex) + ": " + outcome.message,
          warnings,
        );
      }
      projectedMeasures = outcome.projected;

      const projectedCoverage = bandCoverage(projectedMeasures);
      let coverageGrew = false;
      for (let k = 0; k < cfg.bands.length; k++) {
        for (let cell = 0; cell < nCells; cell++) {
          if (projectedCoverage[k][cell] && !coverageAcc[k][cell]) {
            coverageAcc[k][cell] = true;
            coverageGrew = true;
          }
        }
      }
      if (!coverageGrew) {
        coverageReconciled = true;
        break;
      }
    }
    if (!coverageReconciled) {
      return failureResult("numerical_failure", "MESH_PROJECTION_STAGNATED", "band coverage reconciliation exceeded its bound", warnings);
    }
    if (reconciliationRounds > 1) {
      warnings.push("band coverage grew during projection; " + String(reconciliationRounds) + " reconciliation rounds");
    }

    const projectedSum = accumulate(projectedMeasures);
    const segmentNodes = new Array<number>(nCells + 1);
    segmentNodes[0] = segment.rBegin;
    segmentNodes[nCells] = segment.rEnd;
    let prefix = 0.0;
    for (let j = 1; j < nCells; j++) {
      prefix += projectedMeasures[j - 1];
      if (cfg.measure === "width") {
        segmentNodes[j] = segment.rBegin + prefix * ((segment.rEnd - segment.rBegin) / projectedSum);
      } else {
        const target = prefix * (segment.measureTotal / projectedSum);
        segmentNodes[j] = invertCumulative(segment, target, "measure");
      }
    }
    if (segmentIndex === 0) nodes.push(...segmentNodes);
    else nodes.push(...segmentNodes.slice(1));
    segmentMeasureOffset += segment.measureTotal;
  }

  const verificationSegments: SegmentData[] = [];
  for (const segment of segments) {
    const verification = emptySegment(segment.rBegin, segment.rEnd);
    verification.nCells = segment.nCells;
    for (const panel of segment.panels) {
      const verificationPanel = buildVerificationPanel(panel, evaluator);
      if (evaluator.densityInvalid) return densityFailure(evaluator, warnings);
      verification.measureTotal += verificationPanel.measure.total;
      verification.panels.push(verificationPanel);
    }
    verificationSegments.push(verification);
  }

  for (let i = 0; i < nodes.length; i++) {
    if (!Number.isFinite(nodes[i])) {
      return failureResult(
        "numerical_failure",
        "MESH_POSTCHECK_NONFINITE",
        "node " + String(i) + " is non-finite: " + formatDouble(nodes[i]),
        warnings,
      );
    }
  }
  for (let i = 0; i + 1 < nodes.length; i++) {
    if (!(nodes[i + 1] > nodes[i])) {
      return failureResult(
        "numerical_failure",
        "MESH_POSTCHECK_MONOTONE",
        "nodes " +
          String(i) +
          " and " +
          String(i + 1) +
          " are not strictly increasing: " +
          formatDouble(nodes[i]) +
          ", " +
          formatDouble(nodes[i + 1]),
        warnings,
      );
    }
  }
  if (!bitwiseEqual(nodes[0], rMin) || !bitwiseEqual(nodes[nodes.length - 1], rMax)) {
    return failureResult("numerical_failure", "MESH_POSTCHECK_PIN_DRIFT", "domain endpoint drifted from its exact input value", warnings);
  }
  for (const pin of cfg.pins) {
    if (!nodes.some((node) => bitwiseEqual(node, pin.r))) {
      return failureResult(
        "numerical_failure",
        "MESH_POSTCHECK_PIN_DRIFT",
        "pin at r=" + formatDouble(pin.r) + " is not bitwise present in the nodes",
        warnings,
      );
    }
  }

  let minimumWidth = Infinity;
  for (let i = 0; i + 1 < nodes.length; i++) minimumWidth = cmin(minimumWidth, nodes[i + 1] - nodes[i]);
  if (cfg.drMin > 0.0 && minimumWidth < cfg.drMin * (1.0 - kWidthCheckTolerance)) {
    let message = "minimum width " + formatDouble(minimumWidth) + " is below dr_min=" + formatDouble(cfg.drMin);
    if (cfg.measure !== "width") {
      message +=
        "; joint mass-measure + width-floor projection is a planned upgrade; use a " +
        "larger dr_min-compatible budget or profile change";
    }
    return failureResult("numerical_failure", "MESH_POSTCHECK_WIDTH", message, warnings);
  }

  const verificationMeasures: number[] = [];
  let verificationTotal = 0.0;
  for (const segment of verificationSegments) verificationTotal += segment.measureTotal;
  const verificationFractionNodes = new Array<number>(cfg.nCells + 1).fill(0.0);
  let nodeOffset = 0;
  let verificationMeasureOffset = 0.0;
  for (const segment of verificationSegments) {
    for (let i = 0; i < segment.nCells; i++) {
      const cell = nodeOffset + i;
      const localBegin = evaluateCumulative(segment, nodes[cell], "measure");
      const localEnd = evaluateCumulative(segment, nodes[cell + 1], "measure");
      verificationMeasures.push(localEnd - localBegin);
      verificationFractionNodes[cell] = (verificationMeasureOffset + localBegin) / verificationTotal;
      verificationFractionNodes[cell + 1] = (verificationMeasureOffset + localEnd) / verificationTotal;
    }
    nodeOffset += segment.nCells;
    verificationMeasureOffset += segment.measureTotal;
  }

  const crossPinEdges: number[] = [];
  let cumulativeCells = 0;
  for (let i = 0; i + 1 < segments.length; i++) {
    cumulativeCells += segments[i].nCells;
    crossPinEdges.push(cumulativeCells);
  }

  let ratioMax = 1.0;
  let ratioSum = 0.0;
  let softExceedCount = 0;
  for (let i = 0; i + 1 < verificationMeasures.length; i++) {
    let ratio = Infinity;
    const a = verificationMeasures[i];
    const b = verificationMeasures[i + 1];
    if (a > 0.0 && b > 0.0 && Number.isFinite(a) && Number.isFinite(b)) ratio = cmax(a, b) / cmin(a, b);
    const pinIndex = crossPinEdges.indexOf(i + 1);
    const acrossPin = pinIndex >= 0;
    if (ratio > cfg.ratioHardMax * (1.0 + kRatioCheckTolerance)) {
      if (acrossPin) {
        if (!cfg.pins[pinIndex].ratioJumpAllowed) {
          return failureResult(
            "numerical_failure",
            "MESH_POSTCHECK_RATIO_CROSS_PIN",
            "pin at r=" +
              formatDouble(cfg.pins[pinIndex].r) +
              " has achieved adjacent-cell-measure ratio " +
              formatDouble(ratio) +
              "; set ratio_jump_allowed on the pin, or rebalance the intent so the " +
              "per-side cell measures meet at the pin",
            warnings,
          );
        }
      } else {
        return failureResult(
          "numerical_failure",
          "MESH_POSTCHECK_RATIO",
          "interior edge " +
            String(i + 1) +
            " has achieved adjacent-cell-measure ratio " +
            formatDouble(ratio) +
            ", above ratio_hard_max=" +
            formatDouble(cfg.ratioHardMax),
          warnings,
        );
      }
    }
    ratioMax = cmax(ratioMax, ratio);
    ratioSum += ratio;
    if (ratio > cfg.preferredRatio) softExceedCount++;
  }

  for (let i = 0; i < verificationMeasures.length; i++) {
    const measure = verificationMeasures[i];
    if (loMeasure > 0.0 && (!Number.isFinite(measure) || measure < loMeasure * (1.0 - kRatioCheckTolerance))) {
      return failureResult(
        "numerical_failure",
        "MESH_POSTCHECK_CELL_MEASURE_BOX",
        "cell " + String(i) + " has measure " + formatDouble(measure) + ", below lo_measure=" + formatDouble(loMeasure),
        warnings,
      );
    }
    if (hiMeasure > 0.0 && (!Number.isFinite(measure) || measure > hiMeasure * (1.0 + kRatioCheckTolerance))) {
      return failureResult(
        "numerical_failure",
        "MESH_POSTCHECK_CELL_MEASURE_BOX",
        "cell " + String(i) + " has measure " + formatDouble(measure) + ", above hi_measure=" + formatDouble(hiMeasure),
        warnings,
      );
    }
  }

  const bandCellMeasureMaxAchieved = cfg.bands.map(() => 0.0);
  const bandCellMeasureMinAchieved = cfg.bands.map(() => Infinity);
  const bandCoversCell = cfg.bands.map(() => false);
  for (let i = 0; i < verificationMeasures.length; i++) {
    const spanBegin = verificationFractionNodes[i];
    const spanEnd = verificationFractionNodes[i + 1];
    const measure = verificationMeasures[i];
    for (let k = 0; k < cfg.bands.length; k++) {
      const band = cfg.bands[k];
      const overlap = cmin(spanEnd, band.measureFracEnd) - cmax(spanBegin, band.measureFracBegin);
      if (!(overlap > kBandOverlapTolerance)) continue;
      const moved =
        "; cells that moved into the band during projection are bounded only " +
        "at verification time — widen the band, relax the bound, or change " +
        "the budget/profile so the preferred and final cell spans agree";
      if (band.cellMeasureMin > 0.0 && (!Number.isFinite(measure) || measure < band.cellMeasureMin * (1.0 - kRatioCheckTolerance))) {
        return failureResult(
          "numerical_failure",
          "MESH_POSTCHECK_BAND",
          "band " +
            String(k) +
            ", cell " +
            String(i) +
            " has measure " +
            formatDouble(measure) +
            ", below cell_measure_min=" +
            formatDouble(band.cellMeasureMin) +
            moved,
          warnings,
        );
      }
      if (band.cellMeasureMax > 0.0 && (!Number.isFinite(measure) || measure > band.cellMeasureMax * (1.0 + kRatioCheckTolerance))) {
        return failureResult(
          "numerical_failure",
          "MESH_POSTCHECK_BAND",
          "band " +
            String(k) +
            ", cell " +
            String(i) +
            " has measure " +
            formatDouble(measure) +
            ", above cell_measure_max=" +
            formatDouble(band.cellMeasureMax) +
            moved,
          warnings,
        );
      }
      bandCoversCell[k] = true;
      bandCellMeasureMaxAchieved[k] = cmax(bandCellMeasureMaxAchieved[k], measure);
      bandCellMeasureMinAchieved[k] = cmin(bandCellMeasureMinAchieved[k], measure);
    }
  }
  for (let k = 0; k < cfg.bands.length; k++) {
    if (!bandCoversCell[k]) bandCellMeasureMinAchieved[k] = 0.0;
  }

  const verificationSum = accumulate(verificationMeasures);
  const closureResidual = Math.abs(verificationSum - verificationTotal);
  const closureRelative = closureResidual / cmax(Math.abs(verificationTotal), kMuFloor);
  if (closureResidual > kClosureCheckTolerance * verificationTotal) {
    return failureResult(
      "numerical_failure",
      "MESH_POSTCHECK_MEASURE_CLOSURE",
      "verification measure closure residual " +
        formatDouble(closureResidual) +
        " exceeds 1e-8 of total measure " +
        formatDouble(verificationTotal),
      warnings,
    );
  }

  const diag = defaultDiagnostics();
  diag.ratioMaxAchieved = ratioMax;
  diag.ratioMeanAchieved = verificationMeasures.length > 1 ? ratioSum / (verificationMeasures.length - 1) : 1.0;
  diag.nRatioSoftExceed = softExceedCount;
  diag.widthMinAchieved = minimumWidth;
  let measureMin = verificationMeasures[0];
  let measureMax = verificationMeasures[0];
  for (const measure of verificationMeasures) {
    if (measure < measureMin) measureMin = measure;
    if (!(measure < measureMax)) measureMax = measure;
  }
  diag.cellMeasureMinAchieved = measureMin;
  diag.cellMeasureMaxAchieved = measureMax;
  diag.bandCellMeasureMaxAchieved = bandCellMeasureMaxAchieved;
  diag.bandCellMeasureMinAchieved = bandCellMeasureMinAchieved;
  diag.quadratureRelResidual = closureRelative;
  diag.warnings = warnings;
  diag.cellsPerSegment = segments.map((segment) => segment.nCells);
  return { ok: true, nodes, diag };
}
