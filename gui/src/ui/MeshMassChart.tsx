import { useState } from "react";
import { t } from "../i18n";
import type { FormState } from "../core/deck/formState";
import { computeRegionSegments } from "../core/deck/meshAuto";
import { toCanonical } from "../core/units";
import { SwitchField } from "./fields";
import { applyRangePolicy, SvgCartesianFrame } from "@tenryu-common/chart";

const PALETTE = [
  "var(--series-1)",
  "var(--series-2)",
  "var(--series-3)",
  "var(--series-4)",
  "var(--series-5)",
  "var(--series-6)",
  "var(--series-7)",
  "var(--series-8)",
];

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

export interface MeshStats {
  /** max/min cell mass over all cells */
  massRatio: number;
  /** adjacent-cell mass ratio at each material-change junction */
  interfaceRatios: Array<{ left: string; right: string; ratio: number }>;
  /** smallest cell width [µm] */
  drMinUm: number;
  /** innermost cell width [µm] */
  dr0Um: number;
}

export function computeMeshStats(
  edges: number[],
  masses: number[],
  materialNames: string[],
): MeshStats | null {
  const n = masses.length;
  if (n === 0 || edges.length !== n + 1) return null;
  let mMin = Infinity;
  let mMax = 0;
  let drMin = Infinity;
  for (let i = 0; i < n; i++) {
    const m = masses[i];
    if (!(Number.isFinite(m) && m > 0)) return null;
    if (m < mMin) mMin = m;
    if (m > mMax) mMax = m;
    const dr = edges[i + 1] - edges[i];
    if (dr < drMin) drMin = dr;
  }
  const interfaceRatios: MeshStats["interfaceRatios"] = [];
  for (let i = 0; i + 1 < n; i++) {
    if (materialNames[i] !== materialNames[i + 1]) {
      const a = masses[i];
      const b = masses[i + 1];
      interfaceRatios.push({
        left: materialNames[i],
        right: materialNames[i + 1],
        ratio: Math.max(a, b) / Math.min(a, b),
      });
    }
  }
  return {
    massRatio: mMax / mMin,
    interfaceRatios,
    drMinUm: drMin * 1.0e4,
    dr0Um: (edges[1] - edges[0]) * 1.0e4,
  };
}

/** Exact TS mirror of src/mesh/mesh.cu build_graded_nodes (as-built 2026-07-15):
 *  super-Gaussian mass weighting q = w(xi)/(r_est^2 + r_ref^2), per-segment
 *  normalization, then geometric-mean boundary matching between segments. */
export function computeGradedWidths(
  segments: GradedSegment[],
  grading: GradingParams,
): number[][] | null {
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

function applyBoundaryTargets(
  widths: number[],
  length: number,
  left: number | null,
  right: number | null,
): boolean {
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

export default function MeshMassChart({ form }: { form: FormState }) {
  const [logY, setLogY] = useState(false);
  const m = t();

  if (form.main.dimension !== "1D_SPH") return null;

  const totalCells =
    form.mesh.grid1d === "uniform"
      ? form.mesh.nr
      : form.mesh.segmentSource === "regions"
        ? form.mesh.nr
        : form.mesh.segments.reduce((a, s) => a + s.nr, 0);
  if (!(Number.isFinite(totalCells) && totalCells >= 1) || totalCells > 20000) {
    return (
      <div>
        <h2 className="mt-3 text-sm font-semibold">{m.form.massChartTitle}</h2>
        <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
          {m.form.massChartTooLarge}
        </p>
      </div>
    );
  }

  const rMin = toCanonical(form.mesh.rMin, "length");
  const rMax = toCanonical(form.mesh.rMax, "length");
  const edges: number[] = [];
  if (form.mesh.grid1d === "uniform") {
    const nr = form.mesh.nr;
    for (let i = 0; i <= nr; i += 1) {
      edges.push(rMin + (i * (rMax - rMin)) / nr);
    }
  } else {
    let segs: GradedSegment[] | null;
    if (form.mesh.segmentSource === "regions") {
      const auto = computeRegionSegments(form);
      segs = auto === null ? null : auto.map((s) => ({ rStart: s.rStartCm, rEnd: s.rEndCm, nr: s.nr }));
    } else {
      const manual: GradedSegment[] = [];
      let start = rMin;
      for (const segment of form.mesh.segments) {
        const end = toCanonical(segment.rEnd, "length");
        manual.push({ rStart: start, rEnd: end, nr: segment.nr });
        start = end;
      }
      segs = manual;
    }
    const widths = segs === null ? null : computeGradedWidths(segs, {
      edgeRatio: form.mesh.grading.edgeRatio,
      sgOrder: form.mesh.grading.sgOrder,
      sgSigma: form.mesh.grading.sgSigma,
    });
    if (widths !== null) {
      edges.push(rMin);
      let r = rMin;
      for (const w of widths) {
        for (const dw of w) {
          r += dw;
          edges.push(r);
        }
      }
    }
  }

  if (edges.length - 1 > 20000) {
    return (
      <div>
        <h2 className="mt-3 text-sm font-semibold">{m.form.massChartTitle}</h2>
        <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
          {m.form.massChartTooLarge}
        </p>
      </div>
    );
  }

  const invalidEdges = edges.some((edge, i) =>
    !Number.isFinite(edge) || (i > 0 && edge <= edges[i - 1]),
  );
  const empty = (
    <div>
      <h2 className="mt-3 text-sm font-semibold">{m.form.massChartTitle}</h2>
      <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.form.massChartEmpty}</p>
    </div>
  );
  if (
    edges.length < 2
    || !(rMax > rMin)
    || form.geometry.regions.length === 0
    || invalidEdges
  ) {
    return empty;
  }

  const materialNames: string[] = [];
  const masses: number[] = [];
  for (let i = 0; i < edges.length - 1; i += 1) {
    const inner = edges[i];
    const outer = edges[i + 1];
    const rc = 0.5 * (inner + outer);
    const region = form.geometry.regions.find(
      (candidate) => rc < toCanonical(candidate.rOuter, "length"),
    ) ?? form.geometry.regions[form.geometry.regions.length - 1];
    let volume: number;
    if (form.main.geometry1d === "spherical") {
      volume = (4 * Math.PI / 3) * (outer ** 3 - inner ** 3);
    } else if (form.main.geometry1d === "cylindrical") {
      volume = Math.PI * (outer ** 2 - inner ** 2);
    } else {
      volume = outer - inner;
    }
    masses.push(region.rho * volume);
    materialNames.push(region.materialName);
  }

  const maxMass = Math.max(...masses);
  if (maxMass <= 0) return empty;
  const stats = computeMeshStats(edges, masses, materialNames);

  const n = masses.length;
  const minPositiveMass = Math.min(...masses.filter((mass) => mass > 0));
  const minDecade = Math.floor(Math.log10(minPositiveMass));
  const maxDecade = Math.ceil(Math.log10(maxMass));
  // X range policy: zeroBaseline preserves the pre-existing 0-to-cell-count axis.
  const xRange = applyRangePolicy(0, n, { kind: "zeroBaseline" });
  // Y range policy: log uses the decade extent in log10 space; linear uses zeroBaseline.
  const yRange = logY
    ? applyRangePolicy(minDecade, maxDecade, { kind: "log" })
    : applyRangePolicy(0, maxMass, { kind: "zeroBaseline" });

  const materialColor = (materialName: string) => {
    const index = form.materials.findIndex((material) => material.name === materialName);
    return index < 0 ? "#7f8c8d" : PALETTE[index % PALETTE.length];
  };
  const appearingMaterials = Array.from(new Set(materialNames));
  const runs: Array<{ materialName: string; start: number; end: number }> = [];
  for (const materialName of appearingMaterials) {
    let start = -1;
    for (let i = 0; i <= n; i += 1) {
      if (i < n && materialNames[i] === materialName) {
        if (start < 0) start = i;
      } else if (start >= 0) {
        runs.push({ materialName, start, end: i });
        start = -1;
      }
    }
  }

  return (
    <div>
      <h2 className="mt-3 text-sm font-semibold">{m.form.massChartTitle}</h2>
      <SwitchField label={m.form.massChartLog} checked={logY} onChange={setLogY} />
      <SvgCartesianFrame
        width={560}
        height={220}
        xRange={xRange}
        yRange={yRange}
        yLog={logY}
        xLabel={m.form.massChartXLabel}
        yLabel={m.form.massChartYLabel}
      >
        {(x, y) => (
          <>
            {runs.map((run) => {
              const points: string[] = [];
              for (let i = run.start; i < run.end; i += 1) {
                if (!logY || masses[i] > 0) {
                  points.push(`${x(i + 0.5)},${y(logY ? Math.log10(masses[i]) : masses[i])}`);
                }
              }
              if (points.length === 0) return null;
              return (
                <polyline
                  key={`${run.materialName}-${run.start}`}
                  points={points.join(" ")}
                  fill="none"
                  stroke={materialColor(run.materialName)}
                  strokeWidth={1.5}
                />
              );
            })}
          </>
        )}
      </SvgCartesianFrame>
      <div className="flex flex-wrap gap-3">
        {appearingMaterials.map((materialName) => (
          <div className="flex items-center gap-1 text-xs" key={materialName}>
            <div style={{ width: 10, height: 10, background: materialColor(materialName) }} />
            <span>{materialName}</span>
          </div>
        ))}
      </div>
      {stats !== null && (
        <div
          className="text-xs"
          style={{ fontFamily: "var(--mono)", color: "var(--fg-secondary)" }}
        >
          <div>{m.form.meshStatMassRatio}: {stats.massRatio.toPrecision(3)}</div>
          <div>
            {m.form.meshStatDr}: min {stats.drMinUm.toPrecision(3)} µm / center {stats.dr0Um.toPrecision(3)} µm
          </div>
          {stats.interfaceRatios.map((r, i) => (
            <div
              key={i}
              style={{ color: r.ratio > 2 ? "var(--err)" : "var(--fg-secondary)" }}
            >
              {r.left} | {r.right}: ×{r.ratio.toPrecision(3)}
              {r.ratio > 2 ? ` ${m.form.meshStatInterfaceWarn}` : ""}
            </div>
          ))}
        </div>
      )}
      {form.mesh.grid1d === "graded" && (
        <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
          {m.form.massChartGradedNote}
        </p>
      )}
    </div>
  );
}
