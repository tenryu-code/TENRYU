import { useMemo, useState } from "react";
import { Button } from "@tenryu-common/ui/kit";
import { applyRangePolicy, SvgCartesianFrame } from "@tenryu-common/chart";
import { t } from "../../i18n";
import {
  cellMetrics,
  computeMeshNodes1d,
  isMassMeasure,
  solverZonedMethod,
  summarizeMesh,
  zoningIntentNodes1d,
  type MeshCellMetrics,
} from "../../core/deck/mesh1d";
import type { FormState } from "../../core/deck/formState";
import { useApp } from "../../store";
import { SelectField } from "../fields";

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

type Quantity = "width" | "areal" | "reference" | "ratio" | "mass";
type Axis = "r" | "depth" | "index";

const MAX_CELLS = 20000;

function fmt(x: number | null): string {
  return x === null || !Number.isFinite(x) ? "—" : String(Number(x.toPrecision(4)));
}

function sci(x: number | null): string {
  return x === null || !Number.isFinite(x) ? "—" : x.toExponential(3);
}

/** y value of each cell for a quantity (null where it is not shown). */
function cellValues(metrics: MeshCellMetrics, quantity: Quantity): Array<number | null> {
  const n = metrics.masses.length;
  const out: Array<number | null> = [];
  for (let i = 0; i < n; i++) {
    const isVoid = metrics.materials[i] === "VOID";
    if (quantity === "width") out.push(metrics.widths[i] * 1.0e4);
    else if (isVoid) out.push(null);
    else if (quantity === "areal") out.push(metrics.localArealMass[i]);
    else if (quantity === "reference") out.push(metrics.referenceArealMass[i]);
    else if (quantity === "mass") out.push(metrics.masses[i]);
    else if (i + 1 < n && metrics.materials[i + 1] !== "VOID") {
      const a = metrics.masses[i];
      const b = metrics.masses[i + 1];
      out.push(Math.max(a, b) / Math.min(a, b));
    } else out.push(null);
  }
  return out;
}

/** x value of each cell for an axis (null where it is not shown: depth outside the target). */
function cellPositions(metrics: MeshCellMetrics, axis: Axis): Array<number | null> {
  const n = metrics.masses.length;
  const out: Array<number | null> = [];
  for (let i = 0; i < n; i++) {
    const centre = 0.5 * (metrics.edges[i] + metrics.edges[i + 1]);
    if (axis === "index") out.push(i + 0.5);
    else if (axis === "r") out.push(centre * 1.0e4);
    else {
      const depth = (metrics.targetOuterCm - centre) * 1.0e4;
      out.push(depth > 0 ? Math.log10(depth) : null);
    }
  }
  return out;
}

/** Nodes Studio computes for the form's 1D mesh; for a zoning_intent, Studio's copy of the solver's
 *  zoning and, when it refuses the intent, the solver's error. */
function studioNodes(form: FormState): { nodes: number[] | null; error: string | null } {
  if (form.main.dimension === "1D_SPH" && solverZonedMethod(form.mesh.grid1d)) {
    const zoned = zoningIntentNodes1d(form);
    return "nodes" in zoned ? { nodes: zoned.nodes, error: null } : { nodes: null, error: zoned.error };
  }
  return { nodes: computeMeshNodes1d(form), error: null };
}

/** Cell widths, areal masses and adjacent ratios of the 1D mesh, from the nodes Studio computes or
 *  from the solver's preview (validate --mesh-preview), with the solver's resolution requirement. */
export default function MeshDiagnostics({ form }: { form: FormState }) {
  const m = t().mesh1d.ui;
  const deck = useApp((s) => s.deck);
  const preview = useApp((s) => s.meshPreview);
  const previewDeck = useApp((s) => s.meshPreviewDeck);
  const busy = useApp((s) => s.meshPreviewBusy);
  const previewError = useApp((s) => s.meshPreviewError);
  const runMeshPreview = useApp((s) => s.runMeshPreview);
  const [quantity, setQuantity] = useState<Quantity>(form.laser.enabled ? "reference" : "width");
  const [axis, setAxis] = useState<Axis>("r");
  // The zoning of a large intent takes a few ms: computed again only when the form changes.
  const studio = useMemo(() => studioNodes(form), [form]);
  if (form.main.dimension !== "1D_SPH") return null;

  const previewCurrent =
    preview !== null && preview.dim === 1 && preview.rNodes !== null && previewDeck !== null && previewDeck === deck;
  const nodes = previewCurrent ? (preview.rNodes as number[]) : studio.nodes;
  const solverZoned = solverZonedMethod(form.mesh.grid1d);
  // apply="enforce" adds the requirement's ceilings to the intent before the solver zones it
  // (laser on, a mass measure); Studio's zoning does not include them.
  const enforceAddsBands =
    solverZoned &&
    form.mesh.resolutionRequirement.apply === "enforce" &&
    form.laser.enabled &&
    isMassMeasure(form.mesh.zoningIntent.measure);
  const header = (
    <>
      <h2 className="mt-3 text-sm font-semibold">{m.diagTitle}</h2>
      <div className="flex flex-wrap items-center gap-2">
        <Button disabled={busy} onClick={() => void runMeshPreview()}>{busy ? m.diagChecking : m.diagCheck}</Button>
        {previewError !== null && <span className="text-xs" style={{ color: "var(--err)" }}>{previewError}</span>}
      </div>
      {preview !== null && preview.dim === 1 && !previewCurrent && (
        <p className="text-xs" style={{ color: "var(--warn)" }}>{m.diagStale}</p>
      )}
    </>
  );
  if (nodes === null) {
    return (
      <div>
        {header}
        {studio.error !== null ? (
          <p className="text-xs" style={{ color: "var(--err)" }}>{m.diagZoningError(studio.error)}</p>
        ) : (
          <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.diagEmpty}</p>
        )}
      </div>
    );
  }
  if (nodes.length - 1 > MAX_CELLS) {
    return (
      <div>
        {header}
        <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.diagTooLarge}</p>
      </div>
    );
  }
  const metrics = cellMetrics(form, nodes);
  if (metrics === null) {
    return (
      <div>
        {header}
        <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.diagEmpty}</p>
      </div>
    );
  }
  const summary = summarizeMesh(metrics);
  const values = cellValues(metrics, quantity);
  const xs = cellPositions(metrics, axis);
  const logY = quantity !== "ratio";
  const yPlot = values.map((v) => (v === null || !(v > 0) ? null : logY ? Math.log10(v) : v));
  const finiteX = xs.filter((x): x is number => x !== null);
  const finiteY = yPlot.filter((y, i): y is number => y !== null && xs[i] !== null);
  const requirement = previewCurrent ? (preview.requirement ?? null) : null;
  const bands = quantity === "reference" && requirement !== null && requirement.applicable ? requirement.bands : [];
  const capOf = (centre: number) => {
    let cap: number | null = null;
    for (const band of bands) {
      if (centre >= band.rLoCm && centre < band.rHiCm) cap = cap === null ? band.arealMassMaxGcm2 : Math.min(cap, band.arealMassMaxGcm2);
    }
    return cap;
  };
  for (const band of bands) finiteY.push(Math.log10(band.arealMassMaxGcm2));
  if (finiteX.length === 0 || finiteY.length === 0) {
    return (
      <div>
        {header}
        <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.diagEmpty}</p>
      </div>
    );
  }
  const xRange = applyRangePolicy(Math.min(...finiteX), Math.max(...finiteX), { kind: "dataExtent" });
  const yRange = applyRangePolicy(Math.min(...finiteY), Math.max(...finiteY), logY ? { kind: "log" } : { kind: "dataExtent" });
  const materialColor = (name: string) => {
    const index = form.materials.findIndex((material) => material.name === name);
    return index < 0 ? "#7f8c8d" : PALETTE[index % PALETTE.length];
  };
  // Polylines per run of one material.
  const runs: Array<{ name: string; points: Array<[number, number]> }> = [];
  for (let i = 0; i < values.length; i++) {
    const name = metrics.materials[i];
    const px = xs[i];
    const py = yPlot[i];
    if (runs.length === 0 || runs[runs.length - 1].name !== name || py === null || px === null) {
      runs.push({ name, points: [] });
    }
    if (py !== null && px !== null) runs[runs.length - 1].points.push([px, py]);
  }
  const violations: number[] = [];
  if (bands.length > 0) {
    for (let i = 0; i < values.length; i++) {
      const cap = capOf(0.5 * (metrics.edges[i] + metrics.edges[i + 1]));
      const v = values[i];
      if (cap !== null && v !== null && v > cap * (1 + 1e-9)) violations.push(i);
    }
  }
  const xLabel = axis === "r" ? m.axisR : axis === "depth" ? m.axisDepth : m.axisIndex;
  const massUnit = form.main.geometry1d === "spherical" ? "g" : form.main.geometry1d === "cylindrical" ? "g/cm" : "g/cm²";
  const yLabel =
    quantity === "width" ? m.qWidth : quantity === "areal" ? m.qAreal : quantity === "reference" ? m.qRefArealAxis : quantity === "ratio" ? m.qRatio : m.qMass(massUnit);
  const bandX = (r: number) => {
    if (axis === "r") return r * 1.0e4;
    if (axis === "depth") {
      const depth = (metrics.targetOuterCm - r) * 1.0e4;
      return depth > 0 ? Math.log10(depth) : null;
    }
    let index = 0;
    while (index < metrics.edges.length - 1 && metrics.edges[index] < r) index++;
    return index;
  };
  const appearing = Array.from(new Set(metrics.materials));
  return (
    <div>
      {header}
      <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
        {previewCurrent ? m.diagSource.preview : solverZoned ? m.diagSource.guiZoning : m.diagSource.gui}
      </p>
      {!previewCurrent && enforceAddsBands && (
        <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.diagEnforceNote}</p>
      )}
      <div className="flex flex-wrap gap-2">
        <SelectField
          label={m.diagQuantity}
          value={quantity}
          options={[
            { value: "width", label: m.qWidth },
            { value: "areal", label: m.qAreal },
            { value: "reference", label: m.qRefAreal },
            { value: "ratio", label: m.qRatio },
            { value: "mass", label: m.qMass(massUnit) },
          ]}
          onChange={(v) => setQuantity(v as Quantity)}
        />
        <SelectField
          label={m.diagAxis}
          value={axis}
          options={[
            { value: "r", label: m.axisR },
            { value: "depth", label: m.axisDepth },
            { value: "index", label: m.axisIndex },
          ]}
          onChange={(v) => setAxis(v as Axis)}
        />
      </div>
      <SvgCartesianFrame width={560} height={240} xRange={xRange} yRange={yRange} yLog={logY} xLog={axis === "depth"} xLabel={xLabel} yLabel={yLabel}>
        {(x, y) => (
          <>
            {runs.map((run, k) =>
              run.points.length === 0 ? null : (
                <polyline
                  key={k}
                  points={run.points.map(([px, py]) => `${x(px)},${y(py)}`).join(" ")}
                  fill="none"
                  stroke={materialColor(run.name)}
                  strokeWidth={1.5}
                />
              ),
            )}
            {quantity === "ratio" &&
              [1.3, 2.0].map((level) => (
                <line key={level} x1={x(xRange[0])} x2={x(xRange[1])} y1={y(level)} y2={y(level)} stroke="var(--fg-secondary)" strokeDasharray="4 3" />
              ))}
            {bands.map((band, k) => {
              let a = bandX(band.rLoCm);
              let b = bandX(band.rHiCm);
              // On the depth axis an end at or outside the surface has no logarithm; the band
              // then reaches the shallowest cell plotted.
              if (axis === "depth" && (a !== null || b !== null)) {
                a = a ?? xRange[0];
                b = b ?? xRange[0];
              }
              if (a === null || b === null) return null;
              return (
                <line
                  key={`band-${k}`}
                  x1={x(a)}
                  x2={x(b)}
                  y1={y(Math.log10(band.arealMassMaxGcm2))}
                  y2={y(Math.log10(band.arealMassMaxGcm2))}
                  stroke="var(--err)"
                  strokeDasharray="5 3"
                  strokeWidth={1.2}
                />
              );
            })}
            {violations.map((i) =>
              xs[i] === null || yPlot[i] === null ? null : (
                <circle key={`v-${i}`} cx={x(xs[i] as number)} cy={y(yPlot[i] as number)} r={2.5} fill="var(--err)" />
              ),
            )}
          </>
        )}
      </SvgCartesianFrame>
      <div className="flex flex-wrap gap-3">
        {appearing.map((name) => (
          <div className="flex items-center gap-1 text-xs" key={name}>
            <div style={{ width: 10, height: 10, background: materialColor(name) }} />
            <span>{name}</span>
          </div>
        ))}
        {bands.length > 0 && (
          <div className="flex items-center gap-1 text-xs">
            <div style={{ width: 14, height: 0, borderTop: "2px dashed var(--err)" }} />
            <span>{m.diagCeiling}</span>
          </div>
        )}
      </div>
      <div className="text-xs" style={{ fontFamily: "var(--mono)", color: "var(--fg-secondary)" }}>
        <div>{m.diagStats(summary.nCells, fmt(summary.minWidthCm * 1.0e4), fmt(summary.maxAdjacentMassRatio))}</div>
        <div>{m.diagSurface(fmt(summary.surfaceWidthCm === null ? null : summary.surfaceWidthCm * 1.0e4), sci(summary.surfaceLocalArealMass), sci(summary.surfaceReferenceArealMass))}</div>
        {summary.interfaceRatios.map((r, i) => (
          <div key={i} style={{ color: r.ratio > 2 ? "var(--err)" : undefined }}>
            {m.diagInterface(r.left, r.right, fmt(r.ratio))}
          </div>
        ))}
        {requirement !== null &&
          (requirement.applicable ? (
            <div style={{ color: (requirement.ablation?.nViolations ?? 0) + (requirement.shock?.nViolations ?? 0) > 0 ? "var(--err)" : undefined }}>
              {m.diagRequirement(requirement.ablation?.nViolations ?? 0, requirement.shock?.nViolations ?? 0)}
            </div>
          ) : (
            <div>{m.diagRequirementNA(requirement.reason)}</div>
          ))}
        {violations.length > 0 && <div style={{ color: "var(--err)" }}>{m.diagViolation(violations.length)}</div>}
      </div>
    </div>
  );
}
