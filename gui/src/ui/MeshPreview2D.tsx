import { useMemo } from "react";
import { logLine } from "@tenryu-common/core/applog";
import type { FormState } from "../core/deck/formState";
import { resolveShape, shapeOutlinePoints } from "../core/geometry2d";
import { computeLocalMeshPreview } from "../core/localMeshPreview";
import type { SnapshotMesh2D } from "@tenryu-common/core/results/snapshotMesh";
import { toCanonical } from "../core/units";
import { t } from "../i18n";
import { currentProfile, useApp } from "../store";
import { Button } from "@tenryu-common/ui/kit";

const WIDTH = 560;
const HEIGHT = 420;
const PLOT_L = 56;
const PLOT_R = 16;
const PLOT_T = 16;
const PLOT_B = 40;
const GRID_STROKE = "#3d4a63";
const INTERFACE_STROKE = "#ff9f43";
const AXIS_STROKE = "#8b93a7";
// Set false to transpose the snapshot grid if the server layout changes.
const ROW_MAJOR_I_FIRST = true;

const mapErr = (e: string): string =>
  e === "NO_PROFILE"
    ? t().validate.noProfile
    : e === "NO_BIN"
      ? t().server.binMissing
      : e === "NO_TOOLS"
        ? t().geo2d.toolsMissing
        : e;

function linspace(a: number, b: number, n: number): number[] {
  if (n <= 1) return [a];
  return Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
}

function sampledIndices(count: number, stride: number): number[] {
  const indices: number[] = [];
  for (let i = 0; i < count; i += stride) indices.push(i);
  if (indices[indices.length - 1] !== count - 1) indices.push(count - 1);
  return indices;
}

function niceTicks(a: number, b: number): number[] {
  const span = b - a;
  if (!(Number.isFinite(span) && span > 0)) return [a];
  const magnitude = 10 ** Math.floor(Math.log10(span));
  const steps = [0.1, 0.2, 0.5, 1, 2, 5].map((factor) => magnitude * factor);
  let step = steps[0];
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const candidate of steps) {
    const count = Math.floor(b / candidate) - Math.ceil(a / candidate) + 1;
    if (count >= 3 && count <= 7 && Math.abs(count - 5) < bestDistance) {
      step = candidate;
      bestDistance = Math.abs(count - 5);
    }
  }
  const first = Math.ceil(a / step) * step;
  const ticks: number[] = [];
  for (let value = first; value <= b + step * 1e-12; value += step) ticks.push(value);
  return ticks;
}

function PolarInBoxSnapshotPreview({
  form,
  snapshot,
  busy,
  error,
  fetchMeshSnapshot,
}: {
  form: FormState;
  snapshot: SnapshotMesh2D | null;
  busy: boolean;
  error: string | null;
  fetchMeshSnapshot: () => Promise<void>;
}) {
  const m = t();
  let xMin = toCanonical(form.mesh.rMin, "length");
  let xMax = toCanonical(form.mesh.rMax, "length");
  let zMin = toCanonical(form.mesh.zMin, "length");
  let zMax = toCanonical(form.mesh.zMax, "length");
  if (snapshot !== null) {
    xMin = snapshot.xR[0];
    xMax = snapshot.xR[0];
    zMin = snapshot.xZ[0];
    zMax = snapshot.xZ[0];
    for (let k = 1; k < snapshot.xR.length; k++) {
      xMin = Math.min(xMin, snapshot.xR[k]);
      xMax = Math.max(xMax, snapshot.xR[k]);
      zMin = Math.min(zMin, snapshot.xZ[k]);
      zMax = Math.max(zMax, snapshot.xZ[k]);
    }
  }
  const xSpan = xMax - xMin || 1;
  const ySpan = zMax - zMin || 1;
  const innerW = WIDTH - PLOT_L - PLOT_R;
  const innerH = HEIGHT - PLOT_T - PLOT_B;
  const scale = Math.min(innerW / xSpan, innerH / ySpan);
  const x = (r: number) => PLOT_L + (r - xMin) * scale + (innerW - xSpan * scale) / 2;
  const y = (z: number) => PLOT_T + (zMax - z) * scale + (innerH - ySpan * scale) / 2;
  const rTicks = niceTicks(xMin, xMax);
  const zTicks = niceTicks(zMin, zMax);
  const useMicrons = Math.max(Math.abs(xSpan), Math.abs(ySpan)) < 0.1;
  const axisUnit = useMicrons ? "µm" : "cm";
  const formatTick = (value: number) =>
    useMicrons
      ? `${Number((value * 1e4).toFixed(0))}µm`
      : String(Number(value.toPrecision(3)));
  const frameLeft = x(xMin);
  const frameRight = x(xMax);
  const frameTop = y(zMax);
  const frameBottom = y(zMin);
  const csrCellCount = snapshot?.kind === "csr" ? snapshot.offsets.length - 1 : 0;
  const decimated =
    snapshot !== null &&
    (snapshot.kind === "tensor"
      ? snapshot.nRNodes * snapshot.nZNodes > 40000
      : csrCellCount > 20000);
  const iStride =
    snapshot?.kind === "tensor" && decimated
      ? Math.max(1, Math.ceil(snapshot.nRNodes / 200))
      : 1;
  const jStride =
    snapshot?.kind === "tensor" && decimated
      ? Math.max(1, Math.ceil(snapshot.nZNodes / 200))
      : 1;
  const iIndices =
    snapshot?.kind === "tensor" ? sampledIndices(snapshot.nRNodes, iStride) : [];
  const jIndices =
    snapshot?.kind === "tensor" ? sampledIndices(snapshot.nZNodes, jStride) : [];
  const nodeAt = (i: number, j: number): [number, number] => {
    if (snapshot?.kind !== "tensor") return [0, 0];
    const index = ROW_MAJOR_I_FIRST
      ? i * snapshot.nZNodes + j
      : j * snapshot.nRNodes + i;
    return [snapshot.xR[index], snapshot.xZ[index]];
  };
  const csrStride = decimated ? Math.ceil(csrCellCount / 20000) : 1;
  const csrPath = (() => {
    if (snapshot?.kind !== "csr") return "";
    const commands: string[] = [];
    for (let cell = 0; cell < csrCellCount; cell += csrStride) {
      const start = snapshot.offsets[cell];
      const end = snapshot.offsets[cell + 1];
      for (let entry = start; entry < end; entry++) {
        const node = snapshot.indices[entry];
        commands.push(
          `${entry === start ? "M" : "L"} ${x(snapshot.xR[node])} ${y(snapshot.xZ[node])}`,
        );
      }
      commands.push("Z");
    }
    return commands.join(" ");
  })();

  return (
    <div className="mt-3 flex flex-col gap-2">
      {snapshot === null && (
        <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
          {m.geo2d.pibPreviewNeedsFetch}
        </p>
      )}
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        width={WIDTH}
        height={HEIGHT}
        style={{ background: "#0b0e14", borderRadius: 8 }}
      >
        <rect
          x={frameLeft}
          y={frameTop}
          width={frameRight - frameLeft}
          height={frameBottom - frameTop}
          stroke={GRID_STROKE}
          fill="none"
        />
        {rTicks.map((value) => (
          <g key={`r-tick-${value}`}>
            <line
              x1={x(value)}
              y1={frameBottom}
              x2={x(value)}
              y2={frameBottom + 4}
              stroke={AXIS_STROKE}
            />
            <text
              x={x(value)}
              y={frameBottom + 14}
              fill={AXIS_STROKE}
              fontSize={10}
              textAnchor="middle"
            >
              {formatTick(value)}
            </text>
          </g>
        ))}
        {zTicks.map((value) => (
          <g key={`z-tick-${value}`}>
            <line
              x1={frameLeft - 4}
              y1={y(value)}
              x2={frameLeft}
              y2={y(value)}
              stroke={AXIS_STROKE}
            />
            <text
              x={frameLeft - 6}
              y={y(value) + 3}
              fill={AXIS_STROKE}
              fontSize={10}
              textAnchor="end"
            >
              {formatTick(value)}
            </text>
          </g>
        ))}
        <text
          x={(frameLeft + frameRight) / 2}
          y={HEIGHT - 4}
          fill={AXIS_STROKE}
          fontSize={10}
          textAnchor="middle"
        >
          R [{axisUnit}]
        </text>
        <text x={4} y={Math.max(10, frameTop - 4)} fill={AXIS_STROKE} fontSize={10}>
          Z [{axisUnit}]
        </text>
        {snapshot?.kind === "tensor" && (
          <>
            {jIndices.map((j) => (
              <polyline
                key={`snapshot-j-${j}`}
                points={Array.from({ length: snapshot.nRNodes }, (_, i) => {
                  const [r, z] = nodeAt(i, j);
                  return `${x(r)},${y(z)}`;
                }).join(" ")}
                stroke={GRID_STROKE}
                strokeWidth={1}
                fill="none"
              />
            ))}
            {iIndices.map((i) => (
              <polyline
                key={`snapshot-i-${i}`}
                points={Array.from({ length: snapshot.nZNodes }, (_, j) => {
                  const [r, z] = nodeAt(i, j);
                  return `${x(r)},${y(z)}`;
                }).join(" ")}
                stroke={GRID_STROKE}
                strokeWidth={1}
                fill="none"
              />
            ))}
          </>
        )}
        {snapshot?.kind === "csr" && (
          <path d={csrPath} stroke={GRID_STROKE} strokeWidth={1} fill="none" />
        )}
        {form.geometry.shapes2d.map((shape, i) => {
          const resolved = resolveShape(shape);
          const outlines = shapeOutlinePoints(resolved);
          if (outlines === null) return null;
          const svgPoints = (points: Array<[number, number]>) =>
            points.map(([r, z]) => `${x(r)},${y(z)}`).join(" ");
          if (resolved.kind === "shell") {
            return (
              <g key={`shape-${i}`}>
                {outlines.map((points, outlineIndex) => (
                  <polyline
                    key={`shape-${i}-${outlineIndex}`}
                    points={svgPoints(points)}
                    stroke={INTERFACE_STROKE}
                    strokeWidth={1.5}
                    strokeDasharray="6 4"
                    fill="none"
                  />
                ))}
              </g>
            );
          }
          return (
            <polyline
              key={`shape-${i}`}
              points={svgPoints(outlines[0])}
              stroke={INTERFACE_STROKE}
              strokeWidth={1.5}
              strokeDasharray="6 4"
              fill="none"
            />
          );
        })}
      </svg>
      {snapshot !== null && (
        <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
          {m.meshPreview.cells}:{" "}
          {snapshot.kind === "tensor"
            ? `${snapshot.nRNodes - 1}×${snapshot.nZNodes - 1}`
            : csrCellCount}
          {decimated && " (decimated)"}
        </p>
      )}
      <div className="flex items-center gap-2">
        <Button disabled={busy} onClick={() => void fetchMeshSnapshot()}>
          {snapshot === null ? m.geo2d.fetchTrueGrid : m.meshPreview.button}
        </Button>
        {busy && (
          <span className="text-xs" style={{ color: "var(--fg-secondary)" }}>
            {m.meshPreview.running}
          </span>
        )}
        {error !== null && (
          <span
            className="whitespace-pre-wrap break-all text-xs"
            style={{ color: "var(--err)", fontFamily: "var(--mono)" }}
          >
            {mapErr(error)}
          </span>
        )}
      </div>
    </div>
  );
}

export default function MeshPreview2D({ form }: { form: FormState }) {
  const m = t();
  const meshPreview = useApp((s) => s.meshPreview);
  const busy = useApp((s) => s.meshPreviewBusy);
  const error = useApp((s) => s.meshPreviewError);
  const runMeshPreview = useApp((s) => s.runMeshPreview);
  const meshSnapshot = useApp((s) => s.meshSnapshot);
  const meshSnapshotBusy = useApp((s) => s.meshSnapshotBusy);
  const meshSnapshotError = useApp((s) => s.meshSnapshotError);
  const fetchMeshSnapshot = useApp((s) => s.fetchMeshSnapshot);
  const hasProfile = useApp((s) => currentProfile(s) !== null);
  const local = useMemo(() => {
    try {
      return computeLocalMeshPreview(form);
    } catch (e) {
      logLine("error", `preview compute failed: ${e}`);
      return null;
    }
  }, [form]);
  const serverMatch = useMemo(() => {
    try {
      if (meshPreview === null || local === null) return null;
      const serverNodes = meshPreview.rNodes;
      const localNodes = local.rNodes;
      if (serverNodes === null || localNodes === null) {
        return serverNodes === null && localNodes === null;
      }
      return (
        serverNodes.length === localNodes.length &&
        serverNodes.every(
          (node, i) =>
            Math.abs(node - localNodes[i]) <=
            1.0e-9 * Math.max(1, Math.abs(node), Math.abs(localNodes[i])),
        )
      );
    } catch (e) {
      logLine("error", `preview compute failed: ${e}`);
      return null;
    }
  }, [meshPreview, local]);

  const crossCheck = (
    <div className="flex items-center gap-2">
      <Button disabled={busy || !hasProfile} onClick={() => void runMeshPreview()}>
        {m.meshPreview.checkServer}
      </Button>
      {busy && (
        <span className="text-xs" style={{ color: "var(--fg-secondary)" }}>
          {m.meshPreview.running}
        </span>
      )}
      {error !== null && (
        <span
          className="whitespace-pre-wrap break-all text-xs"
          style={{ color: "var(--err)", fontFamily: "var(--mono)" }}
        >
          {mapErr(error)}
        </span>
      )}
      {!busy && error === null && serverMatch !== null && (
        <span
          className="text-xs"
          style={{ color: serverMatch ? "var(--fg-secondary)" : "var(--err)" }}
        >
          {serverMatch ? m.meshPreview.serverMatch : m.meshPreview.serverMismatch}
        </span>
      )}
    </div>
  );

  if (form.mesh.meshMode2d === "polar_in_box") {
    return (
      <PolarInBoxSnapshotPreview
        form={form}
        snapshot={meshSnapshot}
        busy={meshSnapshotBusy}
        error={meshSnapshotError}
        fetchMeshSnapshot={fetchMeshSnapshot}
      />
    );
  }

  if (local === null) {
    return (
      <div className="mt-3 flex flex-col gap-2">
        <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
          {m.meshPreview.unavailableLocal}
        </p>
        {crossCheck}
      </div>
    );
  }

  const data = local;
  if ((data.nr + 1) * (data.nz + 1) > 2_000_000) {
    logLine("error", "preview skipped: grid too large");
    return (
      <div className="mt-3 flex flex-col gap-2">
        <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
          {m.meshPreview.unavailableLocal}
        </p>
        {crossCheck}
      </div>
    );
  }
  const isPolar = data.logicalMesh2d === "spherical_polar_halfplane";
  const polar = data.polar!;
  let rl: number[];
  if (data.rNodes !== null) {
    rl = data.rNodes;
  } else if (isPolar && polar.centerTreatment === "tri_fan") {
    rl = linspace(0, polar.sMax, data.nr + 1);
  } else if (isPolar) {
    const ds = polar.sMax / (data.nr + polar.kappa);
    rl = Array.from({ length: data.nr + 1 }, (_, i) => polar.kappa * ds + i * ds);
  } else {
    rl = linspace(data.rMin, data.rMax, data.nr + 1);
  }

  const thetaLadder = isPolar
    ? Array.from({ length: data.nz + 1 }, (_, j) =>
        polar.equalMu
          ? Math.acos(Math.max(-1, Math.min(1, 1 - (2 * j) / data.nz)))
          : (j * Math.PI) / data.nz,
      )
    : [];
  const zl = isPolar
    ? []
    : (data.zNodes ?? linspace(data.zMin, data.zMax, data.nz + 1));
  const rMax = rl[rl.length - 1];
  const zMin = isPolar ? -rMax : zl[0];
  const zMax = isPolar ? rMax : zl[zl.length - 1];
  const xMin = 0;
  const xMax = rMax;
  const xSpan = xMax - xMin || 1;
  const ySpan = zMax - zMin || 1;
  const innerW = WIDTH - PLOT_L - PLOT_R;
  const innerH = HEIGHT - PLOT_T - PLOT_B;
  const scale = Math.min(innerW / xSpan, innerH / ySpan);
  const x = (r: number) => PLOT_L + (r - xMin) * scale + (innerW - xSpan * scale) / 2;
  const y = (z: number) => PLOT_T + (zMax - z) * scale + (innerH - ySpan * scale) / 2;
  const rTicks = niceTicks(xMin, xMax);
  const zTicks = niceTicks(zMin, zMax);
  const useMicrons = Math.max(Math.abs(xSpan), Math.abs(ySpan)) < 0.1;
  const axisUnit = useMicrons ? "µm" : "cm";
  const formatTick = (value: number) =>
    useMicrons
      ? `${Number((value * 1e4).toFixed(0))}µm`
      : String(Number(value.toPrecision(3)));
  const frameLeft = x(xMin);
  const frameRight = x(xMax);
  const frameTop = y(zMax);
  const frameBottom = y(zMin);
  const decimated = (data.nr + 1) * (data.nz + 1) > 40000;
  const rStride = decimated ? Math.max(1, Math.ceil((data.nr + 1) / 200)) : 1;
  const zStride = decimated ? Math.max(1, Math.ceil((data.nz + 1) / 200)) : 1;
  const rIndices = sampledIndices(data.nr + 1, rStride);
  const zIndices = sampledIndices(data.nz + 1, zStride);

  return (
    <div className="mt-3 flex flex-col gap-2">
      {data.dim === 2 && (
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          width={WIDTH}
          height={HEIGHT}
          style={{ background: "#0b0e14", borderRadius: 8 }}
        >
          <rect
            x={frameLeft}
            y={frameTop}
            width={frameRight - frameLeft}
            height={frameBottom - frameTop}
            stroke={GRID_STROKE}
            fill="none"
          />
          {rTicks.map((value) => (
            <g key={`r-tick-${value}`}>
              <line
                x1={x(value)}
                y1={frameBottom}
                x2={x(value)}
                y2={frameBottom + 4}
                stroke={AXIS_STROKE}
              />
              <text
                x={x(value)}
                y={frameBottom + 14}
                fill={AXIS_STROKE}
                fontSize={10}
                textAnchor="middle"
              >
                {formatTick(value)}
              </text>
            </g>
          ))}
          {zTicks.map((value) => (
            <g key={`z-tick-${value}`}>
              <line
                x1={frameLeft - 4}
                y1={y(value)}
                x2={frameLeft}
                y2={y(value)}
                stroke={AXIS_STROKE}
              />
              <text
                x={frameLeft - 6}
                y={y(value) + 3}
                fill={AXIS_STROKE}
                fontSize={10}
                textAnchor="end"
              >
                {formatTick(value)}
              </text>
            </g>
          ))}
          <text
            x={(frameLeft + frameRight) / 2}
            y={HEIGHT - 4}
            fill={AXIS_STROKE}
            fontSize={10}
            textAnchor="middle"
          >
            R [{axisUnit}]
          </text>
          <text x={4} y={Math.max(10, frameTop - 4)} fill={AXIS_STROKE} fontSize={10}>
            Z [{axisUnit}]
          </text>
          {isPolar ? (
            <>
              {rIndices.map((i) => (
                <polyline
                  key={`radial-${i}`}
                  points={zIndices
                    .map((j) => {
                      const theta = thetaLadder[j];
                      return `${x(rl[i] * Math.sin(theta))},${y(rl[i] * Math.cos(theta))}`;
                    })
                    .join(" ")}
                  stroke={GRID_STROKE}
                  strokeWidth={1}
                  fill="none"
                />
              ))}
              {zIndices.map((j) => {
                const theta = thetaLadder[j];
                return (
                  <line
                    key={`theta-${j}`}
                    x1={x(rl[0] * Math.sin(theta))}
                    y1={y(rl[0] * Math.cos(theta))}
                    x2={x(rMax * Math.sin(theta))}
                    y2={y(rMax * Math.cos(theta))}
                    stroke={GRID_STROKE}
                    strokeWidth={1}
                    fill="none"
                  />
                );
              })}
            </>
          ) : (
            <>
              {rIndices.map((i) => (
                <line
                  key={`r-${i}`}
                  x1={x(rl[i])}
                  y1={y(zMin)}
                  x2={x(rl[i])}
                  y2={y(zMax)}
                  stroke={GRID_STROKE}
                  strokeWidth={1}
                  fill="none"
                />
              ))}
              {zIndices.map((j) => (
                <line
                  key={`z-${j}`}
                  x1={x(0)}
                  y1={y(zl[j])}
                  x2={x(rMax)}
                  y2={y(zl[j])}
                  stroke={GRID_STROKE}
                  strokeWidth={1}
                  fill="none"
                />
              ))}
            </>
          )}
          {form.geometry.shapes2d.map((shape, i) => {
            const resolved = resolveShape(shape);
            const outlines = shapeOutlinePoints(resolved);
            if (outlines === null) return null;
            const svgPoints = (points: Array<[number, number]>) =>
              points.map(([r, z]) => `${x(r)},${y(z)}`).join(" ");
            if (resolved.kind === "shell") {
              return (
                <g key={`shape-${i}`}>
                  {outlines.map((points, outlineIndex) => (
                    <polyline
                      key={`shape-${i}-${outlineIndex}`}
                      points={svgPoints(points)}
                      stroke={INTERFACE_STROKE}
                      strokeWidth={1.5}
                      strokeDasharray="6 4"
                      fill="none"
                    />
                  ))}
                </g>
              );
            }
            return (
              <polyline
                key={`shape-${i}`}
                points={svgPoints(outlines[0])}
                stroke={INTERFACE_STROKE}
                strokeWidth={1.5}
                strokeDasharray="6 4"
                fill="none"
              />
            );
          })}
        </svg>
      )}
      <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
        {m.meshPreview.cells}: {data.nr}×{data.nz} · {m.meshPreview.radialNodes}:{" "}
        {data.rNodes !== null ? m.meshPreview.resolvedBadge : m.meshPreview.uniformBadge}
        {isPolar && ` · ${m.meshPreview.thetaCount}: ${data.nz}`}
        {decimated && " (decimated)"}
      </p>
      {crossCheck}
    </div>
  );
}
