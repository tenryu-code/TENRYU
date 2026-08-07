import { useRef, useState } from "react";
import type { WaveformPoint } from "../core/deck/formState";
import { t } from "../i18n";
import { Button } from "@tenryu-common/ui/kit";

export interface WaveBounds {
  xMax: number;
  yMax: number;
}

export function waveBounds(points: WaveformPoint[]): WaveBounds {
  const xMax = Math.max(1e-9, ...points.map((p) => p.t)) * 1.05;
  const yMax = Math.max(1e-9, ...points.map((p) => p.v)) * 1.15;
  return { xMax, yMax };
}

export function dataToSvg(p: WaveformPoint, b: WaveBounds, w: number, h: number): { x: number; y: number } {
  return { x: (p.t / b.xMax) * w, y: h - (p.v / b.yMax) * h };
}

export function svgToData(x: number, y: number, b: WaveBounds, w: number, h: number): WaveformPoint {
  return { t: (x / w) * b.xMax, v: ((h - y) / h) * b.yMax };
}

const W = 380;
const H = 170;

export default function WaveformEditor({
  points,
  onChange,
  yLabel,
  importHint,
}: {
  points: WaveformPoint[];
  onChange: (p: WaveformPoint[]) => void;
  yLabel: string;
  importHint?: string;
}) {
  const m = t();
  const svgRef = useRef<SVGSVGElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragIdx, setDragIdx] = useState<number | null>(null);
  const [importError, setImportError] = useState("");
  const b = waveBounds(points);

  const clientToLocal = (e: { clientX: number; clientY: number }) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return null;
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const moveTo = (idx: number, e: { clientX: number; clientY: number }) => {
    const loc = clientToLocal(e);
    if (!loc) return;
    const d = svgToData(Math.max(0, Math.min(W, loc.x)), Math.max(0, Math.min(H, loc.y)), b, W, H);
    const next = points.map((p, i) => {
      if (i !== idx) return p;
      const lo = i > 0 ? points[i - 1].t + b.xMax * 1e-4 : 0;
      const hi = i < points.length - 1 ? points[i + 1].t - b.xMax * 1e-4 : b.xMax;
      return {
        t: Number(Math.max(lo, Math.min(hi, d.t)).toPrecision(6)),
        v: Number(Math.max(0, d.v).toPrecision(6)),
      };
    });
    onChange(next);
  };

  const addAt = (e: React.MouseEvent<SVGSVGElement>) => {
    if (dragIdx !== null || e.target !== svgRef.current) return;
    const loc = clientToLocal(e);
    if (!loc) return;
    const d = svgToData(loc.x, loc.y, b, W, H);
    const next = [...points, { t: Number(d.t.toPrecision(6)), v: Number(Math.max(0, d.v).toPrecision(6)) }].sort(
      (a, c) => a.t - c.t,
    );
    onChange(next);
  };

  const removeAt = (idx: number) => {
    if (points.length <= 2) return;
    onChange(points.filter((_, i) => i !== idx));
  };

  const setRow = (idx: number, key: "t" | "v", value: number) => {
    const next = points.map((p, i) => (i === idx ? { ...p, [key]: value } : p));
    onChange(next);
  };

  const importFile = (file: File, input: HTMLInputElement) => {
    const reader = new FileReader();
    reader.onload = () => {
      const parsed: WaveformPoint[] = [];
      const lines = String(reader.result ?? "").split(/\r?\n/);
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].split("#", 1)[0].trim();
        if (line.length === 0) continue;
        const fields = line.split(/[\s,]+/);
        if (fields.length < 2 || !Number.isFinite(Number(fields[0])) || !Number.isFinite(Number(fields[1]))) {
          setImportError(`${m.form.wfImportError}: ${i + 1}`);
          input.value = "";
          return;
        }
        parsed.push({ t: Number(fields[0]), v: Number(fields[1]) });
      }
      if (parsed.length < 2) {
        setImportError(m.form.wfImportError);
        input.value = "";
        return;
      }
      onChange(parsed);
      setImportError("");
      input.value = "";
    };
    reader.onerror = () => {
      setImportError(m.form.wfImportError);
      input.value = "";
    };
    reader.readAsText(file);
  };

  const poly = points.map((p) => {
    const s = dataToSvg(p, b, W, H);
    return `${s.x},${s.y}`;
  });

  const gridYs = [0.25, 0.5, 0.75];

  return (
    <div className="mb-2">
      <div className="mb-1 text-xs" style={{ color: "var(--fg-secondary)" }}>
        {yLabel} — {m.form.waveformHint}
      </div>
      <svg
        ref={svgRef}
        width={W}
        height={H}
        onClick={addAt}
        onPointerMove={(e) => {
          if (dragIdx !== null) moveTo(dragIdx, e);
        }}
        onPointerUp={() => setDragIdx(null)}
        onPointerLeave={() => setDragIdx(null)}
        style={{
          background: "var(--bg-inset)",
          border: "1px solid var(--separator)",
          borderRadius: 6,
          cursor: dragIdx !== null ? "grabbing" : "crosshair",
          touchAction: "none",
        }}
      >
        {gridYs.map((g) => (
          <line key={g} x1={0} x2={W} y1={H * g} y2={H * g} stroke="var(--separator)" strokeWidth={1} />
        ))}
        <polyline points={poly.join(" ")} fill="none" stroke="var(--accent)" strokeWidth={2} />
        {points.map((p, i) => {
          const s = dataToSvg(p, b, W, H);
          return (
            <circle
              key={i}
              cx={s.x}
              cy={s.y}
              r={5}
              fill="var(--accent)"
              stroke="var(--bg-panel)"
              strokeWidth={1.5}
              style={{ cursor: "grab" }}
              onPointerDown={(e) => {
                e.stopPropagation();
                (e.target as Element).setPointerCapture?.(e.pointerId);
                setDragIdx(i);
              }}
              onDoubleClick={(e) => {
                e.stopPropagation();
                removeAt(i);
              }}
            />
          );
        })}
        <text x={4} y={12} fontSize={10} fill="var(--fg-secondary)">
          {Number(b.yMax.toPrecision(3))}
        </text>
        <text x={W - 4} y={H - 4} fontSize={10} textAnchor="end" fill="var(--fg-secondary)">
          {Number(b.xMax.toPrecision(3))} ns
        </text>
      </svg>
      <table className="mt-1 text-xs" style={{ fontFamily: "var(--mono)" }}>
        <thead>
          <tr style={{ color: "var(--fg-secondary)" }}>
            <th className="pr-2 text-left">{m.form.pointT}</th>
            <th className="pr-2 text-left">{yLabel}</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {points.map((p, i) => (
            <tr key={i}>
              <td className="pr-2">
                <input
                  type="number"
                  step="any"
                  value={p.t}
                  onChange={(e) => setRow(i, "t", Number(e.target.value))}
                  style={{
                    width: 90,
                    background: "var(--bg-panel)",
                    border: "1px solid var(--separator)",
                    borderRadius: 4,
                    color: "var(--fg)",
                    padding: "1px 4px",
                  }}
                />
              </td>
              <td className="pr-2">
                <input
                  type="number"
                  step="any"
                  value={p.v}
                  onChange={(e) => setRow(i, "v", Number(e.target.value))}
                  style={{
                    width: 90,
                    background: "var(--bg-panel)",
                    border: "1px solid var(--separator)",
                    borderRadius: 4,
                    color: "var(--fg)",
                    padding: "1px 4px",
                  }}
                />
              </td>
              <td>
                <button
                  onClick={() => removeAt(i)}
                  disabled={points.length <= 2}
                  className="rounded px-1 disabled:opacity-30"
                  style={{ color: "var(--err)" }}
                >
                  ×
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-1 flex items-center gap-2">
        <Button
          onClick={() => {
            const last = points[points.length - 1];
            onChange([...points, { t: Number((last.t + 0.5).toPrecision(6)), v: last.v }]);
          }}
        >
          {m.form.addPoint}
        </Button>
        <Button title={importHint ?? m.form.wfImportHint} onClick={() => fileInputRef.current?.click()}>
          {m.form.wfImport}
        </Button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.txt,.dat"
          className="hidden"
          onChange={(e) => {
            const file = e.currentTarget.files?.[0];
            if (file) importFile(file, e.currentTarget);
            else e.currentTarget.value = "";
          }}
        />
      </div>
      {importError && <p className="text-xs" style={{ color: "var(--err)" }}>{importError}</p>}
    </div>
  );
}
