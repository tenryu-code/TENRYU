import { t } from "../../i18n";
import {
  computeLayerNodes,
  meshLayers1d,
  resolvedLayerZoning,
  type LayerZoningForm,
  type MeshLayer1d,
} from "../../core/deck/mesh1d";
import { useApp } from "../../store";
import { NumInput, SelectField } from "../fields";

function fmt(x: number): string {
  return String(Number(x.toPrecision(4)));
}

export function layerTitle(layer: MeshLayer1d): string {
  const m = t().mesh1d.ui;
  if (layer.kind === "void") return m.layerVoid;
  if (layer.kind === "corona") return m.layerCorona(layer.materialName);
  return layer.materialName;
}

/** Per-layer table of the "layers" 1D mesh method: cell count and spacing of every region, the
 *  corona ramp and the void padding; Studio computes the nodes. */
export default function LayersPanel() {
  const m = t().mesh1d.ui;
  const form = useApp((s) => s.form);
  const update = useApp((s) => s.updateForm);
  const layers = meshLayers1d(form);
  if (layers === null) {
    return <p className="text-xs" style={{ color: "var(--err)" }}>{m.layersUnresolved}</p>;
  }
  const specs = resolvedLayerZoning(form, layers);
  const result = computeLayerNodes(form, layers);
  const setSpec = (index: number, change: Partial<LayerZoningForm>) =>
    update((f) => {
      const current = meshLayers1d(f);
      if (current === null) return;
      f.mesh.layerZoning = resolvedLayerZoning(f, current).map((spec, j) => (j === index ? { ...spec, ...change } : spec));
    });
  const total = specs.reduce((sum, spec) => sum + (Number.isFinite(spec.cells) ? spec.cells : 0), 0);
  return (
    <div className="flex flex-col gap-1">
      {layers.map((layer, i) => {
        const spec = specs[i];
        const nodes = result.perLayer[i];
        const error = result.errors.find((e) => e.layer === i);
        let info = "";
        if (nodes) {
          const widths = nodes.slice(1).map((b, k) => b - nodes[k]);
          const minWidth = Math.min(...widths);
          const n = widths.length;
          const endRatio =
            n >= 2
              ? spec.fineSide === "inner" || spec.fineSide === "both"
                ? widths[1] / widths[0]
                : widths[n - 2] / widths[n - 1]
              : 1;
          info = m.layerInfo(fmt(minWidth * 1.0e4), fmt(endRatio));
        }
        return (
          <div key={i} className="mb-1 rounded border p-2" style={{ borderColor: "var(--separator)" }}>
            <div className="text-xs font-medium">
              {i + 1}. {layerTitle(layer)} — {m.layerRange(fmt(layer.rLoCm * 1.0e4), fmt(layer.rHiCm * 1.0e4))}
            </div>
            <NumInput int label={m.cells} value={spec.cells} onChange={(n) => setSpec(i, { cells: n ?? 0 })} />
            <SelectField
              label={m.spacing}
              value={spec.spacing}
              options={[
                { value: "width", label: m.spacingWidth },
                { value: "mass", label: m.spacingMass },
                { value: "ratio", label: m.spacingRatio },
              ]}
              onChange={(v) => setSpec(i, { spacing: v as LayerZoningForm["spacing"] })}
            />
            {spec.spacing === "ratio" && (
              <>
                <SelectField
                  label={m.fineSide}
                  value={spec.fineSide}
                  options={[
                    { value: "outer", label: m.fineOuter },
                    { value: "inner", label: m.fineInner },
                    { value: "both", label: m.fineBoth },
                  ]}
                  onChange={(v) => setSpec(i, { fineSide: v as LayerZoningForm["fineSide"] })}
                />
                <SelectField
                  label={m.ratioSpec}
                  value={spec.ratioSpec}
                  options={[
                    { value: "ratio", label: m.ratioSpecRatio },
                    { value: "first_width", label: m.ratioSpecFirst },
                    { value: "match_mass", label: m.ratioSpecMatch },
                  ]}
                  onChange={(v) => setSpec(i, { ratioSpec: v as LayerZoningForm["ratioSpec"] })}
                />
                {spec.ratioSpec === "ratio" && (
                  <NumInput label={m.ratio} value={spec.ratio} onChange={(n) => setSpec(i, { ratio: n ?? Number.NaN })} />
                )}
                {spec.ratioSpec === "first_width" && (
                  <NumInput
                    label={m.firstWidthUm}
                    value={spec.firstWidthCm * 1.0e4}
                    onChange={(n) => setSpec(i, { firstWidthCm: (n ?? Number.NaN) * 1.0e-4 })}
                  />
                )}
              </>
            )}
            {info !== "" && (
              <p className="text-xs" style={{ color: "var(--fg-secondary)", fontFamily: "var(--mono)" }}>{info}</p>
            )}
            {error !== undefined && (
              <p className="text-xs" style={{ color: "var(--err)" }}>{t().mesh1d.layerErrors[error.code]}</p>
            )}
          </div>
        );
      })}
      <p className="text-xs">{m.totalCells(total)}</p>
    </div>
  );
}
