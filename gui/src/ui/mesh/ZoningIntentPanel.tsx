import { Button, NumberInput } from "@tenryu-common/ui/kit";
import { t } from "../../i18n";
import { massMeasureFor, type ZoningIntentForm, type ZoningMeasure } from "../../core/deck/mesh1d";
import { useApp } from "../../store";
import { NumInput, SelectField, SwitchField } from "../fields";

type Column<T> = {
  label: string;
  get: (row: T) => number | boolean | null;
  set: (row: T, value: number | boolean | null) => T;
  kind?: "number" | "bool" | "optional";
};

/** Editable list of rows with numeric (or boolean) columns. */
function RowList<T>({
  title,
  rows,
  columns,
  make,
  onChange,
}: {
  title: string;
  rows: T[];
  columns: Column<T>[];
  make: () => T;
  onChange: (rows: T[]) => void;
}) {
  const m = t().mesh1d.ui;
  return (
    <div className="mt-1">
      <div className="text-xs font-medium">{title}</div>
      {rows.length > 0 && (
        <table className="text-xs">
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c.label} className="pr-2 text-left font-normal" style={{ color: "var(--fg-secondary)" }}>{c.label}</th>
              ))}
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {columns.map((c) => {
                  const value = c.get(row);
                  if (c.kind === "bool") {
                    return (
                      <td key={c.label} className="pr-2">
                        <input
                          type="checkbox"
                          checked={value === true}
                          onChange={(e) => onChange(rows.map((r, j) => (j === i ? c.set(r, e.target.checked) : r)))}
                        />
                      </td>
                    );
                  }
                  return (
                    <td key={c.label} className="pr-2">
                      <NumberInput
                        step="any"
                        value={value === null || typeof value === "boolean" || !Number.isFinite(value) ? "" : value}
                        onChange={(e) => {
                          const raw = e.target.value;
                          const next = raw === "" ? (c.kind === "optional" ? null : Number.NaN) : Number(raw);
                          onChange(rows.map((r, j) => (j === i ? c.set(r, next) : r)));
                        }}
                        style={{ width: "7.5rem" }}
                      />
                    </td>
                  );
                })}
                <td>
                  <Button onClick={() => onChange(rows.filter((_, j) => j !== i))}>{m.remove}</Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <Button variant="secondary" onClick={() => onChange([...rows, make()])}>{m.add}</Button>
    </div>
  );
}

const UM = 1.0e-4;

/** Direct editor of Mesh.zoning_intent (SPECIFICATION 6.4.2); the solver computes the nodes. */
export default function ZoningIntentPanel() {
  const m = t().mesh1d.ui;
  const form = useApp((s) => s.form);
  const update = useApp((s) => s.updateForm);
  const z = form.mesh.zoningIntent;
  const set = (change: Partial<ZoningIntentForm>) =>
    update((f) => {
      f.mesh.zoningIntent = { ...f.mesh.zoningIntent, ...change };
      if (f.mesh.recommendation !== null) f.mesh.recommendation = { ...f.mesh.recommendation, edited: true };
    });
  const geometry = form.main.geometry1d;
  const candidates: ZoningMeasure[] = ["width", "areal_mass", massMeasureFor(geometry)];
  const measures = candidates.filter((value, index) => candidates.indexOf(value) === index);
  const unit = m.measureUnit[z.measure];
  const rUm = (label: string) => `${label} [µm]`;
  return (
    <div className="flex flex-col gap-1">
      <NumInput int label={m.intentCells} value={z.nCells} onChange={(n) => set({ nCells: n ?? 0 })} />
      <SelectField
        label={m.intentMeasure}
        value={z.measure}
        options={measures.map((value) => ({ value, label: m.measureNames[value] }))}
        onChange={(v) => set({ measure: v as ZoningMeasure })}
      />
      <SwitchField
        label={m.intentDensityAuto}
        checked={z.densityRegions.length === 0}
        onChange={(auto) => set({ densityRegions: auto ? [] : [{ rEndCm: Number.NaN, rho: 1 }] })}
      />
      {z.densityRegions.length > 0 && (
        <RowList
          title={m.intentDensityRegions}
          rows={z.densityRegions}
          columns={[
            { label: rUm(m.colREnd), get: (r) => r.rEndCm / UM, set: (r, v) => ({ ...r, rEndCm: (v as number) * UM }) },
            { label: m.colRho, get: (r) => r.rho, set: (r, v) => ({ ...r, rho: v as number }) },
          ]}
          make={() => ({ rEndCm: Number.NaN, rho: 1 })}
          onChange={(rows) => set({ densityRegions: rows })}
        />
      )}
      <SwitchField label={m.intentPinInterfaces} checked={z.pinInterfaces} onChange={(b) => set({ pinInterfaces: b })} />
      <RowList
        title={m.intentPins}
        rows={z.pins}
        columns={[
          { label: rUm(m.colR), get: (r) => r.rCm / UM, set: (r, v) => ({ ...r, rCm: (v as number) * UM }) },
          { label: m.colJump, kind: "bool", get: (r) => r.ratioJumpAllowed, set: (r, v) => ({ ...r, ratioJumpAllowed: v === true }) },
        ]}
        make={() => ({ rCm: Number.NaN, ratioJumpAllowed: true })}
        onChange={(rows) => set({ pins: rows })}
      />
      <RowList
        title={m.intentProfile(unit)}
        rows={z.profile}
        columns={[
          { label: rUm(m.colR), get: (r) => r.rCm / UM, set: (r, v) => ({ ...r, rCm: (v as number) * UM }) },
          { label: m.colWeight, get: (r) => r.w, set: (r, v) => ({ ...r, w: v as number }) },
        ]}
        make={() => ({ rCm: Number.NaN, w: 1 })}
        onChange={(rows) => set({ profile: rows })}
      />
      <RowList
        title={m.intentAnchors}
        rows={z.anchors}
        columns={[
          { label: rUm(m.colR), get: (r) => r.rCm / UM, set: (r, v) => ({ ...r, rCm: (v as number) * UM }) },
          { label: rUm(m.colHalfWidth), get: (r) => r.halfWidthCm / UM, set: (r, v) => ({ ...r, halfWidthCm: (v as number) * UM }) },
          { label: m.colLogAmplitude, get: (r) => r.logAmplitude, set: (r, v) => ({ ...r, logAmplitude: v as number }) },
        ]}
        make={() => ({ rCm: Number.NaN, halfWidthCm: Number.NaN, logAmplitude: -1 })}
        onChange={(rows) => set({ anchors: rows })}
      />
      <RowList
        title={m.intentBands(unit)}
        rows={z.bands}
        columns={[
          { label: m.colBegin, get: (r) => r.fracBegin, set: (r, v) => ({ ...r, fracBegin: v as number }) },
          { label: m.colEnd, get: (r) => r.fracEnd, set: (r, v) => ({ ...r, fracEnd: v as number }) },
          { label: m.colMin, kind: "optional", get: (r) => r.cellMeasureMin, set: (r, v) => ({ ...r, cellMeasureMin: v as number | null }) },
          { label: m.colMax, kind: "optional", get: (r) => r.cellMeasureMax, set: (r, v) => ({ ...r, cellMeasureMax: v as number | null }) },
        ]}
        make={() => ({ fracBegin: 0.9, fracEnd: 1, cellMeasureMin: null, cellMeasureMax: null })}
        onChange={(rows) => set({ bands: rows })}
      />
      <RowList
        title={m.intentEvents}
        rows={z.extraEventsCm.map((r) => ({ r }))}
        columns={[{ label: rUm(m.colR), get: (row) => row.r / UM, set: (_row, v) => ({ r: (v as number) * UM }) }]}
        make={() => ({ r: Number.NaN })}
        onChange={(rows) => set({ extraEventsCm: rows.map((row) => row.r) })}
      />
      <NumInput
        allowEmpty
        label={m.intentDrMinUm}
        value={z.drMinCm === null ? null : z.drMinCm / UM}
        onChange={(n) => set({ drMinCm: n === null ? null : n * UM })}
      />
      <NumInput allowEmpty label={m.intentCellMeasureMin(unit)} value={z.cellMeasureMin} onChange={(n) => set({ cellMeasureMin: n })} />
      <NumInput allowEmpty label={m.intentCellMeasureMax(unit)} value={z.cellMeasureMax} onChange={(n) => set({ cellMeasureMax: n })} />
      <NumInput allowEmpty label={m.intentPreferredRatio} value={z.preferredRatio} onChange={(n) => set({ preferredRatio: n })} />
      <NumInput allowEmpty label={m.intentHardRatio} value={z.ratioHardMax} onChange={(n) => set({ ratioHardMax: n })} />
      <NumInput allowEmpty int label={m.intentMinCells} value={z.minCellsPerSegment} onChange={(n) => set({ minCellsPerSegment: n })} />
    </div>
  );
}
