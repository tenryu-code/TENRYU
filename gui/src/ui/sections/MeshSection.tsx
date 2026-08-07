import { t } from "../../i18n";
import { ensureBackgroundGas } from "../../core/deck/formState";
import { computeRegionSegments } from "../../core/deck/meshAuto";
import { defaultShape2D, type Shape2DKind } from "../../core/geometry2d";
import { q } from "../../core/units";
import { useApp } from "../../store";
import { NumInput, QInput, SelectField, SwitchField, TextField } from "../fields";
import { Button } from "@tenryu-common/ui/kit";
import InitialProfile2D from "../InitialProfile2D";
import MeshMassChart from "../MeshMassChart";
import MeshPreview2D from "../MeshPreview2D";

export default function MeshSection() {
  const m = t();
  const form = useApp((s) => s.form);
  const update = useApp((s) => s.updateForm);
  const auto = computeRegionSegments(form);
  return (
    <div className="max-w-xl flex flex-col gap-1">
      <h1 className="mb-2 text-base font-semibold">{m.nav.mesh}</h1>
      <QInput
        label={m.form.rMin}
        kind="length"
        value={form.mesh.rMin}
        onChange={(q) => update((f) => { f.mesh.rMin = q; })}
      />
      <QInput
        label={m.form.rMax}
        kind="length"
        value={form.mesh.rMax}
        onChange={(q) => update((f) => { f.mesh.rMax = q; })}
      />
      {form.main.dimension === "2D_RZ" && (
        <>
          <QInput
            label={m.form.zMin}
            kind="length"
            value={form.mesh.zMin}
            onChange={(q) => update((f) => { f.mesh.zMin = q; })}
          />
          <QInput
            label={m.form.zMax}
            kind="length"
            value={form.mesh.zMax}
            onChange={(q) => update((f) => { f.mesh.zMax = q; })}
          />
          <SelectField
            label={m.form.logicalMesh2d}
            value={form.mesh.meshMode2d}
            options={[
              { value: "rect", label: m.geo2d.meshModeRect },
              { value: "polar_in_box", label: m.geo2d.meshModePib },
            ]}
            onChange={(v) => update((f) => { f.mesh.meshMode2d = v as never; })}
          />
          {form.mesh.meshMode2d === "rect" && (
            <NumInput
              int
              label={m.form.nz}
              value={form.mesh.nz}
              onChange={(n) => update((f) => { f.mesh.nz = n ?? 0; })}
            />
          )}
        </>
      )}
      {form.main.dimension === "1D_SPH" && (
        <SelectField
          label={m.form.gridType}
          value={form.mesh.grid1d}
          options={[
            { value: "uniform", label: m.form.gridUniform },
            { value: "graded", label: m.form.gridGraded },
          ]}
          onChange={(v) => update((f) => { f.mesh.grid1d = v as never; })}
        />
      )}
      {(form.mesh.grid1d === "uniform" || form.main.dimension === "2D_RZ") &&
        !(
          form.main.dimension === "2D_RZ" &&
          (form.mesh.meshMode2d === "polar_in_box" || form.mesh.radialZoning2d === "regions")
        ) && (
        <NumInput
          int
          label={m.form.nr}
          value={form.mesh.nr}
          onChange={(n) => update((f) => { f.mesh.nr = n ?? 0; })}
        />
      )}
      {form.main.dimension === "2D_RZ" && form.mesh.meshMode2d === "rect" && (
        <SelectField
          label={m.form.radialZoning2d}
          value={form.mesh.radialZoning2d}
          options={[
            { value: "uniform", label: m.geo2d.zoningUniform },
            { value: "regions", label: m.geo2d.zoningAuto },
          ]}
          onChange={(v) => update((f) => { f.mesh.radialZoning2d = v as never; })}
        />
      )}
      {form.main.dimension === "2D_RZ" && form.mesh.meshMode2d === "polar_in_box" && (
        <>
          <NumInput
            int
            label={m.geo2d.pibNTheta}
            value={form.mesh.pibNTheta}
            onChange={(n) => update((f) => { f.mesh.pibNTheta = n ?? 0; })}
          />
          <NumInput
            int
            label={m.geo2d.pibNRadial}
            value={form.mesh.pibNRadial}
            onChange={(n) => update((f) => { f.mesh.pibNRadial = n ?? 0; })}
          />
          <NumInput
            int
            label={m.geo2d.pibMorphRings}
            value={form.mesh.pibMorphRings}
            onChange={(n) => update((f) => { f.mesh.pibMorphRings = n ?? 0; })}
          />
          <NumInput
            int
            label={m.geo2d.pibCollarRings}
            value={form.mesh.pibCollarRings}
            onChange={(n) => update((f) => { f.mesh.pibCollarRings = n ?? 0; })}
          />
          <NumInput
            label={m.geo2d.pibMorphGrowthMax}
            value={form.mesh.pibMorphGrowthMax}
            onChange={(n) => update((f) => { f.mesh.pibMorphGrowthMax = n ?? 0; })}
          />
          <NumInput
            int
            label={m.geo2d.pibTailRings}
            value={form.mesh.pibTailRings}
            onChange={(n) => update((f) => { f.mesh.pibTailRings = n ?? 0; })}
          />
          <NumInput
            label={m.geo2d.pibTailRatio}
            value={form.mesh.pibTailRatio}
            onChange={(n) => update((f) => { f.mesh.pibTailRatio = n ?? 0; })}
          />
          <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
            {m.geo2d.pibRunGateNote}
          </p>
        </>
      )}
      {form.main.dimension === "1D_SPH" && form.mesh.grid1d === "graded" && (
        <>
          <h2 className="mt-2 text-sm font-semibold">{m.form.segments}</h2>
          <SelectField
            label={m.form.meshSegSource}
            value={form.mesh.segmentSource}
            options={[
              { value: "manual", label: m.form.meshSegSourceManual },
              { value: "regions", label: m.form.meshSegSourceRegions },
            ]}
            onChange={(v) => update((f) => { f.mesh.segmentSource = v as never; })}
          />
          {form.mesh.segmentSource === "manual" && (
            <>
              {form.mesh.segments.map((seg, i) => (
                <div className="flex items-end gap-2" key={i}>
                  <QInput
                    label={`${m.form.segmentEnd} ${i + 1}`}
                    kind="length"
                    value={seg.rEnd}
                    onChange={(q) => update((f) => { f.mesh.segments[i].rEnd = q; })}
                  />
                  <NumInput
                    int
                    label={m.form.segmentNr}
                    value={seg.nr}
                    onChange={(n) => update((f) => { f.mesh.segments[i].nr = n ?? 0; })}
                  />
                  <Button onClick={() => update((f) => { f.mesh.segments.splice(i, 1); })}>削除</Button>
                </div>
              ))}
              <Button
                variant="secondary"
                onClick={() => update((f) => { f.mesh.segments.push({ rEnd: structuredClone(f.mesh.rMax), nr: 50 }); })}
              >
                {m.form.addSegment}
              </Button>
              <p className="text-xs">{m.form.totalNr}: {form.mesh.segments.reduce((a, s) => a + s.nr, 0)}</p>
            </>
          )}
          {form.mesh.segmentSource === "regions" && (
            <>
              <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.form.meshAutoNote}</p>
              {auto === null ? (
                <p className="text-xs">{m.form.meshAutoUnresolved}</p>
              ) : (
                <>
                  <table className="text-xs">
                    <thead>
                      <tr>
                        <th>{m.form.meshAutoRegionCol}</th>
                        <th>{m.form.meshAutoNrCol}</th>
                        <th>{m.form.meshAutoOverrideCol}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {form.geometry.regions.map((region, i) => (
                        <tr key={i}>
                          <td>{region.materialName}</td>
                          <td>{auto[i].nr}</td>
                          <td>
                            <NumInput
                              int
                              allowEmpty
                              label=""
                              value={form.mesh.regionNrOverrides[i] ?? null}
                              onChange={(n) => update((f) => { f.mesh.regionNrOverrides[i] = n; })}
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="text-xs">{m.form.totalNr}: {auto.reduce((a, s) => a + s.nr, 0)}</p>
                </>
              )}
            </>
          )}
          <details>
            <summary>{m.form.gradingTitle}</summary>
            <NumInput
              label={m.form.edgeRatio}
              value={form.mesh.grading.edgeRatio}
              onChange={(n) => update((f) => { f.mesh.grading.edgeRatio = n ?? 0; })}
            />
            <NumInput
              int
              label={m.form.sgOrder}
              value={form.mesh.grading.sgOrder}
              onChange={(n) => update((f) => { f.mesh.grading.sgOrder = n ?? 0; })}
            />
            <NumInput
              label={m.form.sgSigma}
              value={form.mesh.grading.sgSigma}
              onChange={(n) => update((f) => { f.mesh.grading.sgSigma = n ?? 0; })}
            />
          </details>
        </>
      )}
      {form.main.dimension !== "2D_RZ" && (
        <>
          <h2 className="mt-3 text-sm font-semibold">{m.form.regionsTitle}</h2>
          {form.geometry.regions.map((region, i) => (
            <div key={i} className="mb-2 rounded border p-2" style={{ borderColor: "var(--separator)" }}>
              <SelectField
                label={m.form.regionMaterial}
                value={region.materialName}
                options={form.materials.map((mm) => ({ value: mm.name, label: mm.name }))}
                onChange={(v) => update((f) => { f.geometry.regions[i].materialName = v; })}
              />
              <QInput
                label={m.form.regionROuter}
                kind="length"
                value={region.rOuter}
                onChange={(q) => update((f) => { f.geometry.regions[i].rOuter = q; })}
              />
              <NumInput
                label={m.form.regionRho}
                value={region.rho}
                onChange={(n) => update((f) => { f.geometry.regions[i].rho = n ?? 0; })}
              />
              <QInput
                label={m.form.regionTe}
                kind="temperature"
                value={region.Te}
                onChange={(q) => update((f) => { f.geometry.regions[i].Te = q; })}
              />
              <QInput
                label={m.form.regionTi}
                kind="temperature"
                value={region.Ti}
                onChange={(q) => update((f) => { f.geometry.regions[i].Ti = q; })}
              />
              <Button
                variant="danger"
                disabled={form.geometry.regions.length === 1}
                onClick={() => update((f) => { f.geometry.regions.splice(i, 1); })}
              >
                削除
              </Button>
            </div>
          ))}
          <Button
            onClick={() => update((f) => {
              f.geometry.regions.push({
                materialName: f.materials[0].name,
                rOuter: structuredClone(f.mesh.rMax),
                rho: 1.0,
                Te: { value: 1, unit: "eV" },
                Ti: { value: 1, unit: "eV" },
              });
            })}
          >
            {m.form.addRegion}
          </Button>
          <SwitchField
            label={m.form.vacuumOutside1d}
            checked={form.geometry.vacuumOutside1d}
            onChange={(checked) => update((f) => { f.geometry.vacuumOutside1d = checked; })}
          />
          {form.geometry.vacuumOutside1d && (
            <>
              <SwitchField
                label={m.form.coronaRamp1d}
                checked={form.geometry.coronaRamp1d.enabled}
                onChange={(checked) => update((f) => { f.geometry.coronaRamp1d.enabled = checked; })}
              />
              {form.geometry.coronaRamp1d.enabled && (
                <>
                  <NumInput
                    label={m.form.coronaRampScaleUm}
                    value={form.geometry.coronaRamp1d.scaleUm}
                    onChange={(n) => update((f) => { f.geometry.coronaRamp1d.scaleUm = n ?? 0; })}
                  />
                  <NumInput
                    label={m.form.coronaRampExtentUm}
                    value={form.geometry.coronaRamp1d.extentUm}
                    onChange={(n) => update((f) => { f.geometry.coronaRamp1d.extentUm = n ?? 0; })}
                  />
                  <NumInput
                    label={m.form.coronaRampRho0}
                    value={form.geometry.coronaRamp1d.rho0}
                    onChange={(n) => update((f) => { f.geometry.coronaRamp1d.rho0 = n ?? 0; })}
                  />
                  <NumInput
                    label={m.form.coronaRampRhoMin}
                    value={form.geometry.coronaRamp1d.rhoMin}
                    onChange={(n) => update((f) => { f.geometry.coronaRamp1d.rhoMin = n ?? 0; })}
                  />
                </>
              )}
            </>
          )}
        </>
      )}
      <MeshMassChart form={form} />
      {form.main.dimension === "2D_RZ" && (
        <>
          <h2 className="mt-3 text-sm font-semibold">{m.geo2d.shapesTitle}</h2>
          <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.geo2d.orderHint}</p>
          {form.geometry.shapes2d.map((shape, i) => (
            <div
              key={i}
              className="mb-2 rounded border p-3"
              style={{ borderColor: "var(--separator)", background: "var(--bg-panel)" }}
            >
              <div className="flex flex-wrap items-start gap-2">
                <span className="py-1 text-sm font-semibold">#{i + 1}</span>
                <SelectField
                  label=""
                  value={shape.kind}
                  options={[
                    { value: "solidSphere", label: m.geo2d.kindSolidSphere },
                    { value: "shell", label: m.geo2d.kindShell },
                    { value: "block", label: m.geo2d.kindBlock },
                    { value: "cone", label: m.geo2d.kindCone },
                    { value: "polygon", label: m.geo2d.kindPolygon },
                  ]}
                  onChange={(v) => update((f) => {
                    const s = f.geometry.shapes2d[i];
                    s.kind = v as Shape2DKind;
                    if (s.kind === "polygon" && s.vertices.length < 3) {
                      s.vertices = defaultShape2D("polygon").vertices;
                    }
                  })}
                />
                <Button
                  disabled={i === 0}
                  onClick={() => update((f) => {
                    const shapes = f.geometry.shapes2d;
                    [shapes[i - 1], shapes[i]] = [shapes[i], shapes[i - 1]];
                  })}
                >
                  ↑
                </Button>
                <Button
                  disabled={i === form.geometry.shapes2d.length - 1}
                  onClick={() => update((f) => {
                    const shapes = f.geometry.shapes2d;
                    [shapes[i], shapes[i + 1]] = [shapes[i + 1], shapes[i]];
                  })}
                >
                  ↓
                </Button>
                <Button
                  variant="danger"
                  onClick={() => update((f) => { f.geometry.shapes2d.splice(i, 1); })}
                >
                  削除
                </Button>
              </div>
              <TextField
                label={m.geo2d.label}
                value={shape.label}
                onChange={(v) => update((f) => { f.geometry.shapes2d[i].label = v; })}
              />
              <SelectField
                label={m.form.regionMaterial}
                value={shape.materialName}
                options={form.materials.map((mm) => ({ value: mm.name, label: mm.name }))}
                onChange={(v) => update((f) => { f.geometry.shapes2d[i].materialName = v; })}
              />
              <NumInput
                label={m.form.regionRho}
                value={shape.rho}
                onChange={(n) => update((f) => { f.geometry.shapes2d[i].rho = n ?? 0; })}
              />
              <QInput
                label={m.form.regionTe}
                kind="temperature"
                value={shape.Te}
                onChange={(value) => update((f) => { f.geometry.shapes2d[i].Te = value; })}
              />
              <QInput
                label={m.form.regionTi}
                kind="temperature"
                value={shape.Ti}
                onChange={(value) => update((f) => { f.geometry.shapes2d[i].Ti = value; })}
              />
              {shape.kind === "solidSphere" && (
                <>
                  <QInput
                    label={m.geo2d.z0}
                    kind="length"
                    value={shape.z0}
                    onChange={(value) => update((f) => { f.geometry.shapes2d[i].z0 = value; })}
                  />
                  <QInput
                    label={m.geo2d.radius}
                    kind="length"
                    value={shape.radius}
                    onChange={(value) => update((f) => { f.geometry.shapes2d[i].radius = value; })}
                  />
                </>
              )}
              {shape.kind === "shell" && (
                <>
                  <QInput
                    label={m.geo2d.z0}
                    kind="length"
                    value={shape.z0}
                    onChange={(value) => update((f) => { f.geometry.shapes2d[i].z0 = value; })}
                  />
                  <QInput
                    label={m.geo2d.rIn}
                    kind="length"
                    value={shape.rIn}
                    onChange={(value) => update((f) => { f.geometry.shapes2d[i].rIn = value; })}
                  />
                  <QInput
                    label={m.geo2d.rOut}
                    kind="length"
                    value={shape.radius}
                    onChange={(value) => update((f) => { f.geometry.shapes2d[i].radius = value; })}
                  />
                </>
              )}
              {shape.kind === "block" && (
                <>
                  <QInput
                    label={m.geo2d.r0}
                    kind="length"
                    value={shape.r0}
                    onChange={(value) => update((f) => { f.geometry.shapes2d[i].r0 = value; })}
                  />
                  <QInput
                    label={m.geo2d.r1}
                    kind="length"
                    value={shape.r1}
                    onChange={(value) => update((f) => { f.geometry.shapes2d[i].r1 = value; })}
                  />
                  <QInput
                    label={m.geo2d.zLo}
                    kind="length"
                    value={shape.z0}
                    onChange={(value) => update((f) => { f.geometry.shapes2d[i].z0 = value; })}
                  />
                  <QInput
                    label={m.geo2d.zHi}
                    kind="length"
                    value={shape.z1}
                    onChange={(value) => update((f) => { f.geometry.shapes2d[i].z1 = value; })}
                  />
                </>
              )}
              {shape.kind === "cone" && (
                <>
                  <QInput
                    label={m.geo2d.zApex}
                    kind="length"
                    value={shape.zApex}
                    onChange={(value) => update((f) => { f.geometry.shapes2d[i].zApex = value; })}
                  />
                  <QInput
                    label={m.geo2d.zBase}
                    kind="length"
                    value={shape.zBase}
                    onChange={(value) => update((f) => { f.geometry.shapes2d[i].zBase = value; })}
                  />
                  <QInput
                    label={m.geo2d.baseRadius}
                    kind="length"
                    value={shape.baseRadius}
                    onChange={(value) => update((f) => { f.geometry.shapes2d[i].baseRadius = value; })}
                  />
                </>
              )}
              {shape.kind === "polygon" && (
                <>
                  {shape.vertices.map((vertex, vertexIndex) => (
                    <div className="flex items-end gap-2" key={vertexIndex}>
                      <QInput
                        label="r"
                        kind="length"
                        value={vertex.r}
                        onChange={(value) => update((f) => {
                          f.geometry.shapes2d[i].vertices[vertexIndex].r = value;
                        })}
                      />
                      <QInput
                        label="z"
                        kind="length"
                        value={vertex.z}
                        onChange={(value) => update((f) => {
                          f.geometry.shapes2d[i].vertices[vertexIndex].z = value;
                        })}
                      />
                      <Button
                        variant="danger"
                        onClick={() => update((f) => {
                          f.geometry.shapes2d[i].vertices.splice(vertexIndex, 1);
                        })}
                      >
                        削除
                      </Button>
                    </div>
                  ))}
                  <Button
                    onClick={() => update((f) => {
                      f.geometry.shapes2d[i].vertices.push({ r: q(0, "µm"), z: q(0, "µm") });
                    })}
                  >
                    {m.geo2d.addVertex}
                  </Button>
                </>
              )}
            </div>
          ))}
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => update((f) => {
              const shape = defaultShape2D("solidSphere");
              shape.materialName = f.materials[0]?.name ?? "";
              f.geometry.shapes2d.push(shape);
            })}>
              {m.geo2d.addSolidSphere}
            </Button>
            <Button onClick={() => update((f) => {
              const shape = defaultShape2D("shell");
              shape.materialName = f.materials[0]?.name ?? "";
              f.geometry.shapes2d.push(shape);
            })}>
              {m.geo2d.addShell}
            </Button>
            <Button onClick={() => update((f) => {
              const shape = defaultShape2D("block");
              shape.materialName = f.materials[0]?.name ?? "";
              f.geometry.shapes2d.push(shape);
            })}>
              {m.geo2d.addBlock}
            </Button>
            <Button onClick={() => update((f) => {
              const shape = defaultShape2D("cone");
              shape.materialName = f.materials[0]?.name ?? "";
              f.geometry.shapes2d.push(shape);
            })}>
              {m.geo2d.addCone}
            </Button>
            <Button onClick={() => update((f) => {
              const shape = defaultShape2D("polygon");
              shape.materialName = f.materials[0]?.name ?? "";
              f.geometry.shapes2d.push(shape);
            })}>
              {m.geo2d.addPolygon}
            </Button>
          </div>
          <h2 className="mt-3 text-sm font-semibold">{m.geo2d.backgroundTitle}</h2>
          <SelectField
            label={m.form.regionMaterial}
            value={form.geometry.background2d.materialName}
            options={form.materials.map((mm) => ({ value: mm.name, label: mm.name }))}
            onChange={(v) => update((f) => { f.geometry.background2d.materialName = v; })}
          />
          {(!form.geometry.background2d.materialName ||
            !form.materials.some((material) => material.name === form.geometry.background2d.materialName)) && (
            <Button onClick={() => update((f) => ensureBackgroundGas(f))}>
              {m.geo2d.createGasMaterial}
            </Button>
          )}
          <NumInput
            label={m.form.regionRho}
            value={form.geometry.background2d.rho}
            onChange={(n) => update((f) => { f.geometry.background2d.rho = n ?? 0; })}
          />
          <QInput
            label={m.form.regionTe}
            kind="temperature"
            value={form.geometry.background2d.Te}
            onChange={(value) => update((f) => { f.geometry.background2d.Te = value; })}
          />
          <QInput
            label={m.form.regionTi}
            kind="temperature"
            value={form.geometry.background2d.Ti}
            onChange={(value) => update((f) => { f.geometry.background2d.Ti = value; })}
          />
        </>
      )}
      {form.main.dimension === "2D_RZ" && <MeshPreview2D form={form} />}
      {form.main.dimension === "2D_RZ" && <InitialProfile2D form={form} />}
      <SelectField
        label={m.form.radiationField}
        value={form.geometry.radiationField}
        options={[
          { value: "equilibrium", label: m.form.radFieldEq },
          { value: "zero", label: m.form.radFieldZero },
        ]}
        onChange={(v) => update((f) => { f.geometry.radiationField = v as never; })}
      />
    </div>
  );
}
