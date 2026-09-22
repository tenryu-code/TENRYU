import { useState } from "react";
import { t } from "../../i18n";
import { currentProfile, useApp } from "../../store";
import { NumInput, QInput, SelectField, TextField } from "../fields";
import { Button } from "@tenryu-common/ui/kit";
import RemoteFileBrowser from "../RemoteFileBrowser";
import {
  coldReferenceFromInitialState,
  emptyColdReference,
  type ColdReferenceForm,
  type FormState,
} from "../../core/deck/formState";

/** The material's reference state in an immer-style draft, created empty on first edit. */
function coldReferenceDraft(f: FormState, i: number): ColdReferenceForm {
  const material = f.materials[i];
  if (!material.coldReference) material.coldReference = emptyColdReference();
  return material.coldReference;
}

export default function MaterialsSection() {
  const m = t();
  const form = useApp((s) => s.form);
  const update = useApp((s) => s.updateForm);
  const profile = useApp((s) => currentProfile(s));
  const [browse, setBrowse] = useState<{
    i: number;
    field: "eosFile" | "opacityFile";
  } | null>(null);
  const dirOf = (v: string): string => {
    const s = v.trim();
    const i = s.lastIndexOf("/");
    return s.startsWith("/") && i > 0 ? s.slice(0, i) : "";
  };
  const coldEquilibrium = form.hydro.inactiveCells === "cold_equilibrium";
  return (
    <div className="max-w-xl flex flex-col gap-1">
      <h1 className="mb-2 text-base font-semibold">{m.form.materialsTitle}</h1>
      {form.materials.map((material, i) => (
        <div key={i} className="mb-2 rounded border p-2" style={{ borderColor: "var(--separator)" }}>
          <TextField
            label={m.form.matName}
            value={material.name}
            onChange={(v) => update((f) => { f.materials[i].name = v; })}
          />
          <NumInput
            label={m.form.matA}
            value={material.A}
            onChange={(n) => update((f) => { f.materials[i].A = n ?? 0; })}
          />
          <NumInput
            label={m.form.matZ}
            value={material.Z}
            onChange={(n) => update((f) => { f.materials[i].Z = n ?? 0; })}
          />
          <SelectField
            label={m.form.matEosModel}
            value={material.eosModel}
            options={[
              { value: "ideal_gas", label: m.form.matEosIdealGas },
              { value: "tmat", label: m.form.matEosTmat },
            ]}
            onChange={(v) => update((f) => { f.materials[i].eosModel = v as "ideal_gas" | "tmat"; })}
          />
          {material.eosModel === "ideal_gas" ? (
            <>
              <NumInput
                label={m.form.matGamma}
                value={material.gamma}
                onChange={(n) => update((f) => { f.materials[i].gamma = n ?? 0; })}
              />
              <NumInput
                allowEmpty
                label={m.form.matCvOverride}
                value={material.cvEOverride ?? null}
                onChange={(n) => update((f) => { f.materials[i].cvEOverride = n === null ? undefined : n; })}
              />
            </>
          ) : (
            <div className="flex items-end gap-2">
              <div className="flex-1">
                <TextField
                  mono
                  label={m.form.matEosFile}
                  value={material.eosFile}
                  onChange={(v) => update((f) => { f.materials[i].eosFile = v; })}
                />
              </div>
              <Button
                disabled={profile === null}
                title={profile === null ? m.remoteFs.needProfile : ""}
                onClick={() => setBrowse({ i, field: "eosFile" })}
              >
                {m.remoteFs.browse}
              </Button>
            </div>
          )}
          {coldEquilibrium && (() => {
            const ref = material.coldReference ?? emptyColdReference();
            return (
              <div className="my-1 rounded border p-2" style={{ borderColor: "var(--separator)" }}>
                <div className="mb-1 text-xs font-semibold">{m.form.coldRefTitle}</div>
                <NumInput
                  label={m.form.coldRefRho}
                  value={ref.rhoGcc}
                  onChange={(n) => update((f) => { coldReferenceDraft(f, i).rhoGcc = n ?? Number.NaN; })}
                />
                <QInput
                  label={m.form.coldRefTe0}
                  kind="temperature"
                  value={ref.Te0}
                  onChange={(v) => update((f) => { coldReferenceDraft(f, i).Te0 = v; })}
                />
                <QInput
                  label={m.form.coldRefTi0}
                  kind="temperature"
                  value={ref.Ti0}
                  onChange={(v) => update((f) => { coldReferenceDraft(f, i).Ti0 = v; })}
                />
                <QInput
                  label={m.form.coldRefP0}
                  kind="pressure"
                  value={ref.P0}
                  onChange={(v) => update((f) => { coldReferenceDraft(f, i).P0 = v; })}
                />
                <QInput
                  label={m.form.coldRefK0}
                  kind="pressure"
                  value={ref.K0}
                  onChange={(v) => update((f) => { coldReferenceDraft(f, i).K0 = v; })}
                />
                <Button
                  onClick={() => update((f) => {
                    const initial = coldReferenceFromInitialState(f, f.materials[i].name);
                    const draft = coldReferenceDraft(f, i);
                    draft.rhoGcc = initial.rhoGcc;
                    draft.Te0 = initial.Te0;
                    draft.Ti0 = initial.Ti0;
                  })}
                >
                  {m.form.coldRefFromInitial}
                </Button>
                <p className="mt-1 text-xs" style={{ color: "var(--fg-secondary)" }}>{m.form.coldRefHint}</p>
              </div>
            );
          })()}
          <SelectField
            label={m.form.matOpacityModel}
            value={material.opacityModel}
            options={[
              { value: "constant", label: m.form.matOpacityConstant },
              { value: "tmat", label: m.form.matOpacityTmat },
            ]}
            onChange={(v) => update((f) => { f.materials[i].opacityModel = v as "constant" | "tmat"; })}
          />
          {material.opacityModel === "constant" ? (
            <>
              <NumInput
                label={m.form.matKappaA}
                value={material.kappaA}
                onChange={(n) => update((f) => { f.materials[i].kappaA = n ?? 0; })}
              />
              <NumInput
                label={m.form.matKappaS}
                value={material.kappaS}
                onChange={(n) => update((f) => { f.materials[i].kappaS = n ?? 0; })}
              />
            </>
          ) : (
            <div className="flex items-end gap-2">
              <div className="flex-1">
                <TextField
                  mono
                  label={m.form.matOpacityFile}
                  value={material.opacityFile}
                  onChange={(v) => update((f) => { f.materials[i].opacityFile = v; })}
                />
              </div>
              <Button
                disabled={profile === null}
                title={profile === null ? m.remoteFs.needProfile : ""}
                onClick={() => setBrowse({ i, field: "opacityFile" })}
              >
                {m.remoteFs.browse}
              </Button>
            </div>
          )}
          <Button
            variant="danger"
            disabled={form.materials.length === 1}
            onClick={() => update((f) => { f.materials.splice(i, 1); })}
          >
            削除
          </Button>
        </div>
      ))}
      <Button
        onClick={() => update((f) => {
          f.materials.push({
            name: `mat${f.materials.length + 1}`,
            A: 1.0,
            Z: 1.0,
            eosModel: "ideal_gas",
            gamma: 5 / 3,
            eosFile: "",
            opacityModel: "constant",
            kappaA: 1.0,
            kappaS: 0.0,
            opacityFile: "",
          });
        })}
      >
        {m.form.addMaterial}
      </Button>
      <p className="mt-3 text-xs" style={{ color: "var(--fg-secondary)" }}>
        {m.geo2d.regionsMovedNote}
      </p>
      {browse !== null && (
        <RemoteFileBrowser
          title={m.remoteFs.title}
          initialPath={dirOf(form.materials[browse.i]?.[browse.field] ?? "")}
          onPick={(p) => {
            const b = browse;
            update((f) => { f.materials[b.i][b.field] = p; });
            setBrowse(null);
          }}
          onClose={() => setBrowse(null)}
        />
      )}
    </div>
  );
}
