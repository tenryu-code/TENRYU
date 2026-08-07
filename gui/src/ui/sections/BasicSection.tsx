import { t } from "../../i18n";
import { ensureBackgroundGas } from "../../core/deck/formState";
import { useApp } from "../../store";
import { NumInput, SelectField, TextField } from "../fields";

export default function BasicSection() {
  const m = t();
  const form = useApp((s) => s.form);
  const update = useApp((s) => s.updateForm);
  return (
    <div className="max-w-xl flex flex-col gap-1">
      <h1 className="mb-2 text-base font-semibold">{m.nav.basic}</h1>
      <TextField
        label={m.form.caseName}
        value={form.main.name}
        onChange={(v) => update((f) => { f.main.name = v; })}
      />
      <SelectField
        label={m.form.dimension}
        value={form.main.dimension}
        options={[
          { value: "1D_SPH", label: m.form.dim1d },
          { value: "2D_RZ", label: m.form.dim2d },
        ]}
        onChange={(v) => update((f) => {
          f.main.dimension = v as never;
          if (v === "2D_RZ") ensureBackgroundGas(f);
        })}
      />
      {form.main.dimension === "1D_SPH" && (
        <SelectField
          label={m.form.geometry1d}
          value={form.main.geometry1d}
          options={[
            { value: "spherical", label: m.form.geoSpherical },
            { value: "cylindrical", label: m.form.geoCylindrical },
            { value: "planar", label: m.form.geoPlanar },
          ]}
          onChange={(v) => update((f) => { f.main.geometry1d = v as never; })}
        />
      )}
      <SelectField
        label={m.form.temperatureModel}
        value={form.main.temperatureModel}
        options={[
          { value: "2T", label: "2T" },
          { value: "1T", label: "1T" },
        ]}
        onChange={(v) => update((f) => { f.main.temperatureModel = v as never; })}
      />
      <NumInput
        int
        label={m.form.seed}
        value={form.main.seed}
        onChange={(n) => update((f) => { f.main.seed = n ?? 0; })}
      />
    </div>
  );
}
