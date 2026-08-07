import { t } from "../../i18n";
import { useApp } from "../../store";
import { NumInput, QInput, SwitchField, TextField } from "../fields";

export default function OutputSection() {
  const m = t();
  const form = useApp((s) => s.form);
  const update = useApp((s) => s.updateForm);
  return (
    <div className="max-w-xl flex flex-col gap-1">
      <h1 className="mb-2 text-base font-semibold">{m.form.outputTitle}</h1>
      <h2 className="text-sm font-semibold">{m.form.terminationTitle}</h2>
      <QInput
        label={m.form.tEnd}
        kind="time"
        value={form.main.tEnd}
        onChange={(q) => update((f) => { f.main.tEnd = q; })}
      />
      <NumInput
        int
        label={m.form.maxSteps}
        value={form.main.maxSteps}
        onChange={(n) => update((f) => { f.main.maxSteps = n ?? 0; })}
      />
      <h2 className="mt-3 text-sm font-semibold">{m.form.outputFilesTitle}</h2>
      <TextField
        mono
        label={m.form.outDir}
        value={form.output.directory}
        placeholder={`outputs/${form.main.name}`}
        onChange={(v) => update((f) => { f.output.directory = v; })}
      />
      <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.form.outDirNote}</p>
      <SwitchField
        label={`${m.form.plotEvery} — ${m.form.enableInterval}`}
        checked={form.output.plotEveryS !== null}
        onChange={(b) => update((f) => { f.output.plotEveryS = b ? { value: 50, unit: "ps" } : null; })}
      />
      {form.output.plotEveryS !== null && (
        <QInput
          label={m.form.plotEvery}
          kind="time"
          value={form.output.plotEveryS}
          onChange={(q) => update((f) => { f.output.plotEveryS = q; })}
        />
      )}
      <SwitchField
        label={`${m.form.historyEvery} — ${m.form.enableInterval}`}
        checked={form.output.historyEveryS !== null}
        onChange={(b) => update((f) => { f.output.historyEveryS = b ? { value: 1, unit: "ps" } : null; })}
      />
      {form.output.historyEveryS !== null && (
        <QInput
          label={m.form.historyEvery}
          kind="time"
          value={form.output.historyEveryS}
          onChange={(q) => update((f) => { f.output.historyEveryS = q; })}
        />
      )}
      <SwitchField
        label={`${m.form.checkpointEvery} — ${m.form.enableInterval}`}
        checked={form.output.checkpointEveryS !== null}
        onChange={(b) => update((f) => { f.output.checkpointEveryS = b ? { value: 1, unit: "ps" } : null; })}
      />
      {form.output.checkpointEveryS !== null && (
        <QInput
          label={m.form.checkpointEvery}
          kind="time"
          value={form.output.checkpointEveryS}
          onChange={(q) => update((f) => { f.output.checkpointEveryS = q; })}
        />
      )}
      <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.form.checkpointEveryNote}</p>
      <h2 className="mt-3 text-sm font-semibold">{m.form.customBlockTitle}</h2>
      <textarea
        value={form.customPythonBlock}
        onChange={(e) => update((f) => { f.customPythonBlock = e.target.value; })}
        className="h-40 w-full resize-none rounded border p-2 text-xs"
        style={{
          background: "var(--bg-inset)",
          borderColor: "var(--separator)",
          color: "var(--fg)",
          fontFamily: "var(--mono)",
        }}
      />
    </div>
  );
}
