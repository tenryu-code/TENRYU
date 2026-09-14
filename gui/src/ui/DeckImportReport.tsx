import { useEffect } from "react";
import { Button } from "@tenryu-common/ui/kit";
import { t } from "../i18n";
import { useApp } from "../store";
import { importActivePaths, type ImportRule } from "../core/deck/deckImport";

export default function DeckImportReport() {
  const form = useApp(s=>s.form);
  const open = useApp(s=>s.importReportOpen);
  const close = useApp(s=>s.setImportReportOpen);
  const state = form.deckImport;
  const m = t();
  useEffect(() => {
    if (!open) return;
    const listener = (event: KeyboardEvent) => { if (event.key === "Escape") close(false); };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [open, close]);
  if (!state || !open) return null;
  const groups: Array<[ImportRule["kind"], string]> = [
    ["mapped", m.deck.importMapped], ["approximated", m.deck.importApproximated],
    ["passthrough", m.deck.importPassthrough], ["omitted", m.deck.importOmitted],
  ];
  return <div className="fixed inset-0 z-50 flex items-center justify-center p-6" style={{background:"rgba(0,0,0,0.4)"}}>
    <div role="dialog" aria-modal="true" aria-label={m.deck.importReport} className="flex max-h-full w-[800px] flex-col gap-3 rounded-lg border p-4" style={{background:"var(--bg-panel)",borderColor:"var(--separator)"}}>
      <div className="flex items-center justify-between"><h2>{m.deck.importReport}</h2><Button onClick={()=>close(false)}>{m.common.close}</Button></div>
      <p className="text-xs">{m.deck.importNotice}</p>
      <div className="min-h-0 overflow-auto text-xs">
        {state.evaluation && <details className="text-xs" open>
          <summary>{m.deck.importEvaluationSettings}</summary>
          <div>{m.deck.importVenue}: {state.evaluation.venue === "local" ? m.deck.importLocal : `${m.deck.importServer}: ${state.evaluation.profile?.name} (${state.evaluation.profile?.host})`}</div>
          <div>{m.deck.importWorkingDirectory}: <code>{state.evaluation.workingDirectory}</code></div>
          <div>__file__: <code>{state.evaluation.filename}</code></div>
          {state.evaluation.repoRoot && <div>--repo-root: <code>{state.evaluation.repoRoot}</code></div>}
          <pre className="whitespace-pre-wrap break-words">{Object.entries(state.evaluation.environment).map(([key,value])=>`${key}=${value}`).join("\n") || m.deck.importNone}</pre>
          <p>{m.deck.importSettingsNotice}</p>
        </details>}
        {importActivePaths(form).length > 0 && <p className="text-xs" style={{color:"var(--warn)"}}>{m.deck.importChanged}</p>}
        {groups.map(([kind, label]) => {
          const entries = state.rules.filter(r=>r.kind===kind);
          return <details key={kind} open={kind==="passthrough" || kind==="approximated"} className="mb-3">
            <summary>{label} ({entries.length})</summary>
            {entries.map((r,i)=><div key={i} className="my-1 break-words"><code>{r.path.join(".")}</code> — {r.reason}</div>)}
          </details>;
        })}
        <p>{m.deck.importUnsupported}: {m.deck.importNone}</p>
        {!!state.unverified?.length && <div className="my-3" style={{color:"var(--warn)"}}>
          <p>{m.deck.importUnverified}</p>
          {state.unverified.map((r,i)=><div className="my-1 break-words" key={i}><code>{r.path.join(".")}</code> — {r.reason}</div>)}
        </div>}
        {state.notes.map((note,i)=><p className="my-2 whitespace-pre-wrap break-words" key={i}>{note}</p>)}
      </div>
    </div>
  </div>;
}
