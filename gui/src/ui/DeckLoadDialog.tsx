import { useEffect, useState } from "react";
import { Button } from "@tenryu-common/ui/kit";
import { remoteDirname } from "@tenryu-common/core/ssh";
import { currentProfile, useApp } from "../store";
import { t } from "../i18n";
import RemoteFileBrowser from "./RemoteFileBrowser";
import { appendImportEnvironment, discoverImportEnvironment } from "../core/deck/importEnvironment";

export default function DeckLoadDialog() {
  const pending = useApp(s=>s.pendingDeckImport);
  const settings = useApp(s=>s.importSettings);
  const setSettings = useApp(s=>s.setImportSettings);
  const busy = useApp(s=>s.deckImportBusy);
  const profile = useApp(s=>currentProfile(s));
  const close = useApp(s=>s.setPendingDeckImport);
  const [text, setText] = useState("");
  const [filename, setFilename] = useState("");
  const [error, setError] = useState("");
  const [browse, setBrowse] = useState(false);
  const m = t();
  function updateText(value: string) {
    setText(value);
    const environmentText = useApp.getState().importSettings.environmentText;
    setSettings({environmentText:appendImportEnvironment(environmentText,discoverImportEnvironment(value))});
  }
  useEffect(()=>{ if (pending) updateText(pending.text); setFilename(""); setError(pending?.error ?? ""); setBrowse(false); },[pending]);
  useEffect(()=>{
    if (!pending || busy || browse) return;
    const listener = (e: KeyboardEvent) => { if (e.key === "Escape") close(null); };
    window.addEventListener("keydown",listener);
    return ()=>window.removeEventListener("keydown",listener);
  },[pending,busy,browse,close]);
  if (!pending) return null;
  const defaultDirectory = settings.venue === "local" && pending.filename ? remoteDirname(pending.filename) : m.deck.importDirectoryDefault;
  const environmentNames = discoverImportEnvironment(text);
  async function load() {
    if (!pending) return;
    const result = await useApp.getState().loadDeckText(text,pending.filename,filename.trim() || undefined,{autoDetectWorkingDirectory:true});
    if (result) { setError(result); return; }
    if (pending.filename) useApp.setState({namelistPath:pending.filename,deckIoStatus:{kind:"loaded",detail:pending.name}});
    close(null);
  }
  return <><div className="fixed inset-0 z-40 flex items-center justify-center p-4" style={{background:"rgba(0,0,0,0.4)"}}>
    <div role="dialog" aria-modal="true" aria-label={m.deck.loadTitle} className="flex max-h-full w-[640px] flex-col gap-3 overflow-auto rounded-lg border p-4" style={{background:"var(--bg-panel)",borderColor:"var(--separator)"}}>
      <h3>{m.deck.loadTitle}{pending.name && ` — ${pending.name}`}</h3>
      <textarea aria-label={m.deck.loadTitle} value={text} onChange={e=>updateText(e.target.value)} disabled={busy} spellCheck={false} className="min-h-32 rounded border p-2 font-mono text-xs" />
      <label className="text-xs">{m.deck.importVenue}
        <select aria-label={m.deck.importVenue} value={settings.venue} disabled={busy} onChange={e=>setSettings({venue:e.target.value as "local"|"server",venueExplicit:true})} className="ml-2 rounded border p-1">
          <option value="local">{m.deck.importLocal}</option>
          <option value="server" disabled={!profile}>{m.deck.importServer}{profile ? `: ${profile.name}` : ""}</option>
        </select>
        {!profile && <span className="ml-2" style={{color:"var(--fg-secondary)"}}>{m.deck.importServerHint}</span>}
      </label>
      <label className="text-xs">{m.deck.importWorkingDirectory}
        <div className="flex items-center gap-2">
        <input aria-label={m.deck.importWorkingDirectory} value={settings.workingDirectory} placeholder={defaultDirectory} onChange={e=>setSettings({workingDirectory:e.target.value})} disabled={busy} className="mt-1 w-full rounded border p-2" />
        <Button disabled={busy || settings.venue !== "server" || !profile} onClick={()=>setBrowse(true)}>{m.remoteFs.browse}</Button>
        </div>
      </label>
      <label className="text-xs">{m.deck.importEnvironment}
        <textarea aria-label={m.deck.importEnvironment} value={settings.environmentText} onChange={e=>setSettings({environmentText:e.target.value})} disabled={busy} spellCheck={false} placeholder={m.deck.importEnvironmentPlaceholder} className="mt-1 h-20 w-full rounded border p-2 font-mono" />
        <p>{m.deck.importEnvironmentReads}: {environmentNames.join(", ") || m.deck.importNone}</p>
        <p>{m.deck.importEnvironmentEmpty}</p>
      </label>
      <label className="text-xs">{m.deck.importSourceFilename}
        <input aria-label={m.deck.importSourceFilename} value={filename} onChange={e=>setFilename(e.target.value)} disabled={busy} className="mt-1 w-full rounded border p-2" />
      </label>
      <p className="text-xs">{m.deck.importSettingsNotice}</p>
      <p className="text-xs">{m.deck.importNotice}</p>
      {busy && <p role="status" className="text-xs">{m.deck.importBusy}</p>}
      {error && <pre className="max-h-48 overflow-auto whitespace-pre-wrap text-xs" style={{color:"var(--err)"}}>{error}</pre>}
      <div className="flex justify-end gap-2">
        <Button disabled={busy} onClick={()=>void useApp.getState().loadNamelist()}>{m.deck.loadFromFile}</Button>
        <div className="flex-1" />
        <Button disabled={busy} onClick={()=>close(null)}>{m.common.cancel}</Button>
        <Button disabled={busy || (settings.venue === "server" && !profile)} variant="primary" onClick={()=>void load()}>{m.deck.loadRun}</Button>
      </div>
    </div>
  </div>
    {browse && settings.venue === "server" && profile && <RemoteFileBrowser
      mode="directory" title={m.deck.importWorkingDirectory} initialPath={settings.workingDirectory.trim() || profile.runDir}
      onPick={path=>{setSettings({workingDirectory:path});setBrowse(false);}} onClose={()=>setBrowse(false)} />}
  </>;
}
