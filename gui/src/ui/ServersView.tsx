import { useState } from "react";
import { t } from "../i18n";
import { newProfile, parsePodSsh, validateProfile, type ServerProfile } from "@tenryu-common/core/profiles";
import { useApp } from "../store";
import { Badge, Button, Field, TextInput } from "@tenryu-common/ui/kit";

export default function ServersView() {
  const m = t();
  const profiles = useApp((s) => s.profiles);
  const upsert = useApp((s) => s.upsertProfile);
  const del = useApp((s) => s.deleteProfile);
  const testConnection = useApp((s) => s.testConnection);
  const connTesting = useApp((s) => s.connTesting);
  const connResult = useApp((s) => s.connResult);

  const [editing, setEditing] = useState<ServerProfile | null>(null);
  const [errors, setErrors] = useState<string[]>([]);

  const startAdd = () => {
    setErrors([]);
    setEditing(newProfile());
  };
  const importPod = async () => {
    setErrors([]);
    const picked = await (await import("@tenryu-common/backend")).getBackend().openLocalTextFile([]);
    if (picked === null) return;
    const parsed = parsePodSsh(picked.content);
    if (parsed === null) {
      setErrors([m.server.importPodParseError]);
      return;
    }
    setEditing({
      ...newProfile(),
      name: "RunPod Pod",
      transport: "ssh",
      host: parsed.host,
      port: parsed.port,
      user: "root",
      identityFile: "~/.ssh/runpod_ed25519",
      ephemeralHostKey: true,
      tenryuBin: "/workspace/TENRYU/build/tenryu",
      runDir: "/workspace/TENRYU/tenryu_gui_runs",
    });
  };
  const startEdit = (p: ServerProfile) => {
    setErrors([]);
    setEditing({ ...p });
  };
  const save = async () => {
    if (!editing) return;
    const errs = validateProfile(editing);
    setErrors(errs);
    if (errs.length > 0) return;
    try {
      await upsert(editing);
      setEditing(null);
    } catch (err) {
      setErrors([t().server.saveFailed + String(err)]);
    }
  };

  const upd = (patch: Partial<ServerProfile>) => {
    if (editing) setEditing({ ...editing, ...patch });
  };

  return (
    <div className="max-w-2xl">
      <div className="mb-3 flex items-center justify-between">
        <h1 className="text-base font-semibold">{m.server.title}</h1>
        <div className="flex gap-2">
          <Button onClick={() => void importPod()}>{m.server.importPod}</Button>
          <Button variant="primary" onClick={startAdd}>
            {m.server.add}
          </Button>
        </div>
      </div>
      <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.server.storageNote}</p>

      {profiles.length === 0 && !editing && (
        <p style={{ color: "var(--fg-secondary)" }}>{m.server.empty}</p>
      )}

      <details className="mb-3 rounded border p-2 text-xs" style={{ borderColor: "var(--separator)" }}>
        <summary style={{ color: "var(--fg-secondary)" }}>{m.server.keychainGuideTitle}</summary>
        <pre
          className="mt-1 whitespace-pre-wrap rounded p-2"
          style={{ background: "var(--bg-inset)", fontFamily: "var(--mono)" }}
        >
          {m.server.keychainGuideBody}
        </pre>
      </details>

      <ul className="mb-4 flex flex-col gap-2">
        {profiles.map((p) => (
          <li
            key={p.id}
            className="flex items-center justify-between rounded border p-2"
            style={{ borderColor: "var(--separator)", background: "var(--bg-panel)" }}
          >
            <div>
              <div className="font-medium">{p.name}</div>
              <div className="text-xs" style={{ color: "var(--fg-secondary)", fontFamily: "var(--mono)" }}>
                {p.transport === "ssh" ? `${p.user ? p.user + "@" : ""}${p.host}` : "local"} · {p.tenryuBin}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button onClick={() => testConnection(p)} disabled={connTesting}>
                {connTesting ? m.server.testing : m.server.checkRun}
              </Button>
              <Button onClick={() => startEdit(p)}>{m.server.edit}</Button>
              <Button
                variant="danger"
                onClick={() => {
                  if (window.confirm(m.server.removeConfirm)) void del(p.id);
                }}
              >
                {m.server.remove}
              </Button>
            </div>
          </li>
        ))}
      </ul>

      {connResult && (
        <div className="mb-4 flex flex-col gap-1 text-xs">
          <div className="flex items-center gap-2">
            <span>{m.server.checkSsh}</span>
            <Badge tone={connResult.ok ? "ok" : "err"}>
              {connResult.ok ? m.server.checkOk : m.server.checkNg}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <span>{m.server.checkBin}</span>
            <Badge tone={connResult.binOk ? "ok" : "err"}>
              {connResult.binOk ? m.server.checkOk : m.server.checkNg}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <span>{m.server.checkBinRuns}</span>
            <Badge tone={connResult.binRuns ? "ok" : "err"}>
              {connResult.binRuns ? m.server.checkOk : m.server.checkNg}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <span>{m.server.checkGpu}</span>
            <Badge tone={connResult.gpu !== "" ? "ok" : "err"}>
              {connResult.gpu !== "" ? m.server.checkOk : m.server.checkNg}
            </Badge>
            {connResult.gpu !== "" && (
              <span style={{ color: "var(--fg-secondary)", fontFamily: "var(--mono)" }}>
                {connResult.gpu}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <span>{m.server.checkRunDir}</span>
            <Badge tone={connResult.runDirOk ? "ok" : "err"}>
              {connResult.runDirOk ? m.server.checkOk : m.server.checkNg}
            </Badge>
          </div>
          {connResult.detail && (
            <span className="text-xs" style={{ color: "var(--err)", fontFamily: "var(--mono)" }}>
              {connResult.detail}
            </span>
          )}
        </div>
      )}

      {editing && (
        <div
          className="rounded border p-3"
          style={{ borderColor: "var(--separator)", background: "var(--bg-panel)" }}
        >
          <Field label={m.server.name}>
            <TextInput value={editing.name} onChange={(e) => upd({ name: e.target.value })} />
          </Field>
          <Field label={m.server.transport}>
            <div className="flex gap-3">
              <label className="flex items-center gap-1">
                <input
                  type="radio"
                  checked={editing.transport === "ssh"}
                  onChange={() => upd({ transport: "ssh" })}
                />
                {m.server.transportSsh}
              </label>
              <label className="flex items-center gap-1">
                <input
                  type="radio"
                  checked={editing.transport === "local"}
                  onChange={() => upd({ transport: "local" })}
                />
                {m.server.transportLocal}
              </label>
            </div>
          </Field>
          {editing.transport === "ssh" && (
            <>
              <Field label={m.server.host}>
                <TextInput value={editing.host} onChange={(e) => upd({ host: e.target.value })} />
              </Field>
              <Field label={m.server.user}>
                <TextInput value={editing.user ?? ""} onChange={(e) => upd({ user: e.target.value })} />
              </Field>
              <Field label={m.server.port}>
                <TextInput
                  value={editing.port === undefined ? "" : String(editing.port)}
                  onChange={(e) => {
                    const v = e.target.value.trim();
                    upd({ port: v === "" ? undefined : Number(v) });
                  }}
                />
              </Field>
              <Field label={m.server.identityFile}>
                <TextInput
                  value={editing.identityFile ?? ""}
                  onChange={(e) => upd({ identityFile: e.target.value })}
                  style={{ fontFamily: "var(--mono)" }}
                />
                <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.server.identityFileHint}</p>
              </Field>
              <Field label={m.server.ephemeralHostKey}>
                <label className="flex items-center gap-2 text-xs">
                  <input
                    type="checkbox"
                    checked={editing.ephemeralHostKey === true}
                    onChange={(e) => upd({ ephemeralHostKey: e.target.checked })}
                  />
                  {m.server.ephemeralHostKeyNote}
                </label>
              </Field>
            </>
          )}
          <Field label={m.server.tenryuBin}>
            <TextInput
              value={editing.tenryuBin}
              onChange={(e) => upd({ tenryuBin: e.target.value })}
              style={{ fontFamily: "var(--mono)" }}
            />
            <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.server.tenryuBinHint}</p>
          </Field>
          <Field label={m.server.runDir}>
            <TextInput
              value={editing.runDir}
              onChange={(e) => upd({ runDir: e.target.value })}
              style={{ fontFamily: "var(--mono)" }}
            />
          </Field>
          {errors.length > 0 && (
            <ul className="mb-2 text-xs" style={{ color: "var(--err)" }}>
              {errors.map((e) => (
                <li key={e}>{e}</li>
              ))}
            </ul>
          )}
          <div className="flex gap-2">
            <Button variant="primary" onClick={() => void save()}>
              {t().common.save}
            </Button>
            <Button onClick={() => setEditing(null)}>{t().common.cancel}</Button>
          </div>
        </div>
      )}
    </div>
  );
}
