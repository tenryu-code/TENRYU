import { useEffect, useState } from "react";
import { joinRemoteDir, remoteParentDir } from "@tenryu-common/core/remoteLs";
import { t } from "../i18n";
import { useApp, type RemoteDirListing } from "../store";
import { Button, TextInput } from "@tenryu-common/ui/kit";

export default function RemoteFileBrowser({
  title,
  initialPath,
  onPick,
  onClose,
}: {
  title: string;
  initialPath: string;
  onPick: (path: string) => void;
  onClose: () => void;
}) {
  const m = t();
  const listRemoteDir = useApp((s) => s.listRemoteDir);
  const [path, setPath] = useState("");
  const [listing, setListing] = useState<RemoteDirListing | null>(null);
  const [busy, setBusy] = useState(false);
  const [pathInput, setPathInput] = useState("");
  const [onlyH5, setOnlyH5] = useState(true);

  const load = async (p: string) => {
    setBusy(true);
    const l = await listRemoteDir(p);
    setListing(l);
    if (l.ok) setPath(l.path);
    setPathInput(l.ok ? l.path : pathInput);
    setBusy(false);
  };

  useEffect(() => {
    void load(initialPath);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const entries = (listing?.ok ? listing.entries : []).filter(
    (entry) => entry.dir || !onlyH5 || entry.name.endsWith(".h5"),
  );

  return (
    <div
      className="fixed inset-0 z-10 flex items-center justify-center"
      style={{ background: "rgba(0,0,0,0.4)" }}
      onClick={onClose}
    >
      <div
        className="flex max-h-[70vh] min-h-[320px] w-[640px] flex-col gap-2 rounded-lg border p-4"
        style={{ background: "var(--bg-panel)", borderColor: "var(--separator)" }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold">{title}</h3>
          <div className="flex-1" />
          <Button onClick={onClose}>{m.common.cancel}</Button>
        </div>
        <div className="flex items-center gap-2">
          <TextInput
            value={pathInput}
            onChange={(e) => setPathInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void load(pathInput);
            }}
            style={{ fontFamily: "var(--mono)" }}
          />
          <Button onClick={() => void load(pathInput)}>{m.remoteFs.go}</Button>
        </div>
        {listing !== null && !listing.ok && (
          <div className="text-xs" style={{ color: "var(--err)", fontFamily: "var(--mono)" }}>
            {m.remoteFs.loadError}
            {listing.error === "NO_PROFILE" ? m.validate.noProfile : listing.error}
          </div>
        )}
        <div
          className="min-h-0 flex-1 overflow-y-auto rounded border text-sm"
          style={{ borderColor: "var(--separator)", fontFamily: "var(--mono)" }}
        >
          <div
            className={`px-2 py-1 ${path === "/" ? "cursor-default opacity-40" : "cursor-pointer hover:bg-[var(--bg-inset)]"}`}
            onClick={() => {
              if (path !== "/") void load(remoteParentDir(path));
            }}
          >
            ..
          </div>
          {entries.map((entry) => (
            <div
              key={`${entry.dir ? "d" : "f"}-${entry.name}`}
              className="cursor-pointer px-2 py-1 hover:bg-[var(--bg-inset)]"
              onClick={() => {
                const selected = joinRemoteDir(path, entry.name);
                if (entry.dir) void load(selected);
                else onPick(selected);
              }}
            >
              {entry.name}
              {entry.dir ? "/" : ""}
            </div>
          ))}
          {listing?.ok && entries.length === 0 && (
            <div className="px-2 py-1" style={{ color: "var(--fg-secondary)" }}>
              {m.remoteFs.empty}
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={onlyH5}
              onChange={(e) => setOnlyH5(e.target.checked)}
            />
            {m.remoteFs.onlyH5}
          </label>
          <div className="flex-1" />
          {busy && (
            <span className="text-xs" style={{ color: "var(--fg-secondary)" }}>
              {m.server.testing}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
