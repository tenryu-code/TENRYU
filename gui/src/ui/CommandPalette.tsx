import { useEffect, useMemo, useRef, useState } from "react";

export interface PaletteCommand {
  id: string;
  label: string;
  hint?: string;
  run: () => void;
}

export default function CommandPalette({
  open,
  commands,
  placeholder,
  onClose,
}: {
  open: boolean;
  commands: PaletteCommand[];
  placeholder: string;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (q === "") return commands;
    return commands.filter((c) => c.label.toLowerCase().includes(q) || c.id.includes(q));
  }, [commands, query]);

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setCursor(0);
    const handle = window.setTimeout(() => inputRef.current?.focus(), 0);
    return () => window.clearTimeout(handle);
  }, [open]);

  useEffect(() => {
    setCursor((c) => Math.min(c, Math.max(0, matches.length - 1)));
  }, [matches.length]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-20 flex items-start justify-center pt-24"
      style={{ background: "rgba(0,0,0,0.35)" }}
      onClick={onClose}
    >
      <div
        className="w-[460px] rounded-lg border p-2"
        style={{ background: "var(--bg-raised)", borderColor: "var(--separator)" }}
        onClick={(event) => event.stopPropagation()}
      >
        <input
          ref={inputRef}
          className="w-full rounded border px-2"
          style={{
            background: "var(--bg-panel)",
            borderColor: "var(--separator)",
            color: "var(--fg)",
            borderRadius: "var(--radius-md)",
            height: "var(--control-h)",
            fontSize: 13,
          }}
          value={query}
          placeholder={placeholder}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Escape") {
              event.preventDefault();
              onClose();
            } else if (event.key === "ArrowDown") {
              event.preventDefault();
              setCursor((c) => Math.min(matches.length - 1, c + 1));
            } else if (event.key === "ArrowUp") {
              event.preventDefault();
              setCursor((c) => Math.max(0, c - 1));
            } else if (event.key === "Enter") {
              event.preventDefault();
              const command = matches[cursor];
              if (command !== undefined) {
                onClose();
                command.run();
              }
            }
          }}
        />
        <ul className="mt-1 max-h-72 overflow-auto">
          {matches.map((command, index) => (
            <li key={command.id}>
              <button
                className="flex w-full items-center gap-2 rounded px-2 py-1 text-left text-[13px]"
                style={{
                  background: index === cursor ? "var(--selected-bg)" : "transparent",
                  color: index === cursor ? "var(--selected-fg)" : "var(--fg)",
                }}
                onMouseEnter={() => setCursor(index)}
                onClick={() => {
                  onClose();
                  command.run();
                }}
              >
                <span className="min-w-0 flex-1 truncate">{command.label}</span>
                {command.hint !== undefined && (
                  <span className="text-xs" style={{ color: "var(--fg-secondary)" }}>{command.hint}</span>
                )}
              </button>
            </li>
          ))}
          {matches.length === 0 && (
            <li className="px-2 py-1 text-xs" style={{ color: "var(--fg-secondary)" }}>—</li>
          )}
        </ul>
      </div>
    </div>
  );
}
