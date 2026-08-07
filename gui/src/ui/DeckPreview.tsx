import { Highlight, themes } from "prism-react-renderer";
import { useEffect, useRef, useState } from "react";
import { useSystemDark } from "./hooks";
import { t } from "../i18n";
import { migrateFormState } from "../core/deck/formState";
import { extractGuiState } from "../core/deck/roundtrip";
import { useApp } from "../store";
import { Button } from "@tenryu-common/ui/kit";

export default function DeckPreview() {
  const m = t();
  const deck = useApp((s) => s.deck);
  const formErrors = useApp((s) => s.formErrors);
  const loadForm = useApp((s) => s.loadForm);
  const saveDeckToFile = useApp((s) => s.saveDeckToFile);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [copied, setCopied] = useState(false);
  const [saved, setSaved] = useState(false);
  const [loadOpen, setLoadOpen] = useState(false);
  const [loadText, setLoadText] = useState("");
  const [loadErr, setLoadErr] = useState("");

  const dark = useSystemDark();

  useEffect(() => {
    if (!loadOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setLoadOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [loadOpen]);

  const doCopy = async () => {
    await navigator.clipboard.writeText(deck);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  const doLoad = (text: string = loadText) => {
    const r = extractGuiState(text);
    if (!r.ok) {
      setLoadErr(
        r.reason === "no-marker"
          ? m.deck.loadErrNoMarker
          : r.reason === "bad-json"
            ? m.deck.loadErrBadJson
            : m.deck.loadErrBadVersion,
      );
      return;
    }
    loadForm(migrateFormState(r.state));
    setLoadOpen(false);
    setLoadText("");
    setLoadErr("");
  };

  const doSave = async () => {
    const ok = await saveDeckToFile();
    if (ok) {
      setSaved(true);
      setTimeout(() => setSaved(false), 1200);
    }
  };

  const onLoadFilePicked = (file: File, input: HTMLInputElement) => {
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result ?? "");
      setLoadText(text);
      doLoad(text);
      input.value = "";
    };
    reader.onerror = () => {
      setLoadErr(m.deck.loadErrBadJson);
      input.value = "";
    };
    reader.readAsText(file);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="mb-1 flex items-center gap-2">
        <h2 className="text-sm font-semibold">{m.deck.title}</h2>
        <div className="flex-1" />
        <Button onClick={() => void doCopy()}>{copied ? m.deck.copied : m.deck.copy}</Button>
        <Button disabled={formErrors.length > 0} onClick={() => void doSave()}>
          {saved ? m.deck.savedDone : m.deck.save}
        </Button>
        <Button onClick={() => setLoadOpen(true)}>{m.deck.load}</Button>
      </div>

      {formErrors.length > 0 ? (
        <div className="min-h-0 flex-1 overflow-auto rounded border p-2 text-xs" style={{ borderColor: "var(--separator)" }}>
          <div className="mb-1" style={{ color: "var(--err)" }}>
            {m.deck.formInvalid}
          </div>
          <ul style={{ color: "var(--err)" }}>
            {formErrors.map((e) => (
              <li key={e}>・{e}</li>
            ))}
          </ul>
        </div>
      ) : (
        <div
          className="min-h-0 flex-1 overflow-auto rounded border"
          style={{ borderColor: "var(--separator)", background: "var(--bg-inset)" }}
        >
          <Highlight code={deck} language="python" theme={dark ? themes.oneDark : themes.oneLight}>
            {({ tokens, getLineProps, getTokenProps }) => (
              <pre className="p-2 text-[11px] leading-4" style={{ fontFamily: "var(--mono)", background: "transparent" }}>
                {tokens.map((line, i) => {
                  const lineProps = getLineProps({ line });
                  const raw = line.map((token) => token.content).join("");
                  const style = raw.startsWith("# Units:")
                    ? { ...lineProps.style, whiteSpace: "pre-wrap" as const, overflowWrap: "anywhere" as const }
                    : lineProps.style;
                  return (
                    <div key={i} {...lineProps} style={style}>
                      {line.map((token, k) => (
                        <span key={k} {...getTokenProps({ token })} />
                      ))}
                    </div>
                  );
                })}
              </pre>
            )}
          </Highlight>
        </div>
      )}

      {loadOpen && (
        <div
          className="fixed inset-0 z-10 flex items-center justify-center"
          style={{ background: "rgba(0,0,0,0.4)" }}
        >
          <div
            className="flex w-[560px] flex-col gap-2 rounded-lg border p-4"
            style={{ background: "var(--bg-panel)", borderColor: "var(--separator)" }}
          >
            <h3 className="text-sm font-semibold">{m.deck.loadTitle}</h3>
            <textarea
              value={loadText}
              onChange={(e) => setLoadText(e.target.value)}
              spellCheck={false}
              className="h-64 resize-none rounded border p-2 text-xs"
              style={{
                background: "var(--bg-inset)",
                borderColor: "var(--separator)",
                color: "var(--fg)",
                fontFamily: "var(--mono)",
              }}
            />
            {loadErr && (
              <div className="text-xs" style={{ color: "var(--err)" }}>
                {loadErr}
              </div>
            )}
            <div className="flex justify-end gap-2">
              <Button onClick={() => fileInputRef.current?.click()}>{m.deck.loadFromFile}</Button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".py,.txt"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) onLoadFilePicked(file, e.currentTarget);
                }}
              />
              <div className="flex-1" />
              <Button onClick={() => setLoadOpen(false)}>{m.common.cancel}</Button>
              <Button variant="primary" onClick={() => doLoad()}>
                {m.deck.loadRun}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
