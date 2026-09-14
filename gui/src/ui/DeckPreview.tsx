import { Highlight, themes } from "prism-react-renderer";
import { useState } from "react";
import { useSystemDark } from "./hooks";
import { t } from "../i18n";
import { useApp } from "../store";
import { Button } from "@tenryu-common/ui/kit";

export default function DeckPreview() {
  const m = t();
  const deck = useApp((s) => s.deck);
  const formErrors = useApp((s) => s.formErrors);
  const busy = useApp((s) => s.deckImportBusy);
  const openImport = useApp((s) => s.setPendingDeckImport);
  const imported = useApp((s) => !!s.form.deckImport);
  const showReport = useApp((s) => s.setImportReportOpen);
  const saveDeckToFile = useApp((s) => s.saveDeckToFile);
  const [copied, setCopied] = useState(false);
  const [saved, setSaved] = useState(false);
  const dark = useSystemDark();

  const doCopy = async () => {
    await navigator.clipboard.writeText(deck);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  const doSave = async () => {
    const ok = await saveDeckToFile();
    if (ok) {
      setSaved(true);
      setTimeout(() => setSaved(false), 1200);
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="mb-1 flex flex-wrap items-center gap-2 [&_button]:whitespace-nowrap">
        <h2 className="whitespace-nowrap text-sm font-semibold">{m.deck.title}</h2>
        <div className="flex-1" />
        <Button onClick={() => void doCopy()}>{copied ? m.deck.copied : m.deck.copy}</Button>
        <Button disabled={formErrors.length > 0} onClick={() => void doSave()}>
          {saved ? m.deck.savedDone : m.deck.save}
        </Button>
        {imported && <Button onClick={() => showReport(true)}>{m.deck.importReport}</Button>}
        <Button disabled={busy} onClick={() => openImport({text:"",filename:null,name:""})}>{m.deck.load}</Button>
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

    </div>
  );
}
