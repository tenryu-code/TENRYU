import { translateError as translateRawError } from "../core/errorsJa";
import { getLang, t } from "../i18n";
import { useApp } from "../store";
import { Badge } from "@tenryu-common/ui/kit";

function specialCode(e: string): string | null {
  const m = t();
  if (e === "NO_PROFILE") return m.validate.noProfile;
  if (e === "NO_BIN") return m.server.binMissing;
  if (e === "NO_DECK") return m.validate.noDeck;
  if (e === "FORM_INVALID") return m.validate.formInvalid;
  if (e === "NO_TOOLS") return m.geo2d.toolsMissing;
  return null;
}

function ErrorLine({ e }: { e: string }) {
  const special = specialCode(e);
  if (special !== null) {
    return <li className="whitespace-pre-wrap break-all">{special}</li>;
  }
  const tr = translateRawError(e);
  return (
    <li className="whitespace-pre-wrap break-all">
      {getLang() === "ja" && tr.ja !== null && <div className="font-medium">{tr.ja}</div>}
      <div style={{ opacity: tr.ja !== null ? 0.75 : 1 }}>{tr.raw}</div>
    </li>
  );
}

export default function ValidatePanel() {
  const m = t();
  const validating = useApp((s) => s.validating);
  const result = useApp((s) => s.validateResult);
  const sentTo = useApp((s) => s.validateSentTo);

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex items-center gap-2">
        <h2 className="text-sm font-semibold">{m.validate.title}</h2>
        {validating ? (
          <Badge tone="muted">{m.validate.running}</Badge>
        ) : result ? (
          <Badge tone={result.ok ? "ok" : "err"}>{result.ok ? m.validate.pass : m.validate.fail}</Badge>
        ) : (
          <Badge tone="muted">{m.validate.notRun}</Badge>
        )}
      </div>

      {sentTo && (
        <div className="text-xs" style={{ color: "var(--fg-secondary)", fontFamily: "var(--mono)" }}>
          {m.validate.sentTo}: {sentTo}
        </div>
      )}

      {result && result.summary.length > 0 && (
        <div>
          <h3 className="mb-1 text-xs font-semibold" style={{ color: "var(--fg-secondary)" }}>
            {m.validate.summaryTitle}
          </h3>
          <table className="w-full text-xs" style={{ fontFamily: "var(--mono)" }}>
            <tbody>
              {result.summary.map((row) => (
                <tr key={row.label} className="align-top">
                  <td className="whitespace-nowrap pr-2" style={{ color: "var(--fg-secondary)" }}>
                    {row.label}
                  </td>
                  <td className="break-all">{row.text}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {result && result.errors.length > 0 && (
        <div>
          <h3 className="mb-1 text-xs font-semibold" style={{ color: "var(--err)" }}>
            {m.validate.errorsTitle}
          </h3>
          <ul className="flex flex-col gap-1 text-xs" style={{ color: "var(--err)", fontFamily: "var(--mono)" }}>
            {result.errors.map((e, i) => (
              <ErrorLine key={i} e={e} />
            ))}
          </ul>
        </div>
      )}

      {result && result.warnings.length > 0 && (
        <div>
          <h3 className="mb-1 text-xs font-semibold" style={{ color: "var(--warn)" }}>
            {m.validate.warningsTitle}
          </h3>
          <ul className="flex flex-col gap-1 text-xs" style={{ color: "var(--warn)", fontFamily: "var(--mono)" }}>
            {result.warnings.map((w, i) => (
              <li key={i} className="whitespace-pre-wrap break-all">
                {w}
              </li>
            ))}
          </ul>
        </div>
      )}

      {result && result.raw.length > 0 && (
        <details className="text-xs">
          <summary style={{ color: "var(--fg-secondary)" }}>{m.validate.rawTitle}</summary>
          <pre
            className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap break-all rounded border p-2"
            style={{ borderColor: "var(--separator)", background: "var(--bg-inset)" }}
          >
            {result.raw}
          </pre>
        </details>
      )}
    </div>
  );
}
