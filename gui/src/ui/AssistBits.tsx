import { t } from "../i18n";

/** "CODE: detail" or "CODE" → localized code + mono detail. */
export function AssistError({ text }: { text: string }) {
  const separator = text.indexOf(": ");
  const code = separator >= 0 ? text.slice(0, separator) : text;
  const detail = separator >= 0 ? text.slice(separator + 2) : "";
  const errors = t().assist.errors as Record<string, string>;

  return (
    <div className="text-xs">
      <div style={{ color: "var(--err)" }}>{errors[code] ?? code}</div>
      {detail.length > 0 && (
        <div
          className="whitespace-pre-wrap break-all"
          style={{ fontFamily: "var(--mono)", opacity: 0.75 }}
        >
          {detail}
        </div>
      )}
    </div>
  );
}

function formatValue(value: unknown): string {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return String(value);
  }
  const json = JSON.stringify(value) ?? String(value);
  return json.length > 120 ? `${json.slice(0, 120)}…` : json;
}

/** Two-column key/value table for a flat-ish object. */
export function KvTable({ obj }: { obj: Record<string, unknown> }) {
  return (
    <table className="w-full text-xs" style={{ fontFamily: "var(--mono)" }}>
      <tbody>
        {Object.entries(obj)
          .sort(([left], [right]) => left.localeCompare(right))
          .map(([key, value]) => (
            <tr key={key} className="align-top">
              <td
                className="whitespace-nowrap pr-2"
                style={{ color: "var(--fg-secondary)" }}
              >
                {key}
              </td>
              <td className="break-all">{formatValue(value)}</td>
            </tr>
          ))}
      </tbody>
    </table>
  );
}

export function RawFold({ raw }: { raw: string | undefined }) {
  if (raw === undefined || raw.length === 0) return null;
  return (
    <details className="text-xs">
      <summary style={{ color: "var(--fg-secondary)" }}>{t().assist.rawTitle}</summary>
      <pre
        className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap break-all rounded border p-2"
        style={{
          borderColor: "var(--separator)",
          background: "var(--bg-inset)",
          fontFamily: "var(--mono)",
        }}
      >
        {raw}
      </pre>
    </details>
  );
}
