export interface ValidateSummaryRow {
  label: string;
  text: string;
}

export interface ValidateResult {
  ok: boolean;
  summary: ValidateSummaryRow[];
  errors: string[];
  warnings: string[];
  raw: string;
}

const LOG_PREFIX_RE = /^\[[^\]]*\]\s*\[[a-z]+\]\s*\[TENRYU\]\s*/i;
const SUMMARY_BEGIN = "---- pre-flight summary ----";
const SUMMARY_END_RE = /^-{6,}$/;

export function stripLogPrefix(line: string): string {
  return line.replace(LOG_PREFIX_RE, "");
}

export function parseValidateOutput(
  stdout: string,
  stderr: string,
  code: number | null,
): ValidateResult {
  const raw = stdout + (stderr.length > 0 ? (stdout.endsWith("\n") || stdout.length === 0 ? "" : "\n") + stderr : "");
  const outLines = stdout.split(/\r?\n/).map(stripLogPrefix);
  const errLines = stderr.split(/\r?\n/);

  const ok = code === 0 && outLines.some((l) => l.includes("Configuration validated successfully"));

  const summary: ValidateSummaryRow[] = [];
  let inSummary = false;
  for (const line of outLines) {
    if (!inSummary) {
      if (line.trim() === SUMMARY_BEGIN) inSummary = true;
      continue;
    }
    if (SUMMARY_END_RE.test(line.trim())) break;
    const idx = line.indexOf(":");
    if (idx < 0) continue;
    const label = line.slice(0, idx).trim();
    const text = line.slice(idx + 1).trim();
    if (label.length > 0) summary.push({ label, text });
  }

  const errors: string[] = [];
  for (const line of errLines) {
    if (/^TENRYU ERROR/.test(line)) errors.push(line.trim());
  }
  for (const line of outLines) {
    if (/^TENRYU ERROR/.test(line)) errors.push(line.trim());
  }
  if (!ok && errors.length === 0) {
    const tail = errLines.filter((l) => l.trim().length > 0).slice(-3);
    errors.push(...tail);
  }

  const warnings: string[] = [];
  for (const line of stdout.split(/\r?\n/)) {
    if (/\[warning\]/i.test(line)) warnings.push(stripLogPrefix(line).trim());
  }

  return { ok, summary, errors, warnings, raw };
}
