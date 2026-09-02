export type Parsed<T> =
  | { ok: true; data: T; raw: string }
  | { ok: false; error: string; raw: string };

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseJsonObject(text: string): Parsed<Record<string, unknown>> {
  const raw = text;
  let candidate = text.trim();
  if (!candidate.startsWith("{")) {
    const start = candidate.indexOf("{");
    if (start < 0) {
      return { ok: false, error: "JSON object not found", raw };
    }
    candidate = candidate.slice(start);
  }

  try {
    const value: unknown = JSON.parse(candidate);
    if (!isObject(value)) {
      return { ok: false, error: "JSON value is not an object", raw };
    }
    return { ok: true, data: value, raw };
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return { ok: false, error: `Invalid JSON: ${detail}`, raw };
  }
}

function stringOrNull(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

export interface AssistStatusView {
  enabled: boolean;
  disabledBy: string | null;
  configSource: string | null;
  /** roles joined with providers; model null when the provider is unknown/dry_run. */
  roles: Array<{ role: string; provider: string; model: string | null }>;
  warnings: string[];
}

export function parseAssistStatus(stdout: string): Parsed<AssistStatusView> {
  const parsed = parseJsonObject(stdout);
  if (!parsed.ok) return parsed;

  const providers = isObject(parsed.data.providers)
    ? parsed.data.providers
    : {};
  const roleValues = isObject(parsed.data.roles) ? parsed.data.roles : {};
  const roles = Object.entries(roleValues)
    .filter((entry): entry is [string, string] => typeof entry[1] === "string")
    .map(([role, provider]) => {
      const providerValue = providers[provider];
      const model = isObject(providerValue)
        ? stringOrNull(providerValue.model)
        : null;
      return { role, provider, model };
    })
    .sort((left, right) => left.role.localeCompare(right.role));

  return {
    ok: true,
    data: {
      enabled: parsed.data.enabled === true,
      disabledBy: stringOrNull(parsed.data.disabled_by),
      configSource: stringOrNull(parsed.data.config_source),
      roles,
      warnings: stringArray(parsed.data.warnings),
    },
    raw: parsed.raw,
  };
}

export interface DeckLintView {
  toolError: string | null;
  validate: {
    ok: boolean | null;
    exitCode: number | null;
    stderrTail: string;
  };
  meshPreview: Record<string, unknown> | null;
  lints: Array<{
    id: string;
    severity: string;
    ok: boolean;
    detail: unknown;
  }>;
  intentLock: Array<{
    path: string;
    ok: boolean;
    expected: unknown;
    actual: unknown;
    reason: string | null;
  }> | null;
  defaultsDiff: Record<string, unknown> | null;
}

export function parseDeckLint(stdout: string): Parsed<DeckLintView> {
  const parsed = parseJsonObject(stdout);
  if (!parsed.ok) return parsed;

  const validate = isObject(parsed.data.validate) ? parsed.data.validate : {};
  const lints = Array.isArray(parsed.data.lints)
    ? parsed.data.lints.flatMap((value) => {
        if (
          !isObject(value) ||
          typeof value.id !== "string" ||
          typeof value.severity !== "string" ||
          typeof value.ok !== "boolean"
        ) {
          return [];
        }
        return [
          {
            id: value.id,
            severity: value.severity,
            ok: value.ok,
            detail: value.detail ?? null,
          },
        ];
      })
    : [];

  let intentLock: DeckLintView["intentLock"] = null;
  if (Array.isArray(parsed.data.intent_lock)) {
    intentLock = parsed.data.intent_lock.flatMap((value) => {
      if (
        !isObject(value) ||
        typeof value.path !== "string" ||
        typeof value.ok !== "boolean"
      ) {
        return [];
      }
      return [
        {
          path: value.path,
          ok: value.ok,
          expected: value.expected ?? null,
          actual: value.actual ?? null,
          reason: stringOrNull(value.reason),
        },
      ];
    });
  }

  return {
    ok: true,
    data: {
      toolError: stringOrNull(parsed.data.error),
      validate: {
        ok: typeof validate.ok === "boolean" ? validate.ok : null,
        exitCode: numberOrNull(validate.exit_code),
        stderrTail:
          typeof validate.stderr_tail === "string" ? validate.stderr_tail : "",
      },
      meshPreview: isObject(parsed.data.mesh_preview)
        ? parsed.data.mesh_preview
        : null,
      lints,
      intentLock,
      defaultsDiff: isObject(parsed.data.defaults_diff)
        ? parsed.data.defaults_diff
        : null,
    },
    raw: parsed.raw,
  };
}

export interface GenerateResultView {
  status: "accepted" | "uncertain" | "tool_error" | "exhausted" | "unknown";
  question: string | null;
  iterations: number | null;
  deckPath: string | null;
  error: string | null;
  lint: Record<string, unknown> | null;
}

export function parseGenerateResult(stdout: string): Parsed<GenerateResultView> {
  const parsed = parseJsonObject(stdout);
  if (!parsed.ok) return parsed;

  const knownStatuses = new Set([
    "accepted",
    "uncertain",
    "tool_error",
    "exhausted",
  ]);
  const status =
    typeof parsed.data.status === "string" &&
    knownStatuses.has(parsed.data.status)
      ? (parsed.data.status as GenerateResultView["status"])
      : "unknown";
  const lintValue = isObject(parsed.data.lint)
    ? parsed.data.lint
    : isObject(parsed.data.last_lint)
      ? parsed.data.last_lint
      : null;

  return {
    ok: true,
    data: {
      status,
      question: stringOrNull(parsed.data.question),
      iterations: numberOrNull(parsed.data.iterations),
      deckPath: stringOrNull(parsed.data.deck_path),
      error: stringOrNull(parsed.data.error),
      lint: lintValue,
    },
    raw: parsed.raw,
  };
}

export interface DigestView {
  run: Record<string, unknown> | null;
  frozen: Record<string, unknown> | null;
  derived: Record<string, unknown> | null;
  historyNSamples: number | null;
  historyMissing: string[];
  /** history.series flattened: one row per series, values = its reduction dict. */
  seriesReductions: Array<{
    name: string;
    values: Record<string, unknown>;
  }>;
  tracks: Record<string, unknown> | null;
  notes: string[];
}

export function parseDigest(stdout: string): Parsed<DigestView> {
  const parsed = parseJsonObject(stdout);
  if (!parsed.ok) return parsed;

  const history = isObject(parsed.data.history) ? parsed.data.history : {};
  const series = isObject(history.series) ? history.series : {};
  const seriesReductions = Object.entries(series)
    .filter(
      (entry): entry is [string, Record<string, unknown>] => isObject(entry[1]),
    )
    .map(([name, values]) => ({ name, values }))
    .sort((left, right) => left.name.localeCompare(right.name));

  return {
    ok: true,
    data: {
      run: isObject(parsed.data.run) ? parsed.data.run : null,
      frozen: isObject(parsed.data.frozen_config)
        ? parsed.data.frozen_config
        : null,
      derived: isObject(parsed.data.derived) ? parsed.data.derived : null,
      historyNSamples: numberOrNull(history.n_samples),
      historyMissing: stringArray(history.missing),
      seriesReductions,
      tracks: isObject(history.tracks) ? history.tracks : null,
      notes: stringArray(parsed.data.notes),
    },
    raw: parsed.raw,
  };
}

export interface ZoningReportView {
  applicable: boolean | null;
  reason: string | null;
  geometry: string | null;
  nSnapshotsUsed: number | null;
  critical: Record<string, unknown> | null;
  ablated: Record<string, unknown> | null;
  lintA: Record<string, unknown> | null;
  /** lint_a's boolean verdict when the payload carries one. */
  lintAOk: boolean | null;
  pressureTrack: Record<string, unknown> | null;
  notes: string[];
}

export function parseZoningReport(stdout: string): Parsed<ZoningReportView> {
  const parsed = parseJsonObject(stdout);
  if (!parsed.ok) return parsed;

  const lintA = isObject(parsed.data.lint_a) ? parsed.data.lint_a : null;
  return {
    ok: true,
    data: {
      applicable:
        typeof parsed.data.applicable === "boolean"
          ? parsed.data.applicable
          : null,
      reason: stringOrNull(parsed.data.reason),
      geometry: stringOrNull(parsed.data.geometry),
      nSnapshotsUsed: numberOrNull(parsed.data.n_snapshots_used),
      critical: isObject(parsed.data.critical) ? parsed.data.critical : null,
      ablated: isObject(parsed.data.ablated) ? parsed.data.ablated : null,
      lintA,
      lintAOk: lintA !== null && typeof lintA.ok === "boolean" ? lintA.ok : null,
      pressureTrack: isObject(parsed.data.pressure_track)
        ? parsed.data.pressure_track
        : null,
      notes: stringArray(parsed.data.notes),
    },
    raw: parsed.raw,
  };
}

export interface PromoteZoningView {
  band: Record<string, unknown> | null;
  evidence: Record<string, unknown> | null;
  conservatismNote: string | null;
  /** any remaining top-level scalar fields, for generic display */
  extra: Record<string, unknown>;
}

export function parsePromoteZoning(stdout: string): Parsed<PromoteZoningView> {
  const parsed = parseJsonObject(stdout);
  if (!parsed.ok) return parsed;

  const extra: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(parsed.data)) {
    if (key === "band" || key === "evidence" || key === "conservatism_note") {
      continue;
    }
    if (
      value === null ||
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean"
    ) {
      extra[key] = value;
    }
  }

  return {
    ok: true,
    data: {
      band: isObject(parsed.data.band) ? parsed.data.band : null,
      evidence: isObject(parsed.data.evidence) ? parsed.data.evidence : null,
      conservatismNote: stringOrNull(parsed.data.conservatism_note),
      extra,
    },
    raw: parsed.raw,
  };
}

export interface JournalTailView {
  iterations: number;
  lastKind: string | null;
  lastTs: string | null;
  entries: Array<{ ts: string | null; kind: string }>;
}

export function parseJournalTail(text: string): JournalTailView {
  let iterations = 0;
  const entries: JournalTailView["entries"] = [];

  for (const line of text.split(/\r?\n/)) {
    if (line.trim().length === 0) continue;
    try {
      const value: unknown = JSON.parse(line);
      if (!isObject(value) || typeof value.kind !== "string") continue;
      if (value.kind === "deck_iteration") iterations += 1;
      entries.push({ ts: stringOrNull(value.ts), kind: value.kind });
    } catch {
      // A partial or damaged JSONL record does not invalidate the readable tail.
    }
  }

  const tail = entries.slice(-30);
  const last = tail[tail.length - 1];
  return {
    iterations,
    lastKind: last?.kind ?? null,
    lastTs: last?.ts ?? null,
    entries: tail,
  };
}
