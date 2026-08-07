import { GUI_SCHEMA_VERSION, type FormState } from "./formState";

const STATE_RE = /^# TENRYU-GUI-STATE: (.+)$/m;

export type ExtractResult =
  | { ok: true; state: FormState }
  | { ok: false; reason: "no-marker" | "bad-json" | "bad-version"; detail?: string };

/** Recover the embedded GUI form state from a generated deck. */
export function extractGuiState(deckText: string): ExtractResult {
  const m = STATE_RE.exec(deckText);
  if (!m) return { ok: false, reason: "no-marker" };
  let parsed: unknown;
  try {
    parsed = JSON.parse(m[1]);
  } catch (err) {
    return { ok: false, reason: "bad-json", detail: String(err) };
  }
  const state = parsed as FormState;
  if (state.guiSchemaVersion !== GUI_SCHEMA_VERSION) {
    return { ok: false, reason: "bad-version", detail: String((state as { guiSchemaVersion?: unknown }).guiSchemaVersion) };
  }
  return { ok: true, state };
}
