import type { Backend, ExecOpts } from "@tenryu-common/backend/types";
import type { ServerProfile } from "@tenryu-common/core/profiles";
import { remoteDirname, shQuote, shQuotePath } from "@tenryu-common/core/ssh";
import { t } from "../../i18n";
import { extractGuiState } from "./roundtrip";
import { migrateFormState, type FormState } from "./formState";
import { generateDeck } from "./generate";
import { mapRecordedDeck, type DeckRecord, type ImportRule } from "./deckImport";

// Ship the import verb and its stdlib-only transitive imports from this app.
export const IMPORT_HARNESS_FILES = ["assist.py", "__init__.py", "cli.py", "providers.py", "config.py", "tomlmini.py", "journal.py", "deck_import.py", "deck_import_runtime.py", "deck_import_mesh.py"];
export interface DeckImportOptions {
  venue?: "local" | "server";
  workingDirectory?: string;
  environment?: Record<string,string>;
  sourceFilename?: string;
  profile?: ServerProfile | null;
  autoDetectWorkingDirectory?: boolean;
}
export function parseImportEnvironment(text: string): Record<string,string> {
  const entries = text.split(/\r?\n/).filter(line=>line.trim()).map(line=>{
    const match = /^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$/.exec(line);
    if (!match || line.includes("\0")) throw new Error(t().deck.importEnvironmentInvalid);
    return [match[1], match[2]];
  });
  return Object.fromEntries(entries.filter(([,value])=>value !== ""));
}

/** The deck's directory and up to `levels` ancestors, root excluded, in order, without duplicates. */
export function workingDirectoryCandidates(deckPaths: string[], levels = 3): string[] {
  const out: string[] = [];
  for (const deckPath of deckPaths) {
    let dir = remoteDirname(deckPath);
    for (let i = 0; i <= levels && dir !== "/" && dir !== "."; i++) {
      if (!out.includes(dir)) out.push(dir);
      dir = remoteDirname(dir);
    }
  }
  return out;
}

/** Order server copies of a deck by how many trailing path components they share with the local path (stable). */
export function rankServerCopies(localPath: string, copies: string[]): string[] {
  const local = localPath.split("/").filter(Boolean).reverse();
  const shared = (copy: string) => {
    const parts = copy.split("/").filter(Boolean).reverse();
    let n = 0;
    while (n < local.length && n < parts.length && local[n] === parts[n]) n++;
    return n;
  };
  return copies.map((copy, index) => ({copy, index, score: shared(copy)}))
    .sort((a, b) => b.score - a.score || a.index - b.index).map(entry => entry.copy);
}

export async function loadDeckText(
  backend: Backend, source: string, filename?: string | null, repoRoot?: string | null,
  options: DeckImportOptions = {},
): Promise<FormState> {
  const embedded = extractGuiState(source);
  if (embedded.ok) return migrateFormState(embedded.state);
  if (embedded.reason !== "no-marker") throw new Error(embedded.reason === "bad-json" ? t().deck.loadErrBadJson : t().deck.loadErrBadVersion);
  const venue = options.venue ?? "local";
  const profile = venue === "server" ? options.profile : null;
  const genericHint = t().deck.importSettingsHint;
  let settingsHint = (venue === "server" && !options.workingDirectory?.trim()
    ? `${t().deck.importDefaultDirectoryHint}\n` : "") + genericHint;
  if (venue === "server" && !profile) throw new Error(t().deck.importServerRequired);
  const execute = (script: string, opts: ExecOpts) => profile
    ? backend.exec(profile, ["bash", "-lc", script], opts)
    : backend.execLocal(["bash", "-lc", script], opts);
  const harness = await backend.assistHarnessDir();
  const setup = profile
    ? `mkdir -p ${shQuotePath(profile.runDir)} && cd ${shQuotePath(profile.runDir)} && mktemp -d "$PWD/tenryu-studio-import.XXXXXX"`
    : 'mktemp -d "${TMPDIR:-/tmp}/tenryu-studio-import.XXXXXX"';
  const temp = await execute(`command -v python3 >/dev/null || { echo "python3 is required on the import venue" >&2; exit 127; }; ${setup}`, {timeoutMs:15000});
  if (temp.timedOut) throw new Error(`${t().deck.importTimeout}\n${settingsHint}`);
  if (temp.code !== 0) throw new Error(`${t().deck.importFailed}\n${temp.stderr || t().deck.importPythonRequired}\n${settingsHint}`);
  const directory = temp.stdout.trim();
  if (!directory.startsWith("/") || directory.includes("\n") || !/\/tenryu-studio-import\.[A-Za-z0-9]+$/.test(directory)) throw new Error("Invalid import temporary directory");
  const requestPath = `${directory}/request.json`;
  const write = (path: string, content: string) => profile ? backend.uploadText(profile,path,content) : backend.writeLocalText(path,content);
  let stablePath = "", workingDirectory = "";
  const environment = options.environment ?? {};
  const needsPlanner = /(?:tools\.mesh_planner|TENRYU_REPO)/.test(source);
  const importRepo = needsPlanner ? repoRoot : null;
  let executableHarness = harness;
  async function evaluate(operation: string, candidate?: string): Promise<DeckRecord> {
    await write(requestPath, JSON.stringify({source, filename:stablePath, workingDirectory, environment, operation, candidate}));
    const repo = importRepo ? ` --repo-root ${shQuote(importRepo)}` : "";
    const command = `python3 ${shQuote(`${executableHarness}/assist.py`)} import-deck --request ${shQuote(requestPath)} --timeout 45${repo}`;
    const result = await execute(profile ? `cd ${shQuote(directory)} && ${command}` : command, {timeoutMs:150000});
    if (result.timedOut) throw new Error(`${t().deck.importTimeout}\n${settingsHint}`);
    let parsed: DeckRecord;
    try { parsed = JSON.parse(result.stdout); }
    catch { throw new Error(`${t().deck.importFailed}\n${result.stderr || result.stdout || `exit ${result.code}`}\n${settingsHint}`); }
    if (operation === "compare" && parsed.ok === false && Array.isArray(parsed.failures) && result.code === 2) return parsed;
    if (!parsed.ok || result.code !== 0) throw new Error(`${t().deck.importFailed}\n${parsed.error || parsed.failures?.map(r=>`${r.path.join(".")}: ${r.reason}`).join("\n") || result.stderr}\n${settingsHint}`);
    return parsed;
  }
  try {
    const localPath = filename || (profile ? `${directory}/pasted_deck.py` : `${await backend.appConfigDir()}/pasted_deck.py`);
    // The selected local filename remains provenance; a server __file__ can be
    // supplied explicitly when the deck depends on its original server location.
    stablePath = options.sourceFilename || (profile ? `${directory}/${localPath.split("/").pop()}` : localPath);
    // On the server venue the local deck directory does not exist there; the
    // upload directory is the default until the user names the run directory.
    workingDirectory = options.workingDirectory?.trim() || (profile ? directory : remoteDirname(localPath));
    if (profile) {
      executableHarness = `${directory}/tools/assist`;
      for (const name of IMPORT_HARNESS_FILES) {
        await write(`${executableHarness}/${name}`, await backend.readLocalText(`${harness}/${name}`,4*1024*1024));
      }
      await write(`${directory}/${localPath.split("/").pop()}`,source);
    }
    let record: DeckRecord | null = null;
    if (options.autoDetectWorkingDirectory && !options.workingDirectory?.trim() && filename) {
      // Try the directories the deck is likely to be run from — its own and its
      // ancestors, or those of same-named copies on the server — first success wins.
      settingsHint = genericHint;
      let candidates = profile ? [] : workingDirectoryCandidates([filename]);
      if (profile) {
        const search = await execute(`find "$HOME" -maxdepth 8 -name ${shQuote(filename.split("/").pop() ?? "")} -not -path "*/tenryu_gui_runs/*" 2>/dev/null | head -20`, {timeoutMs:30000});
        const copies = search.stdout.split("\n").map(line=>line.trim()).filter(line=>line.startsWith("/"));
        candidates = workingDirectoryCandidates(rankServerCopies(filename, copies));
      }
      const tried: string[] = [];
      let failure: unknown = null;
      for (const candidate of candidates.slice(0, 8)) {
        workingDirectory = candidate;
        tried.push(candidate);
        try { record = await evaluate("record"); failure = null; break; }
        catch (error) { failure = error; }
      }
      if (!record) {
        settingsHint = profile ? `${t().deck.importDefaultDirectoryHint}\n${genericHint}` : genericHint;
        workingDirectory = profile ? directory : remoteDirname(localPath);
        if (tried.length) throw new Error(`${failure instanceof Error ? failure.message : String(failure)}\n${t().deck.importAutoDetectTried(tried.join("\n"))}`);
        record = await evaluate("record");
      }
    } else {
      record = await evaluate("record");
    }
    stablePath = record.filename || stablePath;
    workingDirectory = record.workingDirectory || workingDirectory;
    const {form, bindings, notes} = mapRecordedDeck(record);
    const verified = await evaluate("verify", generateDeck(form));
    form.deckImport = {source, filename:stablePath, baseline:structuredClone(form), bindings, rules:verified.rules as ImportRule[], notes,
      evaluation:{venue, workingDirectory, environment:{...environment}, filename:stablePath, repoRoot:importRepo,
        ...(profile ? {profile:{id:profile.id,name:profile.name,host:profile.host}} : {})}};
    const comparison = await evaluate("compare", generateDeck(form));
    if (!comparison.ok) {
      const failures = comparison.failures ?? [];
      const retained = failures.filter(failure=>failure.reason.startsWith("Sampling failed:") && verified.rules?.some(rule=>rule.kind==="passthrough" && rule.path.every((part,i)=>failure.path[i]===part)));
      if (!failures.length || retained.length !== failures.length) throw new Error(`${t().deck.importFailed}\n${failures.map(r=>`${r.path.join(".")}: ${r.reason}`).join("\n")}\n${settingsHint}`);
      form.deckImport.unverified = retained;
    }
    return form;
  } finally {
    // Only the directory created by this import is removed. Cleanup must not
    // replace the evaluator's traceback if the connection has been lost.
    await execute(`rm -rf ${shQuote(directory)}`, {timeoutMs:15000}).catch(()=>undefined);
  }
}
