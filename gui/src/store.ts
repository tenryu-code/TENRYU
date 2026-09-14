import { create } from "zustand";
import { newId } from "@tenryu-common/core/ids";
import { getBackend } from "@tenryu-common/backend";
import type { Backend, ExecResult } from "@tenryu-common/backend/types";
import RUN_DETACHED_SH from "./core/assets/run_detached.sh?raw";
import { logLine, logOp } from "@tenryu-common/core/applog";
import { defaultFormState, migrateFormState, validateFormState, type FormState } from "./core/deck/formState";
import { generateDeck, generateDeckForSave } from "./core/deck/generate";
import { extractGuiState } from "./core/deck/roundtrip";
import { parseMeshPreview, type MeshPreviewData } from "./core/meshPreviewParse";
import { profileBinMissing, type ServerProfile } from "@tenryu-common/core/profiles";
import { parseLastProgress } from "./core/progressParse";
import { parseLsOutput, type RemoteDirEntry } from "@tenryu-common/core/remoteLs";
import { parseHistory, type HistorySeries } from "@tenryu-common/core/results/historyParse";
import { parseProfile, type ProfileSnapshot } from "@tenryu-common/core/results/profileParse";
import { parseSnapshotMesh, type SnapshotMesh2D } from "@tenryu-common/core/results/snapshotMesh";
import { parseTmatGroupBounds } from "@tenryu-common/core/results/tmatBounds";
import { isTerminal, parseStatusFile, type RunRecord } from "@tenryu-common/core/runstate";
import { joinRemote, shQuotePath } from "@tenryu-common/core/ssh";
import { toCanonical } from "./core/units";
import { parseValidateOutput, type ValidateResult } from "./core/validateParse";
import {
  buildAskScript,
  buildCancelScript,
  buildGenerateScript,
  buildLatestOutputDirScript,
  buildMirrorProbeScript,
  buildMirrorSyncScript,
  buildMkdirScript,
  buildProbeAssistScript,
  buildRemoteAssistScript,
  buildRemoteWrapperEnv,
  buildStatusScript,
} from "./core/assist/commands";
import {
  parseAskResult,
  parseAssistStatus,
  parseDeckLint,
  parseDigest,
  parseGenerateResult,
  parseJournalTail,
  parsePromoteZoning,
  parseZoningReport,
  type AskChecksView,
  type AssistStatusView,
  type DeckLintView,
  type DigestView,
  type PromoteZoningView,
  type ZoningReportView,
} from "./core/assist/parse";
import {
  addPreset,
  buildAssistantToml,
  formFromStatus,
  validateAssistConfigForm,
  type AssistConfigForm,
  type PresetKey,
} from "./core/assist/config";
import { setLang, t, type Lang } from "./i18n";

let backendOverride: Backend | null = null;
let assistConfigPathCache: string | null = null;
/** Test hook: inject a fake backend (call before first store action). */
export function __setBackendForTest(b: Backend | null): void {
  backendOverride = b;
  homeCache.clear();
  repoRootCache.clear();
  assistConfigPathCache = null;
  assistProbeCache.clear();
  if (assistJournalTimer !== null) {
    clearInterval(assistJournalTimer);
    assistJournalTimer = null;
  }
  if (assistAskJournalTimer !== null) {
    clearInterval(assistAskJournalTimer);
    assistAskJournalTimer = null;
  }
}
function be(): Backend {
  return backendOverride ?? getBackend();
}

export type View = "form" | "servers" | "history" | "assist";
export type SectionKey = "presets" | "basic" | "mesh" | "materials" | "physics" | "laser" | "output";

export interface ConnTestResult {
  ok: boolean;
  binOk: boolean;
  binRuns: boolean;
  gpu: string;
  runDirOk: boolean;
  detail: string;
}

export interface RemoteDirListing {
  ok: boolean;
  path: string;
  entries: RemoteDirEntry[];
  error: string;
}

const MAX_RUN_RECORDS = 50;

const homeCache = new Map<string, string>();
const repoRootCache = new Map<string, string | null>();
const assistProbeCache = new Map<string, boolean>();
let assistJournalTimer: ReturnType<typeof setInterval> | null = null;
let assistAskJournalTimer: ReturnType<typeof setInterval> | null = null;

async function resolveHome(profile: ServerProfile): Promise<string> {
  if (profile.transport === "local") return "";
  const cached = homeCache.get(profile.id);
  if (cached !== undefined) return cached;
  const r = await be().exec(profile, ["printenv", "HOME"], { timeoutMs: 15000 });
  const home = r.stdout.trim().split("\n").pop() ?? "";
  if (r.code !== 0 || !home.startsWith("/")) {
    throw new Error(`リモート HOME を解決できません: ${r.stderr.trim() || `exit=${r.code}`}`);
  }
  homeCache.set(profile.id, home);
  return home;
}

async function resolveAssistConfigPath(): Promise<string> {
  if (assistConfigPathCache === null) {
    assistConfigPathCache = `${await be().appConfigDir()}/assistant.toml`;
  }
  return assistConfigPathCache;
}

function repoRootScript(tenryuBin: string): string {
  return `d=$(dirname ${shQuotePath(tenryuBin)}); for i in 1 2 3 4 5 6 7 8; do if [ -f "$d/tools/mesh_planner.py" ]; then echo "$d"; break; fi; d=$(dirname "$d"); done`;
}

/** Locate the TENRYU checkout on the server by walking up from tenryuBin (cached). */
async function resolveRepoRoot(profile: ServerProfile): Promise<string | null> {
  if (profile.transport === "local" || profileBinMissing(profile)) return null;
  const key = `${profile.id}\0${profile.tenryuBin}`;
  const cached = repoRootCache.get(key);
  if (cached !== undefined) return cached;
  const script = repoRootScript(profile.tenryuBin);
  let resolved: string | null = null;
  try {
    const r = await be().exec(profile, ["bash", "-lc", script], { timeoutMs: 15000 });
    const root = r.code === 0 ? (r.stdout.trim().split("\n").pop() ?? "") : "";
    resolved = root.startsWith("/") ? root : null;
  } catch {
    resolved = null;
  }
  repoRootCache.set(key, resolved);
  return resolved;
}

function withRepoEnv(root: string | null, argv: string[]): string[] {
  return root === null ? argv : ["env", `TENRYU_REPO=${root}`, ...argv];
}

/** Expand a leading ~ against the profile's remote home (ssh) — local requires absolute. */
async function absRemotePath(profile: ServerProfile, p: string): Promise<string> {
  if (!p.startsWith("~")) return p;
  const home = await resolveHome(profile);
  if (home === "") throw new Error("ローカル接続では絶対パスが必要です");
  return p === "~" ? home : home + p.slice(1);
}

function deriveDeck(form: FormState): { formErrors: string[]; deck: string } {
  const formErrors = validateFormState(form);
  if (formErrors.length > 0) {
    return { formErrors, deck: "" };
  }
  try {
    return { formErrors, deck: generateDeck(form) };
  } catch (err) {
    // Fail soft: a generator exception must never take down the UI (a
    // persisted broken form would then brick the app on relaunch too).
    return { formErrors: [err instanceof Error ? err.message : String(err)], deck: "" };
  }
}

function nowStamp(): string {
  return new Date()
    .toISOString()
    .replace(/[-:TZ.]/g, "")
    .slice(0, 14);
}

export interface AppState {
  profiles: ServerProfile[];
  profilesLoaded: boolean;
  loadError: string | null;
  currentProfileId: string | null;
  view: View;
  section: SectionKey;
  lang: Lang;
  chatOpen: boolean;
  form: FormState;
  formErrors: string[];
  deck: string;
  deckIoStatus: { kind: "saved" | "loaded" | "error"; detail: string } | null;
  namelistPath: string | null;
  validating: boolean;
  validateResult: ValidateResult | null;
  validateSentTo: string | null;
  meshPreview: MeshPreviewData | null;
  meshPreviewBusy: boolean;
  meshPreviewError: string | null;
  meshSnapshot: SnapshotMesh2D | null;
  meshSnapshotBusy: boolean;
  meshSnapshotError: string | null;
  tmatBoundsBusy: boolean;
  tmatBoundsError: string | null;
  connTesting: boolean;
  connResult: ConnTestResult | null;
  runs: RunRecord[];
  runLogs: Record<string, string>;
  runRates: Record<string, number>;
  histories: Record<string, { status: "loading" | "ready" | "error"; error?: string; data?: HistorySeries }>;
  profileLists: Record<string, { status: "loading" | "ready" | "error"; error?: string; paths?: string[] }>;
  profileSnaps: Record<string, { status: "loading" | "ready" | "error"; error?: string; data?: ProfileSnapshot }>;
  starting: boolean;
  assistLocalRepo: string;
  assistStatus: { status: "idle" | "loading" | "ready" | "error"; view?: AssistStatusView; raw?: string; error?: string };
  assistMirror: {
    status: "idle" | "syncing" | "ready" | "error";
    root: string | null;
    profileId: string | null;
    lastSyncIso: string | null;
    errorCode: string | null;
    errorDetail: string | null;
  };
  assistConfig: {
    form: AssistConfigForm | null;
    path: string | null;
    status: "idle" | "saving" | "saved" | "error";
    savedPath: string | null;
    error: string | null;
  };
  assistLint: { status: "idle" | "running" | "ready" | "error"; view?: DeckLintView; raw?: string; error?: string; exitOk?: boolean };
  assistSpec: string;
  assistUseTemplate: boolean;
  assistMaxIters: number;
  assistIntentJson: string;
  assistGen: {
    phase: "idle" | "running" | "accepted" | "uncertain" | "error";
    workdir: string | null;
    iterations: number;
    lastKind: string | null;
    question: string | null;
    deckText: string | null;
    deckName: string | null;
    errorCode: string | null;
    errorDetail: string | null;
    resultRaw: string | null;
    lint: Record<string, unknown> | null;
  };
  assistQuestion: string;
  assistAsk: {
    phase: "idle" | "running" | "answered" | "error";
    workdir: string | null;
    turns: Array<{
      turn: number;
      question: string;
      answer: string;
      checks: AskChecksView | null;
    }>;
    /** The question in flight (or the one the last error belongs to). */
    pendingQuestion: string | null;
    lastKind: string | null;
    errorCode: string | null;
    errorDetail: string | null;
    resultRaw: string | null;
  };
  assistDiag: Record<string, {
    digest?: { status: "running" | "ready" | "error"; view?: DigestView; raw?: string; error?: string };
    zoning?: { status: "running" | "ready" | "error"; view?: ZoningReportView; raw?: string; error?: string };
    promote?: { status: "running" | "ready" | "error"; view?: PromoteZoningView; raw?: string; error?: string };
  }>;
  assistDeckValidate: { status: "idle" | "running" | "ready" | "error"; result?: ValidateResult; sentTo?: string | null };

  loadInitial(): Promise<void>;
  setView(v: View): void;
  setSection(s: SectionKey): void;
  setChatOpen(open: boolean): void;
  toggleChat(): void;
  setUiLang(lang: Lang): Promise<void>;
  selectProfile(id: string | null): Promise<void>;
  upsertProfile(p: ServerProfile): Promise<void>;
  deleteProfile(id: string): Promise<void>;
  updateForm(mut: (f: FormState) => void): void;
  loadForm(f: FormState): void;
  listRemoteDir(path: string): Promise<RemoteDirListing>;
  runValidate(): Promise<void>;
  runMeshPreview(): Promise<void>;
  fetchMeshSnapshot(): Promise<void>;
  fetchTmatGroupBounds(): Promise<void>;
  saveNamelist(): Promise<void>;
  saveNamelistAs(): Promise<void>;
  loadNamelist(): Promise<void>;
  saveDeckToFile(): Promise<boolean>;
  saveTextAs(suggestedName: string, content: string): Promise<boolean>;
  testConnection(p: ServerProfile): Promise<void>;
  startRun(override?: { deck: string; name: string }): Promise<void>;
  restartRun(runId: string): Promise<void>;
  pollRunOnce(runId: string): Promise<void>;
  pollActiveRuns(): Promise<void>;
  stopRun(runId: string): Promise<void>;
  fetchLog(runId: string): Promise<void>;
  deleteRunRecord(runId: string): Promise<void>;
  fetchHistory(runId: string): Promise<void>;
  fetchProfileList(runId: string): Promise<void>;
  fetchProfileSnap(runId: string, index: number): Promise<void>;
  setAssistLocalRepo(path: string): Promise<void>;
  syncAssistMirror(): Promise<boolean>;
  resolveAssistRepo(): Promise<string | null>;
  setAssistSpec(spec: string): void;
  setAssistUseTemplate(v: boolean): void;
  setAssistMaxIters(n: number): void;
  setAssistIntentJson(v: string): void;
  fetchAssistStatus(): Promise<void>;
  initAssistConfigForm(): void;
  setAssistConfigForm(form: AssistConfigForm): void;
  addAssistProviderPreset(key: PresetKey): void;
  saveAssistConfig(): Promise<void>;
  runAssistLint(): Promise<void>;
  runAssistDiag(runId: string, verb: "digest" | "zoning" | "promote"): Promise<void>;
  setAssistQuestion(q: string): void;
  askAssistQuestion(): Promise<void>;
  cancelAssistQuestion(): Promise<void>;
  resetAssistConversation(): void;
  generateAssistDeck(): Promise<void>;
  answerAssistClarification(answer: string): Promise<void>;
  cancelAssistGeneration(): Promise<void>;
  resetAssistGeneration(): void;
  validateAssistDeck(): Promise<void>;
  runAssistGeneratedDeck(): Promise<void>;
}

export function currentProfile(s: Pick<AppState, "profiles" | "currentProfileId">): ServerProfile | null {
  return s.profiles.find((p) => p.id === s.currentProfileId) ?? null;
}

const initialForm = defaultFormState();
const initialDerived = deriveDeck(initialForm);

const rateMemo = new Map<string, { step: number; wall: number }>();

const initialAssistGen: AppState["assistGen"] = {
  phase: "idle",
  workdir: null,
  iterations: 0,
  lastKind: null,
  question: null,
  deckText: null,
  deckName: null,
  errorCode: null,
  errorDetail: null,
  resultRaw: null,
  lint: null,
};

const initialAssistAsk: AppState["assistAsk"] = {
  phase: "idle",
  workdir: null,
  turns: [],
  pendingQuestion: null,
  lastKind: null,
  errorCode: null,
  errorDetail: null,
  resultRaw: null,
};

const initialAssistMirror: AppState["assistMirror"] = {
  status: "idle",
  root: null,
  profileId: null,
  lastSyncIso: null,
  errorCode: null,
  errorDetail: null,
};

export const useApp = create<AppState>()((set, get) => {
  function patchRun(runId: string, patch: Partial<RunRecord>): void {
    set({ runs: get().runs.map((r) => (r.id === runId ? { ...r, ...patch } : r)) });
  }

  function stopAssistJournalTimer(): void {
    if (assistJournalTimer !== null) {
      clearInterval(assistJournalTimer);
      assistJournalTimer = null;
    }
  }

  function stopAssistAskJournalTimer(): void {
    if (assistAskJournalTimer !== null) {
      clearInterval(assistAskJournalTimer);
      assistAskJournalTimer = null;
    }
  }

  function failGen(code: string, detail?: string): void {
    set({
      assistGen: {
        ...initialAssistGen,
        phase: "error",
        errorCode: code,
        errorDetail: detail ?? null,
      },
    });
  }

  function failAsk(code: string, detail?: string): void {
    const cur = get().assistAsk;
    const draft = get().assistQuestion;
    const errored: AppState["assistAsk"] = {
      ...cur,
      phase: "error",
      errorCode: code,
      errorDetail: detail ?? null,
    };
    // Give the question back to the input box when a launched ask fails, unless
    // the user already started typing a replacement. Pre-launch failures
    // (NO_QUESTION / NO_SOURCE / NO_LOCAL_ASSIST) never reach this branch:
    // the phase is not yet "running" when they are reported.
    if (cur.phase === "running" && cur.pendingQuestion !== null && draft.trim() === "") {
      set({ assistAsk: errored, assistQuestion: cur.pendingQuestion });
      return;
    }
    set({ assistAsk: errored });
  }

  function errorWithDetail(code: string, detail: string): string {
    const trimmed = detail.trim();
    return trimmed.length > 0 ? `${code}: ${trimmed}` : code;
  }

  async function probeRemoteAssist(profile: ServerProfile, root: string): Promise<boolean> {
    const key = `${profile.id}\0${root}`;
    const cached = assistProbeCache.get(key);
    if (cached !== undefined) return cached;
    const r = await be().exec(profile, ["bash", "-lc", buildProbeAssistScript(root)], {
      timeoutMs: 30000,
    });
    const ok = r.stdout.includes("ASSIST_OK");
    assistProbeCache.set(key, ok);
    return ok;
  }

  async function validateDeckWith(
    profile: ServerProfile,
    root: string | null,
    deckText: string,
    name: string,
  ): Promise<{ result: ValidateResult; sentTo: string | null }> {
    const remoteDeck = joinRemote(profile.runDir, `validate_scratch/${name}.py`);
    try {
      await be().uploadText(profile, remoteDeck, deckText);
      const r: ExecResult = await be().exec(
        profile,
        withRepoEnv(root, [profile.tenryuBin, "validate", remoteDeck]),
        { timeoutMs: 120000 },
      );
      return {
        result: parseValidateOutput(r.stdout, r.stderr, r.code),
        sentTo: `${profile.name}:${remoteDeck}`,
      };
    } catch (err) {
      return {
        result: { ok: false, summary: [], errors: [String(err)], warnings: [], raw: "" },
        sentTo: null,
      };
    }
  }

  async function persistRuns(): Promise<void> {
    const s = get();
    try {
      await be().saveSettings({
        lastProfileId: s.currentProfileId ?? undefined,
        runs: s.runs.slice(0, MAX_RUN_RECORDS),
        lang: s.lang,
        assistLocalRepo: s.assistLocalRepo || undefined,
      });
    } catch {
      /* non-fatal */
    }
  }

  return {
    profiles: [],
    profilesLoaded: false,
    loadError: null,
    currentProfileId: null,
    view: "form",
    section: "basic",
    lang: "ja",
    chatOpen: false,
    form: initialForm,
    formErrors: initialDerived.formErrors,
    deck: initialDerived.deck,
    deckIoStatus: null,
    namelistPath: null,
    validating: false,
    validateResult: null,
    validateSentTo: null,
    meshPreview: null,
    meshPreviewBusy: false,
    meshPreviewError: null,
    meshSnapshot: null,
    meshSnapshotBusy: false,
    meshSnapshotError: null,
    tmatBoundsBusy: false,
    tmatBoundsError: null,
    connTesting: false,
    connResult: null,
    runs: [],
    runLogs: {},
    runRates: {},
    histories: {},
    profileLists: {},
    profileSnaps: {},
    starting: false,
    assistLocalRepo: "",
    assistStatus: { status: "idle" },
    assistMirror: initialAssistMirror,
    assistConfig: {
      form: null,
      path: null,
      status: "idle",
      savedPath: null,
      error: null,
    },
    assistLint: { status: "idle" },
    assistSpec: "",
    assistUseTemplate: true,
    assistMaxIters: 10,
    assistIntentJson: "",
    assistGen: initialAssistGen,
    assistQuestion: "",
    assistAsk: initialAssistAsk,
    assistDiag: {},
    assistDeckValidate: { status: "idle" },

    async loadInitial() {
      try {
        const [profiles, settings] = await Promise.all([be().listProfiles(), be().getSettings()]);
        const currentProfileId =
          settings.lastProfileId && profiles.some((p) => p.id === settings.lastProfileId)
            ? settings.lastProfileId
            : profiles.length > 0
              ? profiles[0].id
              : null;
        const lang: Lang = settings.lang === "en" ? "en" : "ja";
        setLang(lang);
        const form = get().form;
        set({
          profiles,
          currentProfileId,
          profilesLoaded: true,
          loadError: null,
          runs: settings.runs ?? [],
          lang,
          assistLocalRepo: settings.assistLocalRepo ?? "",
          ...deriveDeck(form),
        });
      } catch (err) {
        set({ profilesLoaded: true, loadError: String(err) });
      }
    },

    setView(v) {
      set({ view: v });
    },

    setSection(s) {
      set({ section: s, view: "form" });
    },

    setChatOpen(open) {
      set({ chatOpen: open });
    },

    toggleChat() {
      set({ chatOpen: !get().chatOpen });
    },

    async setUiLang(lang) {
      setLang(lang);
      const form = get().form;
      set({ lang, ...deriveDeck(form) });
      try {
        const s = get();
        await be().saveSettings({
          lastProfileId: s.currentProfileId ?? undefined,
          runs: s.runs.slice(0, MAX_RUN_RECORDS),
          lang,
          assistLocalRepo: s.assistLocalRepo || undefined,
        });
      } catch {
        /* non-fatal */
      }
    },

    async selectProfile(id) {
      set({
        currentProfileId: id,
        connResult: null,
        assistMirror: get().currentProfileId === id
          ? get().assistMirror
          : initialAssistMirror,
      });
      await persistRuns();
    },

    async upsertProfile(p) {
      const profiles = [...get().profiles];
      const i = profiles.findIndex((x) => x.id === p.id);
      if (i >= 0) profiles[i] = p;
      else profiles.push(p);
      set({ profiles });
      homeCache.delete(p.id);
      repoRootCache.clear();
      assistProbeCache.clear();
      await be().saveProfiles(profiles);
      if (get().currentProfileId === null) await get().selectProfile(p.id);
    },

    async deleteProfile(id) {
      const profiles = get().profiles.filter((x) => x.id !== id);
      const currentProfileId =
        get().currentProfileId === id ? (profiles[0]?.id ?? null) : get().currentProfileId;
      set({ profiles, currentProfileId, connResult: null });
      homeCache.delete(id);
      repoRootCache.clear();
      assistProbeCache.clear();
      await be().saveProfiles(profiles);
    },

    updateForm(mut) {
      const next = structuredClone(get().form);
      mut(next);
      set({ form: next, ...deriveDeck(next) });
    },

    loadForm(f) {
      const next = structuredClone(f);
      set({ form: next, ...deriveDeck(next), namelistPath: null });
    },

    async listRemoteDir(path) {
      const profile = currentProfile(get());
      if (profile === null) {
        return { ok: false, path, entries: [], error: "NO_PROFILE" };
      }
      let p = path.trim();
      if (p === "") p = "~";
      try {
        p = await absRemotePath(profile, p);
        const r = await be().exec(
          profile,
          ["bash", "-lc", "cd " + shQuotePath(p) + " && ls -1paL 2>/dev/null"],
          { timeoutMs: 20000 },
        );
        if (r.code !== 0 && r.stdout.trim() === "") {
          return {
            ok: false,
            path: p,
            entries: [],
            error: (r.stderr.trim() || "exit=" + r.code).slice(0, 300),
          };
        }
        return { ok: true, path: p, entries: parseLsOutput(r.stdout), error: "" };
      } catch (err) {
        return { ok: false, path: p, entries: [], error: String(err) };
      }
    },

    async saveDeckToFile() {
      const s = get();
      if (s.formErrors.length > 0 || s.deck.length === 0) return false;
      return (await be().saveLocalTextFile(`${s.form.main.name}.py`, s.deck, "TENRYU deck")) !== null;
    },

    async saveNamelistAs() {
      const s = get();
      const text = s.deck.length > 0 ? s.deck : generateDeckForSave(s.form, s.formErrors);
      try {
        const path = await be().saveLocalTextFile(`${s.form.main.name}.py`, text, "TENRYU deck");
        if (path === null) return;
        const fileName = path.split("/").pop() ?? path;
        set({
          namelistPath: path,
          deckIoStatus: {
            kind: "saved",
            detail: fileName + (s.formErrors.length > 0 ? t().deckIo.savedWithErrors : ""),
          },
        });
      } catch (err) {
        set({ deckIoStatus: { kind: "error", detail: String(err) } });
      }
    },

    async saveNamelist() {
      const s = get();
      const text = s.deck.length > 0 ? s.deck : generateDeckForSave(s.form, s.formErrors);
      if (s.namelistPath === null) {
        await get().saveNamelistAs();
        return;
      }
      try {
        await be().writeLocalTextFile(s.namelistPath, text);
        const fileName = s.namelistPath.split("/").pop() ?? s.namelistPath;
        set({
          deckIoStatus: {
            kind: "saved",
            detail: fileName + (s.formErrors.length > 0 ? t().deckIo.savedWithErrors : ""),
          },
        });
      } catch (err) {
        set({ deckIoStatus: { kind: "error", detail: String(err) } });
      }
    },

    async loadNamelist() {
      try {
        const picked = await be().openLocalTextFile(["py"]);
        if (picked === null) return;
        const r = extractGuiState(picked.content);
        if (!r.ok) {
          const msg = r.reason === "no-marker"
            ? t().deck.loadErrNoMarker
            : r.reason === "bad-json"
              ? t().deck.loadErrBadJson
              : t().deck.loadErrBadVersion;
          set({ deckIoStatus: { kind: "error", detail: msg } });
          return;
        }
        get().loadForm(migrateFormState(r.state));
        set({ namelistPath: picked.path, deckIoStatus: { kind: "loaded", detail: picked.name } });
      } catch (err) {
        set({ deckIoStatus: { kind: "error", detail: String(err) } });
      }
    },

    async saveTextAs(suggestedName, content) {
      if (content.length === 0) return false;
      return (await be().saveLocalTextFile(suggestedName, content)) !== null;
    },

    async setAssistLocalRepo(path) {
      set({ assistLocalRepo: path });
      await persistRuns();
    },

    async syncAssistMirror() {
      const profile = currentProfile(get());
      if (profile === null) {
        set({
          assistMirror: {
            ...get().assistMirror,
            status: "error",
            root: null,
            profileId: null,
            errorCode: "NO_PROFILE",
            errorDetail: null,
          },
        });
        return false;
      }
      if (profileBinMissing(profile)) {
        set({
          assistMirror: {
            ...get().assistMirror,
            status: "error",
            root: null,
            profileId: profile.id,
            errorCode: "NO_BIN",
            errorDetail: null,
          },
        });
        return false;
      }

      try {
        if (profile.transport === "local") {
          const r = await be().execLocal(
            ["bash", "-lc", repoRootScript(profile.tenryuBin)],
            { timeoutMs: 15000 },
          );
          const root = r.code === 0 ? (r.stdout.trim().split("\n").pop() ?? "") : "";
          if (!root.startsWith("/")) {
            set({
              assistMirror: {
                ...get().assistMirror,
                status: "error",
                root: null,
                profileId: profile.id,
                errorCode: "NO_TOOLS",
                errorDetail: null,
              },
            });
            return false;
          }
          set({
            assistMirror: {
              status: "ready",
              root,
              profileId: profile.id,
              lastSyncIso: null,
              errorCode: null,
              errorDetail: null,
            },
          });
          return true;
        }

        const serverRoot = await resolveRepoRoot(profile);
        if (serverRoot === null) {
          set({
            assistMirror: {
              ...get().assistMirror,
              status: "error",
              root: null,
              profileId: profile.id,
              errorCode: "NO_TOOLS",
              errorDetail: null,
            },
          });
          return false;
        }
        const wrapper = buildRemoteWrapperEnv(
          profile,
          serverRoot,
          await absRemotePath(profile, profile.tenryuBin),
        );
        if (wrapper.error !== null) {
          set({
            assistMirror: {
              ...get().assistMirror,
              status: "error",
              root: null,
              profileId: profile.id,
              errorCode: wrapper.error,
              errorDetail: null,
            },
          });
          return false;
        }

        const appConfigDir = await be().appConfigDir();
        const mirrorRoot = `${appConfigDir}/mirror/${profile.id}`;
        set({
          assistMirror: {
            ...get().assistMirror,
            status: "syncing",
            root: null,
            profileId: profile.id,
          },
        });
        await be().execLocal(["bash", "-lc", buildMkdirScript(`${appConfigDir}/mirror`)]);
        const r = await be().execLocal(
          [
            "bash",
            "-lc",
            buildMirrorSyncScript({
              host: wrapper.env.TENRYU_REMOTE_HOST,
              sshOpts: wrapper.env.TENRYU_REMOTE_SSH_OPTS,
              serverRepoRoot: serverRoot,
              mirrorRoot,
              harnessDir: await be().assistHarnessDir(),
            }),
          ],
          { timeoutMs: 600000 },
        );
        if (r.code === 0 && r.stdout.includes("MIRROR_OK")) {
          const lastSyncIso = new Date().toISOString();
          await be().writeLocalText(
            `${mirrorRoot}/mirror.json`,
            JSON.stringify(
              {
                serverRepoRoot: serverRoot,
                host: wrapper.env.TENRYU_REMOTE_HOST,
                lastSyncIso,
              },
              null,
              2,
            ) + "\n",
          );
          set({
            assistMirror: {
              status: "ready",
              root: mirrorRoot,
              profileId: profile.id,
              lastSyncIso,
              errorCode: null,
              errorDetail: null,
            },
          });
          assistProbeCache.clear();
          return true;
        }

        const missingLine = r.stdout.split("\n").find((line) => line.startsWith("MISSING:"))?.trim() ?? "";
        const errorCode = r.code === 3
          ? "MIRROR_INCOMPLETE"
          : r.code === 4
            ? "MIRROR_HARNESS_MISSING"
            : "MIRROR_SYNC_FAILED";
        const fallbackDetail = (r.stderr === "" ? r.stdout : r.stderr).slice(-2000);
        const errorDetail = r.code === 3
          ? `${missingLine} (server: ${serverRoot})`
          : r.code === 4
            ? missingLine
            : fallbackDetail === ""
              ? `exit ${r.code}`
              : fallbackDetail;
        set({
          assistMirror: {
            ...get().assistMirror,
            status: "error",
            root: null,
            profileId: profile.id,
            errorCode,
            errorDetail,
          },
        });
        return false;
      } catch (err) {
        set({
          assistMirror: {
            ...get().assistMirror,
            status: "error",
            root: null,
            profileId: profile.id,
            errorCode: "EXEC_FAILED",
            errorDetail: String(err),
          },
        });
        return false;
      }
    },

    async resolveAssistRepo() {
      const override = get().assistLocalRepo.trim();
      if (override !== "") return override;

      const profile = currentProfile(get());
      if (profile === null) {
        set({
          assistMirror: {
            ...get().assistMirror,
            status: "error",
            root: null,
            profileId: null,
            errorCode: "NO_PROFILE",
            errorDetail: null,
          },
        });
        return null;
      }

      try {
        if (profile.transport === "local") {
          const r = await be().execLocal(
            ["bash", "-lc", repoRootScript(profile.tenryuBin)],
            { timeoutMs: 15000 },
          );
          const root = r.code === 0 ? (r.stdout.trim().split("\n").pop() ?? "") : "";
          if (!root.startsWith("/")) {
            set({
              assistMirror: {
                ...get().assistMirror,
                status: "error",
                root: null,
                profileId: profile.id,
                errorCode: "NO_TOOLS",
                errorDetail: null,
              },
            });
            return null;
          }
          set({
            assistMirror: {
              status: "ready",
              root,
              profileId: profile.id,
              lastSyncIso: null,
              errorCode: null,
              errorDetail: null,
            },
          });
          return root;
        }

        const appConfigDir = await be().appConfigDir();
        const mirrorRoot = `${appConfigDir}/mirror/${profile.id}`;
        const probe = await be().execLocal(
          ["bash", "-lc", buildMirrorProbeScript(mirrorRoot)],
          { timeoutMs: 15000 },
        );
        if (probe.stdout.includes("MIRROR_OK")) {
          let lastSyncIso: string | null = null;
          if (get().assistMirror.root !== mirrorRoot) {
            try {
              const metadata = JSON.parse(
                await be().readLocalText(`${mirrorRoot}/mirror.json`),
              ) as { lastSyncIso?: unknown };
              lastSyncIso = typeof metadata.lastSyncIso === "string"
                ? metadata.lastSyncIso
                : null;
            } catch {
              lastSyncIso = null;
            }
            set({
              assistMirror: {
                status: "ready",
                root: mirrorRoot,
                profileId: profile.id,
                lastSyncIso,
                errorCode: null,
                errorDetail: null,
              },
            });
          }
          return mirrorRoot;
        }

        if (await get().syncAssistMirror()) {
          return get().assistMirror.root;
        }
        return null;
      } catch (err) {
        set({
          assistMirror: {
            ...get().assistMirror,
            status: "error",
            root: null,
            profileId: profile.id,
            errorCode: "EXEC_FAILED",
            errorDetail: String(err),
          },
        });
        return null;
      }
    },

    setAssistSpec(spec) {
      set({ assistSpec: spec });
    },

    setAssistUseTemplate(v) {
      set({ assistUseTemplate: v });
    },

    setAssistMaxIters(n) {
      set({ assistMaxIters: n });
    },

    setAssistIntentJson(v) {
      set({ assistIntentJson: v });
    },

    setAssistQuestion(q) {
      set({ assistQuestion: q });
    },

    initAssistConfigForm() {
      const status = get().assistStatus;
      set({
        assistConfig: {
          ...get().assistConfig,
          form: formFromStatus(
            status.status === "ready" ? (status.view ?? null) : null,
          ),
          status: "idle",
          error: null,
        },
      });
    },

    setAssistConfigForm(form) {
      const current = get().assistConfig;
      set({
        assistConfig: {
          ...current,
          form,
          status:
            current.status === "saved" || current.status === "error"
              ? "idle"
              : current.status,
          error:
            current.status === "saved" || current.status === "error"
              ? null
              : current.error,
        },
      });
    },

    addAssistProviderPreset(key) {
      const current = get().assistConfig;
      if (current.form === null) return;
      set({
        assistConfig: {
          ...current,
          form: addPreset(current.form, key),
        },
      });
    },

    async saveAssistConfig() {
      const initial = get().assistConfig;
      const form = initial.form;
      if (form === null) return;

      const errors = validateAssistConfigForm(form);
      if (errors.length > 0) {
        set({
          assistConfig: {
            ...initial,
            status: "error",
            error: errors
              .map((error) => `${error.code}: ${error.detail}`)
              .join("; "),
          },
        });
        return;
      }

      try {
        const path = await resolveAssistConfigPath();
        set({
          assistConfig: {
            ...initial,
            status: "saving",
            path,
          },
        });
        const parentDirectory = path.slice(0, path.lastIndexOf("/"));
        await be().execLocal([
          "bash",
          "-lc",
          buildMkdirScript(parentDirectory),
        ]);
        await be().writeLocalText(path, buildAssistantToml(form));
        set({
          assistConfig: {
            ...initial,
            status: "saved",
            savedPath: path,
            path,
            error: null,
          },
        });
        const repo = await get().resolveAssistRepo();
        if (repo === null) {
          set({ assistStatus: { status: "error", error: "NO_SOURCE" } });
          return;
        }
        await get().fetchAssistStatus();
      } catch (err) {
        set({
          assistConfig: {
            ...initial,
            status: "error",
            error: `CONFIG_WRITE_FAILED: ${String(err)}`,
          },
        });
      }
    },

    async fetchAssistStatus() {
      const repo = await get().resolveAssistRepo();
      if (repo === null) {
        set({ assistStatus: { status: "error", error: "NO_SOURCE" } });
        return;
      }
      try {
        set({ assistStatus: { status: "loading" } });
        const probe = await be().execLocal(
          ["bash", "-lc", buildProbeAssistScript(repo)],
          { timeoutMs: 30000 },
        );
        if (!probe.stdout.includes("ASSIST_OK")) {
          set({ assistStatus: { status: "error", error: "NO_LOCAL_ASSIST" } });
          return;
        }
        const configPath = await resolveAssistConfigPath();
        set({ assistConfig: { ...get().assistConfig, path: configPath } });
        const r = await be().execLocal(
          ["bash", "-lc", buildStatusScript(repo, await resolveAssistConfigPath())],
          { timeoutMs: 30000 },
        );
        if (r.code !== 0) {
          set({
            assistStatus: {
              status: "error",
              error: errorWithDetail("STATUS_FAILED", r.stderr.slice(-400)),
            },
          });
          return;
        }
        const parsed = parseAssistStatus(r.stdout);
        if (parsed.ok) {
          set({ assistStatus: { status: "ready", view: parsed.data, raw: r.stdout } });
        } else {
          set({
            assistStatus: {
              status: "error",
              error: `PARSE: ${parsed.error}`,
              raw: r.stdout,
            },
          });
        }
      } catch (err) {
        set({
          assistStatus: {
            status: "error",
            error: `EXEC_FAILED: ${String(err)}`,
          },
        });
      }
    },

    async runAssistLint() {
      const s = get();
      const profile = currentProfile(s);
      if (!profile) {
        set({ assistLint: { status: "error", error: "NO_PROFILE" } });
        return;
      }
      if (profileBinMissing(profile)) {
        set({ assistLint: { status: "error", error: "NO_BIN" } });
        return;
      }
      if (s.formErrors.length > 0 || s.deck.length === 0) {
        set({ assistLint: { status: "error", error: "FORM_INVALID" } });
        return;
      }
      try {
        const root = await resolveRepoRoot(profile);
        if (root === null) {
          set({ assistLint: { status: "error", error: "NO_TOOLS" } });
          return;
        }
        if (!(await probeRemoteAssist(profile, root))) {
          set({ assistLint: { status: "error", error: "NO_ASSIST" } });
          return;
        }
        set({ assistLint: { status: "running" } });
        const deckPath = await absRemotePath(
          profile,
          joinRemote(profile.runDir, `validate_scratch/${s.form.main.name}_assist_lint.py`),
        );
        await be().uploadText(profile, deckPath, s.deck);
        const r = await be().exec(
          profile,
          [
            "bash",
            "-lc",
            buildRemoteAssistScript(root, [
              "lint-deck",
              deckPath,
              "--tenryu",
              profile.tenryuBin,
            ]),
          ],
          { timeoutMs: 630000 },
        );
        const parsed = parseDeckLint(r.stdout);
        if (parsed.ok) {
          set({
            assistLint: {
              status: "ready",
              view: parsed.data,
              raw: r.stdout,
              exitOk: r.code === 0,
            },
          });
        } else if (r.code !== 0) {
          set({
            assistLint: {
              status: "error",
              error: errorWithDetail("LINT_FAILED", r.stderr.slice(-400)),
            },
          });
        } else {
          set({
            assistLint: {
              status: "error",
              error: `PARSE: ${parsed.error}`,
              raw: r.stdout,
            },
          });
        }
      } catch (err) {
        set({ assistLint: { status: "error", error: `EXEC_FAILED: ${String(err)}` } });
      }
    },

    async runAssistDiag(runId, verb) {
      const rec = get().runs.find((r) => r.id === runId);
      if (!rec || rec.runDir === "" || !isTerminal(rec.state)) return;

      function patchDiag(
        entry:
          | NonNullable<AppState["assistDiag"][string]["digest"]>
          | NonNullable<AppState["assistDiag"][string]["zoning"]>
          | NonNullable<AppState["assistDiag"][string]["promote"]>,
      ): void {
        set({
          assistDiag: {
            ...get().assistDiag,
            [runId]: { ...get().assistDiag[runId], [verb]: entry },
          },
        });
      }

      const profile = get().profiles.find((p) => p.id === rec.profileId);
      if (!profile) {
        patchDiag({ status: "error", error: "NO_PROFILE" });
        return;
      }
      try {
        const root = await resolveRepoRoot(profile);
        if (root === null) {
          patchDiag({ status: "error", error: "NO_TOOLS" });
          return;
        }
        if (!(await probeRemoteAssist(profile, root))) {
          patchDiag({ status: "error", error: "NO_ASSIST" });
          return;
        }
        patchDiag({ status: "running" });
        const latest = await be().exec(
          profile,
          ["bash", "-lc", buildLatestOutputDirScript(rec.runDir)],
          { timeoutMs: 20000 },
        );
        const dir = latest.stdout
          .split("\n")
          .map((line) => line.trim())
          .filter((line) => line.length > 0)
          .pop() ?? "";
        if (dir === "") {
          patchDiag({ status: "error", error: "NO_OUTPUTS" });
          return;
        }
        const verbArg = verb === "digest"
          ? "digest"
          : verb === "zoning" ? "zoning-report" : "promote-zoning";
        const r = await be().exec(
          profile,
          ["bash", "-lc", buildRemoteAssistScript(root, [verbArg, dir])],
          { timeoutMs: 180000 },
        );
        const failureCode = `${verbArg.toUpperCase().replace("-", "_")}_FAILED`;
        if (verb === "digest") {
          const parsed = parseDigest(r.stdout);
          if (parsed.ok) {
            patchDiag({ status: "ready", view: parsed.data, raw: r.stdout });
          } else if (r.code !== 0) {
            patchDiag({
              status: "error",
              error: errorWithDetail(failureCode, r.stderr.slice(-400)),
            });
          } else {
            patchDiag({ status: "error", error: `PARSE: ${parsed.error}`, raw: r.stdout });
          }
        } else if (verb === "zoning") {
          const parsed = parseZoningReport(r.stdout);
          if (parsed.ok) {
            patchDiag({ status: "ready", view: parsed.data, raw: r.stdout });
          } else if (r.code !== 0) {
            patchDiag({
              status: "error",
              error: errorWithDetail(failureCode, r.stderr.slice(-400)),
            });
          } else {
            patchDiag({ status: "error", error: `PARSE: ${parsed.error}`, raw: r.stdout });
          }
        } else {
          const parsed = parsePromoteZoning(r.stdout);
          if (parsed.ok) {
            patchDiag({ status: "ready", view: parsed.data, raw: r.stdout });
          } else if (r.code !== 0) {
            patchDiag({
              status: "error",
              error: errorWithDetail(failureCode, r.stderr.slice(-400)),
            });
          } else {
            patchDiag({ status: "error", error: `PARSE: ${parsed.error}`, raw: r.stdout });
          }
        }
      } catch (err) {
        patchDiag({ status: "error", error: `EXEC_FAILED: ${String(err)}` });
      }
    },

    async askAssistQuestion() {
      if (get().assistAsk.phase === "running") return;
      const question = get().assistQuestion.trim();
      if (!question) {
        failAsk("NO_QUESTION");
        return;
      }
      const repo = await get().resolveAssistRepo();
      if (repo === null) {
        failAsk("NO_SOURCE");
        return;
      }

      let workdir = get().assistAsk.workdir;
      let launched = false;
      try {
        const probe = await be().execLocal(
          ["bash", "-lc", buildProbeAssistScript(repo)],
          { timeoutMs: 30000 },
        );
        if (!probe.stdout.includes("ASSIST_OK")) {
          failAsk("NO_LOCAL_ASSIST");
          return;
        }

        if (!workdir) {
          workdir = `${await be().appConfigDir()}/ask/${nowStamp()}`;
          await be().execLocal(["bash", "-lc", buildMkdirScript(workdir)]);
        }
        const turnIndex = get().assistAsk.turns.length + 1;
        const questionPath = `${workdir}/question_${turnIndex}.md`;
        await be().writeLocalText(questionPath, question + "\n");

        const current = get().assistAsk;
        set({
          assistAsk: {
            ...current,
            phase: "running",
            workdir,
            pendingQuestion: question,
            lastKind: null,
            errorCode: null,
            errorDetail: null,
            resultRaw: null,
          },
          assistQuestion: "",
        });
        launched = true;

        stopAssistAskJournalTimer();
        assistAskJournalTimer = setInterval(() => {
          void be()
            .readLocalText(`${workdir}/journal.jsonl`, 262144)
            .then((text) => {
              const { entries } = parseJournalTail(text);
              const cur = get().assistAsk;
              if (cur.phase === "running" && cur.workdir === workdir) {
                set({
                  assistAsk: {
                    ...cur,
                    lastKind: entries[entries.length - 1]?.kind ?? null,
                  },
                });
              }
            })
            .catch(() => {});
        }, 2000);

        const r = await be().execLocal(
          [
            "bash",
            "-lc",
            buildAskScript({
              localRepo: repo,
              workdir,
              questionPath,
              configPath: await resolveAssistConfigPath(),
            }),
          ],
          { timeoutMs: 1800000 },
        );
        stopAssistAskJournalTimer();

        const cur = get().assistAsk;
        if (cur.workdir !== workdir || cur.phase !== "running") return;
        const parsed = parseAskResult(r.stdout);
        if (!parsed.ok) {
          failAsk(
            "RESULT_PARSE",
            `${parsed.error}; stderr: ${r.stderr.slice(-2000)}`,
          );
          set({
            assistAsk: { ...get().assistAsk, resultRaw: r.stdout },
          });
          return;
        }

        if (parsed.data.status === "answered" && parsed.data.answer !== null) {
          set({
            assistAsk: {
              ...cur,
              phase: "answered",
              turns: [
                ...cur.turns,
                {
                  turn: parsed.data.turn ?? turnIndex,
                  question,
                  answer: parsed.data.answer,
                  checks: parsed.data.checks,
                },
              ],
              pendingQuestion: null,
              resultRaw: r.stdout,
            },
            assistQuestion: "",
          });
        } else {
          failAsk(
            "ASK_FAILED",
            parsed.data.error ??
              `status ${parsed.data.status}; stderr: ${r.stderr.slice(-2000)}`,
          );
          set({
            assistAsk: { ...get().assistAsk, resultRaw: r.stdout },
          });
        }
      } catch (err) {
        stopAssistAskJournalTimer();
        if (!launched) {
          failAsk("EXEC_FAILED", String(err));
          return;
        }
        const cur = get().assistAsk;
        if (cur.phase === "running" && cur.workdir === workdir) {
          failAsk("EXEC_FAILED", String(err));
        }
      }
    },

    async cancelAssistQuestion() {
      const cur = get().assistAsk;
      if (cur.phase !== "running" || cur.workdir === null) return;
      const draft = get().assistQuestion;
      const cancelled: AppState["assistAsk"] = {
        ...cur,
        phase: "error",
        errorCode: "CANCELLED",
        errorDetail: null,
      };
      // Same draft restore as failAsk: the cancelled question returns to the
      // input box only when the user has not typed a replacement.
      if (draft.trim() === "" && cur.pendingQuestion !== null) {
        set({ assistAsk: cancelled, assistQuestion: cur.pendingQuestion });
      } else {
        set({ assistAsk: cancelled });
      }
      stopAssistAskJournalTimer();
      try {
        await be().execLocal(["bash", "-lc", buildCancelScript(cur.workdir)], {
          timeoutMs: 15000,
        });
      } catch {
        /* cancellation is best-effort */
      }
    },

    resetAssistConversation() {
      if (get().assistAsk.phase === "running") return;
      set({ assistAsk: initialAssistAsk });
    },

    async generateAssistDeck() {
      if (get().assistGen.phase === "running") return;
      const spec = get().assistSpec;
      if (spec.trim() === "") {
        failGen("NO_SPEC");
        return;
      }
      const repo = await get().resolveAssistRepo();
      if (repo === null) {
        failGen("NO_SOURCE");
        return;
      }

      let workdir: string | null = null;
      let launched = false;
      try {
        const probe = await be().execLocal(
          ["bash", "-lc", buildProbeAssistScript(repo)],
          { timeoutMs: 30000 },
        );
        if (!probe.stdout.includes("ASSIST_OK")) {
          failGen("NO_LOCAL_ASSIST");
          return;
        }

        const profile = currentProfile(get());
        if (!profile) {
          failGen("NO_PROFILE");
          return;
        }
        if (profileBinMissing(profile)) {
          failGen("NO_BIN");
          return;
        }

        let tenryuArg: string;
        let env: Record<string, string>;
        if (profile.transport === "local") {
          tenryuArg = profile.tenryuBin;
          env = {};
        } else {
          const root = await resolveRepoRoot(profile);
          if (root === null) {
            failGen("NO_TOOLS");
            return;
          }
          const binAbs = await absRemotePath(profile, profile.tenryuBin);
          const wrapper = buildRemoteWrapperEnv(profile, root, binAbs);
          if (wrapper.error !== null) {
            failGen(wrapper.error);
            return;
          }
          env = wrapper.env;
          tenryuArg = "tools/assist/tenryu_remote.sh";
        }

        const intent = get().assistIntentJson.trim();
        if (intent !== "") {
          try {
            JSON.parse(intent);
          } catch {
            failGen("INTENT_JSON_INVALID");
            return;
          }
        }

        const stamp = nowStamp();
        workdir = `${await be().appConfigDir()}/generate/${stamp}`;
        const specPath = `${workdir}/spec.md`;
        const outDeckPath = `${workdir}/out_deck.py`;
        await be().execLocal(["bash", "-lc", buildMkdirScript(workdir)], {
          timeoutMs: 15000,
        });
        await be().writeLocalText(specPath, spec);

        let intentPath: string | null = null;
        if (intent !== "") {
          intentPath = `${workdir}/intent.json`;
          await be().writeLocalText(intentPath, intent);
        }

        let templatePath: string | null = null;
        if (
          get().assistUseTemplate &&
          get().deck.length > 0 &&
          get().formErrors.length === 0
        ) {
          templatePath = `${workdir}/template.py`;
          await be().writeLocalText(templatePath, get().deck);
        }

        set({
          assistGen: {
            phase: "running",
            workdir,
            iterations: 0,
            lastKind: null,
            question: null,
            deckText: null,
            deckName: `assist_${stamp}`,
            errorCode: null,
            errorDetail: null,
            resultRaw: null,
            lint: null,
          },
        });
        launched = true;

        stopAssistJournalTimer();
        assistJournalTimer = setInterval(() => {
          void be()
            .readLocalText(`${workdir}/journal.jsonl`, 262144)
            .then((text) => {
              const journal = parseJournalTail(text);
              const cur = get().assistGen;
              if (cur.phase === "running" && cur.workdir === workdir) {
                set({
                  assistGen: {
                    ...cur,
                    iterations: journal.iterations,
                    lastKind: journal.lastKind,
                  },
                });
              }
            })
            .catch(() => {});
        }, 2000);

        const r = await be().execLocal(
          [
            "bash",
            "-lc",
            buildGenerateScript({
              localRepo: repo,
              workdir,
              specPath,
              outDeckPath,
              maxIters: get().assistMaxIters,
              templatePath,
              intentPath,
              tenryuArg,
              env,
              configPath: await resolveAssistConfigPath(),
            }),
          ],
          { timeoutMs: 3600000 },
        );
        stopAssistJournalTimer();

        const cur = get().assistGen;
        if (cur.workdir !== workdir || cur.phase !== "running") return;
        const parsed = parseGenerateResult(r.stdout);
        if (!parsed.ok) {
          set({
            assistGen: {
              ...cur,
              phase: "error",
              errorCode: "RESULT_PARSE",
              errorDetail: `${parsed.error}; stderr: ${r.stderr.slice(-2000)}`,
              resultRaw: r.stdout,
            },
          });
          return;
        }

        if (parsed.data.status === "accepted") {
          try {
            const deckText = await be().readLocalText(
              parsed.data.deckPath ?? outDeckPath,
              2000000,
            );
            set({
              assistGen: {
                ...cur,
                phase: "accepted",
                deckText,
                lint: parsed.data.lint,
                iterations: parsed.data.iterations ?? cur.iterations,
                resultRaw: r.stdout,
              },
            });
          } catch (err) {
            set({
              assistGen: {
                ...cur,
                phase: "error",
                errorCode: "READ_DECK_FAILED",
                errorDetail: String(err),
                resultRaw: r.stdout,
              },
            });
          }
        } else if (parsed.data.status === "uncertain") {
          set({
            assistGen: {
              ...cur,
              phase: "uncertain",
              question: parsed.data.question,
              iterations: parsed.data.iterations ?? cur.iterations,
              resultRaw: r.stdout,
            },
          });
        } else {
          set({
            assistGen: {
              ...cur,
              phase: "error",
              errorCode: `GEN_${parsed.data.status.toUpperCase()}`,
              errorDetail: parsed.data.error ?? r.stderr.slice(-2000),
              lint: parsed.data.lint,
              resultRaw: r.stdout,
            },
          });
        }
      } catch (err) {
        stopAssistJournalTimer();
        if (!launched) {
          failGen("EXEC_FAILED", String(err));
          return;
        }
        const cur = get().assistGen;
        if (cur.phase === "running" && cur.workdir === workdir) {
          failGen("EXEC_FAILED", String(err));
        }
      }
    },

    async answerAssistClarification(answer) {
      const cur = get().assistGen;
      if (cur.phase !== "uncertain") return;
      set({
        assistSpec:
          get().assistSpec +
          `\n\n== CLARIFICATION ==\nQ: ${cur.question ?? ""}\nA: ${answer}\n`,
      });
      await get().generateAssistDeck();
    },

    async cancelAssistGeneration() {
      const cur = get().assistGen;
      if (cur.phase !== "running" || cur.workdir === null) return;
      set({
        assistGen: {
          ...cur,
          phase: "error",
          errorCode: "CANCELLED",
          errorDetail: null,
        },
      });
      stopAssistJournalTimer();
      try {
        await be().execLocal(["bash", "-lc", buildCancelScript(cur.workdir)], {
          timeoutMs: 15000,
        });
      } catch {
        /* cancellation is best-effort */
      }
    },

    resetAssistGeneration() {
      set({ assistGen: { ...initialAssistGen }, assistDeckValidate: { status: "idle" } });
    },

    async validateAssistDeck() {
      const gen = get().assistGen;
      if (gen.phase !== "accepted" || gen.deckText === null) return;
      const profile = currentProfile(get());
      if (!profile) {
        set({
          assistDeckValidate: {
            status: "error",
            result: { ok: false, summary: [], errors: ["NO_PROFILE"], warnings: [], raw: "" },
          },
        });
        return;
      }
      if (profileBinMissing(profile)) {
        set({
          assistDeckValidate: {
            status: "error",
            result: { ok: false, summary: [], errors: ["NO_BIN"], warnings: [], raw: "" },
          },
        });
        return;
      }
      set({ assistDeckValidate: { status: "running" } });
      const root = await resolveRepoRoot(profile);
      const { result, sentTo } = await validateDeckWith(
        profile,
        root,
        gen.deckText,
        gen.deckName ?? "assist_deck",
      );
      set({ assistDeckValidate: { status: "ready", result, sentTo } });
    },

    async runAssistGeneratedDeck() {
      const gen = get().assistGen;
      if (
        gen.phase !== "accepted" ||
        gen.deckText === null ||
        gen.deckName === null
      ) return;
      await get().startRun({ deck: gen.deckText, name: gen.deckName });
    },

    async runValidate() {
      const s = get();
      const profile = currentProfile(s);
      if (!profile) {
        set({
          validateResult: { ok: false, summary: [], errors: ["NO_PROFILE"], warnings: [], raw: "" },
          validateSentTo: null,
        });
        return;
      }
      if (profileBinMissing(profile)) {
        set({
          validateResult: { ok: false, summary: [], errors: ["NO_BIN"], warnings: [], raw: "" },
          validateSentTo: null,
        });
        return;
      }
      if (s.formErrors.length > 0 || s.deck.length === 0) {
        set({
          validateResult: {
            ok: false,
            summary: [],
            errors: ["FORM_INVALID", ...s.formErrors],
            warnings: [],
            raw: "",
          },
          validateSentTo: null,
        });
        return;
      }
      const isPib = s.form.main.dimension === "2D_RZ" && s.form.mesh.meshMode2d === "polar_in_box";
      const root = await resolveRepoRoot(profile);
      if (isPib && root === null) {
        set({
          validateResult: { ok: false, summary: [], errors: ["NO_TOOLS"], warnings: [], raw: "" },
          validateSentTo: null,
        });
        return;
      }
      set({ validating: true, validateResult: null });
      const { result, sentTo } = await validateDeckWith(profile, root, s.deck, s.form.main.name);
      set({ validateResult: result, validateSentTo: sentTo, validating: false });
    },

    async runMeshPreview() {
      const s = get();
      const profile = currentProfile(s);
      if (!profile) {
        set({ meshPreviewError: "NO_PROFILE", meshPreviewBusy: false });
        return;
      }
      if (profileBinMissing(profile)) {
        set({ meshPreviewError: "NO_BIN", meshPreviewBusy: false });
        return;
      }
      if (s.formErrors.length > 0 || s.deck.length === 0) {
        set({ meshPreviewError: "FORM_INVALID", meshPreviewBusy: false });
        return;
      }
      const isPib = s.form.main.dimension === "2D_RZ" && s.form.mesh.meshMode2d === "polar_in_box";
      const root = await resolveRepoRoot(profile);
      if (isPib && root === null) {
        set({ meshPreviewError: "NO_TOOLS", meshPreviewBusy: false });
        return;
      }
      set({ meshPreviewBusy: true, meshPreviewError: null });
      const remoteDeck = joinRemote(profile.runDir, `validate_scratch/${s.form.main.name}_preview.py`);
      try {
        await be().uploadText(profile, remoteDeck, s.deck);
        const r: ExecResult = await be().exec(
          profile,
          withRepoEnv(root, [profile.tenryuBin, "validate", remoteDeck, "--mesh-preview"]),
          { timeoutMs: 120000 },
        );
        const data = parseMeshPreview(r.stdout);
        if (r.code !== 0) {
          const result = parseValidateOutput(r.stdout, r.stderr, r.code);
          set({
            meshPreviewError: result.errors[0] ?? "validate failed",
            meshPreviewBusy: false,
          });
        } else if (data === null) {
          set({ meshPreviewError: t().meshPreview.parseError, meshPreviewBusy: false });
        } else {
          set({ meshPreview: data, meshPreviewError: null, meshPreviewBusy: false });
        }
      } catch (err) {
        set({ meshPreviewError: String(err), meshPreviewBusy: false });
      }
    },

    async fetchMeshSnapshot() {
      const s = get();
      const caseName = s.form.main.name;
      logOp(`fetchMeshSnapshot ${caseName}`);
      const profile = currentProfile(s);
      if (!profile) {
        logLine("error", "fetchMeshSnapshot NO_PROFILE");
        set({ meshSnapshotError: "NO_PROFILE", meshSnapshotBusy: false });
        return;
      }
      if (profileBinMissing(profile)) {
        logLine("error", "fetchMeshSnapshot NO_BIN");
        set({ meshSnapshotError: "NO_BIN", meshSnapshotBusy: false });
        return;
      }
      if (s.formErrors.length > 0 || s.deck.length === 0) {
        logLine("error", "fetchMeshSnapshot FORM_INVALID");
        set({ meshSnapshotError: "FORM_INVALID", meshSnapshotBusy: false });
        return;
      }
      const isPib = s.form.main.dimension === "2D_RZ" && s.form.mesh.meshMode2d === "polar_in_box";
      const root = await resolveRepoRoot(profile);
      if (isPib && root === null) {
        logLine("error", "fetchMeshSnapshot NO_TOOLS");
        set({ meshSnapshotError: "NO_TOOLS", meshSnapshotBusy: false });
        return;
      }
      set({ meshSnapshotBusy: true, meshSnapshotError: null });
      const remoteOut = joinRemote(profile.runDir, `validate_scratch/meshsnap_${caseName}`);
      const remoteDeck = joinRemote(profile.runDir, `validate_scratch/${caseName}_meshsnap.py`);
      try {
        await be().uploadText(profile, remoteDeck, s.deck);
        await be().exec(
          profile,
          ["bash", "-lc", "rm -rf " + shQuotePath(remoteOut) + " " + shQuotePath(remoteOut) + "_[0-9][0-9][0-9]"],
          { timeoutMs: 20000 },
        );
        const r = await be().exec(
          profile,
          withRepoEnv(root, [profile.tenryuBin, "run", remoteDeck, "--output-dir", remoteOut]),
          { timeoutMs: 180000 },
        );
        const ls = await be().exec(
          profile,
          ["bash", "-lc", "ls -t " + shQuotePath(remoteOut + "/results") + "/*.h5 2>/dev/null | head -5"],
          { timeoutMs: 20000 },
        );
        const path = ls.stdout
          .split("\n")
          .map((line) => line.trim())
          .find((line) => line !== "" && !line.includes("history"));
        if (path === undefined) {
          const error = `${r.stdout}\n${r.stderr}`.trim().slice(-400);
          logLine("error", `fetchMeshSnapshot snapshot not found: ${error}`);
          set({ meshSnapshotError: error, meshSnapshotBusy: false });
          return;
        }
        const bytes = await be().readBinary(profile, path);
        const mesh = await parseSnapshotMesh(Uint8Array.from(bytes).buffer);
        if (mesh === null) {
          const error = t().meshPreview.parseError;
          logLine("error", `fetchMeshSnapshot ${error}`);
          set({ meshSnapshotError: error, meshSnapshotBusy: false });
          return;
        }
        set({ meshSnapshot: mesh, meshSnapshotError: null, meshSnapshotBusy: false });
      } catch (err) {
        const error = String(err);
        logLine("error", `fetchMeshSnapshot ${error}`);
        set({ meshSnapshotError: error, meshSnapshotBusy: false });
      }
    },

    async fetchTmatGroupBounds() {
      const profile = currentProfile(get());
      if (profile === null) {
        set({ tmatBoundsError: "NO_PROFILE" });
        return;
      }
      const mat = get().form.materials.find(
        (mm) => mm.opacityModel === "tmat" && mm.opacityFile.trim() !== "",
      );
      if (mat === undefined) {
        set({ tmatBoundsError: t().tmatBounds.noTmatMaterial });
        return;
      }
      set({ tmatBoundsBusy: true, tmatBoundsError: null });
      try {
        const bytes = await be().readBinary(profile, mat.opacityFile.trim(), 33554432);
        const bounds = await parseTmatGroupBounds(Uint8Array.from(bytes).buffer);
        if (bounds === null) {
          set({ tmatBoundsError: t().tmatBounds.parseFailed, tmatBoundsBusy: false });
          return;
        }
        get().updateForm((f) => {
          f.radiation.groups = bounds.length - 1;
          f.radiation.groupBoundsEV = bounds;
        });
        set({ tmatBoundsBusy: false, tmatBoundsError: null });
      } catch (err) {
        set({ tmatBoundsError: String(err), tmatBoundsBusy: false });
      }
    },

    async testConnection(p) {
      set({ connTesting: true, connResult: null });
      try {
        const ping = await be().exec(p, ["echo", "tenryu-studio-ping"], { timeoutMs: 20000 });
        if (ping.code !== 0) {
          set({
            connTesting: false,
            connResult: {
              ok: false,
              binOk: false,
              binRuns: false,
              gpu: "",
              runDirOk: false,
              detail: ping.stderr.trim() || `exit=${ping.code}`,
            },
          });
          return;
        }
        let binOk = false;
        let binRuns = false;
        let detail = "";
        if (profileBinMissing(p)) {
          detail = t().server.binSkippedDetail;
        } else {
          const bin = await be().exec(p, ["test", "-x", p.tenryuBin], { timeoutMs: 20000 });
          binOk = bin.code === 0;
          if (binOk) {
            const help = await be().exec(p, [p.tenryuBin, "--help"], { timeoutMs: 20000 });
            binRuns = help.code === 0;
          }
        }
        const gpuResult = await be().exec(
          p,
          ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
          { timeoutMs: 20000 },
        );
        const gpu = gpuResult.code === 0 ? gpuResult.stdout.trim() : "";
        const runDir = p.runDir.trim();
        const probeCommand =
          runDir === ""
            ? "touch ~/.tenryu_studio_probe && rm -f ~/.tenryu_studio_probe"
            : `mkdir -p ${shQuotePath(runDir)} && touch ${shQuotePath(joinRemote(runDir, ".tenryu_studio_probe"))} && rm -f ${shQuotePath(joinRemote(runDir, ".tenryu_studio_probe"))}`;
        const runDirResult = await be().exec(p, ["bash", "-lc", probeCommand], { timeoutMs: 20000 });
        set({
          connTesting: false,
          connResult: { ok: true, binOk, binRuns, gpu, runDirOk: runDirResult.code === 0, detail },
        });
      } catch (err) {
        set({
          connTesting: false,
          connResult: {
            ok: false,
            binOk: false,
            binRuns: false,
            gpu: "",
            runDirOk: false,
            detail: String(err),
          },
        });
      }
    },

    async startRun(override) {
      const s = get();
      const profile = currentProfile(s);
      if (!profile || s.starting) return;
      if (override === undefined && (s.formErrors.length > 0 || s.deck.length === 0)) return;
      if (override !== undefined && override.deck.length === 0) return;
      if (profileBinMissing(profile)) return;
      const root = await resolveRepoRoot(profile);
      if (override === undefined) {
        const isPib = s.form.main.dimension === "2D_RZ" && s.form.mesh.meshMode2d === "polar_in_box";
        if (isPib && root === null) return;
      }
      set({ starting: true, view: "history" });
      const deckText = override?.deck ?? s.deck;
      const runName = `${override?.name ?? s.form.main.name}_${nowStamp()}`;
      const overrideTEnd = override?.deck.match(/t_end\s*=\s*([0-9][0-9eE+.\-]*)/);
      const parsedTEnd = overrideTEnd === null || overrideTEnd === undefined
        ? 0
        : Number(overrideTEnd[1]);
      const record: RunRecord = {
        id: newId(),
        profileId: profile.id,
        profileName: profile.name,
        name: runName,
        runDir: "",
        statusPath: "",
        tEnd: override === undefined
          ? toCanonical(s.form.main.tEnd, "time")
          : Number.isFinite(parsedTEnd) ? parsedTEnd : 0,
        maxSteps: override === undefined ? s.form.main.maxSteps : 0,
        createdAtIso: new Date().toISOString(),
        state: "launching",
        pid: null,
        exitCode: null,
        stopRequested: false,
        lastProgress: null,
        startEpoch: null,
        endEpoch: null,
        launchError: null,
      };
      set({ runs: [record, ...get().runs] });
      try {
        const runDir = await absRemotePath(profile, joinRemote(profile.runDir, runName));
        const deckPath = joinRemote(runDir, "deck.py");
        const scriptPath = joinRemote(runDir, "run_detached.sh");
        patchRun(record.id, { runDir, statusPath: joinRemote(runDir, "status.json") });
        await be().uploadText(profile, deckPath, deckText);
        await be().uploadText(profile, scriptPath, RUN_DETACHED_SH);
        const r = await be().exec(
          profile,
          withRepoEnv(root, ["bash", scriptPath, runDir, profile.tenryuBin, deckPath]),
          { timeoutMs: 30000 },
        );
        if (r.code !== 0) {
          patchRun(record.id, {
            state: "failed",
            launchError: (r.stderr || r.stdout).trim() || `exit=${r.code}`,
          });
        } else {
          const statusPath = r.stdout.trim().split("\n").pop() ?? "";
          patchRun(record.id, {
            statusPath: statusPath.endsWith("status.json")
              ? statusPath
              : joinRemote(runDir, "status.json"),
          });
          await get().pollRunOnce(record.id);
        }
      } catch (err) {
        patchRun(record.id, { state: "failed", launchError: String(err) });
      }
      set({ starting: false });
      await persistRuns();
    },

    async restartRun(runId) {
      const s = get();
      const rec = s.runs.find((r) => r.id === runId);
      if (!rec || rec.runDir === "" || !isTerminal(rec.state) || s.starting) return;
      const profile = s.profiles.find((p) => p.id === rec.profileId);
      if (!profile) return;
      if (profileBinMissing(profile)) {
        patchRun(runId, { launchError: t().server.binMissing });
        return;
      }
      const root = await resolveRepoRoot(profile);
      set({ starting: true });
      try {
        const ls = await be().exec(
          profile,
          ["bash", "-lc", `find ${shQuotePath(rec.runDir)} -name '*_ckpt_*.h5' 2>/dev/null`],
          { timeoutMs: 20000 },
        );
        const checkpoints = ls.stdout
          .split("\n")
          .map((path) => {
            const trimmedPath = path.trim();
            const match = trimmedPath.match(/_ckpt_(\d+)(?:_r\d+)?\.h5$/);
            return match ? { path: trimmedPath, index: Number(match[1]) } : null;
          })
          .filter((item): item is { path: string; index: number } => item !== null);
        if (checkpoints.length === 0) {
          patchRun(runId, { state: "failed", launchError: t().run.noCheckpoints });
          set({ starting: false });
          await persistRuns();
          return;
        }
        const latest = checkpoints.reduce((a, b) => (b.index > a.index ? b : a));
        const restartPrefix = latest.path.replace(/(?:_r\d+)?\.h5$/, "");
        const scriptPath = joinRemote(rec.runDir, "run_detached.sh");
        const deckPath = joinRemote(rec.runDir, "deck.py");
        const startEpoch = Math.floor(Date.now() / 1000);
        const r = await be().exec(
          profile,
          withRepoEnv(root, ["bash", scriptPath, rec.runDir, profile.tenryuBin, deckPath, restartPrefix]),
          { timeoutMs: 30000 },
        );
        if (r.code !== 0) {
          patchRun(runId, {
            state: "failed",
            launchError: (r.stderr || r.stdout).trim() || `exit=${r.code}`,
          });
        } else {
          const statusPath = r.stdout.trim().split("\n").pop() ?? "";
          patchRun(runId, {
            state: "running",
            pid: null,
            exitCode: null,
            stopRequested: false,
            startEpoch,
            endEpoch: null,
            launchError: null,
            statusPath: statusPath.endsWith("status.json")
              ? statusPath
              : joinRemote(rec.runDir, "status.json"),
          });
          await get().pollRunOnce(runId);
        }
      } catch (err) {
        patchRun(runId, { state: "failed", launchError: String(err) });
      }
      set({ starting: false });
      await persistRuns();
    },

    async pollRunOnce(runId) {
      const rec = get().runs.find((r) => r.id === runId);
      if (!rec || isTerminal(rec.state) || rec.statusPath === "") return;
      const profile = get().profiles.find((p) => p.id === rec.profileId);
      if (!profile) {
        patchRun(runId, { state: "unknown" });
        return;
      }
      try {
        const statusText = await be().readText(profile, rec.statusPath, 4096);
        const status = parseStatusFile(statusText.trim());
        if (!status) return;
        if (status.state === "running") {
          const tail = await be().readText(profile, joinRemote(rec.runDir, "run.log"), 16384);
          const prog = parseLastProgress(tail);
          if (prog) {
            const prev = rateMemo.get(runId);
            const wall = Date.now() / 1000;
            if (prev && wall > prev.wall && prog.step > prev.step) {
              set({
                runRates: {
                  ...get().runRates,
                  [runId]: (prog.step - prev.step) / (wall - prev.wall),
                },
              });
            }
            rateMemo.set(runId, { step: prog.step, wall });
          }
          patchRun(runId, {
            state: rec.stopRequested ? "stopping" : "running",
            pid: status.pid,
            startEpoch: status.start_epoch,
            lastProgress: prog ? { step: prog.step, t: prog.t, pct: prog.pct } : rec.lastProgress,
          });
          set({ runLogs: { ...get().runLogs, [runId]: tail } });
        } else {
          const finalState =
            status.state === "finished" ? "finished" : rec.stopRequested ? "stopped" : "failed";
          patchRun(runId, {
            state: finalState,
            pid: status.pid,
            exitCode: status.exit_code,
            startEpoch: status.start_epoch,
            endEpoch: status.end_epoch,
          });
          try {
            const tail = await be().readText(profile, joinRemote(rec.runDir, "run.log"), 16384);
            set({ runLogs: { ...get().runLogs, [runId]: tail } });
            const prog = parseLastProgress(tail);
            if (prog) {
              patchRun(runId, { lastProgress: { step: prog.step, t: prog.t, pct: prog.pct } });
            }
          } catch {
            /* log fetch is best-effort */
          }
          rateMemo.delete(runId);
          await persistRuns();
        }
      } catch {
        /* transient read failure — keep state */
      }
    },

    async pollActiveRuns() {
      for (const r of get().runs) {
        if (!isTerminal(r.state) && r.state !== "unknown") {
          await get().pollRunOnce(r.id);
        }
      }
    },

    async stopRun(runId) {
      const rec = get().runs.find((r) => r.id === runId);
      if (!rec || isTerminal(rec.state) || rec.pid === null) return;
      const profile = get().profiles.find((p) => p.id === rec.profileId);
      if (!profile) return;
      patchRun(runId, { stopRequested: true, state: "stopping" });
      try {
        await be().exec(profile, ["kill", "-TERM", "--", `-${rec.pid}`], { timeoutMs: 15000 });
      } catch {
        /* next poll reflects reality */
      }
    },

    async fetchLog(runId) {
      const rec = get().runs.find((r) => r.id === runId);
      if (!rec || rec.runDir === "") return;
      const profile = get().profiles.find((p) => p.id === rec.profileId);
      if (!profile) return;
      try {
        const tail = await be().readText(profile, joinRemote(rec.runDir, "run.log"), 16384);
        set({ runLogs: { ...get().runLogs, [runId]: tail } });
      } catch (err) {
        set({ runLogs: { ...get().runLogs, [runId]: String(err) } });
      }
    },

    async deleteRunRecord(runId) {
      set({ runs: get().runs.filter((r) => r.id !== runId) });
      rateMemo.delete(runId);
      await persistRuns();
    },

    async fetchHistory(runId) {
      const rec = get().runs.find((r) => r.id === runId);
      if (!rec || rec.runDir === "") return;
      const existing = get().histories[runId];
      if (existing && existing.status !== "error") return;
      const profile = get().profiles.find((p) => p.id === rec.profileId);
      if (!profile) return;
      set({ histories: { ...get().histories, [runId]: { status: "loading" } } });
      try {
        const ls = await be().exec(
          profile,
          ["bash", "-c", `ls ${rec.runDir}/outputs/*/results/*_history.h5 2>/dev/null | head -1`],
          { timeoutMs: 20000 },
        );
        const path = ls.stdout.trim().split("\n").pop() ?? "";
        if (ls.code !== 0 || !path.endsWith("_history.h5")) {
          throw new Error(ls.stderr.trim() || "history.h5 not found");
        }
        const bytes = await be().readBinary(profile, path);
        const data = await parseHistory(bytes);
        set({ histories: { ...get().histories, [runId]: { status: "ready", data } } });
      } catch (err) {
        set({ histories: { ...get().histories, [runId]: { status: "error", error: String(err) } } });
      }
    },

    async fetchProfileList(runId) {
      const rec = get().runs.find((r) => r.id === runId);
      if (!rec || rec.runDir === "") return;
      const existing = get().profileLists[runId];
      if (existing && existing.status !== "error") return;
      const profile = get().profiles.find((p) => p.id === rec.profileId);
      if (!profile) return;
      set({ profileLists: { ...get().profileLists, [runId]: { status: "loading" } } });
      try {
        const ls = await be().exec(
          profile,
          ["bash", "-c", `ls ${rec.runDir}/outputs/*/results/ 2>/dev/null`],
          { timeoutMs: 20000 },
        );
        const dirLine = await be().exec(
          profile,
          ["bash", "-c", `ls -d ${rec.runDir}/outputs/*/results 2>/dev/null | head -1`],
          { timeoutMs: 20000 },
        );
        const dir = dirLine.stdout.trim().split("\n").pop() ?? "";
        const paths = ls.stdout
          .split("\n")
          .map((s) => s.trim())
          .filter((s) => /_\d{4}\.h5$/.test(s))
          .sort()
          .map((s) => `${dir}/${s}`);
        if (dir === "" || paths.length === 0) {
          throw new Error("no profile snapshots found");
        }
        set({ profileLists: { ...get().profileLists, [runId]: { status: "ready", paths } } });
      } catch (err) {
        set({
          profileLists: { ...get().profileLists, [runId]: { status: "error", error: String(err) } },
        });
      }
    },

    async fetchProfileSnap(runId, index) {
      const rec = get().runs.find((r) => r.id === runId);
      const list = get().profileLists[runId];
      if (!rec || !list || list.status !== "ready" || !list.paths) return;
      const path = list.paths[index];
      if (!path) return;
      const key = `${runId}:${index}`;
      const existing = get().profileSnaps[key];
      if (existing && existing.status !== "error") return;
      const profile = get().profiles.find((p) => p.id === rec.profileId);
      if (!profile) return;
      set({ profileSnaps: { ...get().profileSnaps, [key]: { status: "loading" } } });
      try {
        const bytes = await be().readBinary(profile, path);
        const data = await parseProfile(bytes);
        set({ profileSnaps: { ...get().profileSnaps, [key]: { status: "ready", data } } });
      } catch (err) {
        set({
          profileSnaps: { ...get().profileSnaps, [key]: { status: "error", error: String(err) } },
        });
      }
    },
  };
});
