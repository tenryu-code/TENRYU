/** status.json written by ops/gui/run_detached.sh (schema 1). */
export interface RunStatusFile {
  schema: number;
  state: "running" | "finished" | "failed";
  pid: number;
  exit_code: number | null;
  start_epoch: number;
  end_epoch: number | null;
  host?: string;
  deck: string;
  log: string;
}

export function parseStatusFile(text: string): RunStatusFile | null {
  try {
    const j = JSON.parse(text) as RunStatusFile;
    if (j && j.schema === 1 && typeof j.state === "string" && typeof j.pid === "number") {
      return j;
    }
  } catch {
    /* fall through */
  }
  return null;
}

export type RunUiState =
  | "launching"
  | "running"
  | "stopping"
  | "finished"
  | "failed"
  | "stopped"
  | "unknown";

export interface RunProgressBrief {
  step: number;
  t: number;
  pct: number;
}

/** Persisted run-history record (kept small; log tails are not persisted). */
export interface RunRecord {
  id: string;
  profileId: string;
  profileName: string;
  name: string;
  runDir: string;
  statusPath: string;
  tEnd: number;
  maxSteps: number;
  createdAtIso: string;
  state: RunUiState;
  pid: number | null;
  exitCode: number | null;
  stopRequested: boolean;
  lastProgress: RunProgressBrief | null;
  startEpoch: number | null;
  endEpoch: number | null;
  launchError: string | null;
}

export function isTerminal(s: RunUiState): boolean {
  return s === "finished" || s === "failed" || s === "stopped";
}
