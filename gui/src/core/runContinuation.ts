import type { HistorySeries } from "@tenryu-common/core/results/historyParse";
import { sortRunOutputFiles } from "@tenryu-common/core/runOutputs";
import type { RunRecord } from "@tenryu-common/core/runstate";

export { compareOutputDirs, sortRunOutputFiles } from "@tenryu-common/core/runOutputs";

/** Largest Main.max_steps and `tenryu run --max-steps` the solver accepts (2^24 - 1). */
export const MAX_STEPS_UPPER_BOUND = 16_777_215;

/** A run continued from its latest checkpoint with the unchanged deck: the new end time [s] and step limit, passed
 *  as `tenryu run --t-end` / `--max-steps` (SPECIFICATION 7.4). null keeps the deck's value. */
export interface RunContinuation {
  tEndS: number | null;
  maxSteps: number | null;
}

/** The `tenryu run` arguments that follow `--restart <prefix>`. */
export function continuationArgs(c: RunContinuation): string[] {
  const args: string[] = [];
  if (c.tEndS !== null) args.push("--t-end", String(c.tEndS));
  if (c.maxSteps !== null) args.push("--max-steps", String(c.maxSteps));
  return args;
}

export type ContinuationIssue = "tEndInvalid" | "tEndNotLater" | "maxStepsInvalid" | "maxStepsNotLarger";

/** Checks the requested values against the run's last reported progress. The solver checks them again against the
 *  checkpoint it restarts from and refuses an end time or step limit the checkpoint has already reached. */
export function continuationIssues(
  c: RunContinuation,
  rec: Pick<RunRecord, "lastProgress">,
): ContinuationIssue[] {
  const issues: ContinuationIssue[] = [];
  if (c.tEndS !== null) {
    if (!(Number.isFinite(c.tEndS) && c.tEndS > 0)) issues.push("tEndInvalid");
    else if (rec.lastProgress !== null && !(c.tEndS > rec.lastProgress.t)) issues.push("tEndNotLater");
  }
  if (c.maxSteps !== null) {
    if (!(Number.isInteger(c.maxSteps) && c.maxSteps >= 1 && c.maxSteps <= MAX_STEPS_UPPER_BOUND)) {
      issues.push("maxStepsInvalid");
    } else if (rec.lastProgress !== null && !(c.maxSteps > rec.lastProgress.step)) {
      issues.push("maxStepsNotLarger");
    }
  }
  return issues;
}

const CHECKPOINT_FILE = /_ckpt_\d+(?:_r\d+)?\.h5$/;

/** The restart prefix (the path without the rank suffix and ".h5") of the checkpoint a continuation starts from:
 *  the last checkpoint of the latest output directory that has one. null when the run wrote no checkpoint. */
export function latestCheckpointPrefix(paths: string[]): string | null {
  const checkpoints = paths.map((p) => p.trim()).filter((p) => CHECKPOINT_FILE.test(p));
  if (checkpoints.length === 0) return null;
  const sorted = sortRunOutputFiles(checkpoints);
  return sorted[sorted.length - 1].replace(/(?:_r\d+)?\.h5$/, "");
}

/** Joins the history files of a run and its continuations, given in output-directory order. A continuation restarts
 *  from a checkpoint, which may lie before the end of the earlier run, so each segment replaces the earlier rows from
 *  its first time on. Columns absent from a segment are NaN there; `missing` lists the keys absent from every one. */
export function mergeHistorySegments(segments: HistorySeries[]): HistorySeries {
  if (segments.length === 1) return segments[0];
  const keys: string[] = [];
  for (const segment of segments) {
    for (const key of Object.keys(segment.series)) if (!keys.includes(key)) keys.push(key);
  }
  let t: number[] = [];
  let columns: Record<string, number[]> = Object.fromEntries(keys.map((key) => [key, [] as number[]]));
  for (const segment of segments) {
    if (segment.t.length === 0) continue;
    const start = segment.t[0];
    let keep = t.length;
    while (keep > 0 && t[keep - 1] >= start) keep -= 1;
    t = t.slice(0, keep);
    columns = Object.fromEntries(keys.map((key) => [key, columns[key].slice(0, keep)]));
    for (let i = 0; i < segment.t.length; i++) {
      t.push(segment.t[i]);
      for (const key of keys) {
        const values = segment.series[key];
        columns[key].push(values !== undefined ? values[i] : Number.NaN);
      }
    }
  }
  const series: Record<string, Float64Array> = {};
  for (const key of keys) series[key] = Float64Array.from(columns[key]);
  const missing =
    segments.length === 0 ? [] : segments[0].missing.filter((key) => segments.every((s) => s.missing.includes(key)));
  return { t: Float64Array.from(t), series, missing };
}
