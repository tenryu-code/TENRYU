// Output files of one run of `tenryu run`: the deck's Output.directory and, when it already exists, the
// <dir>_001, <dir>_002, ... that later runs of the same deck (restarts and continuations) write to, each with its
// own file numbering from 0000.

function baseName(path: string): string {
  return path.slice(path.lastIndexOf("/") + 1);
}

/** The output directory of a result or checkpoint file: .../<dir>/{results,checkpoints}/<file>. */
export function outputDirOf(path: string): string {
  const parts = path.split("/");
  return parts.length >= 3 ? parts[parts.length - 3] : "";
}

function compareText(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/** Output directories in the order the solver created them: the deck's directory, then <dir>_001, <dir>_002, ...
 *  The suffix has three digits, so shorter names come first and names of equal length sort by text. (The order of
 *  `ls` is not used: UTF-8 collations ignore "/" and "_" and place "<dir>_001" before "<dir>".) */
export function compareOutputDirs(a: string, b: string): number {
  return a.length - b.length || compareText(a, b);
}

/** Result or checkpoint files of a run: by output directory in creation order, then by file name (the four-digit
 *  index grows within a directory). */
export function sortRunOutputFiles(paths: string[]): string[] {
  return [...paths].sort(
    (a, b) => compareOutputDirs(outputDirOf(a), outputDirOf(b)) || compareText(baseName(a), baseName(b)),
  );
}

/** Snapshots of a run and its continuations, each with its time if known. A continuation restarts from a checkpoint,
 *  which may lie before the end of the output it continues, so a frame of an earlier output directory at or after the
 *  first time of a later directory is replaced by that directory's frames (the rule of Studio's history chart).
 *  Returns, per frame, whether it is replaced; frames without a known time are kept. */
export function supersededFrames(frames: Array<{ path: string; time: number | null }>): boolean[] {
  const firstTime = new Map<string, number>();
  for (const { path, time } of frames) {
    if (time === null || !Number.isFinite(time)) continue;
    const dir = outputDirOf(path);
    const known = firstTime.get(dir);
    if (known === undefined || time < known) firstTime.set(dir, time);
  }
  const dirs = [...firstTime.keys()].sort(compareOutputDirs);
  return frames.map(({ path, time }) => {
    if (time === null || !Number.isFinite(time)) return false;
    const own = outputDirOf(path);
    return dirs.some((dir) => compareOutputDirs(dir, own) > 0 && time >= (firstTime.get(dir) as number));
  });
}
