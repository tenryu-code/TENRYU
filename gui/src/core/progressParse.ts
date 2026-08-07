/** Parsed `[progress]` line from the TENRYU driver log (driver.cpp log_progress_if_needed). */
export interface ProgressSample {
  step: number;
  maxSteps: number;
  t: number;
  tEnd: number;
  dt: number;
  pct: number;
}

const PROGRESS_RE =
  /\[progress\] step=(\d+)\/(\d+) t=([0-9.eE+-]+)\/([0-9.eE+-]+) dt=([0-9.eE+-]+) \((\d+(?:\.\d+)?)%\)/;

/** Scan a log tail from the end and return the most recent progress sample. */
export function parseLastProgress(logText: string): ProgressSample | null {
  const lines = logText.split(/\r?\n/);
  for (let i = lines.length - 1; i >= 0; i--) {
    const m = PROGRESS_RE.exec(lines[i]);
    if (m) {
      return {
        step: Number(m[1]),
        maxSteps: Number(m[2]),
        t: Number(m[3]),
        tEnd: Number(m[4]),
        dt: Number(m[5]),
        pct: Number(m[6]),
      };
    }
  }
  return null;
}
