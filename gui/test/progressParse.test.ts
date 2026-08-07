import { describe, expect, it } from "vitest";
import { parseLastProgress } from "../src/core/progressParse";

const REAL_LINE =
  "[2026-07-11 12:00:00.123] [info] [TENRYU] [progress] step=3400/500000 t=1.360e-09/2.000e-09 dt=1.000e-12 (68.0%) elapsed=0:01:07";

describe("parseLastProgress", () => {
  it("parses a real driver line", () => {
    const p = parseLastProgress(REAL_LINE);
    expect(p).not.toBeNull();
    expect(p?.step).toBe(3400);
    expect(p?.maxSteps).toBe(500000);
    expect(p?.t).toBeCloseTo(1.36e-9, 15);
    expect(p?.tEnd).toBeCloseTo(2e-9, 15);
    expect(p?.pct).toBeCloseTo(68.0, 6);
  });
  it("returns the LAST sample of a multi-line tail", () => {
    const log = [
      "[x] [info] [TENRYU] [progress] step=100/500000 t=4.000e-10/2.000e-09 dt=1.000e-12 (20.0%) elapsed=0:00:01",
      "[x] [info] [TENRYU] some other line",
      "[x] [info] [TENRYU] [progress] step=200/500000 t=8.000e-10/2.000e-09 dt=1.000e-12 (40.0%) elapsed=0:00:02",
      "",
    ].join("\n");
    expect(parseLastProgress(log)?.step).toBe(200);
  });
  it("handles single-digit exponents from the mock", () => {
    const p = parseLastProgress("[x] [info] [TENRYU] [progress] step=100/500000 t=4.000e-10/2.000e-9 dt=1.000e-12 (20.0%) elapsed=0:00:01");
    expect(p?.tEnd).toBeCloseTo(2e-9, 15);
  });
  it("null when absent", () => {
    expect(parseLastProgress("no progress here")).toBeNull();
  });
});
