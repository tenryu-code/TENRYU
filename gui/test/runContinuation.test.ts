import { describe, expect, it } from "vitest";
import type { HistorySeries } from "@tenryu-common/core/results/historyParse";
import {
  MAX_STEPS_UPPER_BOUND,
  compareOutputDirs,
  continuationArgs,
  continuationIssues,
  latestCheckpointPrefix,
  mergeHistorySegments,
  sortRunOutputFiles,
} from "../src/core/runContinuation";

const RUN = "/home/u/tenryu_gui_runs/case_20260930";

function segment(t: number[], values: Record<string, number[]>, missing: string[] = []): HistorySeries {
  return {
    t: Float64Array.from(t),
    series: Object.fromEntries(Object.entries(values).map(([k, v]) => [k, Float64Array.from(v)])),
    missing,
  };
}

describe("continuation arguments and checks", () => {
  it("passes only the changed values after --restart", () => {
    expect(continuationArgs({ tEndS: null, maxSteps: null })).toEqual([]);
    expect(continuationArgs({ tEndS: 4e-9, maxSteps: null })).toEqual(["--t-end", "4e-9"]);
    expect(continuationArgs({ tEndS: 2.5e-9, maxSteps: 200000 })).toEqual([
      "--t-end",
      "2.5e-9",
      "--max-steps",
      "200000",
    ]);
    // The shortest decimal that reads back as the same double.
    expect(Number(continuationArgs({ tEndS: 1.2345678901234567e-9, maxSteps: null })[1])).toBe(
      1.2345678901234567e-9,
    );
  });

  it("refuses values the run has already reached", () => {
    const rec = { lastProgress: { step: 1100, t: 1.992e-9, pct: 99.6 } };
    expect(continuationIssues({ tEndS: null, maxSteps: null }, rec)).toEqual([]);
    expect(continuationIssues({ tEndS: 4e-9, maxSteps: 2000 }, rec)).toEqual([]);
    expect(continuationIssues({ tEndS: 1.5e-9, maxSteps: null }, rec)).toEqual(["tEndNotLater"]);
    expect(continuationIssues({ tEndS: 1.992e-9, maxSteps: null }, rec)).toEqual(["tEndNotLater"]);
    expect(continuationIssues({ tEndS: 0, maxSteps: null }, rec)).toEqual(["tEndInvalid"]);
    expect(continuationIssues({ tEndS: Number.NaN, maxSteps: null }, rec)).toEqual(["tEndInvalid"]);
    expect(continuationIssues({ tEndS: null, maxSteps: 1100 }, rec)).toEqual(["maxStepsNotLarger"]);
    expect(continuationIssues({ tEndS: null, maxSteps: 1.5 }, rec)).toEqual(["maxStepsInvalid"]);
    expect(continuationIssues({ tEndS: null, maxSteps: MAX_STEPS_UPPER_BOUND + 1 }, rec)).toEqual([
      "maxStepsInvalid",
    ]);
    expect(continuationIssues({ tEndS: null, maxSteps: MAX_STEPS_UPPER_BOUND }, rec)).toEqual([]);
    // Without a reported progress only the values themselves are checked.
    expect(continuationIssues({ tEndS: 1e-15, maxSteps: 1 }, { lastProgress: null })).toEqual([]);
  });
});

describe("output directories of a continued run", () => {
  it("orders the deck's directory before the numbered ones a restart creates", () => {
    const dirs = ["case_010", "case_002", "case", "case_001"];
    expect([...dirs].sort(compareOutputDirs)).toEqual(["case", "case_001", "case_002", "case_010"]);
  });

  it("sorts files by output directory, then by index", () => {
    const files = [
      `${RUN}/outputs/case_001/results/case_0000.h5`,
      `${RUN}/outputs/case/results/case_0002.h5`,
      `${RUN}/outputs/case/results/case_0000.h5`,
      `${RUN}/outputs/case_001/results/case_0001.h5`,
      `${RUN}/outputs/case/results/case_0001.h5`,
    ];
    expect(sortRunOutputFiles(files)).toEqual([
      `${RUN}/outputs/case/results/case_0000.h5`,
      `${RUN}/outputs/case/results/case_0001.h5`,
      `${RUN}/outputs/case/results/case_0002.h5`,
      `${RUN}/outputs/case_001/results/case_0000.h5`,
      `${RUN}/outputs/case_001/results/case_0001.h5`,
    ]);
  });

  it("restarts from the last checkpoint of the latest directory, not the highest index", () => {
    const listing = [
      `${RUN}/outputs/case/checkpoints/case_ckpt_0003.h5`,
      `${RUN}/outputs/case/checkpoints/case_ckpt_0004.h5`,
      `${RUN}/outputs/case_001/checkpoints/case_ckpt_0000.h5`,
      `${RUN}/outputs/case_001/checkpoints/case_ckpt_0001.h5`,
      "",
    ].join("\n");
    expect(latestCheckpointPrefix(listing.split("\n"))).toBe(`${RUN}/outputs/case_001/checkpoints/case_ckpt_0001`);
  });

  it("skips directories without checkpoints and strips the legacy rank suffix", () => {
    expect(
      latestCheckpointPrefix([
        `${RUN}/outputs/case/checkpoints/case_ckpt_000120_r0000.h5`,
        `${RUN}/outputs/case/checkpoints/case_ckpt_000240_r0000.h5`,
        `${RUN}/outputs/case/results/case_0003.h5`,
      ]),
    ).toBe(`${RUN}/outputs/case/checkpoints/case_ckpt_000240`);
    expect(latestCheckpointPrefix([`${RUN}/outputs/case/results/case_0003.h5`, ""])).toBeNull();
    expect(latestCheckpointPrefix([])).toBeNull();
  });
});

describe("joining the history of a continued run", () => {
  it("replaces the earlier rows from the continuation's first time on", () => {
    // The run went to t=5 with checkpoints at t=2; the continuation restarted at t=2 and went to t=8.
    const run = segment([0, 1, 2, 3, 4, 5], { e: [10, 11, 12, 13, 14, 15] });
    const continued = segment([3, 4, 5, 6, 7, 8], { e: [23, 24, 25, 26, 27, 28] });
    const joined = mergeHistorySegments([run, continued]);
    expect(Array.from(joined.t)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    expect(Array.from(joined.series.e)).toEqual([10, 11, 12, 23, 24, 25, 26, 27, 28]);
  });

  it("appends a continuation from the end and fills columns a segment lacks", () => {
    const run = segment([0, 1, 2], { e: [1, 2, 3] }, ["x"]);
    const continued = segment([3, 4], { e: [4, 5], k: [7, 8] }, ["x"]);
    const joined = mergeHistorySegments([run, continued]);
    expect(Array.from(joined.t)).toEqual([0, 1, 2, 3, 4]);
    expect(Array.from(joined.series.e)).toEqual([1, 2, 3, 4, 5]);
    expect(Array.from(joined.series.k).map((v) => (Number.isNaN(v) ? "nan" : v))).toEqual([
      "nan",
      "nan",
      "nan",
      7,
      8,
    ]);
    expect(joined.missing).toEqual(["x"]);
  });

  it("lets the latest of several continuations win", () => {
    const run = segment([0, 1, 2, 3], { e: [0, 1, 2, 3] });
    const first = segment([2, 3, 4], { e: [12, 13, 14] });
    const second = segment([4, 5], { e: [24, 25] });
    const joined = mergeHistorySegments([run, first, second]);
    expect(Array.from(joined.t)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(Array.from(joined.series.e)).toEqual([0, 1, 12, 13, 24, 25]);
  });

  it("returns a single history unchanged", () => {
    const run = segment([0, 1], { e: [1, 2] });
    expect(mergeHistorySegments([run])).toBe(run);
  });
});
