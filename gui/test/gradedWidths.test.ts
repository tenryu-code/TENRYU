import { describe, expect, it } from "vitest";
import { computeGradedWidths } from "../src/ui/MeshMassChart";

const GRADING = { edgeRatio: 0.1, sgOrder: 4, sgSigma: 0.7 };

describe("computeGradedWidths", () => {
  it("edgeRatio ~ 1 degenerates to uniform", () => {
    const w = computeGradedWidths([{ rStart: 0, rEnd: 0.05, nr: 10 }], {
      edgeRatio: 1.0,
      sgOrder: 4,
      sgSigma: 0.7,
    });
    expect(w).not.toBeNull();
    for (const dw of w![0]) expect(dw).toBeCloseTo(0.005, 12);
  });

  it("preserves segment lengths and positivity", () => {
    const segs = [
      { rStart: 0, rEnd: 0.03, nr: 20 },
      { rStart: 0.03, rEnd: 0.05, nr: 15 },
    ];
    const w = computeGradedWidths(segs, GRADING)!;
    for (let s = 0; s < segs.length; s++) {
      const sum = w[s].reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(segs[s].rEnd - segs[s].rStart, 12);
      for (const dw of w[s]) expect(dw).toBeGreaterThan(0);
    }
  });

  it("matches junction widths to the geometric mean", () => {
    const segs = [
      { rStart: 0, rEnd: 0.03, nr: 20 },
      { rStart: 0.03, rEnd: 0.05, nr: 15 },
    ];
    const w = computeGradedWidths(segs, GRADING)!;
    expect(w[0][w[0].length - 1]).toBeCloseTo(w[1][0], 12);
  });

  it("differs from uniform when edgeRatio < 1", () => {
    const w = computeGradedWidths([{ rStart: 0, rEnd: 0.05, nr: 10 }], GRADING)!;
    const uniform = 0.005;
    const deviates = w[0].some((dw) => Math.abs(dw - uniform) > 1e-4);
    expect(deviates).toBe(true);
  });
});
