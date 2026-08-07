import { describe, expect, it } from "vitest";
import { computeMeshStats } from "../src/ui/MeshMassChart";

describe("computeMeshStats", () => {
  it("computes ratios and widths", () => {
    const edges = [0, 0.01, 0.02, 0.03, 0.04];
    const masses = [1, 2, 4, 4];
    const names = ["a", "a", "b", "b"];
    const s = computeMeshStats(edges, masses, names)!;
    expect(s.massRatio).toBeCloseTo(4, 12);
    expect(s.interfaceRatios).toHaveLength(1);
    expect(s.interfaceRatios[0]).toEqual({ left: "a", right: "b", ratio: 2 });
    expect(s.drMinUm).toBeCloseTo(100, 9);
    expect(s.dr0Um).toBeCloseTo(100, 9);
  });
  it("rejects non-positive masses", () => {
    expect(computeMeshStats([0, 1], [0], ["a"])).toBeNull();
  });
});
