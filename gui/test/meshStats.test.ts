import { describe, expect, it } from "vitest";
import { cellMetrics, summarizeMesh } from "../src/core/deck/mesh1d";
import { defaultFormState } from "../src/core/deck/formState";
import { q } from "../src/core/units";

describe("summarizeMesh", () => {
  it("reports the smallest width, the adjacent mass ratios and the interface ratios", () => {
    const f = defaultFormState();
    f.main.geometry1d = "planar";
    f.mesh.rMax = q(400, "µm");
    f.materials = [{ ...f.materials[0], name: "a" }, { ...f.materials[0], name: "b" }];
    f.geometry.regions = [
      { materialName: "a", rOuter: q(200, "µm"), rho: 1, Te: q(1, "eV"), Ti: q(1, "eV") },
      { materialName: "b", rOuter: q(400, "µm"), rho: 2, Te: q(1, "eV"), Ti: q(1, "eV") },
    ];
    const edges = [0, 0.01, 0.02, 0.03, 0.04];
    const metrics = cellMetrics(f, edges)!;
    [0.01, 0.01, 0.02, 0.02].forEach((mass, i) => expect(metrics.masses[i]).toBeCloseTo(mass, 15));
    const s = summarizeMesh(metrics);
    expect(s.nCells).toBe(4);
    expect(s.minWidthCm).toBeCloseTo(0.01, 15);
    expect(s.maxAdjacentMassRatio).toBeCloseTo(2, 12);
    expect(s.interfaceRatios).toHaveLength(1);
    expect(s.interfaceRatios[0]).toMatchObject({ left: "a", right: "b", rCm: 0.02 });
    expect(s.interfaceRatios[0].ratio).toBeCloseTo(2, 12);
    expect(s.surfaceWidthCm).toBeCloseTo(0.01, 15);
    expect(s.surfaceLocalArealMass).toBeCloseTo(0.02, 15);
  });

  it("excludes the void padding from the surface cell and the ratios", () => {
    const f = defaultFormState();
    f.main.geometry1d = "planar";
    f.mesh.rMax = q(400, "µm");
    f.geometry.regions = [{ materialName: f.materials[0].name, rOuter: q(300, "µm"), rho: 1, Te: q(1, "eV"), Ti: q(1, "eV") }];
    f.geometry.vacuumOutside1d = true;
    const metrics = cellMetrics(f, [0, 0.01, 0.02, 0.03, 0.04])!;
    expect(metrics.materials).toEqual(["CH", "CH", "CH", "VOID"]);
    const s = summarizeMesh(metrics);
    expect(s.maxAdjacentMassRatio).toBeCloseTo(1, 12);
    expect(s.surfaceWidthCm).toBeCloseTo(0.01, 15);
  });
});
