import { describe, expect, it } from "vitest";
import { defaultFormState } from "../src/core/deck/formState";
import { computeRegionSegments } from "../src/core/deck/meshAuto";
import { q } from "../src/core/units";

function twoRegionPlanar() {
  const f = defaultFormState();
  f.main.geometry1d = "planar";
  f.mesh.rMax = q(400, "µm");
  f.mesh.grid1d = "graded";
  f.mesh.segmentSource = "regions";
  f.mesh.nr = 100;
  f.materials.push({
    name: "Au",
    A: 197,
    Z: 79,
    eosModel: "ideal_gas",
    gamma: 5 / 3,
    cvEOverride: undefined,
    eosFile: "",
    opacityModel: "constant",
    kappaA: 1.0,
    kappaS: 0.0,
    opacityFile: "",
  });
  f.geometry.regions = [
    { materialName: "CH", rOuter: q(200, "µm"), rho: 1.0, Te: q(1, "eV"), Ti: q(1, "eV") },
    { materialName: "Au", rOuter: q(400, "µm"), rho: 3.0, Te: q(1, "eV"), Ti: q(1, "eV") },
  ];
  return f;
}

describe("computeRegionSegments", () => {
  it("splits cells in proportion to region mass (planar 1:3)", () => {
    const segs = computeRegionSegments(twoRegionPlanar());
    expect(segs).not.toBeNull();
    expect(segs).toHaveLength(2);
    expect(segs![0].nr).toBe(25);
    expect(segs![1].nr).toBe(75);
    expect(segs![0].rEndCm).toBeCloseTo(0.02, 12);
    expect(segs![1].rEndCm).toBeCloseTo(0.04, 12);
  });

  it("totals exactly mesh.nr for awkward mass shares", () => {
    const f = twoRegionPlanar();
    f.geometry.regions[1].rho = 2.0;
    const segs = computeRegionSegments(f);
    expect(segs).not.toBeNull();
    expect(segs![0].nr + segs![1].nr).toBe(100);
  });

  it("override pins a region and the pool redistributes", () => {
    const f = twoRegionPlanar();
    f.mesh.regionNrOverrides = [10, null];
    const segs = computeRegionSegments(f);
    expect(segs).not.toBeNull();
    expect(segs![0].nr).toBe(10);
    expect(segs![1].nr).toBe(90);
  });

  it("a tiny-mass region still gets at least one cell", () => {
    const f = twoRegionPlanar();
    f.geometry.regions[0].rOuter = q(0.4, "µm");
    const segs = computeRegionSegments(f);
    expect(segs).not.toBeNull();
    expect(segs![0].nr).toBeGreaterThanOrEqual(1);
    expect(segs![0].nr + segs![1].nr).toBe(100);
  });

  it("returns null when overrides exhaust the total", () => {
    const f = twoRegionPlanar();
    f.mesh.regionNrOverrides = [100, null];
    expect(computeRegionSegments(f)).toBeNull();
  });
});
