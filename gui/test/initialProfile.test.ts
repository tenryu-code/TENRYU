import { describe, expect, it } from "vitest";
import { defaultFormState } from "../src/core/deck/formState";
import { defaultShape2D } from "../src/core/geometry2d";
import { sampleInitialProfile, type ProfileResult } from "../src/core/initialProfile";
import { q } from "../src/core/units";

function twoRegion2d(logicalMesh2d: "rectangular_rz" | "spherical_polar_halfplane") {
  const f = defaultFormState();
  const sMaxCm = 3;
  f.main.dimension = "2D_RZ";
  f.mesh.logicalMesh2d = logicalMesh2d;
  f.mesh.radialZoning2d = "regions";
  f.mesh.polarCenterTreatment = "tri_fan";
  f.mesh.rMax = q(sMaxCm, "cm");
  f.mesh.zMin = q(-sMaxCm, "cm");
  f.mesh.zMax = q(sMaxCm, "cm");
  f.mesh.nr = 40;
  f.materials.push({
    name: "outer",
    A: 1,
    Z: 1,
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
    { materialName: "CH", rOuter: q(sMaxCm / 3, "cm"), rho: 1.0, Te: q(1, "eV"), Ti: q(1, "eV") },
    { materialName: "outer", rOuter: q(sMaxCm, "cm"), rho: 0.05, Te: q(1, "eV"), Ti: q(1, "eV") },
  ];
  return f;
}

function valueNearest(result: ProfileResult, s: number): number {
  return result.samples.reduce((nearest, sample) =>
    Math.abs(sample.s - s) < Math.abs(nearest.s - s) ? sample : nearest,
  ).value;
}

describe("sampleInitialProfile", () => {
  it("samples polar density along the equatorial ray", () => {
    const result = sampleInitialProfile(twoRegion2d("spherical_polar_halfplane"), Math.PI / 2, "rho");
    expect(result).not.toBeNull();
    expect(result!.samples).toHaveLength(256);
    expect(result!.samples[0].s).toBe(0);
    expect(result!.samples[255].s).toBe(3);
    expect(valueNearest(result!, 0.5)).toBe(1.0);
    expect(valueNearest(result!, 2.0)).toBe(0.05);
    expect(result!.regionEdges).toEqual([1]);
  });

  it("produces the same polar profile in different directions", () => {
    const f = twoRegion2d("spherical_polar_halfplane");
    const at30 = sampleInitialProfile(f, Math.PI / 6, "rho");
    const at120 = sampleInitialProfile(f, (2 * Math.PI) / 3, "rho");
    expect(at30).toEqual(at120);
  });

  it("samples rectangular density along the R axis", () => {
    const result = sampleInitialProfile(twoRegion2d("rectangular_rz"), Math.PI / 2, "rho");
    expect(result).not.toBeNull();
    expect(valueNearest(result!, 0.5)).toBe(1.0);
    expect(valueNearest(result!, 2.0)).toBe(0.05);
    expect(result!.sEnd).toBe(3);
  });

  it("keeps the first rectangular region along the +Z axis", () => {
    const result = sampleInitialProfile(twoRegion2d("rectangular_rz"), 0, "rho");
    expect(result).not.toBeNull();
    expect(result!.samples.every((sample) => sample.value === 1.0)).toBe(true);
    expect(result!.sEnd).toBe(3);
  });

  it("samples polar Te in eV", () => {
    const result = sampleInitialProfile(twoRegion2d("spherical_polar_halfplane"), Math.PI / 2, "Te");
    expect(result).not.toBeNull();
    expect(valueNearest(result!, 0.5)).toBe(1);
  });

  it("samples a polar shape painter and detects its interface", () => {
    const f = defaultFormState();
    f.main.dimension = "2D_RZ";
    f.mesh.logicalMesh2d = "spherical_polar_halfplane";
    f.mesh.rMax = q(3, "cm");
    const sphere = defaultShape2D("solidSphere");
    sphere.materialName = "CH";
    sphere.rho = 1.0;
    sphere.radius = q(1, "cm");
    sphere.z0 = q(0, "cm");
    f.geometry.regions = [];
    f.geometry.shapes2d = [sphere];
    f.geometry.background2d = {
      materialName: "outer",
      rho: 0.05,
      Te: q(1, "eV"),
      Ti: q(1, "eV"),
    };

    const result = sampleInitialProfile(f, Math.PI / 2, "rho");
    expect(result).not.toBeNull();
    expect(valueNearest(result!, 0.5)).toBe(1.0);
    expect(valueNearest(result!, 2.0)).toBe(0.05);
    expect(result!.regionEdges).toHaveLength(1);
    expect(result!.regionEdges[0]).toBeGreaterThan(0.9);
    expect(result!.regionEdges[0]).toBeLessThan(1.1);
  });

  it("samples off-center shapes according to ray direction", () => {
    const f = defaultFormState();
    f.main.dimension = "2D_RZ";
    f.mesh.logicalMesh2d = "rectangular_rz";
    f.mesh.rMax = q(3, "cm");
    f.mesh.zMin = q(-3, "cm");
    f.mesh.zMax = q(3, "cm");
    const sphere = defaultShape2D("solidSphere");
    sphere.materialName = "CH";
    sphere.rho = 1.0;
    sphere.radius = q(0.5, "cm");
    sphere.z0 = q(1, "cm");
    f.geometry.regions = [];
    f.geometry.shapes2d = [sphere];
    f.geometry.background2d = {
      materialName: "outer",
      rho: 0.05,
      Te: q(1, "eV"),
      Ti: q(1, "eV"),
    };

    const alongZ = sampleInitialProfile(f, 0, "rho");
    const alongR = sampleInitialProfile(f, Math.PI / 2, "rho");
    expect(alongZ).not.toBeNull();
    expect(alongR).not.toBeNull();
    expect(valueNearest(alongZ!, 1.0)).toBe(1.0);
    expect(valueNearest(alongR!, 1.0)).toBe(0.05);
  });
});
