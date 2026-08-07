import { describe, expect, it } from "vitest";
import { defaultFormState, ensureBackgroundGas, validateFormState } from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";
import { computeShapeRadialRegions, computeShapeZSegments } from "../src/core/deck/meshAuto";
import { defaultShape2D } from "../src/core/geometry2d";
import { q } from "../src/core/units";
import { t } from "../src/i18n";

function twoRegion2d(logicalMesh2d: "rectangular_rz" | "spherical_polar_halfplane") {
  const f = defaultFormState();
  const sMaxCm = 3;
  f.main.dimension = "2D_RZ";
  f.mesh.logicalMesh2d = logicalMesh2d;
  f.mesh.radialZoning2d = "regions";
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
  const inner = defaultShape2D("solidSphere");
  inner.materialName = "CH";
  inner.rho = 1.0;
  inner.z0 = q(0, "cm");
  inner.radius = q(sMaxCm / 3, "cm");
  const outer = defaultShape2D("shell");
  outer.materialName = "outer";
  outer.rho = 0.05;
  outer.z0 = q(0, "cm");
  outer.rIn = q(sMaxCm / 3, "cm");
  outer.radius = q(sMaxCm, "cm");
  f.geometry.regions = [];
  f.geometry.shapes2d = [inner, outer];
  f.geometry.background2d = {
    materialName: "outer",
    rho: 0.05,
    Te: q(1, "eV"),
    Ti: q(1, "eV"),
  };
  return f;
}

function polarInBox2d() {
  const f = defaultFormState();
  f.main.dimension = "2D_RZ";
  f.mesh.meshMode2d = "polar_in_box";
  ensureBackgroundGas(f);
  const inner = defaultShape2D("solidSphere");
  inner.materialName = "gas";
  inner.rho = 1.0e-4;
  inner.z0 = q(0, "µm");
  inner.radius = q(200, "µm");
  const shell = defaultShape2D("shell");
  shell.materialName = "CH";
  shell.rho = 1.0;
  shell.z0 = q(0, "µm");
  shell.rIn = q(200, "µm");
  shell.radius = q(300, "µm");
  f.geometry.regions = [];
  f.geometry.shapes2d = [inner, shell];
  return f;
}

describe("computeRegionSegments2d", () => {
  it("allocates polar radial cells in proportion to spherical shell mass", () => {
    const f = twoRegion2d("spherical_polar_halfplane");
    const segs = computeShapeRadialRegions(f);
    expect(segs).not.toBeNull();
    expect(segs).toHaveLength(2);
    const counts = segs!.map((seg) => seg.nz);
    expect(counts.reduce((sum, count) => sum + count, 0)).toBe(40);
    expect(counts.every((count) => count >= 1)).toBe(true);

    const sMax = 3;
    const masses = [1.0 * ((sMax / 3) ** 3 - 0 ** 3), 0.05 * (sMax ** 3 - (sMax / 3) ** 3)];
    const massTotal = masses[0] + masses[1];
    counts.forEach((count, i) => {
      const expected = (40 * masses[i]) / massTotal;
      expect(Math.abs(count - expected)).toBeLessThanOrEqual(1);
    });
    expect(segs![0].rEndCm).toBeCloseTo(1, 12);
    expect(segs![1].rEndCm).toBeCloseTo(3, 12);
  });

  it("uses cylindrical radial weights for rectangular_rz", () => {
    const segs = computeShapeRadialRegions(twoRegion2d("rectangular_rz"));
    expect(segs).not.toBeNull();
    expect(segs).toHaveLength(2);
    expect(segs!.reduce((sum, seg) => sum + seg.nz, 0)).toBe(40);
    expect(segs!.every((seg) => seg.nz >= 1)).toBe(true);
    expect(segs![0].rEndCm).toBeCloseTo(1, 12);
    expect(segs![1].rEndCm).toBeCloseTo(3, 12);
  });
});

describe("computeShapeZSegments", () => {
  it("pins shape interfaces and allocates axial cells by sampled mass", () => {
    const f = defaultFormState();
    f.main.dimension = "2D_RZ";
    f.mesh.zMin = q(-600, "µm");
    f.mesh.zMax = q(600, "µm");
    f.mesh.nz = 64;
    const sphere = defaultShape2D("solidSphere");
    sphere.z0 = q(0, "µm");
    sphere.radius = q(300, "µm");
    sphere.rho = 1.0;
    f.geometry.shapes2d = [sphere];
    f.geometry.background2d.rho = 1.0e-4;

    const segs = computeShapeZSegments(f);
    expect(segs).not.toBeNull();
    expect(segs).toHaveLength(3);
    expect(
      Math.abs(segs![0].zEndCm - q(-300, "µm").value * 1.0e-4) /
        Math.abs(q(-300, "µm").value * 1.0e-4),
    ).toBeLessThanOrEqual(1.0e-12);
    expect(
      Math.abs(segs![1].zEndCm - q(300, "µm").value * 1.0e-4) /
        Math.abs(q(300, "µm").value * 1.0e-4),
    ).toBeLessThanOrEqual(1.0e-12);
    const counts = segs!.map((segment) => segment.count);
    expect(counts.reduce((sum, count) => sum + count, 0)).toBe(64);
    expect(counts.every((count) => count >= 1)).toBe(true);
    expect(counts[1]).toBe(Math.max(...counts));
  });

  it("returns one uniform span for a background-only form", () => {
    const f = defaultFormState();
    f.main.dimension = "2D_RZ";
    f.geometry.shapes2d = [];
    const segs = computeShapeZSegments(f);

    expect(segs).toEqual([
      {
        zStartCm: q(-500, "µm").value * 1.0e-4,
        zEndCm: q(500, "µm").value * 1.0e-4,
        count: f.mesh.nz,
      },
    ]);
  });
});

describe("2D mesh deck generation", () => {
  it("emits rectangular regional auto zoning", () => {
    const f = twoRegion2d("rectangular_rz");
    f.radiation.enabled = false;
    f.laser.enabled = false;
    f.conduction.enabled = false;
    const deck = generateDeck(f);
    const meshBlock = deck.slice(deck.indexOf("Mesh("), deck.indexOf("\n)\n", deck.indexOf("Mesh(")) + 3);
    expect(meshBlock).not.toContain("logical_mesh_2d=");
    expect(meshBlock).not.toContain("spherical_polar_s_max=");
    expect(meshBlock).toContain("auto_regions=[");
    expect(meshBlock).toContain("dict(r_end=");
    expect(meshBlock).not.toContain("topology_scheme=");
    expect(meshBlock).not.toMatch(/^    nr=/m);
    expect(meshBlock).toContain("z_min=-3,");
    expect(meshBlock).toContain("z_max=3,");
    expect(meshBlock).toContain("grid_z=dict(");
    expect(meshBlock).toContain('type="graded"');
    expect(meshBlock.match(/^            dict\(r_start=/gm)).toHaveLength(3);
    expect(meshBlock).toContain("nz=64,");
    expect(deck).toContain("_GUI_SHAPES = [");
    expect(deck).toContain("import math");
  });

  it("preserves rectangular_rz uniform mesh emission", () => {
    const f = defaultFormState();
    f.main.dimension = "2D_RZ";
    f.geometry.regions = [];
    f.geometry.background2d = {
      materialName: "CH",
      rho: 1.0e-4,
      Te: q(1, "eV"),
      Ti: q(1, "eV"),
    };
    const deck = generateDeck(f);
    const meshBlock = deck.slice(deck.indexOf("Mesh("), deck.indexOf("\n)\n", deck.indexOf("Mesh(")) + 3);
    expect(meshBlock).toContain('grid="uniform"');
    expect(meshBlock).toContain("nr=");
    expect(meshBlock).toContain("nz=");
  });
});

describe("polar_in_box improved mesh", () => {
  it("emits the planner recipe and replaces rectangular auto zoning", () => {
    const deck = generateDeck(polarInBox2d());

    expect(deck).toContain("def _tenryu_repo_root():");
    expect(deck).not.toContain("sys.path.insert(0, os.getcwd())");
    expect(deck).toContain("from tools.mesh_planner import (");
    expect(deck).toContain("PolarSolidSphere(");
    expect(deck.match(/PolarShell\(/g)).toHaveLength(1);
    expect(deck).toContain('center_mode="graded_button"');
    expect(deck).toContain('logical_mesh_2d="polar_in_box"');
    expect(deck).toContain("morph_rings=36");
    expect(deck).toContain("nz=48");
    expect(deck).not.toContain("auto_regions=");
    expect(deck).not.toContain("grid_z=");
    expect(deck).not.toContain("r_min=");
  });

  it("accepts a contiguous origin family and rejects a radial gap", () => {
    const f = polarInBox2d();
    expect(validateFormState(f)).toEqual([]);

    f.geometry.shapes2d[1].rIn = q(250, "µm");
    expect(validateFormState(f)).toContain(t().validation.pibFamilyNotContiguous);
  });

  it("requires an origin-centered sphere or shell", () => {
    const f = defaultFormState();
    f.main.dimension = "2D_RZ";
    f.mesh.meshMode2d = "polar_in_box";
    f.geometry.regions = [];
    f.geometry.background2d.materialName = "CH";
    const block = defaultShape2D("block");
    block.materialName = "CH";
    f.geometry.shapes2d = [block];

    expect(validateFormState(f)).toContain(t().validation.pibNeedsOriginSphere);
  });
});
