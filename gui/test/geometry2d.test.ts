import { describe, expect, it } from "vitest";
import { defaultFormState, migrateFormState, type FormState } from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";
import { computeShapeRadialRegions } from "../src/core/deck/meshAuto";
import {
  defaultShape2D,
  resolveGeometry2D,
  resolveShape,
  shapeContains,
  shapeOutlinePoints,
  stateAt,
  type Shape2D,
  type Shape2DKind,
} from "../src/core/geometry2d";
import { q } from "../src/core/units";

describe("shapeContains", () => {
  it.each([
    ["solidSphere", [0.01, 0], [0.03, 0]],
    ["shell", [0.015, 0], [0.005, 0]],
    ["block", [0.01, 0.005], [0.03, 0.005]],
    ["cone", [0.002, 0.01], [0.008, 0.01]],
    ["polygon", [0.005, 0.005], [0.02, 0.02]],
  ] satisfies Array<[Shape2DKind, [number, number], [number, number]]>)(
    "%s includes an inside point and excludes an outside point",
    (kind, inside, outside) => {
      const shape = resolveShape(defaultShape2D(kind));
      expect(shapeContains(shape, inside[0], inside[1])).toBe(true);
      expect(shapeContains(shape, outside[0], outside[1])).toBe(false);
    },
  );
});

describe("shapeOutlinePoints", () => {
  it("returns null for a kind-switched polygon with no vertices", () => {
    const shape = defaultShape2D("block");
    shape.kind = "polygon" as Shape2DKind;

    expect(shapeOutlinePoints(resolveShape(shape))).toBeNull();
  });

  it("returns a closed outline for a triangle polygon", () => {
    const outlines = shapeOutlinePoints(resolveShape(defaultShape2D("polygon")));

    expect(outlines).not.toBeNull();
    expect(outlines![0][0]).toEqual(outlines![0][outlines![0].length - 1]);
  });

  it("returns finite points for a solid sphere", () => {
    const outlines = shapeOutlinePoints(resolveShape(defaultShape2D("solidSphere")));

    expect(outlines).not.toBeNull();
    expect(outlines!.flat().every(([r, z]) => Number.isFinite(r) && Number.isFinite(z))).toBe(true);
  });
});

describe("2D painter composition", () => {
  it("uses the last overlapping shape and falls back to the background", () => {
    const lower = defaultShape2D("solidSphere");
    lower.materialName = "lower";
    lower.rho = 1;
    const upper = defaultShape2D("solidSphere");
    upper.materialName = "upper";
    upper.rho = 2;
    const geometry = resolveGeometry2D([lower, upper], {
      materialName: "ambient",
      rho: 1.0e-4,
      Te: q(3, "eV"),
      Ti: q(4, "eV"),
    });

    expect(stateAt(geometry, 0.01, 0).materialName).toBe("upper");
    expect(stateAt(geometry, 0.03, 0)).toEqual({
      materialName: "ambient",
      rho: 1.0e-4,
      TeEV: 3,
      TiEV: 4,
    });
  });
});

describe("legacy 2D migration", () => {
  it("derives shells and preserves the trailing last-region state", () => {
    const original = defaultFormState();
    original.main.dimension = "2D_RZ";
    original.mesh.rMax = q(400, "µm");
    original.materials.push({
      ...original.materials[0],
      name: "outer",
    });
    original.geometry.regions = [
      { materialName: "CH", rOuter: q(200, "µm"), rho: 1, Te: q(2, "eV"), Ti: q(3, "eV") },
      { materialName: "outer", rOuter: q(400, "µm"), rho: 0.1, Te: q(4, "eV"), Ti: q(5, "eV") },
    ];
    const raw = JSON.parse(JSON.stringify(original)) as {
      geometry: Partial<FormState["geometry"]>;
    };
    delete raw.geometry.shapes2d;
    delete raw.geometry.background2d;

    const migrated = migrateFormState(raw as FormState);
    expect(migrated.geometry.shapes2d).toHaveLength(2);
    expect(migrated.geometry.shapes2d.map((shape) => shape.kind)).toEqual(["shell", "shell"]);
    const geometry = resolveGeometry2D(migrated.geometry.shapes2d, migrated.geometry.background2d);
    expect(stateAt(geometry, 0.01, 0)).toEqual({ materialName: "CH", rho: 1, TeEV: 2, TiEV: 3 });
    expect(stateAt(geometry, 0.03, 0)).toEqual({ materialName: "outer", rho: 0.1, TeEV: 4, TiEV: 5 });
    expect(stateAt(geometry, 0.05, 0)).toEqual({ materialName: "outer", rho: 0.1, TeEV: 4, TiEV: 5 });
  });
});

function shapePainterForm(): FormState {
  const f = defaultFormState();
  f.main.dimension = "2D_RZ";
  f.mesh.logicalMesh2d = "rectangular_rz";
  f.mesh.radialZoning2d = "regions";
  f.mesh.rMax = q(600, "µm");
  f.mesh.zMin = q(-600, "µm");
  f.mesh.zMax = q(600, "µm");
  f.mesh.nr = 40;
  f.radiation.enabled = false;
  f.laser.enabled = false;
  f.conduction.enabled = false;
  f.materials.push(
    { ...f.materials[0], name: "fuel" },
    { ...f.materials[0], name: "ambient" },
  );
  f.geometry.regions[0].rOuter = q(600, "µm");

  const fuel = defaultShape2D("solidSphere");
  fuel.materialName = "fuel";
  fuel.rho = 0.25;
  fuel.radius = q(350, "µm");
  const shell = defaultShape2D("shell");
  shell.materialName = "CH";
  shell.rho = 1.05;
  shell.rIn = q(350, "µm");
  shell.radius = q(400, "µm");
  f.geometry.shapes2d = [fuel, shell] satisfies Shape2D[];
  f.geometry.background2d = {
    materialName: "ambient",
    rho: 1.0e-4,
    Te: q(1, "eV"),
    Ti: q(1, "eV"),
  };
  return f;
}

describe("shape-painter deck generation", () => {
  it("emits the resolved table, painter functions, and material volume fractions", () => {
    const f = shapePainterForm();
    const deck = generateDeck(f);

    expect(deck).toContain("import math");
    expect(deck).toContain("_GUI_SHAPES = [");
    expect(deck).toContain("def _gui_state_at(r_cm, z_cm):");
    expect(deck).toContain("def gui_rho(r_cm, z_cm):");
    for (const material of f.materials) {
      expect(deck).toContain(`def gui_vf_${material.name}(r_cm, z_cm):`);
    }
    expect(deck).not.toContain("def gui_rho(r_cm, z_cm):\n    if ");
  });
});

describe("computeShapeRadialRegions", () => {
  it("derives shape boundaries, mass allocations, and painter-state densities", () => {
    const regions = computeShapeRadialRegions(shapePainterForm());

    expect(regions).not.toBeNull();
    regions!.map((region) => region.rEndCm).forEach((rEnd, i) => {
      expect(rEnd).toBeCloseTo([0.035, 0.04, 0.06][i], 12);
    });
    expect(regions!.reduce((sum, region) => sum + region.nz, 0)).toBe(40);
    expect(regions!.map((region) => region.rhoRefGcc)).toEqual([0.25, 1.05, 1.0e-4]);
  });
});
