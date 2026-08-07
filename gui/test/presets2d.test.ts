import { describe, expect, it } from "vitest";
import {
  defaultFormState,
  ensureBackgroundGas,
  validateFormState,
  type FormState,
} from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";
import {
  preset2dBlank,
  preset2dPolarCapsule,
  preset2dPolarSphere,
  preset2dRectLaser,
  preset2dRectSlabRad,
} from "../src/core/presets";
import { q } from "../src/core/units";

function expectValidAndGenerate(build: () => FormState): string {
  const f = build();
  expect(validateFormState(f).length).toBe(0);
  let deck = "";
  expect(() => {
    deck = generateDeck(f);
  }).not.toThrow();
  return deck;
}

describe("2D presets", () => {
  it("blank validates and generates", () => {
    const f = preset2dBlank();
    expect(f.materials.some((material) => material.name === "gas")).toBe(true);
    expect(f.geometry.background2d.materialName).toBe("gas");
    const deck = expectValidAndGenerate(() => f);
    expect(deck).toContain("_GUI_SHAPES = [");
    expect(deck).toContain('_GUI_BG = ("gas",');
  });

  it("ensures a default 2D form has one background gas material", () => {
    const f = defaultFormState();
    f.main.dimension = "2D_RZ";
    ensureBackgroundGas(f);
    ensureBackgroundGas(f);

    expect(validateFormState(f)).toEqual([]);
    expect(f.materials.filter((material) => material.name === "gas")).toHaveLength(1);
    expect(f.geometry.background2d.materialName).toBe("gas");
  });

  it("solid sphere validates and generates regional rectangular mesh", () => {
    const f = preset2dPolarSphere();
    expect(f.main.name).toBe("solid_sphere_2d");
    expect(f.mesh.logicalMesh2d).toBe("rectangular_rz");
    expect(f.mesh.zMin).toEqual(q(-600, "µm"));
    expect(f.mesh.zMax).toEqual(q(600, "µm"));
    expect(f.mesh.nz).toBe(96);
    const deck = expectValidAndGenerate(() => f);
    expect(deck).not.toContain("logical_mesh_2d=");
    expect(deck).toContain("auto_regions=[");
    expect(deck).toContain("grid_z=dict(");
  });

  it("solid sphere generates gas material without ambient", () => {
    const deck = expectValidAndGenerate(preset2dPolarSphere);
    expect(deck).toContain("gui_vf_gas");
    expect(deck).not.toContain("ambient");
  });

  it("capsule validates and generates regional rectangular mesh", () => {
    const f = preset2dPolarCapsule();
    expect(f.main.name).toBe("capsule_2d");
    expect(f.mesh.logicalMesh2d).toBe("rectangular_rz");
    expect(f.mesh.zMin).toEqual(q(-800, "µm"));
    expect(f.mesh.zMax).toEqual(q(800, "µm"));
    expect(f.mesh.nz).toBe(128);
    const deck = expectValidAndGenerate(() => f);
    expect(deck).not.toContain("logical_mesh_2d=");
    expect(deck).toContain("auto_regions=[");
    expect(deck).toContain("grid_z=dict(");
  });

  it("radiation slab validates and generates Marshak 2D deck", () => {
    const deck = expectValidAndGenerate(preset2dRectSlabRad);
    expect(deck).toContain('dimension="2D_RZ"');
    expect(deck).toContain("marshak");
    expect(deck).toContain("_GUI_SHAPES = [");
    expect(deck).toContain("_GUI_BG");
  });

  it("laser cylinder validates and generates raytrace_3d deck", () => {
    const deck = expectValidAndGenerate(preset2dRectLaser);
    expect(deck).toContain('(\"block\", \"CD\"');
    expect(deck).toContain('mode="raytrace_3d"');
  });
});
