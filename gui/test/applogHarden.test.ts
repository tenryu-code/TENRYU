import { describe, expect, it } from "vitest";
import { logLine, logOp } from "@tenryu-common/core/applog";
import { defaultFormState, type FormState } from "../src/core/deck/formState";
import { computeShapeRadialRegions } from "../src/core/deck/meshAuto";
import { defaultShape2D, resolveShape } from "../src/core/geometry2d";
import { computeLocalMeshPreview } from "../src/core/localMeshPreview";
import { q } from "../src/core/units";

function degenerateShapeForm(): FormState {
  const f = defaultFormState();
  f.main.dimension = "2D_RZ";
  f.mesh.radialZoning2d = "regions";
  const shape = defaultShape2D("solidSphere");
  shape.radius = q(Number.NaN, "µm");
  f.geometry.shapes2d = [shape];
  return f;
}

describe("application logging hardening", () => {
  it("degrades to console-only logging without Tauri", () => {
    expect(() => logLine("info", "test line")).not.toThrow();
    expect(() => logOp("test op")).not.toThrow();
  });

  it("resolves a NaN-valued quantity to finite shape numbers", () => {
    const shape = defaultShape2D("solidSphere");
    shape.radius = { value: Number.NaN, unit: "µm" };
    const resolved = resolveShape(shape);
    const numbers = [
      resolved.rho,
      resolved.TeEV,
      resolved.TiEV,
      resolved.z0,
      resolved.radius,
      resolved.rIn,
      resolved.r0,
      resolved.r1,
      resolved.z1,
      resolved.zApex,
      resolved.zBase,
      resolved.baseRadius,
      ...resolved.vertices.flatMap((vertex) => [vertex.r, vertex.z]),
    ];

    expect(numbers.every(Number.isFinite)).toBe(true);
  });

  it("returns null for shape radial regions with only a NaN radius", () => {
    expect(() => computeShapeRadialRegions(degenerateShapeForm())).not.toThrow();
    expect(computeShapeRadialRegions(degenerateShapeForm())).toBeNull();
  });

  it("returns null for a local regional preview with only a NaN radius", () => {
    expect(() => computeLocalMeshPreview(degenerateShapeForm())).not.toThrow();
    expect(computeLocalMeshPreview(degenerateShapeForm())).toBeNull();
  });
});
