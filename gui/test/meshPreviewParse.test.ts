import { describe, expect, it } from "vitest";
import { parseMeshPreview } from "../src/core/meshPreviewParse";

describe("parseMeshPreview", () => {
  it("parses a polar preview line", () => {
    const rNodes = Array.from({ length: 41 }, (_, i) => (0.03 * i) / 40);
    const payload = {
      dim: 2,
      dimension: "2D_RZ",
      logical_mesh_2d: "spherical_polar_halfplane",
      geometry_1d: "spherical",
      nr: 40,
      nz: 32,
      r_min: 0,
      r_max: 0.03,
      z_min: -0.03,
      z_max: 0.03,
      polar: {
        s_max: 0.03,
        kappa: 0.5,
        center_treatment: "tri_fan",
        equal_mu: false,
      },
      r_nodes: rNodes,
      z_nodes: null,
    };
    const result = parseMeshPreview(`TENRYU-MESH-PREVIEW: ${JSON.stringify(payload)}`);

    expect(result).not.toBeNull();
    expect(result?.polar?.sMax).toBe(0.03);
    expect(result?.polar?.centerTreatment).toBe("tri_fan");
    expect(result?.rNodes).toHaveLength(41);
  });

  it("returns null when the marker is absent", () => {
    expect(parseMeshPreview("Configuration validated successfully.")).toBeNull();
  });

  it("returns null when r_nodes has the wrong length", () => {
    const payload = {
      dim: 2,
      dimension: "2D_RZ",
      logical_mesh_2d: "rectangular_rz",
      geometry_1d: "spherical",
      nr: 4,
      nz: 4,
      r_min: 0,
      r_max: 1,
      z_min: -1,
      z_max: 1,
      polar: null,
      r_nodes: [0, 0.5, 1],
      z_nodes: null,
    };

    expect(parseMeshPreview(`TENRYU-MESH-PREVIEW: ${JSON.stringify(payload)}`)).toBeNull();
  });

  it("parses a rectangular preview with resolved z nodes", () => {
    const payload = {
      dim: 2,
      dimension: "2D_RZ",
      logical_mesh_2d: "rectangular_rz",
      geometry_1d: "spherical",
      nr: 2,
      nz: 2,
      r_min: 0,
      r_max: 1,
      z_min: -1,
      z_max: 1,
      polar: null,
      r_nodes: null,
      z_nodes: [-1, 0, 1],
    };
    const result = parseMeshPreview(`  TENRYU-MESH-PREVIEW: ${JSON.stringify(payload)}`);

    expect(result).not.toBeNull();
    expect(result?.polar).toBeNull();
    expect(result?.zNodes).toEqual([-1, 0, 1]);
  });
});
