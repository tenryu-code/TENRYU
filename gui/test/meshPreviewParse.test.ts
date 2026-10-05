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
  it("parses a 1D preview, whose axial extent the solver writes as null, with its requirement", () => {
    // Field names and nulls as tenryu validate --mesh-preview writes them for a 1D laser deck
    // (main@efa376c7b): z_min, z_max and z_nodes are null and the requirement carries its bands
    // and the check of the cells against them.
    const payload = {
      dim: 1,
      dimension: "1D_SPH",
      logical_mesh_2d: "rectangular_rz",
      geometry_1d: "spherical",
      nr: 3,
      nz: 1,
      r_min: 0,
      r_max: 0.04,
      z_min: null,
      z_max: null,
      polar: null,
      r_nodes: [0, 0.0243, 0.025, 0.04],
      z_nodes: null,
      zoning_pins: [{ r: 0.0243, ratio_jump_allowed: true }],
      zoning_measure: "spherical_cell_mass",
      rho0_cells: [0.02, 1.05, 1e-10],
      material_cells: [0, 1, 2],
      mesh_requirement: {
        applicable: true,
        reason: "",
        params: { apply: "enforce" },
        inputs: { R0_cm: 0.025, area_cm2: 0.007853981633974483 },
        ablation: { dr_min_admissible_cm: 6.958264229083562e-7 },
        bands_recommended: [
          { kind: "formation", r_lo_cm: 0.024877925699027904, r_hi_cm: 0.025, areal_mass_max_g_cm2: 7.30617744053774e-7 },
        ],
        requirement_check: {
          ok: true,
          ablation: { applicable: true, n_checked: 1386, n_violations: 0, max_ratio: 0.8694159742913912, worst: { cell: 1376 } },
          shock: { applicable: true, n_checked: 0, n_violations: 0, max_ratio: 0, worst: { cell: -1 } },
        },
      },
    };
    const result = parseMeshPreview(`TENRYU-MESH-PREVIEW: ${JSON.stringify(payload)}`);

    expect(result).not.toBeNull();
    expect(result?.zMin).toBeNull();
    expect(result?.zMax).toBeNull();
    expect(result?.rNodes).toEqual([0, 0.0243, 0.025, 0.04]);
    expect(result?.rho0Cells).toEqual([0.02, 1.05, 1e-10]);
    expect(result?.requirement?.apply).toBe("enforce");
    expect(result?.requirement?.bands).toEqual([
      { kind: "formation", rLoCm: 0.024877925699027904, rHiCm: 0.025, arealMassMaxGcm2: 7.30617744053774e-7 },
    ]);
    expect(result?.requirement?.ablation?.worstCell).toBe(1376);
    expect(result?.requirement?.drMinAdmissibleCm).toBe(6.958264229083562e-7);
  });

  it("returns null for a 2D preview without its axial extent", () => {
    const payload = {
      dim: 2,
      dimension: "2D_RZ",
      logical_mesh_2d: "rectangular_rz",
      geometry_1d: "spherical",
      nr: 2,
      nz: 2,
      r_min: 0,
      r_max: 1,
      z_min: null,
      z_max: null,
      polar: null,
      r_nodes: null,
      z_nodes: null,
    };

    expect(parseMeshPreview(`TENRYU-MESH-PREVIEW: ${JSON.stringify(payload)}`)).toBeNull();
  });
});
