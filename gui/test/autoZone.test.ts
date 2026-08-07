import { describe, expect, it } from "vitest";
import {
  computeAutoZoneNodes,
  defaultAutoZoneConfig,
  type AutoZoneRegionTS,
} from "../src/core/autoZone";
import { defaultFormState } from "../src/core/deck/formState";
import { computeRegionSegments2d } from "../src/core/deck/meshAuto";
import { computeLocalMeshPreview } from "../src/core/localMeshPreview";
import { q } from "../src/core/units";

const GOLDEN = [
  0,
  0.0039685026299204982,
  0.0050000000000000001,
  0.0057235712127666604,
  0.0062996052494743663,
  0.0067860440414872674,
  0.0072112478515370419,
  0.0075914724296891559,
  0.0079370052598409981,
  0.0082548181222365652,
  0.008549879733383484,
  0.0088258708383151564,
  0.0090856029641606974,
  0.0093312778920431206,
  0.0095646559138619445,
  0.0097871691029221587,
  0.01,
  0.012771823873225884,
  0.014684780191517231,
  0.016198059006387416,
  0.017471609294725979,
  0.018582457992422498,
  0.019574338205844311,
  0.020474752446505033,
  0.021302255094664359,
  0.022070024812210518,
  0.022787798265876402,
  0.023462996943949387,
  0.024101422641752301,
  0.024707707022171385,
  0.025285613460831465,
  0.025838246261818115,
  0.026368199644884009,
  0.026877666342030315,
  0.027368518373992137,
  0.027842368212834204,
  0.028300615829251821,
  0.028744485394724156,
  0.02917505427704235,
  0.029593276209969318,
  0.029999999999999999,
];

const DIRECT_REGIONS: AutoZoneRegionTS[] = [
  { rEnd: 0.01, nz: 16, rhoRef: 1.0, isVoid: false, materialGroup: "" },
  { rEnd: 0.03, nz: 24, rhoRef: 0.05, isVoid: false, materialGroup: "" },
];

function computeDirectNodes(): number[] {
  const cfg = defaultAutoZoneConfig();
  cfg.geometryCode = 0;
  return computeAutoZoneNodes(0, DIRECT_REGIONS, cfg);
}

function twoRegionPolarForm() {
  const f = defaultFormState();
  const sMaxCm = 0.03;
  f.main.dimension = "2D_RZ";
  f.mesh.logicalMesh2d = "spherical_polar_halfplane";
  f.mesh.radialZoning2d = "regions";
  f.mesh.polarCenterTreatment = "tri_fan";
  f.mesh.rMax = q(sMaxCm, "cm");
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
    {
      materialName: "CH",
      rOuter: q(sMaxCm / 3, "cm"),
      rho: 1.0,
      Te: q(1, "eV"),
      Ti: q(1, "eV"),
    },
    {
      materialName: "outer",
      rOuter: q(sMaxCm, "cm"),
      rho: 0.05,
      Te: q(1, "eV"),
      Ti: q(1, "eV"),
    },
  ];
  return f;
}

describe("computeAutoZoneNodes", () => {
  it("reproduces the core-generated GOLDEN ladder", () => {
    const nodes = computeDirectNodes();
    expect(nodes).toHaveLength(GOLDEN.length);
    nodes.forEach((node, i) => {
      expect(Math.abs(node - GOLDEN[i])).toBeLessThanOrEqual(
        1.0e-12 * Math.max(1, Math.abs(GOLDEN[i])),
      );
    });
  });

  it("pins the material interface bit-exactly", () => {
    const nodes = computeDirectNodes();
    expect(nodes[16]).toBe(0.01);
  });

  it("is monotonic and preserves the endpoints", () => {
    const nodes = computeDirectNodes();
    expect(nodes[0]).toBe(0);
    expect(Math.abs(nodes[nodes.length - 1] - 0.03) / 0.03).toBeLessThanOrEqual(1.0e-15);
    for (let i = 1; i < nodes.length; ++i) {
      expect(nodes[i]).toBeGreaterThan(nodes[i - 1]);
    }
  });

  it("wires the polar regional ladder into the local mesh preview", () => {
    const f = twoRegionPolarForm();
    const segs = computeRegionSegments2d(f);
    expect(segs).not.toBeNull();
    const regions: AutoZoneRegionTS[] = segs!.map((segment) => ({
      rEnd: segment.rEndCm,
      nz: segment.nz,
      rhoRef: segment.rhoRefGcc,
      isVoid: false,
      materialGroup: "",
    }));
    const cfg = defaultAutoZoneConfig();
    cfg.geometryCode = 0;
    const expected = computeAutoZoneNodes(0, regions, cfg);
    const local = computeLocalMeshPreview(f);

    expect(local).not.toBeNull();
    expect(local!.rNodes).not.toBeNull();
    expect(local!.rNodes).toHaveLength(expected.length);
    local!.rNodes!.forEach((node, i) => {
      expect(node).toBe(expected[i]);
    });
    expect(local!.nr).toBe(local!.rNodes!.length - 1);
    expect(local!.rNodes![segs![0].nz]).toBe(0.01);
    expect(local!.rNodes![0]).toBe(0);
    expect(Math.abs(local!.rNodes![local!.rNodes!.length - 1] - 0.03) / 0.03).toBeLessThanOrEqual(
      1.0e-15,
    );
    for (let i = 1; i < local!.rNodes!.length; ++i) {
      expect(local!.rNodes![i]).toBeGreaterThan(local!.rNodes![i - 1]);
    }
  });
});
