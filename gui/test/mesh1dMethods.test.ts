import { describe, expect, it } from "vitest";
import { defaultFormState, migrateFormState, validateFormState, type FormState } from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";
import {
  cellMetrics,
  computeLayerNodes,
  computeMeshNodes1d,
  defaultLayerZoning,
  deckRound,
  meshLayers1d,
  nodesFromFrozenConfig,
  parseNodeText,
  recommendationConditionsKey,
  shellVolume,
  solveRatio,
  summarizeMesh,
} from "../src/core/deck/mesh1d";
import { pyExact } from "../src/core/deck/meshEmit";
import { applyRecommendation, parseRecommendation, refinedForConvergence } from "../src/core/deck/meshRecommend";
import { q } from "../src/core/units";

/** A spherical CH shell on a DT gas fill with vacuum and a corona ramp outside. */
function shellForm(): FormState {
  const f = defaultFormState();
  f.main.geometry1d = "spherical";
  f.mesh.rMin = q(0, "cm");
  f.mesh.rMax = q(300, "µm");
  f.materials = [
    { ...f.materials[0], name: "DT", A: 2.5, Z: 1 },
    { ...f.materials[0], name: "CH", A: 6.5, Z: 3.5 },
  ];
  f.geometry.regions = [
    { materialName: "DT", rOuter: q(230, "µm"), rho: 0.01, Te: q(1, "eV"), Ti: q(1, "eV") },
    { materialName: "CH", rOuter: q(250, "µm"), rho: 1.05, Te: q(1, "eV"), Ti: q(1, "eV") },
  ];
  f.geometry.vacuumOutside1d = true;
  f.geometry.coronaRamp1d = { enabled: true, scaleUm: 2, extentUm: 10, rho0: 0.05, rhoMin: 3e-4 };
  f.radiation.enabled = false;
  return f;
}

/** Numbers of the MESH_NODES list of a generated deck. */
function deckNodes(deck: string): number[] {
  const match = /MESH_NODES = \[\n([\s\S]*?)\n\]/.exec(deck);
  if (!match) throw new Error("no MESH_NODES");
  return match[1]
    .split(/[\s,]+/)
    .filter((token) => token.length > 0)
    .map(Number);
}

describe("1D layers", () => {
  it("lists the regions, the corona ramp and the void padding with the radii the deck writes", () => {
    const layers = meshLayers1d(shellForm());
    expect(layers).not.toBeNull();
    expect(layers!.map((layer) => layer.kind)).toEqual(["region", "region", "corona", "void"]);
    expect(layers![1].rLoCm).toBe(deckRound(230e-4));
    expect(layers![2].rLoCm).toBe(deckRound(250e-4));
    expect(layers![2].rHiCm).toBe(deckRound(250e-4 + 10e-4));
    expect(layers![3].rHiCm).toBe(deckRound(300e-4));
    expect(layers![2].rhoAt(250e-4)).toBeCloseTo(0.05, 12);
    expect(layers![2].rhoAt(259e-4)).toBeCloseTo(0.05 * Math.exp(-4.5), 12);
    expect(layers![2].rhoAt(299e-4)).toBe(3e-4);
  });

  it("builds equal-width, equal-mass and geometric layers with every interface on a node", () => {
    const f = shellForm();
    f.mesh.grid1d = "layers";
    f.mesh.layerZoning = [
      { ...defaultLayerZoning(), cells: 50, spacing: "mass" },
      { ...defaultLayerZoning(), cells: 60, spacing: "ratio", ratioSpec: "ratio", ratio: 1.05, fineSide: "outer" },
      { ...defaultLayerZoning(), cells: 20, spacing: "mass" },
      { ...defaultLayerZoning(), cells: 10, spacing: "width" },
    ];
    const layers = meshLayers1d(f)!;
    const result = computeLayerNodes(f, layers);
    expect(result.errors).toEqual([]);
    const edges = result.edges!;
    expect(edges.length).toBe(50 + 60 + 20 + 10 + 1);
    for (const layer of layers) expect(edges).toContain(layer.rHiCm);
    expect(edges[0]).toBe(0);
    // equal mass in the spherical DT fill
    const fill = result.perLayer[0]!;
    const masses = fill.slice(1).map((b, i) => shellVolume("spherical", fill[i], b));
    for (const m of masses) expect(m / masses[0]).toBeCloseTo(1, 10);
    // geometric widths in the shell: each cell is q times its outer neighbour
    const shell = result.perLayer[1]!;
    const widths = shell.slice(1).map((b, i) => b - shell[i]);
    for (let k = 0; k + 1 < widths.length; k++) expect(widths[k] / widths[k + 1]).toBeCloseTo(1.05, 10);
    // equal mass in the corona ramp (varying density)
    const corona = result.perLayer[2]!;
    const coronaLayer = layers[2];
    const coronaMasses = corona.slice(1).map((b, i) => {
      let m = 0;
      const n = 2000;
      for (let j = 0; j < n; j++) {
        const lo = corona[i] + ((b - corona[i]) * j) / n;
        const hi = corona[i] + ((b - corona[i]) * (j + 1)) / n;
        m += coronaLayer.rhoAt(0.5 * (lo + hi)) * shellVolume("spherical", lo, hi);
      }
      return m;
    });
    for (const m of coronaMasses) expect(m / coronaMasses[0]).toBeCloseTo(1, 3);
  });

  it("solves the ratio for a prescribed end cell and matches a neighbour's cell mass", () => {
    const f = shellForm();
    f.mesh.grid1d = "layers";
    f.mesh.layerZoning = [
      { ...defaultLayerZoning(), cells: 80, spacing: "ratio", ratioSpec: "match_mass", fineSide: "outer" },
      { ...defaultLayerZoning(), cells: 100, spacing: "ratio", ratioSpec: "first_width", firstWidthCm: 0.02e-4, fineSide: "both" },
      { ...defaultLayerZoning(), cells: 20, spacing: "ratio", ratioSpec: "match_mass", fineSide: "inner" },
      { ...defaultLayerZoning(), cells: 10, spacing: "width" },
    ];
    const layers = meshLayers1d(f)!;
    const result = computeLayerNodes(f, layers);
    expect(result.errors).toEqual([]);
    const shell = result.perLayer[1]!;
    const outerWidth = shell[shell.length - 1] - shell[shell.length - 2];
    expect(outerWidth / 0.02e-4).toBeCloseTo(1, 9);
    expect((shell[1] - shell[0]) / 0.02e-4).toBeCloseTo(1, 9);
    // The fill's outer cell has the mass of the shell's inner cell.
    const fill = result.perLayer[0]!;
    const fillOuter = 0.01 * shellVolume("spherical", fill[fill.length - 2], fill[fill.length - 1]);
    const shellInner = 1.05 * shellVolume("spherical", shell[0], shell[1]);
    expect(fillOuter / shellInner).toBeCloseTo(1, 8);
    // The corona's inner cell has the mass of the shell's outer cell.
    const corona = result.perLayer[2]!;
    const coronaLayer = layers[2];
    let coronaInner = 0;
    for (let j = 0; j < 4000; j++) {
      const lo = corona[0] + ((corona[1] - corona[0]) * j) / 4000;
      const hi = corona[0] + ((corona[1] - corona[0]) * (j + 1)) / 4000;
      coronaInner += coronaLayer.rhoAt(0.5 * (lo + hi)) * shellVolume("spherical", lo, hi);
    }
    const shellOuter = 1.05 * shellVolume("spherical", shell[shell.length - 2], shell[shell.length - 1]);
    expect(coronaInner / shellOuter).toBeCloseTo(1, 5);
  });

  it("refuses to match a neighbouring cell heavier than the whole layer", () => {
    const f = shellForm();
    f.mesh.grid1d = "layers";
    f.mesh.layerZoning = [
      { ...defaultLayerZoning(), cells: 80, spacing: "ratio", ratioSpec: "match_mass", fineSide: "outer" },
      { ...defaultLayerZoning(), cells: 100, spacing: "ratio", ratioSpec: "first_width", firstWidthCm: 0.02e-4, fineSide: "outer" },
      { ...defaultLayerZoning(), cells: 20 },
      { ...defaultLayerZoning(), cells: 10 },
    ];
    // The shell's widest (inner) cell, 1.05 g/cc x ~1.2 µm, outweighs the whole 0.01 g/cc fill.
    expect(computeLayerNodes(f).errors).toEqual([{ layer: 0, code: "matchTooHeavy" }]);
  });

  it("reports cycles, sides without a neighbour and a both-sided match", () => {
    const f = shellForm();
    f.mesh.grid1d = "layers";
    f.mesh.layerZoning = [
      { ...defaultLayerZoning(), cells: 10, spacing: "ratio", ratioSpec: "match_mass", fineSide: "inner" },
      { ...defaultLayerZoning(), cells: 10, spacing: "ratio", ratioSpec: "match_mass", fineSide: "inner" },
      { ...defaultLayerZoning(), cells: 10, spacing: "ratio", ratioSpec: "match_mass", fineSide: "outer" },
      { ...defaultLayerZoning(), cells: 10, spacing: "ratio", ratioSpec: "match_mass", fineSide: "both" },
    ];
    const codes = computeLayerNodes(f).errors.map((e) => `${e.layer}:${e.code}`).sort();
    expect(codes).toEqual(["0:matchNoNeighbour", "1:matchNoNeighbour", "2:matchNoNeighbour", "3:matchBothSides"]);
    f.mesh.layerZoning[0] = { ...defaultLayerZoning(), cells: 10 };
    f.mesh.layerZoning[1] = { ...defaultLayerZoning(), cells: 10, spacing: "ratio", ratioSpec: "match_mass", fineSide: "outer" };
    f.mesh.layerZoning[2] = { ...defaultLayerZoning(), cells: 10, spacing: "ratio", ratioSpec: "match_mass", fineSide: "inner" };
    f.mesh.layerZoning[3] = { ...defaultLayerZoning(), cells: 10 };
    expect(computeLayerNodes(f).errors.map((e) => e.code).sort()).toEqual(["matchCycle", "matchCycle"]);
  });

  it("solveRatio inverts the geometric sum", () => {
    const exponents = Array.from({ length: 30 }, (_, k) => k);
    const q0 = 1.07;
    const target = exponents.reduce((s, e) => s + Math.pow(q0, e), 0);
    expect(solveRatio(exponents, target)).toBeCloseTo(q0, 12);
    expect(solveRatio(exponents, 30)).toBe(1);
    expect(solveRatio(exponents, 0.5)).toBeNull();
    expect(solveRatio([0, 0], 3)).toBeNull();
  });

  it("writes the layer nodes as explicit_nodes that read back exactly", () => {
    const f = shellForm();
    f.mesh.grid1d = "layers";
    f.mesh.layerZoning = [
      { ...defaultLayerZoning(), cells: 40 },
      { ...defaultLayerZoning(), cells: 70, spacing: "ratio", ratioSpec: "ratio", ratio: 1.08 },
      { ...defaultLayerZoning(), cells: 20 },
      { ...defaultLayerZoning(), cells: 10, spacing: "width" },
    ];
    expect(validateFormState(f)).toEqual([]);
    const deck = generateDeck(f);
    expect(deck).toContain("explicit_nodes=MESH_NODES,");
    expect(deck).not.toMatch(/^\s+nr=/m);
    const nodes = deckNodes(deck);
    expect(nodes).toEqual(computeMeshNodes1d(f));
    expect(deck).toContain(`r_min=${pyExact(nodes[0])},`);
    expect(deck).toContain(`r_max=${pyExact(nodes[nodes.length - 1])},`);
    // Interfaces written by the geometry functions coincide with nodes.
    expect(deck).toContain("if r_cm < 0.023: return");
    expect(nodes).toContain(0.023);
    expect(nodes).toContain(0.025);
  });
});

describe("imported nodes", () => {
  it("parses one value per line, one line of values, and a column of a table", () => {
    expect(parseNodeText("0\n1\n2.5\n", 1e-4)).toEqual({ nodes: [0, 1e-4, 2.5e-4], columns: 1, rows: 3 });
    expect(parseNodeText("0, 1, 2", 1)).toEqual({ nodes: [0, 1, 2], columns: 1, rows: 1 });
    const table = parseNodeText("# zone r [cm] rho\nzone r rho\n0 0.0 1\n1 0.5 1\n2 1.0 1\n", 1, 1);
    expect(table).toEqual({ nodes: [0, 0.5, 1.0], columns: 3, rows: 3 });
    expect(parseNodeText("", 1)).toEqual({ error: "empty" });
    expect(parseNodeText("0 1\n2\n", 1, 1)).toEqual({ error: "column" });
  });

  it("reads the explicit nodes of a frozen configuration", () => {
    expect(nodesFromFrozenConfig({ mesh: { explicit_nodes: [0, 1, 2] } })).toEqual([0, 1, 2]);
    expect(nodesFromFrozenConfig({ mesh: { explicit_nodes: [] } })).toBeNull();
    expect(nodesFromFrozenConfig({ main: {} })).toBeNull();
  });

  it("requires the end nodes at r_min and r_max and warns about interfaces between nodes", () => {
    const f = shellForm();
    f.mesh.grid1d = "explicit";
    f.mesh.explicitNodes = { nodesCm: [0, 0.01, 0.0231, 0.025, 0.026, 0.03], source: "test" };
    expect(validateFormState(f)).toEqual([]);
    f.mesh.explicitNodes = { nodesCm: [0, 0.01, 0.029], source: "test" };
    expect(validateFormState(f).length).toBe(1);
  });
});

describe("cell metrics", () => {
  it("measures each cell with the density at its centre and the target's reference area", () => {
    const f = shellForm();
    f.mesh.grid1d = "uniform";
    f.mesh.nr = 300;
    const metrics = cellMetrics(f, computeMeshNodes1d(f)!)!;
    const summary = summarizeMesh(metrics);
    expect(summary.nCells).toBe(300);
    const area = 4 * Math.PI * 0.025 * 0.025;
    const i = metrics.materials.lastIndexOf("CH");
    expect(metrics.referenceArealMass[i]).toBeCloseTo(metrics.masses[i] / area, 18);
  });
});

/** A recommend-mesh JSON payload shaped like the tool's output (values from a real run). */
function recommendationJson(rMax: number): string {
  return JSON.stringify({
    schema: "tenryu.assist.mesh_recommendation.v1",
    recommendation: { surface_areal_mass_g_cm2: 6.359349436376733e-6, mode: "measured_case" },
    evidence: [{ id: "C28", distance: 0 }],
    flags: [],
    warnings: [],
    confidence: "measured case",
    validation: { status: "validated", attempts: [], achieved_surface_areal_mass_g_cm2: 6.1e-6 },
    mesh: {
      r_min: 0.0,
      r_max: rMax,
      geometry_1d: "spherical",
      zoning_intent: {
        n_cells: 222,
        measure: "spherical_cell_mass",
        density_regions: [
          { r_end: 0.023, rho: 0.01 },
          { r_end: 0.025, rho: 1.05 },
          { r_end: rMax, rho: 1e-9 },
        ],
        pins: [
          { r: 0.023, ratio_jump_allowed: true },
          { r: 0.025, ratio_jump_allowed: true },
        ],
        profile: [
          { r: 0.0, w: 4.74489029934624e-8 },
          { r: 0.025, w: 2.7923135083265143e-8 },
        ],
        bands: [{ measure_frac_begin: 0.0, measure_frac_end: 0.999999970265094, cell_measure_max: 4.74489029934624e-8 }],
        dr_min: 2.39641877934272e-7,
        preferred_ratio: 1.3,
        ratio_hard_max: 1.3,
        min_cells_per_segment: 40,
      },
      resolution_requirement: {
        apply: "enforce",
        empirical: {
          reference_sha256: "9877d1a8c61ead63c6b8a3822ccfc069d149bd849838da79eb5965edd62fd724",
          case_ids: ["C28"],
          surface_ceiling_g_cm2: 6.359349436376733e-6,
          reference_apriori_g_cm2: 7.260161321563045e-7,
        },
      },
    },
  });
}

function laserShellForm(): FormState {
  const f = shellForm();
  f.geometry.coronaRamp1d.enabled = false;
  f.laser.enabled = true;
  f.laser.ghostCorona.enabled = true;
  f.laser.waveformMode = "gaussian";
  f.conduction.solver = "implicit";
  return f;
}

describe("recommended mesh", () => {
  it("applies recommend-mesh's zoning_intent and writes every number exactly", () => {
    const f = laserShellForm();
    const rec = parseRecommendation(recommendationJson(0.03));
    applyRecommendation(f, rec, recommendationConditionsKey(f), "/srv/tenryu", "2026-10-01T00:00:00Z");
    expect(validateFormState(f)).toEqual([]);
    const deck = generateDeck(f);
    expect(deck).toContain("n_cells=222,");
    expect(deck).toContain('measure="spherical_cell_mass",');
    expect(deck).toContain("surface_ceiling_g_cm2=0.000006359349436376733,");
    expect(Number("0.000006359349436376733")).toBe(6.359349436376733e-6);
    expect(deck).toContain("reference_apriori_g_cm2=7.260161321563045e-7,");
    expect(deck).toContain('{"measure_frac_begin": 0, "measure_frac_end": 0.999999970265094, "cell_measure_max": 4.74489029934624e-8}');
    expect(deck).toContain("dr_min=2.39641877934272e-7,");
    expect(deck).toContain('{"r": 0.023, "ratio_jump_allowed": True},');
    expect(deck).toContain('apply="enforce",');
    expect(deck).toContain('solver="implicit"');
    expect(deck).toContain("ghost_corona=dict(");
  });

  it("is stale once the target or the drive changes", () => {
    const f = laserShellForm();
    applyRecommendation(f, parseRecommendation(recommendationJson(0.03)), recommendationConditionsKey(f), "", "");
    const thicker = structuredClone(f);
    thicker.geometry.regions[1].rOuter = q(255, "µm");
    expect(validateFormState(thicker).some((e) => e.includes("推薦メッシュ"))).toBe(true);
    const brighter = structuredClone(f);
    brighter.laser.gaussianPeakW = q(2, "TW");
    expect(validateFormState(brighter).some((e) => e.includes("推薦メッシュ"))).toBe(true);
    const sameInMicrons = structuredClone(f);
    sameInMicrons.mesh.rMax = q(0.03, "cm");
    expect(validateFormState(sameInMicrons)).toEqual([]);
  });

  it("refuses a recommendation for another domain", () => {
    const f = laserShellForm();
    expect(() =>
      applyRecommendation(f, parseRecommendation(recommendationJson(0.04)), recommendationConditionsKey(f), "", ""),
    ).toThrow();
  });
});

describe("convergence pair", () => {
  it("doubles the cells of every method and keeps the progression's shape", () => {
    const f = shellForm();
    f.mesh.grid1d = "layers";
    f.mesh.layerZoning = [
      { ...defaultLayerZoning(), cells: 40 },
      { ...defaultLayerZoning(), cells: 50, spacing: "ratio", ratioSpec: "ratio", ratio: 1.1 },
      { ...defaultLayerZoning(), cells: 20 },
      { ...defaultLayerZoning(), cells: 10, spacing: "width" },
    ];
    const fine = refinedForConvergence(f);
    expect(fine.main.name).toBe(`${f.main.name}_fine`);
    expect(computeMeshNodes1d(fine)!.length - 1).toBe(2 * (computeMeshNodes1d(f)!.length - 1));
    expect(fine.mesh.layerZoning[1].ratio).toBeCloseTo(Math.sqrt(1.1), 14);

    const g = laserShellForm();
    applyRecommendation(g, parseRecommendation(recommendationJson(0.03)), recommendationConditionsKey(g), "", "");
    const gFine = refinedForConvergence(g);
    expect(gFine.mesh.grid1d).toBe("zoning_intent");
    expect(gFine.mesh.zoningIntent.nCells).toBe(444);
    expect(gFine.mesh.zoningIntent.bands[0].cellMeasureMax).toBe(4.74489029934624e-8 / 2);
    expect(gFine.mesh.resolutionRequirement.empirical).toEqual(g.mesh.resolutionRequirement.empirical);
    expect(validateFormState(gFine)).toEqual([]);

    const h = shellForm();
    h.mesh.grid1d = "explicit";
    h.mesh.explicitNodes = { nodesCm: [0, 0.01, 0.023, 0.025, 0.026, 0.03], source: "" };
    const hFine = refinedForConvergence(h);
    expect(hFine.mesh.explicitNodes.nodesCm.length).toBe(11);
    const n = hFine.mesh.explicitNodes.nodesCm;
    const m1 = shellVolume("spherical", n[0], n[1]);
    const m2 = shellVolume("spherical", n[1], n[2]);
    expect(m1 / m2).toBeCloseTo(1, 12);
  });
});

describe("migration and defaults", () => {
  it("fills the new fields of an old GUI state without changing its deck", () => {
    const f = defaultFormState();
    const old = structuredClone(f) as unknown as Record<string, Record<string, unknown>>;
    delete old.mesh.layerZoning;
    delete old.mesh.explicitNodes;
    delete old.mesh.zoningIntent;
    delete old.mesh.resolutionRequirement;
    delete old.mesh.recommendation;
    delete (old.laser as Record<string, unknown>).ghostCorona;
    delete (old.laser as Record<string, unknown>).depositSmoothPasses;
    delete (old.laser as Record<string, unknown>).depositSmoothAlpha;
    delete (old.conduction as Record<string, unknown>).solver;
    const migrated = migrateFormState(old as unknown as FormState);
    expect(migrated.mesh.grid1d).toBe("uniform");
    expect(migrated.conduction.solver).toBe("sts");
    expect(migrated.laser.ghostCorona.enabled).toBe(false);
    const strip = (deck: string) => deck.replace(/^# TENRYU-GUI-STATE: .*$/m, "");
    expect(strip(generateDeck(migrated))).toBe(strip(generateDeck(f)));
    expect(generateDeck(f)).not.toContain("lasermesh");
    expect(generateDeck(f)).not.toContain('solver="implicit"');
  });
});

describe("recommend-mesh input deck", () => {
  it("puts a node on every material interface of the placeholder mesh", async () => {
    const { placeholderForRecommendation } = await import("../src/core/deck/meshRecommend");
    const f = laserShellForm();
    f.geometry.regions[0].rOuter = q(243, "µm");
    const placeholder = placeholderForRecommendation(f);
    const nodes = computeMeshNodes1d(placeholder)!;
    expect(nodes).toContain(deckRound(243e-4));
    expect(nodes).toContain(deckRound(250e-4));
    expect(validateFormState(placeholder)).toEqual([]);
  });
});

describe("zoning density regions against the target", () => {
  it("warns when a recommendation's density regions miss a material interface", async () => {
    const { mesh1dWarnings } = await import("../src/core/deck/mesh1d");
    const f = laserShellForm();
    applyRecommendation(f, parseRecommendation(recommendationJson(0.03)), recommendationConditionsKey(f), "", "");
    expect(mesh1dWarnings(f).some((w) => w.includes("230"))).toBe(false);
    f.mesh.zoningIntent.densityRegions[0].rEndCm = 0.0232;
    expect(mesh1dWarnings(f).some((w) => w.includes("230"))).toBe(true);
  });
});
