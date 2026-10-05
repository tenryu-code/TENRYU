// Studio's copy of the solver's 1D zoning (src/core/deck/zoningIntent.ts) against the C++
// (src/core/zoning_intent.cpp): fixed intents run through compute_zoning_intent_nodes by
// scripts/zoningIntentGolden.cpp (fixtures/zoningIntentGolden.json), and the meshes of the laser
// presets and of the edited forms of zoningIntentForms.ts from the solver's validate --mesh-preview
// (fixtures/presetZoningNodes.json, scripts/presetZoningNodes.py).
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { computeZoningIntentNodes, formatDouble, type ZoningIntentConfig } from "../src/core/deck/zoningIntent";
import { computeMeshNodes1d, zoningIntentNodes1d, zoningPins } from "../src/core/deck/mesh1d";
import { PRESETS_1D } from "../src/core/presets1d";
import type { FormState } from "../src/core/deck/formState";
import { editedIntentForms } from "./zoningIntentForms";

interface GoldenDensity {
  kind: "none" | "regions" | "exponential";
  regions?: Array<{ rEnd: number; rho: number }>;
  rhoInner?: number;
  rStart?: number;
  scale?: number;
  rhoFloor?: number;
}

interface GoldenCase {
  name: string;
  rMin: number;
  rMax: number;
  density: GoldenDensity;
  cfg: ZoningIntentConfig;
  result: {
    ok: boolean;
    status: string;
    code: string;
    message: string;
    warnings: string[];
    cellsPerSegment: number[];
    ratioMaxAchieved: number;
    nRatioSoftExceed: number;
    widthMinAchieved: number;
    cellMeasureMinAchieved: number;
    cellMeasureMaxAchieved: number;
    nodes: number[];
  };
}

const GOLDEN: { cases: GoldenCase[] } = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, "fixtures", "zoningIntentGolden.json"), "utf8"),
);

/** The density of a case, as scripts/zoningIntentGolden.cpp makes it. */
function rho0Of(density: GoldenDensity): ((r: number) => number) | null {
  if (density.kind === "regions") {
    const regions = density.regions as Array<{ rEnd: number; rho: number }>;
    return (r) => {
      let index = regions.findIndex((region) => r < region.rEnd);
      if (index < 0) index = regions.length - 1;
      return regions[index].rho;
    };
  }
  if (density.kind === "exponential") {
    const { rhoInner, rStart, scale, rhoFloor } = density as Required<GoldenDensity>;
    return (r) => (r < rStart ? rhoInner : Math.max(rhoInner * Math.exp(-(r - rStart) / scale), rhoFloor));
  }
  return null;
}

function relative(a: number, b: number, floor: number): number {
  return Math.abs(a - b) / Math.max(Math.abs(b), floor);
}

describe("zoning intent: Studio's copy against the C++", () => {
  for (const c of GOLDEN.cases) {
    it(c.name, () => {
      const result = computeZoningIntentNodes(c.rMin, c.rMax, c.cfg, rho0Of(c.density));
      const expected = c.result;
      expect(result.ok).toBe(expected.ok);
      expect(result.diag.status).toBe(expected.status);
      expect(result.diag.code).toBe(expected.code);
      expect(result.diag.warnings).toEqual(expected.warnings);
      if (!expected.ok) {
        expect(result.diag.message.length).toBeGreaterThan(0);
        return;
      }
      expect(result.diag.cellsPerSegment).toEqual(expected.cellsPerSegment);
      expect(result.diag.nRatioSoftExceed).toBe(expected.nRatioSoftExceed);
      expect(result.nodes.length).toBe(expected.nodes.length);
      const span = c.rMax - c.rMin;
      let worstNode = 0;
      let worstWidth = 0;
      for (let i = 0; i < expected.nodes.length; i++) {
        worstNode = Math.max(worstNode, Math.abs(result.nodes[i] - expected.nodes[i]) / span);
        if (i > 0) {
          const width = result.nodes[i] - result.nodes[i - 1];
          const expectedWidth = expected.nodes[i] - expected.nodes[i - 1];
          worstWidth = Math.max(worstWidth, relative(width, expectedWidth, 0));
        }
      }
      // Pins and the domain ends are exact; the rest agree to round-off (Math.exp/log against the C
      // library: at most 1.2e-16 of the domain in a node, 4.4e-14 in a relative width). The bounds
      // are tight enough to catch a changed quadrature: the bin tolerance at 1e-9 instead of 1e-10
      // moves nodes by 2e-15 to 1e-14 of the domain and widths by 3e-13 to 4e-12.
      expect(result.nodes[0]).toBe(c.rMin);
      expect(result.nodes[result.nodes.length - 1]).toBe(c.rMax);
      for (const pin of c.cfg.pins) expect(result.nodes).toContain(pin.r);
      expect(worstNode, "largest node difference / domain length").toBeLessThan(1e-15);
      expect(worstWidth, "largest relative cell-width difference").toBeLessThan(2e-13);
      expect(relative(result.diag.ratioMaxAchieved, expected.ratioMaxAchieved, 0)).toBeLessThan(1e-9);
      expect(relative(result.diag.widthMinAchieved, expected.widthMinAchieved, 0)).toBeLessThan(1e-9);
      expect(relative(result.diag.cellMeasureMinAchieved, expected.cellMeasureMinAchieved, 0)).toBeLessThan(1e-9);
      expect(relative(result.diag.cellMeasureMaxAchieved, expected.cellMeasureMaxAchieved, 0)).toBeLessThan(1e-9);
    });
  }

  it("covers every measure, a success and a failure of each kind", () => {
    const measures = new Set(GOLDEN.cases.filter((c) => c.result.ok).map((c) => c.cfg.measure));
    expect([...measures].sort()).toEqual(["areal_mass", "cylindrical_line_mass", "spherical_cell_mass", "width"]);
    const statuses = new Set(GOLDEN.cases.map((c) => c.result.status));
    expect([...statuses].sort()).toEqual(["infeasible", "invalid_input", "numerical_failure", "ok"]);
  });

  it("is a copy of the solver's current zoning", () => {
    // After a change to the C++, bring zoningIntent.ts in step, rewrite both fixtures, then this.
    const digest = crypto.createHash("sha256");
    for (const file of ["zoning_intent.cpp", "zoning_intent.hpp"]) {
      digest.update(fs.readFileSync(path.resolve(__dirname, "..", "..", "src", "core", file), "utf8").replace(/\r\n/g, "\n"));
    }
    expect(digest.digest("hex")).toBe("7b98fc98f4c89a9404a4cbea3b82df31567dd7c8743d70c07d570696d8645f40");
  });

  it("formats numbers as printf %.17g does (the solver's messages)", () => {
    expect(formatDouble(1)).toBe("1");
    expect(formatDouble(0.1)).toBe("0.10000000000000001");
    expect(formatDouble(1.5209078354558918e-5)).toBe("1.5209078354558918e-05");
    expect(formatDouble(5.6578927004212906e-49)).toBe("5.6578927004212906e-49");
    expect(formatDouble(0.99999999999900002)).toBe("0.99999999999900002");
    expect(formatDouble(1e21)).toBe("1e+21");
    expect(formatDouble(-0.0001)).toBe("-0.0001");
  });
});

interface PresetNodes {
  nCells: number;
  stride: number;
  nodes: number[];
  last: number;
}

const PRESET_NODES: { presets: Record<string, PresetNodes> } = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, "fixtures", "presetZoningNodes.json"), "utf8"),
);

/** Studio's nodes of a form against the stored solver mesh (every stride-th node and the last). */
function expectSolverMesh(f: FormState, expected: PresetNodes): void {
  const result = zoningIntentNodes1d(f);
  if (!("nodes" in result)) throw new Error(result.error);
  const nodes = result.nodes;
  expect(computeMeshNodes1d(f)).toEqual(nodes);
  // A changed preset, recommendation or form fails here: rewrite the fixture with
  // scripts/presetZoningNodes.py from the solver's preview of the new deck.
  expect(nodes.length - 1).toBe(expected.nCells);
  expect(nodes[nodes.length - 1]).toBe(expected.last);
  const span = expected.last - expected.nodes[0];
  expected.nodes.forEach((node, k) => {
    expect(Math.abs(nodes[k * expected.stride] - node) / span, `node ${k * expected.stride}`).toBeLessThan(1e-15);
  });
  for (const pin of zoningPins(f, f.mesh.zoningIntent)) expect(nodes).toContain(pin.rCm);
}

describe("zoning intent: Studio's zoning of a form against the solver's mesh preview", () => {
  const zoned = PRESETS_1D.filter((preset) => preset.build().mesh.grid1d === "recommended");
  const edited = editedIntentForms();

  it("has the solver's mesh of every preset with a recommended mesh and of every edited form", () => {
    expect([...zoned.map((preset) => preset.id), ...Object.keys(edited)].sort()).toEqual(
      Object.keys(PRESET_NODES.presets).sort(),
    );
  });

  for (const preset of zoned) {
    it(`preset ${preset.id}`, () => expectSolverMesh(preset.build(), PRESET_NODES.presets[preset.id]));
  }
  // The bounds left empty take the solver's defaults; the density and the pins come from the layers.
  for (const [id, f] of Object.entries(edited)) {
    it(`edited form ${id}`, () => expectSolverMesh(f, PRESET_NODES.presets[id]));
  }
});
