import { describe, expect, it, vi } from "vitest";
import { defaultFormState } from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";

// Exercise CBET serialization of legacy saved state independently of form validation.
vi.mock("../src/core/deck/formState", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/core/deck/formState")>();
  return { ...actual, validateFormState: () => [] };
});

describe("CBET deck generation", () => {
  it("emits the GXII port_section configuration", () => {
    const form = defaultFormState();
    form.laser.enabled = true;
    form.laser.mode = "raytrace_2d";
    form.laser.cbet.enabled = true;
    form.laser.cbet.detuneSplitNm = 0.5;
    form.laser.beams[0].powerFraction = 0.25;
    form.laser.beams[0].deltaLambdaNm = 0.3;
    form.laser.beams = [
      form.laser.beams[0],
      { ...form.laser.beams[0], name: "beam_01" },
    ];

    const deck = generateDeck(form);
    const cbetStart = deck.indexOf("cbet=dict(");
    const cbetEnd = deck.indexOf("\n    ),", cbetStart);
    const cbetBlock = deck.slice(cbetStart, cbetEnd + "\n    ),".length);
    const beamsStart = deck.indexOf("    beams=[");
    const beamsEnd = deck.indexOf("\n    ],", beamsStart);
    const beamsBlock = deck.slice(beamsStart, beamsEnd + "\n    ],".length);
    expect(deck).toContain('geometry_mode="port_section"');
    expect(cbetBlock).toContain("n_section_phi=8");
    expect(deck).toContain("port_configuration");
    expect(deck).toContain('normalization="sum_weights_one"');
    expect(deck.match(/port_id=/g) ?? []).toHaveLength(12);
    expect(cbetBlock).not.toMatch(/(^|[\s,(])n_phi=/m);
    expect(beamsBlock.match(/LaserBeam\(/g) ?? []).toHaveLength(1);
    expect(beamsBlock).not.toContain("delta_lambda_nm=");
    expect(beamsBlock).toContain("power=gui_beam_power,");
    expect(deck).toContain("delta_lambda_nm=+0.25");
  });

  it("emits alternating detuning and the OMEGA port count", () => {
    const form = defaultFormState();
    form.laser.enabled = true;
    form.laser.mode = "raytrace_2d";
    form.laser.cbet.enabled = true;
    form.laser.cbet.detuneSplitNm = 0.5;

    const detunedDeck = generateDeck(form);
    expect(detunedDeck).toContain("delta_lambda_nm=+0.25");
    expect(detunedDeck).toContain("delta_lambda_nm=-0.25");

    form.laser.cbet.portPreset = "omega";
    form.laser.cbet.nImpactBins = 2;
    const omegaDeck = generateDeck(form);
    expect(omegaDeck.match(/port_id=/g) ?? []).toHaveLength(60);
  });
});
