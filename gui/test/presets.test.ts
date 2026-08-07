import { describe, expect, it } from "vitest";
import { validateFormState } from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";
import { extractGuiState } from "../src/core/deck/roundtrip";
import { presetIndirectTr, presetLaserSphere, presetSlabRadiation } from "../src/core/presets";

describe("presets", () => {
  it("all presets validate clean and generate", () => {
    for (const build of [presetSlabRadiation, presetLaserSphere, presetIndirectTr]) {
      const f = build();
      expect(validateFormState(f)).toEqual([]);
      expect(generateDeck(f).length).toBeGreaterThan(100);
    }
  });

  it("indirect preset emits piecewise Tr callable and round-trips", () => {
    const f = presetIndirectTr();
    const deck = generateDeck(f);
    expect(deck).toContain("def _gui_pwl(x, xs, ys):");
    expect(deck).toContain("def gui_marshak_tr(t_s):");
    expect(deck).toContain("marshak_Tr=gui_marshak_tr");
    expect(deck).not.toContain("marshak_Tr_eV");
    const r = extractGuiState(deck);
    expect(r.ok && r.state.radiation.marshakPoints.length === 4).toBe(true);
  });

  it("laser table waveform emits a piecewise-linear callable in W and s", () => {
    const f = presetLaserSphere();
    f.laser.waveformMode = "table";
    f.laser.waveformPoints = [
      { t: 0, v: 0.5 },
      { t: 1, v: 1.0 },
      { t: 2, v: 0.0 },
    ];
    const deck = generateDeck(f);
    expect(deck).toContain("def _gui_pwl(x, xs, ys):");
    expect(deck).toContain("def gui_beam_power(t_s):");
    expect(deck).toContain("    if t_s < 0 or t_s > 2e-9: return 0.0");
    expect(deck).toContain(
      "    return _gui_pwl(t_s, [0, 1e-9, 2e-9], [500000000000, 1000000000000, 0])  # editor: t [ns] / P [TW]",
    );
    expect(deck).toContain("            power=gui_beam_power,");
  });

  it("waveform validation rejects non-monotonic t", () => {
    const f = presetIndirectTr();
    f.radiation.marshakPoints = [
      { t: 0, v: 100 },
      { t: 2, v: 150 },
      { t: 1, v: 200 },
    ];
    expect(validateFormState(f).some((e) => e.includes("単調増加"))).toBe(true);
  });
});
