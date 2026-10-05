import { describe, expect, it } from "vitest";
import { validateFormState } from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";
import { computeMeshNodes1d, meshLayers1d, recommendationConditionsKey } from "../src/core/deck/mesh1d";
import { extractGuiState } from "../src/core/deck/roundtrip";
import { PRESET_MESHES, type PresetMeshId } from "../src/core/presetMeshes";
import { PRESETS_1D, presetIndirectCapsule, presetLaserFoil } from "../src/core/presets1d";

describe("1D presets", () => {
  it("validate clean and generate a deck", () => {
    for (const preset of PRESETS_1D) {
      const f = preset.build();
      expect(validateFormState(f), preset.id).toEqual([]);
      expect(generateDeck(f).length, preset.id).toBeGreaterThan(100);
    }
  });

  it("have unique case names (output directories)", () => {
    const names = PRESETS_1D.map((p) => p.build().main.name);
    expect(new Set(names).size).toBe(names.length);
  });

  it("drive lasers with the example suite's conventions", () => {
    for (const preset of PRESETS_1D) {
      const f = preset.build();
      if (!f.laser.enabled) continue;
      const deck = generateDeck(f);
      expect(deck, preset.id).toContain("ghost_corona=dict(");
      // SNB runs on the STS conduction solve; the others use the implicit one.
      if (f.conduction.nonlocalModel === "snb") expect(deck, preset.id).not.toContain('solver="implicit"');
      else expect(deck, preset.id).toContain('solver="implicit"');
      expect(deck, preset.id).toContain("deposit=dict(deposit_smooth_passes=3, deposit_smooth_alpha=0.25)");
    }
  });

  it("carry a recommended mesh made for their own conditions (laser) or a per-layer table", () => {
    for (const preset of PRESETS_1D) {
      if (preset.id === "blank") continue;
      const f = preset.build();
      if (f.laser.enabled) {
        const stored = PRESET_MESHES[preset.id as PresetMeshId];
        expect(stored, `${preset.id}: no stored recommendation (scripts/make_preset_meshes.sh)`).toBeDefined();
        expect(f.mesh.grid1d, preset.id).toBe("recommended");
        expect(stored!.conditionsKey, `${preset.id}: the preset changed after its mesh was made`).toBe(
          recommendationConditionsKey(f),
        );
        expect(f.mesh.recommendation?.status, preset.id).toBe("validated");
      } else {
        expect(f.mesh.grid1d, preset.id).toBe("layers");
        const nodes = computeMeshNodes1d(f)!;
        for (const layer of meshLayers1d(f)!) expect(nodes, preset.id).toContain(layer.rHiCm);
      }
    }
  });

  it("indirect-drive capsule writes its Tr(t) table and round-trips", () => {
    const f = presetIndirectCapsule();
    const deck = generateDeck(f);
    expect(deck).toContain("def _gui_pwl(x, xs, ys):");
    expect(deck).toContain("def gui_marshak_tr(t_s):");
    expect(deck).toContain("marshak_Tr=gui_marshak_tr");
    expect(deck).not.toContain("marshak_Tr_eV");
    const r = extractGuiState(deck);
    expect(r.ok && r.state.radiation.marshakPoints.length === 6).toBe(true);
  });

  it("a laser table waveform emits a piecewise-linear callable in W and s", () => {
    const f = presetLaserFoil();
    // A recommended mesh belongs to its pulse: changing the waveform makes it stale.
    f.mesh.grid1d = "uniform";
    f.mesh.nr = 400;
    f.laser.waveformMode = "table";
    f.laser.waveformPoints = [
      { t: 0, v: 0.5 },
      { t: 1, v: 1.0 },
      { t: 2, v: 0.0 },
    ];
    const deck = generateDeck(f);
    expect(deck).toContain("def _gui_pwl(x, xs, ys):");
    expect(deck).toContain("    if t_s < 0 or t_s > 2e-9: return 0.0");
    expect(deck).toContain(
      "    return _gui_pwl(t_s, [0, 1e-9, 2e-9], [500000000000, 1000000000000, 0])  # editor: t [ns] / P [TW]",
    );
  });

  it("a recommended mesh becomes stale when the pulse changes", () => {
    const f = presetLaserFoil();
    expect(validateFormState(f)).toEqual([]);
    f.laser.powerW = { value: 200, unit: "TW" };
    expect(validateFormState(f).some((e) => e.includes("推薦メッシュ"))).toBe(true);
  });

  it("waveform validation rejects non-monotonic t", () => {
    const f = presetIndirectCapsule();
    f.radiation.marshakPoints = [
      { t: 0, v: 100 },
      { t: 2, v: 150 },
      { t: 1, v: 200 },
    ];
    expect(validateFormState(f).some((e) => e.includes("単調増加"))).toBe(true);
  });
});
