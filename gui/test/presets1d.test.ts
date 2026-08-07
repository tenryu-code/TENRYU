import { describe, expect, it } from "vitest";
import { validateFormState } from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";
import { presetIndirectTr, presetLaserSphere, presetSlabRadiation } from "../src/core/presets";

describe("1D presets", () => {
  it("laser capsule validates and emits the GXII direct-drive deck", () => {
    const f = presetLaserSphere();
    expect(validateFormState(f)).toEqual([]);

    const deck = generateDeck(f);
    expect(deck).toContain('Material(name="VOID"');
    expect(deck).toContain("math.exp(-(");
    expect(deck).toContain('mode="raytrace_2d"');
    expect(deck).toContain("driver_full_step_retry_enabled=True");
    expect(deck).toContain("groups=20");
    expect(deck).toContain("f_number=3");
  });

  it("slab radiation and indirect-drive presets still validate and generate", () => {
    for (const build of [presetSlabRadiation, presetIndirectTr]) {
      const f = build();
      expect(validateFormState(f)).toEqual([]);
      expect(() => generateDeck(f)).not.toThrow();
    }

    const slabDeck = generateDeck(presetSlabRadiation());
    expect(slabDeck).toContain("kappa_a=2000");
    expect(slabDeck).toContain("r_max=0.01");

    const indirectDeck = generateDeck(presetIndirectTr());
    expect(indirectDeck).toContain("kappa_a=2000");
    expect(indirectDeck).toContain("t_end=5e-9");
    expect(indirectDeck).toContain("5e-10");
  });
});
