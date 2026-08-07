import { describe, expect, it } from "vitest";
import { conductionFLimDefault, defaultFormState } from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";

describe("conduction flux-limiter defaults", () => {
  it("uses a per-model default", () => {
    expect(conductionFLimDefault("none")).toBe(0.06);
    expect(conductionFLimDefault("snb")).toBe(0.5);
  });

  it("generates the SNB model with its flux-limiter default", () => {
    const form = defaultFormState();
    form.main.dimension = "1D_SPH";
    form.main.temperatureModel = "2T";
    form.conduction.enabled = true;
    form.conduction.nonlocalModel = "snb";
    form.conduction.fLim = conductionFLimDefault("snb");

    const deck = generateDeck(form);
    expect(deck).toContain("f_lim=0.5");
    expect(deck).toContain('nonlocal_model="snb"');
  });

  it("keeps the untouched local-conduction defaults", () => {
    const deck = generateDeck(defaultFormState());
    expect(deck).toContain("f_lim=0.06");
    expect(deck).not.toContain("nonlocal_model=");
  });
});
