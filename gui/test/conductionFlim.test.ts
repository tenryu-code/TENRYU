import { describe, expect, it } from "vitest";
import { conductionFLimDefault, defaultFormState, validateFormState } from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";
import { t } from "../src/i18n";

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

describe("ion heat conduction", () => {
  it("emits the ion flux limiter with ion conduction and validates in 1D 2T", () => {
    const f = defaultFormState();
    f.conduction.ionConduction = true;
    expect(validateFormState(f)).toEqual([]);
    expect(generateDeck(f)).toContain("ion_conduction=True, ion_f_lim=1)");
    f.conduction.ionFLim = 0.2;
    expect(generateDeck(f)).toContain("ion_f_lim=0.2)");
    expect(generateDeck(defaultFormState())).not.toContain("ion_f_lim");
  });

  it("rejects the combinations the solver refuses", () => {
    const v = t().validation;
    const twoD = defaultFormState();
    twoD.conduction.ionConduction = true;
    twoD.main.dimension = "2D_RZ";
    expect(validateFormState(twoD)).toContain(v.ionCond1dOnly);
    const oneT = defaultFormState();
    oneT.conduction.ionConduction = true;
    oneT.main.temperatureModel = "1T";
    expect(validateFormState(oneT)).toContain(v.ionCondNeeds2T);
    for (const bad of [0, -1, Number.POSITIVE_INFINITY]) {
      const g = defaultFormState();
      g.conduction.ionConduction = true;
      g.conduction.ionFLim = bad;
      expect(validateFormState(g)).toContain(v.ionFLimPositive);
    }
    // With ion conduction off, or conduction disabled, the solver does not read these.
    const off = defaultFormState();
    off.main.dimension = "2D_RZ";
    expect(validateFormState(off)).not.toContain(v.ionCond1dOnly);
    const disabled = defaultFormState();
    disabled.conduction.enabled = false;
    disabled.conduction.ionConduction = true;
    disabled.main.temperatureModel = "1T";
    expect(validateFormState(disabled)).not.toContain(v.ionCondNeeds2T);
  });
});
