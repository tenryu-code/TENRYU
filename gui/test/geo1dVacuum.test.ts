import { describe, expect, it } from "vitest";
import { defaultFormState } from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";
import { q } from "../src/core/units";

describe("1D vacuum exterior", () => {
  it("keeps VOID disabled by default and enables full-step retry", () => {
    const deck = generateDeck(defaultFormState());

    expect(deck).toContain("driver_full_step_retry_enabled=True");
    expect(deck).not.toContain("VOID");
  });

  it("emits a void material and density tail outside the last region", () => {
    const f = defaultFormState();
    f.geometry.vacuumOutside1d = true;
    f.geometry.regions[0].rOuter = q(250, "µm");
    f.mesh.rMax = q(500, "µm");

    const deck = generateDeck(f);

    expect(deck).toContain('Material(name="VOID"');
    expect(deck).toContain("void_config=dict(rho=1.0e-10");
    expect(deck).toContain("gui_vf_VOID");
    expect(deck).toContain("return 1.0e-10");
  });

  it("emits the default corona density ramp", () => {
    const f = defaultFormState();
    f.geometry.vacuumOutside1d = true;
    f.geometry.coronaRamp1d.enabled = true;
    f.geometry.regions[0].rOuter = q(250, "µm");
    f.mesh.rMax = q(500, "µm");

    const deck = generateDeck(f);

    expect(deck).toContain("math.exp(-(");
    expect(deck).toContain("0.0003");
  });

  it("rejects vacuum without exterior room and corona without vacuum", () => {
    const noRoom = defaultFormState();
    noRoom.geometry.vacuumOutside1d = true;
    expect(() => generateDeck(noRoom)).toThrow();

    const noVacuum = defaultFormState();
    noVacuum.geometry.coronaRamp1d.enabled = true;
    expect(() => generateDeck(noVacuum)).toThrow();
  });
});
