import { describe, expect, it } from "vitest";
import { q, toCanonical } from "../src/core/units";

describe("units", () => {
  it("length to cm", () => {
    expect(toCanonical(q(500, "µm"), "length")).toBeCloseTo(0.05, 12);
    expect(toCanonical(q(2, "mm"), "length")).toBeCloseTo(0.2, 12);
    expect(toCanonical(q(0.06, "cm"), "length")).toBe(0.06);
  });
  it("time to s", () => {
    expect(toCanonical(q(2, "ns"), "time")).toBeCloseTo(2e-9, 18);
    expect(toCanonical(q(50, "ps"), "time")).toBeCloseTo(5e-11, 18);
  });
  it("temperature to eV", () => {
    expect(toCanonical(q(1.2, "keV"), "temperature")).toBeCloseTo(1200, 9);
  });
  it("power to W", () => {
    expect(toCanonical(q(1, "TW"), "power")).toBe(1e12);
  });
  it("unknown unit throws", () => {
    expect(() => toCanonical(q(1, "furlong"), "length")).toThrow();
  });
});
