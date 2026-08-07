import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { parseProfile } from "@tenryu-common/core/results/profileParse";

const FIXTURE = path.resolve(__dirname, "fixtures", "profile_small.h5");

describe("parseProfile", () => {
  it("reads centers, time and default fields", async () => {
    const bytes = new Uint8Array(fs.readFileSync(FIXTURE));
    const p = await parseProfile(bytes);
    expect(p.t).toBeCloseTo(1.5e-9, 15);
    expect(p.xUm.length).toBe(40);
    // centers of linspace(0, 0.05 cm, 41): first center = 0.000625 cm = 6.25 µm
    expect(p.xUm[0]).toBeCloseTo(6.25, 6);
    expect(p.missing).toEqual([]);
    expect(p.fields["hydro/Ti"][0]).toBeCloseTo(50.0, 9);
  });

  it("reports absent fields", async () => {
    const bytes = new Uint8Array(fs.readFileSync(FIXTURE));
    const p = await parseProfile(bytes, ["hydro/rho", "no/such"]);
    expect(p.fields["hydro/rho"]).toBeDefined();
    expect(p.missing).toEqual(["no/such"]);
  });
});
