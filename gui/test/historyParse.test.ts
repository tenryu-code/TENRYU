import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { HISTORY_DEFAULT_KEYS, parseHistory } from "@tenryu-common/core/results/historyParse";

const FIXTURE = path.resolve(__dirname, "fixtures", "history_small.h5");

describe("parseHistory", () => {
  it("reads the pinned chart set from the fixture", async () => {
    const bytes = new Uint8Array(fs.readFileSync(FIXTURE));
    const h = await parseHistory(bytes);
    expect(h.t.length).toBe(50);
    expect(h.missing).toEqual([]);
    for (const key of HISTORY_DEFAULT_KEYS) {
      expect(h.series[key]).toBeDefined();
      expect(h.series[key].length).toBe(50);
    }
    // radiation_field = 1e10 * (1 - exp(-t/1e-11)); last sample t = 5e-11
    const rad = h.series["energy/radiation_field"];
    expect(rad[49] / 1e10).toBeCloseTo(1 - Math.exp(-5), 3);
    expect(h.series["energy/E_total"][0]).toBeCloseTo(9.21e10, -6);
  });

  it("reports missing keys without failing", async () => {
    const bytes = new Uint8Array(fs.readFileSync(FIXTURE));
    const h = await parseHistory(bytes, ["energy/radiation_field", "no/such/dataset"]);
    expect(h.series["energy/radiation_field"]).toBeDefined();
    expect(h.missing).toEqual(["no/such/dataset"]);
  });
});
