import { describe, expect, it } from "vitest";
import { BEAM_PRESETS, expandedPairCount, PAIR_CAP } from "../src/core/deck/beamPresets";

describe("beam presets", () => {
  const expectedPortCounts = {
    gxii: 12,
    omega: 60,
    nif: 48,
  } as const;

  for (const id of ["gxii", "omega", "nif"] as const) {
    it(`${id} has normalized contiguous ports`, () => {
      const preset = BEAM_PRESETS[id];
      expect(preset.ports).toHaveLength(expectedPortCounts[id]);
      expect(preset.ports.map((port) => port.portId)).toEqual(
        Array.from({ length: expectedPortCounts[id] }, (_, i) => i),
      );
      for (const port of preset.ports) {
        expect(Math.hypot(...port.dir)).toBeCloseTo(1, 12);
      }
      expect(
        preset.ports.reduce((sum, port) => sum + port.weight, 0),
      ).toBeCloseTo(1, 12);
    });

    it(`${id} is directionally balanced`, () => {
      const sum = BEAM_PRESETS[id].ports.reduce(
        (acc, port) => [
          acc[0] + port.dir[0],
          acc[1] + port.dir[1],
          acc[2] + port.dir[2],
        ],
        [0, 0, 0],
      );
      expect(Math.abs(sum[0])).toBeLessThan(1e-9);
      expect(Math.abs(sum[1])).toBeLessThan(1e-9);
      expect(Math.abs(sum[2])).toBeLessThan(1e-9);
    });
  }

  it("omega has 60 distinct directions separated by more than 10 degrees", () => {
    const directions = BEAM_PRESETS.omega.ports.map((port) => port.dir);
    const unique = new Set(
      directions.map((dir) => dir.map((component) => component.toPrecision(12)).join(",")),
    );
    expect(unique.size).toBe(60);

    let minimumAngleDeg = 180;
    for (let i = 0; i < directions.length; i += 1) {
      for (let j = i + 1; j < directions.length; j += 1) {
        const dot = directions[i].reduce(
          (sum, component, axis) => sum + component * directions[j][axis],
          0,
        );
        const angleDeg = Math.acos(Math.max(-1, Math.min(1, dot))) * 180 / Math.PI;
        minimumAngleDeg = Math.min(minimumAngleDeg, angleDeg);
      }
    }
    expect(minimumAngleDeg).toBeGreaterThan(10);
  });
});

describe("expanded pair count", () => {
  it("enforces the v1 pair cap at preset recommendations", () => {
    expect(expandedPairCount(12, 4)).toBe(4560);
    expect(expandedPairCount(12, 4)).toBeLessThanOrEqual(PAIR_CAP);
    expect(expandedPairCount(60, 2)).toBe(28680);
    expect(expandedPairCount(60, 2)).toBeLessThanOrEqual(PAIR_CAP);
    expect(expandedPairCount(48, 2)).toBe(18336);
    expect(expandedPairCount(48, 2)).toBeLessThanOrEqual(PAIR_CAP);
    expect(expandedPairCount(60, 4)).toBe(114960);
    expect(expandedPairCount(60, 4)).toBeGreaterThan(PAIR_CAP);
  });
});
