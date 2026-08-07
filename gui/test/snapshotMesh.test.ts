import { describe, expect, it } from "vitest";
import { parseSnapshotMesh, validateCsr } from "@tenryu-common/core/results/snapshotMesh";

describe("parseSnapshotMesh", () => {
  it("returns null for an invalid HDF5 buffer", async () => {
    await expect(parseSnapshotMesh(new ArrayBuffer(8))).resolves.toBeNull();
  });
});

describe("validateCsr", () => {
  it("accepts a valid two-quad mesh", () => {
    const offsets = Int32Array.from([0, 4, 8]);
    const indices = Int32Array.from([0, 1, 4, 3, 1, 2, 5, 4]);

    expect(validateCsr(offsets, indices, 6)).toBe(true);
  });

  it("rejects an out-of-range node index", () => {
    const offsets = Int32Array.from([0, 4]);
    const indices = Int32Array.from([0, 1, 2, 4]);

    expect(validateCsr(offsets, indices, 4)).toBe(false);
  });

  it("rejects non-ascending offsets", () => {
    const offsets = Int32Array.from([0, 4, 3]);
    const indices = Int32Array.from([0, 1, 2]);

    expect(validateCsr(offsets, indices, 3)).toBe(false);
  });
});
