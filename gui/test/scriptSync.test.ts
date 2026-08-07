import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

describe("run_detached.sh single-source sync", () => {
  it("ops/gui/run_detached.sh === gui/src/core/assets/run_detached.sh (byte-identical)", () => {
    const opsPath = path.resolve(__dirname, "..", "..", "ops", "gui", "run_detached.sh");
    const assetPath = path.resolve(__dirname, "..", "src", "core", "assets", "run_detached.sh");
    const ops = fs.readFileSync(opsPath, "utf8");
    const asset = fs.readFileSync(assetPath, "utf8");
    expect(asset).toBe(ops);
  });
});
