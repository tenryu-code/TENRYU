import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { parseValidateOutput, stripLogPrefix } from "../src/core/validateParse";

const FIX = path.join(__dirname, "fixtures");
const okText = fs.readFileSync(path.join(FIX, "validate_ok.txt"), "utf8");
const failText = fs.readFileSync(path.join(FIX, "validate_fail.txt"), "utf8");

describe("stripLogPrefix", () => {
  it("removes spdlog prefix", () => {
    expect(
      stripLogPrefix("[2026-07-11 16:09:17.768] [info] [TENRYU] Configuration validated successfully."),
    ).toBe("Configuration validated successfully.");
  });
  it("keeps plain lines", () => {
    expect(stripLogPrefix("TENRYU ERROR [namelist]: x")).toBe("TENRYU ERROR [namelist]: x");
  });
});

describe("parseValidateOutput on real ok fixture", () => {
  const r = parseValidateOutput(okText, "", 0);
  it("ok", () => {
    expect(r.ok).toBe(true);
  });
  it("summary has 10 rows in order", () => {
    expect(r.summary.length).toBe(10);
    expect(r.summary[0].label).toBe("main");
    expect(r.summary[0].text).toContain("dimension=1D_SPH");
    expect(r.summary[1].label).toBe("mesh");
    expect(r.summary[9].label).toBe("output");
  });
  it("no errors", () => {
    expect(r.errors).toEqual([]);
  });
});

describe("parseValidateOutput on real fail fixture", () => {
  const r = parseValidateOutput("", failText, 1);
  it("not ok", () => {
    expect(r.ok).toBe(false);
  });
  it("captures TENRYU ERROR line", () => {
    expect(r.errors.length).toBe(1);
    expect(r.errors[0]).toContain("Mesh.nrr is not a supported key");
    expect(r.errors[0]).toContain("did you mean 'nr'?");
  });
  it("empty summary", () => {
    expect(r.summary).toEqual([]);
  });
});

describe("parseValidateOutput fallback when no TENRYU ERROR", () => {
  const r = parseValidateOutput("", "ssh: connect to host example-host port 22: Connection refused\n", 255);
  it("not ok and captures tail lines", () => {
    expect(r.ok).toBe(false);
    expect(r.errors.length).toBeGreaterThan(0);
    expect(r.errors[0]).toContain("Connection refused");
  });
});
