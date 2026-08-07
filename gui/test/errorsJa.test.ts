import { describe, expect, it } from "vitest";
import { translateError } from "../src/core/errorsJa";

describe("translateError", () => {
  it("translates typo-suggestion errors with the suggested key", () => {
    const tr = translateError(
      "TENRYU ERROR [namelist]: Python execution failed: RuntimeError: Mesh.nrr is not a supported key (did you mean 'nr'?)",
    );
    expect(tr.ja).toContain("もしかして 'nr'");
    expect(tr.raw).toContain("Mesh.nrr");
  });
  it("translates ssh permission failure", () => {
    const tr = translateError("alice@example-host: Permission denied (publickey).");
    expect(tr.ja).toContain("ssh-agent");
  });
  it("translates connection refused", () => {
    const tr = translateError("ssh: connect to host example-host port 22: Connection refused");
    expect(tr.ja).toContain("拒否");
  });
  it("passes through unknown errors with ja=null", () => {
    const tr = translateError("some unknown failure");
    expect(tr.ja).toBeNull();
    expect(tr.raw).toBe("some unknown failure");
  });
});
