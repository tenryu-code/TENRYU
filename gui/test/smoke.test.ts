import { describe, expect, it } from "vitest";
import { ja } from "../src/i18n/ja";

describe("i18n", () => {
  it("has app title", () => {
    expect(ja.app.title).toBe("TENRYU Studio");
  });
});
