import { describe, it } from "vitest";
import * as presets from "../src/core/presets";
import { createImportTestSupport } from "./deckImportSupport";

const { checkForm } = createImportTestSupport("presets");

describe("headerless presets: binary-free equivalence", () => {
  for (const [name,build] of Object.entries(presets)) if (typeof build === "function") {
    it(name,()=>{ checkForm(name,build()); },120000);
  }
});
