import { describe, it } from "vitest";
import { buildCases } from "./fixtures/deckGoldenForms";
import { createImportTestSupport } from "./deckImportSupport";

const { checkForm } = createImportTestSupport("golden");

describe("headerless golden: binary-free equivalence", () => {
  for (const c of buildCases()) it(`golden/${c.name}`,()=>{checkForm(`golden_${c.name}`,c.form);},120000);
});
