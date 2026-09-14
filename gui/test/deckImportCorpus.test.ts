import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { createImportTestSupport, root } from "./deckImportSupport";

const enabled = process.env.TENRYU_IMPORT_CORPUS === "1";
if (!enabled) console.warn("[deck import] examples corpus SKIPPED; set TENRYU_IMPORT_CORPUS=1 to run every examples/ deck");
const support = enabled ? createImportTestSupport("corpus") : null;

function pythonFiles(directory: string): string[] {
  return fs.readdirSync(directory,{withFileTypes:true}).flatMap(entry=>{
    const p = path.join(directory,entry.name);
    return entry.isDirectory()?pythonFiles(p):entry.name.endsWith(".py")?[p]:[];
  }).sort();
}
describe.skipIf(!enabled)("examples corpus (TENRYU_IMPORT_CORPUS=1)",()=>{
  for (const filename of pythonFiles(path.join(root,"examples"))) {
    const name = path.relative(root,filename);
    it(name,()=>{
      const { evaluate, imported, results } = support!;
      const source = fs.readFileSync(filename,"utf8");
      const record = evaluate(source,filename);
      if (!record.ok) {
        results.push({deck:name,result:"evaluation-failure",reason:record.error});
        // Evaluation failures are explicitly reported; they are not equivalence
        // successes and may include helper scripts rather than namelist decks.
        const helpers: Record<string,string> = {
          "examples/verification/compare_ale_identity_diag.py":"SystemExit: 2",
          "examples/verification/i1b_polar_common.py":"No Main block was executed",
        };
        expect(helpers[name],record.error).toBeTruthy();
        expect(record.error).toContain(helpers[name]);
        return;
      }
      const form = imported(source,filename);
      results.push({deck:name,result:"equivalent",rules:form.deckImport!.rules,notes:form.deckImport!.notes});
    },120000);
  }
});
