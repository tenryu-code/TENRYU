import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

// Text shown in the UI comes from the message catalogues (src/i18n/ja.ts and
// en.ts). A Japanese literal written directly in a component stays Japanese
// when the English UI is selected: the remove buttons of the Materials,
// Geometry and Drive tabs did until 2026-09-27. Comments and the list bullet
// "・" are allowed.
const UI_DIR = path.resolve(__dirname, "..", "src");
const JAPANESE = /[\u3040-\u30ff\u3400-\u9fff]/;

function tsxFiles(dir: string): string[] {
  const files: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...tsxFiles(full));
    else if (entry.name.endsWith(".tsx")) files.push(full);
  }
  return files;
}

// Removes comments but keeps their newlines, so line numbers still match.
function stripComments(source: string): string {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, (comment) => comment.replace(/[^\n]/g, ""))
    .replace(/(^|[^:"'`])\/\/.*$/gm, "$1");
}

describe("UI components", () => {
  it("take Japanese text from the message catalogues, not from literals", () => {
    const files = tsxFiles(UI_DIR);
    expect(files.length).toBeGreaterThan(10);
    const offenders: string[] = [];
    for (const file of files) {
      const lines = stripComments(fs.readFileSync(file, "utf8")).split("\n");
      lines.forEach((line, index) => {
        if (JAPANESE.test(line.replace(/・/g, ""))) {
          offenders.push(`${path.relative(UI_DIR, file)}:${index + 1}: ${line.trim()}`);
        }
      });
    }
    expect(offenders, offenders.join("\n")).toEqual([]);
  });
});
