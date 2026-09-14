import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { generateDeck } from "../src/core/deck/generate";
import { parseValidateOutput } from "../src/core/validateParse";
import { buildCases } from "./fixtures/deckGoldenForms";

const BIN = process.env.TENRYU_BIN && fs.existsSync(process.env.TENRYU_BIN) ? process.env.TENRYU_BIN : "";


describe.skipIf(!BIN)("golden gate: generated decks pass real `tenryu validate`", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "tenryu-golden-"));

  for (const c of buildCases()) {
    it(
      `${c.name} validates PASS`,
      () => {
        const deck = generateDeck(c.form);
        const dir = path.join(tmp, c.name);
        fs.mkdirSync(dir, { recursive: true });
        const deckPath = path.join(dir, "deck.py");
        fs.writeFileSync(deckPath, deck);
        const r = spawnSync(BIN, ["validate", deckPath], {
          encoding: "utf8",
          timeout: 120000,
          cwd: dir,
        });
        const parsed = parseValidateOutput(r.stdout ?? "", r.stderr ?? "", r.status);
        if (!parsed.ok) {
          console.error(`--- deck (${c.name}) ---\n${deck}\n--- stderr ---\n${r.stderr}`);
        }
        expect(parsed.ok).toBe(true);
        const byLabel = new Map<string, string>();
        for (const row of parsed.summary) {
          byLabel.set(row.label, (byLabel.get(row.label) ?? "") + row.text + " ");
        }
        for (const [label, substr] of c.expectSummary) {
          const text = byLabel.get(label) ?? "";
          expect(text, `summary[${label}] should contain "${substr}"`).toContain(substr);
        }
      },
      180000,
    );
  }
});

describe("golden gate presence", () => {
  it("warns when TENRYU_BIN is unset", () => {
    if (!BIN) {
      console.warn("[golden] TENRYU_BIN unset or missing — golden validate gate SKIPPED");
    }
    expect(true).toBe(true);
  });
});
