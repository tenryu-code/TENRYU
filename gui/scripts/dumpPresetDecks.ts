// Writes the decks of Studio's 1D presets for checks outside Studio (vite-node):
//   <out>/decks/<id>.py           the deck Studio generates for the preset
//   <out>/recommend/<id>.py       laser presets: the deck with a uniform placeholder mesh, the input of
//                                 tools/assist recommend-mesh --deck
//   <out>/recommend/<id>.key      the conditions key the recommendation is made for
//   <out>/fine/<id>.py            the finer member of the preset's convergence pair
// Usage: npx vite-node scripts/dumpPresetDecks.ts <out>
import fs from "node:fs";
import path from "node:path";
import { generateDeck } from "../src/core/deck/generate";
import { recommendationConditionsKey } from "../src/core/deck/mesh1d";
import { placeholderForRecommendation, refinedForConvergence } from "../src/core/deck/meshRecommend";
import { PRESETS_1D } from "../src/core/presets1d";

const out = process.argv[2];
if (!out) throw new Error("usage: vite-node scripts/dumpPresetDecks.ts <out>");
for (const dir of ["decks", "recommend", "fine"]) fs.mkdirSync(path.join(out, dir), { recursive: true });
for (const preset of PRESETS_1D) {
  if (preset.id === "blank") continue;
  const form = preset.build();
  try {
    fs.writeFileSync(path.join(out, "decks", `${preset.id}.py`), generateDeck(form));
    fs.writeFileSync(path.join(out, "fine", `${preset.id}.py`), generateDeck(refinedForConvergence(form)));
  } catch (err) {
    // A preset whose stored recommendation no longer matches it has no valid deck until
    // make_preset_meshes.sh stores a new one; its recommend-mesh input is written below.
    console.log(`${preset.id}: no deck (${String(err).split("\n").slice(-1)[0]})`);
  }
  if (form.laser.enabled) {
    fs.writeFileSync(path.join(out, "recommend", `${preset.id}.py`), generateDeck(placeholderForRecommendation(form)));
    fs.writeFileSync(path.join(out, "recommend", `${preset.id}.key`), recommendationConditionsKey(form));
  }
}
console.log(`wrote the preset decks under ${out}`);
