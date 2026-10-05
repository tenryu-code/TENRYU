// Writes src/core/presetMeshes.ts from recommend-mesh results (vite-node):
//   <dir>/<id>.json   recommend-mesh -o output for the deck <dir>/<id>.py
//   <dir>/<id>.key    the conditions key written by dumpPresetDecks.ts
// Usage: npx vite-node scripts/writePresetMeshes.ts <dir> <binary description>
import fs from "node:fs";
import path from "node:path";
import { parseRecommendation } from "../src/core/deck/meshRecommend";

const [dir, binary] = process.argv.slice(2);
if (!dir || !binary) throw new Error("usage: vite-node scripts/writePresetMeshes.ts <dir> <binary description>");
const ids = fs
  .readdirSync(dir)
  .filter((name) => name.endsWith(".json"))
  .map((name) => name.slice(0, -5))
  .sort();
const createdAt = new Date().toISOString().slice(0, 10);
const entries: string[] = [];
for (const id of ids) {
  const raw = JSON.parse(fs.readFileSync(path.join(dir, `${id}.json`), "utf8"));
  const parsed = parseRecommendation(JSON.stringify(raw));
  if (parsed.status !== "validated") throw new Error(`${id}: recommend-mesh status ${parsed.status}`);
  const payload = {
    mesh: raw.mesh,
    recommendation: {
      surface_areal_mass_g_cm2: raw.recommendation?.surface_areal_mass_g_cm2,
      mode: raw.recommendation?.mode,
    },
    evidence: (raw.evidence ?? []).map((item: { id: string }) => ({ id: item.id })),
    flags: raw.flags ?? [],
    warnings: raw.warnings ?? [],
    confidence: raw.confidence ?? "",
    validation: {
      status: raw.validation?.status,
      achieved_surface_areal_mass_g_cm2: raw.validation?.achieved_surface_areal_mass_g_cm2,
    },
  };
  const key = fs.readFileSync(path.join(dir, `${id}.key`), "utf8");
  entries.push(
    `  ${id}: {\n    conditionsKey: ${JSON.stringify(key)},\n    createdAt: ${JSON.stringify(createdAt)},\n    binary: ${JSON.stringify(binary)},\n    payload: ${JSON.stringify(payload, null, 2).replace(/\n/g, "\n    ")},\n  },`,
  );
}
const target = path.resolve(__dirname, "..", "src", "core", "presetMeshes.ts");
const source = fs.readFileSync(target, "utf8");
const marker = "export const PRESET_MESHES: Partial<Record<PresetMeshId, PresetMesh>> = ";
const head = source.slice(0, source.indexOf(marker));
fs.writeFileSync(target, `${head}${marker}{\n${entries.join("\n")}\n};\n`);
console.log(`wrote ${ids.length} recommended meshes to ${target}`);
