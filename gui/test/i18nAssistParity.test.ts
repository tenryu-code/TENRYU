import { describe, expect, it } from "vitest";
import { en } from "../src/i18n/en";
import { ja } from "../src/i18n/ja";

function keyPaths(value: unknown, prefix = ""): Set<string> {
  const paths = new Set<string>();
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return paths;
  }
  for (const [key, child] of Object.entries(value)) {
    const path = prefix.length > 0 ? `${prefix}.${key}` : key;
    paths.add(path);
    for (const childPath of keyPaths(child, path)) paths.add(childPath);
  }
  return paths;
}

describe("i18n key parity", () => {
  it("has the same recursive key set in Japanese and English", () => {
    const jaPaths = keyPaths(ja);
    const enPaths = keyPaths(en);
    const missingInEn = [...jaPaths].filter((path) => !enPaths.has(path)).sort();
    const missingInJa = [...enPaths].filter((path) => !jaPaths.has(path)).sort();

    expect(missingInEn, `Missing in en:\n${missingInEn.join("\n")}`).toEqual([]);
    expect(missingInJa, `Missing in ja:\n${missingInJa.join("\n")}`).toEqual([]);
  });
});
