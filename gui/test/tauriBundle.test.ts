import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

interface TauriConfig {
  build: {
    beforeDevCommand: string;
    beforeBuildCommand: string;
  };
  bundle: {
    resources: Record<string, string>;
  };
  plugins: {
    fs: {
      requireLiteralLeadingDot: boolean;
    };
  };
}

interface GuiPackage {
  scripts: Record<string, string>;
}

const guiDir = fileURLToPath(new URL("..", import.meta.url));
const sourceAssistDir = fileURLToPath(new URL("../../tools/assist", import.meta.url));
const stagedAssistDir = fileURLToPath(
  new URL("../src-tauri/assist-staging/tools/assist", import.meta.url),
);
const config = JSON.parse(
  readFileSync(new URL("../src-tauri/tauri.conf.json", import.meta.url), "utf8"),
) as TauriConfig;
const packageJson = JSON.parse(
  readFileSync(new URL("../package.json", import.meta.url), "utf8"),
) as GuiPackage;
const gitignore = readFileSync(new URL("../src-tauri/.gitignore", import.meta.url), "utf8");

function listFiles(root: string, relative = ""): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(path.join(root, relative), { withFileTypes: true })) {
    const relativePath = path.join(relative, entry.name);
    if (entry.isDirectory()) {
      if (entry.name !== "__pycache__") files.push(...listFiles(root, relativePath));
    } else if (entry.isFile() && !entry.name.endsWith(".pyc")) {
      files.push(relativePath.split(path.sep).join("/"));
    }
  }
  return files.sort();
}

function findBytecodeCaches(root: string, relative = ""): string[] {
  const matches: string[] = [];
  for (const entry of readdirSync(path.join(root, relative), { withFileTypes: true })) {
    const relativePath = path.join(relative, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "__pycache__") matches.push(relativePath);
      else matches.push(...findBytecodeCaches(root, relativePath));
    } else if (entry.isFile() && entry.name.endsWith(".pyc")) {
      matches.push(relativePath);
    }
  }
  return matches;
}

describe("Tauri assistant bundle configuration", () => {
  it("allows dot-directories and stages the assistant harness before builds", () => {
    expect(config.plugins.fs.requireLiteralLeadingDot).toBe(false);
    expect(config.bundle.resources).toEqual({
      "assist-staging/tools/assist/": "tools/assist/",
    });
    expect(config.build.beforeDevCommand.startsWith("npm run stage-assist && ")).toBe(true);
    expect(config.build.beforeBuildCommand.startsWith("npm run stage-assist && ")).toBe(true);

    const stageScript = packageJson.scripts["stage-assist"];
    expect(stageScript).toContain("--exclude __pycache__");
    expect(stageScript).toContain("--exclude '*.pyc'");
    expect(stageScript).toContain("--delete");
    expect(gitignore.split(/\r?\n/)).toContain("assist-staging/");
  });

  const rsyncCheck = spawnSync("rsync", ["--version"], { stdio: "ignore" });
  const rsyncUnavailable =
    rsyncCheck.error !== undefined &&
    "code" in rsyncCheck.error &&
    rsyncCheck.error.code === "ENOENT";
  const stagingIt = rsyncUnavailable ? it.skip : it;

  stagingIt("stages every non-bytecode assistant file", () => {
    execFileSync("npm", ["run", "-s", "stage-assist"], { cwd: guiDir, stdio: "pipe" });

    expect(findBytecodeCaches(stagedAssistDir)).toEqual([]);
    expect(existsSync(path.join(stagedAssistDir, "assist.py"))).toBe(true);
    expect(existsSync(path.join(stagedAssistDir, "providers.py"))).toBe(true);
    expect(existsSync(path.join(stagedAssistDir, "skills/codex/tenryu-docs-qa/SKILL.md"))).toBe(true);
    expect(existsSync(path.join(stagedAssistDir, "examples/qa_eval.jsonl"))).toBe(true);
    expect(listFiles(stagedAssistDir)).toEqual(listFiles(sourceAssistDir));
  });
});
