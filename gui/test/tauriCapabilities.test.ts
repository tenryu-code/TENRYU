import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

interface ScopedPermission {
  identifier: string;
  allow?: Array<Record<string, unknown>>;
}

describe("Tauri filesystem capabilities", () => {
  it("allows binary files selected outside app-specific directories to be read", () => {
    const capability = JSON.parse(
      readFileSync(new URL("../src-tauri/capabilities/default.json", import.meta.url), "utf8"),
    ) as { permissions: Array<string | ScopedPermission> };
    const readFile = capability.permissions.find(
      (permission): permission is ScopedPermission =>
        typeof permission !== "string" && permission.identifier === "fs:allow-read-file",
    );

    expect(readFile?.allow).toContainEqual({ path: "**" });
  });

  it("grants shell spawn and kill for the local bash used by execLocal", () => {
    const capability = JSON.parse(
      readFileSync(new URL("../src-tauri/capabilities/default.json", import.meta.url), "utf8"),
    ) as { permissions: Array<string | ScopedPermission> };
    const spawn = capability.permissions.find(
      (permission): permission is ScopedPermission =>
        typeof permission !== "string" && permission.identifier === "shell:allow-spawn",
    );
    expect(spawn?.allow).toContainEqual({ name: "bash", cmd: "bash", args: true, sidecar: false });
    expect(capability.permissions).toContain("shell:allow-kill");
  });
});
