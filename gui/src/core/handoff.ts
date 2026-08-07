import { dataDir } from "@tauri-apps/api/path";
import { writeTextFile } from "@tauri-apps/plugin-fs";

export interface ScopeHandoff {
  profileId: string;
  profileName: string;
  run: string;
  ts: number;
}

// Write the handoff into Scope's app-data dir (exists once Scope has run at least once).
export async function writeScopeHandoff(handoff: Omit<ScopeHandoff, "ts">): Promise<void> {
  const dir = await dataDir();
  const path = `${dir}/jp.osaka-u.ile.tenryu-scope/handoff.json`;
  await writeTextFile(path, JSON.stringify({ ...handoff, ts: Date.now() }));
}
