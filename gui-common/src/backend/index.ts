import { DevBridgeBackend } from "./devbridge";
import { TauriBackend } from "./tauri";
import type { Backend } from "./types";

export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

let cached: Backend | null = null;

export function getBackend(storeFile?: string): Backend {
  if (cached === null) {
    cached = isTauri() ? new TauriBackend(storeFile) : new DevBridgeBackend();
  }
  return cached;
}

export type { Backend } from "./types";
