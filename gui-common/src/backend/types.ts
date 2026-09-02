import type { ServerProfile } from "../core/profiles";
import type { RunRecord } from "../core/runstate";

export interface ExecResult {
  code: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
}

export interface ExecOpts {
  timeoutMs?: number;
}

export interface AppSettings {
  lastProfileId?: string;
  runs?: RunRecord[];
  lang?: "ja" | "en";
  /** Absolute path of a local TENRYU checkout providing tools/assist (assistant). */
  assistLocalRepo?: string;
}

export interface Backend {
  readonly kind: "tauri" | "devbridge";
  listProfiles(): Promise<ServerProfile[]>;
  saveProfiles(profiles: ServerProfile[]): Promise<void>;
  getSettings(): Promise<AppSettings>;
  saveSettings(s: AppSettings): Promise<void>;
  /** Run remoteArgv on the profile's server and collect output. */
  exec(profile: ServerProfile, remoteArgv: string[], opts?: ExecOpts): Promise<ExecResult>;
  /** Write a text file on the profile's server (parent dirs created). */
  uploadText(profile: ServerProfile, remotePath: string, content: string): Promise<void>;
  /** Read the last maxBytes of a text file on the profile's server. */
  readText(profile: ServerProfile, remotePath: string, maxBytes?: number): Promise<string>;
  /** Prompt for a LOCAL save location and write content. Returns the chosen path (or filename) or null when cancelled. */
  saveLocalTextFile(suggestedName: string, content: string, filterName?: string): Promise<string | null>;
  /** Native save dialog + binary write; returns the chosen path, null when cancelled. */
  saveLocalBinaryFile?(suggestedName: string, bytes: Uint8Array): Promise<string | null>;
  /** Write content to an existing LOCAL text file path. */
  writeLocalTextFile(path: string, content: string): Promise<void>;
  /** Native open dialog + read; null when the user cancels. */
  openLocalTextFile(extensions: string[]): Promise<{ name: string; path: string | null; content: string } | null>;
  /** Native open dialog + binary read; null when cancelled. */
  openLocalBinaryFile?(extensions: string[]): Promise<{ name: string; path: string | null; bytes: Uint8Array } | null>;
  /** Read a binary file on the profile's server (base64 over the exec pipe). */
  readBinary(profile: ServerProfile, remotePath: string, maxBytes?: number): Promise<Uint8Array>;
  /** Run argv on the LOCAL machine (assistant harness etc.). Tauri restricts
   *  argv[0] to "bash"; compose local work as `bash -lc <script>`. */
  execLocal(argv: string[], opts?: ExecOpts): Promise<ExecResult>;
  /** Read the last maxBytes of a LOCAL text file at an explicit path (no dialog). */
  readLocalText(path: string, maxBytes?: number): Promise<string>;
  /** Write a LOCAL text file at an explicit path (no dialog; parent must exist). */
  writeLocalText(path: string, content: string): Promise<void>;
}
