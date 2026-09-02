import { open, save } from "@tauri-apps/plugin-dialog";
import { readFile, readTextFile, writeFile, writeTextFile } from "@tauri-apps/plugin-fs";
import { Command } from "@tauri-apps/plugin-shell";
import { LazyStore } from "@tauri-apps/plugin-store";
import type { ServerProfile } from "../core/profiles";
import { planExec, planReadText, planUploadText } from "../core/ssh";
import { readBinaryViaExec } from "./readBinary";
import type { AppSettings, Backend, ExecOpts, ExecResult } from "./types";

const LOCAL_TRANSPORT_ERROR =
  "ローカル transport は開発用 (dev-bridge) 専用です。ssh プロファイルを使ってください。";

export class TauriBackend implements Backend {
  readonly kind = "tauri" as const;
  private readonly store: LazyStore;

  constructor(storeFile: string = "tenryu-studio.json") {
    this.store = new LazyStore(storeFile);
  }

  async listProfiles(): Promise<ServerProfile[]> {
    return (await this.store.get<ServerProfile[]>("profiles")) ?? [];
  }

  async saveProfiles(profiles: ServerProfile[]): Promise<void> {
    await this.store.set("profiles", profiles);
    await this.store.save();
  }

  async getSettings(): Promise<AppSettings> {
    return (await this.store.get<AppSettings>("settings")) ?? {};
  }

  async saveSettings(s: AppSettings): Promise<void> {
    await this.store.set("settings", s);
    await this.store.save();
  }

  async saveLocalTextFile(suggestedName: string, content: string, filterName?: string): Promise<string | null> {
    const extension = suggestedName.split(".").pop()?.toLowerCase() ?? "";
    const path = await save({
      defaultPath: suggestedName,
      filters: [{ name: filterName ?? extension.toUpperCase(), extensions: [extension] }],
    });
    if (!path) return null;
    await writeTextFile(path, content);
    return path;
  }

  async saveLocalBinaryFile(suggestedName: string, bytes: Uint8Array): Promise<string | null> {
    const extension = suggestedName.split(".").pop()?.toLowerCase() ?? "";
    const path = await save({
      defaultPath: suggestedName,
      filters: [{ name: extension.toUpperCase(), extensions: [extension] }],
    });
    if (!path) return null;
    await writeFile(path, bytes);
    return path;
  }

  async writeLocalTextFile(path: string, content: string): Promise<void> {
    await writeTextFile(path, content);
  }

  async openLocalTextFile(extensions: string[]): Promise<{ name: string; path: string; content: string } | null> {
    const path = extensions.length === 0
      ? await open({ multiple: false, directory: false })
      : await open({
          multiple: false,
          directory: false,
          filters: [{ name: "TENRYU deck", extensions }],
        });
    if (typeof path !== "string") return null;
    const content = await readTextFile(path);
    const name = path.split("/").pop() ?? path;
    return { name, path, content };
  }

  async openLocalBinaryFile(extensions: string[]): Promise<{ name: string; path: string; bytes: Uint8Array } | null> {
    const path = extensions.length === 0
      ? await open({ multiple: false, directory: false })
      : await open({
          multiple: false,
          directory: false,
          filters: [{ name: "TENRYU data", extensions }],
        });
    if (typeof path !== "string") return null;
    const bytes = await readFile(path);
    const name = path.split("/").pop() ?? path;
    return { name, path, bytes };
  }

  private async runSsh(argv: string[], _opts?: ExecOpts): Promise<ExecResult> {
    if (argv[0] !== "ssh") throw new Error(`unexpected local command: ${argv[0]}`);
    const out = await Command.create("ssh", argv.slice(1)).execute();
    return { code: out.code, stdout: out.stdout, stderr: out.stderr, timedOut: false };
  }

  async exec(profile: ServerProfile, remoteArgv: string[], opts?: ExecOpts): Promise<ExecResult> {
    if (profile.transport === "local") throw new Error(LOCAL_TRANSPORT_ERROR);
    const plan = planExec(profile, remoteArgv);
    return await this.runSsh(plan.argv, opts);
  }

  async uploadText(profile: ServerProfile, remotePath: string, content: string): Promise<void> {
    if (profile.transport === "local") throw new Error(LOCAL_TRANSPORT_ERROR);
    const plan = planUploadText(profile, remotePath, content);
    if (plan.kind !== "exec") throw new Error(LOCAL_TRANSPORT_ERROR);
    const r = await this.runSsh(plan.argv);
    if (r.code !== 0) {
      throw new Error(`uploadText failed (code=${r.code}): ${r.stderr}`);
    }
  }

  async readText(profile: ServerProfile, remotePath: string, maxBytes = 262144): Promise<string> {
    if (profile.transport === "local") throw new Error(LOCAL_TRANSPORT_ERROR);
    const plan = planReadText(profile, remotePath, maxBytes);
    if (plan.kind !== "exec") throw new Error(LOCAL_TRANSPORT_ERROR);
    const r = await this.runSsh(plan.argv);
    if (r.code !== 0) {
      throw new Error(`readText failed (code=${r.code}): ${r.stderr}`);
    }
    return r.stdout;
  }

  async readBinary(profile: ServerProfile, remotePath: string, maxBytes = 33554432): Promise<Uint8Array> {
    return await readBinaryViaExec(this.exec.bind(this), profile, remotePath, maxBytes);
  }

  async execLocal(argv: string[], _opts?: ExecOpts): Promise<ExecResult> {
    // Capability whitelist mirrors src-tauri/capabilities/default.json: local
    // execution is restricted to bash (login shell restores the user PATH).
    if (argv[0] !== "bash") throw new Error(`execLocal: only bash is permitted (got ${argv[0]})`);
    const out = await Command.create("bash", argv.slice(1)).execute();
    return { code: out.code, stdout: out.stdout, stderr: out.stderr, timedOut: false };
  }

  async readLocalText(path: string, maxBytes = 262144): Promise<string> {
    const content = await readTextFile(path);
    return content.length > maxBytes ? content.slice(content.length - maxBytes) : content;
  }

  async writeLocalText(path: string, content: string): Promise<void> {
    await writeTextFile(path, content);
  }
}
