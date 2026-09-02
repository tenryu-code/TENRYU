import type { ServerProfile } from "../core/profiles";
import { planExec, planReadText, planUploadText } from "../core/ssh";
import { readBinaryViaExec } from "./readBinary";
import type { AppSettings, Backend, ExecOpts, ExecResult } from "./types";

interface SpawnResponse {
  code: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
}

function downloadTextFile(name: string, content: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: "text/x-python" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadBinaryFile(name: string, bytes: Uint8Array): void {
  const extension = name.split(".").pop()?.toLowerCase();
  const type = extension === "mp4"
    ? "video/mp4"
    : extension === "webm"
      ? "video/webm"
      : extension === "png"
        ? "image/png"
        : "application/octet-stream";
  const url = URL.createObjectURL(new Blob([bytes], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

export class DevBridgeBackend implements Backend {
  readonly kind = "devbridge" as const;
  private readonly base: string;

  constructor(baseUrl = "") {
    this.base = baseUrl;
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(this.base + path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`bridge ${path} failed: HTTP ${res.status} ${text}`);
    }
    return (await res.json()) as T;
  }

  private async get<T>(path: string): Promise<T> {
    const res = await fetch(this.base + path);
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`bridge ${path} failed: HTTP ${res.status} ${text}`);
    }
    return (await res.json()) as T;
  }

  private async put<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(this.base + path, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`bridge ${path} failed: HTTP ${res.status} ${text}`);
    }
    return (await res.json()) as T;
  }

  async listProfiles(): Promise<ServerProfile[]> {
    const r = await this.get<{ profiles: ServerProfile[] }>("/api/profiles");
    return r.profiles ?? [];
  }

  async saveProfiles(profiles: ServerProfile[]): Promise<void> {
    await this.put("/api/profiles", { profiles });
  }

  async getSettings(): Promise<AppSettings> {
    const r = await this.get<{ settings: AppSettings }>("/api/settings");
    return r.settings ?? {};
  }

  async saveSettings(s: AppSettings): Promise<void> {
    await this.put("/api/settings", { settings: s });
  }

  async saveLocalTextFile(suggestedName: string, content: string, _filterName?: string): Promise<string> {
    downloadTextFile(suggestedName, content);
    return suggestedName;
  }

  async saveLocalBinaryFile(suggestedName: string, bytes: Uint8Array): Promise<string> {
    downloadBinaryFile(suggestedName, bytes);
    return suggestedName;
  }

  async writeLocalTextFile(path: string, content: string): Promise<void> {
    // Dev bridge cannot overwrite a local path, so degrade to a browser download.
    downloadTextFile(path.split("/").pop() ?? path, content);
  }

  async openLocalTextFile(extensions: string[]): Promise<{ name: string; path: null; content: string } | null> {
    return await new Promise((resolve) => {
      const input = document.createElement("input");
      input.type = "file";
      if (extensions.length > 0) input.accept = extensions.map((e) => "." + e).join(",");
      input.onchange = () => {
        const file = input.files?.[0];
        if (!file) { resolve(null); return; }
        const reader = new FileReader();
        reader.onload = () => resolve({ name: file.name, path: null, content: String(reader.result ?? "") });
        reader.onerror = () => resolve(null);
        reader.readAsText(file);
      };
      input.oncancel = () => resolve(null);
      input.click();
    });
  }

  async openLocalBinaryFile(extensions: string[]): Promise<{ name: string; path: null; bytes: Uint8Array } | null> {
    return await new Promise((resolve) => {
      const input = document.createElement("input");
      input.type = "file";
      if (extensions.length > 0) input.accept = extensions.map((e) => "." + e).join(",");
      input.onchange = () => {
        const file = input.files?.[0];
        if (!file) { resolve(null); return; }
        const reader = new FileReader();
        reader.onload = () => {
          if (!(reader.result instanceof ArrayBuffer)) { resolve(null); return; }
          resolve({ name: file.name, path: null, bytes: new Uint8Array(reader.result) });
        };
        reader.onerror = () => resolve(null);
        reader.readAsArrayBuffer(file);
      };
      input.oncancel = () => resolve(null);
      input.click();
    });
  }

  private spawn(argv: string[], opts?: ExecOpts): Promise<SpawnResponse> {
    return this.post<SpawnResponse>("/api/spawn", {
      argv,
      timeoutMs: opts?.timeoutMs,
    });
  }

  async exec(profile: ServerProfile, remoteArgv: string[], opts?: ExecOpts): Promise<ExecResult> {
    const plan = planExec(profile, remoteArgv);
    return await this.spawn(plan.argv, opts);
  }

  async uploadText(profile: ServerProfile, remotePath: string, content: string): Promise<void> {
    const plan = planUploadText(profile, remotePath, content);
    if (plan.kind === "write-file") {
      await this.post("/api/write-file", { path: plan.path, content });
      return;
    }
    const r = await this.spawn(plan.argv, { timeoutMs: 30000 });
    if (r.code !== 0) {
      throw new Error(`uploadText failed (code=${r.code}): ${r.stderr}`);
    }
  }

  async readText(profile: ServerProfile, remotePath: string, maxBytes = 262144): Promise<string> {
    const plan = planReadText(profile, remotePath, maxBytes);
    if (plan.kind === "read-file") {
      const r = await this.post<{ content: string }>("/api/read-file", {
        path: plan.path,
        maxBytes: plan.maxBytes,
      });
      return r.content;
    }
    const r = await this.spawn(plan.argv, { timeoutMs: 30000 });
    if (r.code !== 0) {
      throw new Error(`readText failed (code=${r.code}): ${r.stderr}`);
    }
    return r.stdout;
  }

  async readBinary(profile: ServerProfile, remotePath: string, maxBytes = 33554432): Promise<Uint8Array> {
    return await readBinaryViaExec(this.exec.bind(this), profile, remotePath, maxBytes);
  }

  async execLocal(argv: string[], opts?: ExecOpts): Promise<ExecResult> {
    return await this.spawn(argv, opts);
  }

  async readLocalText(path: string, maxBytes = 262144): Promise<string> {
    const r = await this.post<{ content: string }>("/api/read-file", { path, maxBytes });
    return r.content;
  }

  async writeLocalText(path: string, content: string): Promise<void> {
    await this.post("/api/write-file", { path, content });
  }
}
