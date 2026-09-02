export type Transport = "ssh" | "local";

export interface ServerProfile {
  id: string;
  name: string;
  transport: Transport;
  /** ssh transport: hostname or ~/.ssh/config alias */
  host: string;
  user?: string;
  port?: number;
  /** ssh -i identity file path (e.g. ~/.ssh/runpod_ed25519); empty/undefined = default keys */
  identityFile?: string;
  /** ephemeral cloud host: disable host-key pinning (StrictHostKeyChecking=no + no known_hosts) */
  ephemeralHostKey?: boolean;
  /** absolute path of the tenryu binary on the server */
  tenryuBin: string;
  /** base directory for GUI runs on the server (absolute or ~/...) */
  runDir: string;
}

export function newProfile(): ServerProfile {
  return {
    id: newId(),
    name: "",
    transport: "ssh",
    host: "",
    user: "",
    port: undefined,
    identityFile: "",
    ephemeralHostKey: false,
    tenryuBin: "",
    runDir: "~/tenryu_gui_runs",
  };
}

/** Returns a list of Japanese validation error messages (empty = valid). */
export function validateProfile(p: ServerProfile): string[] {
  const errs: string[] = [];
  if (!p.name.trim()) errs.push("表示名を入力してください");
  if (p.transport === "ssh") {
    if (!p.host.trim()) errs.push("ホストを入力してください");
    if (/\s/.test(p.host)) errs.push("ホストに空白は使えません");
    if (p.user && /[\s@]/.test(p.user)) errs.push("ユーザー名に空白や @ は使えません");
    if (p.port !== undefined) {
      if (!Number.isInteger(p.port) || p.port < 1 || p.port > 65535) {
        errs.push("ポートは 1-65535 の整数です");
      }
    }
  }
  if (/\n/.test(p.identityFile ?? "")) errs.push("秘密鍵パスに改行は使えません");
  if (p.transport === "local" && p.runDir.startsWith("~")) {
    errs.push("ローカル接続では実行ディレクトリは絶対パスで指定してください");
  }
  if (/\n/.test(p.tenryuBin)) errs.push("tenryu バイナリパスに改行は使えません");
  if (!p.runDir.trim()) errs.push("実行ディレクトリを入力してください");
  return errs;
}

/** Parse ops/runpod/.pod_ssh ("IP PORT" single line); null when malformed. */
export function parsePodSsh(text: string): { host: string; port: number } | null {
  const m = /^\s*([0-9.]+)\s+(\d{1,5})\s*$/.exec(text.split("\n")[0] ?? "");
  if (!m) return null;
  const port = Number(m[2]);
  if (!Number.isInteger(port) || port < 1 || port > 65535) return null;
  return { host: m[1], port };
}

/** True when the profile has no tenryu binary path configured (run/validate need it). */
export function profileBinMissing(p: ServerProfile): boolean {
  return p.tenryuBin.trim() === "";
}
import { newId } from "./ids";
