import type { ServerProfile } from "./profiles";

/** POSIX shell single-argument quoting. Empty string quotes to ''. */
export function shQuote(s: string): string {
  if (s.length > 0 && /^[A-Za-z0-9_\/.:=+,@%^-]+$/.test(s)) return s;
  return "'" + s.replaceAll("'", "'\\''") + "'";
}

/**
 * Quote a remote path while preserving leading tilde expansion:
 * "~/a b" -> ~/'a b' so the remote shell still expands ~.
 */
export function shQuotePath(s: string): string {
  if (s === "~") return "~";
  if (s.startsWith("~/")) return "~/" + shQuote(s.slice(2));
  return shQuote(s);
}

export function remoteDirname(p: string): string {
  const i = p.lastIndexOf("/");
  if (i < 0) return ".";
  if (i === 0) return "/";
  return p.slice(0, i);
}

export function sshDestination(p: ServerProfile): string {
  return p.user && p.user.length > 0 ? `${p.user}@${p.host}` : p.host;
}

export const SSH_BASE_OPTS: readonly string[] = [
  "-o",
  "BatchMode=yes",
  "-o",
  "ConnectTimeout=8",
  "-o",
  "ServerAliveInterval=5",
];

function portOpts(p: ServerProfile): string[] {
  return p.port !== undefined ? ["-p", String(p.port)] : [];
}

function keyOpts(p: ServerProfile): string[] {
  const f = p.identityFile?.trim() ?? "";
  return f.length > 0 ? ["-i", f] : [];
}

function hostKeyOpts(p: ServerProfile): string[] {
  return p.ephemeralHostKey === true
    ? ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    : [];
}

function sshArgv(p: ServerProfile, remoteCommand: string): string[] {
  return [
    "ssh",
    ...SSH_BASE_OPTS,
    ...hostKeyOpts(p),
    ...keyOpts(p),
    ...portOpts(p),
    sshDestination(p),
    "--",
    remoteCommand,
  ];
}

export interface ExecPlan {
  argv: string[];
}

/** Local argv to run `remoteArgv` on the profile's server. */
export function planExec(p: ServerProfile, remoteArgv: string[]): ExecPlan {
  if (remoteArgv.length === 0) throw new Error("planExec: empty argv");
  if (p.transport === "local") return { argv: [...remoteArgv] };
  const remoteCommand = remoteArgv.map(shQuotePath).join(" ");
  return { argv: sshArgv(p, remoteCommand) };
}

export type UploadPlan =
  | { kind: "exec"; argv: string[] }
  | { kind: "write-file"; path: string };

/**
 * Plan writing `content` to remotePath. For ssh the content is embedded in the
 * remote command via printf '%s' (no stdin dependency); caller passes content.
 */
export function planUploadText(p: ServerProfile, remotePath: string, content: string): UploadPlan {
  if (p.transport === "local") return { kind: "write-file", path: remotePath };
  const dir = remoteDirname(remotePath);
  const remoteCommand =
    "mkdir -p " +
    shQuotePath(dir) +
    " && printf '%s' " +
    shQuote(content) +
    " > " +
    shQuotePath(remotePath);
  return { kind: "exec", argv: sshArgv(p, remoteCommand) };
}

export type ReadPlan =
  | { kind: "exec"; argv: string[] }
  | { kind: "read-file"; path: string; maxBytes: number };

/** Plan reading the last maxBytes of remotePath. */
export function planReadText(p: ServerProfile, remotePath: string, maxBytes = 262144): ReadPlan {
  if (p.transport === "local") return { kind: "read-file", path: remotePath, maxBytes };
  const remoteCommand = `tail -c ${Math.max(1, Math.floor(maxBytes))} ` + shQuotePath(remotePath);
  return { kind: "exec", argv: sshArgv(p, remoteCommand) };
}

/** Join a remote base dir and a relative path with a single slash. */
export function joinRemote(base: string, rel: string): string {
  const b = base.endsWith("/") ? base.slice(0, -1) : base;
  const r = rel.startsWith("/") ? rel.slice(1) : rel;
  return `${b}/${r}`;
}
