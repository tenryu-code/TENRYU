import type { ServerProfile } from "../core/profiles";
import type { ExecOpts, ExecResult } from "./types";

export function base64ToBytes(text: string): Uint8Array {
  const clean = text.replace(/\s+/g, "");
  const bin = atob(clean);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

type ExecFn = (profile: ServerProfile, argv: string[], opts?: ExecOpts) => Promise<ExecResult>;

/** Read a remote binary file through the text exec pipe via base64.
 *  Paths follow the GUI contract (no quotes/newlines). */
export async function readBinaryViaExec(
  exec: ExecFn,
  profile: ServerProfile,
  remotePath: string,
  maxBytes: number,
): Promise<Uint8Array> {
  const st = await exec(profile, ["bash", "-c", `stat -c %s '${remotePath}' 2>/dev/null || stat -f %z '${remotePath}'`], {
    timeoutMs: 20000,
  });
  const size = Number(st.stdout.trim().split("\n").pop());
  if (st.code !== 0 || !Number.isFinite(size)) {
    throw new Error(`readBinary: cannot stat ${remotePath}: ${st.stderr.trim() || `exit=${st.code}`}`);
  }
  if (size > maxBytes) {
    throw new Error(`readBinary: ${remotePath} is ${size} bytes (> ${maxBytes})`);
  }
  const r = await exec(profile, ["bash", "-c", `base64 < '${remotePath}'`], { timeoutMs: 120000 });
  if (r.code !== 0) {
    throw new Error(`readBinary: base64 failed (exit=${r.code}): ${r.stderr.trim()}`);
  }
  return base64ToBytes(r.stdout);
}
