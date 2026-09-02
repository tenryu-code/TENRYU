/** Pure builders for assistant commands; two venues per design §2. */

import { shQuote, sshDestination } from "@tenryu-common/core/ssh";
import type { ServerProfile } from "@tenryu-common/core/profiles";

/** Character policy for values embedded in remote command strings (mirrors the
 *  tenryu_remote.sh argument guard). */
const SAFE_REMOTE = /^[A-Za-z0-9_./+=:@-]+$/;
const SAFE_IDENTITY = /^[A-Za-z0-9_./+=:@~-]+$/; // ssh itself expands ~ in -i values

const BASE_SSH_OPTS =
  "-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5";

export interface RemoteWrapperEnv {
  env: Record<string, string>;
  /** null when composable; otherwise one of "HOST_UNSUPPORTED",
   *  "REMOTE_PATH_UNSUPPORTED", "IDENTITY_PATH_UNSUPPORTED". */
  error: string | null;
}

export function buildRemoteWrapperEnv(
  profile: ServerProfile,
  serverRepoRoot: string,
  serverBinAbs: string,
): RemoteWrapperEnv {
  const host = sshDestination(profile);
  if (!SAFE_REMOTE.test(host)) {
    return { env: {}, error: "HOST_UNSUPPORTED" };
  }
  if (
    !serverRepoRoot.startsWith("/") ||
    !SAFE_REMOTE.test(serverRepoRoot) ||
    !serverBinAbs.startsWith("/") ||
    !SAFE_REMOTE.test(serverBinAbs)
  ) {
    return { env: {}, error: "REMOTE_PATH_UNSUPPORTED" };
  }

  const identity = profile.identityFile?.trim() ?? "";
  if (identity.length > 0 && !SAFE_IDENTITY.test(identity)) {
    return { env: {}, error: "IDENTITY_PATH_UNSUPPORTED" };
  }

  let sshOpts = BASE_SSH_OPTS;
  let scpOpts = BASE_SSH_OPTS;
  if (profile.ephemeralHostKey === true) {
    const hostKeyOpts =
      " -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null";
    sshOpts += hostKeyOpts;
    scpOpts += hostKeyOpts;
  }
  if (identity.length > 0) {
    sshOpts += ` -i ${identity}`;
    scpOpts += ` -i ${identity}`;
  }
  if (profile.port !== undefined) {
    sshOpts += ` -p ${profile.port}`;
    scpOpts += ` -P ${profile.port}`;
  }

  return {
    env: {
      TENRYU_REMOTE_HOST: host,
      TENRYU_REMOTE_REPO: serverRepoRoot,
      TENRYU_REMOTE_BIN: serverBinAbs,
      TENRYU_REMOTE_SSH_OPTS: sshOpts,
      TENRYU_REMOTE_SCP_OPTS: scpOpts,
      RSYNC_RSH: `ssh ${sshOpts}`,
    },
    error: null,
  };
}

/** "env K=V ..." prefix with shell-quoted values, keys sorted; "" for empty env. */
export function buildEnvPrefix(env: Record<string, string>): string {
  const keys = Object.keys(env).sort();
  if (keys.length === 0) return "";
  return `env ${keys.map((key) => `${key}=${shQuote(env[key])}`).join(" ")}`;
}

export function buildMkdirScript(dir: string): string {
  return `mkdir -p ${shQuote(dir)}`;
}

export function buildEchoHomeScript(): string {
  return 'echo "$HOME"';
}

export function buildProbeAssistScript(repo: string): string {
  return `test -f ${shQuote(repo)}/tools/assist/assist.py && echo ASSIST_OK`;
}

export function buildStatusScript(localRepo: string): string {
  return `cd ${shQuote(localRepo)} && python3 tools/assist/assist.py status`;
}

export interface GenerateScriptArgs {
  localRepo: string;
  workdir: string;
  specPath: string;
  outDeckPath: string;
  maxIters: number;
  templatePath: string | null;
  intentPath: string | null;
  /** "tools/assist/tenryu_remote.sh" (ssh profiles) or an absolute local binary. */
  tenryuArg: string;
  env: Record<string, string>;
}

export function buildGenerateScript(a: GenerateScriptArgs): string {
  const envPrefix = buildEnvPrefix(a.env);
  const maxIters = Math.max(1, Math.min(10, Math.floor(a.maxIters)));
  let script =
    `cd ${shQuote(a.localRepo)} && ` +
    (envPrefix.length > 0 ? `${envPrefix} ` : "") +
    "python3 tools/assist/assist.py generate-deck " +
    `${shQuote(a.specPath)} --out-deck ${shQuote(a.outDeckPath)} ` +
    `--tenryu ${shQuote(a.tenryuArg)} --workdir ${shQuote(a.workdir)} ` +
    `--max-iters ${maxIters}`;
  if (a.templatePath !== null) {
    script += ` --template ${shQuote(a.templatePath)}`;
  }
  if (a.intentPath !== null) {
    script += ` --intent ${shQuote(a.intentPath)}`;
  }
  return script;
}

/** pkill by the unique workdir path; exit 0 even when nothing matched. */
export function buildCancelScript(workdir: string): string {
  return `pkill -f -- ${shQuote(workdir)} 2>/dev/null || true`;
}

export function buildLatestOutputDirScript(runDir: string): string {
  return `ls -td ${shQuote(runDir)}/outputs/*/ 2>/dev/null | head -1`;
}

export function buildRemoteAssistScript(root: string, args: string[]): string {
  const suffix = args.length > 0 ? ` ${args.map(shQuote).join(" ")}` : "";
  return `cd ${shQuote(root)} && python3 tools/assist/assist.py${suffix}`;
}
