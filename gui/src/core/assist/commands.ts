/** Pure builders for assistant commands; two venues per design §2. */

import { shQuote, sshDestination } from "@tenryu-common/core/ssh";
import type { ServerProfile } from "@tenryu-common/core/profiles";

/** Character policy for values embedded in remote command strings (mirrors the
 *  tenryu_remote.sh argument guard). */
const SAFE_REMOTE = /^[A-Za-z0-9_./+=:@-]+$/;
const SAFE_IDENTITY = /^[A-Za-z0-9_./+=:@~-]+$/; // ssh itself expands ~ in -i values

const BASE_SSH_OPTS =
  "-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5";

/** Checkout subset copied into the local mirror (paths relative to the server repo root). */
export const MIRROR_PATHS: readonly string[] = [
  "docs/site", "docs/SPECIFICATION.md", "docs/TUTORIAL_ja.md", "docs/OUTPUT_SCHEMA.md",
  "docs/POSTPROCESSING.md", "docs/NUMERICS.md", "docs/ARCHITECTURE.md", "docs/VERIFICATION.md",
  "docs/sections", "examples", "tools/mesh_planner.py", "src",
];
export const MIRROR_EXCLUDES: readonly string[] = ["__pycache__", "*.pdf", "*.h5", "*.prp", "*.cn4", "*.o", "*.a"];

export interface MirrorSyncArgs {
  host: string;           // ssh destination (user@host or alias), already validated by buildRemoteWrapperEnv
  sshOpts: string;        // TENRYU_REMOTE_SSH_OPTS from buildRemoteWrapperEnv
  serverRepoRoot: string; // absolute, validated
  mirrorRoot: string;     // absolute local directory
  harnessDir: string;     // absolute local path of the bundled tools/assist
}

/** Fetch the checkout subset with tar over ssh into a temporary directory, then swap it in. */
export function buildMirrorSyncScript(a: MirrorSyncArgs): string {
  const excludes = MIRROR_EXCLUDES.map((e) => `--exclude=${shQuote(e)}`).join(" ");
  const paths = MIRROR_PATHS.join(" ");
  const remoteCmd = `cd ${shQuote(a.serverRepoRoot)} && tar -czf - ${excludes} ${paths} 2>/dev/null`;
  return `tmp=${shQuote(a.mirrorRoot + ".tmp")} && dst=${shQuote(a.mirrorRoot)} && rm -rf "$tmp" && mkdir -p "$tmp" && (ssh ${a.sshOpts} ${a.host} ${shQuote(remoteCmd)} || true) | tar -xzf - -C "$tmp"; rm -rf "$tmp/tools/assist" && mkdir -p "$tmp/tools" && cp -R ${shQuote(a.harnessDir)} "$tmp/tools/assist" && missing=""; for f in docs/SPECIFICATION.md docs/site/ja/index.html src/core/namelist/builder.cpp; do test -f "$tmp/$f" || missing="$missing $f"; done; if [ -n "$missing" ]; then echo "MISSING:$missing"; exit 3; fi; test -f "$tmp/tools/assist/assist.py" || { echo "MISSING: bundled tools/assist/assist.py"; exit 4; }; rm -rf "$dst" && mv "$tmp" "$dst" && echo MIRROR_OK`;
}

export function buildMirrorProbeScript(mirrorRoot: string): string {
  return `test -f ${shQuote(mirrorRoot)}/tools/assist/assist.py && echo MIRROR_OK`;
}

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

export function buildFileExistsScript(path: string): string {
  return `test -f ${shQuote(path)} && echo FILE_OK`;
}

export function buildProbeAssistScript(repo: string): string {
  return `test -f ${shQuote(repo)}/tools/assist/assist.py && echo ASSIST_OK`;
}

/** Local status using the Studio config path or built-in defaults when absent. */
export function buildStatusScript(
  localRepo: string,
  configPath: string,
): string {
  let script = `cd ${shQuote(localRepo)} && python3 tools/assist/assist.py status`;
  script += ` --config-or-defaults ${shQuote(configPath)}`;
  return script;
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
  /** Studio config path; missing files select the built-in defaults. */
  configPath: string;
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
  script += ` --config-or-defaults ${shQuote(a.configPath)}`;
  return script;
}

export interface AskScriptArgs {
  localRepo: string;
  workdir: string;
  questionPath: string;
  /** Studio config path; missing files select the built-in defaults. */
  configPath: string;
}

/** Local question answering: `assist.py ask --json` in the local checkout. */
export function buildAskScript(a: AskScriptArgs): string {
  let script =
    `cd ${shQuote(a.localRepo)} && python3 tools/assist/assist.py ask --json` +
    ` --workdir ${shQuote(a.workdir)} --question-file ${shQuote(a.questionPath)}`;
  script += ` --config-or-defaults ${shQuote(a.configPath)}`;
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
