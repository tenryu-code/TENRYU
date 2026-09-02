import { describe, expect, it } from "vitest";
import type { ServerProfile } from "@tenryu-common/core/profiles";
import {
  buildCancelScript,
  buildEchoHomeScript,
  buildEnvPrefix,
  buildGenerateScript,
  buildLatestOutputDirScript,
  buildMkdirScript,
  buildProbeAssistScript,
  buildRemoteAssistScript,
  buildRemoteWrapperEnv,
  buildStatusScript,
} from "../src/core/assist/commands";

const BASE_OPTS =
  "-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5";

function profile(overrides: Partial<ServerProfile> = {}): ServerProfile {
  return {
    id: "profile",
    name: "Profile",
    transport: "ssh",
    host: "parma",
    tenryuBin: "/srv/TENRYU/build/tenryu",
    runDir: "/srv/tenryu-runs",
    ...overrides,
  };
}

describe("buildRemoteWrapperEnv", () => {
  it("builds the base environment for a plain host alias", () => {
    expect(
      buildRemoteWrapperEnv(
        profile(),
        "/srv/TENRYU",
        "/srv/TENRYU/build/tenryu",
      ),
    ).toEqual({
      env: {
        TENRYU_REMOTE_HOST: "parma",
        TENRYU_REMOTE_REPO: "/srv/TENRYU",
        TENRYU_REMOTE_BIN: "/srv/TENRYU/build/tenryu",
        TENRYU_REMOTE_SSH_OPTS: BASE_OPTS,
        TENRYU_REMOTE_SCP_OPTS: BASE_OPTS,
        RSYNC_RSH: `ssh ${BASE_OPTS}`,
      },
      error: null,
    });
  });

  it("orders host-key, identity, and port options for a full profile", () => {
    const result = buildRemoteWrapperEnv(
      profile({
        host: "gpu.example.org",
        user: "user",
        port: 2222,
        identityFile: "~/.ssh/runpod_ed25519",
        ephemeralHostKey: true,
      }),
      "/opt/TENRYU",
      "/opt/TENRYU/build/tenryu",
    );
    const common =
      `${BASE_OPTS} -o StrictHostKeyChecking=no ` +
      "-o UserKnownHostsFile=/dev/null -i ~/.ssh/runpod_ed25519";

    expect(result).toEqual({
      env: {
        TENRYU_REMOTE_HOST: "user@gpu.example.org",
        TENRYU_REMOTE_REPO: "/opt/TENRYU",
        TENRYU_REMOTE_BIN: "/opt/TENRYU/build/tenryu",
        TENRYU_REMOTE_SSH_OPTS: `${common} -p 2222`,
        TENRYU_REMOTE_SCP_OPTS: `${common} -P 2222`,
        RSYNC_RSH: `ssh ${common} -p 2222`,
      },
      error: null,
    });
  });

  it("rejects an identity path containing a space", () => {
    expect(
      buildRemoteWrapperEnv(
        profile({ identityFile: "/tmp/key file" }),
        "/srv/TENRYU",
        "/srv/TENRYU/build/tenryu",
      ),
    ).toEqual({ env: {}, error: "IDENTITY_PATH_UNSUPPORTED" });
  });

  it("rejects a non-absolute remote repository root", () => {
    expect(
      buildRemoteWrapperEnv(
        profile(),
        "srv/TENRYU",
        "/srv/TENRYU/build/tenryu",
      ),
    ).toEqual({ env: {}, error: "REMOTE_PATH_UNSUPPORTED" });
  });
});

describe("buildEnvPrefix", () => {
  it("returns an empty string for an empty environment", () => {
    expect(buildEnvPrefix({})).toBe("");
  });

  it("sorts keys and quotes values", () => {
    expect(buildEnvPrefix({ B: "y z", A: "x" })).toBe("env A=x B='y z'");
  });
});

describe("buildGenerateScript", () => {
  it("builds the full generate-deck command", () => {
    expect(
      buildGenerateScript({
        localRepo: "/Users/me/TENRYU repo",
        workdir: "/tmp/assist run",
        specPath: "/tmp/assist run/spec.md",
        outDeckPath: "/tmp/assist run/out_deck.py",
        maxIters: 7,
        templatePath: "/tmp/assist run/template.py",
        intentPath: "/tmp/assist run/intent.json",
        tenryuArg: "tools/assist/tenryu_remote.sh",
        env: { B: "y z", A: "x" },
      }),
    ).toBe(
      "cd '/Users/me/TENRYU repo' && env A=x B='y z' " +
        "python3 tools/assist/assist.py generate-deck " +
        "'/tmp/assist run/spec.md' --out-deck '/tmp/assist run/out_deck.py' " +
        "--tenryu tools/assist/tenryu_remote.sh --workdir '/tmp/assist run' " +
        "--max-iters 7 --template '/tmp/assist run/template.py' " +
        "--intent '/tmp/assist run/intent.json'",
    );
  });

  it("omits the env prefix and optional paths", () => {
    expect(
      buildGenerateScript({
        localRepo: "/repo",
        workdir: "/tmp/work",
        specPath: "/tmp/spec.md",
        outDeckPath: "/tmp/out.py",
        maxIters: 5,
        templatePath: null,
        intentPath: null,
        tenryuArg: "/repo/build/tenryu",
        env: {},
      }),
    ).toBe(
      "cd /repo && python3 tools/assist/assist.py generate-deck /tmp/spec.md " +
        "--out-deck /tmp/out.py --tenryu /repo/build/tenryu " +
        "--workdir /tmp/work --max-iters 5",
    );
  });

  it("clamps max-iters to the supported range", () => {
    const base = {
      localRepo: "/repo",
      workdir: "/tmp/work",
      specPath: "/tmp/spec.md",
      outDeckPath: "/tmp/out.py",
      templatePath: null,
      intentPath: null,
      tenryuArg: "/repo/build/tenryu",
      env: {},
    };

    expect(buildGenerateScript({ ...base, maxIters: 0 })).toContain(
      "--max-iters 1",
    );
    expect(buildGenerateScript({ ...base, maxIters: 99 })).toContain(
      "--max-iters 10",
    );
  });
});

describe("assistant shell-script builders", () => {
  it("builds the cancel script", () => {
    expect(buildCancelScript("/tmp/assist run")).toBe(
      "pkill -f -- '/tmp/assist run' 2>/dev/null || true",
    );
  });

  it("builds the latest-output lookup", () => {
    expect(buildLatestOutputDirScript("/srv/run dir")).toBe(
      "ls -td '/srv/run dir'/outputs/*/ 2>/dev/null | head -1",
    );
  });

  it("builds a remote assistant command", () => {
    expect(
      buildRemoteAssistScript("/srv/TENRYU repo", [
        "lint-deck",
        "/tmp/a deck.py",
        "--tenryu",
        "/srv/TENRYU/build/tenryu",
      ]),
    ).toBe(
      "cd '/srv/TENRYU repo' && python3 tools/assist/assist.py " +
        "lint-deck '/tmp/a deck.py' --tenryu /srv/TENRYU/build/tenryu",
    );
  });

  it("builds the assistant probe", () => {
    expect(buildProbeAssistScript("/srv/TENRYU repo")).toBe(
      "test -f '/srv/TENRYU repo'/tools/assist/assist.py && echo ASSIST_OK",
    );
  });

  it("builds the local status command", () => {
    expect(buildStatusScript("/Users/me/TENRYU repo")).toBe(
      "cd '/Users/me/TENRYU repo' && python3 tools/assist/assist.py status",
    );
  });

  it("builds mkdir and HOME lookup scripts", () => {
    expect(buildMkdirScript("/tmp/assist run")).toBe(
      "mkdir -p '/tmp/assist run'",
    );
    expect(buildEchoHomeScript()).toBe('echo "$HOME"');
  });
});
