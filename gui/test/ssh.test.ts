import { describe, expect, it } from "vitest";
import type { ServerProfile } from "@tenryu-common/core/profiles";
import {
  planExec,
  planReadText,
  planUploadText,
  remoteDirname,
  shQuote,
  shQuotePath,
  SSH_BASE_OPTS,
} from "@tenryu-common/core/ssh";

const sshProfile: ServerProfile = {
  id: "p1",
  name: "lab",
  transport: "ssh",
  host: "example-host",
  user: "alice",
  port: undefined,
  tenryuBin: "/opt/tenryu/build/tenryu",
  runDir: "~/tenryu_gui_runs",
};

const localProfile: ServerProfile = {
  ...sshProfile,
  id: "p2",
  transport: "local",
};

describe("shQuote", () => {
  it("passes safe strings through", () => {
    expect(shQuote("abc_1.2/x:y=z+w,@%^-")).toBe("abc_1.2/x:y=z+w,@%^-");
  });
  it("quotes spaces", () => {
    expect(shQuote("a b")).toBe("'a b'");
  });
  it("quotes empty string", () => {
    expect(shQuote("")).toBe("''");
  });
  it("escapes single quotes", () => {
    expect(shQuote("it's")).toBe("'it'\\''s'");
  });
  it("quotes newlines preserving them", () => {
    expect(shQuote("a\nb")).toBe("'a\nb'");
  });
});

describe("shQuotePath", () => {
  it("preserves leading tilde", () => {
    expect(shQuotePath("~/run dir/x")).toBe("~/'run dir/x'");
  });
  it("bare tilde stays bare", () => {
    expect(shQuotePath("~")).toBe("~");
  });
  it("absolute path without specials stays bare", () => {
    expect(shQuotePath("/opt/tenryu")).toBe("/opt/tenryu");
  });
});

describe("remoteDirname", () => {
  it("handles nested", () => {
    expect(remoteDirname("/a/b/c.py")).toBe("/a/b");
  });
  it("handles root", () => {
    expect(remoteDirname("/a.py")).toBe("/");
  });
  it("handles bare name", () => {
    expect(remoteDirname("a.py")).toBe(".");
  });
});

describe("planExec", () => {
  it("builds ssh argv with base opts and destination", () => {
    const plan = planExec(sshProfile, ["/opt/tenryu/build/tenryu", "validate", "/tmp/deck.py"]);
    expect(plan.argv[0]).toBe("ssh");
    expect(plan.argv).toEqual([
      "ssh",
      ...SSH_BASE_OPTS,
      "alice@example-host",
      "--",
      "/opt/tenryu/build/tenryu validate /tmp/deck.py",
    ]);
    expect(plan.argv).not.toContain("-i");
    expect(plan.argv).not.toContain("StrictHostKeyChecking=no");
  });
  it("adds cloud host-key, identity, and port options in order", () => {
    const plan = planExec({
      ...sshProfile,
      identityFile: "~/.ssh/runpod_ed25519",
      ephemeralHostKey: true,
      port: 12345,
    }, ["true"]);
    const optionStart = 1 + SSH_BASE_OPTS.length;
    expect(plan.argv.slice(optionStart, optionStart + 8)).toEqual([
      "-o",
      "StrictHostKeyChecking=no",
      "-o",
      "UserKnownHostsFile=/dev/null",
      "-i",
      "~/.ssh/runpod_ed25519",
      "-p",
      "12345",
    ]);
  });
  it("adds -p when port set", () => {
    const plan = planExec({ ...sshProfile, port: 2222 }, ["true"]);
    expect(plan.argv).toContain("-p");
    expect(plan.argv).toContain("2222");
  });
  it("omits user when empty", () => {
    const plan = planExec({ ...sshProfile, user: "" }, ["true"]);
    expect(plan.argv).toContain("example-host");
    expect(plan.argv.join(" ")).not.toContain("@example-host");
  });
  it("local transport passes argv through", () => {
    const plan = planExec(localProfile, ["/bin/echo", "hi there"]);
    expect(plan.argv).toEqual(["/bin/echo", "hi there"]);
  });
  it("quotes remote args with spaces", () => {
    const plan = planExec(sshProfile, ["cat", "/tmp/a b.txt"]);
    expect(plan.argv[plan.argv.length - 1]).toBe("cat '/tmp/a b.txt'");
  });
});

describe("tilde-preserving exec", () => {
  it("preserves safe tilde paths as bare tokens", () => {
    const plan = planExec(sshProfile, [
      "/opt/tenryu",
      "run",
      "~/d/deck.py",
      "--output-dir",
      "~/d/out",
    ]);
    const remote = plan.argv[plan.argv.length - 1];
    expect(remote).toContain("~/d/deck.py");
    expect(remote).toContain("~/d/out");
    expect(remote).not.toContain("'~/d/deck.py'");
    expect(remote).not.toContain("'~/d/out'");
  });
  it("quotes a spaced tilde path after the tilde prefix", () => {
    const plan = planExec(sshProfile, ["ls", "~/my dir/out"]);
    expect(plan.argv[plan.argv.length - 1]).toContain("~/'my dir/out'");
  });
  it("still quotes non-tilde args containing spaces", () => {
    const plan = planExec(sshProfile, ["echo", "a b"]);
    expect(plan.argv[plan.argv.length - 1]).toBe("echo 'a b'");
  });
  it("preserves a bare tilde as an unquoted token", () => {
    const plan = planExec(sshProfile, ["echo", "~"]);
    expect(plan.argv[plan.argv.length - 1]).toBe("echo ~");
  });
});

describe("planUploadText", () => {
  it("ssh: embeds content via printf and mkdir -p with tilde-safe quoting", () => {
    const plan = planUploadText(sshProfile, "~/tenryu_gui_runs/x/deck.py", "a'b\nc");
    expect(plan.kind).toBe("exec");
    if (plan.kind !== "exec") return;
    const remote = plan.argv[plan.argv.length - 1];
    expect(remote).toBe(
      "mkdir -p ~/tenryu_gui_runs/x && printf '%s' 'a'\\''b\nc' > ~/tenryu_gui_runs/x/deck.py",
    );
  });
  it("local: write-file plan", () => {
    const plan = planUploadText(localProfile, "/tmp/x/deck.py", "abc");
    expect(plan).toEqual({ kind: "write-file", path: "/tmp/x/deck.py" });
  });
});

describe("planReadText", () => {
  it("ssh: tail -c with maxBytes", () => {
    const plan = planReadText(sshProfile, "/var/log/run.log", 1024);
    expect(plan.kind).toBe("exec");
    if (plan.kind !== "exec") return;
    expect(plan.argv[plan.argv.length - 1]).toBe("tail -c 1024 /var/log/run.log");
  });
  it("local: read-file plan with default maxBytes", () => {
    const plan = planReadText(localProfile, "/var/log/run.log");
    expect(plan).toEqual({ kind: "read-file", path: "/var/log/run.log", maxBytes: 262144 });
  });
});
