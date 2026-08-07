import { describe, expect, it } from "vitest";
import { base64ToBytes, readBinaryViaExec } from "@tenryu-common/backend/readBinary";
import type { ServerProfile } from "@tenryu-common/core/profiles";

const PROFILE = { id: "p", name: "p", transport: "ssh", host: "h", tenryuBin: "/t", runDir: "/r" } as ServerProfile;

describe("base64ToBytes", () => {
  it("decodes multiline base64 with whitespace", () => {
    const bytes = base64ToBytes("aGVs\nbG8g\n d29ybGQ=\n");
    expect(new TextDecoder().decode(bytes)).toBe("hello world");
  });
  it("roundtrips arbitrary bytes", () => {
    const src = new Uint8Array([0, 1, 2, 250, 251, 255, 10, 13]);
    const b64 = btoa(String.fromCharCode(...src));
    expect([...base64ToBytes(b64)]).toEqual([...src]);
  });
});

describe("readBinaryViaExec", () => {
  it("stats then decodes", async () => {
    const calls: string[][] = [];
    const exec = async (_p: ServerProfile, argv: string[]) => {
      calls.push(argv);
      if (argv[2].startsWith("stat")) return { code: 0, stdout: "11\n", stderr: "", timedOut: false };
      return { code: 0, stdout: "aGVsbG8gd29ybGQ=\n", stderr: "", timedOut: false };
    };
    const bytes = await readBinaryViaExec(exec, PROFILE, "/x/file.h5", 1024);
    expect(new TextDecoder().decode(bytes)).toBe("hello world");
    expect(calls).toHaveLength(2);
  });
  it("rejects files larger than maxBytes", async () => {
    const exec = async () => ({ code: 0, stdout: "999999\n", stderr: "", timedOut: false });
    await expect(readBinaryViaExec(exec, PROFILE, "/x/big.h5", 1024)).rejects.toThrow("> 1024");
  });
});
