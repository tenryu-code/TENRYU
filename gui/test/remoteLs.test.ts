import { describe, expect, it } from "vitest";
import { joinRemoteDir, parseLsOutput, remoteParentDir } from "@tenryu-common/core/remoteLs";

describe("remote ls helpers", () => {
  it("parses entries with directories first and sorted names", () => {
    expect(parseLsOutput("./\n../\nsub/\nb.h5\nA.h5\nnotes.txt\n")).toEqual([
      { name: "sub", dir: true },
      { name: "A.h5", dir: false },
      { name: "b.h5", dir: false },
      { name: "notes.txt", dir: false },
    ]);
  });

  it("parses empty output", () => {
    expect(parseLsOutput("")).toEqual([]);
  });

  it("resolves parent directories", () => {
    expect(remoteParentDir("/a/b")).toBe("/a");
    expect(remoteParentDir("/a")).toBe("/");
    expect(remoteParentDir("/")).toBe("/");
    expect(remoteParentDir("relative")).toBe("/");
  });

  it("joins remote directory paths", () => {
    expect(joinRemoteDir("/a", "b")).toBe("/a/b");
    expect(joinRemoteDir("/", "b")).toBe("/b");
  });
});
