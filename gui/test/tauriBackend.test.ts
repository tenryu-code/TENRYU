import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";

type ClosePayload = { code: number | null; signal: number | null };
type CommandListener = (payload: ClosePayload | string) => void;
type DataListener = (chunk: Uint8Array) => void;

interface CommandRecord {
  program: string;
  args: string[] | undefined;
  options: { encoding?: string } | undefined;
  command: {
    emitStdout(chunk: Uint8Array): void;
    emitStderr(chunk: Uint8Array): void;
    emitClose(payload: ClosePayload): void;
    emitError(error: string): void;
  };
}

const shellMock = vi.hoisted(() => ({
  commands: [] as CommandRecord[],
  order: [] as string[],
  nextPid: 1000,
  closeOnKill: false,
}));

vi.mock("@tauri-apps/plugin-dialog", () => ({ open: vi.fn(), save: vi.fn() }));
vi.mock("@tauri-apps/api/path", () => ({
  appConfigDir: vi.fn(),
  join: vi.fn(),
  resourceDir: vi.fn(),
}));
vi.mock("@tauri-apps/plugin-fs", () => ({
  readTextFile: vi.fn(),
  writeTextFile: vi.fn(),
  readFile: vi.fn(),
  writeFile: vi.fn(),
}));
vi.mock("@tauri-apps/plugin-shell", () => {
  class MockCommand {
    readonly stdout = {
      on: (_event: "data", listener: DataListener) => {
        this.stdoutListeners.push(listener);
      },
    };
    readonly stderr = {
      on: (_event: "data", listener: DataListener) => {
        this.stderrListeners.push(listener);
      },
    };
    private readonly stdoutListeners: DataListener[] = [];
    private readonly stderrListeners: DataListener[] = [];
    private readonly listeners: Record<"close" | "error", CommandListener[]> = {
      close: [],
      error: [],
    };

    static create(program: string, args?: string[], options?: { encoding?: string }): MockCommand {
      const command = new MockCommand();
      shellMock.commands.push({ program, args, options, command });
      return command;
    }

    on(event: "close" | "error", listener: CommandListener): void {
      this.listeners[event].push(listener);
    }

    async spawn(): Promise<{ pid: number; kill: () => Promise<void> }> {
      const pid = shellMock.nextPid;
      return {
        pid,
        kill: async () => {
          shellMock.order.push("kill");
          if (shellMock.closeOnKill) this.emitClose({ code: null, signal: 9 });
        },
      };
    }

    async execute(): Promise<{ code: number; stdout: string; stderr: string; signal: null }> {
      shellMock.order.push("execute");
      return { code: 0, stdout: "", stderr: "", signal: null };
    }

    emitStdout(chunk: Uint8Array): void {
      for (const listener of this.stdoutListeners) listener(chunk);
    }

    emitStderr(chunk: Uint8Array): void {
      for (const listener of this.stderrListeners) listener(chunk);
    }

    emitClose(payload: ClosePayload): void {
      for (const listener of this.listeners.close) listener(payload);
    }

    emitError(error: string): void {
      for (const listener of this.listeners.error) listener(error);
    }
  }

  return { Command: MockCommand };
});
vi.mock("@tauri-apps/plugin-store", () => ({
  LazyStore: class {
    get = vi.fn();
    set = vi.fn();
    save = vi.fn();
  },
}));

import { TauriBackend } from "@tenryu-common/backend/tauri";

const bytes = (text: string): Uint8Array => new TextEncoder().encode(text);

describe("TauriBackend.execLocal", () => {
  beforeEach(() => {
    shellMock.commands.length = 0;
    shellMock.order.length = 0;
    shellMock.nextPid = 1000;
    shellMock.closeOnKill = false;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("collects raw output and resolves a normal exit", async () => {
    const backend = new TauriBackend();
    const resultPromise = backend.execLocal(["bash", "-lc", "echo hi"], { timeoutMs: 5000 });
    await Promise.resolve();

    const record = shellMock.commands[0];
    record.command.emitStdout(bytes("he"));
    record.command.emitStdout(bytes("llo\n"));
    record.command.emitStderr(bytes("warn\n"));
    record.command.emitClose({ code: 3, signal: null });

    await expect(resultPromise).resolves.toEqual({
      code: 3,
      stdout: "hello\n",
      stderr: "warn\n",
      timedOut: false,
    });
    expect(record).toMatchObject({
      program: "bash",
      args: ["-lc", "echo hi"],
      options: { encoding: "raw" },
    });
  });

  it("terminates a timed-out command before killing its wrapper", async () => {
    vi.useFakeTimers();
    shellMock.nextPid = 4242;
    shellMock.closeOnKill = true;
    const backend = new TauriBackend();
    const resultPromise = backend.execLocal(["bash", "-lc", "sleep 10"], { timeoutMs: 2000 });
    await Promise.resolve();

    await vi.advanceTimersByTimeAsync(2000);

    await expect(resultPromise).resolves.toEqual({
      code: null,
      stdout: "",
      stderr: "",
      timedOut: true,
    });
    expect(shellMock.commands[1]).toMatchObject({
      program: "bash",
      args: ["-lc", "pkill -TERM -P 4242; sleep 2; pkill -KILL -P 4242; exit 0"],
      options: undefined,
    });
    expect(shellMock.order).toEqual(["execute", "kill"]);
  });

  it("uses the default 600000 ms timeout", async () => {
    vi.useFakeTimers();
    shellMock.closeOnKill = true;
    const backend = new TauriBackend();
    const resultPromise = backend.execLocal(["bash", "-lc", "sleep 1000"]);
    await Promise.resolve();

    expect(vi.getTimerCount()).toBe(1);
    await vi.advanceTimersByTimeAsync(599999);
    expect(shellMock.commands).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(shellMock.commands).toHaveLength(2);
    await expect(resultPromise).resolves.toMatchObject({ code: null, timedOut: true });
  });

  it("rejects command errors", async () => {
    const backend = new TauriBackend();
    const resultPromise = backend.execLocal(["bash", "-lc", "exit 1"]);
    await Promise.resolve();

    shellMock.commands[0].command.emitError("spawn failed");

    await expect(resultPromise).rejects.toThrow("spawn failed");
  });

  it("rejects commands other than bash", async () => {
    const backend = new TauriBackend();
    await expect(backend.execLocal(["sh", "-c", "x"])).rejects.toThrow(
      "execLocal: only bash is permitted (got sh)",
    );
  });
});
