type LogLevel = "info" | "error" | "op" | "crash";

let invokeFn: ((cmd: string, args?: Record<string, unknown>) => Promise<unknown>) | null = null;
const buffered: Array<{ level: LogLevel; message: string }> = [];
const tauriDetected = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

void (async () => {
  if (!tauriDetected) return;
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    invokeFn = invoke;
    const pending = buffered.splice(0);
    for (const { level, message } of pending) {
      void invokeFn("app_log", { level, message }).catch(() => undefined);
    }
  } catch {
    invokeFn = null;
  }
})();

export function logLine(level: LogLevel, message: string): void {
  console.log(`[${level}] ${message}`);
  if (invokeFn === null) {
    if (tauriDetected) buffered.push({ level, message });
    return;
  }
  void invokeFn("app_log", { level, message }).catch(() => undefined);
}

export function logOp(message: string): void {
  logLine("op", message);
}

export function installGlobalCapture(): void {
  window.onerror = (message, source, line, col, error) => {
    logLine("crash", `${message} @ ${source}:${line}:${col}\n${error?.stack ?? ""}`);
  };
  window.onunhandledrejection = (event) => {
    const reason = event.reason;
    const stack =
      typeof reason === "object" && reason !== null && "stack" in reason
        ? String((reason as { stack?: unknown }).stack ?? "")
        : "";
    logLine("crash", `unhandledrejection: ${String(reason)}\n${stack}`);
  };
  window.setInterval(() => {
    if (invokeFn !== null) {
      void invokeFn("app_heartbeat").catch(() => undefined);
    }
  }, 2000);
}

export async function getLogPath(): Promise<string | null> {
  if (invokeFn === null) return null;
  try {
    return (await invokeFn("app_log_path")) as string | null;
  } catch {
    return null;
  }
}
