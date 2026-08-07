import { Component, type ErrorInfo, type ReactNode } from "react";
import { getLogPath, logLine } from "@tenryu-common/core/applog";
import { t } from "../i18n";
import { Button } from "@tenryu-common/ui/kit";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
  logPath: string | null;
}

export default class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null, logPath: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error, logPath: null };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    logLine("crash", `render error: ${error.stack}\ncomponentStack: ${info.componentStack}`);
    void getLogPath().then((logPath) => this.setState({ logPath }));
  }

  render(): ReactNode {
    if (this.state.error === null) return this.props.children;
    const m = t();
    return (
      <div
        className="m-4 rounded border p-6"
        style={{ background: "var(--bg-panel)", borderColor: "var(--err)", color: "var(--fg)" }}
      >
        <h1 className="mb-3 text-lg font-semibold" style={{ color: "var(--err)" }}>
          {m.errlog.title}
        </h1>
        <p className="mb-3 whitespace-pre-wrap break-all" style={{ fontFamily: "var(--mono)" }}>
          {this.state.error.message}
        </p>
        <p className="mb-2" style={{ color: "var(--fg-secondary)" }}>
          {m.errlog.hint}
        </p>
        <p className="mb-4 break-all" style={{ fontFamily: "var(--mono)", color: "var(--fg-secondary)" }}>
          {this.state.logPath ?? ""}
        </p>
        <Button onClick={() => window.location.reload()}>{m.errlog.reload}</Button>
      </div>
    );
  }
}
