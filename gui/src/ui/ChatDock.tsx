import { useEffect, useRef, useState } from "react";
import type { AskChecksView } from "../core/assist/parse";
import { buildChatMessages } from "../core/chatModel";
import { Badge, Button } from "@tenryu-common/ui/kit";
import { t } from "../i18n";
import { useApp } from "../store";
import { AssistError } from "./AssistBits";
import { MarkdownLite } from "./MarkdownLite";

const USER_BUBBLE = "self-end max-w-[85%] rounded px-3 py-2";
const ASSISTANT_BUBBLE = "self-start max-w-[96%] rounded border px-3 py-2";
const ASSISTANT_BUBBLE_STYLE = { borderColor: "var(--separator)", background: "var(--bg)" };

function CopyButton({ text }: { text: string }) {
  const m = t();
  const [copied, setCopied] = useState(false);

  async function copy(): Promise<void> {
    if (typeof navigator === "undefined" || !navigator.clipboard) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard access is best-effort */
    }
  }

  return (
    <button
      className="text-[11px]"
      style={{ color: "var(--fg-secondary)" }}
      onClick={() => void copy()}
    >
      {copied ? m.chat.copied : m.chat.copy}
    </button>
  );
}

function EvidenceFooter({ checks }: { checks: AskChecksView }) {
  const m = t();
  const verified = checks.citationsVerified;
  const unverified = checks.citationsUnverified;
  const keysUnknown = checks.keysUnknown;

  return (
    <div className="mt-2 flex flex-wrap items-center gap-1 text-[11px]">
      <Badge tone={unverified.length === 0 ? "ok" : "warn"}>
        {m.chat.citations}: {verified.length}/{verified.length + unverified.length}
      </Badge>
      {keysUnknown.length > 0 && (
        <Badge tone="warn">
          {m.chat.keysUnknown}: {keysUnknown.length}
        </Badge>
      )}
      {(unverified.length > 0 || keysUnknown.length > 0) && (
        <details className="mt-1">
          <summary style={{ color: "var(--fg-secondary)" }}>{m.chat.details}</summary>
          <ul style={{ paddingLeft: "1.2em" }}>
            {unverified.map((item, index) => (
              <li key={`c${index}`} style={{ fontFamily: "var(--mono)" }}>
                {item.citation} — {item.reason}
              </li>
            ))}
            {keysUnknown.map((key, index) => (
              <li key={`k${index}`} style={{ fontFamily: "var(--mono)" }}>
                {key}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

export default function ChatDock() {
  const m = t();
  const assistAsk = useApp((s) => s.assistAsk);
  const assistQuestion = useApp((s) => s.assistQuestion);
  const setAssistQuestion = useApp((s) => s.setAssistQuestion);
  const askAssistQuestion = useApp((s) => s.askAssistQuestion);
  const cancelAssistQuestion = useApp((s) => s.cancelAssistQuestion);
  const resetAssistConversation = useApp((s) => s.resetAssistConversation);
  const assistStatus = useApp((s) => s.assistStatus);
  const fetchAssistStatus = useApp((s) => s.fetchAssistStatus);
  const assistLocalRepo = useApp((s) => s.assistLocalRepo);
  const setChatOpen = useApp((s) => s.setChatOpen);
  const setView = useApp((s) => s.setView);
  const listRef = useRef<HTMLDivElement | null>(null);

  const running = assistAsk.phase === "running";
  const canSend = !running && assistQuestion.trim().length > 0;
  const messages = buildChatMessages(assistAsk);

  const providerRow =
    assistStatus.status === "ready"
      ? assistStatus.view?.roles.find((row) => row.role === "question_answering") ?? null
      : null;
  const providerLabel =
    providerRow === null
      ? ""
      : `${providerRow.provider}${providerRow.model ? ` · ${providerRow.model}` : ""}`;

  useEffect(() => {
    if (assistStatus.status === "idle" && assistLocalRepo.trim() !== "") void fetchAssistStatus();
    // Mount-only probe: later status changes are driven by the assistant view.
  }, []);

  useEffect(() => {
    const list = listRef.current;
    if (list !== null) list.scrollTop = list.scrollHeight;
  }, [messages.length, assistAsk.phase, assistAsk.lastKind]);

  return (
    <aside
      className="flex min-h-0 min-w-0 flex-col border-l"
      style={{ borderColor: "var(--separator)", background: "var(--bg-panel)" }}
    >
      <div
        className="flex items-center gap-2 border-b px-3 py-2"
        style={{ borderColor: "var(--separator)" }}
      >
        <span className="text-sm font-semibold">{m.chat.title}</span>
        <span
          className="min-w-0 flex-1 truncate text-xs"
          style={{ color: "var(--fg-secondary)", fontFamily: "var(--mono)" }}
        >
          {providerLabel}
        </span>
        <Button
          className="whitespace-nowrap"
          disabled={running || assistAsk.turns.length === 0}
          onClick={() => resetAssistConversation()}
        >
          {m.chat.newConversation}
        </Button>
        <Button
          className="whitespace-nowrap"
          onClick={() => setChatOpen(false)}
          title="⌘J / Ctrl+J"
        >
          {m.chat.close}
        </Button>
      </div>

      <div
        ref={listRef}
        className="flex min-h-0 flex-1 flex-col gap-2 overflow-auto p-3 text-xs"
      >
        {messages.length === 0 && (
          <div>
            <div style={{ color: "var(--fg-secondary)" }}>{m.chat.empty}</div>
            <div className="mt-2">
              <Button onClick={() => setView("assist")}>
                {m.chat.openAssistant}
              </Button>
            </div>
          </div>
        )}
        {messages.map((message) => {
          if (message.kind === "user") {
            return (
              <div
                key={message.key}
                className={USER_BUBBLE}
                style={{
                  background: "var(--selected-bg)",
                  color: "var(--selected-fg)",
                  whiteSpace: "pre-wrap",
                  opacity: message.pending ? 0.8 : 1,
                }}
              >
                {message.text}
              </div>
            );
          }
          if (message.kind === "assistant") {
            return (
              <div key={message.key} className={ASSISTANT_BUBBLE} style={ASSISTANT_BUBBLE_STYLE}>
                <MarkdownLite text={message.text} />
                {message.checks !== null && <EvidenceFooter checks={message.checks} />}
                <div className="mt-1">
                  <CopyButton text={message.text} />
                </div>
              </div>
            );
          }
          if (message.kind === "progress") {
            return (
              <div
                key={message.key}
                className={`${ASSISTANT_BUBBLE} animate-pulse`}
                style={ASSISTANT_BUBBLE_STYLE}
              >
                {m.chat.progress[message.stage]}…
              </div>
            );
          }
          return (
            <div key={message.key} className={ASSISTANT_BUBBLE} style={ASSISTANT_BUBBLE_STYLE}>
              <AssistError text={`${message.code}: ${message.detail ?? ""}`} />
              {(message.code === "NO_LOCAL_REPO" || message.code === "NO_LOCAL_ASSIST" || message.code === "NO_SOURCE") && (
                <div className="mt-2">
                  <Button onClick={() => setView("assist")}>{m.chat.openAssistant}</Button>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div
        className="flex flex-col gap-1 border-t p-2"
        style={{ borderColor: "var(--separator)" }}
      >
        <textarea
          rows={2}
          className="w-full rounded border p-2 text-xs"
          style={{
            background: "var(--bg-inset)",
            color: "var(--fg)",
            borderColor: "var(--separator)",
            fontFamily: "var(--font-ui)",
            resize: "none",
          }}
          placeholder={m.chat.placeholder}
          value={assistQuestion}
          onChange={(e) => setAssistQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key !== "Enter" || e.shiftKey) return;
            if (e.nativeEvent.isComposing || e.keyCode === 229) return;
            e.preventDefault();
            if (canSend) void askAssistQuestion();
          }}
        />
        <div className="flex items-center gap-2">
          <span className="text-[11px]" style={{ color: "var(--fg-secondary)" }}>
            {m.chat.hint}
          </span>
          <div className="flex-1" />
          {running ? (
            <Button variant="danger" onClick={() => void cancelAssistQuestion()}>
              {m.chat.cancel}
            </Button>
          ) : (
            <Button variant="primary" disabled={!canSend} onClick={() => void askAssistQuestion()}>
              {m.chat.send}
            </Button>
          )}
        </div>
      </div>
    </aside>
  );
}
