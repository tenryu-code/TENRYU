import type { AskChecksView } from "./assist/parse";

export type ProgressStage = "start" | "exploring" | "checking" | "finishing";

/** The subset of the store's `assistAsk` slice the chat transcript needs. */
export interface AskStateLike {
  phase: "idle" | "running" | "answered" | "error";
  turns: Array<{ turn: number; question: string; answer: string; checks: AskChecksView | null }>;
  pendingQuestion: string | null;
  lastKind: string | null;
  errorCode: string | null;
  errorDetail: string | null;
}

export type ChatMessage =
  | { kind: "user"; key: string; text: string; pending: boolean }
  | { kind: "assistant"; key: string; text: string; checks: AskChecksView | null }
  | { kind: "progress"; key: string; stage: ProgressStage }
  | { kind: "error"; key: string; code: string; detail: string | null };

/** Maps the last journal event kind to the progress wording shown while answering. */
export function progressStage(lastKind: string | null): ProgressStage {
  if (lastKind === null) return "start";
  if (lastKind === "docmap_generated") return "exploring";
  if (lastKind === "llm_invocation") return "checking";
  if (lastKind === "answer_checked") return "finishing";
  return "exploring";
}

export function buildChatMessages(ask: AskStateLike): ChatMessage[] {
  const messages: ChatMessage[] = [];
  for (const turn of ask.turns) {
    messages.push({ kind: "user", key: `u${turn.turn}`, text: turn.question, pending: false });
    messages.push({ kind: "assistant", key: `a${turn.turn}`, text: turn.answer, checks: turn.checks });
  }
  if (ask.pendingQuestion !== null) {
    messages.push({ kind: "user", key: "u-pending", text: ask.pendingQuestion, pending: true });
  }
  if (ask.phase === "running") {
    messages.push({ kind: "progress", key: "progress", stage: progressStage(ask.lastKind) });
  }
  if (ask.phase === "error" && ask.errorCode !== null) {
    messages.push({ kind: "error", key: "error", code: ask.errorCode, detail: ask.errorDetail });
  }
  return messages;
}
