import { describe, expect, it } from "vitest";
import { buildChatMessages, progressStage, type AskStateLike } from "../src/core/chatModel";
import type { AskChecksView } from "../src/core/assist/parse";

const CHECKS: AskChecksView = {
  citationsVerified: ["SPECIFICATION.md:42"],
  citationsUnverified: [],
  keysKnown: [],
  keysUnknown: [],
};

function askState(patch: Partial<AskStateLike> = {}): AskStateLike {
  return {
    phase: "idle",
    turns: [],
    pendingQuestion: null,
    lastKind: null,
    errorCode: null,
    errorDetail: null,
    ...patch,
  };
}

describe("progressStage", () => {
  it("maps the journal event kinds to the shown stage", () => {
    expect(progressStage(null)).toBe("start");
    expect(progressStage("docmap_generated")).toBe("exploring");
    expect(progressStage("llm_invocation")).toBe("checking");
    expect(progressStage("answer_checked")).toBe("finishing");
    expect(progressStage("something_else")).toBe("exploring");
  });
});

describe("buildChatMessages", () => {
  it("returns nothing for an idle, empty conversation", () => {
    expect(buildChatMessages(askState())).toEqual([]);
  });

  it("pairs each turn as a user then an assistant message", () => {
    const messages = buildChatMessages(
      askState({
        phase: "answered",
        turns: [
          { turn: 1, question: "Q1", answer: "A1", checks: null },
          { turn: 2, question: "Q2", answer: "A2", checks: CHECKS },
        ],
      }),
    );
    expect(messages).toEqual([
      { kind: "user", key: "u1", text: "Q1", pending: false },
      { kind: "assistant", key: "a1", text: "A1", checks: null },
      { kind: "user", key: "u2", text: "Q2", pending: false },
      { kind: "assistant", key: "a2", text: "A2", checks: CHECKS },
    ]);
  });

  it("appends the pending question and a progress bubble while running", () => {
    const messages = buildChatMessages(
      askState({
        phase: "running",
        turns: [{ turn: 1, question: "Q1", answer: "A1", checks: null }],
        pendingQuestion: "Q2",
        lastKind: "docmap_generated",
      }),
    );
    expect(messages).toHaveLength(4);
    expect(messages[2]).toEqual({ kind: "user", key: "u-pending", text: "Q2", pending: true });
    expect(messages[3]).toEqual({ kind: "progress", key: "progress", stage: "exploring" });
  });

  it("keeps the pending question and appends the error after a failure", () => {
    const messages = buildChatMessages(
      askState({
        phase: "error",
        pendingQuestion: "Q1",
        errorCode: "ASK_FAILED",
        errorDetail: "provider failed",
      }),
    );
    expect(messages).toEqual([
      { kind: "user", key: "u-pending", text: "Q1", pending: true },
      { kind: "error", key: "error", code: "ASK_FAILED", detail: "provider failed" },
    ]);
  });

  it("omits the error bubble when no error code is recorded", () => {
    const messages = buildChatMessages(askState({ phase: "error", errorCode: null }));
    expect(messages).toEqual([]);
  });
});
