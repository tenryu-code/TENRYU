import { describe, expect, it } from "vitest";
import {
  DEFAULT_BUDGET,
  PROVIDER_PRESETS,
  addPreset,
  buildAssistantToml,
  formFromStatus,
  validateAssistConfigForm,
  type AssistConfigForm,
} from "../src/core/assist/config";
import type { AssistStatusView } from "../src/core/assist/parse";

function validForm(): AssistConfigForm {
  return {
    enabled: true,
    providers: [{ name: "a", command: "run {prompt_file}", model: "model-a" }],
    roles: { question_answering: "a", deck_design: "" },
    extraRoles: {},
    budget: { ...DEFAULT_BUDGET },
  };
}

describe("formFromStatus", () => {
  it("seeds read-only providers and default values without a status", () => {
    expect(formFromStatus(null)).toEqual({
      enabled: false,
      providers: [
        PROVIDER_PRESETS.claude_readonly,
        PROVIDER_PRESETS.codex_readonly,
      ],
      roles: {
        question_answering: "claude_readonly",
        deck_design: "",
      },
      extraRoles: {},
      budget: DEFAULT_BUDGET,
    });
  });

  it("maps providers, managed roles, extra roles, and budget", () => {
    const view: AssistStatusView = {
      enabled: true,
      disabledBy: null,
      configSource: "/tmp/assistant.toml",
      providers: [
        { name: "a", command: "command-a", model: "model-a" },
        { name: "b", command: "command-b", model: "model-b" },
      ],
      roles: [
        { role: "question_answering", provider: "a", model: "model-a" },
        { role: "explanation", provider: "b", model: "model-b" },
      ],
      budget: {
        maxInterventionsPerRun: 5,
        maxTokensPerDecision: 1000,
      },
      warnings: [],
    };

    expect(formFromStatus(view)).toEqual({
      enabled: true,
      providers: [
        { name: "a", command: "command-a", model: "model-a" },
        { name: "b", command: "command-b", model: "model-b" },
      ],
      roles: { question_answering: "a", deck_design: "" },
      extraRoles: { explanation: "b" },
      budget: { maxInterventionsPerRun: 5, maxTokensPerDecision: 1000 },
    });
  });
});

describe("buildAssistantToml", () => {
  it("writes an exact escaped configuration", () => {
    const form: AssistConfigForm = {
      enabled: true,
      providers: [
        {
          name: "a",
          command: 'tool --say "hello" --path C:\\tmp',
          model: "model-a",
        },
        { name: "b", command: "tool-b", model: "model-b" },
      ],
      roles: { question_answering: "a", deck_design: "b" },
      extraRoles: { explanation: "a" },
      budget: { maxInterventionsPerRun: 3, maxTokensPerDecision: 30000 },
    };

    expect(buildAssistantToml(form)).toBe(
      [
        "# TENRYU assistant configuration — written by TENRYU Studio",
        "enabled = true",
        "",
        "[providers.a]",
        'command = "tool --say \\"hello\\" --path C:\\\\tmp"',
        'model = "model-a"',
        "",
        "[providers.b]",
        'command = "tool-b"',
        'model = "model-b"',
        "",
        "[roles]",
        'question_answering = "a"',
        'deck_design = "b"',
        'explanation = "a"',
        "",
        "[budget]",
        "max_interventions_per_run = 3",
        "max_tokens_per_decision = 30000",
        "",
      ].join("\n"),
    );
  });

  it("omits the roles section when every role is unset", () => {
    const form = validForm();
    form.roles.question_answering = "";

    const text = buildAssistantToml(form);
    expect(text).not.toContain("[roles]");
    expect(text).toContain(
      '[providers.a]\ncommand = "run {prompt_file}"\nmodel = "model-a"\n\n[budget]',
    );
  });
});

describe("validateAssistConfigForm", () => {
  it.each([
    ["bad name", { name: "1abc" }, "NAME_INVALID"],
    ["reserved name", { name: "dry_run" }, "NAME_INVALID"],
    ["empty model", { model: " " }, "MODEL_EMPTY"],
    ["newline", { command: "run\nnext" }, "STRING_NEWLINE"],
  ] as const)("reports %s", (_label, providerPatch, code) => {
    const form = validForm();
    form.providers[0] = { ...form.providers[0], ...providerPatch };
    expect(validateAssistConfigForm(form).map((error) => error.code)).toContain(
      code,
    );
  });

  it("reports duplicate names", () => {
    const form = validForm();
    form.providers.push({ ...form.providers[0] });
    expect(validateAssistConfigForm(form).map((error) => error.code)).toContain(
      "NAME_DUPLICATE",
    );
  });

  it("reports a role pointing at an unknown provider", () => {
    const form = validForm();
    form.roles.deck_design = "missing";
    expect(validateAssistConfigForm(form).map((error) => error.code)).toContain(
      "ROLE_UNKNOWN",
    );
  });

  it("reports a zero budget", () => {
    const form = validForm();
    form.budget.maxInterventionsPerRun = 0;
    expect(validateAssistConfigForm(form).map((error) => error.code)).toContain(
      "BUDGET_INVALID",
    );
  });

  it("returns no errors for a valid form", () => {
    expect(validateAssistConfigForm(validForm())).toEqual([]);
  });
});

describe("addPreset", () => {
  it("adds a preset once and fills the matching empty role", () => {
    const form = validForm();
    form.roles.deck_design = "";

    const added = addPreset(form, "codex");
    const addedAgain = addPreset(added, "codex");

    expect(addedAgain.providers.filter((provider) => provider.name === "codex"))
      .toHaveLength(1);
    expect(addedAgain.roles.deck_design).toBe("codex");
  });
});
