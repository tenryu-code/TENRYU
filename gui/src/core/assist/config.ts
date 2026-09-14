import type { AssistStatusView } from "./parse";

export interface AssistProviderForm {
  name: string;
  command: string;
  model: string;
}

export interface AssistConfigForm {
  enabled: boolean;
  providers: AssistProviderForm[];
  roles: { question_answering: string; deck_design: string };
  extraRoles: Record<string, string>;
  budget: {
    maxInterventionsPerRun: number;
    maxTokensPerDecision: number;
  };
}

export type PresetKey =
  | "claude_readonly"
  | "codex_readonly"
  | "claude"
  | "codex";

export const PROVIDER_PRESETS: Record<PresetKey, AssistProviderForm> = {
  claude_readonly: {
    name: "claude_readonly",
    command:
      "sh -c 'claude -p --model {model} --tools Read,Grep,Glob --no-session-persistence --add-dir \"$(dirname {prompt_file})\" < {prompt_file}'",
    model: "opus",
  },
  codex_readonly: {
    name: "codex_readonly",
    command:
      "sh -c 'codex exec --skip-git-repo-check -m {model} -s read-only - < {prompt_file}'",
    model: "gpt-5.6-sol",
  },
  claude: {
    name: "claude",
    command: "sh -c 'claude -p --model {model} < {prompt_file}'",
    model: "opus",
  },
  codex: {
    name: "codex",
    command:
      "sh -c 'codex exec --skip-git-repo-check -m {model} -s workspace-write - < {prompt_file}'",
    model: "gpt-5.6-sol",
  },
};

export const MODEL_SUGGESTIONS: string[] = [
  "opus",
  "sonnet",
  "claude-opus-5",
  "claude-sonnet-5",
  "gpt-5.6-sol",
];

export const DEFAULT_BUDGET = {
  maxInterventionsPerRun: 3,
  maxTokensPerDecision: 30000,
};

export const MANAGED_ROLES = ["question_answering", "deck_design"] as const;

export function formFromStatus(view: AssistStatusView | null): AssistConfigForm {
  const providers =
    view?.providers.map((provider) => ({ ...provider })) ?? [];
  const seeded = providers.length === 0;
  const roleValues = Object.fromEntries(
    (view?.roles ?? []).map((role) => [role.role, role.provider]),
  );
  const extraRoles = Object.fromEntries(
    (view?.roles ?? [])
      .filter(
        (role) =>
          role.role !== "question_answering" && role.role !== "deck_design",
      )
      .map((role) => [role.role, role.provider]),
  );
  const budget = view?.budget;
  let formBudget = { ...DEFAULT_BUDGET };
  if (
    budget !== undefined &&
    budget.maxInterventionsPerRun !== null &&
    budget.maxTokensPerDecision !== null
  ) {
    formBudget = {
      maxInterventionsPerRun: budget.maxInterventionsPerRun,
      maxTokensPerDecision: budget.maxTokensPerDecision,
    };
  }

  return {
    enabled: view?.enabled ?? false,
    providers: seeded
      ? [
          { ...PROVIDER_PRESETS.claude_readonly },
          { ...PROVIDER_PRESETS.codex_readonly },
        ]
      : providers,
    roles: {
      question_answering: seeded
        ? "claude_readonly"
        : (roleValues.question_answering ?? ""),
      deck_design: seeded ? "" : (roleValues.deck_design ?? ""),
    },
    extraRoles,
    budget: formBudget,
  };
}

export interface ConfigError {
  code:
    | "NAME_INVALID"
    | "NAME_DUPLICATE"
    | "COMMAND_EMPTY"
    | "MODEL_EMPTY"
    | "ROLE_UNKNOWN"
    | "BUDGET_INVALID"
    | "STRING_NEWLINE";
  detail: string;
}

export function validateAssistConfigForm(
  form: AssistConfigForm,
): ConfigError[] {
  const errors: ConfigError[] = [];
  const names = new Set<string>();

  for (const provider of form.providers) {
    if (
      !/^[A-Za-z_][A-Za-z0-9_]*$/.test(provider.name) ||
      provider.name === "dry_run"
    ) {
      errors.push({ code: "NAME_INVALID", detail: provider.name });
    }
    if (names.has(provider.name)) {
      errors.push({ code: "NAME_DUPLICATE", detail: provider.name });
    }
    names.add(provider.name);
    if (provider.command.trim() === "") {
      errors.push({ code: "COMMAND_EMPTY", detail: provider.name });
    }
    if (provider.model.trim() === "") {
      errors.push({ code: "MODEL_EMPTY", detail: provider.name });
    }
    if (
      [provider.name, provider.command, provider.model].some((value) =>
        /[\n\r]/.test(value),
      )
    ) {
      errors.push({ code: "STRING_NEWLINE", detail: provider.name });
    }
  }

  for (const [role, value] of [
    ...Object.entries(form.roles),
    ...Object.entries(form.extraRoles),
  ]) {
    if (value !== "" && value !== "dry_run" && !names.has(value)) {
      errors.push({ code: "ROLE_UNKNOWN", detail: `${role}=${value}` });
    }
  }

  if (
    !Number.isInteger(form.budget.maxInterventionsPerRun) ||
    form.budget.maxInterventionsPerRun < 1 ||
    !Number.isInteger(form.budget.maxTokensPerDecision) ||
    form.budget.maxTokensPerDecision < 1
  ) {
    errors.push({
      code: "BUDGET_INVALID",
      detail: `${form.budget.maxInterventionsPerRun}/${form.budget.maxTokensPerDecision}`,
    });
  }

  return errors;
}

export function escapeTomlString(value: string): string {
  return `"${value.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
}

export function buildAssistantToml(form: AssistConfigForm): string {
  const lines = [
    "# TENRYU assistant configuration — written by TENRYU Studio",
    `enabled = ${form.enabled ? "true" : "false"}`,
    "",
  ];

  for (const provider of form.providers) {
    lines.push(
      `[providers.${provider.name}]`,
      `command = ${escapeTomlString(provider.command)}`,
      `model = ${escapeTomlString(provider.model)}`,
      "",
    );
  }

  const extraRoles = Object.entries(form.extraRoles)
    .filter(([, provider]) => provider !== "")
    .sort(([left], [right]) => left.localeCompare(right));
  const hasRoles =
    form.roles.question_answering !== "" ||
    form.roles.deck_design !== "" ||
    extraRoles.length > 0;
  if (hasRoles) {
    lines.push("[roles]");
    if (form.roles.question_answering !== "") {
      lines.push(
        `question_answering = ${escapeTomlString(form.roles.question_answering)}`,
      );
    }
    if (form.roles.deck_design !== "") {
      lines.push(`deck_design = ${escapeTomlString(form.roles.deck_design)}`);
    }
    for (const [role, provider] of extraRoles) {
      lines.push(`${role} = ${escapeTomlString(provider)}`);
    }
    lines.push("");
  }

  lines.push(
    "[budget]",
    `max_interventions_per_run = ${form.budget.maxInterventionsPerRun}`,
    `max_tokens_per_decision = ${form.budget.maxTokensPerDecision}`,
  );
  return `${lines.join("\n")}\n`;
}

export function addPreset(
  form: AssistConfigForm,
  key: PresetKey,
): AssistConfigForm {
  const preset = PROVIDER_PRESETS[key];
  if (form.providers.some((provider) => provider.name === preset.name)) {
    return { ...form };
  }

  const roles = { ...form.roles };
  if (
    (key === "claude_readonly" || key === "codex_readonly") &&
    roles.question_answering === ""
  ) {
    roles.question_answering = preset.name;
  }
  if (
    (key === "claude" || key === "codex") &&
    roles.deck_design === ""
  ) {
    roles.deck_design = preset.name;
  }

  return {
    ...form,
    providers: [...form.providers, { ...preset }],
    roles,
  };
}
