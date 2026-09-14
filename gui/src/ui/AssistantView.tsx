import { Fragment, useEffect, useRef, useState } from "react";
import { profileBinMissing } from "@tenryu-common/core/profiles";
import {
  Badge,
  Button,
  NumberInput,
  Select,
  TextInput,
} from "@tenryu-common/ui/kit";
import { MODEL_SUGGESTIONS } from "../core/assist/config";
import { t } from "../i18n";
import { currentProfile, useApp } from "../store";
import { AssistError, KvTable, RawFold } from "./AssistBits";
import { MarkdownLite } from "./MarkdownLite";

const TEXTAREA_STYLE = {
  borderColor: "var(--separator)",
  background: "var(--bg-inset)",
  color: "var(--fg)",
  fontFamily: "var(--mono)",
};

export default function AssistantView() {
  const m = t();
  const assistLocalRepo = useApp((s) => s.assistLocalRepo);
  const setAssistLocalRepo = useApp((s) => s.setAssistLocalRepo);
  const assistStatus = useApp((s) => s.assistStatus);
  const assistMirror = useApp((s) => s.assistMirror);
  const syncAssistMirror = useApp((s) => s.syncAssistMirror);
  const fetchAssistStatus = useApp((s) => s.fetchAssistStatus);
  const assistConfig = useApp((s) => s.assistConfig);
  const initAssistConfigForm = useApp((s) => s.initAssistConfigForm);
  const setAssistConfigForm = useApp((s) => s.setAssistConfigForm);
  const addAssistProviderPreset = useApp((s) => s.addAssistProviderPreset);
  const saveAssistConfig = useApp((s) => s.saveAssistConfig);
  const assistQuestion = useApp((s) => s.assistQuestion);
  const setAssistQuestion = useApp((s) => s.setAssistQuestion);
  const assistAsk = useApp((s) => s.assistAsk);
  const askAssistQuestion = useApp((s) => s.askAssistQuestion);
  const cancelAssistQuestion = useApp((s) => s.cancelAssistQuestion);
  const resetAssistConversation = useApp((s) => s.resetAssistConversation);
  const setChatOpen = useApp((s) => s.setChatOpen);
  const assistSpec = useApp((s) => s.assistSpec);
  const setAssistSpec = useApp((s) => s.setAssistSpec);
  const assistUseTemplate = useApp((s) => s.assistUseTemplate);
  const setAssistUseTemplate = useApp((s) => s.setAssistUseTemplate);
  const assistMaxIters = useApp((s) => s.assistMaxIters);
  const setAssistMaxIters = useApp((s) => s.setAssistMaxIters);
  const assistIntentJson = useApp((s) => s.assistIntentJson);
  const setAssistIntentJson = useApp((s) => s.setAssistIntentJson);
  const assistGen = useApp((s) => s.assistGen);
  const generateAssistDeck = useApp((s) => s.generateAssistDeck);
  const answerAssistClarification = useApp((s) => s.answerAssistClarification);
  const cancelAssistGeneration = useApp((s) => s.cancelAssistGeneration);
  const resetAssistGeneration = useApp((s) => s.resetAssistGeneration);
  const assistDeckValidate = useApp((s) => s.assistDeckValidate);
  const validateAssistDeck = useApp((s) => s.validateAssistDeck);
  const runAssistGeneratedDeck = useApp((s) => s.runAssistGeneratedDeck);
  const saveTextAs = useApp((s) => s.saveTextAs);
  const deck = useApp((s) => s.deck);
  const formErrors = useApp((s) => s.formErrors);
  const profile = useApp((s) => currentProfile(s));
  const [answer, setAnswer] = useState("");
  const seededFromStatus = useRef(false);
  const binMissing = profile !== null && profileBinMissing(profile);
  const deckText = assistGen.deckText ?? "";
  const configForm = assistConfig.form;

  useEffect(() => {
    if (
      assistStatus.status === "ready" &&
      seededFromStatus.current === false
    ) {
      initAssistConfigForm();
      seededFromStatus.current = true;
      return;
    }
    if (
      assistConfig.form === null &&
      assistStatus.status !== "loading"
    ) {
      initAssistConfigForm();
    }
  }, [assistConfig.form, assistStatus.status]);

  return (
    <div className="flex max-w-[880px] flex-col gap-4">
      <div>
        <div className="flex items-center gap-2">
          <h1 className="text-base font-semibold">{m.assist.title}</h1>
          <Badge tone="warn">{m.assist.experimental}</Badge>
        </div>
        <p className="mt-1 text-xs" style={{ color: "var(--fg-secondary)" }}>
          {m.assist.intro}
        </p>
      </div>

      <section
        className="flex flex-col gap-3 rounded border p-3"
        style={{ borderColor: "var(--separator)", background: "var(--bg-panel)" }}
      >
        <h2 className="text-sm font-semibold">{m.assist.setupTitle}</h2>
        <div className="flex flex-col gap-1 text-xs">
          <div className="flex items-center gap-2">
            <span style={{ color: "var(--fg-secondary)" }}>{m.assist.srcTitle}</span>
            <span>{profile ? profile.name : m.server.noneSelected}</span>
            <div className="flex-1" />
            <Button disabled={assistMirror.status === "syncing" || profile === null} onClick={() => void syncAssistMirror()}>
              {assistMirror.status === "syncing" ? m.assist.srcSyncing : m.assist.srcSync}
            </Button>
            <Button onClick={() => void fetchAssistStatus()} disabled={assistStatus.status === "loading"}>{m.assist.checkStatus}</Button>
          </div>
          <div style={{ fontFamily: "var(--mono)" }}>{m.assist.srcMirrorPath}: {assistMirror.root ?? "-"}</div>
          <div>{m.assist.srcLastSync}: {assistMirror.lastSyncIso ?? m.assist.srcNever}</div>
          <p style={{ color: "var(--fg-secondary)" }}>{m.assist.srcAuto}</p>
          {assistMirror.status === "error" && assistMirror.errorCode && (
            <AssistError text={`${assistMirror.errorCode}: ${assistMirror.errorDetail ?? ""}`} />
          )}
          <details>
            <summary style={{ color: "var(--fg-secondary)" }}>{m.assist.srcOverride}</summary>
            <div className="mt-1 flex items-center gap-2">
              <TextInput className="min-w-0 flex-1" value={assistLocalRepo} onChange={(e) => void setAssistLocalRepo(e.target.value)} style={{ fontFamily: "var(--mono)" }} placeholder={m.assist.srcOverridePlaceholder} />
            </div>
            <p className="mt-1" style={{ color: "var(--fg-secondary)" }}>{m.assist.srcOverrideHint}</p>
          </details>
        </div>
        {assistStatus.status === "loading" && (
          <Badge tone="muted">{m.common.loading}</Badge>
        )}
        {assistStatus.status === "error" && (
          <div className="flex flex-col gap-2">
            <AssistError text={assistStatus.error ?? ""} />
            <RawFold raw={assistStatus.raw} />
          </div>
        )}
        {assistStatus.status === "ready" && assistStatus.view && (
          <div className="flex flex-col gap-2">
            <div>
              <Badge tone={assistStatus.view.enabled ? "ok" : "muted"}>
                {assistStatus.view.enabled
                  ? m.assist.enabledBadge
                  : m.assist.disabledBadge}
              </Badge>
            </div>
            {assistStatus.view.disabledBy && (
              <div className="text-xs">
                {m.assist.disabledBy}: {assistStatus.view.disabledBy}
              </div>
            )}
            <div className="text-xs" style={{ fontFamily: "var(--mono)" }}>
              {m.assist.configSource}: {assistStatus.view.configSource ?? "-"}
            </div>
            <table className="w-full text-xs">
              <thead>
                <tr style={{ color: "var(--fg-secondary)" }}>
                  <th className="pr-2 text-left">{m.assist.roleCol}</th>
                  <th className="pr-2 text-left">{m.assist.providerCol}</th>
                  <th className="text-left">{m.assist.modelCol}</th>
                </tr>
              </thead>
              <tbody style={{ fontFamily: "var(--mono)" }}>
                {assistStatus.view.roles.map((role) => (
                  <tr key={role.role}>
                    <td className="pr-2">{role.role}</td>
                    <td className="pr-2">{role.provider}</td>
                    <td>{role.model ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {assistStatus.view.warnings.length > 0 && (
              <div>
                <h3 className="text-xs font-semibold" style={{ color: "var(--warn)" }}>
                  {m.assist.warningsTitle}
                </h3>
                <ul className="text-xs" style={{ color: "var(--warn)" }}>
                  {assistStatus.view.warnings.map((warning, index) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              </div>
            )}
            <RawFold raw={assistStatus.raw} />
          </div>
        )}
        <div
          className="flex flex-col gap-2 border-t pt-3"
          style={{ borderColor: "var(--separator)" }}
        >
          <div className="flex items-center gap-2">
            <h3 className="text-xs font-semibold">{m.assist.cfgTitle}</h3>
            <div className="flex-1" />
            <Button onClick={() => initAssistConfigForm()}>
              {m.assist.cfgReload}
            </Button>
          </div>
          <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
            {m.assist.cfgIntro}
          </p>
          {configForm === null ? (
            <div className="text-xs">{m.common.loading}</div>
          ) : (
            <div className="flex flex-col gap-3">
              <label className="flex items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={configForm.enabled}
                  onChange={(event) =>
                    setAssistConfigForm({
                      ...configForm,
                      enabled: event.target.checked,
                    })
                  }
                />
                {m.assist.cfgEnabled}
              </label>

              <table className="w-full text-xs">
                <thead>
                  <tr style={{ color: "var(--fg-secondary)" }}>
                    <th className="pr-2 text-left">{m.assist.cfgName}</th>
                    <th className="pr-2 text-left">{m.assist.cfgModel}</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {configForm.providers.map((provider, providerIndex) => (
                    <Fragment key={providerIndex}>
                      <tr>
                        <td className="pr-2 py-1">
                          <TextInput
                            className="w-full"
                            style={{ fontFamily: "var(--mono)" }}
                            value={provider.name}
                            onChange={(event) =>
                              setAssistConfigForm({
                                ...configForm,
                                providers: configForm.providers.map(
                                  (item, index) =>
                                    index === providerIndex
                                      ? { ...item, name: event.target.value }
                                      : item,
                                ),
                              })
                            }
                          />
                        </td>
                        <td className="pr-2 py-1">
                          <TextInput
                            className="w-full"
                            style={{ fontFamily: "var(--mono)" }}
                            list="assist-model-suggestions"
                            value={provider.model}
                            onChange={(event) =>
                              setAssistConfigForm({
                                ...configForm,
                                providers: configForm.providers.map(
                                  (item, index) =>
                                    index === providerIndex
                                      ? { ...item, model: event.target.value }
                                      : item,
                                ),
                              })
                            }
                          />
                        </td>
                        <td className="py-1 text-right">
                          <Button
                            variant="danger"
                            onClick={() => {
                              const removedName = provider.name;
                              setAssistConfigForm({
                                ...configForm,
                                providers: configForm.providers.filter(
                                  (_, index) => index !== providerIndex,
                                ),
                                roles: {
                                  question_answering:
                                    configForm.roles.question_answering ===
                                    removedName
                                      ? ""
                                      : configForm.roles.question_answering,
                                  deck_design:
                                    configForm.roles.deck_design === removedName
                                      ? ""
                                      : configForm.roles.deck_design,
                                },
                                extraRoles: Object.fromEntries(
                                  Object.entries(configForm.extraRoles).map(
                                    ([role, value]) => [
                                      role,
                                      value === removedName ? "" : value,
                                    ],
                                  ),
                                ),
                              });
                            }}
                          >
                            {m.assist.cfgRemove}
                          </Button>
                        </td>
                      </tr>
                      <tr>
                        <td colSpan={3} className="pb-2">
                          <details>
                            <summary>{m.assist.cfgCommand}</summary>
                            <textarea
                              rows={2}
                              style={TEXTAREA_STYLE}
                              className="w-full rounded border p-2 text-xs"
                              value={provider.command}
                              onChange={(event) =>
                                setAssistConfigForm({
                                  ...configForm,
                                  providers: configForm.providers.map(
                                    (item, index) =>
                                      index === providerIndex
                                        ? {
                                            ...item,
                                            command: event.target.value,
                                          }
                                        : item,
                                  ),
                                })
                              }
                            />
                          </details>
                        </td>
                      </tr>
                    </Fragment>
                  ))}
                </tbody>
              </table>
              <datalist id="assist-model-suggestions">
                {MODEL_SUGGESTIONS.map((model) => (
                  <option key={model} value={model} />
                ))}
              </datalist>

              <div className="flex flex-wrap gap-2">
                <Button
                  onClick={() =>
                    addAssistProviderPreset("claude_readonly")
                  }
                >
                  {m.assist.cfgAddClaudeRO}
                </Button>
                <Button
                  onClick={() => addAssistProviderPreset("codex_readonly")}
                >
                  {m.assist.cfgAddCodexRO}
                </Button>
                <Button onClick={() => addAssistProviderPreset("claude")}>
                  {m.assist.cfgAddClaudeRW}
                </Button>
                <Button onClick={() => addAssistProviderPreset("codex")}>
                  {m.assist.cfgAddCodexRW}
                </Button>
              </div>

              <div className="flex flex-col gap-2">
                <label className="flex items-center gap-2 text-xs">
                  <span className="flex-1">{m.assist.cfgRoleAsk}</span>
                  <Select
                    value={configForm.roles.question_answering}
                    onChange={(event) =>
                      setAssistConfigForm({
                        ...configForm,
                        roles: {
                          ...configForm.roles,
                          question_answering: event.target.value,
                        },
                      })
                    }
                  >
                    <option value="">{m.assist.cfgRoleUnset}</option>
                    {configForm.providers.map((provider, index) => (
                      <option key={`${provider.name}-${index}`} value={provider.name}>
                        {provider.name}
                      </option>
                    ))}
                  </Select>
                </label>
                <label className="flex items-center gap-2 text-xs">
                  <span className="flex-1">{m.assist.cfgRoleDeck}</span>
                  <Select
                    value={configForm.roles.deck_design}
                    onChange={(event) =>
                      setAssistConfigForm({
                        ...configForm,
                        roles: {
                          ...configForm.roles,
                          deck_design: event.target.value,
                        },
                      })
                    }
                  >
                    <option value="">{m.assist.cfgRoleUnset}</option>
                    {configForm.providers.map((provider, index) => (
                      <option key={`${provider.name}-${index}`} value={provider.name}>
                        {provider.name}
                      </option>
                    ))}
                  </Select>
                </label>
              </div>

              <div className="text-xs">
                <span style={{ color: "var(--fg-secondary)" }}>{m.assist.cfgLocation}: </span>
                <span style={{ fontFamily: "var(--mono)" }}>{assistConfig.path ?? "…"}</span>
              </div>
              <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.assist.cfgLocationNote}</p>

              <div className="flex items-center gap-2">
                <Button
                  variant="primary"
                  disabled={assistConfig.status === "saving"}
                  onClick={() => void saveAssistConfig()}
                >
                  {m.assist.cfgSave}
                </Button>
                {assistConfig.status === "saving" && (
                  <span className="text-xs">{m.common.loading}</span>
                )}
                {assistConfig.status === "saved" && (
                  <span className="text-xs">
                    {m.assist.cfgSaved}{" "}
                    <span style={{ fontFamily: "var(--mono)" }}>
                      {assistConfig.savedPath}
                    </span>
                  </span>
                )}
                {assistConfig.status === "error" && (
                  <AssistError text={assistConfig.error ?? ""} />
                )}
              </div>
            </div>
          )}
        </div>
        <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
          {m.assist.setupHint}
        </p>
      </section>

      <section
        className="flex flex-col gap-3 rounded border p-3"
        style={{ borderColor: "var(--separator)", background: "var(--bg-panel)" }}
      >
        <h2 className="text-sm font-semibold">{m.assist.askTitle}</h2>
        <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
          {m.assist.askIntro}
        </p>
        <label className="text-xs" style={{ color: "var(--fg-secondary)" }}>
          {m.assist.askLabel}
        </label>
        <textarea
          rows={4}
          className="w-full rounded border p-2 text-xs"
          style={TEXTAREA_STYLE}
          placeholder={m.assist.askPlaceholder}
          value={assistQuestion}
          onChange={(e) => setAssistQuestion(e.target.value)}
        />
        <div className="flex gap-2">
          <Button
            variant="primary"
            disabled={
              assistAsk.phase === "running" || assistQuestion.trim().length === 0
            }
            onClick={() => void askAssistQuestion()}
          >
            {m.assist.askSend}
          </Button>
          {assistAsk.phase === "running" && (
            <Button variant="danger" onClick={() => void cancelAssistQuestion()}>
              {m.assist.askCancel}
            </Button>
          )}
          {(assistAsk.turns.length > 0 || assistAsk.workdir) && (
            <Button
              disabled={assistAsk.phase === "running"}
              onClick={() => resetAssistConversation()}
            >
              {m.assist.askReset}
            </Button>
          )}
          <Button onClick={() => setChatOpen(true)}>{m.chat.openFromAssistant}</Button>
        </div>
        {assistAsk.phase === "running" && (
          <div className="text-xs" style={{ fontFamily: "var(--mono)" }}>
            {m.assist.askRunning}…
            {assistAsk.lastKind !== null &&
              ` · ${m.assist.lastEvent}: ${assistAsk.lastKind}`}
          </div>
        )}
        {assistAsk.phase === "error" && (
          <div className="flex flex-col gap-2">
            <AssistError
              text={`${assistAsk.errorCode}: ${assistAsk.errorDetail ?? ""}`}
            />
            <RawFold raw={assistAsk.resultRaw ?? undefined} />
          </div>
        )}
        {assistAsk.turns.length > 0 && (
          <div className="flex flex-col gap-3">
            {assistAsk.turns.map((turn, index) => (
              <div
                key={`${turn.turn}-${index}`}
                className="flex flex-col gap-2 rounded border p-3 text-xs"
                style={{ borderColor: "var(--separator)" }}
              >
                <strong>
                  {m.assist.askTurnQ} {turn.turn}
                </strong>
                <pre style={{ whiteSpace: "pre-wrap", fontFamily: "inherit" }}>
                  {turn.question}
                </pre>
                <strong>
                  {m.assist.askTurnA} {turn.turn}
                </strong>
                <div className="text-xs">
                  <MarkdownLite text={turn.answer} />
                </div>
                {turn.checks === null ? (
                  <div>{m.assist.askNoChecks}</div>
                ) : (
                  <div className="flex flex-col gap-1">
                    <div>
                      {m.assist.askCitationsVerified}: {turn.checks.citationsVerified.length}
                    </div>
                    <div>
                      {m.assist.askCitationsUnverified}: {turn.checks.citationsUnverified.length}
                      {turn.checks.citationsUnverified.length > 0 && (
                        <ul>
                          {turn.checks.citationsUnverified.map((item, itemIndex) => (
                            <li key={itemIndex}>
                              {item.citation} — {item.reason}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                    <div>
                      {m.assist.askKeysUnknown}: {turn.checks.keysUnknown.length}
                      {turn.checks.keysUnknown.length > 0 && (
                        <span className="ml-1" style={{ fontFamily: "var(--mono)" }}>
                          {turn.checks.keysUnknown.join(", ")}
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        {assistAsk.workdir && (
          <details className="text-xs">
            <summary style={{ color: "var(--fg-secondary)" }}>
              {m.assist.workdirLabel}
            </summary>
            <div className="mt-1 break-all" style={{ fontFamily: "var(--mono)" }}>
              {assistAsk.workdir}
            </div>
            <p className="mt-1 text-xs" style={{ color: "var(--fg-secondary)" }}>
              {m.assist.journalNote}
            </p>
          </details>
        )}
      </section>

      <section
        className="flex flex-col gap-3 rounded border p-3"
        style={{ borderColor: "var(--separator)", background: "var(--bg-panel)" }}
      >
        <h2 className="text-sm font-semibold">{m.assist.generateTitle}</h2>
        <label className="text-xs" style={{ color: "var(--fg-secondary)" }}>
          {m.assist.specLabel}
        </label>
        <textarea
          rows={10}
          className="w-full rounded border p-2 text-xs"
          style={TEXTAREA_STYLE}
          value={assistSpec}
          onChange={(e) => setAssistSpec(e.target.value)}
          placeholder={m.assist.specPlaceholder}
        />
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <label className="flex items-center gap-1">
            <input
              type="checkbox"
              checked={assistUseTemplate}
              onChange={(e) => setAssistUseTemplate(e.target.checked)}
              disabled={deck.length === 0 || formErrors.length > 0}
            />
            {m.assist.useTemplate}
          </label>
          <label className="flex items-center gap-1">
            {m.assist.maxIters}
            <NumberInput
              className="w-20"
              min={1}
              max={10}
              value={assistMaxIters}
              onChange={(e) => setAssistMaxIters(Number(e.target.value) || 1)}
            />
          </label>
        </div>
        <details className="text-xs">
          <summary style={{ color: "var(--fg-secondary)" }}>{m.assist.advanced}</summary>
          <label className="mt-2 block" style={{ color: "var(--fg-secondary)" }}>
            {m.assist.intentLabel}
          </label>
          <textarea
            rows={4}
            className="mt-1 w-full rounded border p-2 text-xs"
            style={TEXTAREA_STYLE}
            value={assistIntentJson}
            onChange={(e) => setAssistIntentJson(e.target.value)}
            placeholder={m.assist.intentPlaceholder}
          />
        </details>
        <div className="flex gap-2">
          <Button
            variant="primary"
            onClick={() => void generateAssistDeck()}
            disabled={assistGen.phase === "running"}
          >
            {m.assist.generate}
          </Button>
          {assistGen.phase === "running" && (
            <Button variant="danger" onClick={() => void cancelAssistGeneration()}>
              {m.assist.cancel}
            </Button>
          )}
          {(assistGen.phase === "accepted" ||
            assistGen.phase === "uncertain" ||
            assistGen.phase === "error") && (
            <Button onClick={() => resetAssistGeneration()}>{m.assist.reset}</Button>
          )}
        </div>
        {assistGen.phase === "running" && (
          <div className="text-xs" style={{ fontFamily: "var(--mono)" }}>
            {m.assist.running}… {m.assist.iterationsLabel} {assistGen.iterations}
            {assistGen.lastKind !== null &&
              ` · ${m.assist.lastEvent}: ${assistGen.lastKind}`}
          </div>
        )}
        {assistGen.phase === "uncertain" && (
          <div
            className="flex flex-col gap-2 rounded border p-3"
            style={{ borderColor: "var(--warn)" }}
          >
            <h3 className="text-sm font-semibold">{m.assist.questionTitle}</h3>
            <pre className="whitespace-pre-wrap">{assistGen.question ?? ""}</pre>
            <label className="text-xs" style={{ color: "var(--fg-secondary)" }}>
              {m.assist.answerLabel}
            </label>
            <textarea
              rows={3}
              className="w-full rounded border p-2 text-xs"
              style={TEXTAREA_STYLE}
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
            />
            <div>
              <Button
                variant="primary"
                disabled={answer.trim().length === 0}
                onClick={() => {
                  void answerAssistClarification(answer);
                  setAnswer("");
                }}
              >
                {m.assist.answerAndRetry}
              </Button>
            </div>
          </div>
        )}
        {assistGen.phase === "error" && (
          <div className="flex flex-col gap-2">
            <AssistError
              text={`${assistGen.errorCode ?? ""}${assistGen.errorDetail ? `: ${assistGen.errorDetail}` : ""}`}
            />
            {assistGen.lint !== null && (
              <details className="text-xs">
                <summary style={{ color: "var(--fg-secondary)" }}>
                  {m.assist.lintTitle}
                </summary>
                <KvTable obj={assistGen.lint} />
              </details>
            )}
            <RawFold raw={assistGen.resultRaw ?? undefined} />
          </div>
        )}
        {assistGen.phase === "accepted" && (
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <Badge tone="ok">{m.assist.accepted}</Badge>
              <span className="text-xs">
                {m.assist.iterationsLabel}: {assistGen.iterations}
              </span>
            </div>
            <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
              {m.assist.noFormImport}
            </p>
            <div>
              <h3 className="mb-1 text-xs font-semibold">{m.assist.deckTitle}</h3>
              <pre
                className="max-h-80 overflow-auto whitespace-pre-wrap break-all rounded border p-2 text-xs"
                style={{
                  borderColor: "var(--separator)",
                  background: "var(--bg-inset)",
                  fontFamily: "var(--mono)",
                }}
              >
                {deckText}
              </pre>
            </div>
            <div className="flex gap-2">
              <Button
                onClick={() => void validateAssistDeck()}
                disabled={assistDeckValidate.status === "running"}
              >
                {m.assist.validateDeck}
              </Button>
              <Button
                onClick={() =>
                  void saveTextAs(`${assistGen.deckName ?? "assist_deck"}.py`, deckText)
                }
              >
                {m.assist.saveDeck}
              </Button>
              <Button
                variant="primary"
                onClick={() => void runAssistGeneratedDeck()}
                disabled={profile === null || binMissing}
              >
                {m.run.run}
              </Button>
            </div>
            {assistDeckValidate.status === "running" && (
              <div>
                <Badge tone="muted">{m.validate.running}</Badge>
              </div>
            )}
            {assistDeckValidate.status === "ready" && assistDeckValidate.result && (
              <div className="flex flex-col gap-2">
                <div>
                  <Badge tone={assistDeckValidate.result.ok ? "ok" : "err"}>
                    {assistDeckValidate.result.ok ? m.validate.pass : m.validate.fail}
                  </Badge>
                </div>
                {!assistDeckValidate.result.ok && (
                  <div className="flex flex-col gap-1" style={{ fontFamily: "var(--mono)" }}>
                    {assistDeckValidate.result.errors.slice(0, 5).map((error, index) => (
                      <AssistError key={index} text={error} />
                    ))}
                  </div>
                )}
              </div>
            )}
            {assistDeckValidate.sentTo && (
              <div
                className="text-xs"
                style={{ color: "var(--fg-secondary)", fontFamily: "var(--mono)" }}
              >
                {m.validate.sentTo}: {assistDeckValidate.sentTo}
              </div>
            )}
            {assistGen.workdir && (
              <details className="text-xs">
                <summary style={{ color: "var(--fg-secondary)" }}>
                  {m.assist.workdirLabel}
                </summary>
                <div className="mt-1 break-all" style={{ fontFamily: "var(--mono)" }}>
                  {assistGen.workdir}
                </div>
                <p className="mt-1 text-xs" style={{ color: "var(--fg-secondary)" }}>
                  {m.assist.journalNote}
                </p>
              </details>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
