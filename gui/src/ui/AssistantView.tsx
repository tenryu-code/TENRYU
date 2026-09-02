import { useState } from "react";
import { profileBinMissing } from "@tenryu-common/core/profiles";
import { Badge, Button, NumberInput, TextInput } from "@tenryu-common/ui/kit";
import { t } from "../i18n";
import { currentProfile, useApp } from "../store";
import { AssistError, KvTable, RawFold } from "./AssistBits";

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
  const fetchAssistStatus = useApp((s) => s.fetchAssistStatus);
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
  const binMissing = profile !== null && profileBinMissing(profile);
  const deckText = assistGen.deckText ?? "";

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
        <div className="flex items-center gap-2">
          <label className="text-xs" style={{ color: "var(--fg-secondary)" }}>
            {m.assist.localRepo}
          </label>
          <TextInput
            className="min-w-0 flex-1"
            value={assistLocalRepo}
            onChange={(e) => void setAssistLocalRepo(e.target.value)}
            style={{ fontFamily: "var(--mono)" }}
          />
          <Button
            onClick={() => void fetchAssistStatus()}
            disabled={assistStatus.status === "loading"}
          >
            {m.assist.checkStatus}
          </Button>
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
        <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>
          {m.assist.setupHint}
        </p>
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
