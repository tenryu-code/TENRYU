import { useEffect, useState } from "react";
import { t } from "../i18n";
import { isTerminal, type RunRecord, type RunUiState } from "@tenryu-common/core/runstate";
import { useApp } from "../store";
import { writeScopeHandoff } from "../core/handoff";
import HistoryChart from "./HistoryChart";
import ProfilePanel from "./ProfilePanel";
import { Badge, Button } from "@tenryu-common/ui/kit";

function stateTone(s: RunUiState): "ok" | "err" | "warn" | "muted" {
  switch (s) {
    case "finished":
      return "ok";
    case "failed":
      return "err";
    case "stopped":
    case "stopping":
      return "warn";
    default:
      return "muted";
  }
}

function Bar({ pct }: { pct: number }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full" style={{ background: "var(--bg-inset)" }}>
      <div
        className="h-full rounded-full"
        style={{
          width: `${Math.min(100, Math.max(0, pct))}%`,
          background: "var(--accent)",
          transition: "width 0.5s",
        }}
      />
    </div>
  );
}

function fmtElapsed(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return "-";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  const mm = String(m).padStart(2, "0");
  const ss = String(s).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

function RunCard({ rec, nowSec }: { rec: RunRecord; nowSec: number }) {
  const m = t();
  const stopRun = useApp((s) => s.stopRun);
  const restartRun = useApp((s) => s.restartRun);
  const fetchLog = useApp((s) => s.fetchLog);
  const deleteRunRecord = useApp((s) => s.deleteRunRecord);
  const starting = useApp((s) => s.starting);
  const log = useApp((s) => s.runLogs[rec.id]);
  const rate = useApp((s) => s.runRates[rec.id]);
  const fetchHistory = useApp((s) => s.fetchHistory);
  const hist = useApp((s) => s.histories[rec.id]);
  const [handoffNote, setHandoffNote] = useState<string | null>(null);

  const pct =
    rec.lastProgress !== null
      ? rec.lastProgress.pct
      : rec.state === "finished"
        ? 100
        : 0;
  const elapsed =
    rec.startEpoch !== null ? fmtElapsed((rec.endEpoch ?? nowSec) - rec.startEpoch) : "-";

  return (
    <div className="mb-2 rounded border p-3" style={{ borderColor: "var(--separator)", background: "var(--bg-panel)" }}>
      <div className="mb-1 flex items-center gap-2">
        <span className="font-medium">{rec.name}</span>
        <Badge tone={stateTone(rec.state)}>{m.run.states[rec.state]}</Badge>
        <span className="text-xs" style={{ color: "var(--fg-secondary)" }}>
          {rec.profileName}
        </span>
        <div className="flex-1" />
        {!isTerminal(rec.state) && rec.state !== "launching" && (
          <Button variant="danger" onClick={() => void stopRun(rec.id)} disabled={rec.state === "stopping"}>
            {m.run.stop}
          </Button>
        )}
        {isTerminal(rec.state) && (
          <>
            <Button onClick={() => void restartRun(rec.id)} disabled={starting}>
              {m.run.restartLatest}
            </Button>
            <Button onClick={() => void deleteRunRecord(rec.id)}>{m.run.deleteRecord}</Button>
          </>
        )}
        <Button
          onClick={() => {
            void writeScopeHandoff({ profileId: rec.profileId, profileName: rec.profileName, run: rec.name })
              .then(() => setHandoffNote(m.run.handoffOk))
              .catch((error) => setHandoffNote(String(error)));
          }}
        >
          {m.run.openInScope}
        </Button>
      </div>
      {handoffNote !== null && (
        <div className="mb-1 text-xs" style={{ color: "var(--fg-secondary)" }}>{handoffNote}</div>
      )}
      <Bar pct={pct} />
      <div className="mt-1 flex items-center gap-3 text-xs" style={{ color: "var(--fg-secondary)", fontFamily: "var(--mono)" }}>
        <span>
          t={rec.lastProgress ? rec.lastProgress.t.toExponential(3) : "0"}/{rec.tEnd.toExponential(3)} s
        </span>
        <span>step {rec.lastProgress ? rec.lastProgress.step : 0}</span>
        {rate !== undefined && <span>{rate.toFixed(1)} steps/s</span>}
        <span>
          {m.run.elapsed} {elapsed}
        </span>
        {rec.exitCode !== null && <span>exit={rec.exitCode}</span>}
      </div>
      {rec.launchError && (
        <div className="mt-1 text-xs" style={{ color: "var(--err)", fontFamily: "var(--mono)" }}>
          {m.run.launchFailed}: {rec.launchError}
        </div>
      )}
      {isTerminal(rec.state) && (
        <details className="mt-1 text-xs" onToggle={(e) => {
          if ((e.target as HTMLDetailsElement).open) void fetchHistory(rec.id);
        }}>
          <summary style={{ color: "var(--fg-secondary)" }}>{m.results.button}</summary>
          <div className="mt-1">
            {!hist || hist.status === "loading" ? (
              <p style={{ color: "var(--fg-secondary)" }}>{m.common.loading}</p>
            ) : hist.status === "error" ? (
              <p style={{ color: "var(--err)" }}>{m.results.error}: {hist.error}</p>
            ) : hist.data ? (
              <>
                <HistoryChart data={hist.data} runName={rec.name} />
                <ProfilePanel runId={rec.id} runName={rec.name} />
              </>
            ) : null}
          </div>
        </details>
      )}
      <details className="mt-1 text-xs" onToggle={(e) => {
        if ((e.target as HTMLDetailsElement).open && log === undefined) void fetchLog(rec.id);
      }}>
        <summary style={{ color: "var(--fg-secondary)" }}>{m.run.log}</summary>
        <pre
          className="mt-1 max-h-52 overflow-auto whitespace-pre-wrap break-all rounded border p-2"
          style={{ borderColor: "var(--separator)", background: "var(--bg-inset)" }}
        >
          {log ?? m.common.loading}
        </pre>
      </details>
    </div>
  );
}

export default function HistoryView() {
  const m = t();
  const runs = useApp((s) => s.runs);
  const [nowSec, setNowSec] = useState(() => Math.floor(Date.now() / 1000));
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    const h = setInterval(() => setNowSec(Math.floor(Date.now() / 1000)), 1000);
    return () => clearInterval(h);
  }, []);

  const selected = runs.find((r) => r.id === selectedId) ?? runs[0] ?? null;

  return (
    <div className="grid h-full min-h-0 gap-4" style={{ gridTemplateColumns: "280px minmax(0,1fr)" }}>
      <div className="min-h-0 overflow-auto pr-1">
        <h1 className="mb-2 text-base font-semibold">{m.nav.history}</h1>
        {runs.length === 0 && <p style={{ color: "var(--fg-secondary)" }}>{m.run.empty}</p>}
        {runs.map((r) => {
          const active = selected !== null && r.id === selected.id;
          const pct = r.lastProgress !== null ? r.lastProgress.pct : r.state === "finished" ? 100 : 0;
          return (
            <button
              key={r.id}
              onClick={() => setSelectedId(r.id)}
              aria-current={active ? "true" : undefined}
              className="mb-1 block w-full rounded border p-2 text-left"
              style={{
                borderColor: active ? "var(--selected-fg)" : "var(--separator)",
                background: active ? "var(--selected-bg)" : "var(--bg-panel)",
              }}
            >
              <div className="mb-1 flex items-center gap-2">
                <span className="min-w-0 flex-1 truncate text-[13px] font-medium">{r.name}</span>
                <Badge tone={stateTone(r.state)}>{m.run.states[r.state]}</Badge>
              </div>
              <Bar pct={pct} />
            </button>
          );
        })}
      </div>
      <div className="min-h-0 overflow-auto">
        {selected === null ? (
          <p style={{ color: "var(--fg-secondary)" }}>{m.run.empty}</p>
        ) : (
          <RunCard rec={selected} nowSec={nowSec} />
        )}
      </div>
    </div>
  );
}
