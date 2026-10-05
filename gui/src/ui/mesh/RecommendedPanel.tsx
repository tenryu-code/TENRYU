import { Badge, Button } from "@tenryu-common/ui/kit";
import { t } from "../../i18n";
import { recommendationConditionsKey } from "../../core/deck/mesh1d";
import { useApp } from "../../store";

function sci(x: number | null): string {
  return x === null ? "—" : x.toExponential(3);
}

/** The mesh recommended by the server's recommend-mesh (tools/assist): run it, read its status. */
export default function RecommendedPanel() {
  const m = t().mesh1d.ui;
  const form = useApp((s) => s.form);
  const update = useApp((s) => s.updateForm);
  const state = useApp((s) => s.meshRecommend);
  const recommendMesh = useApp((s) => s.recommendMesh);
  const meta = form.mesh.recommendation;
  const laser = form.laser.enabled;
  const stale = meta !== null && meta.conditionsKey !== recommendationConditionsKey(form);
  return (
    <div className="flex flex-col gap-1">
      {!laser && <p className="text-xs" style={{ color: "var(--warn)" }}>{m.recNeedsLaser}</p>}
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" disabled={!laser || state.status === "running"} onClick={() => void recommendMesh()}>
          {m.recRun}
        </Button>
        {state.status === "running" && <span className="text-xs">{m.recRunning}</span>}
      </div>
      {state.status === "error" && (
        <div className="text-xs" style={{ color: "var(--err)" }}>
          <div>{m.recErrors[state.error as keyof typeof m.recErrors] ?? state.error}</div>
          {state.detail !== undefined && state.detail.length > 0 && (
            <pre className="whitespace-pre-wrap" style={{ fontFamily: "var(--mono)" }}>{state.detail}</pre>
          )}
        </div>
      )}
      {meta !== null && (
        <div className="rounded border p-2 text-xs" style={{ borderColor: "var(--separator)" }}>
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={meta.status === "validated" ? "ok" : "warn"}>
              {m.statusNames[meta.status as keyof typeof m.statusNames] ?? meta.status}
            </Badge>
            {stale && <Badge tone="err">{m.recStale}</Badge>}
            {meta.edited && <Badge tone="warn">{m.recEdited}</Badge>}
          </div>
          <div style={{ fontFamily: "var(--mono)" }}>
            <div>{m.recSummary(meta.nCells, meta.createdAt)}</div>
            <div>{m.recEvidence(meta.caseIds.join(", "), m.modeNames[meta.mode as keyof typeof m.modeNames] ?? meta.mode)}</div>
            <div>{m.recSurface(sci(meta.surfaceCeilingGcm2), sci(meta.achievedSurfaceGcm2))}</div>
            {meta.binary !== "" && <div>{m.recBinary(meta.binary)}</div>}
          </div>
          {meta.confidence !== "" && <div style={{ color: "var(--fg-secondary)" }}>{meta.confidence}</div>}
          {meta.flags.length > 0 && <div>{m.recFlags}: {meta.flags.join(", ")}</div>}
          {meta.warnings.length > 0 && (
            <ul className="list-disc pl-4" style={{ color: "var(--fg-secondary)" }}>
              {meta.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}
        </div>
      )}
      {meta !== null && form.mesh.grid1d === "recommended" && (
        <div>
          <Button
            onClick={() =>
              update((f) => {
                f.mesh.grid1d = "zoning_intent";
                if (f.mesh.recommendation !== null) f.mesh.recommendation = { ...f.mesh.recommendation, edited: true };
              })
            }
          >
            {m.recEdit}
          </Button>
        </div>
      )}
    </div>
  );
}
