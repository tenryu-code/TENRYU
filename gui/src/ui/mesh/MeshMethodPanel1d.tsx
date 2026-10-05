import { useState } from "react";
import { Button } from "@tenryu-common/ui/kit";
import { t } from "../../i18n";
import { computeMeshNodes1d, mesh1dWarnings, type ResolutionRequirementForm } from "../../core/deck/mesh1d";
import { refinedForConvergence } from "../../core/deck/meshRecommend";
import { validateFormState } from "../../core/deck/formState";
import { useApp } from "../../store";
import { SelectField } from "../fields";
import ExplicitNodesPanel from "./ExplicitNodesPanel";
import LayersPanel from "./LayersPanel";
import RecommendedPanel from "./RecommendedPanel";
import ZoningIntentPanel from "./ZoningIntentPanel";

function sci(x: number): string {
  return x.toExponential(3);
}

/** Mesh.resolution_requirement of the 1D methods beyond uniform and graded. */
function ResolutionRequirementField() {
  const m = t().mesh1d.ui;
  const form = useApp((s) => s.form);
  const update = useApp((s) => s.updateForm);
  const rr = form.mesh.resolutionRequirement;
  const locked = form.mesh.grid1d === "recommended";
  return (
    <div className="flex flex-col gap-1">
      {locked ? (
        <p className="text-xs">{m.rrLabel}: {m.rrEnforce}</p>
      ) : (
        <SelectField
          label={m.rrLabel}
          value={rr.apply}
          options={[
            { value: "default", label: m.rrDefault },
            { value: "report", label: m.rrReport },
            { value: "enforce", label: m.rrEnforce },
          ]}
          onChange={(v) => update((f) => { f.mesh.resolutionRequirement = { ...f.mesh.resolutionRequirement, apply: v as ResolutionRequirementForm["apply"] }; })}
        />
      )}
      {rr.empirical !== null && (
        <div className="flex flex-wrap items-center gap-2 text-xs" style={{ fontFamily: "var(--mono)" }}>
          <span>{m.rrEmpirical(rr.empirical.caseIds.join(", "), sci(rr.empirical.surfaceCeilingGcm2))}</span>
          {!locked && (
            <Button onClick={() => update((f) => { f.mesh.resolutionRequirement = { ...f.mesh.resolutionRequirement, empirical: null }; })}>
              {m.rrRemoveEmpirical}
            </Button>
          )}
        </div>
      )}
    </div>
  );
}

/** The finer member of a convergence pair: surface cells (and every cell) with half the mass. */
export function ConvergencePairPanel() {
  const m = t().mesh1d.ui;
  const form = useApp((s) => s.form);
  const loadForm = useApp((s) => s.loadForm);
  const saveFinePair = useApp((s) => s.saveFinePair);
  const runFinePair = useApp((s) => s.runFinePair);
  const currentProfileId = useApp((s) => s.currentProfileId);
  const [confirming, setConfirming] = useState(false);
  if (form.main.dimension !== "1D_SPH") return null;
  const fine = refinedForConvergence(form);
  const fineOk = validateFormState(fine).length === 0;
  const nodes = computeMeshNodes1d(fine);
  return (
    <div className="mt-3 flex flex-col gap-1">
      <h2 className="text-sm font-semibold">{m.pairTitle}</h2>
      <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.pairHelp}</p>
      <p className="text-xs" style={{ fontFamily: "var(--mono)" }}>
        {nodes !== null
          ? m.pairCells(nodes.length - 1)
          : fine.mesh.grid1d === "zoning_intent"
            ? m.pairCells(fine.mesh.zoningIntent.nCells)
            : m.pairCellsBySolver}
      </p>
      <div className="flex flex-wrap gap-2">
        <Button disabled={!fineOk} onClick={() => void saveFinePair()}>{m.pairSave}</Button>
        <Button disabled={!fineOk || currentProfileId === null} onClick={() => void runFinePair()}>{m.pairRun}</Button>
        {!confirming ? (
          <Button disabled={!fineOk} onClick={() => setConfirming(true)}>{m.pairOpen}</Button>
        ) : (
          <>
            <span className="text-xs" style={{ color: "var(--err)" }}>{m.pairOpenConfirm}</span>
            <Button variant="primary" onClick={() => { loadForm(fine); setConfirming(false); }}>{m.pairOpen}</Button>
            <Button onClick={() => setConfirming(false)}>{t().presets.cancel}</Button>
          </>
        )}
      </div>
    </div>
  );
}

/** Panels of the 1D mesh methods beyond uniform and graded. */
export default function MeshMethodPanel1d() {
  const m = t().mesh1d.ui;
  const form = useApp((s) => s.form);
  const method = form.mesh.grid1d;
  if (form.main.dimension !== "1D_SPH" || method === "uniform" || method === "graded") return null;
  const warnings = mesh1dWarnings(form);
  return (
    <div className="flex flex-col gap-1">
      <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.methodHelp[method]}</p>
      {method === "recommended" && <RecommendedPanel />}
      {method === "layers" && <LayersPanel />}
      {method === "explicit" && <ExplicitNodesPanel />}
      {method === "zoning_intent" && <ZoningIntentPanel />}
      <ResolutionRequirementField />
      {warnings.length > 0 && (
        <ul className="list-disc pl-4 text-xs" style={{ color: "var(--warn)" }}>
          {warnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
