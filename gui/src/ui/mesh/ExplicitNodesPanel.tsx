import { useState } from "react";
import { Button } from "@tenryu-common/ui/kit";
import { t } from "../../i18n";
import { nodesFromFrozenConfig, parseNodeText } from "../../core/deck/mesh1d";
import { q } from "../../core/units";
import { useApp } from "../../store";
import { NumInput, SelectField } from "../fields";

const UNIT_TO_CM: Record<string, number> = { "µm": 1.0e-4, mm: 0.1, cm: 1 };

function fmt(x: number): string {
  return String(Number(x.toPrecision(6)));
}

/** Node list of the "explicit" 1D mesh method: pasted or read from a file, a run or the solver's
 *  preview. Importing sets r_min and r_max to the first and last node. */
export default function ExplicitNodesPanel() {
  const m = t().mesh1d.ui;
  const form = useApp((s) => s.form);
  const update = useApp((s) => s.updateForm);
  const runs = useApp((s) => s.runs);
  const readRunNodes = useApp((s) => s.readRunNodes);
  const meshPreview = useApp((s) => s.meshPreview);
  const [text, setText] = useState("");
  const [unit, setUnit] = useState("µm");
  const [column, setColumn] = useState(0);
  const [message, setMessage] = useState<string | null>(null);
  const [runId, setRunId] = useState<string>("");
  const [reading, setReading] = useState(false);
  const nodes = form.mesh.explicitNodes.nodesCm;

  const apply = (values: number[], source: string) => {
    if (values.length < 2) {
      setMessage(m.explicitParseError.empty);
      return;
    }
    update((f) => {
      f.mesh.explicitNodes = { nodesCm: values, source };
      f.mesh.rMin = q(values[0], "cm");
      f.mesh.rMax = q(values[values.length - 1], "cm");
    });
    setMessage(null);
  };

  const applyText = (body: string, source: string) => {
    const parsed = parseNodeText(body, UNIT_TO_CM[unit], column);
    if ("error" in parsed) {
      setMessage(m.explicitParseError[parsed.error]);
      return;
    }
    apply(parsed.nodes, source);
  };

  const openFile = async () => {
    const picked = await (await import("@tenryu-common/backend"))
      .getBackend()
      .openLocalTextFile(["txt", "csv", "dat", "json"]);
    if (!picked) return;
    if (picked.name.toLowerCase().endsWith(".json")) {
      try {
        const values = nodesFromFrozenConfig(JSON.parse(picked.content));
        if (values === null) setMessage(m.explicitFrozenNoNodes);
        else apply(values, m.sourceFile(picked.name));
      } catch (err) {
        setMessage(String(err));
      }
      return;
    }
    applyText(picked.content, m.sourceFile(picked.name));
  };

  const fromRun = async () => {
    if (runId === "") return;
    setMessage(null);
    setReading(true);
    const result = await readRunNodes(runId);
    setReading(false);
    if ("error" in result) setMessage(m.explicitRunError(result.error));
    else apply(result.nodes, result.source);
  };

  const previewNodes = meshPreview !== null && meshPreview.dim === 1 ? meshPreview.rNodes : null;
  return (
    <div className="flex flex-col gap-1">
      {nodes.length >= 2 ? (
        <p className="text-xs" style={{ fontFamily: "var(--mono)" }}>
          {m.explicitCurrent(nodes.length - 1, fmt(nodes[0] * 1.0e4), fmt(nodes[nodes.length - 1] * 1.0e4), form.mesh.explicitNodes.source)}
        </p>
      ) : (
        <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.explicitEmpty}</p>
      )}
      <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.explicitSetsRange}</p>
      <label className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.explicitPaste}</label>
      <textarea
        rows={5}
        className="w-full rounded border p-2 text-xs"
        style={{ borderColor: "var(--separator)", background: "var(--bg-inset)", color: "var(--fg)", fontFamily: "var(--mono)" }}
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <SelectField
        label={m.explicitUnit}
        value={unit}
        options={["µm", "mm", "cm"].map((u) => ({ value: u, label: u }))}
        onChange={setUnit}
      />
      <NumInput int label={m.explicitColumn} value={column} onChange={(n) => setColumn(n ?? 0)} />
      <div className="flex flex-wrap gap-2">
        <Button variant="primary" disabled={text.trim().length === 0} onClick={() => applyText(text, m.sourcePaste)}>
          {m.explicitApply}
        </Button>
        <Button onClick={() => void openFile()}>{m.explicitOpenFile}</Button>
        {previewNodes !== null && previewNodes !== undefined && (
          <Button onClick={() => apply([...previewNodes], m.sourcePreview)}>{m.explicitFromPreview}</Button>
        )}
      </div>
      {runs.length > 0 && (
        <div className="flex items-end gap-2">
          <SelectField
            label={m.explicitFromRunSelect}
            value={runId}
            options={[{ value: "", label: "—" }, ...runs.map((r) => ({ value: r.id, label: r.name }))]}
            onChange={setRunId}
          />
          <Button disabled={runId === "" || reading} onClick={() => void fromRun()}>{m.explicitFromRun}</Button>
        </div>
      )}
      {reading && <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.explicitReading}</p>}
      {message !== null && <p className="text-xs" style={{ color: "var(--err)" }}>{message}</p>}
    </div>
  );
}
