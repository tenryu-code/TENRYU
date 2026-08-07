import { useEffect, useState } from "react";
import { PROFILE_DEFAULT_FIELDS } from "@tenryu-common/core/results/profileParse";
import { t } from "../i18n";
import { useApp } from "../store";
import { SelectField, SwitchField } from "./fields";
import { Button } from "@tenryu-common/ui/kit";
import { applyRangePolicy, SvgCartesianFrame } from "@tenryu-common/chart";

function ProfileChart({
  xUm,
  values,
  logY,
  xLabel,
  yLabel,
}: {
  xUm: Float64Array;
  values: Float64Array;
  logY: boolean;
  xLabel: string;
  yLabel: string;
}) {
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxValue = 0;
  let minPositive = Number.POSITIVE_INFINITY;
  let maxPositive = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < xUm.length; i += 1) {
    const xValue = xUm[i];
    const value = values[i];
    if (Number.isFinite(xValue)) {
      minX = Math.min(minX, xValue);
      maxX = Math.max(maxX, xValue);
    }
    if (!Number.isFinite(value)) continue;
    maxValue = Math.max(maxValue, value);
    if (value > 0) {
      minPositive = Math.min(minPositive, value);
      maxPositive = Math.max(maxPositive, value);
    }
  }
  const minDecade = Number.isFinite(minPositive) ? Math.floor(Math.log10(minPositive)) : 0;
  const maxDecadeRaw = Number.isFinite(maxPositive) ? Math.ceil(Math.log10(maxPositive)) : 1;
  const maxDecade = Math.max(minDecade + 1, maxDecadeRaw);
  // X range policy: dataExtent preserves the finite x-value extent.
  const xRange = applyRangePolicy(minX, maxX, { kind: "dataExtent" });
  // Y range policy: log uses the decade extent in log10 space; linear uses zeroBaseline.
  const yRange = logY
    ? applyRangePolicy(minDecade, maxDecade, { kind: "log" })
    : applyRangePolicy(0, maxValue > 0 ? maxValue : 1, { kind: "zeroBaseline" });

  return (
    <SvgCartesianFrame
      width={560}
      height={220}
      xRange={xRange}
      yRange={yRange}
      yLog={logY}
      xLabel={xLabel}
      yLabel={yLabel}
    >
      {(x, y) => {
        const points: string[] = [];
        for (let i = 0; i < xUm.length; i += 1) {
          const xValue = xUm[i];
          const value = values[i];
          if (Number.isFinite(xValue) && Number.isFinite(value) && (!logY || value > 0)) {
            points.push(`${x(xValue)},${y(logY ? Math.log10(value) : value)}`);
          }
        }
        return points.length > 0 ? (
          <polyline
            points={points.join(" ")}
            fill="none"
            stroke="#4c8dff"
            strokeWidth={1.5}
          />
        ) : null;
      }}
    </SvgCartesianFrame>
  );
}

export default function ProfilePanel({ runId, runName }: { runId: string; runName: string }) {
  const m = t();
  const fetchProfileList = useApp((s) => s.fetchProfileList);
  const fetchProfileSnap = useApp((s) => s.fetchProfileSnap);
  const saveTextAs = useApp((s) => s.saveTextAs);
  const list = useApp((s) => s.profileLists[runId]);
  const [index, setIndex] = useState(0);
  const [field, setField] = useState("hydro/rho");
  const [logY, setLogY] = useState(false);
  const snap = useApp((s) => s.profileSnaps[`${runId}:${index}`]);

  useEffect(() => {
    void fetchProfileList(runId);
  }, [runId, fetchProfileList]);

  useEffect(() => {
    if (list?.status === "ready") void fetchProfileSnap(runId, index);
  }, [list?.status, index, runId, fetchProfileSnap]);

  const fieldOptions = [
    { value: "hydro/rho", label: m.results.fieldRho },
    { value: "hydro/Te", label: m.results.fieldTe },
    { value: "hydro/Ti", label: m.results.fieldTi },
  ];
  const yLabel = fieldOptions.find((option) => option.value === field)?.label ?? field;

  const exportCsv = () => {
    if (snap?.status !== "ready" || !snap.data) return;
    const keys = PROFILE_DEFAULT_FIELDS.filter((key) => snap.data!.fields[key] !== undefined);
    const rows = [
      `# t_s = ${String(snap.data.t)}`,
      ["r_um", ...keys].join(","),
    ];
    for (let i = 0; i < snap.data.xUm.length; i += 1) {
      rows.push([String(snap.data.xUm[i]), ...keys.map((key) => String(snap.data!.fields[key][i]))].join(","));
    }
    const csv = `${rows.join("\n")}\n`;
    void saveTextAs(`${runName}_snap${String(index).padStart(4, "0")}.csv`, csv);
  };

  return (
    <div>
      <h3 className="mt-2 text-xs font-semibold">{m.results.profilesTitle}</h3>
      {!list || list.status === "loading" ? (
        <p style={{ color: "var(--fg-secondary)" }}>{m.common.loading}</p>
      ) : list.status === "error" ? (
        <p style={{ color: "var(--err)" }}>{m.results.profilesEmpty}: {list.error}</p>
      ) : (
        <>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min={0}
              max={list.paths!.length - 1}
              value={index}
              onChange={(e) => setIndex(Number(e.target.value))}
              className="flex-1"
            />
            <span className="text-xs" style={{ fontFamily: "var(--mono)" }}>
              snapshot {index + 1}/{list.paths!.length}
              {snap?.status === "ready" && snap.data
                ? ` t = ${(snap.data.t * 1e9).toPrecision(4)} ns`
                : ""}
            </span>
          </div>
          <SelectField
            label={m.results.profileField}
            value={field}
            options={fieldOptions}
            onChange={setField}
          />
          <Button onClick={exportCsv} disabled={snap?.status !== "ready" || !snap.data}>
            {m.results.exportCsv}
          </Button>
          <SwitchField label={m.results.logScale} checked={logY} onChange={setLogY} />
          {!snap || snap.status === "loading" ? (
            <p style={{ color: "var(--fg-secondary)" }}>{m.common.loading}</p>
          ) : snap.status === "error" ? (
            <p style={{ color: "var(--err)" }}>{snap.error}</p>
          ) : snap.data && snap.data.fields[field] ? (
            <ProfileChart
              xUm={snap.data.xUm}
              values={snap.data.fields[field]}
              logY={logY}
              xLabel={m.results.axisRUm}
              yLabel={yLabel}
            />
          ) : (
            <p style={{ color: "var(--fg-secondary)" }}>{m.results.profilesEmpty}</p>
          )}
        </>
      )}
    </div>
  );
}
