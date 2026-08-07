import { useState } from "react";
import { HISTORY_DEFAULT_KEYS, type HistorySeries } from "@tenryu-common/core/results/historyParse";
import { t } from "../i18n";
import { useApp } from "../store";
import { SwitchField } from "./fields";
import { Button } from "@tenryu-common/ui/kit";
import { applyRangePolicy, SvgCartesianFrame } from "@tenryu-common/chart";

const PALETTE = [
  "var(--series-1)",
  "var(--series-2)",
  "var(--series-3)",
  "var(--series-4)",
  "var(--series-5)",
  "var(--series-6)",
  "var(--series-7)",
  "var(--series-8)",
];

const ENERGY_KEYS = [
  "energy/radiation_field",
  "energy/internal_electron",
  "energy/internal_ion",
  "energy/kinetic",
  "energy/laser_deposited",
  "energy/E_total",
];

interface ChartSeries {
  key: string;
  values: Float64Array;
  color: string;
}

function HistoryPlot({
  tNs,
  series,
  logY,
  height,
  axisLabel,
}: {
  tNs: Float64Array;
  series: ChartSeries[];
  logY: boolean;
  height: 120 | 220;
  axisLabel: string;
}) {
  let minTime = Number.POSITIVE_INFINITY;
  let maxTime = Number.NEGATIVE_INFINITY;
  for (const value of tNs) {
    if (Number.isFinite(value)) {
      minTime = Math.min(minTime, value);
      maxTime = Math.max(maxTime, value);
    }
  }
  let maxValue = 0;
  let minPositive = Number.POSITIVE_INFINITY;
  let maxPositive = Number.NEGATIVE_INFINITY;
  for (const item of series) {
    for (const value of item.values) {
      if (!Number.isFinite(value)) continue;
      maxValue = Math.max(maxValue, value);
      if (value > 0) {
        minPositive = Math.min(minPositive, value);
        maxPositive = Math.max(maxPositive, value);
      }
    }
  }
  // Energy history semantics: linear axis is zero-baseline; log axis is the
  // positive data extent in log10 space (explicit policies, council P0-C).
  const yRange: [number, number] = logY
    ? [
        Number.isFinite(minPositive) ? Math.log10(minPositive) : 0,
        Number.isFinite(maxPositive) ? Math.log10(maxPositive) : 1,
      ]
    : applyRangePolicy(0, maxValue > 0 ? maxValue : 1, { kind: "zeroBaseline" });
  return (
    <SvgCartesianFrame
      width={560}
      height={height}
      xRange={[minTime, maxTime]}
      yRange={yRange}
      yLog={logY}
      xLabel={axisLabel}
    >
      {(x, y) => (
        <>
          {series.map((item) => {
            const points: string[] = [];
            for (let i = 0; i < tNs.length; i += 1) {
              const value = item.values[i];
              if (Number.isFinite(tNs[i]) && Number.isFinite(value) && (!logY || value > 0)) {
                points.push(`${x(tNs[i])},${y(logY ? Math.log10(value) : value)}`);
              }
            }
            if (points.length === 0) return null;
            return (
              <polyline key={item.key} points={points.join(" ")} fill="none" stroke={item.color} strokeWidth={1.5} />
            );
          })}
        </>
      )}
    </SvgCartesianFrame>
  );
}

export default function HistoryChart({ data, runName }: { data: HistorySeries; runName?: string }) {
  const [logEnergy, setLogEnergy] = useState(true);
  const saveTextAs = useApp((s) => s.saveTextAs);
  const m = t();
  const energyKeys = ENERGY_KEYS.filter((key) => data.series[key] !== undefined);

  const exportCsv = () => {
    const keys = HISTORY_DEFAULT_KEYS.filter((key) => data.series[key] !== undefined);
    const rows = [["t_s", ...keys].join(",")];
    for (let i = 0; i < data.t.length; i += 1) {
      rows.push([String(data.t[i]), ...keys.map((key) => String(data.series[key][i]))].join(","));
    }
    const csv = `${rows.join("\n")}\n`;
    void saveTextAs(`${runName ?? "run"}_history.csv`, csv);
  };

  if (data.t.length < 2 || energyKeys.length === 0) {
    return <p style={{ color: "var(--fg-secondary)" }}>{m.results.empty}</p>;
  }

  const tNs = Float64Array.from(data.t, (value) => value * 1e9);
  const energySeries = energyKeys.map((key, index) => ({
    key,
    values: data.series[key],
    color: PALETTE[index % PALETTE.length],
  }));
  const conservation = data.series["energy/conservation_error"];
  const dt = data.series.dt;

  return (
    <div>
      <h2 className="mt-3 text-sm font-semibold">{m.results.energyTitle}</h2>
      <SwitchField label={m.results.logScale} checked={logEnergy} onChange={setLogEnergy} />
      <Button onClick={exportCsv}>{m.results.exportCsv}</Button>
      <HistoryPlot
        tNs={tNs}
        series={energySeries}
        logY={logEnergy}
        height={220}
        axisLabel={m.results.axisTimeNs}
      />
      <div className="flex flex-wrap gap-3">
        {energySeries.map((item) => (
          <div className="flex items-center gap-1 text-xs" key={item.key}>
            <div style={{ width: 10, height: 10, background: item.color }} />
            <span>{item.key.replace("energy/", "")}</span>
          </div>
        ))}
      </div>
      {conservation !== undefined && (
        <div>
          <h2 className="mt-3 text-sm font-semibold">{m.results.consTitle}</h2>
          <HistoryPlot
            tNs={tNs}
            series={[{
              key: "energy/conservation_error",
              values: Float64Array.from(conservation, (value) => Math.abs(value)),
              color: PALETTE[0],
            }]}
            logY
            height={120}
            axisLabel={m.results.axisTimeNs}
          />
        </div>
      )}
      {dt !== undefined && (
        <div>
          <h2 className="mt-3 text-sm font-semibold">{m.results.dtTitle}</h2>
          <HistoryPlot
            tNs={tNs}
            series={[{ key: "dt", values: dt, color: PALETTE[0] }]}
            logY
            height={120}
            axisLabel={m.results.axisTimeNs}
          />
        </div>
      )}
    </div>
  );
}
