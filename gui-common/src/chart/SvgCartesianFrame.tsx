import { useId, type ReactNode } from "react";
import { formatLogTick, formatTick } from "./format";
import { paddedRange, ticksWithEndpoints } from "./scale";

// Numeric twin of the CSS token --plot-gutter-left (SVG geometry cannot read CSS vars).
export const PLOT_GUTTER_LEFT = 72;
export const PLOT_MARGIN_RIGHT = 18;
export const PLOT_MARGIN_TOP = 10;
export const PLOT_MARGIN_BOTTOM = 34;

// Closed cartesian frame with canonical ticks/labels and a clip region.
// x/y ranges are in PLOT space (already log10-transformed when log), pre-padding
// is applied here via paddedRange.
export function SvgCartesianFrame({
  width = 640,
  height,
  xRange,
  yRange,
  yLog = false,
  xLabel,
  yLabel,
  children,
}: {
  width?: number;
  height: number;
  xRange: [number, number];
  yRange: [number, number];
  yLog?: boolean;
  xLabel?: string;
  yLabel?: string;
  children: (x: (value: number) => number, y: (value: number) => number, clipId: string) => ReactNode;
}) {
  const clipId = `svg-frame-clip-${useId().replace(/:/g, "")}`;
  const [xMin, xMax] = paddedRange(xRange[0], xRange[1]);
  const [yMin, yMax] = paddedRange(yRange[0], yRange[1], yLog);
  const plotRight = width - PLOT_MARGIN_RIGHT;
  const plotBottom = height - PLOT_MARGIN_BOTTOM;
  const plotWidth = plotRight - PLOT_GUTTER_LEFT;
  const plotHeight = plotBottom - PLOT_MARGIN_TOP;
  const x = (value: number) => PLOT_GUTTER_LEFT + ((value - xMin) / (xMax - xMin)) * plotWidth;
  const y = (value: number) => plotBottom - ((value - yMin) / (yMax - yMin)) * plotHeight;
  const yFormat = yLog ? formatLogTick : formatTick;
  const xTicks = ticksWithEndpoints(xMin, xMax, x, formatTick, "horizontal");
  const yTicks = ticksWithEndpoints(yMin, yMax, y, yFormat, "vertical");
  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%" }}>
      <defs>
        <clipPath id={clipId}>
          <rect x={PLOT_GUTTER_LEFT} y={PLOT_MARGIN_TOP} width={plotWidth} height={plotHeight} />
        </clipPath>
      </defs>
      <g stroke="var(--separator)">
        <line x1={PLOT_GUTTER_LEFT} x2={plotRight} y1={plotBottom} y2={plotBottom} />
        <line x1={plotRight} x2={plotRight} y1={plotBottom} y2={PLOT_MARGIN_TOP} />
        <line x1={plotRight} x2={PLOT_GUTTER_LEFT} y1={PLOT_MARGIN_TOP} y2={PLOT_MARGIN_TOP} />
        <line x1={PLOT_GUTTER_LEFT} x2={PLOT_GUTTER_LEFT} y1={PLOT_MARGIN_TOP} y2={plotBottom} />
      </g>
      {xTicks.map((tick) => (
        <g key={`x-${tick}`}>
          <line x1={x(tick)} x2={x(tick)} y1={plotBottom} y2={plotBottom + 5} stroke="var(--separator)" />
          <text
            x={x(tick)}
            y={plotBottom + 17}
            textAnchor={tick === xMin ? "start" : tick === xMax ? "end" : "middle"}
            style={{ fontSize: "var(--fs-chart-tick)" }}
            fill="var(--fg-secondary)"
          >
            {formatTick(tick)}
          </text>
        </g>
      ))}
      {yTicks.map((tick) => (
        <g key={`y-${tick}`}>
          <line x1={PLOT_GUTTER_LEFT - 5} x2={PLOT_GUTTER_LEFT} y1={y(tick)} y2={y(tick)} stroke="var(--separator)" />
          <text
            x={PLOT_GUTTER_LEFT - 8}
            y={y(tick)}
            dy="0.32em"
            textAnchor="end"
            style={{ fontSize: "var(--fs-chart-tick)" }}
            fill="var(--fg-secondary)"
          >
            {yFormat(tick)}
          </text>
        </g>
      ))}
      <g clipPath={`url(#${clipId})`}>{children(x, y, clipId)}</g>
      {xLabel !== undefined && (
        <text
          x={(PLOT_GUTTER_LEFT + plotRight) / 2}
          y={height - 6}
          textAnchor="middle"
          style={{ fontSize: "var(--fs-chart-axis)" }}
          fill="var(--fg-secondary)"
        >
          {xLabel}
        </text>
      )}
      {yLabel !== undefined && (
        <text
          x={13}
          y={(PLOT_MARGIN_TOP + plotBottom) / 2}
          textAnchor="middle"
          style={{ fontSize: "var(--fs-chart-axis)" }}
          fill="var(--fg-secondary)"
          transform={`rotate(-90 13 ${(PLOT_MARGIN_TOP + plotBottom) / 2})`}
        >
          {yLabel}
        </text>
      )}
    </svg>
  );
}
