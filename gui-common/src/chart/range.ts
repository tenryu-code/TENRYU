// Resolve a chart axis range from the data extent plus an optional user request.
// `dataMin`/`dataMax` are in PLOT space (already log10-transformed when `log`);
// `requested` is in RAW value space. Invalid requested endpoints (non-finite, or
// non-positive under log) are ignored per endpoint; an inverted result falls back
// to the data extent.
export function resolveAxisRange(
  dataMin: number,
  dataMax: number,
  requested: { min: number; max: number } | undefined,
  log: boolean,
): [number, number] {
  let min = dataMin;
  let max = dataMax;
  if (requested !== undefined) {
    const requestedMin = log
      ? (requested.min > 0 ? Math.log10(requested.min) : NaN)
      : requested.min;
    const requestedMax = log
      ? (requested.max > 0 ? Math.log10(requested.max) : NaN)
      : requested.max;
    if (Number.isFinite(requestedMin)) min = requestedMin;
    if (Number.isFinite(requestedMax)) max = requestedMax;
    if (!(min <= max)) {
      min = dataMin;
      max = dataMax;
    }
  }
  return [min, max];
}

// Extent of v over points whose s lies inside [xMin, xMax] (inclusive).
// Returns null when no point falls inside the window.
export function visibleValueExtent(
  points: readonly { s: number; v: number }[],
  xMin: number,
  xMax: number,
): { min: number; max: number } | null {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const point of points) {
    if (point.s < xMin || point.s > xMax) continue;
    min = Math.min(min, point.v);
    max = Math.max(max, point.v);
  }
  return Number.isFinite(min) && Number.isFinite(max) ? { min, max } : null;
}

// Explicit range policies so migrating a chart cannot silently change what its
// axis MEANS (council decision: algorithm unification and domain policy are separate).
export type RangePolicy =
  | { kind: "dataExtent" }
  | { kind: "zeroBaseline" }
  | { kind: "fixed"; min: number; max: number }
  | { kind: "log" };

// Resolve the pre-padding axis range for a data extent under a policy.
// "log" returns the extent unchanged: log-space transformation stays at the caller.
export function applyRangePolicy(
  dataMin: number,
  dataMax: number,
  policy: RangePolicy,
): [number, number] {
  switch (policy.kind) {
    case "dataExtent":
      return [dataMin, dataMax];
    case "zeroBaseline":
      return [Math.min(0, dataMin), Math.max(0, dataMax)];
    case "fixed":
      return [policy.min, policy.max];
    case "log":
      return [dataMin, dataMax];
  }
}
