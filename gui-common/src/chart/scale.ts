export function paddedRange(min: number, max: number, logarithmic = false): [number, number] {
  if (min !== max) return [min, max];
  const padding = logarithmic ? 0.5 : Math.abs(min) > 0 ? Math.abs(min) * 0.5 : 0.5;
  return [min - padding, max + padding];
}

export function niceTicks(min: number, max: number): number[] {
  const span = max - min;
  if (!(Number.isFinite(span) && span > 0)) return [];
  const rough = span / 5;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const normalized = rough / magnitude;
  const factor = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  const step = factor * magnitude;
  const first = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let value = first; value <= max + step * 1e-12; value += step) {
    ticks.push(value);
  }
  return ticks;
}

// Estimated glyph advance for the 11px chart tick font. 5.4 (~0.49em) under-predicted
// real digit/exponent widths (~0.6em) and let endpoint labels collide on dense axes.
const TICK_CHAR_PX = 6.6;

export function ticksWithEndpoints(
  min: number,
  max: number,
  toPixel: (value: number) => number,
  format: (value: number) => string,
  orientation: "horizontal" | "vertical",
): number[] {
  if (!(Number.isFinite(min) && Number.isFinite(max) && max >= min)) return [];
  if (min === max) return [min];
  const span = max - min;
  const tolerance = span * 1e-12;
  const endpoints = [min, max];
  const interior = niceTicks(min, max).filter((tick) => {
    if (tick <= min + tolerance || tick >= max - tolerance) return false;
    return endpoints.every((endpoint) => {
      const distance = Math.abs(toPixel(tick) - toPixel(endpoint));
      if (orientation === "vertical") return distance > 12;
      const tickWidth = format(tick).length * TICK_CHAR_PX;
      const endpointWidth = format(endpoint).length * TICK_CHAR_PX;
      const tickStart = toPixel(tick) - tickWidth / 2;
      const tickEnd = toPixel(tick) + tickWidth / 2;
      const endpointPixel = toPixel(endpoint);
      const endpointStart = endpoint === min ? endpointPixel : endpointPixel - endpointWidth;
      const endpointEnd = endpoint === min ? endpointPixel + endpointWidth : endpointPixel;
      return tickEnd + 4 < endpointStart || tickStart - 4 > endpointEnd;
    });
  });
  return [min, ...interior, max];
}
