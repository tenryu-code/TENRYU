export function formatTick(value: number): string {
  if (value === 0) return "0";
  const absolute = Math.abs(value);
  return absolute >= 1e4 || absolute < 1e-3
    ? value.toExponential(2)
    : String(Number(value.toPrecision(3)));
}

export function formatLogTick(value: number): string {
  const rounded = Math.round(value);
  return Math.abs(value - rounded) < 1e-10
    ? `1e${rounded}`
    : formatTick(10 ** value);
}
