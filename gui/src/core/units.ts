export type UnitKind = "length" | "time" | "temperature" | "power" | "pressure";

/** A user-facing quantity with an explicit unit. */
export interface Q {
  value: number;
  unit: string;
}

/** Conversion factors to the frozen internal system (cgs + eV; power in W). */
const FACTORS: Record<UnitKind, Record<string, number>> = {
  length: { "µm": 1e-4, mm: 0.1, cm: 1 },
  time: { ps: 1e-12, ns: 1e-9, "µs": 1e-6, s: 1 },
  temperature: { eV: 1, keV: 1e3 },
  power: { W: 1, TW: 1e12 },
  pressure: { Mbar: 1e12, GPa: 1e10, "dyn/cm²": 1 },
};

export const UNIT_CHOICES: Record<UnitKind, string[]> = {
  length: ["µm", "mm", "cm"],
  time: ["ps", "ns", "µs", "s"],
  temperature: ["eV", "keV"],
  power: ["W", "TW"],
  pressure: ["Mbar", "GPa", "dyn/cm²"],
};

export const CANONICAL_UNIT: Record<UnitKind, string> = {
  length: "cm",
  time: "s",
  temperature: "eV",
  power: "W",
  pressure: "dyn/cm²",
};

export function toCanonical(q: Q, kind: UnitKind): number {
  const f = FACTORS[kind][q.unit];
  if (f === undefined) throw new Error(`unknown ${kind} unit: ${q.unit}`);
  return q.value * f;
}

export function q(value: number, unit: string): Q {
  return { value, unit };
}

/** Human-readable original-unit annotation (used in generated deck comments). */
export function qLabel(qty: Q): string {
  return `${qty.value} ${qty.unit}`;
}
