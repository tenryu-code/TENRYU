import h5wasm from "h5wasm";

export interface HistorySeries {
  /** time axis [s] */
  t: Float64Array;
  /** dataset path -> values (same length as t) */
  series: Record<string, Float64Array>;
  /** requested keys that were absent in the file */
  missing: string[];
}

/** v2a chart set (design doc section 6). */
export const HISTORY_DEFAULT_KEYS = [
  "energy/radiation_field",
  "energy/internal_electron",
  "energy/internal_ion",
  "energy/kinetic",
  "energy/laser_deposited",
  "energy/E_total",
  "energy/conservation_error",
  "dt",
];

let seq = 0;

/** Parse a TENRYU history.h5 byte buffer into typed series. */
export async function parseHistory(
  bytes: Uint8Array,
  keys: string[] = HISTORY_DEFAULT_KEYS,
): Promise<HistorySeries> {
  const { FS } = await h5wasm.ready;
  const name = `history-${seq++}.h5`;
  FS.writeFile(name, bytes);
  const file = new h5wasm.File(name, "r");
  try {
    const tDs = file.get("t");
    if (!tDs || !("value" in tDs)) {
      throw new Error("history.h5: required dataset 't' not found");
    }
    const t = Float64Array.from(tDs.value as ArrayLike<number>);
    const series: Record<string, Float64Array> = {};
    const missing: string[] = [];
    for (const key of keys) {
      const ds = file.get(key);
      if (!ds || !("value" in ds)) {
        missing.push(key);
        continue;
      }
      const v = Float64Array.from(ds.value as ArrayLike<number>);
      if (v.length === t.length) {
        series[key] = v;
      } else {
        missing.push(key);
      }
    }
    return { t, series, missing };
  } finally {
    file.close();
    try {
      FS.unlink(name);
    } catch {
      /* best-effort cleanup */
    }
  }
}
