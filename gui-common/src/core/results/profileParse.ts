import h5wasm from "h5wasm";

export interface ProfileSnapshot {
  /** snapshot time [s] */
  t: number;
  /** cell centers [µm] */
  xUm: Float64Array;
  /** field name -> cell values (same length as xUm) */
  fields: Record<string, Float64Array>;
  /** requested fields absent in the file */
  missing: string[];
}

export const PROFILE_DEFAULT_FIELDS = ["hydro/rho", "hydro/Te", "hydro/Ti"];

let seq = 0;

export async function parseProfile(
  bytes: Uint8Array,
  keys: string[] = PROFILE_DEFAULT_FIELDS,
): Promise<ProfileSnapshot> {
  const { FS } = await h5wasm.ready;
  const name = `profile-${seq++}.h5`;
  FS.writeFile(name, bytes);
  const file = new h5wasm.File(name, "r");
  try {
    const xDs = file.get("mesh/x_r");
    if (!xDs || !("value" in xDs)) {
      throw new Error("snapshot: required dataset 'mesh/x_r' not found");
    }
    const edges = Float64Array.from(xDs.value as ArrayLike<number>);
    if (edges.length < 2) {
      throw new Error("snapshot: mesh/x_r has fewer than 2 nodes");
    }
    const n = edges.length - 1;
    const xUm = new Float64Array(n);
    for (let i = 0; i < n; i++) xUm[i] = 0.5 * (edges[i] + edges[i + 1]) * 1.0e4;
    let t = Number.NaN;
    const attr = (file.attrs as Record<string, { value?: unknown }>)["t"];
    if (attr && typeof attr.value === "number") {
      t = attr.value;
    } else if (attr && Array.isArray(attr.value) && typeof attr.value[0] === "number") {
      t = attr.value[0];
    } else {
      const tDs = file.get("time_state/t");
      if (tDs && "value" in tDs) {
        const v = tDs.value as ArrayLike<number> | number;
        t = typeof v === "number" ? v : Number(v[0]);
      }
    }
    if (!Number.isFinite(t)) {
      throw new Error("snapshot: time attribute 't' not found");
    }
    const fields: Record<string, Float64Array> = {};
    const missing: string[] = [];
    for (const key of keys) {
      const ds = file.get(key);
      if (!ds || !("value" in ds)) {
        missing.push(key);
        continue;
      }
      const v = Float64Array.from(ds.value as ArrayLike<number>);
      if (v.length === n) {
        fields[key] = v;
      } else {
        missing.push(key);
      }
    }
    return { t, xUm, fields, missing };
  } finally {
    file.close();
    try {
      FS.unlink(name);
    } catch {
      /* best-effort cleanup */
    }
  }
}
