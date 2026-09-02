import h5wasm from "h5wasm";

let seq = 0;

/** Read /opacity/grid/group_bounds_eV from a TMAT h5 buffer; null when absent/invalid. */
export async function parseTmatGroupBounds(buf: ArrayBuffer): Promise<number[] | null> {
  try {
    const { FS } = await h5wasm.ready;
    const name = `tmat-bounds-${seq++}.h5`;
    let file: InstanceType<typeof h5wasm.File> | null = null;
    try {
      FS.writeFile(name, new Uint8Array(buf));
      file = new h5wasm.File(name, "r");
      const dataset = file.get("/opacity/grid/group_bounds_eV");
      if (
        !dataset ||
        !("value" in dataset) ||
        dataset.shape === null ||
        dataset.shape.length !== 1 ||
        dataset.shape[0] < 2 ||
        dataset.shape[0] > 81
      ) {
        return null;
      }
      const bounds = Array.from(dataset.value as ArrayLike<number>);
      if (bounds.length !== dataset.shape[0] || bounds.some((value) => !Number.isFinite(value))) {
        return null;
      }
      if (bounds[0] <= 0) return null;
      for (let i = 1; i < bounds.length; i++) {
        if (bounds[i] <= bounds[i - 1]) return null;
      }
      return bounds;
    } finally {
      file?.close();
      try {
        FS.unlink(name);
      } catch {
        /* best-effort cleanup */
      }
    }
  } catch {
    return null;
  }
}
