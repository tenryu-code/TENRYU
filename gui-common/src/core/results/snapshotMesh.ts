import h5wasm from "h5wasm";

export type SnapshotMesh2D =
  | {
      kind: "tensor";
      nRNodes: number;
      nZNodes: number;
      xR: Float64Array;
      xZ: Float64Array;
    }
  | {
      kind: "csr";
      xR: Float64Array;
      xZ: Float64Array;
      offsets: Int32Array;
      indices: Int32Array;
    };

let seq = 0;

function flattenNumeric(value: unknown): Float64Array | null {
  const flat: number[] = [];
  const visit = (item: unknown): boolean => {
    if (typeof item === "number") {
      flat.push(item);
      return true;
    }
    if (Array.isArray(item) || ArrayBuffer.isView(item)) {
      for (const entry of Array.from(item as ArrayLike<unknown>)) {
        if (!visit(entry)) return false;
      }
      return true;
    }
    return false;
  };
  return visit(value) ? Float64Array.from(flat) : null;
}

function flattenInt32(value: unknown): Int32Array | null {
  const flat = flattenNumeric(value);
  if (
    flat === null ||
    flat.some(
      (entry) =>
        !Number.isInteger(entry) || entry < -2147483648 || entry > 2147483647,
    )
  ) {
    return null;
  }
  return Int32Array.from(flat);
}

export function validateCsr(
  offsets: Int32Array,
  indices: Int32Array,
  nNodes: number,
): boolean {
  if (offsets.length === 0 || offsets[0] !== 0 || offsets[offsets.length - 1] !== indices.length) {
    return false;
  }
  for (let i = 1; i < offsets.length; i++) {
    if (offsets[i] < offsets[i - 1]) return false;
  }
  for (const index of indices) {
    if (index < 0 || index >= nNodes) return false;
  }
  return true;
}

export async function parseSnapshotMesh(buf: ArrayBuffer): Promise<SnapshotMesh2D | null> {
  try {
    const { FS } = await h5wasm.ready;
    const name = `snapshot-mesh-${seq++}.h5`;
    let file: InstanceType<typeof h5wasm.File> | null = null;
    try {
      FS.writeFile(name, new Uint8Array(buf));
      file = new h5wasm.File(name, "r");
      const xRDataset = file.get("/mesh/x_r");
      const xZDataset = file.get("/mesh/x_z");
      if (
        !xRDataset ||
        !("value" in xRDataset) ||
        !xZDataset ||
        !("value" in xZDataset) ||
        xRDataset.shape === null ||
        xZDataset.shape === null ||
        xRDataset.shape.length !== xZDataset.shape.length ||
        xRDataset.shape.some((size, dimension) => size !== xZDataset.shape?.[dimension])
      ) {
        return null;
      }
      const xR = flattenNumeric(xRDataset.value);
      const xZ = flattenNumeric(xZDataset.value);
      if (xR === null || xZ === null || xR.length !== xZ.length) return null;

      if (xRDataset.shape.length === 2) {
        const nRNodes = xRDataset.shape[0];
        const nZNodes = xRDataset.shape[1];
        if (xR.length !== nRNodes * nZNodes) return null;
        return { kind: "tensor", nRNodes, nZNodes, xR, xZ };
      }

      if (xRDataset.shape.length !== 1 || xR.length !== xRDataset.shape[0]) return null;

      let offsetsDataset = file.get("/mesh/topology/v2/cell_node_csr_offsets");
      let indicesDataset = file.get("/mesh/topology/v2/cell_node_csr_indices");
      if (!offsetsDataset || !indicesDataset) {
        offsetsDataset = file.get("/mesh/topology/v3/cell_node_csr_offsets");
        indicesDataset = file.get("/mesh/topology/v3/cell_node_csr_indices");
      }
      if (
        !offsetsDataset ||
        !("value" in offsetsDataset) ||
        !indicesDataset ||
        !("value" in indicesDataset) ||
        offsetsDataset.shape === null ||
        indicesDataset.shape === null ||
        offsetsDataset.shape.length !== 1 ||
        indicesDataset.shape.length !== 1
      ) {
        return null;
      }
      const offsets = flattenInt32(offsetsDataset.value);
      const indices = flattenInt32(indicesDataset.value);
      if (
        offsets === null ||
        indices === null ||
        offsets.length !== offsetsDataset.shape[0] ||
        indices.length !== indicesDataset.shape[0] ||
        !validateCsr(offsets, indices, xR.length)
      ) {
        return null;
      }
      return { kind: "csr", xR, xZ, offsets, indices };
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
