export interface MeshPreviewPolar {
  sMax: number;
  kappa: number;
  centerTreatment: string;
  equalMu: boolean;
}

export interface MeshPreviewData {
  dim: number;
  dimension: string;
  logicalMesh2d: string;
  geometry1d: string;
  nr: number;
  nz: number;
  rMin: number;
  rMax: number;
  zMin: number;
  zMax: number;
  polar: MeshPreviewPolar | null;
  rNodes: number[] | null;
  zNodes: number[] | null;
}

const MARKER = "TENRYU-MESH-PREVIEW: ";

function finiteNumberArray(value: unknown, length: number): value is number[] {
  return (
    Array.isArray(value) &&
    value.length === length &&
    value.every((item) => typeof item === "number" && Number.isFinite(item))
  );
}

export function parseMeshPreview(stdout: string): MeshPreviewData | null {
  let payload: string | null = null;
  for (const line of stdout.split(/\r?\n/)) {
    const trimmed = line.trimStart();
    if (trimmed.startsWith(MARKER)) payload = trimmed.slice(MARKER.length);
  }
  if (payload === null) return null;

  try {
    const raw: unknown = JSON.parse(payload);
    if (typeof raw !== "object" || raw === null) return null;
    const value = raw as Record<string, unknown>;
    if (
      typeof value.dim !== "number" ||
      !Number.isFinite(value.dim) ||
      typeof value.nr !== "number" ||
      !Number.isFinite(value.nr) ||
      typeof value.nz !== "number" ||
      !Number.isFinite(value.nz) ||
      typeof value.dimension !== "string" ||
      typeof value.logical_mesh_2d !== "string" ||
      typeof value.geometry_1d !== "string" ||
      typeof value.r_min !== "number" ||
      typeof value.r_max !== "number" ||
      typeof value.z_min !== "number" ||
      typeof value.z_max !== "number"
    ) {
      return null;
    }
    if (
      value.r_nodes !== null &&
      !finiteNumberArray(value.r_nodes, value.nr + 1)
    ) {
      return null;
    }
    if (
      value.z_nodes !== null &&
      !finiteNumberArray(value.z_nodes, value.nz + 1)
    ) {
      return null;
    }

    let polar: MeshPreviewPolar | null;
    if (value.polar === null) {
      polar = null;
    } else {
      if (typeof value.polar !== "object" || value.polar === null) return null;
      const rawPolar = value.polar as Record<string, unknown>;
      if (
        typeof rawPolar.s_max !== "number" ||
        !Number.isFinite(rawPolar.s_max) ||
        typeof rawPolar.kappa !== "number" ||
        !Number.isFinite(rawPolar.kappa) ||
        typeof rawPolar.center_treatment !== "string" ||
        typeof rawPolar.equal_mu !== "boolean"
      ) {
        return null;
      }
      polar = {
        sMax: rawPolar.s_max,
        kappa: rawPolar.kappa,
        centerTreatment: rawPolar.center_treatment,
        equalMu: rawPolar.equal_mu,
      };
    }

    return {
      dim: value.dim,
      dimension: value.dimension,
      logicalMesh2d: value.logical_mesh_2d,
      geometry1d: value.geometry_1d,
      nr: value.nr,
      nz: value.nz,
      rMin: value.r_min,
      rMax: value.r_max,
      zMin: value.z_min,
      zMax: value.z_max,
      polar,
      rNodes: value.r_nodes as number[] | null,
      zNodes: value.z_nodes as number[] | null,
    };
  } catch {
    return null;
  }
}
