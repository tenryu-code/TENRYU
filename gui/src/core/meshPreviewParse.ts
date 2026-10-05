export interface MeshPreviewPolar {
  sMax: number;
  kappa: number;
  centerTreatment: string;
  equalMu: boolean;
}

/** One band of the solver's resolution requirement (mesh_requirement.bands_recommended). */
export interface MeshRequirementBandView {
  kind: string;
  rLoCm: number;
  rHiCm: number;
  arealMassMaxGcm2: number;
}

/** Outcome of one rule of the requirement check (mesh_requirement.requirement_check). */
export interface MeshRequirementRuleView {
  applicable: boolean;
  nChecked: number;
  nViolations: number;
  maxRatio: number | null;
  worstCell: number;
}

/** The parts of the 1D mesh_requirement JSON (SPECIFICATION 6.4.2, OUTPUT_SCHEMA) the mesh
 *  diagnostics use. */
export interface MeshRequirementView {
  applicable: boolean;
  reason: string;
  apply: string;
  R0Cm: number | null;
  areaCm2: number | null;
  bands: MeshRequirementBandView[];
  drMinAdmissibleCm: number | null;
  ablation: MeshRequirementRuleView | null;
  shock: MeshRequirementRuleView | null;
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
  /** null for 1D, where the solver writes no axial extent. */
  zMin: number | null;
  zMax: number | null;
  polar: MeshPreviewPolar | null;
  rNodes: number[] | null;
  zNodes: number[] | null;
  /** 1D: initial density of each cell as the solver samples it, and its material index. */
  rho0Cells?: number[] | null;
  materialCells?: number[] | null;
  requirement?: MeshRequirementView | null;
}

function finiteOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function ruleView(value: unknown): MeshRequirementRuleView | null {
  if (typeof value !== "object" || value === null) return null;
  const rule = value as Record<string, unknown>;
  const worst = (rule.worst ?? {}) as Record<string, unknown>;
  return {
    applicable: rule.applicable === true,
    nChecked: typeof rule.n_checked === "number" ? rule.n_checked : 0,
    nViolations: typeof rule.n_violations === "number" ? rule.n_violations : 0,
    maxRatio: finiteOrNull(rule.max_ratio),
    worstCell: typeof worst.cell === "number" ? worst.cell : -1,
  };
}

export function parseRequirementView(value: unknown): MeshRequirementView | null {
  if (typeof value !== "object" || value === null) return null;
  const req = value as Record<string, unknown>;
  const params = (req.params ?? {}) as Record<string, unknown>;
  const inputs = (req.inputs ?? {}) as Record<string, unknown>;
  const ablation = (req.ablation ?? {}) as Record<string, unknown>;
  const check = (req.requirement_check ?? null) as Record<string, unknown> | null;
  const bands = Array.isArray(req.bands_recommended)
    ? req.bands_recommended.flatMap((band) => {
        if (typeof band !== "object" || band === null) return [];
        const b = band as Record<string, unknown>;
        const lo = finiteOrNull(b.r_lo_cm);
        const hi = finiteOrNull(b.r_hi_cm);
        const cap = finiteOrNull(b.areal_mass_max_g_cm2);
        if (lo === null || hi === null || cap === null) return [];
        return [{ kind: typeof b.kind === "string" ? b.kind : "", rLoCm: lo, rHiCm: hi, arealMassMaxGcm2: cap }];
      })
    : [];
  return {
    applicable: req.applicable === true,
    reason: typeof req.reason === "string" ? req.reason : "",
    apply: typeof params.apply === "string" ? params.apply : "",
    R0Cm: finiteOrNull(inputs.R0_cm),
    areaCm2: finiteOrNull(inputs.area_cm2),
    bands,
    drMinAdmissibleCm: finiteOrNull(ablation.dr_min_admissible_cm),
    ablation: check === null ? null : ruleView(check.ablation),
    shock: check === null ? null : ruleView(check.shock),
  };
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
      typeof value.r_max !== "number"
    ) {
      return null;
    }
    // A 1D deck has no axial extent: the solver writes z_min and z_max as null there.
    const zMin = typeof value.z_min === "number" ? value.z_min : null;
    const zMax = typeof value.z_max === "number" ? value.z_max : null;
    if (value.dim !== 1 && (zMin === null || zMax === null)) return null;
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
      zMin,
      zMax,
      polar,
      rNodes: value.r_nodes as number[] | null,
      zNodes: value.z_nodes as number[] | null,
      rho0Cells: finiteNumberArray(value.rho0_cells, value.nr) ? value.rho0_cells : null,
      materialCells: finiteNumberArray(value.material_cells, value.nr) ? value.material_cells : null,
      requirement: parseRequirementView(value.mesh_requirement),
    };
  } catch {
    return null;
  }
}
