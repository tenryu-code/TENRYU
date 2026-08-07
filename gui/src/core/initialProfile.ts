import type { FormState } from "./deck/formState";
import { resolveGeometry2D, stateAt } from "./geometry2d";
import { toCanonical } from "./units";

export type ProfileField = "rho" | "Te" | "Ti";

export interface ProfileSample {
  s: number;
  value: number;
}

export interface ProfileResult {
  samples: ProfileSample[];
  sEnd: number;
  regionEdges: number[];
}

export function sampleInitialProfile(
  f: FormState,
  thetaRad: number,
  field: ProfileField,
  n = 256,
): ProfileResult | null {
  if (f.main.dimension !== "2D_RZ") return null;
  const usePainter =
    f.geometry.shapes2d.length > 0 || f.geometry.background2d.materialName !== "";
  if (!usePainter && f.geometry.regions.length === 0) return null;

  const isPolar = f.mesh.logicalMesh2d === "spherical_polar_halfplane";
  const sin = Math.sin(thetaRad);
  const cos = Math.cos(thetaRad);

  let sEnd: number;
  if (isPolar) {
    sEnd = toCanonical(f.mesh.rMax, "length");
  } else {
    const rMaxCm = toCanonical(f.mesh.rMax, "length");
    const zMinCm = toCanonical(f.mesh.zMin, "length");
    const zMaxCm = toCanonical(f.mesh.zMax, "length");
    const cands: number[] = [];
    if (sin > 1e-12) cands.push(rMaxCm / sin);
    if (cos > 1e-12) cands.push(zMaxCm / cos);
    if (cos < -1e-12) cands.push(zMinCm / cos);
    if (cands.length === 0) return null;
    sEnd = Math.min(...cands);
  }

  if (usePainter) {
    const g = resolveGeometry2D(f.geometry.shapes2d, {
      materialName: f.geometry.background2d.materialName,
      rho: f.geometry.background2d.rho,
      Te: f.geometry.background2d.Te,
      Ti: f.geometry.background2d.Ti,
    });
    const resolvedSamples = Array.from({ length: n }, (_, i) => {
      const s = (i / (n - 1)) * sEnd;
      return { s, state: stateAt(g, s * sin, s * cos) };
    });
    const samples = resolvedSamples.map(({ s, state }) => ({
      s,
      value: field === "rho" ? state.rho : field === "Te" ? state.TeEV : state.TiEV,
    }));
    const regionEdges = resolvedSamples.flatMap((sample, i) =>
      i > 0 && sample.state.materialName !== resolvedSamples[i - 1].state.materialName
        ? [(sample.s + resolvedSamples[i - 1].s) / 2]
        : [],
    );
    return { samples, sEnd, regionEdges };
  }

  const bounds = f.geometry.regions.map((region) => toCanonical(region.rOuter, "length"));
  const values = f.geometry.regions.map((region) => {
    if (field === "rho") return region.rho;
    return toCanonical(field === "Te" ? region.Te : region.Ti, "temperature");
  });
  const samples = Array.from({ length: n }, (_, i) => {
    const s = (i / (n - 1)) * sEnd;
    const rCm = s * sin;
    const coordinate = isPolar ? s : rCm;
    const regionIndex = bounds.findIndex((bound) => coordinate < bound);
    return {
      s,
      value: values[regionIndex === -1 ? values.length - 1 : regionIndex],
    };
  });

  const regionEdges = isPolar
    ? bounds.filter((bound) => bound < sEnd)
    : sin > 1e-12
      ? bounds.map((bound) => bound / sin).filter((s) => s < sEnd)
      : [];

  return { samples, sEnd, regionEdges };
}
