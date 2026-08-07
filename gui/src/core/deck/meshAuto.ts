import type { FormState } from "./formState";
import { logOp } from "@tenryu-common/core/applog";
import { toCanonical } from "../units";
import { resolveGeometry2D, resolveShape, stateAt } from "../geometry2d";

export interface AutoSegment {
  rStartCm: number;
  rEndCm: number;
  nr: number;
}

export interface AutoRegion2D {
  rEndCm: number;
  nz: number; // per-region radial cell count (namelist key name)
  rhoRefGcc: number;
}

export interface ZSegment2D {
  zStartCm: number;
  zEndCm: number;
  count: number;
}

function allocateByMass(masses: number[], nTotal: number): number[] | null {
  if (nTotal < masses.length) return null;
  const massTotal = masses.reduce((a, mass) => a + mass, 0);
  const ideal = masses.map((mass) => (nTotal * mass) / massTotal);
  const alloc = ideal.map((x) => Math.max(1, Math.floor(x)));
  let used = alloc.reduce((a, count) => a + count, 0);
  while (used > nTotal) {
    let k = 0;
    for (let j = 1; j < alloc.length; j++) {
      if (alloc[j] > alloc[k]) k = j;
    }
    if (alloc[k] <= 1) return null;
    alloc[k] -= 1;
    used -= 1;
  }
  const order = ideal
    .map((x, j) => ({ j, frac: x - Math.floor(x) }))
    .sort((a, b) => b.frac - a.frac || a.j - b.j);
  let rem = nTotal - used;
  let oi = 0;
  while (rem > 0) {
    alloc[order[oi % order.length].j] += 1;
    rem -= 1;
    oi += 1;
  }
  return alloc;
}

/** Region-derived graded segments: one segment per material region, cells
 *  allocated in proportion to region mass (largest-remainder rounding, at
 *  least 1 cell per region), with optional per-region integer overrides
 *  pinning a region's count while the remaining pool redistributes by mass.
 *  Returns null when the form cannot be resolved (validation covers the
 *  user-facing causes). */
export function computeRegionSegments(f: FormState): AutoSegment[] | null {
  if (f.main.dimension !== "1D_SPH") return null;
  const regions = f.geometry.regions;
  if (regions.length === 0) return null;

  const bounds: number[] = [toCanonical(f.mesh.rMin, "length")];
  for (const reg of regions) {
    const ro = toCanonical(reg.rOuter, "length");
    if (!(ro > bounds[bounds.length - 1])) return null;
    bounds.push(ro);
  }

  const masses: number[] = [];
  for (let i = 0; i < regions.length; i++) {
    const a = bounds[i];
    const b = bounds[i + 1];
    let volume: number;
    if (f.main.geometry1d === "spherical") {
      volume = ((4 * Math.PI) / 3) * (b ** 3 - a ** 3);
    } else if (f.main.geometry1d === "cylindrical") {
      volume = Math.PI * (b ** 2 - a ** 2);
    } else {
      volume = b - a;
    }
    const mass = regions[i].rho * volume;
    if (!(Number.isFinite(mass) && mass > 0)) return null;
    masses.push(mass);
  }

  const overrides: Array<number | null> = regions.map((_, i) => {
    const o = f.mesh.regionNrOverrides[i];
    return typeof o === "number" ? o : null;
  });
  for (const o of overrides) {
    if (o !== null && !(Number.isInteger(o) && o >= 1)) return null;
  }
  const nTotal = f.mesh.nr;
  if (!(Number.isInteger(nTotal) && nTotal >= 1 && nTotal <= 2_000_000)) return null;

  const alloc = new Array<number>(regions.length).fill(0);
  for (let i = 0; i < regions.length; i++) {
    const o = overrides[i];
    if (o !== null) alloc[i] = o;
  }
  const freeIdx = overrides.flatMap((o, i) => (o === null ? [i] : []));
  const fixedSum = overrides.reduce((a: number, o) => a + (o ?? 0), 0);

  if (freeIdx.length > 0) {
    const pool = nTotal - fixedSum;
    if (pool < freeIdx.length) return null;
    const massFree = freeIdx.reduce((a, i) => a + masses[i], 0);
    const ideal = freeIdx.map((i) => (pool * masses[i]) / massFree);
    const base = ideal.map((x) => Math.max(1, Math.floor(x)));
    let used = base.reduce((a, b) => a + b, 0);
    while (used > pool) {
      let k = 0;
      for (let j = 1; j < base.length; j++) {
        if (base[j] > base[k]) k = j;
      }
      if (base[k] <= 1) return null;
      base[k] -= 1;
      used -= 1;
    }
    const order = ideal
      .map((x, j) => ({ j, frac: x - Math.floor(x) }))
      .sort((a, b) => b.frac - a.frac || a.j - b.j);
    let rem = pool - used;
    let oi = 0;
    while (rem > 0) {
      base[order[oi % order.length].j] += 1;
      rem -= 1;
      oi += 1;
    }
    freeIdx.forEach((i, j) => {
      alloc[i] = base[j];
    });
  }

  return regions.map((_, i) => ({ rStartCm: bounds[i], rEndCm: bounds[i + 1], nr: alloc[i] }));
}

/** Region-derived radial auto zoning for 2D meshes. */
export function computeRegionSegments2d(f: FormState): AutoRegion2D[] | null {
  if (f.main.dimension !== "2D_RZ") return null;
  const regions = f.geometry.regions;
  if (regions.length === 0) return null;

  const bounds: number[] = [toCanonical(f.mesh.rMin, "length")];
  for (const reg of regions) {
    const ro = toCanonical(reg.rOuter, "length");
    if (!(ro > bounds[bounds.length - 1])) return null;
    bounds.push(ro);
  }
  const rMax = toCanonical(f.mesh.rMax, "length");
  if (Math.abs(bounds[bounds.length - 1] - rMax) > 1e-12 * Math.abs(rMax)) {
    return null;
  }

  const masses: number[] = [];
  for (let i = 0; i < regions.length; i++) {
    const a = bounds[i];
    const b = bounds[i + 1];
    const volume =
      f.mesh.logicalMesh2d === "spherical_polar_halfplane"
        ? ((4 * Math.PI) / 3) * (b ** 3 - a ** 3)
        : Math.PI * (b ** 2 - a ** 2);
    const mass = regions[i].rho * volume;
    if (!(Number.isFinite(mass) && mass > 0)) return null;
    masses.push(mass);
  }

  const nTotal = f.mesh.nr;
  if (!(Number.isInteger(nTotal) && nTotal >= 1 && nTotal <= 2_000_000)) return null;
  const alloc = allocateByMass(masses, nTotal);
  if (alloc === null) return null;

  return regions.map((region, i) => ({
    rEndCm: bounds[i + 1],
    nz: alloc[i],
    rhoRefGcc: region.rho,
  }));
}

/** Shape-derived radial auto zoning for 2D meshes. */
export function computeShapeRadialRegions(f: FormState): AutoRegion2D[] | null {
  try {
    logOp(
      "computeShapeRadialRegions nr=" +
        f.mesh.nr +
        " shapes=" +
        f.geometry.shapes2d.length,
    );
    if (f.main.dimension !== "2D_RZ") return null;

    const radii: number[] = [];
    let hasOriginCenteredSphericalShape = false;
    for (const shape of f.geometry.shapes2d) {
      if (shape.kind !== "solidSphere" && shape.kind !== "shell") continue;
      if (Math.abs(toCanonical(shape.z0, "length")) !== 0) continue;
      hasOriginCenteredSphericalShape = true;
      if (shape.kind === "shell") {
        const rIn = toCanonical(shape.rIn, "length");
        if (rIn > 0) radii.push(rIn);
      }
      radii.push(toCanonical(shape.radius, "length"));
    }
    if (!hasOriginCenteredSphericalShape) return null;

    const validRadii = radii.filter((radius) => Number.isFinite(radius) && radius > 0);
    if (validRadii.length === 0) return null;
    validRadii.sort((a, b) => a - b);
    const boundaries: number[] = [];
    for (const radius of validRadii) {
      const previous = boundaries[boundaries.length - 1];
      if (
        previous === undefined ||
        Math.abs(radius - previous) > 1e-12 * Math.max(Math.abs(radius), Math.abs(previous))
      ) {
        boundaries.push(radius);
      }
    }
    if (boundaries.length === 0) return null;

    const rMax = toCanonical(f.mesh.rMax, "length");
    if (!(Number.isFinite(rMax) && rMax > 0)) return null;
    const largest = boundaries[boundaries.length - 1];
    if (
      largest < rMax &&
      Math.abs(largest - rMax) > 1e-12 * Math.max(Math.abs(largest), Math.abs(rMax))
    ) {
      boundaries.push(rMax);
    }

    const resolved = resolveGeometry2D(f.geometry.shapes2d, f.geometry.background2d);
    const rhoRefs: number[] = [];
    const masses: number[] = [];
    let rStart = 0;
    for (const rEnd of boundaries) {
      if (!(Number.isFinite(rEnd) && rEnd > rStart)) return null;
      const rho = stateAt(resolved, (rStart + rEnd) / 2, 0).rho;
      const volume =
        f.mesh.logicalMesh2d === "spherical_polar_halfplane"
          ? ((4 * Math.PI) / 3) * (rEnd ** 3 - rStart ** 3)
          : Math.PI * (rEnd ** 2 - rStart ** 2);
      const mass = rho * volume;
      if (!(Number.isFinite(mass) && mass > 0)) return null;
      rhoRefs.push(rho);
      masses.push(mass);
      rStart = rEnd;
    }

    const nTotal = f.mesh.nr;
    if (!(Number.isInteger(nTotal) && nTotal >= 1 && nTotal <= 2_000_000)) return null;
    const alloc = allocateByMass(masses, nTotal);
    if (alloc === null) return null;

    return boundaries.map((rEndCm, i) => ({ rEndCm, nz: alloc[i], rhoRefGcc: rhoRefs[i] }));
  } catch {
    return null;
  }
}

/** Shape-derived axial auto zoning for 2D meshes. */
export function computeShapeZSegments(f: FormState): ZSegment2D[] | null {
  try {
    logOp("computeShapeZSegments nz=" + f.mesh.nz + " segs=" + f.geometry.shapes2d.length);
    if (f.main.dimension !== "2D_RZ") return null;

    const zMin = toCanonical(f.mesh.zMin, "length");
    const zMax = toCanonical(f.mesh.zMax, "length");
    if (!(Number.isFinite(zMin) && Number.isFinite(zMax) && zMax > zMin)) return null;

    const nTotal = f.mesh.nz;
    if (!(Number.isInteger(nTotal) && nTotal >= 1 && nTotal <= 2_000_000)) return null;

    const breakpoints = [zMin, zMax];
    const addBreakpoint = (z: number) => {
      const clipped = Math.max(zMin, Math.min(zMax, z));
      const duplicate = breakpoints.some(
        (existing) =>
          Math.abs(clipped - existing) <=
          1e-12 * Math.max(Math.abs(clipped), Math.abs(existing)),
      );
      if (!duplicate) breakpoints.push(clipped);
    };
    for (const shape of f.geometry.shapes2d) {
      const resolved = resolveShape(shape);
      if (resolved.kind === "solidSphere" || resolved.kind === "shell") {
        addBreakpoint(resolved.z0 - resolved.radius);
        addBreakpoint(resolved.z0 + resolved.radius);
      } else if (resolved.kind === "block") {
        addBreakpoint(Math.min(resolved.z0, resolved.z1));
        addBreakpoint(Math.max(resolved.z0, resolved.z1));
      } else if (resolved.kind === "cone") {
        addBreakpoint(Math.min(resolved.zApex, resolved.zBase));
        addBreakpoint(Math.max(resolved.zApex, resolved.zBase));
      } else if (resolved.vertices.length > 0) {
        addBreakpoint(Math.min(...resolved.vertices.map((vertex) => vertex.z)));
        addBreakpoint(Math.max(...resolved.vertices.map((vertex) => vertex.z)));
      }
    }
    breakpoints.sort((a, b) => a - b);

    const segmentCount = breakpoints.length - 1;
    if (nTotal < segmentCount) return null;
    if (segmentCount === 1) {
      return [{ zStartCm: zMin, zEndCm: zMax, count: nTotal }];
    }

    const rMin = toCanonical(f.mesh.rMin, "length");
    const rMax = toCanonical(f.mesh.rMax, "length");
    const geometry = resolveGeometry2D(f.geometry.shapes2d, f.geometry.background2d);
    const weights: number[] = [];
    for (let i = 0; i < segmentCount; i++) {
      const zStart = breakpoints[i];
      const zEnd = breakpoints[i + 1];
      const zMid = (zStart + zEnd) / 2;
      let radialSum = 0;
      for (let k = 0; k < 16; k++) {
        const r = rMin + ((k + 0.5) * (rMax - rMin)) / 16;
        radialSum += stateAt(geometry, r, zMid).rho * r;
      }
      weights.push((zEnd - zStart) * radialSum);
    }
    const weightTotal = weights.reduce((sum, weight) => sum + weight, 0);
    const allocationWeights =
      weights.some((weight) => !Number.isFinite(weight)) ||
      !Number.isFinite(weightTotal) ||
      weightTotal === 0
        ? breakpoints.slice(1).map((zEnd, i) => zEnd - breakpoints[i])
        : weights;
    const alloc = allocateByMass(allocationWeights, nTotal);
    if (alloc === null) return null;

    return alloc.map((count, i) => ({
      zStartCm: breakpoints[i],
      zEndCm: breakpoints[i + 1],
      count,
    }));
  } catch {
    return null;
  }
}
