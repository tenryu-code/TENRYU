import { q, toCanonical, type Q } from "./units";

export type Shape2DKind = "solidSphere" | "shell" | "block" | "cone" | "polygon";

export interface ShapeState {
  materialName: string;
  rho: number;
  Te: Q;
  Ti: Q;
}

export interface Shape2D extends ShapeState {
  kind: Shape2DKind;
  label: string;
  z0: Q;
  radius: Q;
  rIn: Q;
  r0: Q;
  r1: Q;
  z1: Q;
  zApex: Q;
  zBase: Q;
  baseRadius: Q;
  vertices: Array<{ r: Q; z: Q }>;
}

export function defaultShape2D(kind: Shape2DKind): Shape2D {
  return {
    kind,
    materialName: "",
    rho: 1.0,
    Te: q(1, "eV"),
    Ti: q(1, "eV"),
    label: "",
    z0: q(0, "µm"),
    radius: q(200, "µm"),
    rIn: q(100, "µm"),
    r0: q(0, "µm"),
    r1: q(200, "µm"),
    z1: q(100, "µm"),
    zApex: q(0, "µm"),
    zBase: q(200, "µm"),
    baseRadius: q(100, "µm"),
    vertices:
      kind === "polygon"
        ? [
            { r: q(0, "µm"), z: q(0, "µm") },
            { r: q(200, "µm"), z: q(0, "µm") },
            { r: q(0, "µm"), z: q(200, "µm") },
          ]
        : [],
  };
}

export interface ResolvedShape2D {
  kind: Shape2DKind;
  materialName: string;
  rho: number;
  TeEV: number;
  TiEV: number;
  z0: number;
  radius: number;
  rIn: number;
  r0: number;
  r1: number;
  z1: number;
  zApex: number;
  zBase: number;
  baseRadius: number;
  vertices: Array<{ r: number; z: number }>;
}

export function resolveShape(s: Shape2D): ResolvedShape2D {
  const num = (v: Q) => {
    const x = toCanonical(v, "length");
    return Number.isFinite(x) ? x : 0;
  };
  const temperature = (v: Q) => {
    const x = toCanonical(v, "temperature");
    return Number.isFinite(x) ? x : 0;
  };
  return {
    kind: s.kind,
    materialName: s.materialName,
    rho: s.rho,
    TeEV: temperature(s.Te),
    TiEV: temperature(s.Ti),
    z0: num(s.z0),
    radius: num(s.radius),
    rIn: num(s.rIn),
    r0: num(s.r0),
    r1: num(s.r1),
    z1: num(s.z1),
    zApex: num(s.zApex),
    zBase: num(s.zBase),
    baseRadius: num(s.baseRadius),
    vertices: s.vertices.map((v) => ({
      r: num(v.r),
      z: num(v.z),
    })),
  };
}

/** Closed outline polylines for a resolved shape, in (r, z) cm pairs; null when the
 *  shape is degenerate (nothing drawable). */
export function shapeOutlinePoints(s: ResolvedShape2D): Array<Array<[number, number]>> | null {
  const semicircle = (radius: number, z0: number): Array<[number, number]> =>
    Array.from({ length: 64 }, (_, i) => {
      const theta = (Math.PI * i) / 63;
      return [radius * Math.sin(theta), z0 + radius * Math.cos(theta)];
    });

  let outlines: Array<Array<[number, number]>>;
  if (s.kind === "solidSphere") {
    outlines = [semicircle(s.radius, s.z0)];
  } else if (s.kind === "shell") {
    outlines = [semicircle(s.radius, s.z0)];
    if (s.rIn > 0) outlines.push(semicircle(s.rIn, s.z0));
  } else if (s.kind === "block") {
    const zLo = Math.min(s.z0, s.z1);
    const zHi = Math.max(s.z0, s.z1);
    outlines = [
      [
        [s.r0, zLo],
        [s.r1, zLo],
        [s.r1, zHi],
        [s.r0, zHi],
        [s.r0, zLo],
      ],
    ];
  } else if (s.kind === "cone") {
    outlines = [
      [
        [0, s.zApex],
        [s.baseRadius, s.zBase],
        [0, s.zBase],
        [0, s.zApex],
      ],
    ];
  } else {
    if (s.vertices.length < 2) return null;
    outlines = [[...s.vertices, s.vertices[0]].map((vertex) => [vertex.r, vertex.z])];
  }

  return outlines.every((outline) =>
    outline.every(([r, z]) => Number.isFinite(r) && Number.isFinite(z)),
  )
    ? outlines
    : null;
}

export function shapeContains(s: ResolvedShape2D, r: number, z: number): boolean {
  if (s.kind === "solidSphere") {
    return Math.hypot(r, z - s.z0) <= s.radius;
  }
  if (s.kind === "shell") {
    const d = Math.hypot(r, z - s.z0);
    return s.rIn <= d && d <= s.radius;
  }
  if (s.kind === "block") {
    return s.r0 <= r && r <= s.r1 && Math.min(s.z0, s.z1) <= z && z <= Math.max(s.z0, s.z1);
  }
  if (s.kind === "cone") {
    const h = s.zBase - s.zApex;
    if (h === 0) return false;
    const t = (z - s.zApex) / h;
    if (t < 0 || t > 1) return false;
    return r <= s.baseRadius * t;
  }
  let inside = false;
  let j = s.vertices.length - 1;
  for (let i = 0; i < s.vertices.length; i++) {
    const vi = s.vertices[i];
    const vj = s.vertices[j];
    if ((vi.z > z) !== (vj.z > z) && r < ((vj.r - vi.r) * (z - vi.z)) / (vj.z - vi.z) + vi.r) {
      inside = !inside;
    }
    j = i;
  }
  return inside;
}

export interface Background extends ShapeState {}

export interface ResolvedGeometry2D {
  shapes: ResolvedShape2D[];
  background: { materialName: string; rho: number; TeEV: number; TiEV: number };
}

export function resolveGeometry2D(shapes: Shape2D[], background: Background): ResolvedGeometry2D {
  return {
    shapes: shapes.map(resolveShape),
    background: {
      materialName: background.materialName,
      rho: background.rho,
      TeEV: toCanonical(background.Te, "temperature"),
      TiEV: toCanonical(background.Ti, "temperature"),
    },
  };
}

export function stateAt(
  g: ResolvedGeometry2D,
  r: number,
  z: number,
): { materialName: string; rho: number; TeEV: number; TiEV: number } {
  for (let i = g.shapes.length - 1; i >= 0; i--) {
    const s = g.shapes[i];
    if (shapeContains(s, r, z)) {
      return { materialName: s.materialName, rho: s.rho, TeEV: s.TeEV, TiEV: s.TiEV };
    }
  }
  return g.background;
}
