export type PortDef = {
  portId: number;
  dir: [number, number, number];
  weight: number;
};

export type BeamPreset = {
  id: "gxii" | "omega" | "nif";
  ports: PortDef[];
  recommendedBins: number;
  beamCountLabel: number;
};

export const PAIR_CAP = 65536;

export function expandedPairCount(nPorts: number, bins: number): number {
  const expandedGroups = nPorts * 2 * bins;
  return expandedGroups * (expandedGroups - 1) / 2;
}

const GXII_DIRECTIONS: Array<[number, number, number]> = [
  [0.000000000000000, 0.525731112119134, 0.850650808352040],
  [0.525731112119134, 0.850650808352040, 0.000000000000000],
  [0.850650808352040, 0.000000000000000, 0.525731112119134],
  [0.000000000000000, 0.525731112119134, -0.850650808352040],
  [0.525731112119134, -0.850650808352040, 0.000000000000000],
  [-0.850650808352040, 0.000000000000000, 0.525731112119134],
  [0.000000000000000, -0.525731112119134, 0.850650808352040],
  [-0.525731112119134, 0.850650808352040, 0.000000000000000],
  [0.850650808352040, 0.000000000000000, -0.525731112119134],
  [0.000000000000000, -0.525731112119134, -0.850650808352040],
  [-0.525731112119134, -0.850650808352040, 0.000000000000000],
  [-0.850650808352040, 0.000000000000000, -0.525731112119134],
];

function makeOmegaDirections(): Array<[number, number, number]> {
  const phi = (1 + Math.sqrt(5)) / 2;
  const vertices: Array<[number, number, number]> = [];
  for (const s1 of [-1, 1]) {
    for (const s2 of [-1, 1]) {
      vertices.push(
        [0, s1, s2 * phi],
        [s1, s2 * phi, 0],
        [s2 * phi, 0, s1],
      );
    }
  }

  const edges: Array<[number[], number[]]> = [];
  for (let i = 0; i < vertices.length; i += 1) {
    for (let j = i + 1; j < vertices.length; j += 1) {
      const distanceSquared = vertices[i].reduce(
        (sum, component, axis) => sum + (component - vertices[j][axis]) ** 2,
        0,
      );
      if (Math.abs(distanceSquared - 4) < 1e-12) {
        edges.push([vertices[i], vertices[j]]);
      }
    }
  }
  if (edges.length !== 30) {
    throw new Error(`OMEGA preset construction expected 30 edges, got ${edges.length}`);
  }

  const directions: Array<[number, number, number]> = [];
  for (const [a, b] of edges) {
    for (const fraction of [1 / 3, 2 / 3]) {
      const point = a.map(
        (component, axis) => component + fraction * (b[axis] - component),
      );
      const norm = Math.hypot(...point);
      directions.push(point.map((component) => component / norm) as [number, number, number]);
    }
  }
  if (directions.length !== 60) {
    throw new Error(`OMEGA preset construction expected 60 points, got ${directions.length}`);
  }
  for (let i = 0; i < directions.length; i += 1) {
    for (let j = i + 1; j < directions.length; j += 1) {
      const dot = directions[i].reduce(
        (sum, component, axis) => sum + component * directions[j][axis],
        0,
      );
      if (dot >= 1 - 1e-9) {
        throw new Error("OMEGA preset construction produced duplicate points");
      }
    }
  }
  return directions;
}

function makeNifDirections(): Array<[number, number, number]> {
  // Ring-uniform idealization of NIF's 192 beams grouped into 48 four-beam
  // quad-center ports for 1D spherical modeling.
  const rings: Array<[number, number]> = [
    [23.5, 4],
    [30, 4],
    [44.5, 8],
    [50, 8],
  ];
  const directions: Array<[number, number, number]> = [];
  for (const upperHemisphere of [true, false]) {
    for (const [thetaDeg, count] of rings) {
      const theta = (upperHemisphere ? thetaDeg : 180 - thetaDeg) * Math.PI / 180;
      const offset = upperHemisphere ? 0 : Math.PI / count;
      for (let i = 0; i < count; i += 1) {
        const azimuth = offset + 2 * Math.PI * i / count;
        directions.push([
          Math.sin(theta) * Math.cos(azimuth),
          Math.sin(theta) * Math.sin(azimuth),
          Math.cos(theta),
        ]);
      }
    }
  }
  return directions;
}

function makePorts(
  directions: Array<[number, number, number]>,
): PortDef[] {
  return directions.map((dir, portId) => ({
    portId,
    dir,
    weight: 1 / directions.length,
  }));
}

export const BEAM_PRESETS: Record<"gxii" | "omega" | "nif", BeamPreset> = {
  gxii: {
    id: "gxii",
    ports: makePorts(GXII_DIRECTIONS),
    recommendedBins: 4,
    beamCountLabel: 12,
  },
  omega: {
    id: "omega",
    ports: makePorts(makeOmegaDirections()),
    recommendedBins: 2,
    beamCountLabel: 60,
  },
  nif: {
    id: "nif",
    ports: makePorts(makeNifDirections()),
    recommendedBins: 2,
    beamCountLabel: 192,
  },
};
