// Deck text of the 1D mesh methods of mesh1d.ts: zoning_intent (directly edited or recommended by
// recommend-mesh) with its resolution requirement, and the explicit node list of the per-layer and
// imported meshes.
import type { FormState } from "./formState";
import {
  computeLayerNodes,
  isMassMeasure,
  meshLayers1d,
  resolvedLayerZoning,
  zoningDensityRegions,
  zoningPins,
  type MeshLayer1d,
} from "./mesh1d";

export class MeshEmitError extends Error {}

/** Shortest decimal that reads back as the same double in Python (round-trip exact): the
 *  recommender's numbers and Studio's node radii must reach the solver unchanged. */
export function pyExact(x: number): string {
  if (!Number.isFinite(x)) throw new MeshEmitError(`non-finite number: ${x}`);
  return String(x);
}

function pyStringLiteral(s: string): string {
  return JSON.stringify(s);
}

function pyBoolLiteral(b: boolean): string {
  return b ? "True" : "False";
}

/** Numbers as indented Python list lines, `perLine` values per line. */
export function numberListLines(values: number[], indent: string, perLine = 6): string[] {
  const lines: string[] = [];
  for (let i = 0; i < values.length; i += perLine) {
    lines.push(indent + values.slice(i, i + perLine).map(pyExact).join(", ") + ",");
  }
  return lines;
}

function emitResolutionRequirement(f: FormState, lines: string[]): void {
  const rr = f.mesh.resolutionRequirement;
  if (rr.apply === "default" && rr.empirical === null) return;
  lines.push("    resolution_requirement=dict(");
  lines.push(`        apply=${pyStringLiteral(rr.apply === "default" ? "report" : rr.apply)},`);
  if (rr.empirical !== null) {
    const e = rr.empirical;
    lines.push("        empirical=dict(");
    lines.push(`            reference_sha256=${pyStringLiteral(e.referenceSha256)},`);
    lines.push(`            case_ids=[${e.caseIds.map(pyStringLiteral).join(", ")}],`);
    lines.push(`            surface_ceiling_g_cm2=${pyExact(e.surfaceCeilingGcm2)},`);
    lines.push(`            reference_apriori_g_cm2=${pyExact(e.referenceAprioriGcm2)},`);
    lines.push("        ),");
  }
  lines.push("    ),");
}

/** Mesh(...) of the "zoning_intent" and "recommended" methods. */
export function emitZoningIntentMesh(f: FormState, rMinText: string, rMaxText: string): string[] {
  const z = f.mesh.zoningIntent;
  const lines: string[] = [];
  const density = zoningDensityRegions(f, z);
  if (density === null && isMassMeasure(z.measure)) {
    throw new MeshEmitError("zoning_intent: the density regions cannot be derived from the initial regions");
  }
  if (f.mesh.grid1d === "recommended") {
    lines.push("# Mesh recommended by tools/assist recommend-mesh (TENRYU Studio); regenerate it in Studio's");
    lines.push("# mesh section after changing the target, the drive or the physics switches.");
  }
  lines.push("Mesh(");
  lines.push(`    r_min=${rMinText}`);
  lines.push(`    r_max=${rMaxText}`);
  lines.push(`    geometry_1d=${pyStringLiteral(f.main.geometry1d)},`);
  lines.push("    zoning_intent=dict(");
  lines.push(`        n_cells=${pyExact(z.nCells)},`);
  lines.push(`        measure=${pyStringLiteral(z.measure)},`);
  if (density !== null && (z.densityRegions.length > 0 || isMassMeasure(z.measure))) {
    lines.push("        density_regions=[");
    for (const region of density) {
      lines.push(`            {"r_end": ${pyExact(region.rEndCm)}, "rho": ${pyExact(region.rho)}},`);
    }
    lines.push("        ],");
  }
  const pins = zoningPins(f, z);
  if (pins.length > 0) {
    lines.push("        pins=[");
    for (const pin of pins) {
      lines.push(`            {"r": ${pyExact(pin.rCm)}, "ratio_jump_allowed": ${pyBoolLiteral(pin.ratioJumpAllowed)}},`);
    }
    lines.push("        ],");
  }
  if (z.profile.length > 0) {
    lines.push("        profile=[");
    for (const point of z.profile) lines.push(`            {"r": ${pyExact(point.rCm)}, "w": ${pyExact(point.w)}},`);
    lines.push("        ],");
  }
  if (z.anchors.length > 0) {
    lines.push("        anchors=[");
    for (const anchor of z.anchors) {
      lines.push(
        `            {"r": ${pyExact(anchor.rCm)}, "half_width": ${pyExact(anchor.halfWidthCm)}, "log_amplitude": ${pyExact(anchor.logAmplitude)}},`,
      );
    }
    lines.push("        ],");
  }
  if (z.bands.length > 0) {
    lines.push("        bands=[");
    for (const band of z.bands) {
      const parts = [
        `"measure_frac_begin": ${pyExact(band.fracBegin)}`,
        `"measure_frac_end": ${pyExact(band.fracEnd)}`,
      ];
      if (band.cellMeasureMin !== null) parts.push(`"cell_measure_min": ${pyExact(band.cellMeasureMin)}`);
      if (band.cellMeasureMax !== null) parts.push(`"cell_measure_max": ${pyExact(band.cellMeasureMax)}`);
      lines.push(`            {${parts.join(", ")}},`);
    }
    lines.push("        ],");
  }
  if (z.extraEventsCm.length > 0) lines.push(`        extra_events=[${z.extraEventsCm.map(pyExact).join(", ")}],`);
  if (z.drMinCm !== null) lines.push(`        dr_min=${pyExact(z.drMinCm)},`);
  if (z.cellMeasureMin !== null) lines.push(`        cell_measure_min=${pyExact(z.cellMeasureMin)},`);
  if (z.cellMeasureMax !== null) lines.push(`        cell_measure_max=${pyExact(z.cellMeasureMax)},`);
  if (z.preferredRatio !== null) lines.push(`        preferred_ratio=${pyExact(z.preferredRatio)},`);
  if (z.ratioHardMax !== null) lines.push(`        ratio_hard_max=${pyExact(z.ratioHardMax)},`);
  if (z.minCellsPerSegment !== null) lines.push(`        min_cells_per_segment=${pyExact(z.minCellsPerSegment)},`);
  lines.push("    ),");
  emitResolutionRequirement(f, lines);
  lines.push(")");
  return lines;
}

function layerLabel(layer: MeshLayer1d): string {
  if (layer.kind === "void") return "VOID padding";
  if (layer.kind === "corona") return `${layer.materialName} corona ramp`;
  return layer.materialName;
}

function spacingLabel(f: FormState, index: number, layers: MeshLayer1d[]): string {
  const spec = resolvedLayerZoning(f, layers)[index];
  if (spec.spacing === "width") return "equal width";
  if (spec.spacing === "mass") return "equal mass";
  const side = spec.fineSide === "both" ? "both ends" : `${spec.fineSide} side`;
  if (spec.ratioSpec === "ratio") return `width ratio ${spec.ratio}, finest at the ${side}`;
  if (spec.ratioSpec === "first_width") return `finest cell ${spec.firstWidthCm * 1.0e4} µm at the ${side}`;
  return `finest cell matched in mass to the neighbouring layer at the ${side}`;
}

/** Node list and comment of the "layers" and "explicit" methods. */
export function explicitMeshNodes(f: FormState): { nodes: number[]; comment: string[] } {
  if (f.mesh.grid1d === "layers") {
    const layers = meshLayers1d(f);
    const result = computeLayerNodes(f, layers);
    if (layers === null || result.edges === null) {
      throw new MeshEmitError("layers: the per-layer mesh cannot be built (see the mesh section)");
    }
    const comment = ["# Mesh nodes [cm] computed by TENRYU Studio from its per-layer table:"];
    layers.forEach((layer, i) => {
      const cells = (result.perLayer[i] as number[]).length - 1;
      comment.push(
        `#   ${layerLabel(layer)} ${pyExact(layer.rLoCm * 1.0e4)}-${pyExact(layer.rHiCm * 1.0e4)} µm: ${cells} cells, ${spacingLabel(f, i, layers)}`,
      );
    });
    return { nodes: result.edges, comment };
  }
  const source = f.mesh.explicitNodes.source.replace(/[\r\n]+/g, " ").trim();
  return {
    nodes: f.mesh.explicitNodes.nodesCm,
    comment: [`# Mesh nodes [cm] imported into TENRYU Studio${source.length > 0 ? ` (${source})` : ""}.`],
  };
}

/** MESH_NODES list and Mesh(...) of the "layers" and "explicit" methods. */
export function emitExplicitMesh(f: FormState): string[] {
  const { nodes, comment } = explicitMeshNodes(f);
  if (nodes.length < 2) throw new MeshEmitError("explicit_nodes: at least two nodes are required");
  const lines = [...comment, "MESH_NODES = ["];
  lines.push(...numberListLines(nodes, "    "));
  lines.push("]");
  lines.push("");
  lines.push("Mesh(");
  lines.push(`    r_min=${pyExact(nodes[0])},`);
  lines.push(`    r_max=${pyExact(nodes[nodes.length - 1])},`);
  lines.push("    explicit_nodes=MESH_NODES,");
  lines.push(`    geometry_1d=${pyStringLiteral(f.main.geometry1d)},`);
  emitResolutionRequirement(f, lines);
  lines.push(")");
  return lines;
}
