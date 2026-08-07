import type { FormState } from "./deck/formState";
import { logOp } from "@tenryu-common/core/applog";
import {
  computeRegionSegments2d,
  computeShapeRadialRegions,
  computeShapeZSegments,
} from "./deck/meshAuto";
import { toCanonical } from "./units";
import {
  computeAutoZoneNodes,
  defaultAutoZoneConfig,
  type AutoZoneRegionTS,
} from "./autoZone";
import type { MeshPreviewData } from "./meshPreviewParse";

export function computeLocalMeshPreview(f: FormState): MeshPreviewData | null {
  if (f.main.dimension !== "2D_RZ") return null;

  const isPolar = f.mesh.logicalMesh2d === "spherical_polar_halfplane";
  const s = toCanonical(f.mesh.rMax, "length");
  const data: MeshPreviewData = {
    dim: 2,
    dimension: "2D_RZ",
    logicalMesh2d: f.mesh.logicalMesh2d,
    geometry1d: "spherical",
    nr: f.mesh.nr,
    nz: f.mesh.nz,
    rMin: isPolar ? 0 : toCanonical(f.mesh.rMin, "length"),
    rMax: s,
    zMin: isPolar ? -s : toCanonical(f.mesh.zMin, "length"),
    zMax: isPolar ? s : toCanonical(f.mesh.zMax, "length"),
    polar: isPolar
      ? {
          sMax: s,
          kappa: f.mesh.polarKappa,
          centerTreatment: f.mesh.polarCenterTreatment,
          equalMu: false,
        }
      : null,
    rNodes: null,
    zNodes: null,
  };

  if (f.mesh.radialZoning2d === "regions") {
    try {
      const segments =
        f.geometry.shapes2d.length > 0
          ? computeShapeRadialRegions(f)
          : computeRegionSegments2d(f);
      if (segments === null) return null;
      const regions: AutoZoneRegionTS[] = segments.map((segment) => ({
        rEnd: segment.rEndCm,
        nz: segment.nz,
        rhoRef: segment.rhoRefGcc,
        isVoid: false,
        materialGroup: "",
      }));
      const cfg = defaultAutoZoneConfig();
      cfg.geometryCode = isPolar ? 0 : 1;
      logOp("computeAutoZoneNodes rings=" + regions.length);
      data.rNodes = computeAutoZoneNodes(0, regions, cfg);
      const zSegments = computeShapeZSegments(f);
      if (zSegments !== null && zSegments.length >= 1) {
        const zNodes: number[] = [];
        for (const [segmentIndex, segment] of zSegments.entries()) {
          if (segmentIndex === 0) zNodes.push(segment.zStartCm);
          for (let i = 1; i <= segment.count; i++) {
            zNodes.push(
              segment.zStartCm +
                ((segment.zEndCm - segment.zStartCm) * i) / segment.count,
            );
          }
        }
        data.zNodes = zNodes;
        data.nz = zNodes.length - 1;
      }
    } catch {
      return null;
    }
    data.nr = data.rNodes.length - 1;
  }

  return data;
}
