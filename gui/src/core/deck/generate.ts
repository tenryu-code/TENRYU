import { qLabel, toCanonical, type Q } from "../units";
import { resolveGeometry2D, resolveShape } from "../geometry2d";
import {
  effectiveLaserRaysPerBeam,
  GUI_SCHEMA_VERSION,
  validateFormState,
  type FormState,
} from "./formState";
import {
  computeRegionSegments,
  computeRegionSegments2d,
  computeShapeRadialRegions,
  computeShapeZSegments,
} from "./meshAuto";
import { t } from "../../i18n";
import { BEAM_PRESETS } from "./beamPresets";

export class GeneratorError extends Error {}

export function pyNum(n: number): string {
  if (!Number.isFinite(n)) throw new GeneratorError(`non-finite number: ${n}`);
  // Trim one-ulp float noise (e.g. 300 µm -> 0.030000000000000002 cm) while
  // preserving user-entered precision up to 15 significant digits.
  return String(Number(n.toPrecision(15)));
}

export function pyStr(s: string): string {
  return JSON.stringify(s);
}

export function pyList(ns: number[]): string {
  return "[" + ns.map(pyNum).join(", ") + "]";
}

export function pyBool(b: boolean): string {
  return b ? "True" : "False";
}

const cm = (q: Q) => toCanonical(q, "length");
const sec = (q: Q) => toCanonical(q, "time");
const eV = (q: Q) => toCanonical(q, "temperature");
const watt = (q: Q) => toCanonical(q, "power");

/** `<value>,` plus an original-unit comment when the input unit is not canonical. */
function numC(value: number, orig: Q): string {
  const canonical = { length: "cm", time: "s", temperature: "eV", power: "W" } as const;
  const isCanonical = Object.values(canonical).includes(orig.unit as never);
  return isCanonical ? `${pyNum(value)},` : `${pyNum(value)},  # ${qLabel(orig)}`;
}

/** One-line chained-if piecewise-constant function of one or two coordinates. */
function emitPiecewise(
  fnName: string,
  argList: string,
  cmpArg: string,
  innerBoundsCm: number[],
  values: number[],
): string[] {
  if (values.length !== innerBoundsCm.length + 1) {
    throw new GeneratorError(`emitPiecewise: ${values.length} values vs ${innerBoundsCm.length} bounds`);
  }
  const lines = [`def ${fnName}(${argList}):`];
  const comparison = cmpArg.includes("(") ? "_s" : cmpArg;
  if (comparison === "_s") {
    lines.push(`    _s = ${cmpArg}`);
  }
  for (let i = 0; i < innerBoundsCm.length; i++) {
    lines.push(`    if ${comparison} < ${pyNum(innerBoundsCm[i])}: return ${pyNum(values[i])}`);
  }
  lines.push(`    return ${pyNum(values[values.length - 1])}`);
  return lines;
}

export function generateDeck(f: FormState): string {
  const errs = validateFormState(f);
  if (errs.length > 0) {
    throw new GeneratorError(t().validation.formHasErrors + "\n- " + errs.join("\n- "));
  }
  const is2d = f.main.dimension === "2D_RZ";
  const usePolarInBox = is2d && f.mesh.meshMode2d === "polar_in_box";
  const useShapePainter =
    is2d &&
    (f.geometry.shapes2d.length > 0 ||
      (f.geometry.regions.length === 0 &&
        f.materials.some((material) => material.name === f.geometry.background2d.materialName)));
  const useVacuumOutside1d = !is2d && f.geometry.vacuumOutside1d;
  const useCoronaRamp1d = useVacuumOutside1d && f.geometry.coronaRamp1d.enabled;
  const L: string[] = [];

  // --- header -------------------------------------------------------------
  L.push("# TENRYU Studio generated deck");
  L.push(`# TENRYU-GUI-STATE: ${JSON.stringify(f)}`);
  L.push("# Units: cgs + eV (length cm / time s / temperature eV / density g/cm3 / power W)");
  if (usePolarInBox) {
    L.push("import os, sys");
    L.push("def _tenryu_repo_root():");
    L.push("    cands = [os.environ.get(\"TENRYU_REPO\", \"\"), os.getcwd()]");
    L.push("    d = os.path.dirname(os.path.abspath(__file__))");
    L.push("    for _ in range(8):");
    L.push("        cands.append(d)");
    L.push("        d = os.path.dirname(d)");
    L.push("    for c in cands:");
    L.push("        if c and os.path.isfile(os.path.join(c, \"tools\", \"mesh_planner.py\")):");
    L.push("            return c");
    L.push("    raise RuntimeError(\"TENRYU repo root not found (tools/mesh_planner.py); set TENRYU_REPO or run from the checkout root\")");
    L.push("sys.path.insert(0, _tenryu_repo_root())");
    L.push("from tools.mesh_planner import (");
    L.push("    PolarSolidSphere, PolarShell, PolarBase, Quality, SmoothZoning, plan_mesh,");
    L.push(")");
  }
  L.push("from tenryu_namelist import *");
  if ((f.laser.enabled && f.laser.waveformMode === "gaussian") || useShapePainter || useCoronaRamp1d) {
    L.push("import math");
  }
  L.push("");

  // --- Main ----------------------------------------------------------------
  L.push("Main(");
  L.push(`    name=${pyStr(f.main.name)},`);
  L.push(`    dimension=${pyStr(f.main.dimension)},`);
  L.push(`    temperature_model=${pyStr(f.main.temperatureModel)},`);
  L.push(`    t_end=${numC(sec(f.main.tEnd), f.main.tEnd)}`);
  L.push(`    seed=${pyNum(f.main.seed)},`);
  L.push(`    max_steps=${pyNum(f.main.maxSteps)},`);
  L.push('    verbosity="normal",');
  L.push(")");
  L.push("");

  // --- Mesh ----------------------------------------------------------------
  if (usePolarInBox) {
    const originFamily = f.geometry.shapes2d
      .map(resolveShape)
      .filter(
        (shape) =>
          (shape.kind === "solidSphere" || shape.kind === "shell") &&
          Math.abs(shape.z0) <= 1e-12,
      )
      .map((shape) => ({
        inner: shape.kind === "solidSphere" ? 0 : shape.rIn,
        outer: shape.radius,
        rho: shape.rho,
        materialName: shape.materialName,
      }))
      .sort((a, b) => a.inner - b.inner || a.outer - b.outer);
    const sTarget = Math.max(...originFamily.map((interval) => interval.outer));
    L.push("# non-spherical shapes paint materials only; the mesh follows the origin family");
    L.push("PRIMITIVES = [");
    for (const [i, interval] of originFamily.entries()) {
      if (i === 0) {
        L.push(
          `    PolarSolidSphere(r=${pyNum(interval.outer)}, rho=${pyNum(interval.rho)}, material=${pyStr(interval.materialName)}),`,
        );
      } else {
        L.push(
          `    PolarShell(${pyNum(interval.inner)}, ${pyNum(interval.outer)}, ${pyNum(interval.rho)}, ${pyStr(interval.materialName)}),`,
        );
      }
    }
    L.push("]");
    L.push(
      `BASE = PolarBase(s_max=${pyNum(sTarget)}, center_treatment="tri_fan",`,
    );
    L.push(
      `                 center_mode="graded_button", n_theta_default=${pyNum(f.mesh.pibNTheta)})`,
    );
    L.push(
      `PLAN = plan_mesh(PRIMITIVES, BASE, Quality(), SmoothZoning(n_total=${pyNum(f.mesh.pibNRadial)}))`,
    );
    L.push("");
    L.push('EXPLICIT_NODES = list(PLAN.mesh_kwargs["explicit_nodes"])');
    L.push("d = EXPLICIT_NODES[-1] - EXPLICIT_NODES[-2]");
    L.push(`for exponent in range(1, ${pyNum(f.mesh.pibTailRings)}+1):`);
    L.push(
      `    EXPLICIT_NODES.append(EXPLICIT_NODES[-1] + d * ${pyNum(f.mesh.pibTailRatio)}**exponent)`,
    );
    L.push("");
    L.push("MESH_KW = dict(PLAN.mesh_kwargs)");
    L.push('MESH_KW.pop("nr", None)');
    L.push("MESH_KW.update(dict(");
    L.push('    logical_mesh_2d="polar_in_box",');
    L.push("    spherical_polar_s_max=EXPLICIT_NODES[-1],");
    L.push("    explicit_nodes=EXPLICIT_NODES,");
    L.push(
      `    box_r_max=${pyNum(cm(f.mesh.rMax))}, box_z_min=${pyNum(cm(f.mesh.zMin))}, box_z_max=${pyNum(cm(f.mesh.zMax))}, box_center_z=0.0,`,
    );
    L.push(
      `    morph_rings=${pyNum(f.mesh.pibMorphRings)}, collar_rings=${pyNum(f.mesh.pibCollarRings)}, morph_growth_max=${pyNum(f.mesh.pibMorphGrowthMax)},`,
    );
    L.push(`    nz=${pyNum(f.mesh.pibNTheta)},`);
    L.push('    grid="uniform", motion="lagrangian",');
    L.push("))");
    L.push("Mesh(**MESH_KW)");
  } else {
    L.push("Mesh(");
    L.push(`    r_min=${numC(cm(f.mesh.rMin), f.mesh.rMin)}`);
    L.push(`    r_max=${numC(cm(f.mesh.rMax), f.mesh.rMax)}`);
  if (is2d) {
    L.push(`    z_min=${numC(cm(f.mesh.zMin), f.mesh.zMin)}`);
    L.push(`    z_max=${numC(cm(f.mesh.zMax), f.mesh.zMax)}`);
    L.push('    grid="uniform",');
    L.push('    motion="lagrangian",');
    if (f.mesh.radialZoning2d === "regions") {
      const auto =
        f.geometry.shapes2d.length > 0 ? computeShapeRadialRegions(f) : computeRegionSegments2d(f);
      if (auto === null) throw new GeneratorError(t().validation.meshAutoUnresolvable);
      const zsegs = computeShapeZSegments(f);
      if (zsegs === null) throw new GeneratorError(t().validation.meshAutoUnresolvable);
      L.push("    auto_regions=[");
      for (const region of auto) {
        L.push(
          `        dict(r_end=${pyNum(region.rEndCm)}, nz=${pyNum(region.nz)}, rho_ref=${pyNum(region.rhoRefGcc)}),`,
        );
      }
      L.push("    ],");
      if (zsegs.length === 1) {
        L.push(`    nz=${pyNum(f.mesh.nz)},`);
      } else {
        L.push("    grid_z=dict(");
        L.push('        type="graded",');
        L.push("        grading=dict(edge_ratio=0.9999999999999999),");
        L.push("        segments=[");
        for (const segment of zsegs) {
          L.push(
            `            dict(r_start=${pyNum(segment.zStartCm)}, r_end=${pyNum(segment.zEndCm)}, nr=${pyNum(segment.count)}),`,
          );
        }
        L.push("        ],");
        L.push("    ),");
        L.push(`    nz=${pyNum(zsegs.reduce((sum, segment) => sum + segment.count, 0))},`);
      }
    } else {
      L.push(`    nr=${pyNum(f.mesh.nr)},`);
      L.push(`    nz=${pyNum(f.mesh.nz)},`);
    }
  } else if (f.mesh.grid1d === "graded") {
    let cmSegs: Array<{ rStart: number; rEnd: number; nr: number }>;
    if (f.mesh.segmentSource === "regions") {
      const auto = computeRegionSegments(f);
      if (auto === null) throw new GeneratorError(t().validation.meshAutoUnresolvable);
      cmSegs = auto.map((s) => ({ rStart: s.rStartCm, rEnd: s.rEndCm, nr: s.nr }));
    } else {
      cmSegs = [];
      let prev = cm(f.mesh.rMin);
      for (const s of f.mesh.segments) {
        const e = cm(s.rEnd);
        cmSegs.push({ rStart: prev, rEnd: e, nr: s.nr });
        prev = e;
      }
    }
    const total = cmSegs.reduce((a, s) => a + s.nr, 0);
    L.push(`    nr=${pyNum(total)},`);
    L.push("    grid=dict(");
    L.push('        type="graded",');
    if (f.mesh.segmentSource === "regions") {
      L.push("        # segments auto-derived from material regions (mass-proportional cell allocation)");
    }
    L.push("        segments=[");
    for (const s of cmSegs) {
      L.push(
        `            {"r_start": ${pyNum(s.rStart)}, "r_end": ${pyNum(s.rEnd)}, "nr": ${pyNum(s.nr)}},`,
      );
    }
    L.push("        ],");
    const g = f.mesh.grading;
    L.push(
      `        grading=dict(edge_ratio=${pyNum(g.edgeRatio)}, sg_order=${pyNum(g.sgOrder)}, sg_sigma=${pyNum(g.sgSigma)}),`,
    );
    L.push("    ),");
    L.push(`    geometry_1d=${pyStr(f.main.geometry1d)},`);
  } else {
    L.push(`    nr=${pyNum(f.mesh.nr)},`);
    L.push('    grid="uniform",');
    L.push(`    geometry_1d=${pyStr(f.main.geometry1d)},`);
  }
  L.push(")");
  }
  L.push("");

  // --- Materials -------------------------------------------------------------
  L.push("Materials(");
  L.push("    materials=[");
  for (const m of f.materials) {
    L.push("        Material(");
    L.push(`            name=${pyStr(m.name)},`);
    L.push(`            A=${pyNum(m.A)},`);
    L.push(`            Z=${pyNum(m.Z)},`);
    if (m.eosModel === "tmat") {
      L.push(`            eos=dict(model="tmat", file=${pyStr(m.eosFile.trim())}),`);
    } else {
      const cv = m.cvEOverride !== undefined && m.cvEOverride !== null ? `, cv_e_override=${pyNum(m.cvEOverride)}` : "";
      L.push(`            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=${pyNum(m.gamma)})${cv}),`);
    }
    if (m.opacityModel === "tmat") {
      L.push(`            opacity=dict(model="tmat", file=${pyStr(m.opacityFile.trim())}),`);
    } else {
      L.push(
        `            opacity=dict(model="constant", kappa_a=${pyNum(m.kappaA)}, kappa_s=${pyNum(m.kappaS)}, units="cm2_per_g"),`,
      );
    }
    L.push("        ),");
  }
  if (useVacuumOutside1d) {
    L.push('        Material(name="VOID", A=1.0, Z=1.0, is_void=True),');
  }
  L.push("    ],");
  if (useVacuumOutside1d) {
    L.push("    void_config=dict(rho=1.0e-10, Te=1.0e-3, Ti=1.0e-3),");
  }
  if (f.zbarFixedValue !== null) {
    L.push(`    zbar=dict(model="fixed", fixed_value=${pyNum(f.zbarFixedValue)}),`);
  }
  L.push(")");
  L.push("");

  // --- Geometry (piecewise-constant radial regions) ---------------------------
  const args = is2d ? "r_cm, z_cm" : "r_cm";
  const cmpVar = "r_cm";
  const regions = f.geometry.regions;
  const innerBounds = regions.slice(0, -1).map((r) => cm(r.rOuter));
  if (useShapePainter) {
    const geometry = resolveGeometry2D(f.geometry.shapes2d, f.geometry.background2d);
    L.push("_GUI_SHAPES = [");
    for (const shape of geometry.shapes) {
      let params: string;
      if (shape.kind === "solidSphere") {
        params = `{"z0": ${pyNum(shape.z0)}, "radius": ${pyNum(shape.radius)}}`;
      } else if (shape.kind === "shell") {
        params = `{"z0": ${pyNum(shape.z0)}, "rIn": ${pyNum(shape.rIn)}, "radius": ${pyNum(shape.radius)}}`;
      } else if (shape.kind === "block") {
        params = `{"r0": ${pyNum(shape.r0)}, "r1": ${pyNum(shape.r1)}, "z0": ${pyNum(shape.z0)}, "z1": ${pyNum(shape.z1)}}`;
      } else if (shape.kind === "cone") {
        params = `{"zApex": ${pyNum(shape.zApex)}, "zBase": ${pyNum(shape.zBase)}, "baseRadius": ${pyNum(shape.baseRadius)}}`;
      } else {
        const vertices = shape.vertices.map((vertex) => `(${pyNum(vertex.r)}, ${pyNum(vertex.z)})`).join(", ");
        params = `{"vertices": [${vertices}]}`;
      }
      L.push(
        `    (${pyStr(shape.kind)}, ${pyStr(shape.materialName)}, ${pyNum(shape.rho)}, ${pyNum(shape.TeEV)}, ${pyNum(shape.TiEV)}, ${params}),`,
      );
    }
    L.push("]");
    L.push(
      `_GUI_BG = (${pyStr(geometry.background.materialName)}, ${pyNum(geometry.background.rho)}, ${pyNum(geometry.background.TeEV)}, ${pyNum(geometry.background.TiEV)})`,
    );
    L.push("");
    L.push("def _gui_shape_contains(kind, p, r_cm, z_cm):");
    L.push('    if kind == "solidSphere":');
    L.push('        return math.hypot(r_cm, z_cm - p["z0"]) <= p["radius"]');
    L.push('    if kind == "shell":');
    L.push('        d = math.hypot(r_cm, z_cm - p["z0"])');
    L.push('        return p["rIn"] <= d <= p["radius"]');
    L.push('    if kind == "block":');
    L.push('        zlo = min(p["z0"], p["z1"]); zhi = max(p["z0"], p["z1"])');
    L.push('        return p["r0"] <= r_cm <= p["r1"] and zlo <= z_cm <= zhi');
    L.push('    if kind == "cone":');
    L.push('        h = p["zBase"] - p["zApex"]');
    L.push("        if h == 0.0: return False");
    L.push('        t = (z_cm - p["zApex"]) / h');
    L.push('        return 0.0 <= t <= 1.0 and r_cm <= p["baseRadius"] * t');
    L.push('    if kind == "polygon":');
    L.push('        vs = p["vertices"]; inside = False');
    L.push("        j = len(vs) - 1");
    L.push("        for i in range(len(vs)):");
    L.push("            ri, zi = vs[i]; rj, zj = vs[j]");
    L.push("            if (zi > z_cm) != (zj > z_cm) and r_cm < (rj - ri) * (z_cm - zi) / (zj - zi) + ri:");
    L.push("                inside = not inside");
    L.push("            j = i");
    L.push("        return inside");
    L.push("    return False");
    L.push("");
    L.push("def _gui_state_at(r_cm, z_cm):");
    L.push("    for kind, mat, rho, te, ti, p in reversed(_GUI_SHAPES):");
    L.push("        if _gui_shape_contains(kind, p, r_cm, z_cm):");
    L.push("            return (mat, rho, te, ti)");
    L.push("    return (_GUI_BG[0], _GUI_BG[1], _GUI_BG[2], _GUI_BG[3])");
    L.push("");
    L.push("def gui_rho(r_cm, z_cm):");
    L.push("    return _gui_state_at(r_cm, z_cm)[1]");
    L.push("");
    L.push("def gui_Te(r_cm, z_cm):");
    L.push("    return _gui_state_at(r_cm, z_cm)[2]");
    L.push("");
    L.push("def gui_Ti(r_cm, z_cm):");
    L.push("    return _gui_state_at(r_cm, z_cm)[3]");
    L.push("");
    for (const m of f.materials) {
      L.push(`def gui_vf_${m.name}(r_cm, z_cm):`);
      L.push(`    return 1.0 if _gui_state_at(r_cm, z_cm)[0] == ${pyStr(m.name)} else 0.0`);
      L.push("");
    }
  } else {
    const lastRegion = regions[regions.length - 1];
    const rLast = cm(lastRegion.rOuter);
    const corona = f.geometry.coronaRamp1d;
    const rEnd = useCoronaRamp1d ? rLast + corona.extentUm * 1.0e-4 : rLast;
    if (useVacuumOutside1d) {
      const rhoLines = emitPiecewise("gui_rho", args, cmpVar, innerBounds, regions.map((r) => r.rho));
      rhoLines.pop();
      L.push(...rhoLines);
      L.push(`    if ${cmpVar} < ${pyNum(rLast)}: return ${pyNum(lastRegion.rho)}`);
      if (useCoronaRamp1d) {
        L.push(
          `    if ${cmpVar} <= ${pyNum(rEnd)}: return max(${pyNum(corona.rho0)} * math.exp(-(${cmpVar} - ${pyNum(rLast)}) / ${pyNum(corona.scaleUm * 1.0e-4)}), ${pyNum(corona.rhoMin)})`,
        );
      }
      L.push("    return 1.0e-10");
    } else {
      L.push(...emitPiecewise("gui_rho", args, cmpVar, innerBounds, regions.map((r) => r.rho)));
    }
    L.push("");
    L.push(...emitPiecewise("gui_Te", args, cmpVar, innerBounds, regions.map((r) => eV(r.Te))));
    L.push("");
    L.push(...emitPiecewise("gui_Ti", args, cmpVar, innerBounds, regions.map((r) => eV(r.Ti))));
    L.push("");
    for (const m of f.materials) {
      const vfLines = emitPiecewise(
        `gui_vf_${m.name}`,
        args,
        cmpVar,
        innerBounds,
        regions.map((r) => (r.materialName === m.name ? 1.0 : 0.0)),
      );
      if (useVacuumOutside1d) {
        const guardR = m.name === lastRegion.materialName ? rEnd : rLast;
        vfLines.splice(1, 0, `    if ${cmpVar} >= ${pyNum(guardR)}: return 0.0`);
      }
      L.push(...vfLines);
      L.push("");
    }
    if (useVacuumOutside1d) {
      L.push(`def gui_vf_VOID(${args}):`);
      L.push(`    return 1.0 if ${cmpVar} >= ${pyNum(rEnd)} else 0.0`);
      L.push("");
    }
  }
  L.push("Geometry(");
  L.push("    volfrac=dict(");
  for (const m of f.materials) {
    L.push(`        ${m.name}=gui_vf_${m.name},`);
  }
  if (useVacuumOutside1d) {
    L.push("        VOID=gui_vf_VOID,");
  }
  L.push("    ),");
  L.push("    rho=gui_rho,");
  L.push("    Te=gui_Te,");
  L.push("    Ti=gui_Ti,");
  L.push(`    radiation_field=${pyStr(f.geometry.radiationField)},`);
  L.push(")");
  L.push("");

  // --- piecewise-linear waveform helper ----------------------------------------------
  const marshakTable =
    f.radiation.enabled && f.radiation.outerR === "marshak" && f.radiation.marshakMode === "table";
  const laserTrapezoid =
    f.laser.enabled &&
    f.laser.waveformMode === "square" &&
    (sec(f.laser.riseTime) > 0 || sec(f.laser.fallTime) > 0);
  const laserTable = f.laser.enabled && (f.laser.waveformMode === "table" || laserTrapezoid);
  const hoteTable =
    f.laser.enabled &&
    f.laser.hotE.enabled &&
    !f.laser.hotE.useChannels &&
    f.laser.hotE.etaMode === "table";
  const hoteChannelTables =
    !is2d &&
    f.laser.hotE.enabled &&
    f.laser.hotE.useChannels &&
    f.laser.hotE.etaEvolution === "legacy" &&
    f.laser.hotE.channels.some((ch) => ch.etaMode === "table");
  const pressureDrive = f.main.dimension === "1D_SPH" && f.hydro.boundary1d === "pressure";
  const pressureTable = pressureDrive && f.hydro.boundaryPressure.mode === "table";
  if (marshakTable || laserTable || hoteTable || hoteChannelTables || pressureTable) {
    L.push("def _gui_pwl(x, xs, ys):");
    L.push("    if x <= xs[0]: return ys[0]");
    L.push("    if x >= xs[-1]: return ys[-1]");
    L.push("    for i in range(len(xs) - 1):");
    L.push("        if x <= xs[i + 1]:");
    L.push("            w = (x - xs[i]) / (xs[i + 1] - xs[i])");
    L.push("            return ys[i] * (1.0 - w) + ys[i + 1] * w");
    L.push("    return ys[-1]");
    L.push("");
  }
  if (pressureTable) {
    const ts = f.hydro.boundaryPressure.points.map((p) => p.t * 1e-9);
    const vs = f.hydro.boundaryPressure.points.map((p) => p.v * 1e12);
    L.push("def gui_boundary_pressure(t_s):");
    L.push(`    return _gui_pwl(t_s, ${pyList(ts)}, ${pyList(vs)})  # t [ns] -> P_ext [dyn/cm^2] (input Mbar)`);
    L.push("");
  }

  // --- Numerics ----------------------------------------------------------------
  L.push("Numerics(");
  L.push("    dt=dict(");
  L.push(`        initial_s=${numC(sec(f.numerics.dtInitial), f.numerics.dtInitial)}`);
  L.push(`        max_s=${numC(sec(f.numerics.dtMax), f.numerics.dtMax)}`);
  L.push(`        min_s=${numC(sec(f.numerics.dtMin), f.numerics.dtMin)}`);
  L.push(`        growth_factor=${pyNum(f.numerics.growthFactor)},`);
  L.push("    ),");
  const tStart = `, T_start_eV=${pyNum(f.hydro.tStartEV)}`;
  const pv = f.hydro.plasmaVisc;
  const pvisc = pv.enabled
    ? `, plasma_viscosity=dict(enabled=True, model=${pyStr(pv.model)}, species=${pyStr(pv.species)}, eta_const=${pyNum(pv.etaConst)}, eta0_scale=${pyNum(pv.eta0Scale)}, mfp_cap_cells=${pyNum(pv.mfpCapCells)}, lnlambda_fixed=${pyNum(pv.lnLambdaFixed)}, dt_safety=${pyNum(pv.dtSafety)})`
    : "";
  const bpConst = toCanonical(f.hydro.boundaryPressure.value, "pressure");
  const bp = !pressureDrive
    ? ""
    : pressureTable
      ? ", boundary_pressure=gui_boundary_pressure"
      : `, boundary_pressure=(lambda t_s: ${pyNum(bpConst)})`;
  if (is2d) {
    L.push(`    hydro=dict(enabled=${pyBool(f.hydro.enabled)}${tStart}),`);
  } else {
    L.push(
      `    hydro=dict(enabled=${pyBool(f.hydro.enabled)}, boundary_1d=${pyStr(f.hydro.boundary1d)}${bp}${tStart}${pvisc}, driver_full_step_retry_enabled=True),`,
    );
  }
  const cn = f.conduction;
  const ionCond = cn.ionConduction ? ", ion_conduction=True" : "";
  const snb =
    cn.nonlocalModel === "snb"
      ? `, nonlocal_model="snb", snb_n_groups=${pyNum(cn.snbNGroups)}, snb_E_max_over_Te=${pyNum(cn.snbEMaxOverTe)}, snb_mfp=${pyStr(cn.snbMfp)}, snb_efield=${pyStr(cn.snbEfield)}, snb_picard_max_iters=${pyNum(cn.snbPicardMaxIters)}, snb_picard_rtol=${pyNum(cn.snbPicardRtol)}`
      : "";
  L.push(
    `    conduction=dict(enabled=${pyBool(cn.enabled)}, f_lim=${pyNum(cn.fLim)}${ionCond}${snb}),`,
  );
  const flo = f.numerics.floors;
  L.push(
    `    floors=dict(rho_floor_gcc=${pyNum(flo.rhoFloorGcc)}, Te_floor_eV=${pyNum(flo.TeFloorEV)}, Ti_floor_eV=${pyNum(flo.TiFloorEV)}),`,
  );
  L.push(")");
  L.push("");

  // --- marshak Tr(t) and hot-electron callables --------------------------------------
  if (marshakTable) {
    const ts = f.radiation.marshakPoints.map((p) => p.t * 1e-9);
    const vs = f.radiation.marshakPoints.map((p) => p.v);
    L.push("def gui_marshak_tr(t_s):");
    L.push(`    return _gui_pwl(t_s, ${pyList(ts)}, ${pyList(vs)})  # t [ns] -> Tr [eV]`);
    L.push("");
  }
  if (hoteTable) {
    const ts = f.laser.hotE.etaPoints.map((p) => p.t * 1e-9);
    const vs = f.laser.hotE.etaPoints.map((p) => p.v);
    L.push("def gui_hote_eta(t_s):");
    L.push(`    return _gui_pwl(t_s, ${pyList(ts)}, ${pyList(vs)})  # hot-e eta(t)`);
    L.push("");
  }
  if (hoteChannelTables) {
    f.laser.hotE.channels.forEach((ch, i) => {
      if (ch.etaMode !== "table") return;
      const ts = ch.etaPoints.map((p) => p.t * 1e-9);
      const vs = ch.etaPoints.map((p) => p.v);
      L.push(`def gui_hote_eta_ch${i}(t_s):`);
      L.push(`    return _gui_pwl(t_s, ${pyList(ts)}, ${pyList(vs)})  # hot-e eta(t)`);
      L.push("");
    });
  }
  // --- Radiation ------------------------------------------------------------------
  if (!f.radiation.enabled) {
    L.push("Radiation(enabled=False)");
    L.push("");
  } else {
    const r = f.radiation;
    const lo = Math.min(0.1, r.groupBoundsEV[0]);
    const hi = Math.max(1.0e5, r.groupBoundsEV[r.groupBoundsEV.length - 1]);
    L.push("Radiation(");
    L.push("    enabled=True,");
    L.push(`    mode=${pyStr(r.mode)},`);
    L.push(`    groups=${pyNum(r.groups)},`);
    L.push(`    group_bounds_eV=${pyList(r.groupBoundsEV)},`);
    L.push(`    compute_T_range_eV=${pyList([lo, hi])},`);
    if (r.mode === "multigroup_diffusion") {
      L.push("    multigroup_diffusion=dict(");
      if (is2d) {
        L.push(
          `        boundary=dict(inner_r="reflect", outer_r=${pyStr(r.outerR)}, z=${pyStr(r.zBc)}),`,
        );
      } else {
        L.push(`        boundary=dict(inner_r="reflect", outer_r=${pyStr(r.outerR)}),`);
      }
      L.push("    ),");
      if (r.outerR === "marshak") {
        L.push("    boundary=dict(");
        if (r.marshakMode === "table") {
          L.push("        marshak_Tr=gui_marshak_tr,");
        } else {
          L.push(`        marshak_Tr_eV=${pyNum(r.marshakTrEV)},`);
        }
        L.push("    ),");
      }
    } else {
      L.push("    sn_transport=dict(");
      L.push(`        n_angles=${pyNum(r.snNAngles)},`);
      if (is2d) {
        L.push(
          `        boundary=dict(inner_r="reflect_parity", outer_r=${pyStr(r.outerR)}, z=${pyStr(r.zBc)}),`,
        );
      } else {
        L.push(`        boundary=dict(inner_r="reflect_parity", outer_r=${pyStr(r.outerR)}),`);
      }
      L.push("    ),");
      if (r.outerR === "marshak") {
        if (r.marshakMode === "table") {
          L.push(`    boundary=dict(inner_r="reflect", outer_r=${pyStr(r.outerR)}, marshak_Tr=gui_marshak_tr),`);
        } else {
          L.push(`    boundary=dict(inner_r="reflect", outer_r=${pyStr(r.outerR)}, marshak_Tr_eV=${pyNum(r.marshakTrEV)}),`);
        }
      } else {
        L.push(`    boundary=dict(inner_r="reflect", outer_r=${pyStr(r.outerR)}),`);
      }
    }
    L.push(")");
    L.push("");
  }

  // --- Laser ------------------------------------------------------------------------
  if (!f.laser.enabled) {
    L.push("Laser(enabled=False)");
    L.push("");
  } else {
    if (f.laser.waveformMode === "table") {
      const ts = f.laser.waveformPoints.map((p) => p.t * 1e-9);
      const ps = f.laser.waveformPoints.map((p) => p.v * 1e12);
      const tLast = ts[ts.length - 1];
      L.push("def gui_beam_power(t_s):");
      L.push(`    if t_s < 0 or t_s > ${pyNum(tLast)}: return 0.0`);
      L.push(`    return _gui_pwl(t_s, ${pyList(ts)}, ${pyList(ps)})  # editor: t [ns] / P [TW]`);
      L.push("");
    } else if (f.laser.waveformMode === "gaussian") {
      const tau = sec(f.laser.gaussianFwhm);
      const t0 = sec(f.laser.gaussianCenter);
      // E_total = P0 * tau * sqrt(pi / (4 ln 2))  <=>  P0 = E * 2 sqrt(ln2/pi) / tau
      const p0 =
        f.laser.gaussianSpec === "peak"
          ? watt(f.laser.gaussianPeakW)
          : (f.laser.gaussianEnergyJ * 0.9394372786996513) / tau;
      L.push("def gui_beam_power(t_s):");
      L.push("    # gaussian pulse: P0 * exp(-4 ln2 ((t - t0)/tau)^2), tau = FWHM");
      L.push(
        `    return ${pyNum(p0)} * math.exp(-2.772588722239781 * ((t_s - ${pyNum(t0)}) / ${pyNum(tau)}) ** 2)`,
      );
      L.push("");
    } else if (laserTrapezoid) {
      const rise = sec(f.laser.riseTime);
      const flat = sec(f.laser.pulseDuration);
      const fall = sec(f.laser.fallTime);
      const pw = watt(f.laser.powerW);
      const rawT = [0, rise, rise + flat, rise + flat + fall];
      const rawP = [rise > 0 ? 0 : pw, pw, pw, fall > 0 ? 0 : pw];
      const ts: number[] = [];
      const ps: number[] = [];
      for (let i = 0; i < rawT.length; i++) {
        if (i > 0 && !(rawT[i] > ts[ts.length - 1])) continue;
        ts.push(rawT[i]);
        ps.push(rawP[i]);
      }
      L.push("def gui_beam_power(t_s):");
      L.push(`    if t_s < 0 or t_s > ${pyNum(ts[ts.length - 1])}: return 0.0`);
      L.push(`    return _gui_pwl(t_s, ${pyList(ts)}, ${pyList(ps)})  # trapezoid: rise/flat/fall`);
      L.push("");
    } else {
      const dur = sec(f.laser.pulseDuration);
      const pw = watt(f.laser.powerW);
      L.push("def gui_beam_power(t_s):");
      L.push(`    if t_s <= ${pyNum(dur)}: return ${pyNum(pw)}  # ${qLabel(f.laser.powerW)}, ${qLabel(f.laser.pulseDuration)}`);
      L.push("    return 0.0");
      L.push("");
    }
    L.push("Laser(");
    L.push("    enabled=True,");
    L.push(`    wavelength_nm=${pyNum(f.laser.wavelengthNm)},`);
    L.push(`    mode=${pyStr(is2d ? "raytrace_3d" : f.laser.mode)},`);
    if (!is2d && f.laser.mode === "raytrace_2d") {
      L.push(`    rays_per_beam=${pyNum(f.laser.raysPerBeam)},`);
    }
    if (is2d && f.laser.raysPerBeam !== 1000) {
      L.push(`    rays_per_beam=${pyNum(f.laser.raysPerBeam)},  # 2D default is 128/side when omitted`);
    }
    if (f.laser.rayOutputTrajectory) {
      const rayOutputCount = Math.min(
        f.laser.rayOutputCount,
        effectiveLaserRaysPerBeam(f),
      );
      L.push("    ray_output_trajectory=True,");
      L.push(`    ray_output_count=${pyNum(rayOutputCount)},`);
    }
    L.push("    beams=[");
    const emittedBeams = is2d
      ? f.laser.beams
      : f.laser.beams.slice(0, 1);
    emittedBeams.forEach((b, i) => {
      const bname = b.name.trim().length > 0 ? b.name.trim() : `beam_${String(i).padStart(2, "0")}`;
      L.push("        LaserBeam(");
      L.push(`            name=${pyStr(bname)},`);
      if (is2d) {
        const norm = Math.hypot(b.dirX, b.dirY, b.dirZ);
        L.push(`            direction=(${pyNum(b.dirX / norm)}, ${pyNum(b.dirY / norm)}, ${pyNum(b.dirZ / norm)}),`);
      } else {
        const dz = b.axialDirection === "plus_z" ? 1.0 : -1.0;
        L.push(`            direction=(0.0, 0.0, ${pyNum(dz)}),`);
      }
      if (!is2d || b.powerFraction === 1.0) {
        L.push("            power=gui_beam_power,");
      } else {
        L.push(`            power=lambda t_s, _f=${pyNum(b.powerFraction)}: _f * gui_beam_power(t_s),`);
      }
      L.push(`            f_number=${pyNum(b.fNumber)},`);
      L.push(
        `            focus=(0.0, 0.0, ${pyNum(b.focusZUm * 1e-4)}),  # ${pyNum(b.focusZUm)} µm on the beam axis`,
      );
      if (b.profileModel === "table") {
        const rs = b.profilePoints.map((p) => p.t);
        const is = b.profilePoints.map((p) => p.v);
        L.push(
          `            profile=dict(model="table", r_um=${pyList(rs)}, I_rel=${pyList(is)}),`,
        );
      } else {
        L.push(
          `            profile=dict(model="super_gaussian", w0_um=${pyNum(b.w0Um)}, m=${pyNum(b.superGaussianM)}),`,
        );
      }
      L.push(i === emittedBeams.length - 1 ? "        )" : "        ),");
    });
    L.push("    ],");
    if (!is2d && f.laser.hotE.enabled) {
      const he = f.laser.hotE;
      L.push("    hot_electron=dict(");
      L.push("        enable=True,");
      if (!he.useChannels) {
        L.push(`        source_nc_fraction=${pyNum(he.sourceNcFraction)},`);
        if (he.etaMode === "table") {
          L.push("        eta_hot_table=gui_hote_eta,");
        } else {
          L.push(`        eta_hot=${pyNum(he.etaHot)},`);
        }
        L.push(`        T_hot_eV=${pyNum(he.THotEV)},`);
        L.push(`        n_energy_groups=${pyNum(he.nEnergyGroups)},`);
        L.push(`        E_min_over_Th=${pyNum(he.EMinOverTh)}, E_max_over_Th=${pyNum(he.EMaxOverTh)},`);
        L.push(`        angular_model=${pyStr(he.angularModel)},`);
        if (he.angularModel === "cone") {
          L.push(`        theta_div_deg=${pyNum(he.thetaDivDeg)},`);
        } else {
          L.push(`        inner_bc=${pyStr(he.innerBc)},`);
        }
        L.push(`        n_mu=${pyNum(he.nMu)}, n_phi=${pyNum(he.nPhi)},`);
        L.push(`        subtract_from_laser=${pyBool(he.subtractFromLaser)},`);
        L.push(`        explicit_source_limit=${pyNum(he.explicitSourceLimit)},`);
      } else {
        L.push(`        angular_model=${pyStr(he.angularModel)},`);
        if (he.angularModel === "radial") {
          L.push(`        inner_bc=${pyStr(he.innerBc)},`);
        }
        L.push(`        subtract_from_laser=${pyBool(he.subtractFromLaser)},`);
        L.push(`        explicit_source_limit=${pyNum(he.explicitSourceLimit)},`);
        if (he.etaEvolution === "model") {
          L.push('        eta_mode="model",');
          L.push(
            `        eta_model=dict(ln_filter_tau_s=${pyNum(he.lnFilterTauS)}, eta_total_cap=${pyNum(he.etaTotalCap)}),`,
          );
        }
        L.push("        sources=[");
        he.channels.forEach((ch, i) => {
          L.push("            dict(");
          L.push(`                mechanism=${pyStr(ch.mechanism)},`);
          L.push(`                capture_nc_fraction=${pyNum(ch.captureNcFraction)},`);
          L.push(`                T_hot_eV=${pyNum(ch.THotEV)},`);
          L.push(`                n_energy_groups=${pyNum(ch.nEnergyGroups)},`);
          L.push(
            `                E_min_over_Th=${pyNum(ch.EMinOverTh)}, E_max_over_Th=${pyNum(ch.EMaxOverTh)},`,
          );
          L.push(`                n_mu=${pyNum(ch.nMu)}, n_phi=${pyNum(ch.nPhi)},`);
          if (ch.mechanism === "tpd") {
            L.push(`                tpd_theta_deg=${pyNum(ch.tpdThetaDeg)},`);
            L.push(`                tpd_delta_deg=${pyNum(ch.tpdDeltaDeg)},`);
          } else {
            L.push(`                theta_div_deg=${pyNum(ch.thetaDivDeg)},`);
          }
          if (he.etaEvolution === "legacy") {
            if (ch.etaMode === "table") {
              L.push(`                eta_table=gui_hote_eta_ch${i},`);
            } else {
              L.push(`                eta=${pyNum(ch.etaHot)},`);
            }
          } else {
            L.push(`                eval_nc_fraction=${pyNum(ch.evalNcFraction)},`);
            L.push(`                threshold_multiplier=${pyNum(ch.thresholdMultiplier)},`);
            L.push(`                eta_inf=${pyNum(ch.etaInf)},`);
            L.push(`                eta_hard_cap=${pyNum(ch.etaHardCap)},`);
            L.push(`                shape_coefficient=${pyNum(ch.shapeCoefficient)},`);
            L.push(`                relaxation_model=${pyStr(ch.relaxationModel)},`);
            L.push(`                relaxation_tau_s=${pyNum(ch.relaxationTauS)},`);
            L.push(`                relaxation_tau_min_s=${pyNum(ch.relaxationTauMinS)},`);
            L.push(`                relaxation_tau_max_s=${pyNum(ch.relaxationTauMaxS)},`);
          }
          L.push(i === he.channels.length - 1 ? "            )" : "            ),");
        });
        L.push("        ],");
      }
      L.push("    ),");
    }
    if (!is2d && f.laser.cbet.enabled) {
      const cb = f.laser.cbet;
      const preset = BEAM_PRESETS[cb.portPreset];
      L.push("    cbet=dict(");
      L.push("        enable=True,");
      L.push('        geometry_mode="port_section",');
      L.push(`        n_section_phi=${pyNum(cb.nSectionPhi)},`);
      L.push(`        f_cbet=${pyNum(cb.fCbet)},`);
      L.push(`        alpha_iaw=${pyNum(cb.alphaIaw)},`);
      L.push(`        theta_cap=${pyNum(cb.thetaCap)},`);
      L.push(`        tol=${pyNum(cb.tol)}, max_iters=${pyNum(cb.maxIters)},`);
      L.push(`        n_impact_bins=${pyNum(cb.nImpactBins)},`);
      L.push(`        ne_frac_cutoff=${pyNum(cb.neFracCutoff)}, k_a_floor=${pyNum(cb.kAFloor)},`);
      L.push("    ),");
      L.push("    port_configuration=dict(");
      L.push('        normalization="sum_weights_one",');
      L.push("        ports=[");
      preset.ports.forEach((port) => {
        const direction = port.dir.map((component) => component.toPrecision(15)).join(", ");
        const detune = cb.detuneSplitNm > 0
          ? `, delta_lambda_nm=${port.portId % 2 === 0
            ? `+${pyNum(cb.detuneSplitNm / 2)}`
            : pyNum(-cb.detuneSplitNm / 2)}`
          : "";
        L.push(
          `            dict(port_id=${port.portId}, direction=(${direction}), power_weight=${pyNum(port.weight)}${detune}),`,
        );
      });
      L.push("        ],");
      L.push("    ),");
    }
    L.push(")");
    L.push("");
  }

  // --- Burn / Output / Diagnostics ------------------------------------------------------
  if (f.burn.enabled) {
    const bu = f.burn;
    const fuels = (["DT", "DD", "D3He"] as const).filter((k) => bu.fuels[k]);
    const mats = bu.fuelMaterials.split(",").map((s) => s.trim()).filter((s) => s.length > 0);
    L.push("Burn(");
    L.push("    enabled=True,");
    L.push(`    fuels=[${fuels.map((k) => pyStr(k)).join(", ")}],`);
    L.push(`    scheme=${pyStr(bu.scheme)},`);
    if (bu.scheme === "mc") {
      L.push(`    mc_particles_per_cell=${pyNum(bu.mcParticlesPerCell)},`);
    }
    if (bu.scheme === "diffusion") {
      L.push(`    diffusion_groups=${pyNum(bu.diffusionGroups)},`);
      L.push(`    diffusion_E_min_keV=${pyNum(bu.diffusionEMinKeV)},`);
    }
    L.push(`    partition=${pyStr(bu.partition)},`);
    L.push(`    screening=${pyStr(bu.screening)},`);
    L.push(`    fuel_materials=[${mats.map((s) => pyStr(s)).join(", ")}],`);
    L.push(`    x_D=${pyNum(bu.xD)}, x_T=${pyNum(bu.xT)}, x_He3=${pyNum(bu.xHe3)},`);
    L.push(`    T_floor_keV=${pyNum(bu.TFloorKeV)},`);
    L.push(`    explicit_source_limit=${pyNum(bu.explicitSourceLimit)},`);
    L.push(`    eps_deplete=${pyNum(bu.epsDeplete)},`);
    L.push(`    subcycle_max=${pyNum(bu.subcycleMax)},`);
    L.push(`    vf_threshold=${pyNum(bu.vfThreshold)},`);
    L.push(`    neutron_heating=${pyBool(bu.neutronHeating)},`);
    if (bu.neutronHeating) {
      L.push(`    neutron_heating_n_mu=${pyNum(bu.neutronHeatingNMu)},`);
    }
    L.push(")");
  } else {
    L.push("Burn(enabled=False)");
  }
  L.push("");
  const outDir = f.output.directory.trim().length > 0 ? f.output.directory.trim() : `outputs/${f.main.name}`;
  L.push("Output(");
  L.push(`    directory=${pyStr(outDir)},`);
  L.push("    plot_every=0,");
  L.push("    history_every=1,");
  L.push("    checkpoint_every=0,");
  L.push(
    `    plot_every_s=${f.output.plotEveryS ? numC(sec(f.output.plotEveryS), f.output.plotEveryS) : "-1.0,"}`,
  );
  L.push(
    `    history_every_s=${f.output.historyEveryS ? numC(sec(f.output.historyEveryS), f.output.historyEveryS) : "-1.0,"}`,
  );
  L.push(
    `    checkpoint_every_s=${f.output.checkpointEveryS ? numC(sec(f.output.checkpointEveryS), f.output.checkpointEveryS) : "-1.0,"}`,
  );
  L.push(")");
  L.push("");
  L.push("Diagnostics(enabled=True)");

  // --- custom python block --------------------------------------------------------------
  if (f.customPythonBlock.trim().length > 0) {
    L.push("");
    L.push("# ---- custom python block (GUI-unmanaged, emitted verbatim) ----");
    L.push(f.customPythonBlock.replace(/\s+$/, ""));
  }
  L.push("");
  return L.join("\n");
}

export const DECK_SCHEMA_VERSION = GUI_SCHEMA_VERSION;

function wipWarningHeader(formErrors: string[]): string {
  if (formErrors.length === 0) return "";
  return [
    "# WARNING: saved with form validation errors — this namelist may not run:",
    ...formErrors.map((e) => `#   - ${e}`),
    "",
  ].join("\n");
}

/** Best-effort deck for saving work-in-progress: always embeds the GUI state. */
export function generateDeckForSave(f: FormState, formErrors: string[]): string {
  try {
    const body = generateDeck(f);
    if (body.length > 0) {
      return wipWarningHeader(formErrors) + body;
    }
  } catch {
    /* fall through to the state-only stub */
  }
  return (
    wipWarningHeader(formErrors) +
    [
      "# TENRYU Studio form snapshot (deck generation unavailable)",
      `# TENRYU-GUI-STATE: ${JSON.stringify(f)}`,
      "# Fix the validation errors and save again to obtain a runnable namelist.",
    ].join("\n") +
    "\n"
  );
}
