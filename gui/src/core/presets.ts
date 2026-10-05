import { defaultFormState, ensureBackgroundGas, type FormState } from "./deck/formState";
import { defaultShape2D } from "./geometry2d";
import { q } from "./units";

// 1D presets: presets1d.ts.
export {
  presetBlank,
  presetBurnPusher,
  presetCbetHotElectrons,
  presetColdEquilibriumLayers,
  presetIndirectCapsule,
  presetLaserFoil,
  presetLaserShell,
  presetMarshakWave,
  presetPressureShock,
  presetSnbFoil,
  presetSnRadiationFoil,
  presetTableRadiationFoil,
} from "./presets1d";

export function preset2dBlank(): FormState {
  const f = defaultFormState();
  f.main.name = "blank_2d";
  f.main.dimension = "2D_RZ";
  ensureBackgroundGas(f);
  f.geometry.regions = [];
  f.geometry.shapes2d = [];
  f.geometry.background2d.rho = 1.0e-4;
  return f;
}

export function preset2dPolarSphere(): FormState {
  const f = defaultFormState();
  f.main.name = "solid_sphere_2d";
  f.main.dimension = "2D_RZ";
  f.main.tEnd = q(2.0, "ns");
  f.mesh.radialZoning2d = "regions";
  f.mesh.rMax = q(600, "µm");
  f.mesh.zMin = q(-600, "µm");
  f.mesh.zMax = q(600, "µm");
  f.mesh.nr = 64;
  f.mesh.nz = 96;
  f.materials[0].name = "solid";
  f.materials.push({
    name: "gas",
    A: 1,
    Z: 1,
    eosModel: "ideal_gas",
    gamma: 5 / 3,
    cvEOverride: undefined,
    eosFile: "",
    opacityModel: "constant",
    kappaA: 0,
    kappaS: 0,
    opacityFile: "",
  });
  const solid = defaultShape2D("solidSphere");
  solid.label = "solid";
  solid.materialName = "solid";
  solid.rho = 1.0;
  solid.z0 = q(0, "µm");
  solid.radius = q(300, "µm");
  f.geometry.regions = [];
  f.geometry.shapes2d = [solid];
  f.geometry.background2d = {
    materialName: "gas",
    rho: 1.0e-4,
    Te: q(1, "eV"),
    Ti: q(1, "eV"),
  };
  f.radiation.enabled = false;
  f.laser.enabled = false;
  f.conduction.enabled = false;
  f.hydro.enabled = true;
  return f;
}

export function preset2dPolarCapsule(): FormState {
  const f = defaultFormState();
  f.main.name = "capsule_2d";
  f.main.dimension = "2D_RZ";
  f.main.tEnd = q(3.0, "ns");
  f.mesh.radialZoning2d = "regions";
  f.mesh.rMax = q(800, "µm");
  f.mesh.zMin = q(-800, "µm");
  f.mesh.zMax = q(800, "µm");
  f.mesh.nr = 96;
  f.mesh.nz = 128;
  f.materials[0].name = "fuel";
  f.materials[0].A = 2.5;
  f.materials[0].Z = 1;
  f.materials.push({ ...f.materials[0], name: "shell", A: 6.5, Z: 3.5 });
  f.materials.push({ ...f.materials[0], name: "gas", A: 1, Z: 1 });
  const fuel = defaultShape2D("solidSphere");
  fuel.label = "fuel";
  fuel.materialName = "fuel";
  fuel.rho = 1.0e-3;
  fuel.z0 = q(0, "µm");
  fuel.radius = q(350, "µm");
  const shell = defaultShape2D("shell");
  shell.label = "shell-material";
  shell.materialName = "shell";
  shell.rho = 1.05;
  shell.z0 = q(0, "µm");
  shell.rIn = q(350, "µm");
  shell.radius = q(400, "µm");
  f.geometry.regions = [];
  f.geometry.shapes2d = [fuel, shell];
  f.geometry.background2d = {
    materialName: "gas",
    rho: 1.0e-4,
    Te: q(1, "eV"),
    Ti: q(1, "eV"),
  };
  f.radiation.enabled = false;
  f.laser.enabled = false;
  f.conduction.enabled = false;
  f.hydro.enabled = true;
  return f;
}

export function preset2dRectSlabRad(): FormState {
  const f = defaultFormState();
  f.main.name = "slab_radiation_2d";
  f.main.dimension = "2D_RZ";
  f.main.geometry1d = "planar";
  f.main.temperatureModel = "1T";
  f.main.tEnd = q(1.0, "ns");
  f.mesh.rMax = q(0.06, "cm");
  f.mesh.nr = 128;
  f.mesh.zMin = q(-150, "µm");
  f.mesh.zMax = q(150, "µm");
  f.mesh.nz = 16;
  f.materials[0].name = "ch_slab";
  f.materials[0].cvEOverride = 8.68e11;
  f.geometry.regions = [];
  f.geometry.shapes2d = [];
  f.geometry.background2d = {
    materialName: "ch_slab",
    rho: 0.2,
    Te: q(1, "eV"),
    Ti: q(1, "eV"),
  };
  f.geometry.radiationField = "zero";
  f.radiation.outerR = "marshak";
  f.radiation.marshakMode = "constant";
  f.radiation.marshakTrEV = 120.0;
  f.hydro.enabled = false;
  f.conduction.enabled = false;
  f.numerics.floors.TeFloorEV = 1.0;
  f.numerics.floors.TiFloorEV = 1.0;
  return f;
}

export function preset2dRectLaser(): FormState {
  const f = defaultFormState();
  f.main.name = "laser_cylinder_2d";
  f.main.dimension = "2D_RZ";
  f.mesh.rMax = q(500, "µm");
  f.mesh.nr = 96;
  f.mesh.zMin = q(-400, "µm");
  f.mesh.zMax = q(400, "µm");
  f.mesh.nz = 64;
  f.materials[0].name = "CD";
  f.materials[0].A = 7.0;
  f.materials.push({ ...f.materials[0], name: "gas", A: 1, Z: 1 });
  const column = defaultShape2D("block");
  column.label = "CD column";
  column.materialName = "CD";
  column.rho = 1.05;
  column.r0 = q(0, "µm");
  column.r1 = q(250, "µm");
  column.z0 = q(-400, "µm");
  column.z1 = q(400, "µm");
  f.geometry.regions = [];
  f.geometry.shapes2d = [column];
  f.geometry.background2d = {
    materialName: "gas",
    rho: 1.0e-4,
    Te: q(1, "eV"),
    Ti: q(1, "eV"),
  };
  f.laser.enabled = true;
  f.conduction.fLim = 0.06;
  f.hydro.enabled = true;
  f.numerics.dtInitial = q(5e-14, "s");
  f.numerics.dtMax = q(2e-12, "s"); // adaptive ceiling (form default); the pinned 5e-14 forced ~40k steps/ns
  f.numerics.floors.TeFloorEV = 0.1;
  f.numerics.floors.TiFloorEV = 0.1;
  return f;
}
