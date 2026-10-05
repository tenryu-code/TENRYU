// Directly edited zoning_intent forms whose solver mesh is stored in
// fixtures/presetZoningNodes.json next to the laser presets'. Each makes one part of the way Studio
// hands a form to the zoning change the nodes (checked when the forms were made: switching it
// moves nodes by 0.3 to 9 % of the domain):
//   shell: the solver's default ratio cap (2.0) and minimum cells per segment (1) bind, the
//          density comes from the regions and the void padding, the pins from the interfaces,
//          and r_max = 300 µm is 0.030000000000000002 cm before the deck's rounding;
//   foil:  the width measure with dr_min binding and the default ratio cap, r_max = 130 µm;
//   two-layer foil: an unpinned density jump that the quadrature needs as an event.
import { defaultResolutionRequirement, defaultZoningIntent, type ZoningIntentForm } from "../src/core/deck/mesh1d";
import type { FormState } from "../src/core/deck/formState";
import { presetLaserFoil, presetLaserShell } from "../src/core/presets1d";
import { q } from "../src/core/units";

function intentForm(f: FormState, intent: Partial<ZoningIntentForm>): FormState {
  f.mesh.grid1d = "zoning_intent";
  f.mesh.recommendation = null;
  f.mesh.resolutionRequirement = defaultResolutionRequirement();
  f.mesh.zoningIntent = { ...defaultZoningIntent(), ...intent };
  return f;
}

export function editedIntentForms(): Record<string, FormState> {
  const shell = presetLaserShell();
  shell.mesh.rMax = q(300, "µm");
  intentForm(shell, {
    measure: "spherical_cell_mass",
    nCells: 400,
    profile: [
      { rCm: 0.0, w: 1.0 },
      { rCm: 0.023, w: 1.0 },
      { rCm: 0.0236, w: 0.02 },
    ],
  });

  const foil = presetLaserFoil();
  foil.mesh.rMax = q(130, "µm");
  intentForm(foil, {
    measure: "width",
    nCells: 300,
    drMinCm: 1.0e-5,
    profile: [
      { rCm: 0.0, w: 1.0 },
      { rCm: 0.0024, w: 1.0 },
      { rCm: 0.0025, w: 0.01 },
      { rCm: 0.013, w: 1.0 },
    ],
  });

  const twoLayer = presetLaserFoil();
  twoLayer.materials = [...twoLayer.materials, { ...twoLayer.materials[0], name: "Al" }];
  const ch = twoLayer.geometry.regions[0];
  twoLayer.geometry.regions = [
    { ...ch, rOuter: q(15, "µm") },
    { ...ch, materialName: "Al", rOuter: q(25, "µm"), rho: 2.7 },
  ];
  twoLayer.geometry.vacuumOutside1d = false;
  twoLayer.mesh.rMax = q(25, "µm");
  intentForm(twoLayer, {
    measure: "areal_mass",
    nCells: 200,
    pinInterfaces: false,
    profile: [
      { rCm: 0.0, w: 1.0 },
      { rCm: 0.0025, w: 0.05 },
    ],
  });

  return { editedShellIntent: shell, editedFoilWidthIntent: foil, editedTwoLayerIntent: twoLayer };
}
