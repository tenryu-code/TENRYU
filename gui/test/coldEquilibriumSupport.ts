import { defaultFormState, setInactiveCellMode, type FormState } from "../src/core/deck/formState";
import { q } from "../src/core/units";

/** A 1D liquid-D2 fuel inside a CH shell, both TMAT, set up for cold_equilibrium
 *  with the literature bulk moduli (liquid D2 0.12 GPa, polystyrene 5.8 GPa). */
export function coldEquilibriumForm(): FormState {
  const f = defaultFormState();
  f.materials = [
    { ...f.materials[0], name: "D2", A: 2.014, Z: 1, eosModel: "tmat", eosFile: "TMAT-H5/D2.tmat.h5" },
    { ...f.materials[0], name: "CH", eosModel: "tmat", eosFile: "TMAT-H5/CH.tmat.h5" },
  ];
  f.geometry.regions = [
    { materialName: "D2", rOuter: q(900, "µm"), rho: 0.17, Te: q(1e-3, "eV"), Ti: q(1e-3, "eV") },
    { materialName: "CH", rOuter: q(1000, "µm"), rho: 1.05, Te: q(1e-3, "eV"), Ti: q(1e-3, "eV") },
  ];
  f.mesh.rMax = q(1000, "µm");
  f.hydro.tStartEV = 2.0;
  setInactiveCellMode(f, "cold_equilibrium");
  f.materials[0].coldReference!.K0 = q(0.12, "GPa");
  f.materials[1].coldReference!.K0 = q(5.8, "GPa");
  return f;
}
