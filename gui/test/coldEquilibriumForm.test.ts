import { describe, expect, it } from "vitest";
import {
  coldReferenceFromInitialState,
  defaultFormState,
  emptyColdReference,
  migrateFormState,
  setInactiveCellMode,
  validateFormState,
  type FormState,
} from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";
import { extractGuiState } from "../src/core/deck/roundtrip";
import { t } from "../src/i18n";
import { q } from "../src/core/units";
import { coldEquilibriumForm } from "./coldEquilibriumSupport";

describe("inactive-cell treatment, exchange heat capacity and cold-equilibrium reference states", () => {
  it("decks with the default settings do not change", () => {
    const deck = generateDeck(defaultFormState());
    for (const key of ["T_start_inactive_cells", "qei_heat_capacity", "cold_equilibrium", "cold_reference"]) {
      expect(deck).not.toContain(key);
    }
  });

  it("selecting cold_equilibrium prefills each reference state from the material's region and selects the table heat capacity", () => {
    const f = coldEquilibriumForm();
    expect(f.hydro.qeiHeatCapacity).toBe("table");
    expect(f.materials[0].coldReference).toEqual({
      rhoGcc: 0.17,
      Te0: { value: 1e-3, unit: "eV" },
      Ti0: { value: 1e-3, unit: "eV" },
      P0: { value: 0, unit: "GPa" },
      K0: { value: 0.12, unit: "GPa" },
    });
    expect(f.materials[1].coldReference!.rhoGcc).toBe(1.05);
    expect(validateFormState(f)).toEqual([]);
  });

  it("an existing reference state is kept when the mode is selected again", () => {
    const f = coldEquilibriumForm();
    f.materials[0].coldReference!.rhoGcc = 0.2;
    setInactiveCellMode(f, "passive_fill");
    expect(f.hydro.qeiHeatCapacity).toBe("table");
    setInactiveCellMode(f, "cold_equilibrium");
    expect(f.materials[0].coldReference!.rhoGcc).toBe(0.2);
  });

  it("a material that no region uses gets an empty reference state", () => {
    const f = defaultFormState();
    f.materials.push({ ...f.materials[0], name: "unused" });
    const ref = coldReferenceFromInitialState(f, "unused");
    expect(Number.isNaN(ref.rhoGcc)).toBe(true);
    expect(Number.isNaN(ref.Te0.value)).toBe(true);
    expect(Number.isNaN(ref.Ti0.value)).toBe(true);
    expect(Number.isNaN(ref.K0.value)).toBe(true);
    expect(ref.P0).toEqual({ value: 0, unit: "GPa" });
  });

  it("cold_equilibrium writes the mode, its parameters, the table heat capacity and each reference state in cgs + eV", () => {
    const deck = generateDeck(coldEquilibriumForm());
    expect(deck).toContain(
      'T_start_eV=2, T_start_inactive_cells="cold_equilibrium", cold_equilibrium=dict(transition_begin_fraction=0.5, density_core_ratio=1.1, density_outer_ratio=1.5, inverse_max_iterations=80), qei_heat_capacity="table", driver_full_step_retry_enabled=True',
    );
    expect(deck).toContain(
      'eos=dict(model="tmat", file="TMAT-H5/D2.tmat.h5", cold_reference=dict(rho_gcc=0.17, Te0_eV=0.001, Ti0_eV=0.001, P0_dyn_cm2=0, bulk_modulus_dyn_cm2=1200000000)),',
    );
    expect(deck).toContain(
      'eos=dict(model="tmat", file="TMAT-H5/CH.tmat.h5", cold_reference=dict(rho_gcc=1.05, Te0_eV=0.001, Ti0_eV=0.001, P0_dyn_cm2=0, bulk_modulus_dyn_cm2=58000000000)),',
    );
  });

  it("rigid_wall and the table heat capacity are written without the cold-equilibrium keys", () => {
    const f = defaultFormState();
    f.hydro.tStartEV = 1.5;
    f.hydro.inactiveCells = "rigid_wall";
    f.hydro.qeiHeatCapacity = "table";
    f.materials[0].coldReference = emptyColdReference();
    const deck = generateDeck(f);
    expect(deck).toContain('T_start_eV=1.5, T_start_inactive_cells="rigid_wall", qei_heat_capacity="table"');
    expect(deck).not.toContain("cold_equilibrium");
    expect(deck).not.toContain("cold_reference");
  });

  it("validation reports each condition the solver checks for cold_equilibrium", () => {
    const v = t().validation;
    const errors = (mutate: (f: FormState) => void): string[] => {
      const f = coldEquilibriumForm();
      mutate(f);
      return validateFormState(f);
    };
    expect(errors((f) => { f.main.temperatureModel = "1T"; })).toContain(v.coldEqNeeds2T);
    expect(errors((f) => { f.hydro.tStartEV = 0; })).toContain(v.coldEqNeedsTStart);
    expect(errors((f) => { f.hydro.qeiHeatCapacity = "ideal_gas"; })).toContain(v.coldEqNeedsQeiTable);
    expect(errors((f) => { f.materials[1].eosModel = "ideal_gas"; })).toContain(v.coldEqNeedsTableEos("CH"));
    expect(errors((f) => { delete f.materials[1].coldReference; })).toContain(v.coldRefMissing("CH"));
    expect(errors((f) => { f.materials[0].coldReference!.rhoGcc = Number.NaN; })).toContain(v.coldRefRhoPositive("D2"));
    expect(errors((f) => { f.materials[0].coldReference!.Te0 = q(0, "eV"); })).toContain(v.coldRefTe0Positive("D2"));
    expect(errors((f) => { f.materials[0].coldReference!.Ti0 = q(-1, "eV"); })).toContain(v.coldRefTi0Positive("D2"));
    expect(errors((f) => { f.materials[0].coldReference!.P0 = q(Number.NaN, "GPa"); })).toContain(v.coldRefP0Finite("D2"));
    expect(errors((f) => { f.materials[0].coldReference!.K0 = q(Number.NaN, "GPa"); })).toContain(v.coldRefK0Positive("D2"));
    expect(errors((f) => { f.hydro.coldEquilibrium.transitionBeginFraction = 1; })).toContain(v.coldEqBeginFractionRange);
    expect(errors((f) => { f.hydro.coldEquilibrium.densityCoreRatio = 1; })).toContain(v.coldEqDensityRatios);
    expect(errors((f) => { f.hydro.coldEquilibrium.densityOuterRatio = 1.1; })).toContain(v.coldEqDensityRatios);
    expect(errors((f) => { f.hydro.coldEquilibrium.inverseMaxIterations = 7; })).toContain(v.coldEqMaxIterations);
    expect(errors((f) => { f.hydro.coldEquilibrium.inverseMaxIterations = 8.5; })).toContain(v.coldEqMaxIterations);
    // Boundary values the solver accepts.
    expect(errors((f) => { f.hydro.coldEquilibrium.inverseMaxIterations = 8; })).toEqual([]);
    expect(errors((f) => { f.materials[0].coldReference!.P0 = q(-1, "GPa"); })).toEqual([]);
  });

  it("1D-only settings are rejected in 2D", () => {
    const v = t().validation;
    const f = coldEquilibriumForm();
    f.main.dimension = "2D_RZ";
    expect(validateFormState(f)).toContain(v.coldEq1dOnly);
    const g = defaultFormState();
    g.main.dimension = "2D_RZ";
    g.hydro.inactiveCells = "rigid_wall";
    g.hydro.qeiHeatCapacity = "table";
    const errs = validateFormState(g);
    expect(errs).toContain(v.hydroRigidWall1dOnly);
    expect(errs).toContain(v.qeiTable1dOnly);
  });

  it("older GUI states load with the defaults, and saved settings survive the embedded state", () => {
    const old = structuredClone(defaultFormState()) as any;
    delete old.hydro.inactiveCells;
    delete old.hydro.qeiHeatCapacity;
    delete old.hydro.coldEquilibrium;
    const migrated = migrateFormState(old);
    expect(migrated.hydro.inactiveCells).toBe("passive_fill");
    expect(migrated.hydro.qeiHeatCapacity).toBe("ideal_gas");
    expect(migrated.hydro.coldEquilibrium).toEqual(defaultFormState().hydro.coldEquilibrium);
    expect(migrated.materials[0].coldReference).toBeUndefined();

    const unknown = structuredClone(defaultFormState()) as any;
    unknown.hydro.inactiveCells = "frozen";
    unknown.hydro.qeiHeatCapacity = "measured";
    unknown.hydro.coldEquilibrium = { densityCoreRatio: 1.2 };
    const sanitized = migrateFormState(unknown);
    expect(sanitized.hydro.inactiveCells).toBe("passive_fill");
    expect(sanitized.hydro.qeiHeatCapacity).toBe("ideal_gas");
    expect(sanitized.hydro.coldEquilibrium).toEqual({ ...defaultFormState().hydro.coldEquilibrium, densityCoreRatio: 1.2 });

    const cold = coldEquilibriumForm();
    const restored = extractGuiState(generateDeck(cold));
    expect(restored.ok).toBe(true);
    if (restored.ok) expect(migrateFormState(restored.state)).toEqual(cold);
  });
});
