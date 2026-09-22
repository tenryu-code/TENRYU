import { describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToString } from "react-dom/server";
import { defaultFormState, setInactiveCellMode, type FormState } from "../src/core/deck/formState";
import { ja } from "../src/i18n/ja";
import { coldEquilibriumForm } from "./coldEquilibriumSupport";

// zustand renders the initial store state on the server, so the screens get a
// minimal store whose selectors read a form the test controls.
const store = vi.hoisted(() => ({ form: null as unknown }));
vi.mock("../src/store", () => ({
  useApp: (selector: (state: unknown) => unknown) =>
    selector({
      form: store.form,
      updateForm: () => {},
      fetchTmatGroupBounds: () => {},
      tmatBoundsBusy: false,
      tmatBoundsError: null,
      listRemoteDir: async () => ({ ok: false, path: "", entries: [] }),
    }),
  currentProfile: () => null,
}));

const { default: PhysicsSection } = await import("../src/ui/sections/PhysicsSection");
const { default: MaterialsSection } = await import("../src/ui/sections/MaterialsSection");

function render(form: FormState): { physics: string; materials: string } {
  store.form = form;
  return {
    physics: renderToString(createElement(PhysicsSection)),
    materials: renderToString(createElement(MaterialsSection)),
  };
}

describe("physics and materials screens with the inactive-cell controls", () => {
  it("default mode shows the two selects and neither the details nor a reference-state panel", () => {
    const { physics, materials } = render(defaultFormState());
    expect(physics).toContain(ja.form.hydroInactiveCells);
    expect(physics).toContain(ja.form.qeiHeatCapacity);
    expect(physics).toContain(ja.form.hydroInactivePassiveFillHint);
    expect(physics).not.toContain(ja.form.coldEqAdvanced);
    expect(materials).not.toContain(ja.form.coldRefTitle);
  });

  it("cold_equilibrium shows the details block and one reference-state panel per material", () => {
    const { physics, materials } = render(coldEquilibriumForm());
    expect(physics).toContain(ja.form.coldEqAdvanced);
    expect(physics).toContain(ja.form.hydroInactiveColdEquilibriumHint);
    expect(materials.split(ja.form.coldRefTitle).length - 1).toBe(2);
    expect(materials).toContain(ja.form.coldRefK0);
    expect(materials).toContain(ja.form.coldRefFromInitial);
  });

  it("a material without a reference state and a rigid_wall hint render", () => {
    const f = coldEquilibriumForm();
    delete f.materials[1].coldReference;
    expect(render(f).materials.split(ja.form.coldRefTitle).length - 1).toBe(2);
    const g = defaultFormState();
    setInactiveCellMode(g, "rigid_wall");
    expect(render(g).physics).toContain(ja.form.hydroInactiveRigidWallHint);
  });
});
