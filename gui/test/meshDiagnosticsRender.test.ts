// The mesh section's diagnostics draw the mesh of every 1D preset without a server check: the
// recommended meshes from Studio's copy of the solver's zoning, the others from Studio's nodes.
import { describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToString } from "react-dom/server";
import type { FormState } from "../src/core/deck/formState";
import { ja } from "../src/i18n/ja";
import { PRESETS_1D } from "../src/core/presets1d";
import { editedIntentForms } from "./zoningIntentForms";

// No server preview: the diagnostics show what Studio computes.
vi.mock("../src/store", () => ({
  useApp: (selector: (state: unknown) => unknown) =>
    selector({
      deck: "",
      meshPreview: null,
      meshPreviewDeck: null,
      meshPreviewBusy: false,
      meshPreviewError: null,
      runMeshPreview: async () => {},
    }),
}));

const { default: MeshDiagnostics } = await import("../src/ui/mesh/MeshDiagnostics");

function render(form: FormState): string {
  return renderToString(createElement(MeshDiagnostics, { form }));
}

const ui = ja.mesh1d.ui;

describe("mesh diagnostics without a server check", () => {
  for (const preset of PRESETS_1D) {
    it(`draws the mesh of the preset ${preset.id}`, () => {
      const f = preset.build();
      const html = render(f);
      expect(html).toContain("<polyline");
      expect(html).not.toContain(ui.diagEmpty);
      if (f.mesh.grid1d === "recommended") {
        expect(html).toContain(ui.diagSource.guiZoning);
        // The laser presets' requirement is on enforce: the note says the solver adds its ceilings.
        expect(html).toContain(ui.diagEnforceNote);
      } else {
        expect(html).toContain(ui.diagSource.gui);
        expect(html).not.toContain(ui.diagEnforceNote);
      }
    });
  }

  it("draws an edited zoning_intent without the enforce note", () => {
    const html = render(editedIntentForms().editedShellIntent);
    expect(html).toContain("<polyline");
    expect(html).toContain(ui.diagSource.guiZoning);
    expect(html).not.toContain(ui.diagEnforceNote);
  });

  it("shows the solver's error for an intent its zoning refuses", () => {
    const f = editedIntentForms().editedShellIntent;
    // Three segments of at least 300 cells do not fit in 400.
    f.mesh.zoningIntent.minCellsPerSegment = 300;
    const html = render(f);
    expect(html).not.toContain("<polyline");
    expect(html).toContain("MESH_SEGMENT_MIN_COUNT_INFEASIBLE");
    expect(html).toContain(ui.diagZoningError("").trim());
  });
});
