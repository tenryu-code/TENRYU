import { describe, expect, it } from "vitest";
import { defaultFormState, migrateFormState, validateFormState } from "../src/core/deck/formState";
import { generateDeck } from "../src/core/deck/generate";
import { preset2dRectLaser } from "../src/core/presets";
import { ja } from "../src/i18n/ja";

describe("2D beam direction vectors", () => {
  it("defaults to the minus-z unit vector", () => {
    const beam = defaultFormState().laser.beams[0];
    expect([beam.dirX, beam.dirY, beam.dirZ]).toEqual([0, 0, -1]);
  });

  it("migrates a legacy plus-z axial direction", () => {
    const raw = JSON.parse(JSON.stringify(defaultFormState()));
    delete raw.laser.beams[0].dirX;
    delete raw.laser.beams[0].dirY;
    delete raw.laser.beams[0].dirZ;
    raw.laser.beams[0].axialDirection = "plus_z";

    const migrated = migrateFormState(raw);

    expect([
      migrated.laser.beams[0].dirX,
      migrated.laser.beams[0].dirY,
      migrated.laser.beams[0].dirZ,
    ]).toEqual([0, 0, 1]);
  });

  it("normalizes the emitted 2D direction", () => {
    const f = preset2dRectLaser();
    f.laser.beams[0].dirX = 1;
    f.laser.beams[0].dirY = 0;
    f.laser.beams[0].dirZ = -1;

    expect(generateDeck(f)).toContain("direction=(0.70710678");
  });

  it("rejects a zero 2D direction vector", () => {
    const f = preset2dRectLaser();
    f.laser.beams[0].dirX = 0;
    f.laser.beams[0].dirY = 0;
    f.laser.beams[0].dirZ = 0;

    expect(validateFormState(f)).toContain(ja.validation.beamDirZero(1));
  });
});
