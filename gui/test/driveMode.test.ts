import { describe, expect, it } from "vitest";
import { defaultFormState, driveMode, setDriveMode } from "../src/core/deck/formState";

describe("driveMode / setDriveMode", () => {
  it("default form (laser off, vacuum outer) is none", () => {
    const f = defaultFormState();
    f.laser.enabled = false;
    expect(driveMode(f)).toBe("none");
  });

  it("laser mode enables the laser and clears a marshak outer", () => {
    const f = defaultFormState();
    f.radiation.outerR = "marshak";
    setDriveMode(f, "laser");
    expect(f.laser.enabled).toBe(true);
    expect(f.radiation.outerR).toBe("vacuum");
    expect(driveMode(f)).toBe("laser");
  });

  it("radiation_tr mode disables the laser and sets marshak", () => {
    const f = defaultFormState();
    f.laser.enabled = true;
    setDriveMode(f, "radiation_tr");
    expect(f.laser.enabled).toBe(false);
    expect(f.radiation.outerR).toBe("marshak");
    expect(driveMode(f)).toBe("radiation_tr");
  });

  it("pressure mode disables laser, clears marshak, and round-trips to laser", () => {
    const f = defaultFormState();
    f.laser.enabled = true;
    f.radiation.outerR = "marshak";
    setDriveMode(f, "pressure");
    expect(f.laser.enabled).toBe(false);
    expect(f.radiation.outerR).not.toBe("marshak");
    expect(f.hydro.boundary1d).toBe("pressure");
    expect(driveMode(f)).toBe("pressure");
    setDriveMode(f, "laser");
    expect(f.hydro.boundary1d).toBe("free");
    expect(driveMode(f)).toBe("laser");
  });

  it("none mode clears a pressure boundary", () => {
    const f = defaultFormState();
    setDriveMode(f, "pressure");
    setDriveMode(f, "none");
    expect(f.hydro.boundary1d).toBe("free");
    expect(driveMode(f)).toBe("none");
  });

  it("none mode clears both", () => {
    const f = defaultFormState();
    f.laser.enabled = true;
    setDriveMode(f, "none");
    expect(driveMode(f)).toBe("none");
    setDriveMode(f, "radiation_tr");
    setDriveMode(f, "none");
    expect(f.laser.enabled).toBe(false);
    expect(f.radiation.outerR).toBe("vacuum");
  });
});
