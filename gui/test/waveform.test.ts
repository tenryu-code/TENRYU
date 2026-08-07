import { describe, expect, it } from "vitest";
import { dataToSvg, svgToData, waveBounds } from "../src/ui/WaveformEditor";

describe("waveform coordinate mapping", () => {
  const pts = [
    { t: 0, v: 120 },
    { t: 3, v: 200 },
  ];
  it("bounds pad the data", () => {
    const b = waveBounds(pts);
    expect(b.xMax).toBeCloseTo(3.15, 6);
    expect(b.yMax).toBeCloseTo(230, 6);
  });
  it("data->svg->data round-trips", () => {
    const b = waveBounds(pts);
    const s = dataToSvg({ t: 1.5, v: 100 }, b, 380, 170);
    const d = svgToData(s.x, s.y, b, 380, 170);
    expect(d.t).toBeCloseTo(1.5, 9);
    expect(d.v).toBeCloseTo(100, 9);
  });
});
