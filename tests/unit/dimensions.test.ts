import { describe, expect, it } from "vitest";
import { fitWithinLongEdge, resolveExportSize } from "../../src/utils/dimensions";

describe("fitWithinLongEdge", () => {
  it("does not upscale images smaller than the max edge", () => {
    expect(fitWithinLongEdge({ width: 800, height: 600 }, 1280)).toEqual({ width: 800, height: 600 });
  });

  it("scales down a landscape image preserving aspect ratio", () => {
    expect(fitWithinLongEdge({ width: 4000, height: 2000 }, 2000)).toEqual({ width: 2000, height: 1000 });
  });

  it("scales down a portrait image preserving aspect ratio", () => {
    expect(fitWithinLongEdge({ width: 2000, height: 4000 }, 2000)).toEqual({ width: 1000, height: 2000 });
  });

  it("never returns a zero-pixel dimension", () => {
    const result = fitWithinLongEdge({ width: 10000, height: 1 }, 100);
    expect(result.width).toBe(100);
    expect(result.height).toBeGreaterThanOrEqual(1);
  });
});

describe("resolveExportSize", () => {
  const source = { width: 4000, height: 3000 };

  it("returns the (capped) source size for the 'original' preset", () => {
    expect(resolveExportSize(source, "original", 4096, 4096)).toEqual({ width: 4000, height: 3000 });
  });

  it("clamps the 'original' preset to the export max edge", () => {
    expect(resolveExportSize(source, "original", 2000, 4096)).toEqual({ width: 2000, height: 1500 });
  });

  it("resolves the 1920 preset relative to the long edge", () => {
    expect(resolveExportSize(source, "1920", 4096, 4096)).toEqual({ width: 1920, height: 1440 });
  });

  it("resolves the 1280 preset relative to the long edge", () => {
    expect(resolveExportSize(source, "1280", 4096, 4096)).toEqual({ width: 1280, height: 960 });
  });

  it("clamps to a smaller GPU MAX_TEXTURE_SIZE when necessary", () => {
    expect(resolveExportSize(source, "original", 4096, 1024)).toEqual({ width: 1024, height: 768 });
  });

  it("never upscales when the source is smaller than the preset target", () => {
    const small = { width: 800, height: 600 };
    expect(resolveExportSize(small, "1920", 4096, 4096)).toEqual({ width: 800, height: 600 });
  });
});
