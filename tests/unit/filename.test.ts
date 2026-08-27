import { describe, expect, it } from "vitest";
import { buildExportFilename, formatTimestamp } from "../../src/utils/filename";

describe("formatTimestamp", () => {
  it("formats as YYYYMMDD-HHmmss with zero-padding", () => {
    const date = new Date(2026, 0, 5, 3, 7, 9); // 2026-01-05 03:07:09 local
    expect(formatTimestamp(date)).toBe("20260105-030709");
  });
});

describe("buildExportFilename", () => {
  it("builds the expected pattern for jpeg", () => {
    const date = new Date(2026, 7, 21, 15, 30, 0);
    expect(buildExportFilename("rotoscope", "image/jpeg", date)).toBe("portrait-rotoscope-20260821-153000.jpg");
  });

  it("maps output formats to their file extensions", () => {
    const date = new Date(2026, 0, 1, 0, 0, 0);
    expect(buildExportFilename("watercolor", "image/png", date)).toMatch(/\.png$/);
    expect(buildExportFilename("watercolor", "image/webp", date)).toMatch(/\.webp$/);
  });

  it("slugifies filter ids that contain unexpected characters", () => {
    const date = new Date(2026, 0, 1, 0, 0, 0);
    expect(buildExportFilename("Soft Anime!!", "image/jpeg", date)).toBe("portrait-soft-anime-20260101-000000.jpg");
  });
});
