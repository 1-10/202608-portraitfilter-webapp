import { describe, expect, it } from "vitest";
import type { NormalizedLandmark } from "../../src/vision/faceMeshTopology";
import {
  cropAround,
  detectTwoPass,
  remapLandmarks,
  scaleLadder,
  scoutTiers,
  type DetectInRegion,
  type Region,
} from "../../src/vision/faceSearch";

/** A square "face" as four corner landmarks, in full-image pixels. */
type PxFace = { cx: number; cy: number; span: number };

function cornersIn(face: PxFace, region: Region): NormalizedLandmark[] {
  const h = face.span / 2;
  return [
    [face.cx - h, face.cy - h],
    [face.cx + h, face.cy - h],
    [face.cx - h, face.cy + h],
    [face.cx + h, face.cy + h],
  ].map(([x, y]) => ({ x: (x! - region.x) / region.w, y: (y! - region.y) / region.h }));
}

/**
 * A fake landmarker over a W x H image. A face is "seen" when its span in the
 * resampled input lies within [minPx, maxPx] and it lies entirely inside the region,
 * which models both failure directions the two-pass search exists for.
 */
function fakeDetector(faces: PxFace[], minPx: number, maxPx: number, calls: string[] = []): DetectInRegion {
  return (region, longSide) => {
    const scale = Math.min(1, longSide / Math.max(region.w, region.h));
    calls.push(`${region.w}x${region.h}@${longSide}`);
    return faces
      .filter((f) => {
        const px = f.span * scale;
        const h = f.span / 2;
        const inside =
          f.cx - h >= region.x && f.cx + h <= region.x + region.w && f.cy - h >= region.y && f.cy + h <= region.y + region.h;
        return inside && px >= minPx && px <= maxPx;
      })
      .map((f) => cornersIn(f, region));
  };
}

describe("scaleLadder", () => {
  it("climbs by octaves from 256 and ends at the long side itself", () => {
    expect(scaleLadder(3840)).toEqual([256, 512, 1024, 2048, 3840]);
    expect(scaleLadder(200)).toEqual([200]);
  });
});

describe("scoutTiers", () => {
  it("starts with the whole image, then covers it with square windows that stay inside it", () => {
    const [w, h] = [2160, 3840];
    const tiers = scoutTiers(w, h);
    expect(tiers[0]).toEqual([{ x: 0, y: 0, w, h }]);
    for (const tier of tiers.slice(1)) {
      const size = tier[0]!.w;
      for (const r of tier) {
        expect(r.w).toBe(size);
        expect(r.h).toBe(size);
        expect(r.x + r.w).toBeLessThanOrEqual(w);
        expect(r.y + r.h).toBeLessThanOrEqual(h);
      }
      for (const [x, y] of [[0, 0], [w - 1, h - 1], [w / 2, h / 2], [0, h - 1]] as const) {
        expect(tier.some((r) => x >= r.x && x < r.x + size && y >= r.y && y < r.y + size)).toBe(true);
      }
    }
    expect(tiers.slice(1).map((t) => t[0]!.w)).toEqual([2160, 1296]);
  });
});

describe("remapLandmarks", () => {
  it("maps region-normalized points back into the full frame", () => {
    const region = { x: 100, y: 300, w: 400, h: 400 };
    const [p] = remapLandmarks([{ x: 0.5, y: 0.25, z: 0.1 }], region, 800, 1600);
    expect(p!.x).toBeCloseTo(300 / 800);
    expect(p!.y).toBeCloseTo(400 / 1600);
    expect(p!.z).toBeCloseTo((0.1 * 400) / 800);
  });
});

describe("cropAround", () => {
  it("clamps the crop to the image", () => {
    expect(cropAround({ cx: 50, cy: 50, span: 40 }, 200, 300, 300)).toEqual({ x: 0, y: 0, w: 150, h: 150 });
  });
});

describe("detectTwoPass", () => {
  const [W, H] = [2160, 3840];

  it("returns null rather than coarse pass-1 points when the refine pass cannot see the face", () => {
    // Visible only when its resampled span is 40-60px: true at the 512 rung of the
    // whole image (span 300 * 512/3840 = 40px), false at every other whole-image rung.
    const face = { cx: 1080, cy: 900, span: 300 };
    const result = detectTwoPass(fakeDetector([face], 40, 60), W, H);
    // The refine crop is 600px; 300px there is out of range, but the 256 rung
    // (300 * 256/600 = 128) is not in range either — so pass 2 must fail cleanly.
    expect(result).toBeNull();
  });

  it("refines on a full-resolution crop and returns points in full-image space", () => {
    const face = { cx: 1080, cy: 900, span: 300 };
    const calls: string[] = [];
    const result = detectTwoPass(fakeDetector([face], 20, 400, calls), W, H);
    expect(result).not.toBeNull();
    const xs = result!.map((p) => p.x * W);
    const ys = result!.map((p) => p.y * H);
    expect(Math.min(...xs)).toBeCloseTo(930);
    expect(Math.max(...ys)).toBeCloseTo(1050);
    // The last call is pass 2 on the 600px crop, tried from its largest rung first.
    expect(calls.at(-1)).toBe("600x600@600");
  });

  it("picks the largest face as the subject and ignores a neighbour caught in the crop", () => {
    const subject = { cx: 1000, cy: 1000, span: 400 };
    const neighbour = { cx: 1500, cy: 1000, span: 200 };
    const result = detectTwoPass(fakeDetector([neighbour, subject], 10, 2000), W, H);
    const cx = (Math.min(...result!.map((p) => p.x)) + Math.max(...result!.map((p) => p.x))) / 2;
    expect(cx * W).toBeCloseTo(1000);
  });

  it("falls back to windows when the face is too small a fraction of the whole frame", () => {
    // A fake finder that, like the real one, needs the face to be at least 12% of the
    // region it is given, whatever the resolution.
    const face = { cx: 1080, cy: 700, span: 300 };
    const fraction: DetectInRegion = (region, longSide) =>
      face.span / Math.max(region.w, region.h) >= 0.12 ? fakeDetector([face], 1, 1e6)(region, longSide) : [];
    const result = detectTwoPass(fraction, W, H);
    expect(result).not.toBeNull();
    const cy = (Math.min(...result!.map((p) => p.y)) + Math.max(...result!.map((p) => p.y))) / 2;
    expect(cy * H).toBeCloseTo(700);
  });

  it("returns null when nothing is found", () => {
    expect(detectTwoPass(fakeDetector([], 1, 1e6), W, H)).toBeNull();
  });
});
