/**
 * Two-pass face detection: find the subject at several scales, then re-detect on a
 * full-resolution crop around it.
 *
 * The landmarker is itself "find a face -> crop around it -> fit 478 points", and its
 * finder looks at the whole input shrunk to a small square. What this module adds is
 * the layer that chooses what resolution that finder sees, because both directions
 * fail:
 *
 *   too large  the landmarker's own downscale aliases, and a clear frontal face is
 *              missed
 *   too small  the face is a dozen pixels and cannot be located
 *
 * and the good resolution is NOT monotonic — one photo passes at 512 and fails at
 * 1024, another the reverse. So no single size is chosen; a ladder is searched.
 *
 *   pass 1 (scout)   the whole image at EVERY rung of the ladder, collecting every
 *                    face found. Not stopped at the first rung that passes: which
 *                    faces are visible changes with the rung, so stopping early
 *                    makes the chosen subject depend on resolution.
 *                    When the whole image yields nothing, the same is done over
 *                    square windows of it (SCOUT_WINDOW_FRACTIONS). In a full-length
 *                    shot the face is so small a fraction of the frame that the
 *                    finder's fixed input square loses it at EVERY rung; a window
 *                    makes the face a larger fraction of what the finder sees,
 *                    which no rescale of the whole frame can do.
 *   pass 2 (refine)  a square of REFINE_SPAN_FACTOR x the subject's size, cut from
 *                    the source at full resolution, tried from the LARGEST rung
 *                    down (the crop is already just the face, so more detail is
 *                    better and it is too small to alias). Stops at the first rung
 *                    that finds the same face.
 *
 * Pass 1 only locates; in a full-length portrait its face is tens of pixels across
 * and its points are imprecise. If pass 2 fails the result is "no face" rather than
 * the coarse pass-1 points, which downstream code would otherwise trust as exact.
 *
 * The detector is injected, so the coordinate round trips and the subject matching
 * are testable without a model.
 */
import type { NormalizedLandmark } from "./faceMeshTopology";

/** A region of the source image, in source pixels. */
export type Region = { x: number; y: number; w: number; h: number };

/**
 * Runs the landmarker once on `region` of the source, resampled so its long edge is
 * at most `longSide`, and returns every face found with coordinates normalized to
 * that region. Never upsamples.
 */
export type DetectInRegion = (region: Region, longSide: number) => NormalizedLandmark[][];

/** Bottom of the ladder, as a long edge in pixels. Bounds the search; carries no other meaning. */
const LADDER_FLOOR = 256;
/** One octave per rung. There is no evidence the pass/fail boundary is finer than that. */
const LADDER_STEP = 2;

/**
 * Refine crop side / subject size. Pass 1's size estimate is coarse, and twice the
 * size keeps the face inside the crop when it is off by up to a factor of two. Too
 * large costs little, since pass 2 re-chooses the resolution anyway.
 */
const REFINE_SPAN_FACTOR = 2;

/**
 * A pass-2 face is the subject when its centre moved less than this fraction of the
 * smaller of the two face sizes. Two distinct faces are at least (a + b) / 2 apart,
 * so half the smaller size cannot reach a neighbour.
 */
const SAME_FACE_SHIFT = 0.5;

/**
 * Scout window sides as fractions of the image's short edge, tried as tiers, larger
 * first; a tier runs only if every larger one found nothing. Windows overlap their
 * neighbours by half, so a face straddling one window's border sits inside the next.
 */
const SCOUT_WINDOW_FRACTIONS = [1, 0.6] as const;

/** Positions along one axis so windows of `size` cover `length` with half-window overlap. */
function axisPositions(length: number, size: number): number[] {
  if (size >= length) return [0];
  const steps = Math.ceil((length - size) / (size / 2));
  return Array.from({ length: steps + 1 }, (_, i) => Math.round(((length - size) * i) / steps));
}

/** Scout regions as tiers: the whole image, then each window size in turn. */
export function scoutTiers(width: number, height: number): Region[][] {
  const tiers: Region[][] = [[{ x: 0, y: 0, w: width, h: height }]];
  const shortEdge = Math.min(width, height);
  for (const fraction of SCOUT_WINDOW_FRACTIONS) {
    const size = Math.round(shortEdge * fraction);
    // A window the size of the whole image repeats the first tier.
    if (size < 64 || (size === width && size === height)) continue;
    const tier: Region[] = [];
    for (const y of axisPositions(height, size)) {
      for (const x of axisPositions(width, size)) tier.push({ x, y, w: size, h: size });
    }
    tiers.push(tier);
  }
  return tiers;
}

/** `LADDER_FLOOR` up by `LADDER_STEP` to below `longSide`, then `longSide` itself. Ascending. */
export function scaleLadder(longSide: number): number[] {
  const rungs: number[] = [];
  for (let side = LADDER_FLOOR; side < longSide; side *= LADDER_STEP) rungs.push(side);
  rungs.push(longSide);
  return rungs;
}

/** Face square in source pixels: centre and the larger side of the bounding box. */
export type FaceSquare = { cx: number; cy: number; span: number };

export function faceSquare(face: readonly NormalizedLandmark[], width: number, height: number): FaceSquare {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of face) {
    minX = Math.min(minX, p.x * width);
    minY = Math.min(minY, p.y * height);
    maxX = Math.max(maxX, p.x * width);
    maxY = Math.max(maxY, p.y * height);
  }
  return { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, span: Math.max(maxX - minX, maxY - minY) };
}

/** Maps landmarks normalized to `region` back to coordinates normalized to the full image. */
export function remapLandmarks(
  face: readonly NormalizedLandmark[],
  region: Region,
  width: number,
  height: number,
): NormalizedLandmark[] {
  return face.map((p) => ({
    x: (region.x + p.x * region.w) / width,
    y: (region.y + p.y * region.h) / height,
    // z is normalized by the input width, so it scales with the crop as x does, and
    // is relative to the head rather than the crop origin, so it is not translated.
    ...(p.z === undefined ? {} : { z: (p.z * region.w) / width }),
  }));
}

/** Square crop of `span` around a face, clamped to the image. */
export function cropAround(face: FaceSquare, span: number, width: number, height: number): Region {
  const half = span / 2;
  const x0 = Math.max(0, Math.floor(face.cx - half));
  const y0 = Math.max(0, Math.floor(face.cy - half));
  const x1 = Math.min(width, Math.ceil(face.cx + half));
  const y1 = Math.min(height, Math.ceil(face.cy + half));
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

function sameFace(a: FaceSquare, b: FaceSquare): boolean {
  const shift = Math.hypot(a.cx - b.cx, a.cy - b.cy);
  return shift < SAME_FACE_SHIFT * Math.min(a.span, b.span);
}

/**
 * The subject's landmarks normalized to the full image, or `null`.
 *
 * The subject is the LARGEST face found in pass 1.
 */
export function detectTwoPass(detect: DetectInRegion, width: number, height: number): NormalizedLandmark[] | null {
  let scouted: NormalizedLandmark[][] = [];
  for (const tier of scoutTiers(width, height)) {
    scouted = tier.flatMap((region) =>
      scaleLadder(Math.max(region.w, region.h)).flatMap((rung) =>
        detect(region, rung).map((face) => remapLandmarks(face, region, width, height)),
      ),
    );
    if (scouted.length > 0) break;
  }
  if (scouted.length === 0) return null;
  const subject = scouted
    .map((face) => faceSquare(face, width, height))
    .reduce((best, sq) => (sq.span > best.span ? sq : best));
  if (subject.span <= 0) return null;

  const crop = cropAround(subject, subject.span * REFINE_SPAN_FACTOR, width, height);
  if (crop.w < 1 || crop.h < 1) return null;
  for (const rung of scaleLadder(Math.max(crop.w, crop.h)).reverse()) {
    for (const face of detect(crop, rung)) {
      const mapped = remapLandmarks(face, crop, width, height);
      // Pass 2 can also catch someone else at the edge of the crop, so the face has
      // to be matched to the subject, not just taken.
      if (sameFace(faceSquare(mapped, width, height), subject)) return mapped;
    }
  }
  return null;
}
