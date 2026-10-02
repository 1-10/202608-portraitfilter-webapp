/**
 * Measures this subject's own skin tone (OKLab lightness and a/b), so filters can
 * tell hair from skin, and a flushed cheek from this person's ordinary complexion,
 * without a threshold baked in at authoring time.
 *
 * Chroma alone separates black hair from skin, but not brown hair: a warm fringe
 * can sit inside the skin chroma range, and it is inside the face oval, so a
 * region-plus-chroma test accepts it and an aggressive retouch bleaches the bangs.
 * Lightness separates them cleanly — hair is a fraction of skin's lightness — but
 * only RELATIVE to the subject, because an absolute floor is exactly the mistake
 * that makes a filter work on pale skin and fail on dark skin.
 *
 * So it is measured per image, from pixels the landmarks say are skin, as a MEDIAN:
 * a few samples will inevitably land on a stray lock of hair or a specular
 * highlight, and a median ignores them where a mean would not.
 */
import { distance, type FaceFrame, type Pt } from "./faceMeshTopology";

/** Samples per axis over the face bounding box. */
const GRID = 20;

/** This subject's median skin colour, in OKLab. */
export type SkinTone = { l: number; a: number; b: number };

/**
 * Used when the measurement cannot be taken. Mid-tone lightness — neither gate nor
 * free pass — and the a/b of typical skin, so a relative redness test stays inert.
 */
export const DEFAULT_SKIN_TONE: SkinTone = { l: 0.62, a: 0.05, b: 0.033 };

function srgbChannelToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** OKLab of an sRGB byte triple. Mirrors linearRgbToOklab in the shaders. */
function oklab(r: number, g: number, b: number): SkinTone {
  const lr = srgbChannelToLinear(r / 255);
  const lg = srgbChannelToLinear(g / 255);
  const lb = srgbChannelToLinear(b / 255);
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  return {
    l: 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    a: 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    b: 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  };
}

function pointInPolygon(p: Pt, polygon: readonly Pt[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i]!;
    const b = polygon[j]!;
    if (a.y > p.y !== b.y > p.y && p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x) {
      inside = !inside;
    }
  }
  return inside;
}

/**
 * Per-channel median OKLab of the subject's skin, or `null` if it cannot be read.
 *
 * `source` must be the same canvas the landmarks were detected on, since the
 * landmark pixel positions are taken at its resolution.
 */
export function measureSkinTone(
  source: HTMLCanvasElement,
  points: readonly Pt[],
  ovalRing: readonly number[],
  frame: FaceFrame,
): SkinTone | null {
  const oval = ovalRing.map((i) => points[i]).filter((p): p is Pt => !!p);
  if (oval.length < 3) return null;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of oval) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }
  const x0 = Math.max(0, Math.floor(minX));
  const y0 = Math.max(0, Math.floor(minY));
  const x1 = Math.min(source.width, Math.ceil(maxX));
  const y1 = Math.min(source.height, Math.ceil(maxY));
  if (x1 - x0 < GRID || y1 - y0 < GRID) return null;

  const ctx = source.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  const width = x1 - x0;
  const height = y1 - y0;
  const image = ctx.getImageData(x0, y0, width, height);

  // Features are excluded generously: an eyebrow or a lip is skin-adjacent and far
  // from skin-coloured, and including them would drag the median down.
  const exclusions = [
    ...frame.eyes.map((eye) => ({ center: eye.center, radius: eye.radius * 1.8 })),
    { center: frame.lipsCenter, radius: frame.faceWidth * 0.3 },
  ];

  const samples: SkinTone[] = [];
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      const px = minX + ((gx + 0.5) / GRID) * (maxX - minX);
      const py = minY + ((gy + 0.5) / GRID) * (maxY - minY);
      const at: Pt = { x: px, y: py };
      if (!pointInPolygon(at, oval)) continue;
      if (exclusions.some((e) => distance(at, e.center) < e.radius)) continue;
      const ix = Math.min(width - 1, Math.max(0, Math.round(px - x0)));
      const iy = Math.min(height - 1, Math.max(0, Math.round(py - y0)));
      const o = (iy * width + ix) * 4;
      samples.push(oklab(image.data[o]!, image.data[o + 1]!, image.data[o + 2]!));
    }
  }
  if (samples.length < 12) return null;
  const median = (channel: keyof SkinTone): number => {
    const values = samples.map((t) => t[channel]).sort((x, y) => x - y);
    return values[Math.floor(values.length / 2)]!;
  };
  return { l: median("l"), a: median("a"), b: median("b") };
}
