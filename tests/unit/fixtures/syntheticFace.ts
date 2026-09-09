import type { NormalizedLandmark } from "../../../src/vision/faceMeshTopology";

/**
 * A synthetic, exactly symmetric face in normalized coordinates, plus a mesh over
 * it.
 *
 * Real landmarks would make every assertion approximate and every failure hard to
 * read. A face that is symmetric to the last digit lets the deformation tests state
 * what they actually mean: that the left and right halves get mirrored treatment,
 * that the field is zero where nothing should move, and that the frame is upright.
 */

const OVAL_COUNT = 16;
const EYE_COUNT = 8;
const LIPS_COUNT = 8;

function ellipse(cx: number, cy: number, rx: number, ry: number, count: number, phase = 0): NormalizedLandmark[] {
  const points: NormalizedLandmark[] = [];
  for (let i = 0; i < count; i++) {
    const t = phase + (i / count) * Math.PI * 2;
    points.push({ x: cx + Math.cos(t) * rx, y: cy + Math.sin(t) * ry });
  }
  return points;
}

const landmarks: NormalizedLandmark[] = [];

// Face oval, starting at the right temple and running clockwise in image space so
// the lowest point (the chin) is unambiguous.
const ovalStart = landmarks.length;
landmarks.push(...ellipse(0.5, 0.5, 0.15, 0.21, OVAL_COUNT));

const eyeAStart = landmarks.length;
landmarks.push(...ellipse(0.44, 0.44, 0.035, 0.018, EYE_COUNT));

const eyeBStart = landmarks.length;
landmarks.push(...ellipse(0.56, 0.44, 0.035, 0.018, EYE_COUNT));

const lipsOuterStart = landmarks.length;
landmarks.push(...ellipse(0.5, 0.62, 0.055, 0.024, LIPS_COUNT));

const lipsInnerStart = landmarks.length;
landmarks.push(...ellipse(0.5, 0.62, 0.03, 0.006, LIPS_COUNT));

const browAStart = landmarks.length;
landmarks.push(...ellipse(0.44, 0.4, 0.04, 0.008, EYE_COUNT));

const browBStart = landmarks.length;
landmarks.push(...ellipse(0.56, 0.4, 0.04, 0.008, EYE_COUNT));

// Fan centres, so the rings below can be triangulated without a real tessellation.
const ovalCentre = landmarks.length;
landmarks.push({ x: 0.5, y: 0.5 });
const eyeACentre = landmarks.length;
landmarks.push({ x: 0.44, y: 0.44 });
const eyeBCentre = landmarks.length;
landmarks.push({ x: 0.56, y: 0.44 });

const range = (start: number, count: number): number[] => Array.from({ length: count }, (_, i) => start + i);

const ovalRing = range(ovalStart, OVAL_COUNT);
const eyeARing = range(eyeAStart, EYE_COUNT);
const eyeBRing = range(eyeBStart, EYE_COUNT);
const lipsOuterRing = range(lipsOuterStart, LIPS_COUNT);
const lipsInnerRing = range(lipsInnerStart, LIPS_COUNT);
const browARing = range(browAStart, EYE_COUNT);
const browBRing = range(browBStart, EYE_COUNT);

function fan(centre: number, ring: readonly number[]): number[] {
  const tris: number[] = [];
  for (let i = 0; i < ring.length; i++) {
    tris.push(centre, ring[i]!, ring[(i + 1) % ring.length]!);
  }
  return tris;
}

/**
 * Oval fan first, then the eye fans, because the rasterizer is last-write-wins and
 * the eye triangles sit inside the oval ones.
 */
const triangles = new Uint16Array([
  ...fan(ovalCentre, ovalRing),
  ...fan(eyeACentre, eyeARing),
  ...fan(eyeBCentre, eyeBRing),
]);

export const SYNTHETIC_FACE = {
  landmarks,
  triangles,
  rings: {
    faceOval: ovalRing,
    leftEye: eyeARing,
    rightEye: eyeBRing,
    lips: lipsOuterRing,
  },
  lipsOuter: lipsOuterRing,
  lipsInner: lipsInnerRing,
  brows: [browARing, browBRing],
};

/** The same face, rotated `degrees` clockwise about the image centre. */
export function rotatedFace(degrees: number): NormalizedLandmark[] {
  const angle = (degrees * Math.PI) / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return landmarks.map((p) => {
    const dx = p.x - 0.5;
    const dy = p.y - 0.5;
    return { x: 0.5 + dx * cos - dy * sin, y: 0.5 + dx * sin + dy * cos };
  });
}
