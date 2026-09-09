/**
 * Turns MediaPipe face landmarks into the RGBA region mask that filters sample as
 * `uFaceMask`. Split out from the detector so the geometry is unit-testable without
 * a WebGL context, a model download, or a DOM face.
 *
 * Channel contract (must stay in sync with COMMON_UNIFORMS in glslCommon.ts):
 *   .r skin   — face oval minus eyes, brows and lips: smooth/flatten this hard
 *   .g eyes   — keep crisp
 *   .b lips   — keep crisp
 *   .a face   — whole oval, generously feathered: "this is the subject"
 */
import { alphaPlane, blurPlane, packPlanes, type AuxTextureData } from "./auxTexture";
import type { NormalizedLandmark, Pt } from "./faceMeshTopology";

/** Long edge of the generated mask, in pixels. */
export const FACE_MASK_LONG_EDGE = 256;

export type FaceMaskResult = {
  texture: AuxTextureData;
  /** Interocular distance as a fraction of the image long edge. */
  faceScale: number;
};

/** Axis-aligned bounds of the given landmarks, in normalized 0..1 coordinates. */
export function boundsOf(points: readonly NormalizedLandmark[]): {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
} {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  return { minX, minY, maxX, maxY };
}

function fillRing(
  ctx: CanvasRenderingContext2D,
  landmarks: readonly NormalizedLandmark[],
  ring: readonly number[],
  w: number,
  h: number,
  dilatePx: number,
): void {
  if (ring.length < 3) return;
  ctx.beginPath();
  ring.forEach((idx, i) => {
    const p = landmarks[idx];
    if (!p) return;
    const x = p.x * w;
    const y = p.y * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fill();
  if (dilatePx > 0) {
    // Stroking the same path widens the region by half the line width on each
    // side — used to make feature exclusions generous, so smoothing never creeps
    // onto an eyelash or lip edge.
    ctx.lineWidth = dilatePx * 2;
    ctx.lineJoin = "round";
    ctx.stroke();
  }
}

type RegionSpec = {
  ring: readonly number[];
  featherFraction: number;
  subtract?: readonly (readonly number[])[];
  dilateFraction?: number;
};

/**
 * Renders one region into a single-channel coverage plane.
 *
 * Feather and dilation radii are expressed as fractions of FACE WIDTH rather than
 * image width, so mask softness looks the same whether the face fills the frame or
 * sits small in a full-body shot.
 *
 * Feathering is applied to the extracted plane rather than with `ctx.filter`,
 * which is optional in the 2D context and differs between engines.
 */
function renderRegion(
  ctx: CanvasRenderingContext2D,
  landmarks: readonly NormalizedLandmark[],
  spec: RegionSpec,
  w: number,
  h: number,
  faceWidthPx: number,
): Uint8ClampedArray {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = "source-over";
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff";
  ctx.strokeStyle = "#fff";
  fillRing(ctx, landmarks, spec.ring, w, h, (spec.dilateFraction ?? 0) * faceWidthPx);

  if (spec.subtract?.length) {
    ctx.globalCompositeOperation = "destination-out";
    for (const sub of spec.subtract) {
      fillRing(ctx, landmarks, sub, w, h, faceWidthPx * 0.02);
    }
    ctx.globalCompositeOperation = "source-over";
  }

  const plane = alphaPlane(ctx.getImageData(0, 0, w, h).data, w * h);
  blurPlane(plane, w, h, faceWidthPx * spec.featherFraction);
  return plane;
}

export type FaceMaskRings = {
  faceOval: readonly number[];
  leftEye: readonly number[];
  rightEye: readonly number[];
  leftBrow: readonly number[];
  rightBrow: readonly number[];
  /** Outer lip contour. */
  lips: readonly number[];
};

/**
 * Composites the region masks into one packed RGBA buffer ready for GPU upload.
 *
 * Packed as raw bytes rather than through `putImageData`: a 2D canvas stores
 * premultiplied colour, so writing the four masks as an ImageData would quantize
 * .r/.g/.b everywhere .a (the feathered oval) is below 255 — silently degrading
 * exactly the soft border the mask exists to provide.
 */
export function buildFaceMaskTexture(
  landmarks: readonly NormalizedLandmark[],
  rings: FaceMaskRings,
  imageWidth: number,
  imageHeight: number,
): FaceMaskResult | null {
  if (landmarks.length === 0 || rings.faceOval.length < 3) return null;
  const imageAspect = imageWidth / Math.max(imageHeight, 1);

  const w = imageAspect >= 1 ? FACE_MASK_LONG_EDGE : Math.max(1, Math.round(FACE_MASK_LONG_EDGE * imageAspect));
  const h = imageAspect >= 1 ? Math.max(1, Math.round(FACE_MASK_LONG_EDGE / imageAspect)) : FACE_MASK_LONG_EDGE;

  const ovalPoints = rings.faceOval.map((i) => landmarks[i]).filter((p): p is NormalizedLandmark => !!p);
  if (ovalPoints.length < 3) return null;
  const ovalBounds = boundsOf(ovalPoints);
  const faceWidthPx = Math.max(1, (ovalBounds.maxX - ovalBounds.minX) * w);

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;

  const features = [rings.leftEye, rings.rightEye, rings.leftBrow, rings.rightBrow, rings.lips].filter(
    (r) => r.length >= 3,
  );

  const skin = renderRegion(
    ctx,
    landmarks,
    { ring: rings.faceOval, featherFraction: 0.04, subtract: features },
    w,
    h,
    faceWidthPx,
  );
  const eyesLeft = renderRegion(
    ctx,
    landmarks,
    { ring: rings.leftEye, featherFraction: 0.015, dilateFraction: 0.01 },
    w,
    h,
    faceWidthPx,
  );
  const eyesRight = renderRegion(
    ctx,
    landmarks,
    { ring: rings.rightEye, featherFraction: 0.015, dilateFraction: 0.01 },
    w,
    h,
    faceWidthPx,
  );
  const lips = renderRegion(
    ctx,
    landmarks,
    { ring: rings.lips, featherFraction: 0.015, dilateFraction: 0.01 },
    w,
    h,
    faceWidthPx,
  );
  const face = renderRegion(ctx, landmarks, { ring: rings.faceOval, featherFraction: 0.06 }, w, h, faceWidthPx);

  const eyes = new Uint8ClampedArray(w * h);
  for (let i = 0; i < eyes.length; i++) eyes[i] = Math.max(eyesLeft[i]!, eyesRight[i]!);

  // Interocular distance drives line widths and blemish-band frequencies, so that
  // a headshot and a full-body shot get proportionally identical-looking results.
  const leftEyePts = rings.leftEye.map((i) => landmarks[i]).filter((p): p is NormalizedLandmark => !!p);
  const rightEyePts = rings.rightEye.map((i) => landmarks[i]).filter((p): p is NormalizedLandmark => !!p);
  let faceScale = (ovalBounds.maxX - ovalBounds.minX) * 0.5;
  if (leftEyePts.length && rightEyePts.length) {
    const lb = boundsOf(leftEyePts);
    const rb = boundsOf(rightEyePts);
    const lc: Pt = { x: (lb.minX + lb.maxX) / 2, y: (lb.minY + lb.maxY) / 2 };
    const rc: Pt = { x: (rb.minX + rb.maxX) / 2, y: (rb.minY + rb.maxY) / 2 };
    faceScale = Math.hypot(rc.x - lc.x, rc.y - lc.y);
  }

  return { texture: packPlanes([skin, eyes, lips, face], w, h), faceScale };
}
