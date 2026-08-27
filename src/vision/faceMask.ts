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

/** Long edge of the generated mask, in pixels. */
export const FACE_MASK_LONG_EDGE = 256;

export type NormalizedLandmark = { x: number; y: number; z?: number };

export type FaceMaskResult = {
  canvas: HTMLCanvasElement;
  /** Interocular distance as a fraction of the image long edge. */
  faceScale: number;
};

type Connection = { start: number; end: number };

/**
 * Walks MediaPipe's connection pairs into an ordered ring of landmark indices.
 *
 * MediaPipe publishes region outlines as unordered edge lists, not polygons, so
 * they cannot be filled directly. Deriving the ring at runtime also means we never
 * hardcode literal landmark indices, which silently shift between model revisions.
 */
export function ringFromConnections(connections: readonly Connection[]): number[] {
  if (connections.length === 0) return [];
  const next = new Map<number, number>();
  for (const c of connections) next.set(c.start, c.end);

  const first = connections[0]!.start;
  const ring = [first];
  const seen = new Set([first]);
  let cur = next.get(first);
  while (cur !== undefined && cur !== first && !seen.has(cur) && ring.length <= connections.length) {
    ring.push(cur);
    seen.add(cur);
    cur = next.get(cur);
  }
  return ring;
}

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

/** Box blur fallback for browsers without ctx.filter, at this size it is trivial. */
function boxBlurAlpha(data: Uint8ClampedArray, w: number, h: number, radius: number): void {
  if (radius < 1) return;
  const tmp = new Uint8ClampedArray(data.length);
  for (let pass = 0; pass < 2; pass++) {
    const src = pass === 0 ? data : tmp;
    const dst = pass === 0 ? tmp : data;
    const horizontal = pass === 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0;
        let count = 0;
        for (let k = -radius; k <= radius; k++) {
          const sx = horizontal ? Math.min(w - 1, Math.max(0, x + k)) : x;
          const sy = horizontal ? y : Math.min(h - 1, Math.max(0, y + k));
          sum += src[(sy * w + sx) * 4]!;
          count++;
        }
        dst[(y * w + x) * 4] = sum / count;
      }
    }
  }
}

type RegionSpec = {
  ring: readonly number[];
  featherFraction: number;
  subtract?: readonly (readonly number[])[];
  dilateFraction?: number;
};

/**
 * Renders one region into a single-channel byte array.
 *
 * Feather and dilation radii are expressed as fractions of FACE WIDTH rather than
 * image width, so mask softness looks the same whether the face fills the frame or
 * sits small in a full-body shot.
 */
function renderRegion(
  landmarks: readonly NormalizedLandmark[],
  spec: RegionSpec,
  w: number,
  h: number,
  faceWidthPx: number,
): Uint8ClampedArray {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return new Uint8ClampedArray(w * h * 4);

  const featherPx = Math.max(0, faceWidthPx * spec.featherFraction);
  const supportsFilter = typeof ctx.filter === "string";
  if (supportsFilter && featherPx >= 0.5) ctx.filter = `blur(${featherPx.toFixed(2)}px)`;

  ctx.fillStyle = "#fff";
  ctx.strokeStyle = "#fff";
  fillRing(ctx, landmarks, spec.ring, w, h, (spec.dilateFraction ?? 0) * faceWidthPx);

  if (spec.subtract?.length) {
    ctx.globalCompositeOperation = "destination-out";
    ctx.fillStyle = "#000";
    ctx.strokeStyle = "#000";
    for (const sub of spec.subtract) {
      fillRing(ctx, landmarks, sub, w, h, faceWidthPx * 0.02);
    }
    ctx.globalCompositeOperation = "source-over";
  }

  const image = ctx.getImageData(0, 0, w, h);
  if (!supportsFilter && featherPx >= 1) {
    boxBlurAlpha(image.data, w, h, Math.round(featherPx));
  }
  return image.data;
}

export type FaceRings = {
  faceOval: readonly number[];
  leftEye: readonly number[];
  rightEye: readonly number[];
  leftBrow: readonly number[];
  rightBrow: readonly number[];
  lips: readonly number[];
};

/**
 * Composites the region masks into one RGBA canvas ready for GPU upload.
 * Drawn in top-down image coordinates; the vertical flip is applied once at
 * upload time by TextureManager, matching how the source image is uploaded.
 */
export function buildFaceMaskCanvas(
  landmarks: readonly NormalizedLandmark[],
  rings: FaceRings,
  imageAspect: number,
): FaceMaskResult | null {
  if (landmarks.length === 0 || rings.faceOval.length < 3) return null;

  const w = imageAspect >= 1 ? FACE_MASK_LONG_EDGE : Math.max(1, Math.round(FACE_MASK_LONG_EDGE * imageAspect));
  const h = imageAspect >= 1 ? Math.max(1, Math.round(FACE_MASK_LONG_EDGE / imageAspect)) : FACE_MASK_LONG_EDGE;

  const ovalPoints = rings.faceOval.map((i) => landmarks[i]).filter((p): p is NormalizedLandmark => !!p);
  if (ovalPoints.length < 3) return null;
  const ovalBounds = boundsOf(ovalPoints);
  const faceWidthPx = Math.max(1, (ovalBounds.maxX - ovalBounds.minX) * w);

  const features = [rings.leftEye, rings.rightEye, rings.leftBrow, rings.rightBrow, rings.lips].filter(
    (r) => r.length >= 3,
  );

  const skin = renderRegion(
    landmarks,
    { ring: rings.faceOval, featherFraction: 0.04, subtract: features },
    w,
    h,
    faceWidthPx,
  );
  const eyes = renderRegion(
    landmarks,
    { ring: rings.leftEye, featherFraction: 0.015, dilateFraction: 0.01 },
    w,
    h,
    faceWidthPx,
  );
  const eyesRight = renderRegion(
    landmarks,
    { ring: rings.rightEye, featherFraction: 0.015, dilateFraction: 0.01 },
    w,
    h,
    faceWidthPx,
  );
  const lips = renderRegion(
    landmarks,
    { ring: rings.lips, featherFraction: 0.015, dilateFraction: 0.01 },
    w,
    h,
    faceWidthPx,
  );
  const face = renderRegion(landmarks, { ring: rings.faceOval, featherFraction: 0.06 }, w, h, faceWidthPx);

  const out = document.createElement("canvas");
  out.width = w;
  out.height = h;
  const outCtx = out.getContext("2d");
  if (!outCtx) return null;
  const packed = outCtx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    packed.data[o] = skin[o]!;
    packed.data[o + 1] = Math.max(eyes[o]!, eyesRight[o]!);
    packed.data[o + 2] = lips[o]!;
    packed.data[o + 3] = face[o]!;
  }
  outCtx.putImageData(packed, 0, 0);

  // Interocular distance drives line widths and blemish-band frequencies, so that
  // a headshot and a full-body shot get proportionally identical-looking results.
  const leftEyePts = rings.leftEye.map((i) => landmarks[i]).filter((p): p is NormalizedLandmark => !!p);
  const rightEyePts = rings.rightEye.map((i) => landmarks[i]).filter((p): p is NormalizedLandmark => !!p);
  let faceScale = (ovalBounds.maxX - ovalBounds.minX) * 0.5;
  if (leftEyePts.length && rightEyePts.length) {
    const lb = boundsOf(leftEyePts);
    const rb = boundsOf(rightEyePts);
    const lc = { x: (lb.minX + lb.maxX) / 2, y: (lb.minY + lb.maxY) / 2 };
    const rc = { x: (rb.minX + rb.maxX) / 2, y: (rb.minY + rb.maxY) / 2 };
    faceScale = Math.hypot(rc.x - lc.x, rc.y - lc.y);
  }

  return { canvas: out, faceScale };
}
