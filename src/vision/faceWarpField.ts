/**
 * Bakes the face-mesh deformation into a displacement-field texture.
 *
 * The renderer can only draw one fullscreen triangle per pass — there is no vertex
 * buffer and no way to hand it a mesh. So the mesh is solved here, on the CPU, and
 * the RESULT is uploaded as a small texture that an ordinary fragment shader reads:
 *
 *     out(uv) = source(uv + displacement(uv))
 *
 * Three properties make this work as well as drawing the mesh would:
 *
 * - **Backward map.** The field is rasterized over the DEFORMED vertex positions
 *   carrying `original - deformed`, so every output pixel knows where to fetch
 *   from. Rasterizing the forward map instead leaves holes wherever the mesh
 *   expands.
 * - **Two independent fields, not one.** Displacement is stored per group (face
 *   shape in RG, eyes in BA) so the sliders are uniform weights the shader sums.
 *   Nothing is re-rasterized while the user drags. Summing the groups is an
 *   approximation, exact only where they do not overlap — which holds here,
 *   because the eye field has decayed to zero long before the jaw field rises.
 * - **Normalized UV addressing.** The field is stored over the face rect in
 *   normalized coordinates, so preview, thumbnail and 4096px export all read the
 *   same texture. Nothing is rebuilt when the render resolution changes.
 */
import type { AuxTextureData } from "./auxTexture";
import {
  buildSkirtRing,
  clamp01,
  distance,
  project,
  SKIRT_OUTWARD_FACE_FRACTION,
  smoothstep,
  type FaceFrame,
  type GeomRect,
  type Pt,
} from "./faceMeshTopology";

/** Long edge of the displacement field, in texels. */
export const WARP_FIELD_LONG_EDGE = 512;

/**
 * Half-range of the encoded displacement, as a fraction of face width.
 *
 * Displacement is stored in 8 bits per axis, so this trades range against
 * precision. 0.14 leaves headroom above the largest displacement the rules below
 * can produce (jaw contraction at the oval edge, ~0.10 of face width) and still
 * resolves under a pixel at a 4096px export. Quantization does not show as
 * terracing because the field itself is bilinearly filtered before being applied.
 * The unit test asserts no channel saturates, which is what catches this constant
 * being left behind when the rules get stronger.
 */
const WARP_RANGE_FACE_FRACTION = 0.14;

export type WarpFieldResult = {
  texture: AuxTextureData;
  /**
   * Image-space UV displacement that an encoded channel value of 1.0 represents,
   * per axis. The shader decodes with `(texel - 128/255) * range`.
   */
  rangeU: number;
  rangeV: number;
};

/**
 * The face outline, prepared for nearest-point queries.
 *
 * Slimming needs to know where the SILHOUETTE is, not just how far a point is from
 * the midline — see displaceShape for why that distinction is the whole ballgame.
 */
type Contour = {
  points: Pt[];
  /** Unit inward direction at each outline point. */
  inward: Pt[];
};

function buildContour(points: readonly Pt[], ovalRing: readonly number[]): Contour {
  const ring = ovalRing.map((i) => points[i]).filter((p): p is Pt => !!p);
  let cx = 0;
  let cy = 0;
  for (const p of ring) {
    cx += p.x / ring.length;
    cy += p.y / ring.length;
  }
  const centre = { x: cx, y: cy };
  const inward = ring.map((p) => {
    const dx = centre.x - p.x;
    const dy = centre.y - p.y;
    const length = Math.hypot(dx, dy) || 1;
    return { x: dx / length, y: dy / length };
  });
  return { points: ring, inward };
}

/**
 * Distance to the outline and the inward direction there, both smoothed over the
 * outline rather than taken from the single nearest vertex.
 *
 * Nearest-vertex is the obvious implementation and it is visibly wrong: the outline
 * is 36 points, so as a mesh vertex moves the nearest one JUMPS, and the inward
 * direction jumps with it. The displacement comes out faceted along the jaw, and on
 * a perfectly symmetric face the left and right halves disagree by ~12% purely from
 * which vertex won each tie. A Gaussian weighting over every outline point costs 36
 * exponentials per vertex and removes both.
 */
function nearContour(p: Pt, contour: Contour, sigma: number): { distance: number; inward: Pt } {
  let weightSum = 0;
  let distanceSum = 0;
  let ix = 0;
  let iy = 0;
  const invTwoSigmaSq = 1 / (2 * sigma * sigma);
  for (let i = 0; i < contour.points.length; i++) {
    const d = distance(p, contour.points[i]!);
    const w = Math.exp(-d * d * invTwoSigmaSq);
    weightSum += w;
    distanceSum += w * d;
    ix += w * contour.inward[i]!.x;
    iy += w * contour.inward[i]!.y;
  }
  if (weightSum < 1e-12) {
    // Far outside the kernel's reach; the caller's falloff is zero there anyway.
    return { distance: Infinity, inward: { x: 0, y: 0 } };
  }
  const length = Math.hypot(ix, iy) || 1;
  return { distance: distanceSum / weightSum, inward: { x: ix / length, y: iy / length } };
}

/**
 * Per-vertex displacement for the "face shape" group, in working-image pixels.
 *
 * The slimming term moves the SILHOUETTE inward and leaves the interior of the face
 * alone, and that is the difference between this reading as a slimmer face and
 * reading as a squashed one.
 *
 * The obvious formulation — contract every point in the lower face toward the
 * midline in proportion to its distance from it — is a lateral scale, and a lateral
 * scale is wrong in a way that is easy to miss on a face crop and obvious on a
 * whole head: it compresses the cheek's roundness horizontally while the shading
 * that describes that roundness stays where it was, so the face flattens and reads
 * as melted. It also drags the nose base, the mouth corners and everything else
 * inward, changing features that were not meant to change.
 *
 * Moving the outline instead — full displacement AT the boundary, decaying to zero
 * about a quarter of a face width inside it — changes the shape of the face and
 * nothing else. The interior keeps its own geometry, so its shading stays correct.
 */
function displaceShape(p: Pt, f: FaceFrame, contour: Contour): Pt {
  const halfWidth = f.faceWidth * 0.5;
  const halfHeight = f.faceHeight * 0.5;
  const lat = project(p, f.eyeMid, f.right);
  const vert = project(p, f.eyeMid, f.up);
  const lv = vert / halfHeight;

  // Silhouette contraction, concentrated on the JAW.
  //
  // The vertical ramp used to span 0.6 down to -0.2 of half-height, which reached
  // the temples as well. Measured on the output, that contracted the whole outline
  // by the same factor: face width/height went 0.864 to 0.818 while jaw width over
  // face width stayed at 0.950. In other words it made the face smaller without
  // making the jaw any sharper, which is not what a jawline slider is for.
  //
  // Starting the ramp at the eye line and completing it by half-height down leaves
  // the widest part of the face — the cheekbone — where it was, so the change shows
  // up as jaw shape rather than as overall scale.
  const near = nearContour(p, contour, f.faceWidth * 0.12);
  const boundaryFalloff = 1 - smoothstep(0.0, 0.26, near.distance / f.faceWidth);
  const slim = 0.075 * f.faceWidth * boundaryFalloff * smoothstep(-0.12, -0.6, lv);
  const dx = near.inward.x * slim;
  const dy = near.inward.y * slim;

  let dLat = 0;
  let dVert = 0;

  // Chin, pulled up. Radial around the lowest outline point, so it shortens the
  // face without moving the jaw corners.
  const chinDistance = distance(p, f.chin) / halfHeight;
  dVert += 0.06 * halfHeight * (1 - smoothstep(0.0, 0.55, chinDistance));

  // Nose. Kept small: the nose reads as smaller mostly because the face around it
  // got narrower, and pushing this term instead produces a pinched, boneless nose.
  const noseFalloff = 1 - smoothstep(0.1, 0.42, distance(p, f.noseCenter) / halfWidth);
  dLat += -lat * 0.08 * noseFalloff;

  // Lips, thickened by pushing away from the lip centre along the face's up axis.
  const mouthFalloff = 1 - smoothstep(0.1, 0.6, distance(p, f.lipsCenter) / halfWidth);
  dVert += project(p, f.lipsCenter, f.up) * 0.1 * mouthFalloff;

  return {
    x: dx + dLat * f.right.x + dVert * f.up.x,
    y: dy + dLat * f.right.y + dVert * f.up.y,
  };
}

/**
 * Per-vertex displacement for the "eyes" group, in working-image pixels.
 *
 * A local scale-up around each eye centre, close to UNIFORM. The tempting version
 * pushes vertically about twice as hard as horizontally, on the theory that a bigger
 * eye is mostly a more open one — but scaling the axes unequally changes the eye's
 * aspect ratio, and an eye whose aspect ratio has changed is somebody else's eye. It
 * comes out round where it was almond, which is most of what makes a filtered eye
 * look doll-like. Slightly more vertical than horizontal is as far as this goes.
 *
 * The outer corner also lifts, which is the difference between "enlarged" and
 * "lifted", and unlike an aspect change it is a real thing eyes do.
 */
function displaceEyes(p: Pt, f: FaceFrame): Pt {
  let dx = 0;
  let dy = 0;
  for (const eye of f.eyes) {
    const r = distance(p, eye.center) / eye.radius;
    const w = 1 - smoothstep(0.3, 2.0, r);
    if (w <= 0) continue;

    const lat = project(p, eye.center, f.right);
    const vert = project(p, eye.center, f.up);
    const dLat = lat * 0.16 * w;
    const dVert = vert * 0.2 * w;

    // Outer-corner lift, applied only on the temple side of the eye centre.
    const towardOuter = project(p, eye.center, eye.outerDir) / eye.radius;
    const lift = 0.09 * eye.radius * clamp01(towardOuter) * w;

    dx += dLat * f.right.x + (dVert + lift) * f.up.x;
    dy += dLat * f.right.y + (dVert + lift) * f.up.y;
  }
  return { x: dx, y: dy };
}

/**
 * Fills `out` (2 floats per texel) by barycentric interpolation over `triangles`.
 *
 * Triangles are rasterized in the field's own texel grid at the DEFORMED vertex
 * positions. Overlaps are last-write-wins, which is acceptable because an overlap
 * only happens where two triangles fold over each other and their values there are
 * already close.
 */
function rasterizeField(
  fieldWidth: number,
  fieldHeight: number,
  out: Float32Array,
  posX: Float64Array,
  posY: Float64Array,
  valX: Float64Array,
  valY: Float64Array,
  triangles: Uint32Array,
): void {
  for (let t = 0; t + 2 < triangles.length; t += 3) {
    const i0 = triangles[t]!;
    const i1 = triangles[t + 1]!;
    const i2 = triangles[t + 2]!;
    const x0 = posX[i0]!;
    const y0 = posY[i0]!;
    const x1 = posX[i1]!;
    const y1 = posY[i1]!;
    const x2 = posX[i2]!;
    const y2 = posY[i2]!;

    const area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if (Math.abs(area) < 1e-9) continue;
    const invArea = 1 / area;

    const minX = Math.max(0, Math.floor(Math.min(x0, x1, x2)));
    const maxX = Math.min(fieldWidth - 1, Math.ceil(Math.max(x0, x1, x2)));
    const minY = Math.max(0, Math.floor(Math.min(y0, y1, y2)));
    const maxY = Math.min(fieldHeight - 1, Math.ceil(Math.max(y0, y1, y2)));
    if (minX > maxX || minY > maxY) continue;

    const v0x = valX[i0]!;
    const v0y = valY[i0]!;
    const v1x = valX[i1]!;
    const v1y = valY[i1]!;
    const v2x = valX[i2]!;
    const v2y = valY[i2]!;

    for (let py = minY; py <= maxY; py++) {
      const sy = py + 0.5;
      for (let px = minX; px <= maxX; px++) {
        const sx = px + 0.5;
        const b1 = ((sx - x0) * (y2 - y0) - (x2 - x0) * (sy - y0)) * invArea;
        const b2 = ((x1 - x0) * (sy - y0) - (sx - x0) * (y1 - y0)) * invArea;
        const b0 = 1 - b1 - b2;
        if (b0 < 0 || b1 < 0 || b2 < 0) continue;
        const o = (py * fieldWidth + px) * 2;
        out[o] = b0 * v0x + b1 * v1x + b2 * v2x;
        out[o + 1] = b0 * v0y + b1 * v1y + b2 * v2y;
      }
    }
  }
}

/**
 * Builds the two-group displacement field for one face.
 *
 * `meshTriangles` indexes into `points`; the skirt vertices are appended after
 * them, so the combined index space is [points, skirt].
 */
export function buildWarpFieldCanvas(
  points: readonly Pt[],
  frame: FaceFrame,
  meshTriangles: Uint16Array,
  ovalRing: readonly number[],
  rect: GeomRect,
  imageWidth: number,
  imageHeight: number,
): WarpFieldResult {
  const aspect = rect.w / Math.max(rect.h, 1e-6);
  const fieldWidth = aspect >= 1 ? WARP_FIELD_LONG_EDGE : Math.max(2, Math.round(WARP_FIELD_LONG_EDGE * aspect));
  const fieldHeight = aspect >= 1 ? Math.max(2, Math.round(WARP_FIELD_LONG_EDGE / aspect)) : WARP_FIELD_LONG_EDGE;

  const skirt = buildSkirtRing(points, ovalRing, frame.faceWidth * SKIRT_OUTWARD_FACE_FRACTION);
  const total = points.length + skirt.length;

  // Combined vertex list: the mesh, then the zero-displacement skirt ring.
  const shapeDx = new Float64Array(total);
  const shapeDy = new Float64Array(total);
  const eyeDx = new Float64Array(total);
  const eyeDy = new Float64Array(total);
  const contour = buildContour(points, ovalRing);
  for (let i = 0; i < points.length; i++) {
    const p = points[i]!;
    const s = displaceShape(p, frame, contour);
    shapeDx[i] = s.x;
    shapeDy[i] = s.y;
    const e = displaceEyes(p, frame);
    eyeDx[i] = e.x;
    eyeDy[i] = e.y;
  }

  const triangles = new Uint32Array(meshTriangles.length + ovalRing.length * 6);
  triangles.set(meshTriangles, 0);
  // Stitch the annulus between the oval ring and the skirt ring: two triangles per
  // segment. Both rings are the same length and in the same order, so the strip is
  // a straight zip with no triangulation needed.
  let at = meshTriangles.length;
  for (let i = 0; i < ovalRing.length; i++) {
    const nextI = (i + 1) % ovalRing.length;
    const inner = ovalRing[i]!;
    const innerNext = ovalRing[nextI]!;
    const outer = points.length + i;
    const outerNext = points.length + nextI;
    triangles[at++] = inner;
    triangles[at++] = innerNext;
    triangles[at++] = outer;
    triangles[at++] = innerNext;
    triangles[at++] = outerNext;
    triangles[at++] = outer;
  }

  const toFieldX = (x: number): number => ((x - rect.x) / rect.w) * fieldWidth;
  const toFieldY = (y: number): number => ((y - rect.y) / rect.h) * fieldHeight;

  const rangePx = WARP_RANGE_FACE_FRACTION * frame.faceWidth;

  const bake = (dispX: Float64Array, dispY: Float64Array): Float32Array => {
    const posX = new Float64Array(total);
    const posY = new Float64Array(total);
    const valX = new Float64Array(total);
    const valY = new Float64Array(total);
    for (let i = 0; i < points.length; i++) {
      const p = points[i]!;
      posX[i] = toFieldX(p.x + dispX[i]!);
      posY[i] = toFieldY(p.y + dispY[i]!);
      // The value is the BACKWARD offset: where the deformed pixel came from.
      valX[i] = -dispX[i]!;
      valY[i] = -dispY[i]!;
    }
    for (let i = 0; i < skirt.length; i++) {
      const p = skirt[i]!;
      posX[points.length + i] = toFieldX(p.x);
      posY[points.length + i] = toFieldY(p.y);
    }
    const field = new Float32Array(fieldWidth * fieldHeight * 2);
    rasterizeField(fieldWidth, fieldHeight, field, posX, posY, valX, valY, triangles);
    return field;
  };

  const shapeField = bake(shapeDx, shapeDy);
  const eyeField = bake(eyeDx, eyeDy);

  // Encode so byte 128 decodes to EXACTLY zero: 127 steps per side plus a bias of
  // 128. The usual `*255 + 127.5` cannot land on an integer, and the half-step of
  // drift it leaves would displace the whole frame by a fraction of a pixel.
  const encode = (valuePx: number): number => {
    const n = Math.max(-1, Math.min(1, valuePx / rangePx));
    return Math.round(n * 127) + 128;
  };

  const data = new Uint8Array(fieldWidth * fieldHeight * 4);
  for (let y = 0; y < fieldHeight; y++) {
    // Bottom-up: array uploads ignore UNPACK_FLIP_Y_WEBGL (see auxTexture.ts).
    const srcRow = (fieldHeight - 1 - y) * fieldWidth;
    const dstRow = y * fieldWidth;
    for (let x = 0; x < fieldWidth; x++) {
      const src = (srcRow + x) * 2;
      const dst = (dstRow + x) * 4;
      data[dst] = encode(shapeField[src]!);
      // Negated: image y grows downward while the shader's v grows upward, so a
      // downward image displacement is an upward UV displacement.
      data[dst + 1] = encode(-shapeField[src + 1]!);
      data[dst + 2] = encode(eyeField[src]!);
      data[dst + 3] = encode(-eyeField[src + 1]!);
    }
  }

  return {
    texture: { data, width: fieldWidth, height: fieldHeight },
    // 127 encoded steps span `rangePx`, so one full unit of texel value (255 steps)
    // is 255/127 of it.
    rangeU: ((rangePx / imageWidth) * 255) / 127,
    rangeV: ((rangePx / imageHeight) * 255) / 127,
  };
}
