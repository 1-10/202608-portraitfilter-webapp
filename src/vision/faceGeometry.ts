/**
 * Assembles everything the makeup filter needs from one detection: the mesh
 * deformation field and the makeup coverage masks, both baked over a shared face
 * rect.
 *
 * All three textures are addressed in normalized coordinates, so they are built
 * ONCE per image and reused unchanged for the preview, the thumbnails and the
 * full-resolution export. Nothing here re-runs while the user drags a slider.
 */
import type { AuxTextureData } from "./auxTexture";
import { buildMakeupMasks, type MakeupRings } from "./faceMakeupMasks";
import {
  buildFaceFrame,
  buildSkirtRing,
  computeGeomRect,
  SKIRT_OUTWARD_FACE_FRACTION,
  type FaceFrame,
  type NormalizedLandmark,
} from "./faceMeshTopology";
import { buildWarpFieldCanvas } from "./faceWarpField";
import { DEFAULT_SKIN_TONE, measureSkinTone, type SkinTone } from "./skinTone";

/**
 * The mesh topology and region rings, derived once from the model's published
 * connection lists. Held separately from the landmarks because it is the same for
 * every face the model will ever return.
 */
export type FaceTopology = {
  triangles: Uint16Array;
  faceOval: readonly number[];
  eyes: readonly [readonly number[], readonly number[]];
  brows: readonly (readonly number[])[];
  lipsOuter: readonly number[];
  lipsInner: readonly number[];
  irises: readonly (readonly number[])[];
};

export type FaceGeometry = {
  warp: AuxTextureData;
  makeupA: AuxTextureData;
  makeupB: AuxTextureData;
  /**
   * The face rect in the shader's UV space: `xy` is the min corner, `zw` the size.
   * `y` is flipped relative to image space because texture v grows upward.
   */
  rectUv: [number, number, number, number];
  /** Image-UV displacement represented by a full unit of encoded channel value. */
  warpRange: [number, number];
  /** This subject's own median skin colour, in OKLab. */
  skinTone: SkinTone;
};

/**
 * Fraction of face width the geometry rect is grown by, beyond the skirt ring.
 *
 * The margin guarantees a border of zero-displacement, fully transparent texels
 * around the baked content, so bilinear sampling at the rect edge decays to
 * "no effect" instead of smearing the edge value outward.
 */
const RECT_MARGIN_FACE_FRACTION = 0.06;

/**
 * `source` must be the canvas the landmarks were detected on: its dimensions define
 * the pixel space every measurement below works in, and its pixels are read to
 * measure the subject's skin lightness.
 */
export function buildFaceGeometry(
  source: HTMLCanvasElement,
  landmarks: readonly NormalizedLandmark[],
  topology: FaceTopology,
): FaceGeometry | null {
  const imageWidth = source.width;
  const imageHeight = source.height;
  const built = buildFaceFrame(
    landmarks,
    {
      faceOval: topology.faceOval,
      leftEye: topology.eyes[0],
      rightEye: topology.eyes[1],
      lips: topology.lipsOuter,
    },
    imageWidth,
    imageHeight,
  );
  if (!built) {
    if (import.meta.env.DEV) console.warn("[faceGeometry] no face frame: rings too degenerate");
    return null;
  }
  const { frame, points } = built;
  if (topology.triangles.length === 0) {
    if (import.meta.env.DEV) console.warn("[faceGeometry] no mesh triangles recovered from the tessellation");
    return null;
  }

  // The rect must contain the skirt, not just the oval: the displacement field
  // decays across the annulus between them, and cropping it would reinstate the
  // hard boundary the skirt exists to remove.
  const skirt = buildSkirtRing(points, topology.faceOval, frame.faceWidth * SKIRT_OUTWARD_FACE_FRACTION);
  const rect = computeGeomRect(skirt, frame.faceWidth * RECT_MARGIN_FACE_FRACTION);

  const warp = buildWarpFieldCanvas(
    points,
    frame,
    topology.triangles,
    topology.faceOval,
    rect,
    imageWidth,
    imageHeight,
  );

  const makeupRings: MakeupRings = {
    lipsOuter: topology.lipsOuter,
    lipsInner: topology.lipsInner,
    brows: topology.brows,
    faceOval: topology.faceOval,
    irises: topology.irises,
  };
  const makeup = buildMakeupMasks(points, frame, makeupRings, rect);
  if (!makeup) {
    if (import.meta.env.DEV) console.warn("[faceGeometry] makeup masks unavailable (no 2D canvas context)");
    return null;
  }
  return {
    warp: warp.texture,
    makeupA: makeup.a,
    makeupB: makeup.b,
    rectUv: [
      rect.x / imageWidth,
      // Image y grows downward, UV v grows upward, so the rect's UV origin is its
      // BOTTOM edge measured from the top of the image.
      1 - (rect.y + rect.h) / imageHeight,
      rect.w / imageWidth,
      rect.h / imageHeight,
    ],
    warpRange: [warp.rangeU, warp.rangeV],
    skinTone: measureSkinTone(source, points, topology.faceOval, frame) ?? DEFAULT_SKIN_TONE,
  };
}

/** Re-exported so callers do not need to reach into the topology module. */
export type { FaceFrame };
