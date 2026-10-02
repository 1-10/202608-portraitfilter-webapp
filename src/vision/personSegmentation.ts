/**
 * Per-pixel person parts — hair, face skin, body skin, clothes — from MediaPipe's
 * SelfieMulticlass segmenter, run locally like the landmarker.
 *
 * This is what lets a filter know "is this skin" and "is this hair" from a model
 * instead of guessing from colour: grey hair sits at skin lightness, a brown fringe
 * inside skin's chroma range, and a warm wall inside skin's hue band, and every
 * colour test misjudges one of them.
 *
 * Best-effort by the same contract as detection: every failure resolves to `null`
 * and filters fall back to their colour heuristics.
 *
 * The model sees a 256x256 input. Run over a whole full-length portrait, a face
 * would be a few dozen of those pixels, so when the face is known the segmenter
 * runs on a square around the head and shoulders instead, and the result is
 * addressed over that rect.
 */
import { packPlanes, type AuxTextureData } from "./auxTexture";
import type { FaceSquare } from "./faceSearch";
import { assetBase, LOAD_TIMEOUT_MS, withTimeout } from "./mediapipeAssets";

/** SelfieMulticlass class order (the model's specification). */
const CLASS_HAIR = 1;
const CLASS_BODY_SKIN = 2;
const CLASS_FACE_SKIN = 3;
const CLASS_CLOTHES = 4;
const CLASS_COUNT = 6;

/** The model's input edge; resampling to it here keeps the downscale high quality. */
const MODEL_INPUT = 256;

/**
 * Crop side / face size, and how far below the face centre the crop is centred,
 * in face sizes. Enough to take in the hair above, the neck and the shoulders,
 * which is where a retouch is judged.
 */
const CROP_SPAN_FACTOR = 4;
const CROP_DROP_FACTOR = 0.6;

type MaskLike = { width: number; height: number; getAsFloat32Array(): Float32Array };
type Segmenter = {
  segment(image: TexImageSource, callback: (result: { confidenceMasks?: MaskLike[] }) => void): void;
  close(): void;
};

export type PersonSegmentation = {
  /** .r hair  .g face skin  .b body skin  .a clothes — model confidences, bottom-up. */
  texture: AuxTextureData;
  /** Region in UV space: xy = min corner, zw = size. y is measured from the bottom. */
  rectUv: [number, number, number, number];
};

let segmenterPromise: Promise<Segmenter | null> | null = null;

async function loadSegmenter(): Promise<Segmenter | null> {
  try {
    const { FilesetResolver, ImageSegmenter } = await import("@mediapipe/tasks-vision");
    const base = assetBase();
    const fileset = await FilesetResolver.forVisionTasks(`${base}wasm`);
    const segmenter = await ImageSegmenter.createFromOptions(fileset, {
      // CPU for the same reason as the landmarker: a GPU delegate opens a second
      // WebGL context next to the renderer's.
      baseOptions: { modelAssetPath: `${base}selfie_multiclass_256x256.tflite`, delegate: "CPU" },
      runningMode: "IMAGE",
      // Soft confidences: a binary category mask would give every gate a staircase edge.
      outputConfidenceMasks: true,
      outputCategoryMask: false,
    });
    return segmenter as unknown as Segmenter;
  } catch (err) {
    if (import.meta.env.DEV) console.warn("[personSegmentation] model unavailable:", err);
    return null;
  }
}

/** Square region around the head and shoulders, clamped to the image; the whole image without a face. */
export function segmentationRegion(
  face: FaceSquare | null,
  width: number,
  height: number,
): { x: number; y: number; w: number; h: number } {
  if (!face) return { x: 0, y: 0, w: width, h: height };
  const half = (face.span * CROP_SPAN_FACTOR) / 2;
  const cy = face.cy + face.span * CROP_DROP_FACTOR;
  const x0 = Math.max(0, Math.floor(face.cx - half));
  const y0 = Math.max(0, Math.floor(cy - half));
  const x1 = Math.min(width, Math.ceil(face.cx + half));
  const y1 = Math.min(height, Math.ceil(cy + half));
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

function toBytes(mask: MaskLike): Uint8ClampedArray {
  const values = mask.getAsFloat32Array();
  const bytes = new Uint8ClampedArray(values.length);
  for (let i = 0; i < values.length; i++) bytes[i] = Math.round(values[i]! * 255);
  return bytes;
}

/**
 * Segments the person in `source` (full resolution, `width` x `height`), around
 * `face` when it is known. Never rejects.
 */
export async function segmentPerson(
  source: CanvasImageSource,
  width: number,
  height: number,
  face: FaceSquare | null,
): Promise<PersonSegmentation | null> {
  try {
    if (!segmenterPromise) segmenterPromise = withTimeout(loadSegmenter(), LOAD_TIMEOUT_MS);
    const segmenter = await segmenterPromise;
    if (!segmenter) return null;

    const region = segmentationRegion(face, width, height);
    if (region.w < 1 || region.h < 1) return null;
    const canvas = document.createElement("canvas");
    const scale = Math.min(1, MODEL_INPUT / Math.max(region.w, region.h));
    canvas.width = Math.max(1, Math.round(region.w * scale));
    canvas.height = Math.max(1, Math.round(region.h * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(source, region.x, region.y, region.w, region.h, 0, 0, canvas.width, canvas.height);

    let texture: AuxTextureData | null = null;
    // The callback form: mask data is only valid inside it, and it avoids the copy
    // the returning overload makes.
    segmenter.segment(canvas, (result) => {
      const masks = result.confidenceMasks ?? [];
      if (masks.length !== CLASS_COUNT) return;
      const { width: mw, height: mh } = masks[0]!;
      texture = packPlanes(
        [
          toBytes(masks[CLASS_HAIR]!),
          toBytes(masks[CLASS_FACE_SKIN]!),
          toBytes(masks[CLASS_BODY_SKIN]!),
          toBytes(masks[CLASS_CLOTHES]!),
        ],
        mw,
        mh,
      );
    });
    if (!texture) return null;
    return {
      texture,
      rectUv: [region.x / width, 1 - (region.y + region.h) / height, region.w / width, region.h / height],
    };
  } catch (err) {
    if (import.meta.env.DEV) console.warn("[personSegmentation] segmentation failed:", err);
    return null;
  }
}

/** Releases the cached model. Safe to call when nothing was ever loaded. */
export async function disposePersonSegmenter(): Promise<void> {
  const pending = segmenterPromise;
  segmenterPromise = null;
  if (!pending) return;
  try {
    (await pending)?.close();
  } catch {
    // Disposal is best-effort.
  }
}
