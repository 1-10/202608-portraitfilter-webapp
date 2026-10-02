/**
 * Local, on-device face-landmark detection.
 *
 * Everything here is best-effort by contract: `detectFace` resolves to `null` on
 * every failure path and never throws or surfaces an error to the user. A portrait
 * app must keep working on a landscape photo, on a browser without wasm SIMD, and
 * with the model file missing from the server — in all of those cases filters fall
 * back to the shader-side `skinLikelihood()` heuristic.
 *
 * No image data leaves the device: the model and wasm are served from this app's
 * own origin and inference runs locally.
 */
import type { FaceTopology } from "./faceGeometry";
import type { FaceMaskRings } from "./faceMask";
import {
  pickInnerRing,
  pickOuterRing,
  ringsFromConnections,
  trianglesFromConnections,
  type Connection,
  type NormalizedLandmark,
} from "./faceMeshTopology";
import { detectTwoPass, type DetectInRegion, type Region } from "./faceSearch";
import { assetBase, LOAD_TIMEOUT_MS, withTimeout } from "./mediapipeAssets";

type Landmarker = {
  detect: (image: TexImageSource) => { faceLandmarks: NormalizedLandmark[][] };
  close: () => void;
};

/**
 * Everything derived from the model's published connection lists. Independent of
 * any particular face, so it is built once when the model loads.
 */
type Topology = {
  maskRings: Omit<FaceMaskRings, "lips">;
  lipsRings: number[][];
  browRings: number[][];
  irisRings: number[][];
  eyeRings: [number[], number[]];
  faceOval: number[];
  triangles: Uint16Array;
};

export type FaceDetection = {
  landmarks: NormalizedLandmark[];
  /** Ring set for `buildFaceMaskTexture`. */
  maskRings: FaceMaskRings;
  /** Mesh and region topology for `buildFaceGeometry`. */
  topology: FaceTopology;
  imageWidth: number;
  imageHeight: number;
};

let landmarkerPromise: Promise<Landmarker | null> | null = null;
let cachedTopology: Topology | null = null;

/** First ring of a connection list, for regions that have exactly one contour. */
function singleRing(connections: readonly Connection[]): number[] {
  return ringsFromConnections(connections)[0] ?? [];
}

async function loadLandmarker(): Promise<Landmarker | null> {
  try {
    // Dynamic import keeps ~2MB of glue out of the initial bundle and means
    // nothing is fetched at all until the user actually loads a photo.
    const vision = await import("@mediapipe/tasks-vision");
    const { FilesetResolver, FaceLandmarker } = vision;
    const base = assetBase();
    const fileset = await FilesetResolver.forVisionTasks(`${base}wasm`);
    const landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: `${base}face_landmarker.task`,
        // CPU, not GPU: the GPU delegate creates its OWN WebGL context alongside
        // the renderer's. Two live contexts plus a 4096px export is exactly how
        // you provoke context loss. Detection runs once per image, in the
        // background after the first preview is on screen.
        delegate: "CPU",
      },
      runningMode: "IMAGE",
      // A ceiling, not a cost: only faces actually present are paid for. More than one
      // is needed because the most confident face is not necessarily the subject.
      numFaces: 5,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });

    const eyeLeft = singleRing(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE);
    const eyeRight = singleRing(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE);
    const browLeft = singleRing(FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW);
    const browRight = singleRing(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW);
    const faceOval = singleRing(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL);

    cachedTopology = {
      maskRings: { faceOval, leftEye: eyeLeft, rightEye: eyeRight, leftBrow: browLeft, rightBrow: browRight },
      // The lips are TWO concentric contours; which is outer depends on the actual
      // landmark positions, so the choice is deferred to detection time.
      lipsRings: ringsFromConnections(FaceLandmarker.FACE_LANDMARKS_LIPS),
      browRings: [browLeft, browRight].filter((r) => r.length >= 3),
      // Iris rings only exist on models built with iris refinement. Absence is
      // normal and only costs the eye catchlight its exact centring, so the slots
      // are kept positional (left, right) and may be empty — filtering them out
      // would silently pair the right iris with the left eye.
      irisRings: [
        singleRing((FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS as Connection[] | undefined) ?? []),
        singleRing((FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS as Connection[] | undefined) ?? []),
      ],
      eyeRings: [eyeLeft, eyeRight],
      faceOval,
      triangles: trianglesFromConnections(FaceLandmarker.FACE_LANDMARKS_TESSELATION),
    };

    return landmarker as unknown as Landmarker;
  } catch (err) {
    if (import.meta.env.DEV) {
      console.warn("[faceLandmarks] model unavailable, falling back to the color heuristic:", err);
    }
    return null;
  }
}

/**
 * One landmarker run on `region` of `source`, resampled to at most `longSide`.
 *
 * The resample canvas is reused across runs; smoothing is forced to high quality
 * because the scout pass exists precisely to avoid aliased downscales.
 */
function regionDetector(landmarker: Landmarker, source: CanvasImageSource): DetectInRegion | null {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  return (region: Region, longSide: number) => {
    const scale = Math.min(1, longSide / Math.max(region.w, region.h));
    canvas.width = Math.max(1, Math.round(region.w * scale));
    canvas.height = Math.max(1, Math.round(region.h * scale));
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(source, region.x, region.y, region.w, region.h, 0, 0, canvas.width, canvas.height);
    return landmarker.detect(canvas)?.faceLandmarks ?? [];
  };
}

/**
 * Detects the largest face in `source`, or returns `null` if detection is
 * unavailable or finds nothing.
 *
 * Pass the FULL-resolution image: the refine pass crops around the face from it, and
 * that is where the landmark precision comes from (see faceSearch.ts). Landmarks are
 * normalized, so everything derived from them works at any resolution.
 */
export async function detectFace(
  source: CanvasImageSource,
  width: number,
  height: number,
): Promise<FaceDetection | null> {
  try {
    if (!landmarkerPromise) landmarkerPromise = withTimeout(loadLandmarker(), LOAD_TIMEOUT_MS);
    const landmarker = await landmarkerPromise;
    if (!landmarker || !cachedTopology) return null;

    const detect = regionDetector(landmarker, source);
    if (!detect) return null;
    const best = detectTwoPass(detect, width, height);
    if (!best) return null;

    const topology = cachedTopology;
    // Ring area is compared in normalized coordinates. The per-axis scale factors
    // are constant across rings, so the ordering — which is all that matters — is
    // the same as it would be in pixels.
    const lipsOuter = pickOuterRing(topology.lipsRings, best);
    const lipsInner = pickInnerRing(topology.lipsRings, best);

    return {
      landmarks: best,
      maskRings: { ...topology.maskRings, lips: lipsOuter },
      topology: {
        triangles: topology.triangles,
        faceOval: topology.faceOval,
        eyes: topology.eyeRings,
        brows: topology.browRings,
        lipsOuter,
        lipsInner,
        irises: topology.irisRings,
      },
      imageWidth: width,
      imageHeight: height,
    };
  } catch (err) {
    if (import.meta.env.DEV) console.warn("[faceLandmarks] detection failed:", err);
    return null;
  }
}

/** Releases the cached model. Safe to call when nothing was ever loaded. */
export async function disposeFaceLandmarker(): Promise<void> {
  const pending = landmarkerPromise;
  landmarkerPromise = null;
  cachedTopology = null;
  if (!pending) return;
  try {
    (await pending)?.close();
  } catch {
    // Disposal is best-effort.
  }
}
