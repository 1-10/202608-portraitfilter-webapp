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

/** Give up rather than make the user wait on a wedged model load. */
const DETECT_TIMEOUT_MS = 8000;

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

function assetBase(): string {
  // Relative to baseURI rather than a rooted path, so the app still works if it is
  // ever deployed under a sub-path.
  return new URL("mediapipe/", document.baseURI).toString();
}

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
        // you provoke context loss. Detection runs once per image on a <=1280px
        // still, where CPU inference is fast enough to be invisible.
        delegate: "CPU",
      },
      runningMode: "IMAGE",
      numFaces: 1,
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

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T | null> {
  return Promise.race([promise, new Promise<null>((resolve) => setTimeout(() => resolve(null), ms))]);
}

/** Bounding-box area of a landmark set, used to pick the subject among several faces. */
function boundsArea(face: readonly NormalizedLandmark[]): number {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of face) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }
  return (maxX - minX) * (maxY - minY);
}

/**
 * Detects the largest face in `source`, or returns `null` if detection is
 * unavailable or finds nothing.
 *
 * Pass the working-resolution canvas rather than the full-size bitmap: landmarks
 * are normalized so everything derived from them is identical either way, and
 * inference is several times faster.
 */
export async function detectFace(source: HTMLCanvasElement): Promise<FaceDetection | null> {
  try {
    if (!landmarkerPromise) landmarkerPromise = withTimeout(loadLandmarker(), DETECT_TIMEOUT_MS);
    const landmarker = await landmarkerPromise;
    if (!landmarker || !cachedTopology) return null;

    const result = landmarker.detect(source);
    const faces = result?.faceLandmarks ?? [];
    if (faces.length === 0) return null;

    // numFaces is 1, but stay explicit: if several are ever returned, the subject
    // is the largest one, not whichever the model happened to list first.
    let best = faces[0]!;
    if (faces.length > 1) {
      let bestArea = -1;
      for (const face of faces) {
        const area = boundsArea(face);
        if (area > bestArea) {
          bestArea = area;
          best = face;
        }
      }
    }

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
      imageWidth: source.width,
      imageHeight: source.height,
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
