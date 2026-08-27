/**
 * Local, on-device face-region detection.
 *
 * Everything here is best-effort by contract: `detectFaceMask` resolves to `null`
 * on every failure path and never throws or surfaces an error to the user. A
 * portrait app must keep working on a landscape photo, on a browser without wasm
 * SIMD, and with the model file missing from the server — in all of those cases
 * filters fall back to the shader-side `skinLikelihood()` heuristic.
 *
 * No image data leaves the device: the model and wasm are served from this app's
 * own origin and inference runs locally.
 */
import { buildFaceMaskCanvas, ringFromConnections, type FaceMaskResult, type FaceRings } from "./faceMask";

/** Give up rather than make the user wait on a wedged model load. */
const DETECT_TIMEOUT_MS = 8000;

type Landmarker = {
  detect: (image: TexImageSource) => { faceLandmarks: { x: number; y: number; z?: number }[][] };
  close: () => void;
};

let landmarkerPromise: Promise<Landmarker | null> | null = null;
let cachedRings: FaceRings | null = null;

function assetBase(): string {
  // Relative to baseURI rather than a rooted path, so the app still works if it is
  // ever deployed under a sub-path.
  return new URL("mediapipe/", document.baseURI).toString();
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

    cachedRings = {
      faceOval: ringFromConnections(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL),
      leftEye: ringFromConnections(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE),
      rightEye: ringFromConnections(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE),
      leftBrow: ringFromConnections(FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW),
      rightBrow: ringFromConnections(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW),
      lips: ringFromConnections(FaceLandmarker.FACE_LANDMARKS_LIPS),
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
  return Promise.race([
    promise,
    new Promise<null>((resolve) => setTimeout(() => resolve(null), ms)),
  ]);
}

/**
 * Detects the largest face in `source` and returns its region mask, or `null` if
 * detection is unavailable or finds nothing.
 *
 * Pass the working-resolution canvas rather than the full-size bitmap: landmarks
 * are normalized so the mask is identical either way, and inference is several
 * times faster.
 */
export async function detectFaceMask(
  source: HTMLCanvasElement,
): Promise<(FaceMaskResult & { faceScale: number }) | null> {
  try {
    if (!landmarkerPromise) landmarkerPromise = withTimeout(loadLandmarker(), DETECT_TIMEOUT_MS);
    const landmarker = await landmarkerPromise;
    if (!landmarker || !cachedRings) return null;

    const result = landmarker.detect(source);
    const faces = result?.faceLandmarks ?? [];
    if (faces.length === 0) return null;

    // numFaces is 1, but stay explicit: if several are ever returned, the subject
    // is the largest one, not whichever the model happened to list first.
    let best = faces[0]!;
    if (faces.length > 1) {
      let bestArea = -1;
      for (const face of faces) {
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
        const area = (maxX - minX) * (maxY - minY);
        if (area > bestArea) {
          bestArea = area;
          best = face;
        }
      }
    }

    return buildFaceMaskCanvas(best, cachedRings, source.width / Math.max(1, source.height));
  } catch (err) {
    if (import.meta.env.DEV) console.warn("[faceLandmarks] detection failed:", err);
    return null;
  }
}

/** Releases the cached model. Safe to call when nothing was ever loaded. */
export async function disposeFaceLandmarker(): Promise<void> {
  const pending = landmarkerPromise;
  landmarkerPromise = null;
  cachedRings = null;
  if (!pending) return;
  try {
    (await pending)?.close();
  } catch {
    // Disposal is best-effort.
  }
}
