/**
 * Where the locally bundled MediaPipe wasm and models live, and the shared helpers
 * every MediaPipe task loader needs. Nothing is fetched from a CDN.
 */

/** Give up rather than make the user wait on a wedged model load. */
export const LOAD_TIMEOUT_MS = 8000;

export function assetBase(): string {
  // Relative to baseURI rather than a rooted path, so the app still works if it is
  // ever deployed under a sub-path.
  return new URL("mediapipe/", document.baseURI).toString();
}

export function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T | null> {
  return Promise.race([promise, new Promise<null>((resolve) => setTimeout(() => resolve(null), ms))]);
}
