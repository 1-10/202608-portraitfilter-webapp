import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import { defaultParamValues } from "../filters/definitions";
import { exportImage } from "../rendering/exportImage";
import { WebGLRenderer, WebGLUnsupportedError } from "../rendering/WebGLRenderer";
import type { AppError, FilterDefinition, FilterParamValues, LoadedImage, OutputFormat } from "../types";
import { pickPreviewMaxEdge } from "../utils/dimensions";
import { createWorkingCanvas } from "../utils/workingCanvas";
import { buildFaceGeometry } from "../vision/faceGeometry";
import { detectFace, disposeFaceLandmarker } from "../vision/faceLandmarks";
import { buildFaceMaskTexture } from "../vision/faceMask";

type WorkingImage = { canvas: HTMLCanvasElement; width: number; height: number };

/**
 * Interocular distance as a fraction of the image long edge, assumed when no face
 * was detected. Roughly a typical head-and-shoulders portrait, so face-relative
 * radii stay sensible on the fallback path instead of collapsing to zero.
 */
const DEFAULT_FACE_SCALE = 0.14;

export function useFilterRenderer(canvasRef: RefObject<HTMLCanvasElement>) {
  const rendererRef = useRef<WebGLRenderer | null>(null);
  const fullBitmapRef = useRef<ImageBitmap | null>(null);
  const workingRef = useRef<WorkingImage | null>(null);
  const rafRef = useRef<number | null>(null);
  const pendingRenderRef = useRef<(() => void) | null>(null);

  const [rendererError, setRendererError] = useState<AppError | null>(null);
  const [contextLost, setContextLost] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [faceScale, setFaceScale] = useState(DEFAULT_FACE_SCALE);
  const [faceMaskActive, setFaceMaskActive] = useState(false);

  const ensureRenderer = useCallback((): WebGLRenderer | null => {
    if (rendererRef.current) return rendererRef.current;
    if (!canvasRef.current) return null;
    try {
      const renderer = new WebGLRenderer(canvasRef.current);
      renderer.setContextEventHandlers(
        () => setContextLost(true),
        () => setContextLost(false),
      );
      rendererRef.current = renderer;
      return renderer;
    } catch (err) {
      setRendererError(
        err instanceof WebGLUnsupportedError
          ? { kind: "webgl-unsupported", message: "WebGL2に対応していません。最新のChromeまたはEdgeでお試しください。" }
          : { kind: "unknown", message: "描画エンジンの初期化に失敗しました。" },
      );
      return null;
    }
  }, [canvasRef]);

  useEffect(() => {
    ensureRenderer();
    return () => {
      rendererRef.current?.dispose();
      rendererRef.current = null;
      fullBitmapRef.current?.close();
      fullBitmapRef.current = null;
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      void disposeFaceLandmarker();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setImage = useCallback(
    (loaded: LoadedImage) => {
      const renderer = ensureRenderer();
      if (!renderer) return;
      if (fullBitmapRef.current && fullBitmapRef.current !== loaded.bitmap) {
        fullBitmapRef.current.close();
      }
      fullBitmapRef.current = loaded.bitmap;
      const working = createWorkingCanvas(loaded.bitmap, loaded.width, loaded.height, pickPreviewMaxEdge());
      workingRef.current = working;
      // Drop the previous subject's mask and baked geometry synchronously, BEFORE
      // anything can render the new image. Detection is async, so without this the
      // old data would be applied to the new photo for as long as detection takes —
      // slimming a jaw and painting lipstick using another person's landmarks.
      renderer.setFaceMask(null);
      renderer.setFaceGeometry(null);
      setFaceScale(DEFAULT_FACE_SCALE);
      setFaceMaskActive(false);
      renderer.setImage(working.canvas, working.width, working.height);
    },
    [ensureRenderer],
  );

  /**
   * Runs face detection on the current working image and bakes everything derived
   * from it: the region mask every filter samples, and the mesh deformation and
   * makeup masks the makeup filter samples.
   *
   * Resolves to true only when something was actually applied, so callers know
   * whether a re-render is needed. Never rejects — detection is optional, and a
   * photo with no face in it is a valid input.
   */
  const runFaceDetection = useCallback(async (): Promise<boolean> => {
    const renderer = rendererRef.current;
    const working = workingRef.current;
    if (!renderer || !working) return false;
    const detection = await detectFace(working.canvas);
    // The renderer may have moved on to a different image while we were away.
    if (!detection || workingRef.current !== working) return false;

    const mask = buildFaceMaskTexture(
      detection.landmarks,
      detection.maskRings,
      detection.imageWidth,
      detection.imageHeight,
    );
    if (!mask) return false;
    renderer.setFaceMask(mask.texture, mask.faceScale);
    setFaceScale(mask.faceScale);
    setFaceMaskActive(true);

    // The geometry bake is a further best-effort step on top: it can fail on its
    // own (degenerate rings, a tessellation the model did not publish) without
    // costing the mask, so filters that only need the mask are unaffected.
    renderer.setFaceGeometry(buildFaceGeometry(working.canvas, detection.landmarks, detection.topology));
    return true;
  }, []);

  const renderNow = useCallback(
    (filter: FilterDefinition, params: FilterParamValues, strengthPercent: number, debugPassId?: string) => {
      const renderer = rendererRef.current;
      const working = workingRef.current;
      if (!renderer || !working || renderer.isContextLost()) return;
      try {
        // Named marks so render cost is inspectable in the Chrome DevTools Performance panel.
        performance.mark("filter-render-start");
        renderer.renderPreview(filter, params, strengthPercent, working.width, working.height, debugPassId);
        performance.mark("filter-render-end");
        performance.measure("filter-render", "filter-render-start", "filter-render-end");
        setRendererError(null);
      } catch (err) {
        setRendererError({
          kind: "shader-compile-failed",
          message: "フィルターの描画に失敗しました。",
          detail: err instanceof Error ? err.message : String(err),
        });
      }
    },
    [],
  );

  /** Batches rapid calls (e.g. slider drags) into at most one render per animation frame. */
  const scheduleRender = useCallback(
    (filter: FilterDefinition, params: FilterParamValues, strengthPercent: number) => {
      pendingRenderRef.current = () => renderNow(filter, params, strengthPercent);
      if (rafRef.current == null) {
        setIsProcessing(true);
        rafRef.current = requestAnimationFrame(() => {
          rafRef.current = null;
          pendingRenderRef.current?.();
          setIsProcessing(false);
        });
      }
    },
    [renderNow],
  );

  const generateThumbnail = useCallback(
    (filter: FilterDefinition, size = 140, paramsOverride?: FilterParamValues, strengthOverride = 100): string | null => {
      const renderer = rendererRef.current;
      const working = workingRef.current;
      if (!renderer || !working || renderer.isContextLost()) return null;
      const aspect = working.height / working.width;
      const w = size;
      const h = Math.max(1, Math.round(size * aspect));
      try {
        const params = paramsOverride ?? defaultParamValues(filter);
        const pixels = renderer.renderToPixels(filter, params, strengthOverride, w, h);
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) return null;
        ctx.putImageData(new ImageData(pixels.data, w, h), 0, 0);
        return canvas.toDataURL("image/png");
      } catch {
        return null;
      }
    },
    [],
  );

  const exportCurrent = useCallback(
    async (
      filter: FilterDefinition,
      params: FilterParamValues,
      strengthPercent: number,
      targetWidth: number,
      targetHeight: number,
      format: OutputFormat,
      quality: number,
    ): Promise<Blob> => {
      const renderer = ensureRenderer();
      const fullBitmap = fullBitmapRef.current;
      if (!renderer || !fullBitmap) {
        throw new Error("画像が読み込まれていません。");
      }
      try {
        return await exportImage({
          renderer,
          filter,
          params,
          strengthPercent,
          sourceBitmap: fullBitmap,
          targetWidth,
          targetHeight,
          format,
          quality,
        });
      } finally {
        const working = workingRef.current;
        if (working) renderer.setImage(working.canvas, working.width, working.height);
      }
    },
    [ensureRenderer],
  );

  const drawOriginalTo = useCallback((target: HTMLCanvasElement) => {
    const working = workingRef.current;
    if (!working) return;
    target.width = working.width;
    target.height = working.height;
    const ctx = target.getContext("2d");
    ctx?.drawImage(working.canvas, 0, 0);
  }, []);

  const maxTextureSize = rendererRef.current?.maxTextureSize ?? 4096;

  return {
    setImage,
    runFaceDetection,
    scheduleRender,
    renderNow,
    generateThumbnail,
    exportCurrent,
    drawOriginalTo,
    rendererError,
    contextLost,
    isProcessing,
    faceScale,
    faceMaskActive,
    maxTextureSize,
    getWorkingSize: () => workingRef.current,
  };
}
