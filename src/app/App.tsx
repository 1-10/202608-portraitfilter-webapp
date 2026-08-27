import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CameraCapture from "../components/CameraCapture";
import ErrorBanner from "../components/ErrorBanner";
import ExportDialog from "../components/ExportDialog";
import FilterControls from "../components/FilterControls";
import FilterPicker from "../components/FilterPicker";
import Header from "../components/Header";
import ImageDropzone from "../components/ImageDropzone";
import ImagePreview from "../components/ImagePreview";
import FilterDebugPanel from "../components/FilterDebugPanel";
import { DEFAULT_FILTER_ID, FILTERS, defaultParamValues, getFilterById } from "../filters/definitions";
import type { CapturedFrame } from "../hooks/useCamera";
import { useFilterRenderer } from "../hooks/useFilterRenderer";
import { useImageInput } from "../hooks/useImageInput";
import { downloadBlob } from "../rendering/exportImage";
import type { AppError, AppState, FilterParamValues, LoadedImage, OutputFormat } from "../types";
import { buildExportFilename } from "../utils/filename";
import styles from "./App.module.css";

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const prevContextLostRef = useRef(false);
  /** Guards against a slow detection for image A landing after image B was loaded. */
  const faceDetectTokenRef = useRef(0);

  const { loadFile, error: imageInputError, clearError } = useImageInput();
  const filterRenderer = useFilterRenderer(canvasRef);

  const [status, setStatus] = useState<AppState>("idle");
  const [image, setImage] = useState<LoadedImage | null>(null);
  const [selectedFilterId, setSelectedFilterId] = useState<string>(DEFAULT_FILTER_ID);
  const [strength, setStrength] = useState(100);
  const [paramValues, setParamValues] = useState<FilterParamValues>({});
  const [thumbnails, setThumbnails] = useState<Record<string, string | null>>({});
  const [compareHeld, setCompareHeld] = useState(false);
  const [showExportDialog, setShowExportDialog] = useState(false);
  const [updateAvailable, setUpdateAvailable] = useState(false);

  useEffect(() => {
    const handler = () => setUpdateAvailable(true);
    window.addEventListener("app-update-available", handler);
    return () => window.removeEventListener("app-update-available", handler);
  }, []);

  const selectedFilter = useMemo(() => getFilterById(selectedFilterId), [selectedFilterId]);

  const regenerateThumbnails = useCallback(() => {
    const next: Record<string, string | null> = {};
    for (const filter of FILTERS) {
      next[filter.id] = filterRenderer.generateThumbnail(filter);
    }
    setThumbnails(next);
  }, [filterRenderer]);

  const applyNewImage = useCallback(
    (loaded: LoadedImage) => {
      // setImage clears any previous face mask synchronously, so nothing below
      // can render this image using the last subject's regions.
      filterRenderer.setImage(loaded);
      setImage(loaded);
      setSelectedFilterId(DEFAULT_FILTER_ID);
      setStrength(100);
      setParamValues({});
      filterRenderer.renderNow(getFilterById(DEFAULT_FILTER_ID), {}, 100);
      if (overlayCanvasRef.current) filterRenderer.drawOriginalTo(overlayCanvasRef.current);
      setStatus("ready");
      regenerateThumbnails();

      // Face detection is a progressive enhancement: the preview above is already
      // on screen, and this refines it a few hundred ms later rather than making
      // the user wait on a model load. Failure is silent by design.
      const token = ++faceDetectTokenRef.current;
      void filterRenderer.runFaceDetection().then((applied) => {
        if (!applied || token !== faceDetectTokenRef.current) return;
        filterRenderer.renderNow(getFilterById(DEFAULT_FILTER_ID), {}, 100);
        // Thumbnails bake a PNG data URL, so a maskless one would stay wrong for
        // the lifetime of this image — they have to be redrawn, not just the preview.
        regenerateThumbnails();
      });
    },
    [filterRenderer, regenerateThumbnails],
  );

  const handleFileSelected = useCallback(
    async (file: File) => {
      setStatus("loading-image");
      const loaded = await loadFile(file);
      if (!loaded) {
        setStatus("error");
        return;
      }
      applyNewImage(loaded);
    },
    [loadFile, applyNewImage],
  );

  const handleCameraRequested = useCallback(() => {
    clearError();
    setStatus("camera-preview");
  }, [clearError]);

  const handleCameraCapture = useCallback(
    (frame: CapturedFrame) => {
      applyNewImage({ bitmap: frame.bitmap, width: frame.width, height: frame.height, sourceFileName: null });
    },
    [applyNewImage],
  );

  const handleCameraCancel = useCallback(() => {
    setStatus("idle");
  }, []);

  const handleChangeImage = useCallback(() => {
    setImage(null);
    setCompareHeld(false);
    setShowExportDialog(false);
    setStatus("idle");
  }, []);

  const handleSelectFilter = useCallback(
    (filterId: string) => {
      const filter = getFilterById(filterId);
      const defaults = defaultParamValues(filter);
      setSelectedFilterId(filterId);
      setParamValues(defaults);
      filterRenderer.scheduleRender(filter, defaults, strength);
    },
    [strength, filterRenderer],
  );

  const handleStrengthChange = useCallback(
    (value: number) => {
      setStrength(value);
      filterRenderer.scheduleRender(selectedFilter, paramValues, value);
    },
    [selectedFilter, paramValues, filterRenderer],
  );

  const handleParamChange = useCallback(
    (paramId: string, value: number) => {
      setParamValues((prev) => {
        const next = { ...prev, [paramId]: value };
        filterRenderer.scheduleRender(selectedFilter, next, strength);
        return next;
      });
    },
    [selectedFilter, strength, filterRenderer],
  );

  const handleApplyPreset = useCallback(
    (presetValues: FilterParamValues) => {
      const next = { ...defaultParamValues(selectedFilter), ...presetValues };
      setParamValues(next);
      filterRenderer.scheduleRender(selectedFilter, next, strength);
    },
    [selectedFilter, strength, filterRenderer],
  );

  const handleReset = useCallback(() => {
    setSelectedFilterId(DEFAULT_FILTER_ID);
    setStrength(100);
    setParamValues({});
    filterRenderer.renderNow(getFilterById(DEFAULT_FILTER_ID), {}, 100);
  }, [filterRenderer]);

  const handleExport = useCallback(
    async (options: { format: OutputFormat; quality: number; width: number; height: number }) => {
      const blob = await filterRenderer.exportCurrent(
        selectedFilter,
        paramValues,
        strength,
        options.width,
        options.height,
        options.format,
        options.quality,
      );
      const filename = buildExportFilename(selectedFilterId, options.format);
      downloadBlob(blob, filename);
    },
    [filterRenderer, selectedFilter, selectedFilterId, paramValues, strength],
  );

  // Re-render the current filter once the WebGL context comes back after being lost.
  useEffect(() => {
    if (prevContextLostRef.current && !filterRenderer.contextLost && status === "ready") {
      filterRenderer.renderNow(selectedFilter, paramValues, strength);
    }
    prevContextLostRef.current = filterRenderer.contextLost;
  }, [filterRenderer, filterRenderer.contextLost, status, selectedFilter, paramValues, strength]);

  useEffect(() => {
    if (filterRenderer.rendererError?.kind === "webgl-unsupported" && status !== "ready") {
      setStatus("error");
    }
  }, [filterRenderer.rendererError, status]);

  const displayedError: AppError | null =
    status === "error" ? (imageInputError ?? filterRenderer.rendererError ?? { kind: "unknown", message: "予期しないエラーが発生しました。" }) : null;

  return (
    <div className={styles.app}>
      <Header hasImage={status === "ready" && image !== null} onChangeImage={handleChangeImage} />

      {updateAvailable && (
        <div className={styles.updateBanner} role="status">
          新しいバージョンが利用可能です。
          <button type="button" onClick={() => window.location.reload()}>
            再読み込み
          </button>
        </div>
      )}

      <main className={styles.main}>
        {/* The rendering canvas and its original-image overlay are always mounted so
            useFilterRenderer can bind to them as soon as the app starts. */}
        <div className={status === "ready" ? styles.editLayout : styles.hiddenLayout}>
          <ImagePreview
            canvasRef={canvasRef}
            overlayCanvasRef={overlayCanvasRef}
            compareHeld={compareHeld}
            isProcessing={filterRenderer.isProcessing}
            onCompareStart={() => setCompareHeld(true)}
            onCompareEnd={() => setCompareHeld(false)}
          />

          <aside className={styles.panel}>
            {filterRenderer.contextLost && (
              <ErrorBanner
                error={{ kind: "context-lost", message: "GPU描画がリセットされました。復帰を待っています…" }}
              />
            )}
            {/* A render failure while editing (e.g. a shader that fails to compile)
                used to be recorded and never shown, because the error surface below
                only renders when status === "error" — which a render failure never
                sets. That made a broken filter look like one that simply does nothing. */}
            {!filterRenderer.contextLost && filterRenderer.rendererError && (
              <ErrorBanner error={filterRenderer.rendererError} />
            )}
            <FilterPicker
              filters={FILTERS}
              selectedId={selectedFilterId}
              thumbnails={thumbnails}
              onSelect={handleSelectFilter}
            />
            {/* Everything below can grow past the panel's height (some filters have
                many sliders), so only THIS region scrolls — the picker above and the
                download button below stay put regardless of how tall it gets. */}
            <div className={styles.controlsScroll}>
              <FilterControls
                filter={selectedFilter}
                strength={strength}
                paramValues={paramValues}
                onStrengthChange={handleStrengthChange}
                onParamChange={handleParamChange}
                onApplyPreset={handleApplyPreset}
                onReset={handleReset}
              />
              {import.meta.env.DEV && selectedFilter.passes.length > 1 && (
                <FilterDebugPanel
                  filter={selectedFilter}
                  paramValues={paramValues}
                  strength={strength}
                  renderNow={filterRenderer.renderNow}
                  generateThumbnail={filterRenderer.generateThumbnail}
                  originalFilter={getFilterById(DEFAULT_FILTER_ID)}
                  faceMaskActive={filterRenderer.faceMaskActive}
                  faceScale={filterRenderer.faceScale}
                />
              )}
            </div>
            <button type="button" className={styles.downloadButton} onClick={() => setShowExportDialog(true)}>
              ダウンロード
            </button>
          </aside>
        </div>

        {status === "idle" && (
          <div className={styles.centerStage}>
            <ImageDropzone onFileSelected={handleFileSelected} onCameraRequested={handleCameraRequested} busy={false} />
            {imageInputError && <ErrorBanner error={imageInputError} onDismiss={clearError} />}
          </div>
        )}

        {status === "loading-image" && (
          <div className={styles.centerStage}>
            <p className={styles.loadingText}>画像を読み込んでいます…</p>
          </div>
        )}

        {status === "camera-preview" && (
          <div className={styles.centerStage}>
            <CameraCapture onCapture={handleCameraCapture} onCancel={handleCameraCancel} />
          </div>
        )}

        {status === "error" && displayedError && (
          <div className={styles.centerStage}>
            <ErrorBanner error={displayedError} onRetry={() => setStatus("idle")} />
          </div>
        )}
      </main>

      {image && (
        <ExportDialog
          open={showExportDialog}
          filterId={selectedFilterId}
          sourceSize={{ width: image.width, height: image.height }}
          maxTextureSize={filterRenderer.maxTextureSize}
          onClose={() => setShowExportDialog(false)}
          onExport={handleExport}
        />
      )}
    </div>
  );
}
