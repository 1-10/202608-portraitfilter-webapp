import { useEffect, useRef, useState } from "react";
import { DEFAULT_EXPORT_MAX_EDGE, resolveExportSize, type ExportSizePreset, type Size } from "../utils/dimensions";
import { buildExportFilename } from "../utils/filename";
import type { OutputFormat } from "../types";
import styles from "./ExportDialog.module.css";

type Props = {
  open: boolean;
  filterId: string;
  sourceSize: Size;
  maxTextureSize: number;
  onClose: () => void;
  onExport: (options: { format: OutputFormat; quality: number; width: number; height: number }) => Promise<void>;
};

const FORMAT_OPTIONS: { value: OutputFormat; label: string }[] = [
  { value: "image/jpeg", label: "JPEG" },
  { value: "image/png", label: "PNG" },
  { value: "image/webp", label: "WebP" },
];

const SIZE_OPTIONS: { value: ExportSizePreset; label: string }[] = [
  { value: "original", label: "元のサイズ" },
  { value: "1920", label: "長辺1920px" },
  { value: "1280", label: "長辺1280px" },
];

export default function ExportDialog({ open, filterId, sourceSize, maxTextureSize, onClose, onExport }: Props) {
  const [format, setFormat] = useState<OutputFormat>("image/jpeg");
  const [quality, setQuality] = useState(92);
  const [sizePreset, setSizePreset] = useState<ExportSizePreset>("original");
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const previouslyFocused = document.activeElement as HTMLElement | null;
    const dialog = dialogRef.current;
    const focusable = dialog?.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
    );
    focusable?.[0]?.focus();

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key !== "Tab" || !focusable || focusable.length === 0) return;
      const first = focusable[0]!;
      const last = focusable[focusable.length - 1]!;
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      previouslyFocused?.focus();
    };
  }, [open, onClose]);

  if (!open) return null;

  const exportMaxEdge = Math.min(DEFAULT_EXPORT_MAX_EDGE, maxTextureSize);
  const resolved = resolveExportSize(sourceSize, sizePreset, exportMaxEdge, maxTextureSize);
  const previewFilename = buildExportFilename(filterId, format);
  const showQuality = format !== "image/png";

  const handleExport = async () => {
    setIsExporting(true);
    setExportError(null);
    try {
      await onExport({ format, quality: quality / 100, width: resolved.width, height: resolved.height });
    } catch (err) {
      setExportError(err instanceof Error ? err.message : "画像の書き出しに失敗しました。");
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className={styles.backdrop} onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="export-dialog-title"
        ref={dialogRef}
      >
        <h2 id="export-dialog-title" className={styles.title}>
          ダウンロード
        </h2>

        <fieldset className={styles.fieldset}>
          <legend className={styles.legend}>形式</legend>
          <div className={styles.optionRow}>
            {FORMAT_OPTIONS.map((opt) => (
              <label key={opt.value} className={styles.radioLabel}>
                <input
                  type="radio"
                  name="export-format"
                  value={opt.value}
                  checked={format === opt.value}
                  onChange={() => setFormat(opt.value)}
                />
                {opt.label}
              </label>
            ))}
          </div>
        </fieldset>

        {showQuality && (
          <div className={styles.row}>
            <div className={styles.rowHeader}>
              <label htmlFor="export-quality" className={styles.label}>
                画質
              </label>
              <span className={styles.value}>{quality}</span>
            </div>
            <input
              id="export-quality"
              type="range"
              min={70}
              max={100}
              step={1}
              value={quality}
              onChange={(e) => setQuality(Number(e.target.value))}
              className={styles.slider}
            />
          </div>
        )}

        <fieldset className={styles.fieldset}>
          <legend className={styles.legend}>サイズ</legend>
          <div className={styles.optionRow}>
            {SIZE_OPTIONS.map((opt) => (
              <label key={opt.value} className={styles.radioLabel}>
                <input
                  type="radio"
                  name="export-size"
                  value={opt.value}
                  checked={sizePreset === opt.value}
                  onChange={() => setSizePreset(opt.value)}
                />
                {opt.label}
              </label>
            ))}
          </div>
          <p className={styles.hint}>
            出力サイズ：{resolved.width} × {resolved.height} px
          </p>
        </fieldset>

        <p className={styles.filenamePreview}>ファイル名：{previewFilename}</p>

        {exportError && (
          <p className={styles.error} role="alert">
            {exportError}
          </p>
        )}

        <div className={styles.actions}>
          <button type="button" className={styles.cancelButton} onClick={onClose} disabled={isExporting}>
            閉じる
          </button>
          <button type="button" className={styles.downloadButton} onClick={handleExport} disabled={isExporting}>
            {isExporting ? "書き出し中…" : "ダウンロード"}
          </button>
        </div>
      </div>
    </div>
  );
}
