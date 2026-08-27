import type { FilterDefinition, FilterParamValues, OutputFormat } from "../types";
import type { WebGLRenderer } from "./WebGLRenderer";

export type ExportImageOptions = {
  renderer: WebGLRenderer;
  filter: FilterDefinition;
  params: FilterParamValues;
  strengthPercent: number;
  sourceBitmap: ImageBitmap;
  targetWidth: number;
  targetHeight: number;
  format: OutputFormat;
  /** 0..1, only meaningful for image/jpeg and image/webp. */
  quality: number;
};

export class ExportError extends Error {}

/**
 * Re-uploads the source image at the requested export resolution, runs the
 * filter pipeline at full quality, and encodes the result to a Blob.
 * Mutates the renderer's currently-bound original texture as a side effect;
 * callers that also drive a live preview must re-upload the preview-resolution
 * image afterward.
 */
export async function exportImage(opts: ExportImageOptions): Promise<Blob> {
  const { renderer, filter, params, strengthPercent, sourceBitmap, targetWidth, targetHeight, format, quality } =
    opts;

  const scaledCanvas = document.createElement("canvas");
  scaledCanvas.width = targetWidth;
  scaledCanvas.height = targetHeight;
  const scaledCtx = scaledCanvas.getContext("2d");
  if (!scaledCtx) throw new ExportError("書き出し用キャンバスを作成できませんでした。");
  scaledCtx.drawImage(sourceBitmap, 0, 0, targetWidth, targetHeight);

  renderer.setImage(scaledCanvas, targetWidth, targetHeight);
  const pixels = renderer.renderToPixels(filter, params, strengthPercent, targetWidth, targetHeight);

  const outCanvas = document.createElement("canvas");
  outCanvas.width = pixels.width;
  outCanvas.height = pixels.height;
  const outCtx = outCanvas.getContext("2d");
  if (!outCtx) throw new ExportError("書き出し画像の生成に失敗しました。");
  const imageData = new ImageData(pixels.data, pixels.width, pixels.height);
  outCtx.putImageData(imageData, 0, 0);

  const blob = await new Promise<Blob | null>((resolve) => {
    outCanvas.toBlob(resolve, format, quality);
  });
  if (!blob) {
    throw new ExportError("画像の書き出しに失敗しました。");
  }
  return blob;
}

/** Triggers a browser download of `blob` as `filename` via a temporary object URL, then revokes it. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
