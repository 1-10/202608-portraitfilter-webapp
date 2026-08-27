import { fitWithinLongEdge } from "./dimensions";

/**
 * Draws `source` scaled down (never up) so its long edge is at most `maxEdge`,
 * returning a persistent canvas element suitable for repeated GPU texture
 * uploads (e.g. re-uploading after WebGL context restore).
 */
export function createWorkingCanvas(
  source: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  maxEdge: number,
): { canvas: HTMLCanvasElement; width: number; height: number } {
  const { width, height } = fitWithinLongEdge({ width: sourceWidth, height: sourceHeight }, maxEdge);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("キャンバスの作成に失敗しました。");
  ctx.drawImage(source, 0, 0, width, height);
  return { canvas, width, height };
}
