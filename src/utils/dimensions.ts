export const DEFAULT_EXPORT_MAX_EDGE = 4096;
export const DEFAULT_PREVIEW_MAX_EDGE = 1280;
export const MIN_PREVIEW_MAX_EDGE = 768;

export type Size = { width: number; height: number };

/**
 * Scales `size` down so its longest edge does not exceed `maxEdge`.
 * Never upscales. Dimensions are rounded to the nearest integer, minimum 1px.
 */
export function fitWithinLongEdge(size: Size, maxEdge: number): Size {
  const longEdge = Math.max(size.width, size.height);
  if (longEdge <= maxEdge) {
    return { width: Math.round(size.width), height: Math.round(size.height) };
  }
  const scale = maxEdge / longEdge;
  return {
    width: Math.max(1, Math.round(size.width * scale)),
    height: Math.max(1, Math.round(size.height * scale)),
  };
}

/**
 * Picks a preview working resolution long edge based on the source image size
 * and a device-capability hint (defaults to the full range's upper bound).
 */
export function pickPreviewMaxEdge(devicePerformanceHint: "low" | "medium" | "high" = "high"): number {
  switch (devicePerformanceHint) {
    case "low":
      return MIN_PREVIEW_MAX_EDGE;
    case "medium":
      return 1024;
    case "high":
    default:
      return DEFAULT_PREVIEW_MAX_EDGE;
  }
}

export type ExportSizePreset = "original" | "1920" | "1280";

/**
 * Resolves the requested export preset into concrete pixel dimensions,
 * clamped by the source size, the configured export cap, and the GPU's
 * MAX_TEXTURE_SIZE.
 */
export function resolveExportSize(
  sourceSize: Size,
  preset: ExportSizePreset,
  exportMaxEdge: number = DEFAULT_EXPORT_MAX_EDGE,
  maxTextureSize: number = DEFAULT_EXPORT_MAX_EDGE,
): Size {
  const hardCap = Math.min(exportMaxEdge, maxTextureSize);
  const sourceClamped = fitWithinLongEdge(sourceSize, hardCap);

  if (preset === "original") {
    return sourceClamped;
  }

  const target = preset === "1920" ? 1920 : 1280;
  const effectiveTarget = Math.min(target, hardCap);
  return fitWithinLongEdge(sourceClamped, effectiveTarget);
}
