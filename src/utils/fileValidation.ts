export const ACCEPTED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"] as const;
export type AcceptedMimeType = (typeof ACCEPTED_MIME_TYPES)[number];

/** Files larger than this are rejected before attempting to decode, to avoid OOM. */
export const MAX_FILE_SIZE_BYTES = 80 * 1024 * 1024; // 80MB

/** Decoded bitmaps with a long edge above this are rejected as too large to process safely. */
export const MAX_DECODED_LONG_EDGE = 12000;

export type FileValidationResult =
  | { ok: true }
  | { ok: false; reason: "unsupported-format" | "too-large" };

export function isAcceptedMimeType(type: string): type is AcceptedMimeType {
  return (ACCEPTED_MIME_TYPES as readonly string[]).includes(type);
}

export function validateImageFile(file: File): FileValidationResult {
  if (!isAcceptedMimeType(file.type)) {
    return { ok: false, reason: "unsupported-format" };
  }
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return { ok: false, reason: "too-large" };
  }
  return { ok: true };
}

export function validateDecodedDimensions(width: number, height: number): FileValidationResult {
  if (Math.max(width, height) > MAX_DECODED_LONG_EDGE) {
    return { ok: false, reason: "too-large" };
  }
  return { ok: true };
}
