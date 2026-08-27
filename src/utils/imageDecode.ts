import type { AppError, LoadedImage } from "../types";
import { getOrientationTransform, readJpegExifOrientation } from "./exifOrientation";
import { MAX_DECODED_LONG_EDGE, validateDecodedDimensions, validateImageFile } from "./fileValidation";

export class ImageDecodeError extends Error {
  appError: AppError;
  constructor(appError: AppError) {
    super(appError.message);
    this.appError = appError;
  }
}

/**
 * Decodes an image File into an oriented ImageBitmap.
 * - Uses createImageBitmap for GPU-friendly, off-main-thread-capable decode.
 * - Reads EXIF orientation by hand and bakes it into the bitmap via a 2D canvas,
 *   since we explicitly request `imageOrientation: "none"` for deterministic
 *   behavior across browsers.
 */
export async function decodeImageFile(file: File): Promise<LoadedImage> {
  const validation = validateImageFile(file);
  if (!validation.ok) {
    throw new ImageDecodeError(
      validation.reason === "unsupported-format"
        ? {
            kind: "unsupported-format",
            message: "対応していない画像形式です。JPEG、PNG、WebPをご利用ください。",
          }
        : {
            kind: "too-large",
            message: "ファイルサイズが大きすぎます。80MB以下の画像をご利用ください。",
          },
    );
  }

  let orientation = 1 as ReturnType<typeof readJpegExifOrientation>;
  if (file.type === "image/jpeg") {
    try {
      const buffer = await file.arrayBuffer();
      orientation = readJpegExifOrientation(buffer);
    } catch {
      orientation = 1;
    }
  }

  let rawBitmap: ImageBitmap;
  try {
    rawBitmap = await createImageBitmap(file, { imageOrientation: "none" });
  } catch (cause) {
    throw new ImageDecodeError({
      kind: "decode-failed",
      message: "画像を読み込めませんでした。ファイルが破損している可能性があります。",
      detail: cause instanceof Error ? cause.message : String(cause),
    });
  }

  const dimCheck = validateDecodedDimensions(rawBitmap.width, rawBitmap.height);
  if (!dimCheck.ok) {
    rawBitmap.close();
    throw new ImageDecodeError({
      kind: "too-large",
      message: `画像の解像度が大きすぎます（長辺${MAX_DECODED_LONG_EDGE}px以下にしてください）。`,
    });
  }

  if (orientation === 1) {
    return {
      bitmap: rawBitmap,
      width: rawBitmap.width,
      height: rawBitmap.height,
      sourceFileName: file.name,
    };
  }

  const { canvasWidth, canvasHeight, matrix } = getOrientationTransform(
    orientation,
    rawBitmap.width,
    rawBitmap.height,
  );

  const canvas = document.createElement("canvas");
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    rawBitmap.close();
    throw new ImageDecodeError({
      kind: "decode-failed",
      message: "画像の向き補正に失敗しました。",
    });
  }
  ctx.transform(...matrix);
  ctx.drawImage(rawBitmap, 0, 0);
  rawBitmap.close();

  const oriented = await createImageBitmap(canvas);
  return {
    bitmap: oriented,
    width: oriented.width,
    height: oriented.height,
    sourceFileName: file.name,
  };
}
