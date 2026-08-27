/**
 * Minimal EXIF Orientation (tag 0x0112) reader for JPEG files.
 * Implemented by hand to avoid pulling in an external EXIF library.
 * Returns 1 (normal) for non-JPEG data, missing EXIF, or any parse failure.
 */
export type ExifOrientation = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;

const JPEG_SOI = 0xffd8;
const APP1_MARKER = 0xffe1;
const EXIF_TAG_ORIENTATION = 0x0112;

export function readJpegExifOrientation(buffer: ArrayBuffer): ExifOrientation {
  try {
    return readJpegExifOrientationUnsafe(buffer);
  } catch {
    return 1;
  }
}

function readJpegExifOrientationUnsafe(buffer: ArrayBuffer): ExifOrientation {
  const view = new DataView(buffer);
  if (view.byteLength < 4 || view.getUint16(0) !== JPEG_SOI) {
    return 1;
  }

  let offset = 2;
  while (offset + 4 <= view.byteLength) {
    const marker = view.getUint16(offset);
    if ((marker & 0xff00) !== 0xff00) {
      break;
    }
    if (marker === 0xffd9 || marker === 0xffda) {
      break; // EOI or start of scan: no more metadata markers follow
    }
    const segmentLength = view.getUint16(offset + 2);
    if (marker === APP1_MARKER) {
      const orientation = readOrientationFromApp1(view, offset + 4);
      if (orientation !== null) return orientation;
    }
    offset += 2 + segmentLength;
  }
  return 1;
}

function readOrientationFromApp1(view: DataView, exifStart: number): ExifOrientation | null {
  if (exifStart + 6 > view.byteLength) return null;
  const isExif =
    view.getUint8(exifStart) === 0x45 && // E
    view.getUint8(exifStart + 1) === 0x78 && // x
    view.getUint8(exifStart + 2) === 0x69 && // i
    view.getUint8(exifStart + 3) === 0x66; // f
  if (!isExif) return null;

  const tiffStart = exifStart + 6;
  if (tiffStart + 8 > view.byteLength) return null;
  const byteOrderMark = view.getUint16(tiffStart);
  const little = byteOrderMark === 0x4949;
  if (!little && byteOrderMark !== 0x4d4d) return null;

  const firstIfdOffset = view.getUint32(tiffStart + 4, little);
  const dirStart = tiffStart + firstIfdOffset;
  if (dirStart + 2 > view.byteLength) return null;

  const entryCount = view.getUint16(dirStart, little);
  for (let i = 0; i < entryCount; i++) {
    const entryOffset = dirStart + 2 + i * 12;
    if (entryOffset + 12 > view.byteLength) break;
    const tag = view.getUint16(entryOffset, little);
    if (tag === EXIF_TAG_ORIENTATION) {
      const value = view.getUint16(entryOffset + 8, little);
      if (value >= 1 && value <= 8) return value as ExifOrientation;
      return 1;
    }
  }
  return 1;
}

export type OrientationTransform = {
  canvasWidth: number;
  canvasHeight: number;
  /** Args for CanvasRenderingContext2D#transform(a,b,c,d,e,f) to bake in the orientation. */
  matrix: [number, number, number, number, number, number];
};

/**
 * Given the raw (un-rotated) bitmap size and its EXIF orientation, returns the
 * corrected canvas size and the transform to apply before drawing the bitmap
 * at (0,0) so the result appears upright.
 */
export function getOrientationTransform(
  orientation: ExifOrientation,
  sourceWidth: number,
  sourceHeight: number,
): OrientationTransform {
  const swapped = orientation >= 5 && orientation <= 8;
  const canvasWidth = swapped ? sourceHeight : sourceWidth;
  const canvasHeight = swapped ? sourceWidth : sourceHeight;

  let matrix: [number, number, number, number, number, number];
  switch (orientation) {
    case 2:
      matrix = [-1, 0, 0, 1, sourceWidth, 0];
      break;
    case 3:
      matrix = [-1, 0, 0, -1, sourceWidth, sourceHeight];
      break;
    case 4:
      matrix = [1, 0, 0, -1, 0, sourceHeight];
      break;
    case 5:
      matrix = [0, 1, 1, 0, 0, 0];
      break;
    case 6:
      matrix = [0, 1, -1, 0, sourceHeight, 0];
      break;
    case 7:
      matrix = [0, -1, -1, 0, sourceHeight, sourceWidth];
      break;
    case 8:
      matrix = [0, -1, 1, 0, 0, sourceWidth];
      break;
    default:
      matrix = [1, 0, 0, 1, 0, 0];
      break;
  }

  return { canvasWidth, canvasHeight, matrix };
}
