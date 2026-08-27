import { describe, expect, it } from "vitest";
import { getOrientationTransform, readJpegExifOrientation } from "../../src/utils/exifOrientation";

/** Builds a minimal JPEG byte sequence with an APP1/EXIF segment declaring the given orientation. */
function buildJpegWithOrientation(orientation: number, littleEndian = true): ArrayBuffer {
  const bytes: number[] = [0xff, 0xd8]; // SOI

  // TIFF header (8 bytes) + one IFD entry (12 bytes) + next-IFD-offset (4 bytes) = 24 bytes
  const tiff: number[] = [];
  const pushU16 = (arr: number[], v: number) => {
    if (littleEndian) {
      arr.push(v & 0xff, (v >> 8) & 0xff);
    } else {
      arr.push((v >> 8) & 0xff, v & 0xff);
    }
  };
  const pushU32 = (arr: number[], v: number) => {
    if (littleEndian) {
      arr.push(v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff);
    } else {
      arr.push((v >> 24) & 0xff, (v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff);
    }
  };

  if (littleEndian) {
    tiff.push(0x49, 0x49, 0x2a, 0x00); // "II" + 42
  } else {
    tiff.push(0x4d, 0x4d, 0x00, 0x2a); // "MM" + 42
  }
  pushU32(tiff, 8); // offset to first IFD

  pushU16(tiff, 1); // 1 entry
  pushU16(tiff, 0x0112); // tag: Orientation
  pushU16(tiff, 3); // type: SHORT
  pushU32(tiff, 1); // count
  // TIFF values shorter than 4 bytes are always left-justified within the value field.
  if (littleEndian) {
    tiff.push(orientation & 0xff, (orientation >> 8) & 0xff, 0, 0);
  } else {
    tiff.push((orientation >> 8) & 0xff, orientation & 0xff, 0, 0);
  }
  pushU32(tiff, 0); // next IFD offset

  const exifHeader = [0x45, 0x78, 0x69, 0x66, 0x00, 0x00]; // "Exif\0\0"
  const app1Payload = [...exifHeader, ...tiff];
  const app1Length = app1Payload.length + 2; // includes the length field itself

  bytes.push(0xff, 0xe1);
  bytes.push((app1Length >> 8) & 0xff, app1Length & 0xff);
  bytes.push(...app1Payload);

  bytes.push(0xff, 0xd9); // EOI
  return new Uint8Array(bytes).buffer;
}

describe("readJpegExifOrientation", () => {
  it("returns 1 (normal) for non-JPEG data", () => {
    const buffer = new Uint8Array([0x89, 0x50, 0x4e, 0x47]).buffer; // PNG signature
    expect(readJpegExifOrientation(buffer)).toBe(1);
  });

  it("returns 1 when a JPEG has no EXIF segment", () => {
    const buffer = new Uint8Array([0xff, 0xd8, 0xff, 0xd9]).buffer;
    expect(readJpegExifOrientation(buffer)).toBe(1);
  });

  it("reads orientation 6 (rotated 90 CW) from little-endian EXIF", () => {
    expect(readJpegExifOrientation(buildJpegWithOrientation(6, true))).toBe(6);
  });

  it("reads orientation 3 (rotated 180) from big-endian EXIF", () => {
    expect(readJpegExifOrientation(buildJpegWithOrientation(3, false))).toBe(3);
  });

  it("reads orientation 8 correctly", () => {
    expect(readJpegExifOrientation(buildJpegWithOrientation(8, true))).toBe(8);
  });
});

describe("getOrientationTransform", () => {
  it("keeps dimensions unchanged for normal orientation", () => {
    const t = getOrientationTransform(1, 1000, 600);
    expect(t.canvasWidth).toBe(1000);
    expect(t.canvasHeight).toBe(600);
  });

  it("swaps width/height for 90-degree rotations (6 and 8)", () => {
    expect(getOrientationTransform(6, 1000, 600)).toMatchObject({ canvasWidth: 600, canvasHeight: 1000 });
    expect(getOrientationTransform(8, 1000, 600)).toMatchObject({ canvasWidth: 600, canvasHeight: 1000 });
  });

  it("keeps dimensions unchanged for a 180-degree rotation", () => {
    const t = getOrientationTransform(3, 1000, 600);
    expect(t.canvasWidth).toBe(1000);
    expect(t.canvasHeight).toBe(600);
  });
});
