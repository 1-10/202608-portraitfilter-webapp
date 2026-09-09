import { describe, expect, it } from "vitest";
import {
  buildFaceFrame,
  buildSkirtRing,
  computeGeomRect,
  SKIRT_OUTWARD_FACE_FRACTION,
} from "../../src/vision/faceMeshTopology";
import { buildWarpFieldCanvas } from "../../src/vision/faceWarpField";
import { SYNTHETIC_FACE } from "./fixtures/syntheticFace";

const IMAGE_WIDTH = 1000;
const IMAGE_HEIGHT = 1000;

function build() {
  const built = buildFaceFrame(SYNTHETIC_FACE.landmarks, SYNTHETIC_FACE.rings, IMAGE_WIDTH, IMAGE_HEIGHT);
  if (!built) throw new Error("the synthetic face must produce a frame");
  const { frame, points } = built;
  const skirt = buildSkirtRing(points, SYNTHETIC_FACE.rings.faceOval, frame.faceWidth * SKIRT_OUTWARD_FACE_FRACTION);
  const rect = computeGeomRect(skirt, frame.faceWidth * 0.06);
  const result = buildWarpFieldCanvas(
    points,
    frame,
    SYNTHETIC_FACE.triangles,
    SYNTHETIC_FACE.rings.faceOval,
    rect,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
  );
  return { ...result, frame, points, rect };
}

/** Channel value at a field texel. Rows are packed bottom-up, as GL wants them. */
function at2(
  texture: { data: Uint8Array; width: number; height: number },
  x: number,
  yFromBottom: number,
  channel: 0 | 1 | 2 | 3,
): number {
  return texture.data[(yFromBottom * texture.width + x) * 4 + channel]!;
}
const at = at2;

describe("buildWarpFieldCanvas", () => {
  const field = build();

  it("encodes zero displacement as exactly 128", () => {
    // Anything else leaves a constant sub-pixel drift across the entire frame,
    // which is why the encoder scales by 127 and biases by 128 rather than 255/127.5.
    const { texture } = field;
    for (const [x, y] of [
      [0, 0],
      [texture.width - 1, 0],
      [0, texture.height - 1],
      [texture.width - 1, texture.height - 1],
    ]) {
      for (const channel of [0, 1, 2, 3] as const) {
        expect(at(texture, x!, y!, channel), `corner ${x},${y} channel ${channel}`).toBe(128);
      }
    }
  });

  it("leaves a zero border all the way round, so bilinear sampling decays to no effect", () => {
    const { texture } = field;
    for (let x = 0; x < texture.width; x++) {
      expect(at(texture, x, 0, 0)).toBe(128);
      expect(at(texture, x, texture.height - 1, 0)).toBe(128);
    }
    for (let y = 0; y < texture.height; y++) {
      expect(at(texture, 0, y, 0)).toBe(128);
      expect(at(texture, texture.width - 1, y, 0)).toBe(128);
    }
  });

  it("pulls the lower face inward, mirrored about the midline", () => {
    const { texture } = field;
    // The lower half of the field in image terms is the BOTTOM half of the rect,
    // which is the low-y half of a bottom-up buffer.
    const jawRow = Math.round(texture.height * 0.3);
    const at = (x: number) => at2(texture, x, jawRow, 0) - 128;

    // The stored field is the BACKWARD map: to draw a narrower jaw, the output pixel
    // on the right of the face fetches from further right, hence a positive x offset
    // there and a negative one on the left.
    // Peak magnitude, per side, rather than per texel. The field is antisymmetric in
    // sign and shape but NOT to the last texel: the mesh triangles overlap once
    // deformed and the rasterizer is last-write-wins, so which of two overlapping
    // triangles supplies a texel depends on their order in the list, and that order
    // is not mirror-symmetric. On a real (asymmetric) face the difference is
    // invisible; asserting per-texel equality would only be asserting the fan
    // ordering of the synthetic fixture.
    let leftPeak = 0;
    let rightPeak = 0;
    const middle = texture.width / 2;
    for (let x = 0; x < texture.width; x++) {
      if (x < middle) leftPeak = Math.max(leftPeak, -at(x));
      else rightPeak = Math.max(rightPeak, at(x));
    }
    expect(leftPeak).toBeGreaterThan(4);
    expect(rightPeak).toBeGreaterThan(4);
    expect(Math.abs(leftPeak - rightPeak)).toBeLessThanOrEqual(Math.max(leftPeak, rightPeak) * 0.15);

    // And it points inward on both sides, not outward.
    const quarter = Math.round(texture.width * 0.25);
    expect(at(quarter)).toBeLessThan(0);
    expect(at(texture.width - 1 - quarter)).toBeGreaterThan(0);
  });

  it("fetches from lower down at the chin, which is what shortens the face", () => {
    const { texture, frame, rect } = field;
    const chinX = Math.round(((frame.chin.x - rect.x) / rect.w) * texture.width);
    // A little above the chin in image terms, so the sample lands inside the mesh.
    const chinImageY = frame.chin.y - frame.faceHeight * 0.04;
    const chinRowFromTop = ((chinImageY - rect.y) / rect.h) * texture.height;
    const chinRow = Math.round(texture.height - 1 - chinRowFromTop);
    // v grows upward while image y grows downward, so "fetch from lower in the
    // image" is a NEGATIVE v offset, encoded below 128.
    expect(at(texture, chinX, chinRow, 1)).toBeLessThan(128);
  });

  it("keeps the eye field local: zero at the chin, non-zero at the eyes", () => {
    const { texture, frame, rect } = field;
    const toField = (x: number, y: number): [number, number] => [
      Math.round(((x - rect.x) / rect.w) * texture.width),
      Math.round(texture.height - 1 - ((y - rect.y) / rect.h) * texture.height),
    ];

    const [chinX, chinY] = toField(frame.chin.x, frame.chin.y - frame.faceHeight * 0.04);
    expect(at(texture, chinX, chinY, 2)).toBe(128);
    expect(at(texture, chinX, chinY, 3)).toBe(128);

    // Just inside the outer half of one eye, where the enlargement is strongest.
    const eye = frame.eyes[0];
    const [eyeX, eyeY] = toField(eye.center.x + eye.radius * 0.6, eye.center.y);
    expect(Math.abs(at(texture, eyeX, eyeY, 2) - 128)).toBeGreaterThan(1);
  });

  it("reports a range that covers its own encoded values without clipping", () => {
    const { texture, rangeU, rangeV } = field;
    expect(rangeU).toBeGreaterThan(0);
    expect(rangeV).toBeGreaterThan(0);
    // A saturated channel means a displacement was clamped, i.e. the range is too
    // small for the deformation rules and part of the face is silently capped.
    let min = 255;
    let max = 0;
    for (let i = 0; i < texture.data.length; i++) {
      const value = texture.data[i]!;
      if (value < min) min = value;
      if (value > max) max = value;
    }
    expect(min).toBeGreaterThan(0);
    expect(max).toBeLessThan(255);
  });

  it("scales the reported range with each axis, so a non-square image is not skewed", () => {
    const built = buildFaceFrame(SYNTHETIC_FACE.landmarks, SYNTHETIC_FACE.rings, 1000, 2000)!;
    const skirt = buildSkirtRing(
      built.points,
      SYNTHETIC_FACE.rings.faceOval,
      built.frame.faceWidth * SKIRT_OUTWARD_FACE_FRACTION,
    );
    const rect = computeGeomRect(skirt, built.frame.faceWidth * 0.06);
    const tall = buildWarpFieldCanvas(
      built.points,
      built.frame,
      SYNTHETIC_FACE.triangles,
      SYNTHETIC_FACE.rings.faceOval,
      rect,
      1000,
      2000,
    );
    // Same displacement in pixels, but v spans twice as many pixels, so the UV
    // range per axis has to differ by the same factor.
    expect(tall.rangeU / tall.rangeV).toBeCloseTo(2, 6);
  });
});
