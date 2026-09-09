/**
 * Packing for CPU-produced auxiliary textures.
 *
 * These textures carry DATA in all four channels — masks, displacement vectors —
 * not a picture. That rules out routing them through a canvas: a 2D canvas stores
 * premultiplied colour, so `putImageData` with any alpha below 255 quantizes the
 * RGB channels it was supposed to be preserving, and the loss is silent. Canvas 2D
 * is still the right tool for RASTERIZING each layer (paths, gradients, blur), but
 * the packed result is handed to WebGL as a plain byte array instead.
 *
 * Row-order contract: aux data is packed BOTTOM-UP, row 0 being the bottom of the
 * region. `UNPACK_FLIP_Y_WEBGL` has no effect on array uploads, so the flip that
 * image-source uploads get for free has to happen here instead.
 */

export type AuxTextureData = {
  data: Uint8Array;
  width: number;
  height: number;
};

/** One channel's worth of coverage, in top-down image order, `width * height` bytes. */
export type Plane = Uint8Array | Uint8ClampedArray;

/**
 * Interleaves four top-down single-channel planes into one bottom-up RGBA buffer.
 * A missing plane is written as zero.
 */
export function packPlanes(
  planes: readonly [Plane | null, Plane | null, Plane | null, Plane | null],
  width: number,
  height: number,
): AuxTextureData {
  const data = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    const srcRow = (height - 1 - y) * width;
    const dstRow = y * width;
    for (let x = 0; x < width; x++) {
      const src = srcRow + x;
      const dst = (dstRow + x) * 4;
      data[dst] = planes[0] ? planes[0][src]! : 0;
      data[dst + 1] = planes[1] ? planes[1][src]! : 0;
      data[dst + 2] = planes[2] ? planes[2][src]! : 0;
      data[dst + 3] = planes[3] ? planes[3][src]! : 0;
    }
  }
  return { data, width, height };
}

/**
 * Extracts the alpha channel of an RGBA canvas readback as a plane.
 *
 * Alpha is the only channel a canvas readback reports exactly: colour comes back
 * un-premultiplied, which is lossy at low alpha. Every layer is therefore drawn as
 * opaque white and read as coverage.
 */
export function alphaPlane(rgba: Uint8ClampedArray, pixelCount: number): Uint8ClampedArray {
  const plane = new Uint8ClampedArray(pixelCount);
  for (let i = 0; i < pixelCount; i++) plane[i] = rgba[i * 4 + 3]!;
  return plane;
}

/**
 * Separable box blur over a single-channel plane, repeated three times to
 * approximate a Gaussian. Used where `ctx.filter` is unavailable, and wherever a
 * layer needs a blur radius of its own after being rasterized.
 */
export function blurPlane(plane: Uint8ClampedArray, width: number, height: number, radius: number): void {
  const r = Math.round(radius);
  if (r < 1) return;
  const temp = new Uint8ClampedArray(plane.length);
  for (let pass = 0; pass < 3; pass++) {
    boxBlurAxis(plane, temp, width, height, r, true);
    boxBlurAxis(temp, plane, width, height, r, false);
  }
}

function boxBlurAxis(
  src: Uint8ClampedArray,
  dst: Uint8ClampedArray,
  width: number,
  height: number,
  radius: number,
  horizontal: boolean,
): void {
  const outer = horizontal ? height : width;
  const inner = horizontal ? width : height;
  const window = radius * 2 + 1;
  for (let o = 0; o < outer; o++) {
    const at = (i: number): number => (horizontal ? o * width + i : i * width + o);
    // Running sum: the window is clamped at both ends, so the first and last
    // `radius` samples are counted repeatedly rather than shrinking the window,
    // which keeps the plane's edges from darkening.
    let sum = 0;
    for (let k = -radius; k <= radius; k++) {
      sum += src[at(Math.min(inner - 1, Math.max(0, k)))]!;
    }
    for (let i = 0; i < inner; i++) {
      dst[at(i)] = sum / window;
      const leaving = src[at(Math.min(inner - 1, Math.max(0, i - radius)))]!;
      const entering = src[at(Math.min(inner - 1, Math.max(0, i + radius + 1)))]!;
      sum += entering - leaving;
    }
  }
}
