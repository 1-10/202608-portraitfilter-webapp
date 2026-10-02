import { buildFragmentShader, inputUniform } from "../../rendering/shaders/glslCommon";

/**
 * Self-guided filter on OKLab lightness: an edge-preserving smooth.
 *
 * A Gaussian band split removes pores by blurring, and blurs every edge the gate
 * does not fully protect along with them — nostrils, the lip line, the edge of a
 * brow — which is what separates "retouched" skin from the clean-interior,
 * crisp-edge finish of a generated portrait. The guided filter flattens deviations
 * smaller than sqrt(eps) and keeps larger ones as they are, so the interior goes
 * flat while the edges stay put.
 *
 * Output L at a pixel is a * L + b, with (a, b) the window-averaged linear fit of L
 * on itself:
 *
 *   pass coef   per window: a = var / (var + eps), b = mean * (1 - a)
 *   pass mean   window-average of (a, b)
 *
 * The render targets are 8-bit, which cannot hold a or b to the precision the
 * recombination needs, so each is packed into two channels (high and low byte). The
 * window statistics are accumulated in shader floats, never stored, which is why
 * the first pass computes mean and variance itself instead of storing L and L^2.
 *
 * The window radius is a fraction of face width. Taps are spread over the window on
 * a fixed grid, so the cost is constant whatever the radius.
 */

/** GLSL for the two-byte encoding of a 0..1 value. */
export const PACK16_GLSL = `
vec2 pack16(float v) {
  float x = clamp(v, 0.0, 1.0) * 255.0;
  float hi = floor(x);
  return vec2(hi / 255.0, x - hi);
}
float unpack16(vec2 p) {
  return p.x + p.y / 255.0;
}`;

const TAPS = 6;

function windowLoop(radiusFactor: number, sampleBody: string): string {
  return `
  float faceWidthPx = max(uFaceScale * uResolution.x * 2.5, 24.0);
  float radiusPx = max(faceWidthPx * ${radiusFactor.toFixed(4)}, 1.0);
  vec2 stepUv = (radiusPx / ${TAPS.toFixed(1)}) / uResolution;
  float count = 0.0;
  for (int y = -${TAPS}; y <= ${TAPS}; y++) {
    for (int x = -${TAPS}; x <= ${TAPS}; x++) {
      vec2 uv = vUv + vec2(float(x), float(y)) * stepUv;
${sampleBody}
      count += 1.0;
    }
  }`;
}

/** Per-window coefficients (a, b) from the filter's original image, packed. */
export function guidedCoefPass(radiusFactor: number, eps: number): string {
  return buildFragmentShader({
    extraUniforms: PACK16_GLSL,
    body: `
  float sum = 0.0;
  float sum2 = 0.0;${windowLoop(
    radiusFactor,
    `      float l = linearRgbToOklab(srgbToLinear(texture(uOriginal, uv).rgb)).x;
      sum += l;
      sum2 += l * l;`,
  )}
  float mean = sum / count;
  float variance = max(sum2 / count - mean * mean, 0.0);
  float a = variance / (variance + ${eps.toFixed(6)});
  float b = mean * (1.0 - a);
  fragColor = vec4(pack16(a), pack16(b));`,
  });
}

/** Window-average of the packed (a, b) from the previous pass, packed again. */
export function guidedMeanPass(radiusFactor: number): string {
  return buildFragmentShader({
    extraUniforms: PACK16_GLSL,
    body: `
  float sumA = 0.0;
  float sumB = 0.0;${windowLoop(
    radiusFactor,
    `      vec4 p = texture(uSource, uv);
      sumA += unpack16(p.xy);
      sumB += unpack16(p.zw);`,
  )}
  fragColor = vec4(pack16(sumA / count), pack16(sumB / count));`,
  });
}

/** GLSL expression for the guided-filter output at `vUv`, given the mean pass's sampler and the pixel's L. */
export function guidedOutput(meanPassId: string, lightness: string): string {
  return `(unpack16(texture(${inputUniform(meanPassId)}, vUv).xy) * (${lightness}) + unpack16(texture(${inputUniform(meanPassId)}, vUv).zw))`;
}
