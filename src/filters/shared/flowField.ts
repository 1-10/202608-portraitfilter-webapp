import { buildFragmentShader, inputUniform } from "../../rendering/shaders/glslCommon";

/**
 * A smooth local flow field from the structure tensor: the direction intensity
 * varies least at every pixel, which follows hair strands and jawlines.
 *
 * Three passes, in order: tensor (from a source image), tensor blur, flow. The
 * flow pass's output is
 *   rg = local edge TANGENT encoded to 0..1   b = anisotropy   a = edge strength
 * and filters read it through extraInputs.
 *
 * The tensor pass's input should be free of sensor noise and compression blocking
 * where that matters (a denoise pass), since those enter the gradients directly;
 * the blur in tensor space removes much of it either way.
 */

/**
 * Pass 2 — structure tensor. Encodes how intensity varies locally, which is what
 * lets later passes know the direction of the "brush stroke" at every pixel.
 * Packed into 8 bits, which is lossy, but pass 3 immediately blurs it and the
 * resulting orientation error is far below a degree at any useful anisotropy.
 */
export function flowTensorPass(source: "original" | { pass: string }): string {
  const sampler = source === "original" ? "uOriginal" : inputUniform(source.pass);
  return buildFragmentShader({
  extraUniforms: source === "original" ? "" : `uniform sampler2D ${sampler};`,
  body: `
  vec2 texel = 1.0 / uResolution;
  vec3 gx = vec3(0.0);
  vec3 gy = vec3(0.0);
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      vec3 s = texture(${sampler}, vUv + vec2(float(i), float(j)) * texel).rgb;
      float wx = float(i) * (j == 0 ? 2.0 : 1.0);
      float wy = float(j) * (i == 0 ? 2.0 : 1.0);
      gx += s * wx;
      gy += s * wy;
    }
  }
  gx *= 0.25;
  gy *= 0.25;
  float E = dot(gx, gx);
  float F = dot(gx, gy);
  float G = dot(gy, gy);
  fragColor = vec4(E / 3.0, F / 6.0 + 0.5, G / 3.0, 1.0);`,
  });
}

/**
 * Pass 3 — blur the tensor. Averaging in tensor space (rather than averaging
 * directions, which would cancel out) is what turns a noisy per-pixel gradient
 * into a smooth, coherent flow field that follows hair and jawlines.
 */
export const flowTensorBlurPass = buildFragmentShader({
  body: `
  vec2 texel = 1.0 / uResolution;
  vec3 sum = vec3(0.0);
  float wsum = 0.0;
  for (int j = -2; j <= 2; j++) {
    for (int i = -2; i <= 2; i++) {
      float w = gaussW(length(vec2(float(i), float(j))), 1.6);
      sum += texture(uSource, vUv + vec2(float(i), float(j)) * texel).rgb * w;
      wsum += w;
    }
  }
  fragColor = vec4(sum / wsum, 1.0);`,
});

/**
 * Pass 4 — eigen-decompose into a flow field.
 *   rg = local edge TANGENT (direction of least variation) encoded to 0..1
 *   b  = anisotropy: how directional this neighbourhood is
 *   a  = smoothed edge strength
 *
 * That alpha is the honest structural replacement for sobelEdge(uOriginal, ...):
 * computed from a denoised image and blurred in tensor space, so pores and
 * compression artifacts do not survive it but a jawline does.
 */
export const flowPass = buildFragmentShader({
  body: `
  vec3 t = texture(uSource, vUv).rgb;
  float E = t.r * 3.0;
  float F = (t.g - 0.5) * 6.0;
  float G = t.b * 3.0;

  float d = sqrt(max((E - G) * (E - G) + 4.0 * F * F, 0.0));
  float l1 = 0.5 * (E + G + d);
  float l2 = 0.5 * (E + G - d);

  // Eigenvector of the MINOR eigenvalue = direction of least intensity change,
  // i.e. the direction the edge runs along.
  vec2 v = vec2(l1 - E, -F);
  vec2 tangent = length(v) > 1.0e-6 ? normalize(v) : vec2(0.0, 1.0);
  float aniso = (l1 + l2) > 1.0e-6 ? (l1 - l2) / (l1 + l2) : 0.0;

  fragColor = vec4(tangent * 0.5 + 0.5, aniso, clamp(sqrt(l1), 0.0, 1.0));`,
});
