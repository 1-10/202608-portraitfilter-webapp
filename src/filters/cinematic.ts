import { buildFragmentShader, inputUniform, paramUniform } from "../rendering/shaders/glslCommon";
import type { FilterDefinition } from "../types";

/**
 * Cinematic — a film response rather than a color overlay.
 *
 * This is the "remove the rawness without breaking the reality line" approach done
 * purely tonally: a filmic curve compresses the highlights the way an emulsion
 * does, halation bleeds warm light out of the brightest areas, and split-toning
 * separates shadows from highlights by hue. Skin stops looking clinically
 * photographic without a single pixel of its geometry being touched.
 *
 * Two things here are deliberate and are what keep it from looking like a preset
 * Instagram filter:
 *  - Tone mapping happens in LINEAR light, because that is where film's response
 *    actually lives. Applying an S-curve to sRGB values crushes shadows unevenly.
 *  - Split-toning happens in OKLab and touches only the a/b axes, never lightness.
 *    HSV-based split-toning always darkens as it saturates, which is exactly why
 *    "teal shadows" usually mud up the blacks.
 */

/** Pass 1 — exposure, log-space contrast, filmic curve, saturation. */
const gradePass = buildFragmentShader({
  extraUniforms: `uniform float ${paramUniform("exposure")};\nuniform float ${paramUniform("contrast")};\nuniform float ${paramUniform("warmth")};\nuniform float ${paramUniform("saturation")};`,
  body: `
  vec3 lin = srgbToLinear(texture(uOriginal, vUv).rgb);
  lin *= exp2((${paramUniform("exposure")} - 50.0) / 50.0 * 0.8);

  // White balance before the curve, so the curve shapes the corrected image.
  float warmth = (${paramUniform("warmth")} - 50.0) / 50.0;
  lin *= vec3(1.0 + warmth * 0.10, 1.0, 1.0 - warmth * 0.10);

  // Contrast pivoted at 18% grey in log space, so the slider changes contrast
  // without also shifting exposure.
  float c = 0.7 + ${paramUniform("contrast")} / 100.0 * 0.8;
  lin = vec3(0.18) * pow(max(lin / 0.18, vec3(1.0e-4)), vec3(c));

  // Narkowicz ACES fit: a filmic toe and shoulder in one rational function.
  const float A = 2.51, B = 0.03, C2 = 2.43, D = 0.59, E = 0.14;
  lin = clamp((lin * (A * lin + B)) / (lin * (C2 * lin + D) + E), 0.0, 1.0);

  vec3 lab = linearRgbToOklab(lin);
  lab.yz *= 0.6 + ${paramUniform("saturation")} / 100.0 * 0.9;
  lin = max(oklabToLinearRgb(lab), 0.0);

  fragColor = vec4(linearToSrgb(clamp(lin, 0.0, 1.0)), 1.0);`,
});

/**
 * Pass 2 — highlight extraction for halation, at quarter resolution.
 * Weighted toward red because film halation IS red: long wavelengths scatter
 * furthest through the emulsion before the anti-halation backing absorbs them.
 */
const bloomCutPass = buildFragmentShader({
  extraUniforms: `uniform float ${paramUniform("halation")};`,
  body: `
  vec3 lin = srgbToLinear(texture(uSource, vUv).rgb);
  vec3 h = max(lin - 0.62, 0.0);
  fragColor = vec4(linearToSrgb(clamp(h * vec3(1.0, 0.42, 0.18) * 2.0, 0.0, 1.0)), 1.0);`,
});

/** Passes 3-4 — separable blur of the halation, radius in image-relative units. */
function bloomBlurPass(axis: "x" | "y"): string {
  const offset =
    axis === "x" ? "vec2(float(i) * radius / uResolution.x, 0.0)" : "vec2(0.0, float(i) * radius / uResolution.y)";
  return buildFragmentShader({
    body: `
  // Radius as a fraction of the long edge, so the glow is the same size relative
  // to the subject at preview and at export.
  float radius = 0.012 * max(uResolution.x, uResolution.y);
  vec3 sum = vec3(0.0);
  float wsum = 0.0;
  for (int i = -6; i <= 6; i++) {
    float w = gaussW(float(i), 3.0);
    sum += texture(uSource, vUv + ${offset}).rgb * w;
    wsum += w;
  }
  fragColor = vec4(sum / wsum, 1.0);`,
  });
}

/** Pass 5 — add halation, then split-tone in OKLab. */
const tonePass = buildFragmentShader({
  extraUniforms: `uniform sampler2D ${inputUniform("grade")};\nuniform float ${paramUniform("halation")};\nuniform float ${paramUniform("splitTone")};\nuniform float ${paramUniform("monochrome")};`,
  body: `
  vec3 lin = srgbToLinear(texture(${inputUniform("grade")}, vUv).rgb);
  vec3 glow = srgbToLinear(texture(uSource, vUv).rgb);
  lin += glow * (${paramUniform("halation")} / 100.0) * 0.8;
  lin = clamp(lin, 0.0, 1.0);

  vec3 lab = linearRgbToOklab(lin);

  // Monochrome is a preset of this filter rather than a separate one: collapse
  // chroma first, then let split-toning tint the result if it is enabled.
  float mono = ${paramUniform("monochrome")} / 100.0;
  lab.yz = mix(lab.yz, vec2(0.0), mono);

  float shadowW = 1.0 - smoothstep(0.15, 0.55, lab.x);
  float highW = smoothstep(0.50, 0.95, lab.x);
  vec2 shadowTint = vec2(-0.020, -0.030);
  vec2 highTint = vec2(0.018, 0.030);
  // Only a/b move; lightness is untouched, so tinting provably cannot darken.
  lab.yz += (shadowTint * shadowW + highTint * highW) * (${paramUniform("splitTone")} / 100.0);

  lin = max(oklabToLinearRgb(lab), 0.0);
  fragColor = vec4(linearToSrgb(clamp(lin, 0.0, 1.0)), 1.0);`,
});

/** Pass 6 — grain and vignette, then blend with the original by strength. */
const finishPass = buildFragmentShader({
  extraUniforms: `uniform float ${paramUniform("grain")};\nuniform float ${paramUniform("vignette")};`,
  body: `
  vec3 col = texture(uSource, vUv).rgb;

  // Resolution-independent: the same grain size relative to the frame at 140px,
  // 1280px and 4096px, instead of getting ~3x finer in a big export.
  vec2 gu = stableNoiseUv(vUv, uResolution, 620.0);
  float L = luminance(col);
  // Real film grain peaks in the midtones and disappears in clipped blacks/whites.
  float amp = ${paramUniform("grain")} / 100.0 * 0.16 * L * (1.0 - L) * 4.0;
  vec3 noise = vec3(
    valueNoise(gu) - 0.5,
    valueNoise(gu + 37.0) - 0.5,
    valueNoise(gu + 91.0) - 0.5
  );
  col += noise * vec3(1.0, 0.7, 0.7) * amp;

  vec2 d = vUv - 0.5;
  d.x *= uResolution.x / max(uResolution.y, 1.0);
  float vig = 1.0 - smoothstep(0.35, 0.85, length(d)) * (${paramUniform("vignette")} / 100.0) * 0.55;
  col *= vig;

  vec3 original = texture(uOriginal, vUv).rgb;
  fragColor = vec4(mix(original, clamp(col, 0.0, 1.0), uStrength), 1.0);`,
});

export const cinematicFilter: FilterDefinition = {
  id: "cinematic",
  name: "シネマティック",
  description:
    "フィルミックなトーンカーブ、ハレーション、OKLabのスプリットトーンで、形はそのままに写真の生々しさだけを抜いて映画的な質感にします。",
  parameters: [
    { id: "exposure", label: "露出", min: 0, max: 100, step: 1, defaultValue: 50 },
    { id: "contrast", label: "コントラスト", min: 0, max: 100, step: 1, defaultValue: 55 },
    { id: "warmth", label: "色温度", min: 0, max: 100, step: 1, defaultValue: 55 },
    { id: "saturation", label: "彩度", min: 0, max: 100, step: 1, defaultValue: 50 },
    { id: "halation", label: "ハレーション", min: 0, max: 100, step: 1, defaultValue: 35 },
    { id: "splitTone", label: "スプリットトーン", min: 0, max: 100, step: 1, defaultValue: 55 },
    { id: "grain", label: "粒子", min: 0, max: 100, step: 1, defaultValue: 30 },
    { id: "vignette", label: "周辺減光", min: 0, max: 100, step: 1, defaultValue: 30 },
    { id: "monochrome", label: "モノクロ", min: 0, max: 100, step: 1, defaultValue: 0 },
  ],
  passes: [
    { id: "grade", fragmentSource: gradePass },
    { id: "bloomCut", fragmentSource: bloomCutPass, outputScale: 0.25 },
    { id: "bloomH", fragmentSource: bloomBlurPass("x"), outputScale: 0.25 },
    { id: "bloomV", fragmentSource: bloomBlurPass("y"), outputScale: 0.25 },
    { id: "tone", fragmentSource: tonePass, extraInputs: ["grade"] },
    { id: "finish", fragmentSource: finishPass },
  ],
  presets: [
    {
      id: "teal-orange",
      label: "ティール&オレンジ",
      values: { exposure: 50, contrast: 62, warmth: 58, saturation: 55, halation: 35, splitTone: 75, grain: 25, vignette: 35, monochrome: 0 },
    },
    {
      id: "warm-film",
      label: "ウォームフィルム",
      values: { exposure: 54, contrast: 48, warmth: 68, saturation: 45, halation: 55, splitTone: 35, grain: 45, vignette: 30, monochrome: 0 },
    },
    {
      id: "monochrome",
      label: "モノクローム",
      values: { exposure: 52, contrast: 68, warmth: 50, saturation: 0, halation: 30, splitTone: 20, grain: 55, vignette: 40, monochrome: 100 },
    },
  ],
};
