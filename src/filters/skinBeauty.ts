import { buildFragmentShader, inputUniform, paramUniform } from "../rendering/shaders/glslCommon";
import type { FilterDefinition } from "../types";

/**
 * Natural skin retouch.
 *
 * The failure mode of every naive "beauty" filter is plastic skin, and it has one
 * cause: they split the image into low and high frequency and then attenuate the
 * HIGH band — which is exactly where pores and fine hair live.
 *
 * This splits luminance into THREE bands instead:
 *
 *   Llow    = wide blur      -> shading and 3-D form. Carries the likeness. Kept.
 *   blemish = Lmid - Llow    -> blotches, oil sheen, uneven tone. Attenuated.
 *   pore    = L    - Lmid    -> pores, fine hair, real skin texture. KEPT (and the
 *                               slider can push it above 1.0 to add texture back).
 *
 * Attenuating only the middle band removes what people actually mean by "blotchy"
 * while leaving both the modelling that makes a face look three-dimensional and the
 * microtexture that makes it read as skin rather than vinyl.
 *
 * The second half of the effect is chroma: blotchy redness is overwhelmingly a
 * CHROMA variance, not a luminance one, so the a/b axes are smoothed far harder
 * than lightness. The eye cannot resolve chroma detail at that scale, so this
 * removes redness and unevenness at essentially no cost in apparent sharpness.
 */

/** Separable blur of OKLab a/b, at half resolution — chroma is perceptually low-frequency. */
function chromaBlurPass(axis: "x" | "y", readFromOriginal: boolean): string {
  const src = readFromOriginal ? "uOriginal" : "uSource";
  const offset =
    axis === "x" ? "vec2(float(i) * stepPx / uResolution.x, 0.0)" : "vec2(0.0, float(i) * stepPx / uResolution.y)";
  const decode = readFromOriginal
    ? `linearRgbToOklab(srgbToLinear(texture(${src}, uv).rgb)).yz`
    : `texture(${src}, uv).xy * 0.6 - 0.3`;
  return buildFragmentShader({
    body: `
  // Sigma is a fraction of FACE WIDTH so a headshot and a full-body shot get the
  // same evening-out on the skin itself. The 13 taps are spaced at sigma/2, giving
  // a total reach of 3 sigma — deliberately, because treating this value as the
  // per-tap step instead made the kernel reach 6x further than intended and drag
  // hair and wall color onto the forehead.
  float faceWidthPx = max(uFaceScale * uResolution.x * 2.5, 24.0);
  float sigmaPx = max(faceWidthPx * 0.05, 1.0);
  float stepPx = sigmaPx * 0.5;
  vec2 sum = vec2(0.0);
  float wsum = 0.0;
  for (int i = -6; i <= 6; i++) {
    float w = gaussW(float(i), 2.0);
    vec2 uv = vUv + ${offset};
    sum += (${decode}) * w;
    wsum += w;
  }
  vec2 ab = sum / wsum;
  // Packed into 0..1; a/b stay well inside +-0.3 for real images.
  fragColor = vec4((ab + 0.3) / 0.6, 0.0, 1.0);`,
  });
}

/** Separable blur of OKLab lightness at a given face-relative radius. */
function lumaBlurPass(axis: "x" | "y", readFromOriginal: boolean, radiusFactor: number): string {
  const src = readFromOriginal ? "uOriginal" : "uSource";
  const offset =
    axis === "x" ? "vec2(float(i) * stepPx / uResolution.x, 0.0)" : "vec2(0.0, float(i) * stepPx / uResolution.y)";
  const decode = readFromOriginal ? `linearRgbToOklab(srgbToLinear(texture(${src}, uv).rgb)).x` : `texture(${src}, uv).r`;
  return buildFragmentShader({
    body: `
  float faceWidthPx = max(uFaceScale * uResolution.x * 2.5, 24.0);
  float sigmaPx = max(faceWidthPx * ${radiusFactor.toFixed(4)}, 0.6);
  float stepPx = sigmaPx * 0.5;
  float sum = 0.0;
  float wsum = 0.0;
  for (int i = -6; i <= 6; i++) {
    float w = gaussW(float(i), 2.0);
    vec2 uv = vUv + ${offset};
    sum += (${decode}) * w;
    wsum += w;
  }
  fragColor = vec4(sum / wsum, 0.0, 0.0, 1.0);`,
  });
}

/**
 * Final recombination. Everything is gated by THREE multiplied terms, and all
 * three are needed:
 *   - the face/skin region (MediaPipe, or the color heuristic as fallback)
 *   - explicit exclusion of eyes, brows and lips
 *   - an edge term, so hair falling across a cheek, glasses frames and the jaw
 *     boundary are not smoothed
 * Without that last term you get the classic "wax mask with a glowing rim" halo.
 */
const compositePass = buildFragmentShader({
  extraUniforms: `uniform sampler2D ${inputUniform("chromaV")};\nuniform sampler2D ${inputUniform("lumaLowV")};\nuniform sampler2D ${inputUniform("lumaMidV")};\nuniform float ${paramUniform("evenness")};\nuniform float ${paramUniform("dullness")};\nuniform float ${paramUniform("shine")};\nuniform float ${paramUniform("texture")};\nuniform float ${paramUniform("brightness")};`,
  body: `
  vec3 origSrgb = texture(uOriginal, vUv).rgb;
  vec3 origLin = srgbToLinear(origSrgb);
  vec3 lab = linearRgbToOklab(origLin);

  float Llow = texture(${inputUniform("lumaLowV")}, vUv).r;
  float Lmid = texture(${inputUniform("lumaMidV")}, vUv).r;
  vec2 abLow = texture(${inputUniform("chromaV")}, vUv).xy * 0.6 - 0.3;

  float blemish = Lmid - Llow;
  float pore = lab.x - Lmid;

  // Gate 1: is this skin at all?
  float gate = faceSkinWeight(origLin, vUv);
  // Gate 2: never touch the features that carry the likeness.
  gate *= 1.0 - faceFeatureWeight(vUv);
  // Gate 3: stay off structural boundaries (hair over cheek, glasses, jawline).
  float edge = sobelEdge01(uOriginal, vUv, uOriginalTexelSize);
  gate *= 1.0 - smoothstep(0.10, 0.32, edge);
  gate = clamp(gate, 0.0, 1.0);

  float kBlemish = mix(1.0, 0.18, ${paramUniform("evenness")} / 100.0 * gate);
  float kPore = 0.85 + ${paramUniform("texture")} / 100.0 * 0.45;
  float L2 = Llow + blemish * kBlemish + pore * kPore;

  // Chroma is attacked far harder than luma: blotchy redness IS chroma variance,
  // and smoothing it costs no apparent sharpness.
  vec2 ab2 = mix(lab.yz, abLow, (${paramUniform("dullness")} / 100.0) * 0.85 * gate);

  // Specular shine is bright AND desaturated, which is what separates it from a
  // simply light complexion.
  float C = length(ab2);
  float spec = smoothstep(0.72, 0.95, L2) * (1.0 - smoothstep(0.04, 0.10, C)) * gate;
  L2 -= spec * (${paramUniform("shine")} / 100.0) * 0.12;
  ab2 *= 1.0 + spec * (${paramUniform("shine")} / 100.0) * 0.25;

  L2 += (${paramUniform("brightness")} - 50.0) / 50.0 * 0.05 * gate;

  vec3 outLin = max(oklabToLinearRgb(vec3(L2, ab2)), 0.0);
  vec3 outSrgb = linearToSrgb(clamp(outLin, 0.0, 1.0));
  fragColor = vec4(mix(origSrgb, outSrgb, uStrength), 1.0);`,
});

export const skinBeautyFilter: FilterDefinition = {
  id: "skin-beauty",
  name: "ナチュラル美肌",
  description:
    "毛穴や質感を残したまま、肌のムラ・赤み・くすみ・テカリだけを抑えます。周波数を3帯域に分け、中間帯だけを弱めることでプラスチック肌を避けています。",
  parameters: [
    { id: "evenness", label: "肌の均一さ", min: 0, max: 100, step: 1, defaultValue: 55 },
    { id: "dullness", label: "赤み・くすみ除去", min: 0, max: 100, step: 1, defaultValue: 60 },
    { id: "shine", label: "テカリ抑制", min: 0, max: 100, step: 1, defaultValue: 45 },
    { id: "texture", label: "質感キープ", min: 0, max: 100, step: 1, defaultValue: 55 },
    { id: "brightness", label: "明るさ", min: 0, max: 100, step: 1, defaultValue: 55 },
  ],
  passes: [
    { id: "chromaH", fragmentSource: chromaBlurPass("x", true), outputScale: 0.5 },
    { id: "chromaV", fragmentSource: chromaBlurPass("y", false), outputScale: 0.5 },
    { id: "lumaLowH", fragmentSource: lumaBlurPass("x", true, 0.060), outputScale: 0.5 },
    { id: "lumaLowV", fragmentSource: lumaBlurPass("y", false, 0.060), outputScale: 0.5 },
    { id: "lumaMidH", fragmentSource: lumaBlurPass("x", true, 0.010) },
    { id: "lumaMidV", fragmentSource: lumaBlurPass("y", false, 0.010) },
    {
      id: "composite",
      fragmentSource: compositePass,
      extraInputs: ["chromaV", "lumaLowV", "lumaMidV"],
    },
  ],
  presets: [
    {
      id: "natural",
      label: "ナチュラル",
      values: { evenness: 45, dullness: 50, shine: 35, texture: 65, brightness: 53 },
    },
    {
      id: "polished",
      label: "しっかり",
      values: { evenness: 75, dullness: 75, shine: 65, texture: 45, brightness: 58 },
    },
    {
      id: "glow",
      label: "グロウ",
      values: { evenness: 60, dullness: 65, shine: 25, texture: 55, brightness: 68 },
    },
  ],
};
