import { buildFragmentShader, inputUniform, paramUniform } from "../rendering/shaders/glslCommon";
import type { FilterDefinition } from "../types";
import { chromaBlurPass, lumaBlurPass } from "./shared/skinRetouch";

/**
 * Band radii, as fractions of face width. See shared/skinRetouch.ts for what the
 * pair means. This filter can afford a wide low band because it only attenuates
 * the middle one modestly — it is not trying to erase anything structural.
 */
const BAND_LOW_RADIUS = 0.06;
const BAND_MID_RADIUS = 0.01;

/**
 * Natural skin retouch.
 *
 * The three-band frequency split this is built on lives in shared/skinRetouch.ts,
 * which explains why the MIDDLE band is the one that gets attenuated. What is
 * specific to this filter is the recombination below: the blotch band is weakened
 * but the pore band is KEPT, and the texture slider can push it above 1.0 to add
 * microtexture back. That is the whole difference between this and the makeup
 * filter, which crushes both bands on purpose.
 */

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
    { id: "chromaH", fragmentSource: chromaBlurPass("x", "original"), outputScale: 0.5 },
    { id: "chromaV", fragmentSource: chromaBlurPass("y", "chain"), outputScale: 0.5 },
    { id: "lumaLowH", fragmentSource: lumaBlurPass("x", "original", BAND_LOW_RADIUS), outputScale: 0.5 },
    { id: "lumaLowV", fragmentSource: lumaBlurPass("y", "chain", BAND_LOW_RADIUS), outputScale: 0.5 },
    { id: "lumaMidH", fragmentSource: lumaBlurPass("x", "original", BAND_MID_RADIUS) },
    { id: "lumaMidV", fragmentSource: lumaBlurPass("y", "chain", BAND_MID_RADIUS) },
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
