import { buildFragmentShader, paramUniform } from "../rendering/shaders/glslCommon";
import type { FilterDefinition } from "../types";

export const DUOTONE_PALETTE_NAMES = ["ネイビー×クリーム", "セピア", "ティール×コーラル", "パープル×ゴールド", "モノブルー"];

const duotonePass = buildFragmentShader({
  extraUniforms: `uniform float ${paramUniform("palette")};\nuniform float ${paramUniform("contrast")};`,
  body: `
  vec3 color = texture(uSource, vUv).rgb;
  float contrastAmt = mix(0.8, 1.8, ${paramUniform("contrast")} / 100.0);
  float lum = luminance(color);
  lum = clamp((lum - 0.5) * contrastAmt + 0.5, 0.0, 1.0);

  vec3 shadowColor;
  vec3 highColor;
  int preset = int(${paramUniform("palette")} + 0.5);
  if (preset <= 0) {
    shadowColor = vec3(0.106, 0.165, 0.290);
    highColor = vec3(0.961, 0.914, 0.788);
  } else if (preset == 1) {
    shadowColor = vec3(0.227, 0.141, 0.082);
    highColor = vec3(0.910, 0.788, 0.627);
  } else if (preset == 2) {
    shadowColor = vec3(0.059, 0.239, 0.243);
    highColor = vec3(1.0, 0.620, 0.490);
  } else if (preset == 3) {
    shadowColor = vec3(0.169, 0.106, 0.290);
    highColor = vec3(0.941, 0.788, 0.416);
  } else {
    shadowColor = vec3(0.063, 0.075, 0.102);
    highColor = vec3(0.812, 0.910, 1.0);
  }

  vec3 duotoned = mix(shadowColor, highColor, lum);
  vec3 original = texture(uOriginal, vUv).rgb;
  fragColor = vec4(mix(original, duotoned, uStrength), 1.0);`,
});

export const duotoneFilter: FilterDefinition = {
  id: "duotone",
  name: "デュオトーン",
  description: "輝度を基準に2色へマッピングし、はっきりとした配色に仕上げます。",
  parameters: [
    {
      id: "palette",
      label: "配色プリセット",
      min: 0,
      max: DUOTONE_PALETTE_NAMES.length - 1,
      step: 1,
      defaultValue: 0,
      options: DUOTONE_PALETTE_NAMES,
    },
    { id: "contrast", label: "コントラスト", min: 0, max: 100, step: 1, defaultValue: 55 },
  ],
  passes: [{ id: "duotone", fragmentSource: duotonePass }],
};
