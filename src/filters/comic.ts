import { buildFragmentShader, inputUniform, paramUniform } from "../rendering/shaders/glslCommon";
import type { FilterDefinition } from "../types";

const posterizePass = buildFragmentShader({
  extraUniforms: `uniform float ${paramUniform("contrast")};`,
  body: `
  vec3 color = texture(uSource, vUv).rgb;
  float contrastAmt = mix(0.8, 2.2, ${paramUniform("contrast")} / 100.0);
  color = clamp((color - 0.5) * contrastAmt + 0.5, 0.0, 1.0);
  color = quantizeColor(color, 4.0);
  fragColor = vec4(color, 1.0);`,
});

const edgePass = buildFragmentShader({
  body: `
  float edgeMag = sobelEdge(uOriginal, vUv, uTexelSize);
  float ink = step(0.35, edgeMag);
  fragColor = vec4(vec3(1.0 - ink), 1.0);`,
});

const halftonePass = buildFragmentShader({
  extraUniforms: `uniform sampler2D ${inputUniform("posterize")};\nuniform float ${paramUniform("dotSize")};`,
  body: `
  vec3 posterized = texture(${inputUniform("posterize")}, vUv).rgb;
  float edgeMask = texture(uSource, vUv).r;
  float lum = luminance(posterized);

  float dotSizePx = mix(4.0, 18.0, ${paramUniform("dotSize")} / 100.0);
  float angle = 0.4363;
  mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
  vec2 grid = rot * (vUv * uResolution) / dotSizePx;
  vec2 cellUv = fract(grid) - 0.5;
  float dist = length(cellUv);

  float shadowAmount = 1.0 - smoothstep(0.15, 0.6, lum);
  float dotRadius = shadowAmount * 0.62;
  float dot = 1.0 - smoothstep(dotRadius - 0.06, dotRadius + 0.06, dist);
  vec3 halftoned = mix(posterized, vec3(0.05), dot * shadowAmount);

  vec3 inked = halftoned * edgeMask;
  vec3 original = texture(uOriginal, vUv).rgb;
  fragColor = vec4(mix(original, inked, uStrength), 1.0);`,
});

export const comicFilter: FilterDefinition = {
  id: "comic",
  name: "コミック",
  description: "高コントラストな輪郭線とシャドウのハーフトーンで印刷コミック風に仕上げます。",
  parameters: [
    { id: "dotSize", label: "網点サイズ", min: 0, max: 100, step: 1, defaultValue: 45 },
    { id: "contrast", label: "コントラスト", min: 0, max: 100, step: 1, defaultValue: 60 },
  ],
  passes: [
    { id: "posterize", fragmentSource: posterizePass },
    { id: "edge", fragmentSource: edgePass },
    { id: "halftone", fragmentSource: halftonePass, extraInputs: ["posterize"] },
  ],
};
