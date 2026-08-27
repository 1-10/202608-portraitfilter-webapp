import { buildFragmentShader, inputUniform, paramUniform } from "../rendering/shaders/glslCommon";
import type { FilterDefinition } from "../types";

// Shared inline snippets (not texture fetches — pure ALU reused verbatim across passes).
const DETAIL_FROM_EDGE = "smoothstep(0.04, 0.3, edgeMagRaw)";
const GRADIENT_TANGENT_FROM_ORIGINAL = `
  float otx = uOriginalTexelSize.x;
  float oty = uOriginalTexelSize.y;
  float lL = luminance(srgbToLinear(texture(uOriginal, vUv - vec2(otx, 0.0)).rgb));
  float lR = luminance(srgbToLinear(texture(uOriginal, vUv + vec2(otx, 0.0)).rgb));
  float lT = luminance(srgbToLinear(texture(uOriginal, vUv - vec2(0.0, oty)).rgb));
  float lB = luminance(srgbToLinear(texture(uOriginal, vUv + vec2(0.0, oty)).rgb));
  float gx = lR - lL;
  float gy = lB - lT;
  float gradMag = length(vec2(gx, gy)) + 1.0e-4;
  vec2 tangent = vec2(-gy, gx) / gradMag;`;

/**
 * Pass 1/11 — "structure" (50% res): linearize + mild exposure shaping, then a
 * 4-quadrant Kuwahara whose radius shrinks near Sobel-detected detail (eyes,
 * brows, lips, hairline) and grows in flat regions (skin, background), so
 * consolidation strength follows local structure rather than one fixed
 * radius. Quadrant selection uses a perceptually-weighted linear-RGB variance
 * (cheap); the winning mean is then nudged toward OKLab lightness steps for a
 * gentle painterly quantization without posterizing. Raw Sobel magnitude is
 * carried in alpha for every later pass to reuse without resampling.
 */
const structurePass = buildFragmentShader({
  extraUniforms: `uniform float ${paramUniform("detail")};`,
  body: `
  vec3 origLin = srgbToLinear(texture(uOriginal, vUv).rgb);
  float lum0 = luminance(origLin);
  origLin += (0.05 - origLin * 0.05) * smoothstep(0.5, 0.0, lum0);
  origLin = mix(origLin, vec3(0.94), smoothstep(0.85, 1.05, lum0) * 0.15);

  float edgeMag = sobelEdge(uOriginal, vUv, uOriginalTexelSize);
  float detailStrength = ${paramUniform("detail")} / 100.0;
  float detail = clamp(smoothstep(0.04, 0.3, edgeMag) * mix(0.4, 1.3, detailStrength), 0.0, 1.0);
  float radiusScale = mix(1.6, 0.35, detail);

  vec2 dir[4] = vec2[4](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
  float baseRadius = 3.2 * radiusScale;
  vec3 bestMeanLin = origLin;
  float bestVar = 1.0e6;
  for (int q = 0; q < 4; q++) {
    vec3 sumLin = vec3(0.0);
    vec3 sumSqLin = vec3(0.0);
    float count = 0.0;
    for (int i = 0; i <= 2; i++) {
      for (int j = 0; j <= 2; j++) {
        vec2 offset = dir[q] * vec2(float(i), float(j)) * uOriginalTexelSize * baseRadius;
        vec3 cLin = srgbToLinear(texture(uOriginal, vUv + offset).rgb);
        sumLin += cLin;
        sumSqLin += cLin * cLin;
        count += 1.0;
      }
    }
    vec3 meanLin = sumLin / count;
    vec3 varLin = sumSqLin / count - meanLin * meanLin;
    float v = varLin.r * 0.3 + varLin.g * 0.59 + varLin.b * 0.11;
    if (v < bestVar) {
      bestVar = v;
      bestMeanLin = meanLin;
    }
  }

  vec3 bestOklab = linearRgbToOklab(max(bestMeanLin, 0.0));
  float lQuant = floor(bestOklab.x * 9.0 + 0.5) / 9.0;
  bestOklab.x = mix(bestOklab.x, lQuant, 0.25);
  vec3 structuredLin = max(oklabToLinearRgb(bestOklab), 0.0);

  // Stored back in sRGB (like every other intermediate color buffer in this
  // pipeline) — persisting linear values in 8-bit RGBA8 across many passes
  // visibly banded/hazed the result, since sRGB's gamma curve is exactly what
  // allocates enough 8-bit precision to the shadows/midtones our eyes are
  // sensitive to. Linear space is used transiently within each pass instead.
  fragColor = vec4(linearToSrgb(clamp(structuredLin, 0.0, 1.0)), edgeMag);`,
});

/**
 * Pass 2/11 — "region" (50% res): a cheap skin-likelihood heuristic from
 * linear-RGB chromaticity (explicitly approximate — warm backgrounds can be
 * misread as skin, a known limitation) combined with the structure pass's
 * detail signal into an overall "protection" value, from which a baseline
 * wetness (inverse of protection) is derived. OKLab lightness is cached here
 * too so later passes don't need to reconvert it.
 */
const regionPass = buildFragmentShader({
  body: `
  vec4 structureSample = texture(uSource, vUv);
  vec3 baseLin = srgbToLinear(structureSample.rgb);
  float edgeMagRaw = structureSample.a;

  vec3 oklab = linearRgbToOklab(baseLin);
  float L = oklab.x;

  float sum = baseLin.r + baseLin.g + baseLin.b + 1.0e-4;
  float rNorm = baseLin.r / sum;
  float gNorm = baseLin.g / sum;
  float skinChroma = smoothstep(0.36, 0.7, rNorm) * smoothstep(0.24, 0.42, gNorm) * smoothstep(0.02, 0.25, rNorm - gNorm);
  float skinLum = smoothstep(0.05, 0.18, L) * (1.0 - smoothstep(0.85, 0.98, L));
  float skinMask = clamp(skinChroma * skinLum, 0.0, 1.0);

  float detail01 = ${DETAIL_FROM_EDGE};
  float protection = clamp(detail01 + skinMask * 0.5, 0.0, 1.0);
  float wetnessBase = clamp(1.0 - protection, 0.0, 1.0);

  fragColor = vec4(skinMask, wetnessBase, L, 1.0);`,
});

/**
 * Pass 3/11 — "pigmentWash" (50% res): pigment density from OKLab
 * lightness/chroma (highlights thin, shadows thick but capped, saturated
 * hues like lips keep their pigment), a "low-saturation dark" granulation
 * flag, and a wetness value (baseline reduced in skin regions, per §7.1).
 * Two fbm scales perturb density/wetness/deposition — never color directly.
 */
const pigmentWashPass = buildFragmentShader({
  extraUniforms: `uniform sampler2D ${inputUniform("structure")};\nuniform float ${paramUniform("pigment")};`,
  body: `
  vec4 regionSample = texture(uSource, vUv);
  float skinMask = regionSample.r;
  float wetnessBase = regionSample.g;

  vec4 structureSample = texture(${inputUniform("structure")}, vUv);
  vec3 baseLin = srgbToLinear(structureSample.rgb);
  float edgeMagRaw = structureSample.a;
  float detail01 = ${DETAIL_FROM_EDGE};

  vec3 oklab = linearRgbToOklab(baseLin);
  float L = oklab.x;
  float chroma = length(oklab.yz);

  float density = smoothstep(0.92, 0.15, L);
  density = max(density, chroma * smoothstep(0.06, 0.22, chroma) * 0.6);
  density += detail01 * 0.1;
  density = clamp(density, 0.03, 0.95);

  float lowSatDark = (1.0 - clamp(chroma * 4.0, 0.0, 1.0)) * smoothstep(0.5, 0.05, L);

  float washA = fbm(vUv * 3.2);
  float washB = fbm(vUv * 8.2 + 12.0);
  float washNoise = mix(washA, washB, 0.4);

  float pigmentAmt = ${paramUniform("pigment")} / 100.0;
  float unevenDensity = clamp(density * mix(0.7, 1.3, washNoise), 0.0, 1.0);
  unevenDensity = clamp(mix(density, unevenDensity, 0.5 + pigmentAmt * 0.5), 0.0, 1.0);

  float wetness = wetnessBase * mix(1.0, 0.4, skinMask);
  wetness = clamp(wetness * mix(0.7, 1.3, fbm(vUv * 5.2 + 30.0)), 0.0, 1.0);

  float depositMod = mix(0.65, 1.35, fbm(vUv * 6.6 + 55.0));

  fragColor = vec4(unevenDensity, wetness, depositMod, lowSatDark);`,
});

/**
 * Passes 4-7/11 — "bleedH1/V1/H2/V2" (50% res): two iterations of a
 * direction-aware separable blur (closer to real diffusion than one big
 * kernel, per §6.7's "5〜9タップ...2〜3回反復"). Each axis's radius is scaled
 * by wetness, shrunk near protected regions (detail + skin), and biased by
 * how strongly the local gradient's tangent aligns with that axis — so
 * diffusion runs along edges rather than crossing them. The second iteration
 * uses a reduced multiplier so total spread doesn't compound unboundedly.
 */
function bleedPassSource(axis: "x" | "y", iteration: 1 | 2, colorFromChain: boolean): string {
  // colorFromChain is true whenever this pass's uSource (the previous pass in
  // the array) is already a blurred COLOR (bleedV1/H2/V2, each chained off the
  // prior bleed pass) rather than the pigmentWash MAP (only true for bleedH1,
  // which chains directly off pigmentWash). Wetness must come from wherever
  // the map actually is: uSource itself for H1, the pigmentWash extraInput otherwise.
  const iterFactor = iteration === 1 ? 1.0 : 0.6;
  const colorSampler = colorFromChain ? "uSource" : inputUniform("structure");
  const wetnessExpr = colorFromChain ? `texture(${inputUniform("pigmentWash")}, vUv).g` : "texture(uSource, vUv).g";
  const extra = colorFromChain
    ? `uniform sampler2D ${inputUniform("pigmentWash")};\nuniform sampler2D ${inputUniform("structure")};\nuniform sampler2D ${inputUniform("region")};\nuniform float ${paramUniform("bleed")};`
    : `uniform sampler2D ${inputUniform("structure")};\nuniform sampler2D ${inputUniform("region")};\nuniform float ${paramUniform("bleed")};`;
  const axisOffset = axis === "x" ? "vec2(float(i) * uTexelSize.x * radius, 0.0)" : "vec2(0.0, float(i) * uTexelSize.y * radius)";
  const allowExpr = axis === "x" ? "abs(tangent.x)" : "abs(tangent.y)";

  return buildFragmentShader({
    extraUniforms: extra,
    body: `
  float wetness = ${wetnessExpr};
  float edgeMagRaw = texture(${inputUniform("structure")}, vUv).a;
  float detail01 = ${DETAIL_FROM_EDGE};
  float skinMask = texture(${inputUniform("region")}, vUv).r;
  float protect = clamp(detail01 + skinMask * 0.5, 0.0, 1.0);
${GRADIENT_TANGENT_FROM_ORIGINAL}

  float bleedAmt = ${paramUniform("bleed")} / 100.0;
  float allow = mix(1.0, ${allowExpr}, smoothstep(0.05, 0.3, gradMag));
  float radius = (0.4 + bleedAmt * 1.6) * ${iterFactor.toFixed(2)} * mix(0.5, 1.4, wetness) * mix(1.0, 0.2, protect) * allow;

  vec3 sum = vec3(0.0);
  float weightSum = 0.0;
  for (int i = -3; i <= 3; i++) {
    float w = exp(-float(i * i) / 6.0);
    vec2 uv = vUv + ${axisOffset};
    sum += texture(${colorSampler}, uv).rgb * w;
    weightSum += w;
  }
  fragColor = vec4(sum / weightSum, 1.0);`,
  });
}

// Only bleedH1's predecessor (pigmentWash) is a data map rather than blurred
// color, so it alone passes colorFromChain=false; every later iteration
// chains progressively off the previous pass's already-blurred color.
const bleedH1Pass = bleedPassSource("x", 1, false);
const bleedV1Pass = bleedPassSource("y", 1, true);
const bleedH2Pass = bleedPassSource("x", 2, true);
const bleedV2Pass = bleedPassSource("y", 2, true);

/**
 * Pass 8/11 — "edgeDeposit" (50% res): two independently-triggered
 * accumulation effects. Backrun deposits a darker, more chroma-saturated
 * (OKLab) version of the local color near structural (Sobel) boundaries,
 * broken by noise. Cauliflower blooms trigger separately from local wetness
 * *differences* (sampled from the pigment/wash map) with a coarser, wavier
 * noise mask — both dampened in skin regions. Granulation mixes dark,
 * low-saturation pixels partway toward a small blue-grey/umber palette so
 * shadows never read as flat black.
 */
const edgeDepositPass = buildFragmentShader({
  extraUniforms: `uniform sampler2D ${inputUniform("pigmentWash")};\nuniform sampler2D ${inputUniform("structure")};\nuniform sampler2D ${inputUniform("region")};\nuniform float ${paramUniform("pigment")};`,
  body: `
  vec3 color = texture(uSource, vUv).rgb;
  vec4 mapSample = texture(${inputUniform("pigmentWash")}, vUv);
  float density = mapSample.r;
  float lowSatDark = mapSample.a;
  float edgeMagRaw = texture(${inputUniform("structure")}, vUv).a;
  float skinMask = texture(${inputUniform("region")}, vUv).r;
  float pigmentAmt = ${paramUniform("pigment")} / 100.0;

  float breakNoise = fbm(vUv * 15.0 + 70.0);
  float edgeChance = smoothstep(0.32, 0.7, breakNoise);
  float backrunMask = smoothstep(0.05, 0.22, edgeMagRaw) * edgeChance * mix(0.35, 1.0, density) * pigmentAmt * mix(1.0, 0.4, skinMask);

  vec3 oklab = linearRgbToOklab(srgbToLinear(color));
  vec3 pooledOklab = vec3(oklab.x * 0.62, oklab.yz * 1.35);
  vec3 pooled = linearToSrgb(max(oklabToLinearRgb(pooledOklab), 0.0));
  color = mix(color, pooled, backrunMask);

  float wtx = uTexelSize.x;
  float wty = uTexelSize.y;
  float wL = texture(${inputUniform("pigmentWash")}, vUv - vec2(wtx, 0.0)).g;
  float wR = texture(${inputUniform("pigmentWash")}, vUv + vec2(wtx, 0.0)).g;
  float wT = texture(${inputUniform("pigmentWash")}, vUv - vec2(0.0, wty)).g;
  float wB = texture(${inputUniform("pigmentWash")}, vUv + vec2(0.0, wty)).g;
  float wetGrad = abs(wR - wL) + abs(wB - wT);
  float cauliNoise = fbm(vUv * 6.0 + 200.0);
  float cauliMask = smoothstep(0.15, 0.4, wetGrad) * smoothstep(0.3, 0.7, cauliNoise) * pigmentAmt * mix(1.0, 0.3, skinMask);
  vec3 cauliOklab = vec3(oklab.x * 0.75, oklab.yz * 1.2);
  vec3 cauli = linearToSrgb(max(oklabToLinearRgb(cauliOklab), 0.0));
  color = mix(color, cauli, cauliMask * 0.7);

  vec3 granOklabA = vec3(0.24, -0.02, -0.08);
  vec3 granOklabB = vec3(0.22, 0.03, 0.06);
  vec3 granPick = mix(granOklabA, granOklabB, fbm(vUv * 9.0 + 3.0));
  vec3 granColor = linearToSrgb(max(oklabToLinearRgb(granPick), 0.0));
  float granAmt = lowSatDark * pigmentAmt * 0.45;
  color = mix(color, mix(color, granColor, 0.5), granAmt);

  fragColor = vec4(max(color, 0.0), 1.0);`,
});

/**
 * Pass 9/11 — "detailRestore" (100% res): a Difference-of-Gaussians stand-in
 * computed for free from data we already have — |original luminance −
 * structure baseColor luminance| — highlights exactly what the structure
 * pass's consolidation removed. Gated by the Sobel detail mask (so it fires
 * on real edges, not generic photo grain) and broken by noise, it restores a
 * faint line in the original's own darkened pigment color (never black).
 */
const detailRestorePass = buildFragmentShader({
  extraUniforms: `uniform sampler2D ${inputUniform("structure")};\nuniform sampler2D ${inputUniform("region")};\nuniform float ${paramUniform("detail")};`,
  body: `
  vec3 color = texture(uSource, vUv).rgb;
  vec3 originalLin = srgbToLinear(texture(uOriginal, vUv).rgb);
  vec4 structureSample = texture(${inputUniform("structure")}, vUv);
  vec3 baseLin = srgbToLinear(structureSample.rgb);
  float edgeMagRaw = structureSample.a;
  float skinMask = texture(${inputUniform("region")}, vUv).r;

  float dogSignal = abs(luminance(originalLin) - luminance(baseLin));
  float detail01 = ${DETAIL_FROM_EDGE};

  float breakNoise = fbm(vUv * 24.0 + 90.0);
  float breakMask = smoothstep(0.28, 0.66, breakNoise);

  float detailAmt = ${paramUniform("detail")} / 100.0;
  float lineAlpha = detail01 * smoothstep(0.03, 0.18, dogSignal) * breakMask * detailAmt;
  lineAlpha *= mix(0.8, 1.15, skinMask);

  vec3 lineOklab = linearRgbToOklab(originalLin);
  lineOklab.x *= 0.55;
  vec3 lineColor = linearToSrgb(max(oklabToLinearRgb(lineOklab), 0.0));
  color = mix(color, lineColor, clamp(lineAlpha, 0.0, 1.0) * 0.7);

  fragColor = vec4(max(color, 0.0), 1.0);`,
});

/**
 * Pass 10/11 — "paperInteract" (100% res): a procedural paper *height* map
 * (low/mid bumps + a stretched-noise directional fiber term + fine grain,
 * frequency tied to output resolution so it doesn't over-scale). Peaks
 * expose paper white and — in dry (low-wetness) areas — drop pigment
 * entirely (drybrush); valleys accumulate a touch more pigment. Never a flat
 * overlay multiply.
 */
const paperInteractPass = buildFragmentShader({
  extraUniforms: `uniform sampler2D ${inputUniform("pigmentWash")};\nuniform sampler2D ${inputUniform("region")};\nuniform float ${paramUniform("paperGrain")};`,
  body: `
  vec3 color = texture(uSource, vUv).rgb;
  float density = texture(${inputUniform("pigmentWash")}, vUv).r;
  float wetness = texture(${inputUniform("pigmentWash")}, vUv).g;
  float skinMask = texture(${inputUniform("region")}, vUv).r;

  float grain = ${paramUniform("paperGrain")} / 100.0;
  vec2 fiberUv = vUv * uResolution * 0.022;
  float fiber = fbm(vec2(fiberUv.x * 0.3, fiberUv.y * 1.6) + 8.0);
  float bumps = fbm(vUv * uResolution * 0.026 + 20.0);
  float fineGrain = fbm(vUv * uResolution * 0.07 + 44.0);
  float height = clamp(bumps * 0.5 + fiber * 0.3 + fineGrain * 0.2, 0.0, 1.0);

  float dryness = clamp(1.0 - wetness, 0.0, 1.0) * mix(1.0, 0.5, skinMask);
  float peak = smoothstep(0.68, 0.9, height);
  float valley = smoothstep(0.4, 0.1, height);

  // Kept in sRGB — color at this point in the pipeline is sRGB, not linear.
  vec3 paperColorSrgb = vec3(0.975, 0.968, 0.94);
  float paperShow = (1.0 - density) * peak * grain;
  color = mix(color, paperColorSrgb, paperShow * 0.35);
  float dryBrushDrop = peak * dryness * grain * 0.35;
  color = mix(color, paperColorSrgb, dryBrushDrop);

  float valleyAccum = valley * grain * density * 0.18;
  color *= mix(1.0, 0.94, valleyAccum);

  fragColor = vec4(max(color, 0.0), 1.0);`,
});

/**
 * Pass 11/11 — "finalComposite" (100% res): linear → sRGB with a tiny
 * hash-based dither to guard against 8-bit banding after so much
 * multi-pass math, then mixes with the untouched original via the common
 * "効果の強さ" (= 水彩の強さ from the spec — the same mix ratio concept).
 */
const finalCompositePass = buildFragmentShader({
  body: `
  // color is already sRGB here (every intermediate buffer in this pipeline
  // persists sRGB; linear space is only ever used transiently within a pass).
  vec3 srgbColor = clamp(texture(uSource, vUv).rgb, 0.0, 1.0);
  vec3 originalSrgb = texture(uOriginal, vUv).rgb;

  srgbColor += orderedDither(gl_FragCoord.xy);

  fragColor = vec4(mix(originalSrgb, clamp(srgbColor, 0.0, 1.0), uStrength), 1.0);`,
});

export const watercolorFilter: FilterDefinition = {
  id: "watercolor",
  name: "水彩",
  description:
    "線形色空間とOKLabで色面を整理し、顔料濃度・水分量・異方性のにじみ・バックラン/カリフラワー・紙の高さマップで透明水彩のポートレートらしさを再現します。",
  parameters: [
    { id: "bleed", label: "にじみ", min: 0, max: 100, step: 1, defaultValue: 45 },
    { id: "pigment", label: "顔料", min: 0, max: 100, step: 1, defaultValue: 55 },
    { id: "paperGrain", label: "紙目", min: 0, max: 100, step: 1, defaultValue: 40 },
    { id: "detail", label: "ディテール", min: 0, max: 100, step: 1, defaultValue: 55 },
  ],
  passes: [
    { id: "structure", fragmentSource: structurePass, outputScale: 0.5 },
    { id: "region", fragmentSource: regionPass, outputScale: 0.5 },
    { id: "pigmentWash", fragmentSource: pigmentWashPass, outputScale: 0.5, extraInputs: ["structure"] },
    { id: "bleedH1", fragmentSource: bleedH1Pass, outputScale: 0.5, extraInputs: ["structure", "region"] },
    { id: "bleedV1", fragmentSource: bleedV1Pass, outputScale: 0.5, extraInputs: ["pigmentWash", "structure", "region"] },
    { id: "bleedH2", fragmentSource: bleedH2Pass, outputScale: 0.5, extraInputs: ["pigmentWash", "structure", "region"] },
    { id: "bleedV2", fragmentSource: bleedV2Pass, outputScale: 0.5, extraInputs: ["pigmentWash", "structure", "region"] },
    { id: "edgeDeposit", fragmentSource: edgeDepositPass, outputScale: 0.5, extraInputs: ["pigmentWash", "structure", "region"] },
    { id: "detailRestore", fragmentSource: detailRestorePass, outputScale: 1, extraInputs: ["structure", "region"] },
    { id: "paperInteract", fragmentSource: paperInteractPass, outputScale: 1, extraInputs: ["pigmentWash", "region"] },
    { id: "finalComposite", fragmentSource: finalCompositePass, outputScale: 1 },
  ],
  presets: [
    {
      id: "light-wash",
      label: "淡彩ポートレート",
      values: { bleed: 25, pigment: 32, paperGrain: 32, detail: 72 },
    },
    {
      id: "bleeding-watercolor",
      label: "にじみ水彩",
      values: { bleed: 72, pigment: 58, paperGrain: 45, detail: 48 },
    },
    {
      id: "rich-illustration",
      label: "濃彩イラスト",
      values: { bleed: 38, pigment: 88, paperGrain: 55, detail: 78 },
    },
  ],
};
