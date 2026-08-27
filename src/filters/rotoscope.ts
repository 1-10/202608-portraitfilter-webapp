import { buildFragmentShader, inputUniform, paramUniform } from "../rendering/shaders/glslCommon";
import type { FilterDefinition } from "../types";

/**
 * Rotoscope — photoreal geometry, cel-painted surface.
 *
 * The reference is the digital rotoscoping of *A Scanner Darkly*: the drawing sits
 * exactly on top of the photograph, so composition, proportion and identity are
 * untouched, while the SURFACE becomes flat painted regions with hand-drawn
 * outlines. That is the "reality line intact, rawness removed" brief.
 *
 * It exists because the previous cel-anime filter failed at this on real photos,
 * and it is built to defeat each specific cause:
 *
 *  - Its 9-tap radius-1.5 blur could not flatten real skin at all, so quantization
 *    posterized skin NOISE into blotches. Here, flattening is a genuine anisotropic
 *    Kuwahara run twice, which smooths ALONG local structure and refuses to average
 *    across it — flat planes with hard borders, not a smear.
 *  - It quantized per RGB channel, so channels crossed their thresholds at
 *    different luminances and skin picked up hue fringes. Here, quantization is in
 *    OKLab on lightness and chroma only; hue is never touched, so fringing is
 *    structurally impossible.
 *  - Every stylized filter took its edges from the RAW photo, so pores, stray hair
 *    and JPEG noise all became "lines". Here, lines come from the FLATTENED,
 *    quantized buffer, which has no pores left to find.
 *  - Its outlines were a uniform black multiply. Here they are drawn in a darkened
 *    version of the adjacent region's own color and vary in width and opacity.
 *
 * All radii are expressed as fractions of the image long edge (and scaled by
 * uFaceScale where they should track the subject), never in raw texels, so the
 * result is the same at thumbnail, preview and export sizes.
 */

/**
 * Radii here are expressed relative to the FACE, not the frame. A portrait where
 * the head fills the frame and one where it occupies a fifth of a 9:16 shot must
 * get the same-looking flattening and the same-looking line weight, and only a
 * face-relative unit gives that.
 *
 * uFaceScale is the interocular distance as a fraction of image WIDTH, so it is
 * converted through uResolution.x specifically — using the long edge would be
 * wrong by the aspect ratio on any non-square image.
 */
const LONG_EDGE = `
  vec2 texel = 1.0 / uResolution;
  float faceWidthPx = max(uFaceScale * uResolution.x * 2.5, 24.0);`;

/**
 * Pass 1 — denoise. A mild edge-preserving smooth before anything measures
 * gradients, so sensor noise and JPEG blocking never enter the structure tensor.
 * Alpha carries OKLab lightness for later passes.
 */
const denoisePass = buildFragmentShader({
  body: `
${LONG_EDGE}
  vec3 c = bilateral8(uOriginal, vUv, uOriginalTexelSize, 1.2, 22.0);
  float L = linearRgbToOklab(srgbToLinear(c)).x;
  fragColor = vec4(c, L);`,
});

/**
 * Pass 2 — structure tensor. Encodes how intensity varies locally, which is what
 * lets later passes know the direction of the "brush stroke" at every pixel.
 * Packed into 8 bits, which is lossy, but pass 3 immediately blurs it and the
 * resulting orientation error is far below a degree at any useful anisotropy.
 */
const tensorPass = buildFragmentShader({
  extraUniforms: `uniform sampler2D ${inputUniform("denoise")};`,
  body: `
${LONG_EDGE}
  vec3 gx = vec3(0.0);
  vec3 gy = vec3(0.0);
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      vec3 s = texture(${inputUniform("denoise")}, vUv + vec2(float(i), float(j)) * texel).rgb;
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

/**
 * Pass 3 — blur the tensor. Averaging in tensor space (rather than averaging
 * directions, which would cancel out) is what turns a noisy per-pixel gradient
 * into a smooth, coherent flow field that follows hair and jawlines.
 */
const tensorBlurPass = buildFragmentShader({
  body: `
${LONG_EDGE}
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
const flowPass = buildFragmentShader({
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

/**
 * Passes 5-6 — anisotropic Kuwahara, run twice.
 *
 * The window is an ellipse stretched along the local tangent and squashed across
 * it. It is divided into 8 overlapping sectors; each sector's mean and variance are
 * accumulated, and the output favours the LOWEST-variance sectors. Averaging only
 * within the most uniform sector is what produces a flat region that stops dead at
 * a boundary instead of blurring through it.
 *
 * Two moderate iterations beat one large one: Kuwahara converges toward piecewise
 * constant regions under iteration, so this flattens harder AND costs less than a
 * single pass of twice the radius.
 */
function kuwaharaPass(sourceInput: string): string {
  // The first iteration reads the denoised buffer via an extra input; the second
  // chains off its predecessor through uSource, which is already declared.
  const extraSampler = sourceInput === "uSource" ? "" : `uniform sampler2D ${sourceInput};\n`;
  return buildFragmentShader({
    extraUniforms: `${extraSampler}uniform sampler2D ${inputUniform("flow")};\nuniform float ${paramUniform("flatten")};`,
    body: `
${LONG_EDGE}
  vec4 f = texture(${inputUniform("flow")}, vUv);
  vec2 dir = f.rg * 2.0 - 1.0;
  float aniso = f.b;

  // Flatten skin hardest; hold back sharply on eyes, brows and lips, because
  // flattening the catchlights and iris out of the eyes is the fastest way to
  // destroy a likeness.
  vec3 srcLin = srgbToLinear(texture(uOriginal, vUv).rgb);
  float skin = faceSkinWeight(srcLin, vUv);
  float feature = faceFeatureWeight(vUv);

  // A few percent of face width. Larger than this and the kernel spans real facial
  // features, and because the sample grid is fixed at (2*MAXR+1)^2 the taps also
  // spread far enough apart to alias into streaks rather than flatten.
  float radiusPx = faceWidthPx * (0.012 + ${paramUniform("flatten")} / 100.0 * 0.030)
                 * mix(1.0, 1.35, skin) * mix(1.0, 0.30, feature);
  radiusPx = clamp(radiusPx, 1.0, 14.0);

  float a = radiusPx * (1.0 + aniso);
  float b = radiusPx / (1.0 + aniso);
  mat2 rot = mat2(dir.x, -dir.y, dir.y, dir.x);
  mat2 shrink = mat2(1.0 / max(a, 0.5), 0.0, 0.0, 1.0 / max(b, 0.5));
  mat2 toUnit = shrink * rot;

  vec3 mean[8];
  vec3 sq[8];
  float wgt[8];
  for (int k = 0; k < 8; k++) { mean[k] = vec3(0.0); sq[k] = vec3(0.0); wgt[k] = 0.0; }

  const int MAXR = 4;
  for (int j = -MAXR; j <= MAXR; j++) {
    for (int i = -MAXR; i <= MAXR; i++) {
      vec2 o = vec2(float(i), float(j)) * (radiusPx / float(MAXR));
      vec2 e = toUnit * o;
      float q = dot(e, e);
      if (q > 1.0) continue;
      vec3 c = texture(${sourceInput}, vUv + o * texel).rgb;
      float radial = 1.0 - q;

      // A sample's angle places it between exactly two adjacent sectors, and the
      // triangular sector weights are a partition of unity — so it contributes to
      // those two and to no others. Resolving them directly instead of testing all
      // eight is mathematically identical and cuts this inner loop from 8 to 2,
      // which matters because it runs once per sample per pixel.
      float angNorm = (atan(e.y, e.x) + 3.14159265) * (8.0 / 6.28318531);
      float base = floor(angNorm);
      float frac = angNorm - base;
      int k0 = int(mod(base, 8.0));
      int k1 = int(mod(base + 1.0, 8.0));

      float w0 = (1.0 - frac) * radial;
      mean[k0] += c * w0; sq[k0] += c * c * w0; wgt[k0] += w0;
      float w1 = frac * radial;
      mean[k1] += c * w1; sq[k1] += c * c * w1; wgt[k1] += w1;
    }
  }

  vec3 acc = vec3(0.0);
  float accW = 0.0;
  for (int k = 0; k < 8; k++) {
    if (wgt[k] < 1.0e-4) continue;
    vec3 m = mean[k] / wgt[k];
    vec3 variance = abs(sq[k] / wgt[k] - m * m);
    float sd = sqrt(dot(variance, vec3(0.299, 0.587, 0.114)));
    float w = 1.0 / (1.0 + pow(sd * 40.0, 4.0));
    acc += m * w;
    accW += w;
  }
  vec3 outColor = accW > 1.0e-4 ? acc / accW : texture(${sourceInput}, vUv).rgb;
  fragColor = vec4(outColor, 1.0);`,
  });
}

/**
 * Pass 7 — cel quantization in OKLab.
 *
 * Lightness is stepped through a soft staircase whose transition is one band wide,
 * so bands read as painted planes rather than as aliased contours. Chroma is
 * stepped separately and pushed up, because cel paint is more saturated than
 * photography. Hue is deliberately untouched.
 */
const quantizePass = buildFragmentShader({
  extraUniforms: `uniform float ${paramUniform("levels")};\nuniform float ${paramUniform("saturation")};`,
  body: `
  vec3 flat3 = texture(uSource, vUv).rgb;
  vec3 lab = linearRgbToOklab(srgbToLinear(flat3));
  float C = length(lab.yz);
  float hueA = lab.y;
  float hueB = lab.z;

  float levels = max(${paramUniform("levels")}, 2.0);
  float step0 = floor(lab.x * levels + 0.5) / levels;
  // Soft staircase. t is the position within the band, -1..1; the tanh is
  // normalized by tanh(hardness) so the curve still reaches the band edge exactly
  // and stays continuous across band boundaries. Without that normalization the
  // curve saturates early and the mapping degenerates back to near-identity —
  // i.e. it stops quantizing at all, which is what it was doing before.
  float t = clamp((lab.x - step0) * levels * 2.0, -1.0, 1.0);
  const float hardness = 2.2;
  float L2 = step0 + (0.5 / levels) * (tanh(t * hardness) / tanh(hardness));

  // Chroma is quantized in FIXED perceptual steps, not in "levels" spanning 0..1.
  // Skin sits at C ~= 0.03-0.08 in OKLab, so a step of 1/6 would snap any faintly
  // ruddy pixel straight to full saturation — which showed up as orange blotches
  // on the cheeks. A ~0.022 step quantizes visibly without inventing color.
  const float cStep = 0.022;
  float Cq = floor(C / cStep + 0.5) * cStep;
  float C2 = mix(C, Cq, 0.6) * (0.75 + ${paramUniform("saturation")} / 100.0 * 0.7);
  // Hard ceiling on how much a single pixel's chroma may grow, so a near-neutral
  // pixel can never be amplified into a saturated blob by rounding alone.
  C2 = min(C2, C * 1.5 + 0.012);

  // Eyes and lips stay continuous-tone: they carry the likeness.
  float feature = faceFeatureWeight(vUv);
  L2 = mix(L2, lab.x, feature);
  C2 = mix(C2, C, feature);

  float scale = C > 1.0e-4 ? clamp(C2 / C, 0.0, 1.8) : 0.0;
  vec3 outLin = oklabToLinearRgb(vec3(L2, hueA * scale, hueB * scale));
  fragColor = vec4(linearToSrgb(clamp(outLin, 0.0, 1.0)), L2);`,
});

/**
 * Pass 8 — the drawn line.
 *
 * A difference-of-Gaussians is taken ACROSS the flow (along the edge normal) on the
 * quantized buffer, then smeared ALONG the flow. That second step is what turns a
 * per-pixel edge response into a continuous stroke, and it is why this produces
 * strokes instead of the salt-and-pepper speckle a plain threshold gives.
 *
 * Width and opacity vary from three sources — local ambiguity, local darkness, and
 * a low-frequency noise wobble — so the outline reads as drawn rather than detected.
 * The ink color is sampled from the darker side of the edge, so hair outlines come
 * out deep brown and skin outlines terracotta; a single global black ink is the
 * thing that makes filters like this look like a cartoon.
 */
/** Line width, in pixels. Identical in the DoG and smear passes, so they agree. */
const LINE_WIDTH = `
  float wobble = fbm(stableNoiseUv(vUv, uResolution, 14.0)) - 0.5;
  float widthPx = max(0.7,
      faceWidthPx * (0.004 + ${paramUniform("lineWidth")} / 100.0 * 0.016)
    * mix(1.35, 1.0, aniso)
    * (1.0 + wobble * 0.5));`;

const dogPass = buildFragmentShader({
  extraUniforms: `uniform sampler2D ${inputUniform("flow")};\nuniform float ${paramUniform("lineWidth")};`,
  body: `
${LONG_EDGE}
  vec4 f = texture(${inputUniform("flow")}, vUv);
  vec2 tangent = f.rg * 2.0 - 1.0;
  vec2 normalDir = vec2(-tangent.y, tangent.x);
  float aniso = f.b;
${LINE_WIDTH}

  float sigmaCenter = widthPx * 0.5;
  float sigmaSurround = widthPx * 1.6;

  // The sampling extent must cover the SURROUND, not the line width. Sampling only
  // +-widthPx truncated the surround Gaussian to its own peak, so the two averages
  // came out nearly equal, the difference never went negative, and no line was ever
  // produced anywhere in the image.
  float extent = sigmaSurround * 2.5;
  float centerSum = 0.0, centerW = 0.0, surroundSum = 0.0, surroundW = 0.0;
  const int TAPS = 6;
  for (int i = -TAPS; i <= TAPS; i++) {
    float x = float(i) * extent / float(TAPS);
    float L = texture(uSource, vUv + normalDir * x * texel).a;
    float wc = gaussW(x, sigmaCenter);
    float ws = gaussW(x, sigmaSurround);
    centerSum += L * wc; centerW += wc;
    surroundSum += L * ws; surroundW += ws;
  }
  float dog = centerSum / centerW - 0.99 * (surroundSum / surroundW);
  // Encoded around 0.5 because the render target is unsigned 8-bit.
  fragColor = vec4(dog * 8.0 + 0.5, 0.0, 0.0, 1.0);`,
});

const linePass = buildFragmentShader({
  extraUniforms: `uniform sampler2D ${inputUniform("flow")};\nuniform sampler2D ${inputUniform("quantize")};\nuniform float ${paramUniform("lineWidth")};\nuniform float ${paramUniform("lineOpacity")};`,
  body: `
${LONG_EDGE}
  vec4 f = texture(${inputUniform("flow")}, vUv);
  vec2 tangent = f.rg * 2.0 - 1.0;
  vec2 normalDir = vec2(-tangent.y, tangent.x);
  float aniso = f.b;
${LINE_WIDTH}

  // Smear the DoG response ALONG the flow. This is what turns a per-pixel edge
  // response into a continuous stroke, and it is why this yields strokes rather
  // than the salt-and-pepper speckle a plain threshold produces on real skin.
  float acc = (texture(uSource, vUv).r - 0.5) / 8.0;
  float accW = 1.0;
  vec2 p = vUv, dcur = tangent;
  for (int s = 1; s <= 4; s++) {
    vec2 dn = texture(${inputUniform("flow")}, p).rg * 2.0 - 1.0;
    if (dot(dn, dcur) < 0.0) dn = -dn;
    dcur = dn;
    p += dcur * texel * widthPx * 0.8;
    float w = gaussW(float(s), 3.0);
    acc += ((texture(uSource, p).r - 0.5) / 8.0) * w;
    accW += w;
  }
  vec2 pb = vUv; vec2 db = -tangent;
  for (int s = 1; s <= 4; s++) {
    vec2 dn = texture(${inputUniform("flow")}, pb).rg * 2.0 - 1.0;
    if (dot(dn, db) < 0.0) dn = -dn;
    db = dn;
    pb += db * texel * widthPx * 0.8;
    float w = gaussW(float(s), 3.0);
    acc += ((texture(uSource, pb).r - 0.5) / 8.0) * w;
    accW += w;
  }
  float lineField = acc / accW;

  // Soft threshold: anti-aliased, and darker areas cross it sooner, so lines
  // thicken in shadow the way a pen bearing down would.
  float localL = texture(${inputUniform("quantize")}, vUv).a;
  float eps = mix(-0.004, 0.004, localL);
  float line = lineField >= eps ? 0.0 : clamp(-tanh(24.0 * (lineField - eps)), 0.0, 1.0);
  line *= ${paramUniform("lineOpacity")} / 100.0;
  line *= 0.55 + 0.45 * smoothstep(0.04, 0.30, f.a);

  vec3 cA = texture(${inputUniform("quantize")}, vUv + normalDir * widthPx * 1.6 * texel).rgb;
  vec3 cB = texture(${inputUniform("quantize")}, vUv - normalDir * widthPx * 1.6 * texel).rgb;
  vec3 darker = luminance(cA) < luminance(cB) ? cA : cB;
  vec3 inkLab = linearRgbToOklab(srgbToLinear(darker));
  inkLab.x *= 0.38;
  inkLab.yz *= 1.25;
  vec3 ink = linearToSrgb(clamp(oklabToLinearRgb(inkLab), 0.0, 1.0));

  fragColor = vec4(line, ink);`,
});

/**
 * Pass 9 — composite. Lays the ink over the cel colors, then re-injects a little
 * original micro-detail inside the eyes and lips, which is what keeps the subject
 * recognisably themselves rather than a generic illustration.
 */
const compositePass = buildFragmentShader({
  extraUniforms: `uniform sampler2D ${inputUniform("quantize")};`,
  body: `
  vec3 cel = texture(${inputUniform("quantize")}, vUv).rgb;
  vec4 ln = texture(uSource, vUv);
  vec3 col = mix(cel, ln.gba, clamp(ln.r, 0.0, 1.0));

  vec3 original = texture(uOriginal, vUv).rgb;
  float feature = faceFeatureWeight(vUv);
  col = mix(col, original, feature * 0.28 * (1.0 - ln.r));

  fragColor = vec4(mix(original, col, uStrength), 1.0);`,
});

export const rotoscopeFilter: FilterDefinition = {
  id: "rotoscope",
  name: "ロトスコープ",
  description:
    "写真の構図と本人らしさを保ったまま、表面だけを平坦な色面と手描き風の輪郭に置き換えます。『スキャナー・ダークリー』的なデジタル・ロトスコープ表現です。",
  parameters: [
    { id: "flatten", label: "色面の平坦さ", min: 0, max: 100, step: 1, defaultValue: 55 },
    { id: "levels", label: "色の段数", min: 3, max: 12, step: 1, defaultValue: 7 },
    { id: "lineWidth", label: "線の太さ", min: 0, max: 100, step: 1, defaultValue: 45 },
    { id: "lineOpacity", label: "線の濃さ", min: 0, max: 100, step: 1, defaultValue: 75 },
    { id: "saturation", label: "彩度", min: 0, max: 100, step: 1, defaultValue: 55 },
  ],
  passes: [
    { id: "denoise", fragmentSource: denoisePass, maxOutputLongEdge: 1600 },
    { id: "tensor", fragmentSource: tensorPass, outputScale: 0.5, maxOutputLongEdge: 800, extraInputs: ["denoise"] },
    { id: "tensorBlur", fragmentSource: tensorBlurPass, outputScale: 0.5, maxOutputLongEdge: 800 },
    { id: "flow", fragmentSource: flowPass, outputScale: 0.5, maxOutputLongEdge: 800 },
    {
      id: "kuwahara1",
      fragmentSource: kuwaharaPass(inputUniform("denoise")),
      maxOutputLongEdge: 1600,
      extraInputs: ["denoise", "flow"],
    },
    { id: "kuwahara2", fragmentSource: kuwaharaPass("uSource"), maxOutputLongEdge: 1600, extraInputs: ["flow"] },
    { id: "quantize", fragmentSource: quantizePass, maxOutputLongEdge: 1600 },
    // The line work is capped too. Left uncapped these two ran at the full 8.3Mpx
    // of a 4K export with ~27 texture reads each and dominated export time. Capping
    // them means the whole look is computed at a bounded resolution and the final
    // composite upscales it — which is also what keeps a large export looking like
    // the preview the user actually approved, rather than a finer re-derivation.
    { id: "dog", fragmentSource: dogPass, maxOutputLongEdge: 1600, extraInputs: ["flow"] },
    { id: "line", fragmentSource: linePass, maxOutputLongEdge: 1600, extraInputs: ["flow", "quantize"] },
    { id: "composite", fragmentSource: compositePass, extraInputs: ["quantize"] },
  ],
  presets: [
    {
      id: "photoreal",
      label: "写実ロトスコープ",
      values: { flatten: 45, levels: 9, lineWidth: 35, lineOpacity: 60, saturation: 50 },
    },
    {
      id: "anime",
      label: "アニメ調",
      values: { flatten: 75, levels: 5, lineWidth: 65, lineOpacity: 90, saturation: 70 },
    },
    {
      id: "graphic-novel",
      label: "グラフィックノベル",
      values: { flatten: 60, levels: 4, lineWidth: 80, lineOpacity: 100, saturation: 30 },
    },
  ],
};
