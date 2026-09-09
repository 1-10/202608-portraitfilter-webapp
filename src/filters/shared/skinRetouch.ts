import { buildFragmentShader, inputUniform } from "../../rendering/shaders/glslCommon";

/**
 * The frequency-separation passes shared by every skin-retouch filter.
 *
 * The failure mode of every naive "beauty" filter is plastic skin, and it has one
 * cause: they split the image into low and high frequency and then attenuate the
 * HIGH band — which is exactly where pores and fine hair live.
 *
 * These passes produce a THREE-band split instead:
 *
 *   Llow    = wide blur      -> shading and 3-D form. Carries the likeness.
 *   Lmid    = narrow blur    -> Lmid - Llow is blotches, oil sheen, uneven tone.
 *   L       = the source     -> L - Lmid is pores and fine hair: real texture.
 *
 * Attenuating only the MIDDLE band removes what people actually mean by "blotchy"
 * while leaving both the modelling that makes a face look three-dimensional and the
 * microtexture that makes it read as skin rather than vinyl. How hard each band is
 * attenuated is the composite's business, not these passes'.
 *
 * The second half of the effect is chroma: blotchy redness is overwhelmingly a
 * CHROMA variance, not a luminance one, so a/b get their own much wider blur. The
 * eye cannot resolve chroma detail at that scale, so smoothing it removes redness
 * and unevenness at essentially no cost in apparent sharpness.
 *
 * Both blurs are separable, so each is two passes: the first reads the filter's
 * source image, the second reads the first's packed output.
 */

/**
 * Choosing the two radii, both fractions of FACE WIDTH.
 *
 * The pair defines what counts as a "blemish": everything between them lands in
 * the attenuated middle band, everything wider than the low radius is untouchable
 * shading, everything finer than the mid radius is texture.
 *
 * That makes the LOW radius the dangerous one. Set it wide (0.06) and the band
 * swallows the shadow beside the nose, the alar crease and the modelling of the
 * chin — all of which are 0.05-0.15 of face width — so a strong setting flattens
 * the nose into a blob with two dark dots where the nostrils were. A retouch that
 * only evens the skin can afford 0.06; one that pushes hard cannot, and has to
 * take blotches out of CHROMA instead, where removing them costs no form.
 *
 * Each filter therefore declares its own pair rather than sharing one here.
 */

/**
 * Where a blur's FIRST axis reads its image from.
 *
 * - `"original"` — the filter's untouched input, already declared in every shader.
 * - `{ pass: id }` — an earlier pass's output, for filters that retouch something
 *   other than the original (the makeup filter blurs the deformed frame). The
 *   sampler uniform is declared by the builder, and the caller must also list `id`
 *   in that pass's `extraInputs` so the renderer actually binds it.
 * - `"chain"` — the previous pass in the chain, which is how each blur's second
 *   axis reads the first axis's packed result.
 */
export type SkinBlurSource = "original" | "chain" | { pass: string };

type ResolvedSource = { sampler: string; declaration: string; fromPackedPair: boolean };

function resolveSource(source: SkinBlurSource): ResolvedSource {
  if (source === "original") return { sampler: "uOriginal", declaration: "", fromPackedPair: false };
  if (source === "chain") return { sampler: "uSource", declaration: "", fromPackedPair: true };
  const sampler = inputUniform(source.pass);
  return { sampler, declaration: `uniform sampler2D ${sampler};`, fromPackedPair: false };
}

/** Separable blur of OKLab a/b at half resolution. */
export function chromaBlurPass(axis: "x" | "y", source: SkinBlurSource): string {
  const { sampler, declaration, fromPackedPair } = resolveSource(source);
  const offset =
    axis === "x" ? "vec2(float(i) * stepPx / uResolution.x, 0.0)" : "vec2(0.0, float(i) * stepPx / uResolution.y)";
  const decode = fromPackedPair
    ? `texture(${sampler}, uv).xy * 0.6 - 0.3`
    : `linearRgbToOklab(srgbToLinear(texture(${sampler}, uv).rgb)).yz`;
  return buildFragmentShader({
    extraUniforms: declaration,
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
export function lumaBlurPass(axis: "x" | "y", source: SkinBlurSource, radiusFactor: number): string {
  const { sampler, declaration, fromPackedPair } = resolveSource(source);
  const offset =
    axis === "x" ? "vec2(float(i) * stepPx / uResolution.x, 0.0)" : "vec2(0.0, float(i) * stepPx / uResolution.y)";
  const decode = fromPackedPair
    ? `texture(${sampler}, uv).r`
    : `linearRgbToOklab(srgbToLinear(texture(${sampler}, uv).rgb)).x`;
  return buildFragmentShader({
    extraUniforms: declaration,
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
