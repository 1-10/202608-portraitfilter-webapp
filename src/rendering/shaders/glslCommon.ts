export const FULLSCREEN_VERT = `#version 300 es
const vec2 POSITIONS[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
out vec2 vUv;
void main() {
  vec2 pos = POSITIONS[gl_VertexID];
  vUv = (pos + 1.0) * 0.5;
  gl_Position = vec4(pos, 0.0, 1.0);
}
`;

/** Shared GLSL helper functions available to every fragment shader pass. */
export const GLSL_LIB = `
vec3 rgb2hsv(vec3 c) {
  vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y);
  float e = 1.0e-10;
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float luminance(vec3 c) {
  return dot(c, vec3(0.299, 0.587, 0.114));
}

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float valueNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  float a = hash21(i);
  float b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0, 1.0));
  float d = hash21(i + vec2(1.0, 1.0));
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm(vec2 p) {
  float sum = 0.0;
  float amp = 0.5;
  // Rotate between octaves. Without this the value-noise lattices all line up on
  // the same axes and the result shows a faint square grid — most visible in
  // paper/grain textures, which are supposed to read as organic fibre.
  mat2 rot = mat2(0.80, 0.60, -0.60, 0.80);
  for (int i = 0; i < 4; i++) {
    sum += amp * valueNoise(p);
    p = rot * p * 2.02;
    amp *= 0.5;
  }
  return sum;
}

vec3 quantizeColor(vec3 color, float levels) {
  return floor(color * levels + 0.5) / max(levels, 1.0);
}

float srgbChannelToLinear(float c) {
  return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}
vec3 srgbToLinear(vec3 c) {
  return vec3(srgbChannelToLinear(c.r), srgbChannelToLinear(c.g), srgbChannelToLinear(c.b));
}
float linearChannelToSrgb(float c) {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}
vec3 linearToSrgb(vec3 c) {
  return vec3(linearChannelToSrgb(c.r), linearChannelToSrgb(c.g), linearChannelToSrgb(c.b));
}

// OKLab (Bjorn Ottosson). Input/output linear RGB. Perceptually near-uniform,
// used so color-region grouping and density math don't follow raw RGB's
// perceptually skewed distances.
vec3 linearRgbToOklab(vec3 c) {
  float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
  float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
  float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
  float l_ = pow(max(l, 0.0), 1.0 / 3.0);
  float m_ = pow(max(m, 0.0), 1.0 / 3.0);
  float s_ = pow(max(s, 0.0), 1.0 / 3.0);
  return vec3(
    0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
    1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
    0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
  );
}
vec3 oklabToLinearRgb(vec3 c) {
  float l_ = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
  float m_ = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
  float s_ = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;
  float l = l_ * l_ * l_;
  float m = m_ * m_ * m_;
  float s = s_ * s_ * s_;
  return vec3(
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
  );
}

float orderedDither(vec2 fragCoord) {
  return (hash21(fragCoord) - 0.5) / 255.0;
}

float sobelEdge(sampler2D tex, vec2 uv, vec2 texel) {
  float tl = luminance(texture(tex, uv + vec2(-texel.x, -texel.y)).rgb);
  float t  = luminance(texture(tex, uv + vec2(0.0, -texel.y)).rgb);
  float tr = luminance(texture(tex, uv + vec2(texel.x, -texel.y)).rgb);
  float l  = luminance(texture(tex, uv + vec2(-texel.x, 0.0)).rgb);
  float r  = luminance(texture(tex, uv + vec2(texel.x, 0.0)).rgb);
  float bl = luminance(texture(tex, uv + vec2(-texel.x, texel.y)).rgb);
  float b  = luminance(texture(tex, uv + vec2(0.0, texel.y)).rgb);
  float br = luminance(texture(tex, uv + vec2(texel.x, texel.y)).rgb);
  float gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br;
  float gy = -tl - 2.0 * t - tr + bl + 2.0 * b + br;
  return sqrt(gx * gx + gy * gy);
}

/**
 * 8-neighbour edge-preserving smooth (a cheap bilateral). falloff controls how
 * strongly a color difference rejects a neighbour — larger keeps more edges.
 * Shared because four filters previously carried near-identical copies of this
 * loop that differed only in radius and the falloff constant.
 */
vec3 bilateral8(sampler2D tex, vec2 uv, vec2 texel, float radius, float falloff) {
  vec3 center = texture(tex, uv).rgb;
  vec3 sum = center;
  float weightSum = 1.0;
  vec2 offsets[8] = vec2[8](
    vec2(1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0),
    vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0)
  );
  for (int i = 0; i < 8; i++) {
    vec3 c = texture(tex, uv + offsets[i] * texel * radius).rgb;
    float d = distance(c, center);
    float w = exp(-d * d * falloff);
    sum += c * w;
    weightSum += w;
  }
  return sum / weightSum;
}

/**
 * UV for procedural noise whose feature size is constant relative to the IMAGE
 * rather than the pixel grid, with aspect corrected so features stay square.
 * cycles ~= how many noise features span the image height.
 *
 * Filters MUST use this instead of vUv * uResolution * k: this pipeline renders
 * the same filter at ~140px (thumbnail), <=1280px (preview) and up to 4096px
 * (export), so pixel-locked noise makes the exported grain/paper/halftone several
 * times finer than the preview the user actually approved.
 */
vec2 stableNoiseUv(vec2 uv, vec2 resolution, float cycles) {
  float aspect = resolution.x / max(resolution.y, 1.0);
  return vec2(uv.x * aspect, uv.y) * cycles;
}

/**
 * Rough skin likelihood, used as the FALLBACK wherever MediaPipe face landmarks
 * are unavailable (see uHasFaceMask).
 *
 * Tests OKLab HUE rather than RGB channel ratios: human skin of every tone sits
 * in a tight hue band (~20deg-70deg, centre ~45deg = 0.785 rad) and varies mostly
 * in lightness and chroma. The earlier rNorm/gNorm formulation this replaces had
 * a lightness floor that silently excluded deep skin tones, and fired readily on
 * wood and warm-lit walls.
 *
 * Still approximate: warm-toned surfaces inside the same hue band can score, so
 * callers should treat it as a weight to be combined with other evidence rather
 * than a segmentation.
 */
float skinLikelihood(vec3 linearRgb) {
  vec3 lab = linearRgbToOklab(max(linearRgb, 0.0));
  float C = length(lab.yz);
  float h = atan(lab.z, lab.y);
  float hueScore = 1.0 - smoothstep(0.0, 0.45, abs(h - 0.785));
  float chromaScore = smoothstep(0.015, 0.045, C) * (1.0 - smoothstep(0.16, 0.26, C));
  float lightScore = smoothstep(0.02, 0.12, lab.x) * (1.0 - smoothstep(0.90, 0.99, lab.x));
  return clamp(hueScore * chromaScore * lightScore, 0.0, 1.0);
}

/** Gaussian weight, for hand-rolled separable blurs. */
float gaussW(float x, float sigma) {
  return exp(-0.5 * x * x / max(sigma * sigma, 1.0e-6));
}

/**
 * Sobel magnitude normalized to roughly 0..1. The raw sobelEdge() returns an
 * unnormalized magnitude up to ~5.66, which is why every existing call site
 * invented its own threshold constant.
 */
float sobelEdge01(sampler2D tex, vec2 uv, vec2 texel) {
  return clamp(sobelEdge(tex, uv, texel) * 0.177, 0.0, 1.0);
}

/**
 * Scale factor for any radius expressed in texels, relative to a 1280px
 * reference. Multiply texel-space radii by this so a 3px outline at the 1280
 * preview stays proportionally 3px-equivalent at a 4096 export instead of
 * becoming a 3x thinner hairline.
 */
float imageScale(vec2 resolution) {
  return max(resolution.x, resolution.y) / 1280.0;
}

/**
 * How strongly this pixel should be treated as retouchable skin: the MediaPipe
 * skin region when a face was detected, otherwise the color heuristic. Every
 * filter should go through this rather than sampling uFaceMask directly, so the
 * no-detection path keeps working.
 */
float faceSkinWeight(vec3 linearRgb, vec2 uv) {
  float heuristic = skinLikelihood(linearRgb);
  float detected = texture(uFaceMask, uv).r;
  return mix(heuristic, detected, uHasFaceMask);
}

/**
 * Skin weight for AGGRESSIVE retouching: the detected region, additionally gated
 * by chroma.
 *
 * faceSkinWeight() trusts the region outright when there is a detection, and the
 * region is a face oval — which includes the fringe of hair hanging over the
 * forehead. That is fine for a gentle retouch and wrong for a strong one: crushing
 * texture and averaging chroma across a fringe turns black hair beige.
 *
 * The extra gate is chroma ALONE, deliberately, and not skinLikelihood(): that
 * function's job is to find skin against an unknown background, so it also rejects
 * anything MORE chromatic than typical skin — which is exactly what a red blotch
 * is, and blotches are what this filter exists to remove. Hair is the opposite
 * case, sitting far below skin in chroma whatever its hue, so the low-chroma ramp
 * alone separates it, and it does so across skin tones.
 */
float faceSkinWeightStrict(vec3 linearRgb, vec2 uv) {
  vec3 lab = linearRgbToOklab(max(linearRgb, 0.0));
  // Measured on real portraits: skin chroma runs about 0.025-0.10, hair 0.004-0.010.
  // The ramp has to sit in that gap. Set too high it silently rejects the paler,
  // lower-chroma parts of the skin as well, which is indistinguishable from the
  // retouch simply not working.
  float chromatic = smoothstep(0.012, 0.030, length(lab.yz));
  // Chroma alone cannot reject a BROWN fringe, which can sit inside skin's chroma
  // range. Lightness relative to this subject's own skin can: shadowed skin still
  // reaches two thirds of the median, hair does not come close.
  float bright = smoothstep(0.55, 0.8, lab.x / max(uFaceSkinLightness, 0.05));
  float detected = texture(uFaceMask, uv).r;
  return mix(skinLikelihood(linearRgb), detected * chromatic * bright, uHasFaceMask);
}

/**
 * Skin ANYWHERE in frame, not only inside the detected oval.
 *
 * The region mask is a face oval and nothing else, so a strong treatment gated on it
 * alone puts a doll's head on the original body: the face is evened out, lifted and
 * contoured while the neck, chest and arms keep every blotch and shadow they had.
 * The step at the jaw is the most conspicuous artifact this kind of filter can
 * produce, and it is invisible in a head-and-shoulders crop — which is exactly why
 * it survives review against reference images that are all head crops. Phone beauty
 * filters treat all visible skin, so this returns the oval judgement OR a
 * colour-only one, whichever is higher.
 *
 * The colour-only branch leans on CHROMA, from measurement: skin runs 0.04-0.10 in
 * OKLab while a neutral wall runs under 0.03, hair under 0.01, and a grey shirt
 * near zero. That gap is real but not wide, so a warm and strongly coloured
 * background can score here. The oval stays the strong prior and this only ever
 * ADDS to it, so the failure mode is a slightly evened-out wall rather than a
 * missed face.
 */
float faceSkinWeightBroad(vec3 linearRgb, vec2 uv) {
  vec3 lab = linearRgbToOklab(max(linearRgb, 0.0));
  float C = length(lab.yz);
  float h = atan(lab.z, lab.y);
  float hue = 1.0 - smoothstep(0.0, 0.5, abs(h - 0.785));
  float chromatic = smoothstep(0.032, 0.05, C) * (1.0 - smoothstep(0.16, 0.26, C));
  // Relative to the subject's own measured skin, so shadowed skin still counts and
  // no absolute lightness floor decides which skin tones the filter works on.
  float bright = smoothstep(0.45, 0.72, lab.x / max(uFaceSkinLightness, 0.05));
  return max(faceSkinWeightStrict(linearRgb, uv), clamp(hue * chromatic * bright, 0.0, 1.0));
}

/** The whole face oval, generously feathered. 0 with no detection. */
float faceRegionWeight(vec2 uv) {
  return texture(uFaceMask, uv).a * uHasFaceMask;
}

/**
 * Eyes + lips + brows: the features that must stay sharp when everything else is
 * being flattened or smoothed. Returns 0 with no detection, so callers degrade to
 * "protect nothing specifically" rather than protecting the wrong region.
 */
float faceFeatureWeight(vec2 uv) {
  vec4 m = texture(uFaceMask, uv);
  return clamp(max(m.g, m.b), 0.0, 1.0) * uHasFaceMask;
}

/** Segmentation confidences at \`uv\` (see uPersonSeg). Zero outside its rect. */
vec4 personSeg(vec2 uv) {
  vec2 g = (uv - uPersonSegRect.xy) / max(uPersonSegRect.zw, vec2(1.0e-6));
  if (any(lessThan(g, vec2(0.0))) || any(greaterThan(g, vec2(1.0)))) return vec4(0.0);
  return texture(uPersonSeg, g);
}

/**
 * How far to trust personSeg() at \`uv\`, 0..1: zero without a segmentation, and
 * fading out over the outer 8% of its rect, so a filter that mixes it with a colour
 * heuristic hands over smoothly instead of drawing the rect's edge into the image.
 */
float personSegCoverage(vec2 uv) {
  vec2 g = (uv - uPersonSegRect.xy) / max(uPersonSegRect.zw, vec2(1.0e-6));
  vec2 edge = smoothstep(vec2(0.0), vec2(0.08), g) * smoothstep(vec2(0.0), vec2(0.08), 1.0 - g);
  return edge.x * edge.y * uHasPersonSeg;
}

/** Image UV -> face-rect local UV. Outside the rect the result leaves 0..1. */
vec2 faceGeomUv(vec2 uv) {
  return (uv - uFaceGeomRect.xy) / max(uFaceGeomRect.zw, vec2(1.0e-6));
}

bool insideGeom(vec2 g) {
  return all(greaterThanEqual(g, vec2(0.0))) && all(lessThanEqual(g, vec2(1.0)));
}

/**
 * Samples a face-rect texture, returning 0 outside the rect.
 *
 * The explicit test is not redundant with CLAMP_TO_EDGE: clamping would repeat the
 * rect's border texels across the whole frame, painting a streak of lipstick out to
 * the image edge.
 */
vec4 sampleGeom(sampler2D tex, vec2 g) {
  if (!insideGeom(g)) return vec4(0.0);
  return texture(tex, g);
}

/**
 * The mesh deformation, as a UV offset to add before sampling.
 *
 * Two independent fields are stored (face shape in .rg, eyes in .ba) and weighted
 * here, so the deformation sliders are uniforms and nothing is re-rasterized on the
 * CPU while the user drags. See vision/faceWarpField.ts for the encoding and for
 * why summing the two groups is sound.
 *
 * Byte 128 is exactly zero displacement, so the bias below must stay 128/255.
 */
vec2 faceWarpOffset(vec2 uv, float shapeAmount, float eyeAmount) {
  if (uHasFaceGeometry < 0.5) return vec2(0.0);
  vec2 g = faceGeomUv(uv);
  if (!insideGeom(g)) return vec2(0.0);
  vec4 f = texture(uFaceWarp, g);
  vec2 shape = (f.rg - vec2(128.0 / 255.0)) * uFaceWarpRange;
  vec2 eyes = (f.ba - vec2(128.0 / 255.0)) * uFaceWarpRange;
  return shape * shapeAmount + eyes * eyeAmount;
}

/**
 * Where the pixel now at \`uv\` came from before the deformation.
 *
 * Passes AFTER the warp must look up the face mask and the makeup masks through
 * this, because both were baked against the undeformed landmarks. Sampling them at
 * the raw uv instead slides the lipstick off the lip by exactly the displacement.
 */
vec2 faceWarpedUv(vec2 uv, float shapeAmount, float eyeAmount) {
  return uv + faceWarpOffset(uv, shapeAmount, eyeAmount);
}
`;

const COMMON_UNIFORMS = `
precision highp float;
in vec2 vUv;
out vec4 fragColor;
uniform sampler2D uSource;
uniform sampler2D uOriginal;
uniform vec2 uResolution;
uniform vec2 uTexelSize;
/** Always 1/(original working-resolution size), regardless of this pass's own outputScale. */
uniform vec2 uOriginalTexelSize;
uniform float uStrength;
/**
 * Face regions from MediaPipe, at a fixed low resolution sampled by normalized vUv:
 *   .r = skin (face oval minus eyes/brows/mouth)   .g = eyes
 *   .b = lips                                      .a = whole face, feathered
 * Always bound (a 1x1 opaque white texture when there is no face), so sampling is
 * never undefined. Gate on uHasFaceMask instead of assuming the contents.
 */
uniform sampler2D uFaceMask;
/** 1.0 when uFaceMask holds a real detection, 0.0 when it is the placeholder. */
uniform float uHasFaceMask;
/**
 * Interocular distance as a fraction of the image long edge (~0.14 for a typical
 * head-and-shoulders portrait). Scale outline widths and blemish-band sizes by
 * this so a headshot and a full-body shot get proportionally identical results
 * instead of the effect shrinking as the subject gets smaller in frame.
 */
uniform float uFaceScale;
/**
 * Mesh geometry baked on the CPU from the face landmarks, all three addressed over
 * the face rect (uFaceGeomRect) rather than the whole frame — that is what gives a
 * 1024px makeup texture several hundred pixels across an eye.
 *
 *   uFaceWarp   .rg = face-shape displacement   .ba = eye displacement
 *               signed, biased by 128/255; decode through faceWarpOffset()
 *   uMakeupA    .r lip fill   .g lip gloss   .b blush     .a eyeshadow
 *   uMakeupB    .r lash/liner .g brow        .b highlight .a catchlight
 *
 * Always bound (1x1 stand-ins when there is no face), so sampling is never
 * undefined. Gate on uHasFaceGeometry rather than assuming the contents.
 */
uniform sampler2D uFaceWarp;
uniform sampler2D uMakeupA;
uniform sampler2D uMakeupB;
/** Face rect in UV space: xy = min corner, zw = size. (0,0,1,1) when absent. */
uniform vec4 uFaceGeomRect;
/** UV displacement per unit of encoded warp value, per axis. */
uniform vec2 uFaceWarpRange;
/** 1.0 when the three textures above hold a real bake, 0.0 when they are stand-ins. */
uniform float uHasFaceGeometry;
/**
 * This subject's own median skin lightness (OKLab L), measured per image.
 *
 * The reference a relative test needs: hair is a fraction of skin's lightness, but
 * an ABSOLUTE floor is the mistake that makes a filter work on pale skin and fail
 * on dark skin. Falls back to a mid-tone when no face was measured.
 */
uniform float uFaceSkinLightness;
/**
 * The same subject's median skin OKLab a/b. The reference for a RELATIVE redness
 * test: median skin a/b differs a lot between people, so an absolute threshold
 * flags one person's ordinary complexion as flushed and misses another's flush.
 */
uniform vec2 uFaceSkinAb;
/**
 * Person parts from the SelfieMulticlass segmenter, addressed over uPersonSegRect:
 *   .r hair   .g face skin   .b body skin   .a clothes   (model confidences)
 * Always bound (a 1x1 stand-in when absent). Read through personSeg(), which returns
 * zero outside the rect, and weigh it with personSegCoverage().
 */
uniform sampler2D uPersonSeg;
uniform vec4 uPersonSegRect;
uniform float uHasPersonSeg;
`;

export type BuildFragmentShaderOptions = {
  /** Extra `uniform ...;` declarations specific to this pass (parameters, extra inputs). */
  extraUniforms?: string;
  /** GLSL statements for `void main() { ... }`. Must write to `fragColor`. */
  body: string;
};

export function buildFragmentShader(opts: BuildFragmentShaderOptions): string {
  return `#version 300 es
${COMMON_UNIFORMS}
${opts.extraUniforms ?? ""}
${GLSL_LIB}
void main() {
${opts.body}
}
`;
}

/** Uniform name for a parameter with the given id. */
export function paramUniform(paramId: string): string {
  return `uParam_${paramId}`;
}

/** Uniform name for an extra input bound from another pass's output (or "original"). */
export function inputUniform(passId: string): string {
  return `uInput_${passId}`;
}
