import { buildFragmentShader, inputUniform, paramUniform } from "../rendering/shaders/glslCommon";
import type { FilterDefinition } from "../types";
import { flowPass, flowTensorBlurPass, flowTensorPass } from "./shared/flowField";
import { guidedCoefPass, guidedMeanPass, guidedOutput, PACK16_GLSL } from "./shared/guidedFilter";
import { chromaBlurPass, lumaBlurPass } from "./shared/skinRetouch";

/**
 * "AI look": the finish of a generative portrait retouch, rebuilt as a filter.
 *
 * Measured against image-to-image AI output aligned pixel-for-pixel with its source,
 * the generator does not reshape the face at all. What it does to the subject is:
 *
 *   - skin lightness converges on one value whatever the source exposure
 *   - skin loses redness and yellowness, most where it is flushed, and its shadows
 *     stay warm while its highlights go neutral
 *   - fine skin detail (pores, fine lines) is almost entirely removed, while the
 *     broad shading that gives the face its form survives untouched
 *   - lips get lighter
 *   - hair strands and silhouette edges get SHARPER, the opposite of the skin, and
 *     hair and dark clothing get deeper blacks
 *
 * So there is no mesh deformation and no painted makeup here. Skin and tone are one
 * pass over a three-band split, and the hair/edge sharpening is a second pass that
 * is gated to exactly the places the skin pass leaves alone.
 */

/**
 * Band radii, as fractions of face width. See shared/skinRetouch.ts.
 *
 * The skin itself is smoothed by the guided filter below, not by these bands. They
 * serve the operations that need a local reference: the low band is the broad
 * shading the shine layer and the warm-shadow term are measured against, the mid
 * band separates fine from mid detail on the lips. The low radius stays below the
 * 0.05-0.15 face-width scale of the nose and cheekbone modelling, so measuring
 * against it never treats form as shine.
 */
const BAND_LOW_RADIUS = 0.03;
const BAND_MID_RADIUS = 0.007;

/**
 * Guided-filter window radius (fraction of face width) and eps (squared OKLab L).
 * Deviations below sqrt(eps) = 0.04 L — pores, blotches, fine lines — are flattened;
 * nostrils, the lip line, brow and lid edges are larger and are kept.
 */
const GF_RADIUS = 0.012;
const GF_EPS = 0.04 * 0.04;

/**
 * Radius of the opening that regularizes skin shine, as a fraction of face width.
 * Glints smaller than this — the irregular oily speckle and streaks of a real face —
 * are removed; compact highlights larger than it survive.
 */
const SHINE_OPEN_RADIUS = 0.008;

/** Shine is stored x4 so its small range uses more of the 16-bit packing. */
const SHINE_SCALE = 4;

/**
 * Pass: the shine layer (lightness above the broad shading), eroded over a small
 * disk, or dilated back over the same disk ("min" then "max" is a morphological
 * opening).
 */
function shineOpenPass(op: "min" | "max"): string {
  const read =
    op === "min"
      ? `max(linearRgbToOklab(srgbToLinear(texture(uOriginal, uv).rgb)).x - texture(${inputUniform("lumaLowV")}, uv).r, 0.0) * ${SHINE_SCALE.toFixed(1)}`
      : "unpack16(texture(uSource, uv).xy)";
  return buildFragmentShader({
    extraUniforms: `${PACK16_GLSL}${op === "min" ? `
uniform sampler2D ${inputUniform("lumaLowV")};` : ""}`,
    body: `
  float faceWidthPx = max(uFaceScale * uResolution.x * 2.5, 24.0);
  vec2 stepUv = (max(faceWidthPx * ${SHINE_OPEN_RADIUS.toFixed(4)}, 1.0) / 2.0) / uResolution;
  float acc = ${op === "min" ? "1.0" : "0.0"};
  for (int y = -2; y <= 2; y++) {
    for (int x = -2; x <= 2; x++) {
      vec2 o = vec2(float(x), float(y));
      if (dot(o, o) > 5.0) continue;
      vec2 uv = vUv + o * stepUv;
      acc = ${op}(acc, ${read});
    }
  }
  fragColor = vec4(pack16(acc), 0.0, 1.0);`,
  });
}

/** Long-edge cap for the blur passes; their radii are face-relative, so this costs precision only. */
const BAND_CAP = 1600;

/**
 * Range of OKLab lightness the skin can be pulled toward, set by the "仕上がりの白さ"
 * slider. The reference lands every subject's skin in a narrow band around 0.745
 * regardless of how dark the source exposure was; the slider's default sits just
 * below it because the restored skin sheen adds its own lightness on top.
 */
const TARGET_SKIN_L_MIN = 0.7;
const TARGET_SKIN_L_RANGE = 0.1;

/**
 * Skin: band recombination, complexion, and the lightness pull.
 */
const skinPass = buildFragmentShader({
  extraUniforms: `uniform sampler2D ${inputUniform("chromaV")};
uniform sampler2D ${inputUniform("lumaLowV")};
uniform sampler2D ${inputUniform("lumaMidV")};
uniform sampler2D ${inputUniform("gfMean")};
uniform sampler2D ${inputUniform("shineMax")};
${PACK16_GLSL}
uniform float ${paramUniform("smooth")};
uniform float ${paramUniform("tone")};
uniform float ${paramUniform("clarity")};
uniform float ${paramUniform("lips")};
uniform float ${paramUniform("target")};
uniform float ${paramUniform("shine")};
uniform float ${paramUniform("warmth")};
uniform float ${paramUniform("flush")};`,
  body: `
  vec3 srcSrgb = texture(uOriginal, vUv).rgb;
  vec3 srcLin = srgbToLinear(srcSrgb);
  vec3 lab = linearRgbToOklab(srcLin);

  float Llow = texture(${inputUniform("lumaLowV")}, vUv).r;
  float Lmid = texture(${inputUniform("lumaMidV")}, vUv).r;
  vec2 abLow = texture(${inputUniform("chromaV")}, vUv).xy * 0.6 - 0.3;

  float blemish = Lmid - Llow;
  float pore = lab.x - Lmid;
  float smoothAmount = ${paramUniform("smooth")} / 100.0;

  // The strict-plus-broad weight keeps a dark fringe over the forehead out of the
  // smoothing (its chroma is far below skin's) while still reaching neck and chest,
  // so the retouch does not stop in a step at the jaw.
  //
  // Where the person segmentation reaches, its skin classes replace the colour
  // judgement outright: they know a grey fringe or a warm wall is not skin, which no
  // colour test does.
  vec4 seg = personSeg(vUv);
  float segCov = personSegCoverage(vUv);
  // Thresholded rather than taken raw: the model's confidence on unambiguous skin sits
  // around 0.7-0.9, and used as a weight that alone would weaken every skin operation.
  float segSkin = smoothstep(0.2, 0.6, seg.g + seg.b);
  float region = mix(faceSkinWeightBroad(srcLin, vUv), segSkin, segCov) * (1.0 - faceFeatureWeight(vUv));
  // Weak on purpose: fine lines and creases are edges, and they are what the reference
  // removes. Hair across a cheek is kept out by the chroma term in region, not here.
  // Sampled at a face-relative step: a one-texel Sobel weakens as resolution rises,
  // so the gate would open at a 4K export where it stays shut in the preview.
  float faceWidthPx = max(uFaceScale * uResolution.x * 2.5, 24.0);
  float edgeStep = max(faceWidthPx * 0.0025, 1.0);
  float edge = sobelEdge01(uOriginal, vUv, uOriginalTexelSize * edgeStep);
  float gate = clamp(region * (1.0 - smoothstep(0.3, 0.75, edge)), 0.0, 1.0);

  // The reference does not soften the subject's own pores, it removes them, and it
  // keeps every feature edge crisp: flat interiors with sharp edges is the doll
  // finish, flat interiors with softened edges is a retouched photo. So the skin is
  // replaced by an edge-preserving (guided) smooth of itself rather than by a blend
  // of Gaussian bands, which would soften every nostril and lip line the gate does
  // not fully protect. The edge gate is correspondingly weak: the filter itself
  // keeps the edges.
  float sAmt = min(smoothAmount * 1.25, 1.0);
  float gfGate = clamp(region * (1.0 - smoothstep(0.5, 0.95, edge)), 0.0, 1.0);
  float L2 = mix(lab.x, ${guidedOutput("gfMean", "lab.x")}, sAmt * gfGate);

  // Shine: a real face has irregular oily speckle and streaks; the reference has a
  // satin field with compact highlights. The shine layer is replaced by its opening,
  // which removes glints smaller than the opening disk and keeps larger highlights.
  // Eyes and lips are left out: their catchlights and gloss are glints by this test.
  float shine = max(lab.x - Llow, 0.0);
  float shineOpen = min(unpack16(texture(${inputUniform("shineMax")}, vUv).xy) / ${SHINE_SCALE.toFixed(1)}, shine);
  float glintW = (1.0 - ${paramUniform("shine")} / 100.0) * region;

  // Chroma blotches are smoothed toward the local mean; the eye cannot resolve chroma
  // at this scale, so it costs no sharpness.
  vec2 ab2 = mix(lab.yz, abLow, 0.4 * smoothAmount * gate);

  vec4 regions = texture(uFaceMask, vUv);
  float lipW = clamp(regions.b, 0.0, 1.0) * uHasFaceMask;
  // Lips: the reference's are smooth, without creases. Lightness only, since a chroma
  // blur would bleed skin colour into them; the clamps keep the lip's darkness against
  // the skin and the vermilion edge, and the edge gate keeps the outline.
  glintW *= (1.0 - lipW) * (1.0 - clamp(regions.g, 0.0, 1.0) * uHasFaceMask);
  // The shine layer of the smoothed skin is rebuilt, not just trimmed: the guided
  // filter flattens soft highlights along with pores (a cheek or forehead sheen is a
  // deviation below its eps), and that is where gloss goes. Rebuilding the layer from
  // the source's own shine — glints opened away, compact highlights kept — brings
  // the gloss back without the oily speckle.
  float targetShine = mix(shine, shineOpen, glintW);
  float smoothedShine = max(L2 - Llow, 0.0);
  // Only ever ADDED back: where the source is darker than its surroundings the
  // target is zero, and pulling the smoothed skin down to it would put the pores back.
  L2 += max(targetShine - smoothedShine, 0.0) * region * (1.0 - lipW);
  float lipSmooth = 0.8 * lipW * (1.0 - smoothstep(0.3, 0.75, edge)) * sAmt;
  L2 -= lipSmooth * (clamp(pore, -0.04, 0.04) + 0.4 * clamp(blemish, -0.015, 0.015));
  float skinL = uFaceSkinLightness;
  float rel = L2 / max(skinL, 0.05);
  // Eye white is bright and neutral; it is released from the eye hold below because
  // the reference lifts it with the skin.
  float eyesRegion = clamp(regions.g, 0.0, 1.0) * uHasFaceMask;
  float eyesHold = eyesRegion;
  eyesHold *= 1.0 - smoothstep(0.85, 1.0, rel) * (1.0 - smoothstep(0.02, 0.04, length(lab.yz)));
  float browHold = sampleGeom(uMakeupB, faceGeomUv(vUv)).g * uHasFaceGeometry;

  // Complexion weight. Not the oval: a colour shift of this size gated by the oval's
  // few-percent feather draws a seam along the jaw. Not skinLikelihood() either: its
  // hue window is centred at 45deg while measured skin sits at 30-38deg, so it scores
  // a flushed cheek, the very pixel that needs the most correction, lowest. Blurred
  // chroma, so compression blocks do not get individually different shifts.
  float hueSkin = atan(abLow.y, abLow.x);
  float toneSkin = (1.0 - smoothstep(0.0, 0.6, abs(hueSkin - 0.58)))
                 * smoothstep(0.015, 0.035, length(abLow))
                 * smoothstep(0.55, 0.8, rel) * (1.0 - lipW);
  toneSkin = mix(toneSkin, segSkin * (1.0 - lipW), segCov);
  // Eyes and brows are inside the face-skin class, and the warm-shadow term below
  // would redden a brow.
  toneSkin *= (1.0 - eyesHold) * (1.0 - browHold);

  // The reference moves the complexion's MEAN and keeps its variation: rosy cheeks,
  // peach highlights. So the cut is a SUBTRACTION sized from this subject's median
  // skin a/b, not a scale, which would shrink the variation by the same factor and
  // leave the face grey. Yellow (b) is cut at least as hard as red (a), and redness
  // that runs above the median gets part of its excess removed on top.
  //
  // Below median chroma the shift tapers away, so stubble, brows and already pale
  // pixels are never pushed through neutral into green or blue.
  float clarity = ${paramUniform("clarity")} / 100.0;
  float flush = smoothstep(0.004, 0.016, abLow.x - uFaceSkinAb.x);
  float guard = clamp(length(ab2) / max(length(uFaceSkinAb), 1.0e-3), 0.0, 1.0);
  vec2 shift = vec2(0.36 * uFaceSkinAb.x + ${paramUniform("flush")} / 100.0 * flush * max(abLow.x - uFaceSkinAb.x, 0.0),
                    0.42 * uFaceSkinAb.y);
  vec2 abSub = ab2 - clarity * toneSkin * guard * shift;
  // Without a measured median there is nothing to size the shift from; a plain scale
  // is the safe fallback.
  vec2 abMul = ab2 * vec2(1.0 - 0.3 * clarity * toneSkin, 1.0 - 0.45 * clarity * toneSkin);
  ab2 = mix(abMul, abSub, uHasFaceGeometry);
  // Warm shadows, neutral highlights: the reference's skin a* rises as the broad
  // shading darkens, by a slope that is the same for every subject measured.
  ab2.x -= 0.06 * ${paramUniform("warmth")} / 100.0 * (Llow - skinL) * toneSkin * clarity;

  // Eye whites: clean and white in the reference. Only the dark specks are filled (the
  // dark side of the local deviation), so the iris edge stays crisp and no grey halo
  // forms around it, and the sclera's colour cast is reduced.
  float sclera = eyesRegion * smoothstep(0.85, 1.0, rel) * (1.0 - smoothstep(0.035, 0.07, length(lab.yz)));
  L2 -= 0.8 * sclera * min(lab.x - Llow, 0.0);
  ab2 *= 1.0 - 0.3 * sclera;

  // Lightness pull toward the target, sized by how far THIS subject's measured skin
  // is from it, so a dark exposure gets a large lift and a bright one almost none.
  //
  // Below the skin median it is a pure gain (the added amount is proportional to L),
  // which stretches the shadows the way an exposure change does instead of flattening
  // them; above the median it is a constant offset, so highlights keep their spread,
  // and a soft shoulder keeps them from clipping. Monotone throughout.
  //
  // The floor is relative to the subject's skin and sits above hair, so black hair
  // keeps its black. The rising ramp is kept wide because its slope multiplies local
  // contrast: a narrow one makes compression blocks stand out.
  float target = ${TARGET_SKIN_L_MIN.toFixed(3)} + ${TARGET_SKIN_L_RANGE.toFixed(3)} * ${paramUniform("target")} / 100.0;
  float delta = clamp(target - skinL, -0.02, 0.18);
  // Without a measured face the skin value is a guess, so only part of the pull applies.
  delta *= mix(0.4, 1.0, uHasFaceGeometry);
  float lift = smoothstep(0.55, 0.92, rel) * min(rel, 1.0);
  // Grey and white hair sits at skin lightness, so the lightness floor cannot keep it
  // out; chroma can, since skin never drops below ~0.03 and hair sits well under it.
  //
  // The segmentation may only REDUCE what the chroma gate allows, never raise it:
  // otherwise a grey shirt or wall inside the segmentation rect gets the full lift
  // while the same surface just outside it does not, and the rect's edge shows. The
  // hair class only counts where chroma agrees, so forehead skin the coarse mask calls
  // hair at the hairline is not left as a dark band.
  float chromaGate = mix(0.1, 1.0, smoothstep(0.012, 0.03, length(abLow)));
  float lowChroma = 1.0 - smoothstep(0.02, 0.04, length(abLow));
  float segGate = max(chromaGate, segSkin) * (1.0 - 0.9 * seg.r * lowChroma);
  lift *= mix(chromaGate, min(segGate, chromaGate), segCov);

  // Eyes and brows are held back. Lips get three quarters of the skin's lift at the
  // default, which is what the reference does on average.
  float hold = 1.0 - 0.85 * clamp(max(eyesHold, browHold), 0.0, 1.0);
  hold *= 1.0 - lipW * (1.0 - 0.75 * ${paramUniform("lips")} / 50.0);

  float Lpre = L2;
  L2 += delta * (${paramUniform("tone")} / 100.0) * lift * clamp(hold, 0.0, 1.0);
  L2 = L2 < 0.92 ? L2 : 1.0 - 0.08 * exp(-(L2 - 0.92) / 0.08);
  // Chroma follows the lift. Adding lightness at constant a/b lowers relative
  // saturation, which is what makes a heavily lifted face look ashy.
  ab2 *= mix(1.0, clamp(L2 / max(Lpre, 0.05), 1.0, 1.4), 0.8);

  vec3 outLin = max(oklabToLinearRgb(vec3(L2, ab2)), 0.0);
  fragColor = vec4(linearToSrgb(clamp(outLin, 0.0, 1.0)), 1.0);`,
});

/**
 * Hair strands smoothed ALONG the local flow (a short line-integral convolution).
 *
 * A compressed phone photo's hair is ragged — broken strands, blocky edges — and
 * sharpening makes that worse; the reference's strands are long and coherent.
 * Averaging along the strand direction joins the broken pieces without blurring
 * across strands, and the detail pass then sharpens across them. Gated by the hair
 * class and by anisotropy, so only genuinely directional hair is touched.
 */
const hairFlowPass = buildFragmentShader({
  extraUniforms: `uniform sampler2D ${inputUniform("flow")};
uniform float ${paramUniform("hair")};`,
  body: `
  vec3 base = texture(uSource, vUv).rgb;
  vec4 seg = personSeg(vUv);
  vec4 f = texture(${inputUniform("flow")}, vUv);
  float w = personSegCoverage(vUv) * seg.r * smoothstep(0.2, 0.5, f.b) * clamp(${paramUniform("hair")} / 55.0, 0.0, 2.0);
  if (w <= 0.0) {
    fragColor = vec4(base, 1.0);
    return;
  }
  float faceWidthPx = max(uFaceScale * uResolution.x * 2.5, 24.0);
  vec2 stepUv = max(faceWidthPx * 0.006, 1.0) / uResolution;
  vec2 tangent = f.rg * 2.0 - 1.0;
  vec3 acc = base;
  float accW = 1.0;
  for (int side = 0; side < 2; side++) {
    vec2 p = vUv;
    vec2 d = side == 0 ? tangent : -tangent;
    for (int s = 1; s <= 4; s++) {
      vec2 dn = texture(${inputUniform("flow")}, p).rg * 2.0 - 1.0;
      if (dot(dn, d) < 0.0) dn = -dn;
      d = dn;
      p += d * stepUv;
      float sw = gaussW(float(s), 2.5);
      acc += texture(uSource, p).rgb * sw;
      accW += sw;
    }
  }
  fragColor = vec4(mix(base, acc / accW, clamp(0.5 * w, 0.0, 1.0)), 1.0);`,
});

/**
 * Hair, eye and silhouette sharpening, plus the hair tone change, applied to the
 * retouched frame.
 *
 * An unsharp mask on lightness, gated to NOT-skin near the head. The skin gate is the
 * same broad weight the skin pass smooths with, so the two passes partition the head
 * between them instead of fighting over the same pixels. The correction is clamped so
 * a hard edge against a bright background gets crispness rather than a halo.
 */
const detailPass = buildFragmentShader({
  extraUniforms: `uniform float ${paramUniform("detail")};
uniform float ${paramUniform("hair")};`,
  body: `
  vec3 baseSrgb = texture(uSource, vUv).rgb;
  vec3 lab = linearRgbToOklab(srgbToLinear(baseSrgb));

  float faceWidthPx = max(uFaceScale * uResolution.x * 2.5, 24.0);
  float sigmaPx = max(faceWidthPx * 0.004, 1.0);
  float sum = 0.0;
  float wsum = 0.0;
  for (int y = -2; y <= 2; y++) {
    for (int x = -2; x <= 2; x++) {
      vec2 o = vec2(float(x), float(y));
      float w = gaussW(length(o), 1.0);
      sum += linearRgbToOklab(srgbToLinear(texture(uSource, vUv + o * sigmaPx * uTexelSize).rgb)).x * w;
      wsum += w;
    }
  }
  float hp = lab.x - sum / wsum;

  vec3 origLin = srgbToLinear(texture(uOriginal, vUv).rgb);
  vec4 seg = personSeg(vUv);
  float segCov = personSegCoverage(vUv);
  float notSkin = 1.0 - mix(faceSkinWeightBroad(origLin, vUv), clamp(seg.g + seg.b, 0.0, 1.0), segCov);

  // Distance from the face in face widths, aspect-corrected so the radius is a circle.
  // The rect is the face oval grown by the geometry bake's margin, hence the constant.
  float aspect = uResolution.x / max(uResolution.y, 1.0);
  vec2 rectCentre = uFaceGeomRect.xy + uFaceGeomRect.zw * 0.5;
  vec2 offset = (vUv - rectCentre) * vec2(aspect, 1.0);
  float faceW = max(uFaceGeomRect.z * aspect / 1.54, 1.0e-4);
  float nearHead = 1.0 - smoothstep(1.0, 2.2, length(offset) / faceW);
  nearHead = mix(1.0, nearHead, uHasFaceGeometry);

  // Hair strands and silhouettes are strong edges; the block noise of a compressed
  // source on a flat wall is a weak one, and sharpening it makes the blocks visible.
  // Face-relative Sobel step, for the same resolution reason as the skin gate.
  float structure = smoothstep(0.05, 0.18, sobelEdge01(uOriginal, vUv, uOriginalTexelSize * sigmaPx));

  float detail = ${paramUniform("detail")} / 100.0;
  // Segmented, the sharpening goes to hair by class rather than to
  // "not skin near the head".
  float sharpenWhere = mix(notSkin * nearHead, seg.r, segCov);
  float amount = detail * sharpenWhere * structure;
  // The eyes are sharpened too, more gently: the skin smoothing around them otherwise
  // leaves them softer than the source, and the reference has them crisper.
  float eyes = clamp(texture(uFaceMask, vUv).g, 0.0, 1.0) * uHasFaceMask;
  amount = max(amount, detail * 0.75 * eyes);
  float add = hp * 2.2 * amount;
  // A phone's own over-sharpening leaves a bright halo outside the hair silhouette;
  // brightening there, on what is not hair, would amplify it. The eyes are exempt, or
  // their sharpening could only ever darken and the catchlight never gets crisper.
  if (add > 0.0) add *= mix(1.0, max(smoothstep(0.4, 0.8, seg.r), eyes), segCov);
  float L2 = lab.x + clamp(add, -0.08, 0.08);

  // Feature edges — brows, lip outline, nostrils, the jaw — crisped along strong
  // structure only. The skin pass leaves interiors flat; this gives their edges the
  // clean, drawn quality the reference has. The gradient is taken at a few times the
  // unsharp radius so pore-scale noise, already removed, cannot trigger it, and the
  // gain is small and clamped so an edge gets crisper rather than haloed.
  vec2 gStep = 2.0 * sigmaPx * uTexelSize;
  float gx = linearRgbToOklab(srgbToLinear(texture(uSource, vUv + vec2(gStep.x, 0.0)).rgb)).x
           - linearRgbToOklab(srgbToLinear(texture(uSource, vUv - vec2(gStep.x, 0.0)).rgb)).x;
  float gy = linearRgbToOklab(srgbToLinear(texture(uSource, vUv + vec2(0.0, gStep.y)).rgb)).x
           - linearRgbToOklab(srgbToLinear(texture(uSource, vUv - vec2(0.0, gStep.y)).rgb)).x;
  float strong = smoothstep(0.02, 0.05, 0.5 * length(vec2(gx, gy)));
  float faceOval = texture(uFaceMask, vUv).a * uHasFaceMask;
  // Hair inside the face oval (a fringe) is already sharpened as hair; doubling it
  // up turns strand edges into ink lines.
  float featW = detail / 0.55 * faceOval * (1.0 - eyes) * (1.0 - seg.r * segCov) * strong;
  L2 += clamp(hp * 0.6 * featW, -0.04, 0.04);

  // Hair: contrast stretched about a pivot above most hair tones, so strands
  // separate, blacks deepen, and only the strand highlights get brighter (a lower
  // pivot brightens grey hair as a whole). Hair already at or below 0.20 is left
  // alone, so black hair keeps its sheen instead of being crushed further.
  // Monotone (slope 1 + 0.3 w). Gated by the hair class alone, which ends where the
  // hair does.
  float hairW = segCov * seg.r * ${paramUniform("hair")} / 55.0;
  L2 = max(L2 + 0.3 * (L2 - 0.45) * hairW, min(L2, max(0.8 * L2, 0.20)));

  vec3 outLin = max(oklabToLinearRgb(vec3(L2, lab.yz)), 0.0);
  vec3 result = mix(texture(uOriginal, vUv).rgb, linearToSrgb(clamp(outLin, 0.0, 1.0)), uStrength);
  fragColor = vec4(result + orderedDither(vUv * uResolution), 1.0);`,
});

export const aiLookFilter: FilterDefinition = {
  id: "ai-look",
  name: "AIルック",
  description:
    "生成AIのポートレート補正の仕上がりを再現します。顔の形は変えず、肌の明るさを一定にそろえ、赤み・黄みと細かい凹凸だけを消し、髪と輪郭はシャープにします。",
  parameters: [
    { id: "tone", label: "肌の明るさ", min: 0, max: 100, step: 1, defaultValue: 85 },
    { id: "smooth", label: "肌のなめらかさ", min: 0, max: 100, step: 1, defaultValue: 80 },
    { id: "clarity", label: "赤み・黄み除去", min: 0, max: 100, step: 1, defaultValue: 70 },
    { id: "lips", label: "唇の明るさ", min: 0, max: 100, step: 1, defaultValue: 50 },
    { id: "detail", label: "くっきり感（目・輪郭）", min: 0, max: 100, step: 1, defaultValue: 55 },
    { id: "target", label: "仕上がりの白さ", min: 0, max: 100, step: 1, defaultValue: 45 },
    { id: "hair", label: "髪のツヤ・黒さ", min: 0, max: 100, step: 1, defaultValue: 55 },
    { id: "shine", label: "ツヤの残し具合", min: 0, max: 100, step: 1, defaultValue: 30 },
    { id: "warmth", label: "陰影の暖かみ", min: 0, max: 100, step: 1, defaultValue: 50 },
    { id: "flush", label: "頬の赤み除去", min: 0, max: 100, step: 1, defaultValue: 70 },
  ],
  passes: [
    { id: "chromaH", label: "chromaH (彩度ぼかし H)", fragmentSource: chromaBlurPass("x", "original"), outputScale: 0.5, maxOutputLongEdge: BAND_CAP },
    { id: "chromaV", label: "chromaV (彩度ぼかし V)", fragmentSource: chromaBlurPass("y", "chain"), outputScale: 0.5, maxOutputLongEdge: BAND_CAP },
    { id: "lumaLowH", label: "lumaLowH (低周波 H)", fragmentSource: lumaBlurPass("x", "original", BAND_LOW_RADIUS), outputScale: 0.5, maxOutputLongEdge: BAND_CAP },
    { id: "lumaLowV", label: "lumaLowV (低周波 V)", fragmentSource: lumaBlurPass("y", "chain", BAND_LOW_RADIUS), outputScale: 0.5, maxOutputLongEdge: BAND_CAP },
    { id: "lumaMidH", label: "lumaMidH (中間帯 H)", fragmentSource: lumaBlurPass("x", "original", BAND_MID_RADIUS), maxOutputLongEdge: 2048 },
    { id: "lumaMidV", label: "lumaMidV (中間帯 V)", fragmentSource: lumaBlurPass("y", "chain", BAND_MID_RADIUS), maxOutputLongEdge: 2048 },
    { id: "gfCoef", label: "gfCoef (ガイデッド係数)", fragmentSource: guidedCoefPass(GF_RADIUS, GF_EPS), outputScale: 0.5, maxOutputLongEdge: BAND_CAP },
    { id: "gfMean", label: "gfMean (ガイデッド平均)", fragmentSource: guidedMeanPass(GF_RADIUS), outputScale: 0.5, maxOutputLongEdge: BAND_CAP },
    { id: "shineMin", label: "shineMin (テカリ収縮)", fragmentSource: shineOpenPass("min"), outputScale: 0.5, maxOutputLongEdge: BAND_CAP, extraInputs: ["lumaLowV"] },
    { id: "shineMax", label: "shineMax (テカリ膨張)", fragmentSource: shineOpenPass("max"), outputScale: 0.5, maxOutputLongEdge: BAND_CAP },
    { id: "tensor", label: "tensor (構造テンソル)", fragmentSource: flowTensorPass("original"), outputScale: 0.5, maxOutputLongEdge: 800 },
    { id: "tensorBlur", label: "tensorBlur (テンソル平滑)", fragmentSource: flowTensorBlurPass, outputScale: 0.5, maxOutputLongEdge: 800 },
    { id: "flow", label: "flow (流れ場)", fragmentSource: flowPass, outputScale: 0.5, maxOutputLongEdge: 800 },
    {
      id: "skin",
      label: "skin (肌・トーン)",
      fragmentSource: skinPass,
      extraInputs: ["chromaV", "lumaLowV", "lumaMidV", "gfMean", "shineMax"],
    },
    { id: "hairFlow", label: "hairFlow (髪の流れ平滑)", fragmentSource: hairFlowPass, extraInputs: ["flow"] },
    { id: "detail", label: "detail (髪・輪郭シャープ)", fragmentSource: detailPass },
  ],
  presets: [
    {
      id: "subtle",
      label: "ほんのり",
      values: { tone: 55, smooth: 55, clarity: 45, lips: 40, detail: 35, hair: 35 },
    },
    {
      id: "reference",
      label: "AIルック",
      values: { tone: 85, smooth: 80, clarity: 70, lips: 50, detail: 55, hair: 55 },
    },
    {
      id: "strong",
      label: "しっかり",
      values: { tone: 100, smooth: 92, clarity: 85, lips: 60, detail: 70, hair: 70 },
    },
  ],
};
