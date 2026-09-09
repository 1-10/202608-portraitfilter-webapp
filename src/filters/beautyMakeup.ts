import { buildFragmentShader, inputUniform, paramUniform } from "../rendering/shaders/glslCommon";
import type { FilterDefinition } from "../types";
import { chromaBlurPass, lumaBlurPass } from "./shared/skinRetouch";

/**
 * Beauty makeup: the "cute filter" of a phone camera app, built the way those are
 * actually built rather than as a stack of blurs.
 *
 * Three mechanisms, in the order they are applied:
 *
 * 1. **Mesh deformation.** A displacement field baked from the MediaPipe face mesh
 *    slims the jaw and cheeks, shortens the chin, narrows the nose and mouth, and
 *    enlarges and lifts the eyes. Blur cannot do any of this — the geometry of the
 *    face is most of what separates "retouched photo" from "beauty filter".
 * 2. **Skin.** The three-band frequency split shared with the natural retouch
 *    filter, driven much harder, plus a brighter and pinker complexion and
 *    landmark-placed highlights under the eyes, along the nose and on the forehead.
 * 3. **Makeup.** Lip colour and gloss, blush, eyeshadow, lashes and liner, brows
 *    and an iris catchlight, all rasterized from the landmarks into coverage masks
 *    and coloured here.
 *
 * Hair gloss and a background defocus finish the look, because both move in the
 * same direction across the reference ladder this was matched against.
 *
 * WHERE THE WORK HAPPENS: everything face-shaped is baked to three normalized-UV
 * textures once per image (see vision/faceGeometry.ts). Every slider on this filter
 * is a uniform weight over those textures, so dragging one costs a single re-render
 * and no CPU geometry work at all.
 *
 * WITHOUT A FACE: uHasFaceGeometry is 0, the deformation and every makeup layer
 * collapse to nothing, and what remains is a plain skin-and-tone filter. No error,
 * no exception — a landscape photo is a valid input.
 */

/**
 * Band radii, as fractions of face width. See shared/skinRetouch.ts.
 *
 * Both are far narrower than the natural retouch filter's, and that is the whole
 * reason this filter can be pushed to a porcelain finish without flattening the
 * face: the attenuated band spans 0.008-0.030 of face width, which is pores and
 * small blotches. The nose, the alar crease and the chin are all wider than that,
 * so they stay in the untouched low band. Large patches of redness are wider too —
 * those are removed from CHROMA, where taking them out costs no form at all.
 */
const BAND_LOW_RADIUS = 0.03;
const BAND_MID_RADIUS = 0.008;

/**
 * The deformed lookup, recomputed identically by every pass after the warp.
 *
 * Shared as one string rather than retyped: the face mask and the makeup masks were
 * baked against the UNDEFORMED landmarks, so any pass that disagrees with the warp
 * pass about the displacement slides the lipstick off the lip by the difference.
 *
 * `uStrength` is folded in HERE and nowhere else in the colour work. A deformation
 * cannot be cross-faded — mixing a deformed frame with an undeformed one ghosts
 * every edge — so the strength slider has to scale the displacement itself, and
 * the mask lookups have to follow it.
 */
const WARP_UV = `
  float warpShape = ${paramUniform("shape")} / 100.0 * uStrength;
  float warpEyes = ${paramUniform("eyes")} / 100.0 * uStrength;
  vec2 srcUv = faceWarpedUv(vUv, warpShape, warpEyes);`;

const WARP_UNIFORMS = `uniform float ${paramUniform("shape")};
uniform float ${paramUniform("eyes")};`;

/**
 * Pass 1: resample the original through the displacement field, and restore the
 * resolution that magnifying it costs.
 *
 * Magnification IS a blur: an enlarged eye is drawn from fewer source pixels than
 * the output has room for, and a single bilinear tap cannot invent the difference.
 * The result is a soft patch surrounded by sharp skin, and that mismatch is most of
 * what makes a warped face look handled rather than photographed — the deformation
 * itself is not the tell, the softness it leaves behind is.
 *
 * So the local magnification is measured, not guessed: the Jacobian determinant of
 * the backward map says exactly how much source area this output pixel is drawn
 * from, and an unsharp mask is applied in proportion. Where nothing is magnified the
 * determinant is 1 and this costs nothing but the taps.
 */
const warpPass = buildFragmentShader({
  extraUniforms: WARP_UNIFORMS,
  body: `${WARP_UV}
  vec2 texel = uOriginalTexelSize;
  vec2 ddx = faceWarpedUv(vUv + vec2(texel.x, 0.0), warpShape, warpEyes) - srcUv;
  vec2 ddy = faceWarpedUv(vUv + vec2(0.0, texel.y), warpShape, warpEyes) - srcUv;
  float jacobian = abs((ddx.x / texel.x) * (ddy.y / texel.y) - (ddx.y / texel.y) * (ddy.x / texel.x));
  // Below 1 the output pixel is drawn from less than a pixel of source: magnified.
  // Clamped so a fold in the field cannot ask for an unbounded amount of sharpening.
  float magnify = clamp(1.0 - jacobian, 0.0, 0.5);

  vec3 centre = texture(uOriginal, srcUv).rgb;
  vec3 neighbourhood = (
    texture(uOriginal, srcUv + vec2(texel.x, 0.0)).rgb +
    texture(uOriginal, srcUv - vec2(texel.x, 0.0)).rgb +
    texture(uOriginal, srcUv + vec2(0.0, texel.y)).rgb +
    texture(uOriginal, srcUv - vec2(0.0, texel.y)).rgb) * 0.25;
  fragColor = vec4(clamp(centre + (centre - neighbourhood) * magnify * 2.2, 0.0, 1.0), 1.0);`,
});

/**
 * Pass 8: recombine the frequency bands into porcelain skin.
 *
 * This differs from the natural retouch filter in one deliberate way: the PORE band
 * is attenuated too, not just the blotch band. Keeping pores is what makes the
 * natural filter look like skin; removing them is what makes this one look like the
 * reference. That trade is exactly what the "肌の仕上がり" slider chooses.
 */
const skinPass = buildFragmentShader({
  extraUniforms: `${WARP_UNIFORMS}
uniform sampler2D ${inputUniform("warp")};
uniform sampler2D ${inputUniform("chromaV")};
uniform sampler2D ${inputUniform("lumaLowV")};
uniform sampler2D ${inputUniform("lumaMidV")};
uniform float ${paramUniform("skin")};
uniform float ${paramUniform("texture")};`,
  body: `${WARP_UV}
  vec3 srcSrgb = texture(${inputUniform("warp")}, vUv).rgb;
  vec3 srcLin = srgbToLinear(srcSrgb);
  vec3 lab = linearRgbToOklab(srcLin);

  float Llow = texture(${inputUniform("lumaLowV")}, vUv).r;
  float Lmid = texture(${inputUniform("lumaMidV")}, vUv).r;
  vec2 abLow = texture(${inputUniform("chromaV")}, vUv).xy * 0.6 - 0.3;

  float blemish = Lmid - Llow;
  float pore = lab.x - Lmid;
  float amount = ${paramUniform("skin")} / 100.0;

  // Three multiplied gates, all three needed: is this skin, is it a feature that
  // must stay sharp, and is it a structural boundary. Dropping the last one is what
  // produces the classic wax mask with a glowing rim.
  //
  // The BROAD skin weight, so the treatment reaches the neck, chest and arms and
  // does not stop at the jaw — see glslCommon.ts for why that step is the worst
  // artifact available to this filter. It is built on the strict weight, which is
  // itself load-bearing: with the plain region weight the face oval alone counts as
  // skin, so the fringe of hair over the forehead gets its texture crushed and its
  // chroma replaced by the surrounding skin's, and black bangs come out beige.
  float region = faceSkinWeightBroad(srcLin, srcUv) * (1.0 - faceFeatureWeight(srcUv));
  // The edge term is much weaker than in the natural retouch filter, on purpose.
  // Nasolabial folds, under-eye creases and the shadow beside the nose ARE edges,
  // and they are precisely what the reference ladder removes; a gate tight enough
  // to protect a lock of hair over a cheek also protects every fold in the face.
  // What keeps hair safe here is the chroma term above, not this one.
  float edge = sobelEdge01(${inputUniform("warp")}, vUv, uOriginalTexelSize);
  float gate = clamp(region * (1.0 - smoothstep(0.2, 0.6, edge)), 0.0, 1.0);

  // Attenuate each band, then CLAMP how far the pixel is allowed to move.
  //
  // The clamp is what keeps the face from turning into a balloon. A blotch is a
  // small deviation from the local mean (a few hundredths of L); a nostril, the
  // shadow beside the nose and the mouth line are large ones, and they land in the
  // same band because the band is just "narrow blur minus wide blur". Scaling the
  // band by a constant therefore erases the nostril in the act of erasing the
  // blotch — with no edge gate tight enough to tell them apart, since the interior
  // of a nostril is not an edge. Capping the correction instead lets an unlimited
  // amount of blotch removal coexist with facial structure that survives intact.
  // The blotch band is always crushed. How much of the PORE band survives is the
  // user's call, not a baked-in one: keeping pores is what makes skin read as skin,
  // removing them is what makes it read as porcelain, and which of those is wanted
  // is a matter of taste rather than something this code can decide. At 100 the pore
  // band passes through untouched and the filter never produces wax.
  float keepTexture = ${paramUniform("texture")} / 100.0;
  float kBlemish = mix(1.0, 0.06, amount * gate);
  float kPore = mix(mix(1.0, 0.3, amount * gate), 1.0, keepTexture);
  float blemishShift = clamp(blemish * kBlemish - blemish, -0.035, 0.035);
  float poreShift = clamp(pore * kPore - pore, -0.03, 0.03);
  float L2 = Llow + blemish + blemishShift + pore + poreShift;

  // Skin CONTRAST, expanded about this subject's own measured skin lightness.
  //
  // Measured: the reference raises skin luminance std from 0.094 to 0.101-0.115,
  // while this pipeline was lowering it to 0.085. Two of its own operations do that
  // — the lift below is weighted by (1 - L), so bright pixels get less of it and the
  // top end packs together, and the glow is screened on, which raises darks more
  // than highlights. Both are right in themselves and both cost contrast.
  //
  // Flat and bright reads as BLOWN OUT even when the peak values are lower than the
  // reference's: measured, this filter's skin peaks below all three reference images
  // (p90 0.828 against 0.851-0.883) and still looked more blown than they do. So the
  // correction is contrast, not exposure.
  //
  // Applied here, before the lift, so the pivot is still the lightness that was
  // actually measured on the source.
  L2 = mix(L2, uFaceSkinLightness + (L2 - uFaceSkinLightness) * 1.55, amount * region);

  // Chroma is attacked hardest of all: blotchy redness IS chroma variance, and the
  // eye cannot resolve chroma detail at this scale, so it costs no sharpness.
  //
  vec2 ab2 = mix(lab.yz, abLow, 0.8 * amount * gate);

  // Specular shine is bright AND desaturated, which is what separates it from a
  // simply light complexion.
  float chroma = length(ab2);
  float spec = smoothstep(0.70, 0.94, L2) * (1.0 - smoothstep(0.04, 0.10, chroma)) * gate;
  L2 -= spec * amount * 0.16;

  // Complexion: a lift and a desaturation, neither of which is a variance change.
  // A filter that only evens the skin out leaves it the same dull colour it was, and
  // that is what "no processing feel" looks like.
  //
  // Measured on the reference ladder against the source, skin sits at OKLab L 0.710
  // before and 0.781 / 0.788 / 0.767 after, with chroma 0.056 before and
  // 0.047 / 0.040 / 0.054 after. Two things follow, and both are load-bearing here:
  //
  //   1. The lift is LARGE — about +0.07 — and an order of magnitude bigger than
  //      what evening the skin out contributes on its own.
  //   2. It is essentially the SAME at all three strengths. Brightness is not the
  //      strength axis in the reference at all; the ladder is carried by geometry
  //      and makeup. So the presets keep this high throughout and vary the rest.
  //
  // Neither term is gated by the face REGION, and that is deliberate. The oval mask
  // has a feather of a few percent of face width: invisible when it gates a texture
  // change, and a hard seam down the cheek and along the jaw when it gates a tone
  // change of this size — which is exactly what it drew. A tone shift has to be
  // either far more softly bounded than a mask can be, or not bounded at all. Not
  // bounded is also closer to the target: across the ladder the whole frame
  // brightens, the wall behind the subject included.
  //
  // Desaturation follows the colour heuristic, so it lands on every piece of skin in
  // frame (neck and chest included) rather than stopping at the jaw.
  float toneSkin = skinLikelihood(srcLin);
  ab2 *= 1.0 - amount * toneSkin * 0.18;
  ab2 += vec2(0.003, -0.002) * amount * toneSkin;

  // The lift is ungated, and shaped so it opens midtones and highlights while
  // leaving the darks alone.
  //
  // Not a gamma curve and not an add weighted by (1 - L), both of which were tried:
  // a gamma lifts L 0.25 to 0.37, which turns black hair grey, and a plain (1 - L)
  // weighting gives the DARKEST pixels the most lift, which inverted the tone
  // relationships it was meant to preserve — the neck came out brighter than the
  // face. The smoothstep floor pins the darks and has to sit ABOVE HAIR, not merely
  // above black: this subject's fringe is brown at L 0.30-0.40, and a floor at 0.30
  // lifted it into grey. (1 - L) keeps the highlights from clipping, and the result
  // is still monotone, so nothing swaps order.
  //
  // Eyes, brows and lips are held back from it. Measured, the reference lifts skin
  // from L 0.710 to 0.788 and leaves the lip at 0.62-0.64: it brightens the
  // complexion, not the features. Without this the lip came out at 0.72, which reads
  // as a pale washed-out mouth however well its colour is matched. Excluding a region
  // from a tone lift is safe here in a way it is not on the cheek, because these
  // regions have real edges of their own for the transition to hide in.
  // The brows have to be held back as well, and they are NOT in faceFeatureWeight:
  // the region mask's channels are skin, eyes, lips and oval, with the brows only
  // subtracted OUT of the skin channel. Lifting them turned them thin and grey. The
  // makeup masks already carry a brow coverage plane, so it doubles as the hold.
  float browRegion = sampleGeom(uMakeupB, faceGeomUv(srcUv)).g;
  float featureHold = 1.0 - 0.85 * clamp(max(faceFeatureWeight(srcUv), browRegion), 0.0, 1.0);
  // 0.26, not 0.34. Three separate things brighten the frame — this lift, the glow
  // screened on in the final pass, and the background compression — and stacking
  // them at full strength took the whole image up by +0.12 in mean L where the
  // reference moves +0.05 to +0.07. Nothing clips, but the frame goes flat and
  // washed, which is what "blown out" looks like before it becomes clipping.
  L2 += 0.26 * amount * smoothstep(0.42, 0.62, L2) * (1.0 - L2) * featureHold;

  // Contouring, from the landmark-placed signed map: highlights on the nose bridge,
  // cheekbones, under-eye, forehead, chin and cupid's bow; shadows under the
  // cheekbones, at the temples, either side of the nose bridge and along the jaw.
  //
  // This is where the reference ladder's luminosity comes from. Evening the skin out
  // and lifting it removes the modelling that made the face look three-dimensional,
  // and the result is a flat chalky mask however bright it is; putting shaped light
  // and shadow back is what a beauty filter actually does.
  //
  // Gated by the region, NOT by the edge-narrowed gate: the under-eye highlight's
  // whole purpose is to fill the crease under the eye, and a crease is an edge, so
  // the edge term would switch it off exactly where it is needed. Note the map is
  // sampled with a 0.5 default outside the face rect, so it is neutral there.
  float contour = sampleGeom(uMakeupB, faceGeomUv(srcUv)).b;
  contour = insideGeom(faceGeomUv(srcUv)) ? (contour - 128.0 / 255.0) * 2.0 : 0.0;
  // Shadow at under half the weight of highlight. Painted-on shading is far more
  // conspicuous than painted-on light — a highlight that lands slightly wrong looks
  // like sheen, a shadow that lands slightly wrong looks like dirt.
  L2 += (max(contour, 0.0) * 0.1 + min(contour, 0.0) * 0.055) * amount * region;

  vec3 outLin = max(oklabToLinearRgb(vec3(L2, ab2)), 0.0);
  fragColor = vec4(linearToSrgb(clamp(outLin, 0.0, 1.0)), 1.0);`,
});

/**
 * Pass 9: hair gloss.
 *
 * Strand highlights are smeared ALONG the strand, so the sample line is the
 * perpendicular of the local luminance gradient — the same "flow" idea the
 * rotoscope filter uses, reduced to one pass. Taking the maximum lightness along
 * that line and adding the excess extends existing highlights into streaks instead
 * of inventing them.
 *
 * Hair is identified by four multiplied terms, because no one of them is specific:
 * dark, not skin, near the head, and locally textured. The texture term is what
 * keeps flat dark clothing out of it.
 */
const hairPass = buildFragmentShader({
  extraUniforms: `${WARP_UNIFORMS}
uniform float ${paramUniform("hair")};`,
  body: `${WARP_UV}
  vec3 baseSrgb = texture(uSource, vUv).rgb;
  vec3 lin = srgbToLinear(baseSrgb);
  vec3 lab = linearRgbToOklab(lin);
  float amount = ${paramUniform("hair")} / 100.0;

  // Wide enough to include the lighter brown at the crown and the roots, which is
  // where this subject's colour unevenness actually lives. Skin is excluded by the
  // notSkin term below, not by this one.
  float dark = 1.0 - smoothstep(0.3, 0.7, lab.x);
  float notSkin = 1.0 - faceSkinWeightBroad(lin, srcUv);
  // Loose texture gate. It exists to keep flat dark CLOTHING out, and the head
  // proximity term already does most of that job — set tight, it also excluded the
  // smooth top of the head and left the treatment patchy.
  float textured = smoothstep(0.01, 0.1, sobelEdge01(uSource, vUv, uTexelSize));

  // Distance from the face, in face widths, measured in aspect-corrected UV so the
  // radius is a circle rather than an ellipse on a portrait-format photo. The rect
  // is the face oval grown by the skirt and margin, hence the constant.
  float aspect = uResolution.x / max(uResolution.y, 1.0);
  vec2 rectCentre = uFaceGeomRect.xy + uFaceGeomRect.zw * 0.5;
  vec2 offset = (vUv - rectCentre) * vec2(aspect, 1.0);
  float faceW = max(uFaceGeomRect.z * aspect / 1.54, 1.0e-4);
  float headProximity = 1.0 - smoothstep(0.75, 1.7, length(offset) / faceW);
  // With no detection there is no head to be near, so the term is dropped rather
  // than guessed: what is left is "dark and textured", which is a fair description
  // of hair and a harmless one to brighten slightly.
  float nearHead = mix(1.0, headProximity, uHasFaceGeometry);

  float gx = luminance(texture(uSource, vUv + vec2(uTexelSize.x, 0.0)).rgb)
           - luminance(texture(uSource, vUv - vec2(uTexelSize.x, 0.0)).rgb);
  float gy = luminance(texture(uSource, vUv + vec2(0.0, uTexelSize.y)).rgb)
           - luminance(texture(uSource, vUv - vec2(0.0, uTexelSize.y)).rgb);
  // Perpendicular to the gradient is along the strand. Where the gradient vanishes
  // the direction is meaningless, so fall back to vertical — the way hair falls.
  vec2 dir = length(vec2(gx, gy)) > 1.0e-5 ? normalize(vec2(-gy, gx)) : vec2(0.0, 1.0);

  float faceWidthPx = max(uFaceScale * uResolution.x * 2.5, 24.0);
  float stepPx = max(faceWidthPx * 0.055, 3.0) / 4.0;
  float peak = lab.x;
  for (int i = -4; i <= 4; i++) {
    vec2 uv = vUv + dir * (float(i) * stepPx) * uTexelSize;
    peak = max(peak, linearRgbToOklab(srgbToLinear(texture(uSource, uv).rgb)).x);
  }

  float hairW = clamp(dark * notSkin * nearHead * textured, 0.0, 1.0) * amount;

  // Uniform, glossy, near-black — which is what the reference actually does to hair.
  // Boosting the sheen alone left this subject's patchy brown roots and uneven
  // colour in place, and that unevenness is a large part of why the result read as
  // "the same photo, retouched" next to a reference that had redrawn the hair.
  //
  // So the hair's own lightness variation is COMPRESSED toward a deep value and the
  // sheen is added back on top: the patchiness goes, the highlights stay. Chroma is
  // cut hard, because brown roots are a chroma signal.
  // Capped: at the hairline the strand direction can run along the hair/skin
  // boundary, so the brightest tap is skin rather than a highlight. Without the cap
  // that lifts the whole fringe to skin brightness.
  float sheen = min(max(0.0, peak - lab.x), 0.1);
  float L2 = mix(lab.x, mix(0.2, lab.x, 0.5), hairW) + sheen * hairW * 1.4;
  // A little extra contrast about the hair's own midtone, so the gloss reads as
  // separated strands rather than an overall haze.
  L2 += (lab.x - 0.32) * 0.45 * hairW;
  vec2 ab2 = lab.yz * (1.0 - hairW * 0.85);

  vec3 outLin = max(oklabToLinearRgb(vec3(L2, ab2)), 0.0);
  fragColor = vec4(linearToSrgb(clamp(outLin, 0.0, 1.0)), 1.0);`,
});

/**
 * Passes 10-11: the defocused background plate, at quarter resolution.
 *
 * The blur RADIUS grows with the slider as well as the mix, because a faint amount
 * of a very large blur reads as a smeared lens rather than a shallow one.
 */
function backgroundBlurPass(axis: "x" | "y"): string {
  const offset =
    axis === "x" ? "vec2(float(i) * stepPx / uResolution.x, 0.0)" : "vec2(0.0, float(i) * stepPx / uResolution.y)";
  // The first axis reads the hair pass explicitly rather than uSource: the glow
  // plate now sits between them in the chain, and uSource would blur that instead.
  const source = axis === "x" ? inputUniform("hair") : "uSource";
  return buildFragmentShader({
    extraUniforms:
      axis === "x"
        ? `uniform sampler2D ${inputUniform("hair")};
uniform float ${paramUniform("bokeh")};`
        : `uniform float ${paramUniform("bokeh")};`,
    body: `
  float faceWidthPx = max(uFaceScale * uResolution.x * 2.5, 12.0);
  float sigmaPx = max(faceWidthPx * (0.012 + 0.03 * ${paramUniform("bokeh")} / 100.0), 1.0);
  float stepPx = sigmaPx * 0.5;
  vec3 sum = vec3(0.0);
  float wsum = 0.0;
  for (int i = -6; i <= 6; i++) {
    float w = gaussW(float(i), 2.0);
    sum += texture(${source}, vUv + ${offset}).rgb * w;
    wsum += w;
  }
  fragColor = vec4(sum / wsum, 1.0);`,
  });
}

/**
 * The glow plate: the skin's own highlights, blurred wide, to be screened back on.
 *
 * This is the piece that was missing, and measurement is what found it. The
 * reference ladder barely changes the face's PROPORTIONS at all — measured against
 * the source, face width/height goes 0.857 to 0.868-0.876 (very slightly WIDER),
 * eye width 0.200 to 0.203-0.207 of face width, eye height 0.071 to 0.068. Nothing
 * is slimmed and nothing is enlarged. The only real geometric change is the lip,
 * about 8% thicker.
 *
 * So the "smaller face, bigger eyes" read that a beauty filter gives is not
 * geometry: it is the crop, the lash line, the contouring and the light. And the
 * light is diffuse — skin lit from inside, highlights bleeding into their
 * surroundings. A frequency-band retouch cannot produce that however hard it is
 * pushed, because it only ever removes; a glow has to be ADDED.
 *
 * Highlight-weighted rather than a plain blur, so it blooms the lit side of the
 * face and leaves the shadows alone — a flat blur screened back is just a haze.
 */
function glowPass(axis: "x" | "y" | "cut"): string {
  if (axis === "cut") {
    return buildFragmentShader({
      extraUniforms: `${WARP_UNIFORMS}
uniform sampler2D ${inputUniform("skin")};`,
      body: `${WARP_UV}
  vec3 srgb = texture(${inputUniform("skin")}, vUv).rgb;
  vec3 lin = srgbToLinear(srgb);
  float L = linearRgbToOklab(lin).x;
  // Skin only, and only its brighter half. Blooming hair or the background would
  // read as a lens fault rather than as complexion.
  float weight = faceSkinWeightBroad(lin, srcUv) * smoothstep(0.55, 0.92, L);
  fragColor = vec4(srgb * weight, 1.0);`,
    });
  }
  const offset =
    axis === "x" ? "vec2(float(i) * stepPx / uResolution.x, 0.0)" : "vec2(0.0, float(i) * stepPx / uResolution.y)";
  return buildFragmentShader({
    body: `
  // Wide: the radius is a fraction of FACE WIDTH, so the bloom spreads over the
  // cheek rather than over a few pixels, which is the difference between a glow and
  // a soft-focus smear.
  float faceWidthPx = max(uFaceScale * uResolution.x * 2.5, 12.0);
  float sigmaPx = max(faceWidthPx * 0.09, 1.0);
  float stepPx = sigmaPx * 0.5;
  vec3 sum = vec3(0.0);
  float wsum = 0.0;
  for (int i = -6; i <= 6; i++) {
    float w = gaussW(float(i), 2.0);
    sum += texture(uSource, vUv + ${offset}).rgb * w;
    wsum += w;
  }
  fragColor = vec4(sum / wsum, 1.0);`,
  });
}

/**
 * Pass 12: makeup composite, background mix and the strength cross-fade.
 *
 * Makeup is applied in OKLab, and each layer touches only the axes it physically
 * would: lipstick and blush are chroma, lashes and brows are lightness, gloss and
 * catchlight are added light. Nothing here replaces lightness wholesale, which is
 * the single most common way filter makeup ends up looking pasted on.
 */
const finishPass = buildFragmentShader({
  extraUniforms: `${WARP_UNIFORMS}
uniform sampler2D ${inputUniform("warp")};
uniform sampler2D ${inputUniform("hair")};
uniform sampler2D ${inputUniform("bgV")};
uniform sampler2D ${inputUniform("glowV")};
uniform float ${paramUniform("makeup")};
uniform float ${paramUniform("lash")};
uniform float ${paramUniform("glow")};
uniform float ${paramUniform("bokeh")};`,
  body: `${WARP_UV}
  vec3 baseSrgb = texture(${inputUniform("hair")}, vUv).rgb;
  vec2 geomUv = faceGeomUv(srcUv);
  vec4 mA = sampleGeom(uMakeupA, geomUv);
  vec4 mB = sampleGeom(uMakeupB, geomUv);

  float makeupAmount = ${paramUniform("makeup")} / 100.0;
  float lashAmount = ${paramUniform("lash")} / 100.0;

  vec3 lin = srgbToLinear(baseSrgb);
  vec3 lab = linearRgbToOklab(lin);
  float L = lab.x;
  vec2 ab = lab.yz;

  // Strict for the blush, which would otherwise tint a lock of hair over the cheek.
  float skinW = faceSkinWeightBroad(lin, srcUv);
  float eyeW = texture(uFaceMask, srcUv).g * uHasFaceMask;

  float blush = mA.b * makeupAmount * skinW;
  ab = mix(ab, vec2(0.115, 0.035), blush * 0.22);
  L += blush * 0.012;

  // Eyeshadow: a warm shade that darkens, kept off the eyeball itself. Deliberately
  // light — in the references the visible dark band above the eye is the LASH line,
  // and a heavy shadow here reads as a smudge rather than makeup.
  float shadow = mA.a * makeupAmount * (1.0 - eyeW);
  ab = mix(ab, vec2(0.050, 0.035), shadow * 0.30);
  L *= 1.0 - shadow * 0.06;

  // Rotate the lip's own chroma toward the lipstick hue and scale its magnitude,
  // rather than replacing a/b with a constant. Replacing them flattens the lip into
  // one slab of colour and takes the vermilion border and the inner shadow with it;
  // rotating keeps every variation the lip already had.
  float lip = mA.r * makeupAmount * 0.62;
  float lipChroma = length(ab);
  vec2 lipDir = lipChroma > 1.0e-4 ? ab / lipChroma : vec2(1.0, 0.0);
  vec2 lipTarget = normalize(vec2(0.115, 0.062));
  ab = normalize(mix(lipDir, lipTarget, lip)) * lipChroma * (1.0 + lip * 0.26);
  L *= 1.0 - lip * 0.05;
  float gloss = mA.g * makeupAmount;
  L += gloss * 0.4 * (1.0 - L);

  float lash = mB.r * lashAmount;
  L = mix(L, L * 0.25, lash * 0.8);
  ab *= 1.0 - lash * 0.75;

  float brow = mB.g * lashAmount;
  L = mix(L, L * 0.72, brow * 0.5);

  // Eye clarity, driven by the ENLARGEMENT rather than by the makeup: magnifying the
  // eye resamples it, and magnification is a blur. Without putting the contrast back
  // the iris comes out milky, while the references keep it dark and crisp — and a
  // catchlight only reads against a dark iris.
  float eyeClarity = eyeW * warpEyes;
  L = mix(L, clamp((L - 0.45) * 1.28 + 0.45, 0.0, 1.0), eyeClarity * 0.42);
  // Deepen the iris. A large dark iris with a hard limbal edge is most of what makes
  // the reference eye read as bigger; the pupil and iris are the dark half of the
  // eye region, so pushing the darks down inside the eye mask does it without any
  // extra geometry. Chroma comes down with it, which is what keeps a brown iris from
  // going orange as it darkens.
  float irisDark = eyeW * (1.0 - smoothstep(0.18, 0.5, L)) * lashAmount;
  L *= 1.0 - irisDark * 0.35;
  ab *= 1.0 - irisDark * 0.4;

  float catchlight = mB.a * lashAmount * eyeW;
  L = mix(L, 1.0, catchlight * 0.95);
  ab *= 1.0 - catchlight * 0.9;

  vec3 styled = linearToSrgb(clamp(max(oklabToLinearRgb(vec3(L, ab)), 0.0), 0.0, 1.0));

  // Glow, screened on. Screen rather than add: it cannot clip, and it lifts the
  // midtones more than the highlights, which is what makes skin read as luminous
  // instead of blown out.
  vec3 glow = texture(${inputUniform("glowV")}, vUv).rgb * (${paramUniform("glow")} / 100.0) * 0.6;
  styled = 1.0 - (1.0 - styled) * (1.0 - clamp(glow, 0.0, 1.0));

  // Background defocus by distance from the face, with the detected face oval
  // subtracted so the subject is never blurred. There is no person segmentation
  // here, so the neck and shoulders soften along with the background — which the
  // reference ladder also does, and which is the honest limit of this approach.
  float aspect = uResolution.x / max(uResolution.y, 1.0);
  vec2 rectCentre = uFaceGeomRect.xy + uFaceGeomRect.zw * 0.5;
  vec2 offset = (vUv - rectCentre) * vec2(aspect, 1.0);
  float faceW = max(uFaceGeomRect.z * aspect / 1.54, 1.0e-4);
  // Starts further out and ramps more slowly than a lens would. With no person
  // segmentation this is a radial falloff, and a radial falloff that ramps quickly
  // reads as a smeared vignette rather than depth of field — it also destroys
  // whatever happens to sit near the frame edge, a ceiling light in this case.
  float far = smoothstep(1.6, 3.4, length(offset) / faceW);
  far *= 1.0 - texture(uFaceMask, srcUv).a * uHasFaceMask;
  // Gated on detection: with no face there is no subject to isolate, and blurring
  // outward from the frame centre would just be wrong.
  float bokeh = far * (${paramUniform("bokeh")} / 100.0) * uHasFaceGeometry;

  // The plate is lifted and desaturated as well as blurred. Across the reference
  // ladder the background goes creamy, not just soft, and blur alone reads as a
  // smeared lens rather than a shallow depth of field.
  vec3 bgLab = linearRgbToOklab(srgbToLinear(texture(${inputUniform("bgV")}, vUv).rgb));
  bgLab.x += 0.05 * (1.0 - bgLab.x);
  bgLab.yz *= 0.82;
  vec3 bgSrgb = linearToSrgb(clamp(max(oklabToLinearRgb(bgLab), 0.0), 0.0, 1.0));
  styled = mix(styled, bgSrgb, bokeh);

  // Cross-fade against the WARPED frame, not the original: the deformation is
  // already scaled by uStrength inside the warp pass, so mixing back to the
  // original here would ghost every deformed edge.
  // Background LIFT, separate from the defocus and much wider-reaching. The
  // reference takes the wall to near-white; blurring alone leaves it the same dull
  // beige it was, and that difference reads immediately at head scale.
  //
  // Gated on "not skin AND already bright": hair is dark so it is excluded by the
  // brightness term rather than by a region, and the face is excluded by the skin
  // term. No radial falloff here — a brightness change does not need one, and the
  // radial term the defocus uses would leave a dark halo hugging the hair.
  // Compressed TOWARD white rather than lifted proportionally. A lift weighted by
  // brightness raises the lit wall more than the shadow on it, which amplifies the
  // very gradient it was supposed to remove — the reference wall is not just
  // brighter, it is flat. Pulling every background pixel a fixed fraction of the way
  // to near-white flattens it instead.
  //
  // The floor sits at L 0.30-0.50 so hair (0.20-0.40) and the dark dress are left
  // alone while the wall shadow (~0.55) is fully included.
  float bgAmount = ${paramUniform("bokeh")} / 100.0;
  vec3 styledLab = linearRgbToOklab(srgbToLinear(styled));
  float bgLift = (1.0 - skinW) * smoothstep(0.3, 0.5, styledLab.x) * (1.0 - faceRegionWeight(srcUv)) * bgAmount;
  styledLab.x += bgLift * (0.88 - styledLab.x) * 0.32;
  styledLab.yz *= 1.0 - bgLift * 0.5;
  styled = linearToSrgb(clamp(max(oklabToLinearRgb(styledLab), 0.0), 0.0, 1.0));

  vec3 warped = texture(${inputUniform("warp")}, vUv).rgb;
  vec3 result = mix(warped, styled, uStrength);
  fragColor = vec4(result + orderedDither(vUv * uResolution), 1.0);`,
});

/**
 * Cap for the frequency-separation passes.
 *
 * These carry low-frequency bands whose radii are face-relative, so running them at
 * a lower resolution changes precision rather than appearance — the cheapest place
 * to buy back the cost of a twelve-pass pipeline at a 4096px export.
 */
const BAND_CAP = 1600;

export const beautyMakeupFilter: FilterDefinition = {
  id: "beauty-makeup",
  name: "ビューティーメイク",
  description:
    "顔ランドマークから三角形メッシュを張り、頂点を動かして小顔・目もとを整え、UV上に描いたリップ・チーク・アイシャドウ・まつ毛・キャッチライトを合成します。肌はつるんと仕上げ、髪にツヤ、背景に軽いぼかしを加えます。",
  parameters: [
    { id: "shape", label: "顎のライン", min: 0, max: 100, step: 1, defaultValue: 25 },
    { id: "eyes", label: "目もと", min: 0, max: 100, step: 1, defaultValue: 6 },
    { id: "skin", label: "肌の均し", min: 0, max: 100, step: 1, defaultValue: 62 },
    { id: "glow", label: "ツヤ・透明感", min: 0, max: 100, step: 1, defaultValue: 34 },
    { id: "texture", label: "質感キープ", min: 0, max: 100, step: 1, defaultValue: 55 },
    { id: "makeup", label: "メイクの濃さ", min: 0, max: 100, step: 1, defaultValue: 18 },
    { id: "lash", label: "まつ毛・アイライン", min: 0, max: 100, step: 1, defaultValue: 15 },
    { id: "hair", label: "髪の艶", min: 0, max: 100, step: 1, defaultValue: 55 },
    { id: "bokeh", label: "背景の処理", min: 0, max: 100, step: 1, defaultValue: 25 },
  ],
  passes: [
    { id: "warp", label: "warp (メッシュ変形)", fragmentSource: warpPass },
    { id: "chromaH", label: "chromaH (彩度ぼかし H)", fragmentSource: chromaBlurPass("x", { pass: "warp" }), outputScale: 0.5, maxOutputLongEdge: BAND_CAP, extraInputs: ["warp"] },
    { id: "chromaV", label: "chromaV (彩度ぼかし V)", fragmentSource: chromaBlurPass("y", "chain"), outputScale: 0.5, maxOutputLongEdge: BAND_CAP },
    { id: "lumaLowH", label: "lumaLowH (低周波 H)", fragmentSource: lumaBlurPass("x", { pass: "warp" }, BAND_LOW_RADIUS), outputScale: 0.5, maxOutputLongEdge: BAND_CAP, extraInputs: ["warp"] },
    { id: "lumaLowV", label: "lumaLowV (低周波 V)", fragmentSource: lumaBlurPass("y", "chain", BAND_LOW_RADIUS), outputScale: 0.5, maxOutputLongEdge: BAND_CAP },
    { id: "lumaMidH", label: "lumaMidH (中間帯 H)", fragmentSource: lumaBlurPass("x", { pass: "warp" }, BAND_MID_RADIUS), maxOutputLongEdge: 2048, extraInputs: ["warp"] },
    { id: "lumaMidV", label: "lumaMidV (中間帯 V)", fragmentSource: lumaBlurPass("y", "chain", BAND_MID_RADIUS), maxOutputLongEdge: 2048 },
    {
      id: "skin",
      label: "skin (肌の再合成)",
      fragmentSource: skinPass,
      maxOutputLongEdge: 3072,
      extraInputs: ["warp", "chromaV", "lumaLowV", "lumaMidV"],
    },
    { id: "hair", label: "hair (髪のツヤ)", fragmentSource: hairPass, maxOutputLongEdge: 2048 },
    { id: "glowCut", label: "glowCut (ツヤ抽出)", fragmentSource: glowPass("cut"), outputScale: 0.25, extraInputs: ["skin"] },
    { id: "glowH", label: "glowH (ツヤ H)", fragmentSource: glowPass("x"), outputScale: 0.25 },
    { id: "glowV", label: "glowV (ツヤ V)", fragmentSource: glowPass("y"), outputScale: 0.25 },
    { id: "bgH", label: "bgH (背景ぼかし H)", fragmentSource: backgroundBlurPass("x"), outputScale: 0.25, extraInputs: ["hair"] },
    { id: "bgV", label: "bgV (背景ぼかし V)", fragmentSource: backgroundBlurPass("y"), outputScale: 0.25 },
    { id: "finish", label: "finish (メイク合成+最終)", fragmentSource: finishPass, extraInputs: ["warp", "hair", "bgV", "glowV"] },
  ],
  /**
   * The ladder is carried by GEOMETRY, and the makeup and tone are kept well below
   * it. That split is measured, not a preference.
   *
   * Swept on a real photograph with the skin and makeup held fixed, the deformation
   * survives being pushed a very long way: at shape 85 / eyes 65 the jaw is markedly
   * narrower and the eyes markedly larger, the hair silhouette still follows the
   * face, and it still reads as the same person under a strong filter. The painted
   * layers do not have that headroom — past roughly a quarter of their range the lip
   * becomes a slab, the forehead flattens into a featureless pale field and the
   * result reads as "a different person, wrongly".
   *
   * An earlier version of these presets pulled BOTH down after the strong setting was
   * judged bad, which threw away the thing a phone beauty filter mostly does. The
   * artifacts were coming from the painted side; the deformation was carrying its
   * weight. Sliders stay open to 100 either way.
   */
  presets: [
    {
      id: "subtle",
      label: "ほんのり",
      values: { shape: 12, eyes: 2, skin: 48, texture: 68, glow: 20, makeup: 10, lash: 12, hair: 25, bokeh: 12 },
    },
    {
      id: "natural",
      label: "ナチュラル",
      values: { shape: 25, eyes: 4, skin: 64, texture: 52, glow: 34, makeup: 22, lash: 24, hair: 55, bokeh: 25 },
    },
    {
      id: "polished",
      label: "しっかり",
      values: { shape: 42, eyes: 8, skin: 76, texture: 38, glow: 48, makeup: 34, lash: 34, hair: 85, bokeh: 40 },
    },
  ],
};
