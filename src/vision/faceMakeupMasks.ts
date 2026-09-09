/**
 * Rasterizes makeup as coverage masks over the face rect.
 *
 * The masks carry WHERE, never what colour: colour and intensity live in the
 * shader as uniforms. That split is what keeps the filter interactive — dragging
 * "makeup strength" changes a uniform, and nothing here re-runs. These masks are
 * rebuilt once per image, right after detection.
 *
 * Every layer is drawn as opaque white on transparent and read back as its alpha
 * channel (see auxTexture.ts for why a canvas's colour channels are not
 * trustworthy here), then the eight planes are packed into two RGBA textures:
 *
 *   makeupA   .r lip fill    .g lip gloss   .b blush   .a eyeshadow
 *   makeupB   .r lash/liner  .g brow        .b contour .a catchlight
 *
 * `.b` of makeupB is the one SIGNED channel: 128 is neutral, above it is highlight
 * and below it is shadow. See the contour block below for why.
 *
 * Softness is applied to the extracted PLANE, never with `ctx.filter`. A canvas
 * filter is applied in the current user space, so the local scale that `softBlob`
 * uses to make an ellipse would multiply the blur radius by that ellipse's size.
 *
 * Radii are fractions of FACE WIDTH throughout, so a headshot and a full-body
 * shot get the same makeup, proportionally.
 */
import { alphaPlane, blurPlane, packPlanes, type AuxTextureData, type Plane } from "./auxTexture";
import { distance, project, type FaceFrame, type GeomRect, type Pt } from "./faceMeshTopology";

/** Long edge of each makeup texture, in texels. */
export const MAKEUP_LONG_EDGE = 1024;

export type MakeupRings = {
  lipsOuter: readonly number[];
  lipsInner: readonly number[];
  brows: readonly (readonly number[])[];
  faceOval: readonly number[];
  /** Iris rings, when the model provides them. Empty falls back to the eye centre. */
  irises: readonly (readonly number[])[];
};

export type MakeupMasks = {
  a: AuxTextureData;
  b: AuxTextureData;
};

type Ctx = CanvasRenderingContext2D;

function addScaled(base: Pt, dir: Pt, amount: number): Pt {
  return { x: base.x + dir.x * amount, y: base.y + dir.y * amount };
}

/**
 * A soft blob: an ellipse aligned to the face frame, filled with a radial gradient
 * so it has no edge of its own. Everything diffuse — blush, highlights, gloss — is
 * built from these.
 */
function softBlob(
  ctx: Ctx,
  center: Pt,
  latRadius: number,
  vertRadius: number,
  frame: FaceFrame,
  midStop: number,
): void {
  if (latRadius <= 0.5 || vertRadius <= 0.5) return;
  ctx.save();
  ctx.translate(center.x, center.y);
  // transform(a, b, c, d, e, f) maps x onto (a, b) and y onto (c, d): feeding it
  // the frame's axes rotates the blob with the head.
  ctx.transform(frame.right.x, frame.right.y, frame.up.x, frame.up.y, 0, 0);
  ctx.scale(latRadius, vertRadius);
  const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(midStop, "rgba(255,255,255,0.55)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(0, 0, 1, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function fillPolygon(ctx: Ctx, points: readonly Pt[]): void {
  if (points.length < 3) return;
  ctx.beginPath();
  points.forEach((p, i) => {
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.closePath();
  ctx.fill();
}

/**
 * Coverage strip along a lid: the lid points, then the same points pushed away
 * from the lid by `thicknessAt(t)`, walked back.
 *
 * Canvas 2D has no tapered stroke, and an eyeliner that does not taper to nothing
 * at the inner corner reads as a black smudge — hence an explicit polygon.
 */
function lidStrip(
  ctx: Ctx,
  lidPoints: readonly Pt[],
  eyeCenter: Pt,
  frame: FaceFrame,
  thicknessAt: (t: number) => number,
  side: 1 | -1,
): void {
  if (lidPoints.length < 3) return;
  const back: Pt[] = [];
  ctx.beginPath();
  lidPoints.forEach((p, i) => {
    const t = i / (lidPoints.length - 1);
    // Push mostly along the face's up axis, so liner sits above the lid rather
    // than fanning out radially, with a little radial so it follows the curve.
    const radialX = p.x - eyeCenter.x;
    const radialY = p.y - eyeCenter.y;
    const radialLength = Math.hypot(radialX, radialY) || 1;
    const dirX = (frame.up.x * 0.78 + (radialX / radialLength) * 0.22) * side;
    const dirY = (frame.up.y * 0.78 + (radialY / radialLength) * 0.22) * side;
    const dirLength = Math.hypot(dirX, dirY) || 1;
    const thickness = thicknessAt(t);
    back.push({ x: p.x + (dirX / dirLength) * thickness, y: p.y + (dirY / dirLength) * thickness });
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  for (let i = back.length - 1; i >= 0; i--) ctx.lineTo(back[i]!.x, back[i]!.y);
  ctx.closePath();
  ctx.fill();
}

/** In-place `plane *= mask`, both the same length. */
function multiplyPlane(plane: Uint8ClampedArray, mask: Uint8ClampedArray): void {
  for (let i = 0; i < plane.length; i++) plane[i] = (plane[i]! * mask[i]!) / 255;
}

export function buildMakeupMasks(
  points: readonly Pt[],
  frame: FaceFrame,
  rings: MakeupRings,
  rect: GeomRect,
): MakeupMasks | null {
  const longEdge = Math.max(rect.w, rect.h);
  const width = Math.max(2, Math.round((MAKEUP_LONG_EDGE * rect.w) / longEdge));
  const height = Math.max(2, Math.round((MAKEUP_LONG_EDGE * rect.h) / longEdge));
  // One uniform scale for both axes: with separate scales, a circle around the eye
  // would come out as an ellipse whenever the rect is not square.
  const scale = width / rect.w;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;

  const local = (p: Pt): Pt => ({ x: (p.x - rect.x) * scale, y: (p.y - rect.y) * scale });
  const localRing = (ring: readonly number[]): Pt[] =>
    ring
      .map((i) => points[i])
      .filter((p): p is Pt => !!p)
      .map(local);

  const faceWidth = frame.faceWidth * scale;
  const faceHalfHeight = frame.faceHeight * 0.5 * scale;
  const pixelCount = width * height;

  /** Draws one layer and returns its coverage, softened by `blurFraction` of face width. */
  const renderLayer = (draw: (c: Ctx) => void, blurFraction: number): Uint8ClampedArray => {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#fff";
    draw(ctx);
    const plane = alphaPlane(ctx.getImageData(0, 0, width, height).data, pixelCount);
    blurPlane(plane, width, height, faceWidth * blurFraction);
    return plane;
  };

  // A soft face-oval coverage plane. Diffuse layers are multiplied by this rather
  // than clipped to the oval, because a clip is a hard cut and would draw a line
  // along the jaw.
  const ovalPlane = renderLayer((c) => fillPolygon(c, localRing(rings.faceOval)), 0.035);

  const lipsOuter = localRing(rings.lipsOuter);
  const lipsInner = localRing(rings.lipsInner);
  const lipsCenter = local(frame.lipsCenter);

  let lipLatRadius = faceWidth * 0.2;
  let lipVertRadius = faceWidth * 0.07;
  if (lipsOuter.length >= 3) {
    let minLat = Infinity;
    let maxLat = -Infinity;
    let minVert = Infinity;
    let maxVert = -Infinity;
    for (const p of lipsOuter) {
      const lat = project(p, lipsCenter, frame.right);
      const vert = project(p, lipsCenter, frame.up);
      minLat = Math.min(minLat, lat);
      maxLat = Math.max(maxLat, lat);
      minVert = Math.min(minVert, vert);
      maxVert = Math.max(maxVert, vert);
    }
    lipLatRadius = Math.max((maxLat - minLat) * 0.5, 2);
    lipVertRadius = Math.max((maxVert - minVert) * 0.5, 2);
  }

  const eyeMid = local(frame.eyeMid);
  const noseCenter = local(frame.noseCenter);
  const chin = local(frame.chin);
  const bridgeCenter = { x: (eyeMid.x + noseCenter.x) * 0.5, y: (eyeMid.y + noseCenter.y) * 0.5 };
  const bridgeLength = distance(eyeMid, noseCenter);
  const cheekOf = (eye: FaceFrame["eyes"][number]): Pt =>
    addScaled(addScaled(local(eye.center), frame.up, -faceHalfHeight * 0.4), eye.outerDir, faceWidth * 0.04);

  const lipPlane = renderLayer((c) => {
    fillPolygon(c, lipsOuter);
    if (lipsInner.length >= 3 && rings.lipsInner !== rings.lipsOuter) {
      // Excludes the mouth OPENING so an open mouth does not get its teeth
      // tinted. On a closed mouth this ring is a sliver and removes nothing.
      c.globalCompositeOperation = "destination-out";
      fillPolygon(c, lipsInner);
      c.globalCompositeOperation = "source-over";
    }
    // A soft edge, not a crisp one: lipstick has a vermilion border, but a mask that
    // stops in two pixels puts a cut-out slab of colour on the mouth.
  }, 0.02);

  const glossPlane = renderLayer((c) => {
    // A tight streak on the lower lip plus two small peaks on the upper one. Tight
    // is the point: a broad soft blob lightens the whole lip instead of catching a
    // light on it, and a pale lip reads as no lipstick at all.
    softBlob(
      c,
      addScaled(lipsCenter, frame.up, -lipVertRadius * 0.45),
      lipLatRadius * 0.26,
      lipVertRadius * 0.13,
      frame,
      0.55,
    );
    for (const side of [-1, 1]) {
      const peak = addScaled(
        addScaled(lipsCenter, frame.right, lipLatRadius * 0.26 * side),
        frame.up,
        lipVertRadius * 0.42,
      );
      softBlob(c, peak, lipLatRadius * 0.11, lipVertRadius * 0.09, frame, 0.5);
    }
  }, 0.004);
  // Keep gloss on the lip. It is added as light, and light spilling past the lip
  // border is exactly what makes a filter look pasted on.
  multiplyPlane(glossPlane, lipPlane);

  const blushPlane = renderLayer((c) => {
    for (const eye of frame.eyes) {
      softBlob(c, cheekOf(eye), faceWidth * 0.15, faceWidth * 0.11, frame, 0.28);
    }
  }, 0.02);
  multiplyPlane(blushPlane, ovalPlane);

  const eyeshadowPlane = renderLayer((c) => {
    for (const eye of frame.eyes) {
      const lid = eye.upperLid
        .map((i) => points[i])
        .filter((p): p is Pt => !!p)
        .map(local);
      const radius = eye.radius * scale;
      const center = local(eye.center);
      lidStrip(
        c,
        lid,
        center,
        frame,
        // Widest just outside the middle of the lid, closing to nothing at both
        // corners so the crescent has no blunt ends.
        (t) => radius * 0.5 * Math.pow(Math.sin(Math.PI * t), 0.7) * (0.55 + 0.75 * t),
        1,
      );
      // Carry the shadow a little past the outer corner.
      const outer = local(eye.outerCorner);
      softBlob(
        c,
        addScaled(addScaled(outer, eye.outerDir, radius * 0.2), frame.up, radius * 0.28),
        radius * 0.34,
        radius * 0.26,
        frame,
        0.3,
      );
    }
  }, 0.03);

  const lashPlane = renderLayer((c) => {
    for (const eye of frame.eyes) {
      const radius = eye.radius * scale;
      const center = local(eye.center);
      const upper = eye.upperLid
        .map((i) => points[i])
        .filter((p): p is Pt => !!p)
        .map(local);
      const lower = eye.lowerLid
        .map((i) => points[i])
        .filter((p): p is Pt => !!p)
        .map(local);

      if (upper.length >= 3) {
        lidStrip(
          c,
          upper,
          center,
          frame,
          (t) => radius * 0.17 * (0.18 + 0.82 * Math.pow(Math.sin(Math.PI * t), 0.45)) * (0.7 + 0.6 * t),
          1,
        );
        // The wing: a triangle continuing the lid tangent past the outer corner.
        const outer = local(eye.outerCorner);
        const tangentFrom = upper[Math.max(0, upper.length - 3)]!;
        const tx = outer.x - tangentFrom.x;
        const ty = outer.y - tangentFrom.y;
        const tLength = Math.hypot(tx, ty) || 1;
        const tip = addScaled(
          { x: outer.x + (tx / tLength) * radius * 0.3, y: outer.y + (ty / tLength) * radius * 0.3 },
          frame.up,
          radius * 0.17,
        );
        c.beginPath();
        c.moveTo(outer.x, outer.y);
        c.lineTo(tip.x, tip.y);
        c.lineTo(outer.x + frame.up.x * radius * 0.14, outer.y + frame.up.y * radius * 0.14);
        c.closePath();
        c.fill();
      }

      if (lower.length >= 3) {
        // Lower line only along the outer third, and much thinner: a full lower
        // line closes the eye up instead of opening it.
        lidStrip(c, lower, center, frame, (t) => radius * 0.03 * Math.pow(t, 2.6), -1);
      }
    }
  }, 0.0035);

  const browPlane = renderLayer((c) => {
    for (const brow of rings.brows) fillPolygon(c, localRing(brow));
  }, 0.009);

  const litPlane = renderLayer((c) => {
    for (const eye of frame.eyes) {
      const radius = eye.radius * scale;
      // Under-eye: the single change that most reads as "rested".
      softBlob(c, addScaled(local(eye.center), frame.up, -radius * 0.6), radius * 0.75, radius * 0.24, frame, 0.3);
      // Top of the cheekbone.
      softBlob(c, addScaled(cheekOf(eye), frame.up, faceWidth * 0.1), faceWidth * 0.15, faceWidth * 0.07, frame, 0.3);
    }
    // Nose bridge, as an elongated blob spanning the eye line to the nose tip.
    softBlob(c, bridgeCenter, faceWidth * 0.04, bridgeLength * 0.62, frame, 0.2);
    // Forehead. Small relative to the forehead itself: a blob that covers it edge to
    // edge removes the last of its modelling and leaves a featureless pale field,
    // which was the most conspicuous thing wrong with the strong settings.
    softBlob(c, addScaled(eyeMid, frame.up, faceHalfHeight * 0.5), faceWidth * 0.13, faceHalfHeight * 0.1, frame, 0.35);
    softBlob(c, addScaled(chin, frame.up, faceHalfHeight * 0.13), faceWidth * 0.1, faceWidth * 0.07, frame, 0.25);
    // Cupid's bow, just above the upper lip. Small: any wider and it stops reading
    // as a highlight on the philtrum and becomes a pale patch around the mouth.
    softBlob(c, addScaled(lipsCenter, frame.up, lipVertRadius * 1.35), lipLatRadius * 0.2, lipVertRadius * 0.16, frame, 0.4);
  }, 0.04);
  multiplyPlane(litPlane, ovalPlane);

  const shadowedPlane = renderLayer((c) => {
    for (const eye of frame.eyes) {
      // Under the cheekbone, running toward the ear.
      softBlob(c, addScaled(cheekOf(eye), frame.up, -faceWidth * 0.11), faceWidth * 0.16, faceWidth * 0.055, frame, 0.35);
      // No temple shading. At 0.46 of face width from the midline the blob sits on
      // the outer brow rather than the temple, and it reads as a dirty smudge there.
      // Either side of the nose bridge, which is what makes the nose read narrower.
      softBlob(
        c,
        addScaled(bridgeCenter, eye.outerDir, faceWidth * 0.06),
        faceWidth * 0.03,
        bridgeLength * 0.55,
        frame,
        0.3,
      );
    }
    // Along the jaw, just inside the oval. Drawn as blobs on the lower oval
    // vertices rather than a stroke, so it has no hard ends at the chin or ears.
    const ovalPts = localRing(rings.faceOval);
    const ovalCentre = ovalPts.reduce((acc, p) => ({ x: acc.x + p.x / ovalPts.length, y: acc.y + p.y / ovalPts.length }), {
      x: 0,
      y: 0,
    });
    for (const p of ovalPts) {
      if (project(p, eyeMid, frame.up) > -faceHalfHeight * 0.25) continue;
      const inward = { x: ovalCentre.x - p.x, y: ovalCentre.y - p.y };
      const length = Math.hypot(inward.x, inward.y) || 1;
      const at = addScaled(p, { x: inward.x / length, y: inward.y / length }, faceWidth * 0.055);
      softBlob(c, at, faceWidth * 0.075, faceWidth * 0.075, frame, 0.35);
    }
  }, 0.05);
  multiplyPlane(shadowedPlane, ovalPlane);

  /**
   * The two are packed into ONE signed channel: 128 is neutral, above it is light
   * and below it is shadow.
   *
   * Contouring is where the reference ladder's luminosity actually comes from — it
   * is modelling, not saturation. Highlights alone brighten the face into a flat
   * chalky mask; the shadows are what put the bones back. There is no spare channel
   * for a second map, and there does not need to be: no pixel is both lit and
   * shaded, so one signed channel carries both losslessly.
   */
  const contourPlane = new Uint8ClampedArray(pixelCount);
  for (let i = 0; i < pixelCount; i++) {
    contourPlane[i] = 128 + (litPlane[i]! - shadowedPlane[i]!) / 2;
  }

  const catchlightPlane = renderLayer((c) => {
    frame.eyes.forEach((eye, index) => {
      const irisRing = rings.irises[index];
      const irisPts = irisRing ? localRing(irisRing) : [];
      let center = local(eye.center);
      let irisRadius = eye.radius * scale * 0.39;
      if (irisPts.length >= 3) {
        let sumX = 0;
        let sumY = 0;
        for (const p of irisPts) {
          sumX += p.x;
          sumY += p.y;
        }
        center = { x: sumX / irisPts.length, y: sumY / irisPts.length };
        let maxRadius = 0;
        for (const p of irisPts) maxRadius = Math.max(maxRadius, distance(p, center));
        irisRadius = Math.max(maxRadius, 1);
      }
      // Both eyes get the highlight on the SAME side of the iris, because a
      // catchlight is the reflection of one light source. Mirroring it per eye is
      // the classic tell of a fake one.
      const spot = addScaled(addScaled(center, frame.up, irisRadius * 0.32), frame.right, -irisRadius * 0.32);
      // Small and tight. A soft wide blob does not read as a reflection, it just
      // milks the whole iris — which is the difference between a catchlight and a
      // cataract.
      softBlob(c, spot, irisRadius * 0.22, irisRadius * 0.22, frame, 0.6);
    });
  }, 0.002);

  const pack = (planes: readonly [Plane, Plane, Plane, Plane]): AuxTextureData =>
    packPlanes([planes[0], planes[1], planes[2], planes[3]], width, height);

  return {
    a: pack([lipPlane, glossPlane, blushPlane, eyeshadowPlane]),
    b: pack([lashPlane, browPlane, contourPlane, catchlightPlane]),
  };
}
