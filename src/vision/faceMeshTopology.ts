/**
 * Face-mesh topology and the anchor frame every deformation and makeup layer is
 * expressed in.
 *
 * Two rules shape this module:
 *
 * 1. **No literal landmark indices.** MediaPipe renumbers points between model
 *    revisions, so everything here is derived at runtime from the published
 *    connection lists (`FaceLandmarker.FACE_LANDMARKS_*`) or from geometry.
 * 2. **Pixels, not normalized UV.** Landmarks arrive normalized 0..1 on each axis
 *    independently, so dot products and distances taken on them are skewed by the
 *    image aspect ratio — a circle around the eye would come out as an ellipse on
 *    a 9:16 portrait. Every function here works in working-image pixels and the
 *    conversion back to UV happens once, at texture-encode time.
 */

export type Connection = { start: number; end: number };
export type Pt = { x: number; y: number };
export type NormalizedLandmark = { x: number; y: number; z?: number };

/**
 * Walks a connection list into every closed ring it contains, treating the edges
 * as UNDIRECTED.
 *
 * Undirected is the whole point. MediaPipe publishes region outlines as edge
 * lists, and the direction of those edges does not encode a traversal: the lips
 * are published as concentric contours each made of an upper arc and a lower arc
 * that both run from one mouth corner to the other. Following edge direction
 * therefore walks up one arc and dead-ends at the far corner — no cycle, no lip
 * region, and every layer that depends on the lips silently disappears. Ignoring
 * direction closes both contours.
 *
 * Edges are consumed as they are walked, so each cycle is returned exactly once
 * and the two lip contours come back separately.
 */
export function ringsFromConnections(connections: readonly Connection[]): number[][] {
  const edgeA: number[] = [];
  const edgeB: number[] = [];
  const incident = new Map<number, number[]>();
  const seen = new Set<string>();
  const attach = (vertex: number, edgeId: number): void => {
    const list = incident.get(vertex);
    if (list) list.push(edgeId);
    else incident.set(vertex, [edgeId]);
  };
  for (const c of connections) {
    if (c.start === c.end) continue;
    const lo = Math.min(c.start, c.end);
    const hi = Math.max(c.start, c.end);
    const key = `${lo}-${hi}`;
    // A list may name the same edge in both directions; a duplicate would let the
    // walk bounce straight back where it came from.
    if (seen.has(key)) continue;
    seen.add(key);
    const id = edgeA.length;
    edgeA.push(c.start);
    edgeB.push(c.end);
    attach(c.start, id);
    attach(c.end, id);
  }

  const used = new Array<boolean>(edgeA.length).fill(false);
  const takeUnused = (vertex: number): number => {
    for (const id of incident.get(vertex) ?? []) {
      if (!used[id]) return id;
    }
    return -1;
  };

  const rings: number[][] = [];
  for (let id = 0; id < edgeA.length; id++) {
    if (used[id]) continue;
    const start = edgeA[id]!;
    used[id] = true;
    const ring = [start];
    let cur = edgeB[id]!;
    // The guard is the edge count: a malformed list must not spin forever.
    while (cur !== start && ring.length <= edgeA.length) {
      ring.push(cur);
      const next = takeUnused(cur);
      if (next < 0) break;
      used[next] = true;
      cur = edgeA[next] === cur ? edgeB[next]! : edgeA[next]!;
    }
    if (cur === start && ring.length >= 3) rings.push(ring);
  }
  return rings;
}

/** Signed-area magnitude of a polygon, in whatever units the points are given in. */
export function polygonArea(points: readonly Pt[]): number {
  if (points.length < 3) return 0;
  let sum = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i]!;
    const b = points[(i + 1) % points.length]!;
    sum += a.x * b.y - b.x * a.y;
  }
  return Math.abs(sum) * 0.5;
}

/** The largest-area ring, used where a region has concentric contours (the lips). */
export function pickOuterRing(rings: readonly number[][], landmarks: readonly Pt[]): number[] {
  let best: number[] = [];
  let bestArea = -1;
  for (const ring of rings) {
    const pts = ring.map((i) => landmarks[i]).filter((p): p is Pt => !!p);
    const area = polygonArea(pts);
    if (area > bestArea) {
      bestArea = area;
      best = ring;
    }
  }
  return best;
}

/** The smallest-area ring, i.e. the inner mouth border. */
export function pickInnerRing(rings: readonly number[][], landmarks: readonly Pt[]): number[] {
  let best: number[] = [];
  let bestArea = Infinity;
  for (const ring of rings) {
    const pts = ring.map((i) => landmarks[i]).filter((p): p is Pt => !!p);
    const area = polygonArea(pts);
    if (area < bestArea) {
      bestArea = area;
      best = ring;
    }
  }
  return best;
}

/**
 * Recovers the mesh triangles from `FACE_LANDMARKS_TESSELATION`.
 *
 * The tessellation is published as an EDGE list, not a triangle list, so it cannot
 * be drawn or rasterized directly. A triangle is any three mutually connected
 * vertices: for each undirected edge (lo, hi) we take every common neighbour
 * `t > hi`, which emits each triangle {a<b<c} exactly once — from edge (a,b) with
 * t=c, and from no other edge.
 *
 * A closed mesh can in principle contain a 3-cycle that is not a face. Here that
 * is harmless: these triangles only carry a smooth displacement field, so a
 * spurious one interpolates between values that are already nearly equal.
 */
export function trianglesFromConnections(connections: readonly Connection[]): Uint16Array {
  const adjacency = new Map<number, Set<number>>();
  const addEdge = (a: number, b: number): void => {
    let set = adjacency.get(a);
    if (!set) {
      set = new Set<number>();
      adjacency.set(a, set);
    }
    set.add(b);
  };
  for (const c of connections) {
    if (c.start === c.end) continue;
    addEdge(c.start, c.end);
    addEdge(c.end, c.start);
  }

  const out: number[] = [];
  const seenEdge = new Set<number>();
  for (const c of connections) {
    const lo = Math.min(c.start, c.end);
    const hi = Math.max(c.start, c.end);
    if (lo === hi) continue;
    const edgeKey = lo * 65536 + hi;
    if (seenEdge.has(edgeKey)) continue;
    seenEdge.add(edgeKey);

    const nLo = adjacency.get(lo);
    const nHi = adjacency.get(hi);
    if (!nLo || !nHi) continue;
    const [probe, other] = nLo.size <= nHi.size ? [nLo, nHi] : [nHi, nLo];
    for (const t of probe) {
      if (t > hi && other.has(t)) out.push(lo, hi, t);
    }
  }
  return new Uint16Array(out);
}

export type EyeInfo = {
  center: Pt;
  /** Half the eye's width along the face's lateral axis, in pixels. */
  radius: number;
  /** Unit lateral direction pointing away from the face midline for this eye. */
  outerDir: Pt;
  /** Ring indices ordered from the inner corner to the outer corner along the UPPER lid. */
  upperLid: number[];
  /** The same walk along the LOWER lid. */
  lowerLid: number[];
  innerCorner: Pt;
  outerCorner: Pt;
};

/**
 * The face's own coordinate frame: an origin, an up axis and a lateral axis
 * derived from the eyes and mouth rather than from the image.
 *
 * Everything downstream is expressed in this frame so a tilted head is deformed
 * and made up along ITS axes. Anchored on image axes instead, a 15-degree head
 * tilt slims one cheek and widens the other.
 */
export type FaceFrame = {
  /** Unit vector from the mouth toward the eyes. */
  up: Pt;
  /** Unit vector perpendicular to `up`. */
  right: Pt;
  /** Midpoint of the two eye centres — the origin of the frame. */
  eyeMid: Pt;
  lipsCenter: Pt;
  noseCenter: Pt;
  chin: Pt;
  /** Oval extent along `right`, in pixels. */
  faceWidth: number;
  /** Oval extent along `up`, in pixels. */
  faceHeight: number;
  eyes: [EyeInfo, EyeInfo];
  /** Interocular distance in pixels. */
  interocular: number;
};

function centroid(points: readonly Pt[]): Pt {
  let x = 0;
  let y = 0;
  for (const p of points) {
    x += p.x;
    y += p.y;
  }
  const n = Math.max(1, points.length);
  return { x: x / n, y: y / n };
}

function toPixels(landmarks: readonly NormalizedLandmark[], width: number, height: number): Pt[] {
  return landmarks.map((p) => ({ x: p.x * width, y: p.y * height }));
}

/** Projection of `p - origin` onto unit vector `axis`. */
export function project(p: Pt, origin: Pt, axis: Pt): number {
  return (p.x - origin.x) * axis.x + (p.y - origin.y) * axis.y;
}

export function distance(a: Pt, b: Pt): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function buildEyeInfo(
  pts: readonly Pt[],
  ring: readonly number[],
  axes: { up: Pt; right: Pt },
  midline: Pt,
): EyeInfo | null {
  const ringPts = ring.map((i) => pts[i]).filter((p): p is Pt => !!p);
  if (ringPts.length < 4) return null;
  const center = centroid(ringPts);

  // Corners are the ring's lateral extremes. Which one counts as "outer" is decided
  // by which side of the face midline the eye sits on, so this works for either eye
  // and for a mirrored (selfie) image without a left/right convention to get wrong.
  let minLat = Infinity;
  let maxLat = -Infinity;
  let minIdx = 0;
  let maxIdx = 0;
  ring.forEach((landmarkIndex, i) => {
    const p = pts[landmarkIndex];
    if (!p) return;
    const lat = project(p, center, axes.right);
    if (lat < minLat) {
      minLat = lat;
      minIdx = i;
    }
    if (lat > maxLat) {
      maxLat = lat;
      maxIdx = i;
    }
  });
  const radius = Math.max((maxLat - minLat) * 0.5, 1);

  const centerLat = project(center, midline, axes.right);
  const outerSign = centerLat >= 0 ? 1 : -1;
  const outerDir: Pt = { x: axes.right.x * outerSign, y: axes.right.y * outerSign };
  const outerIdx = outerSign > 0 ? maxIdx : minIdx;
  const innerIdx = outerSign > 0 ? minIdx : maxIdx;

  // Walk the ring both ways from the inner corner to the outer corner and keep the
  // run whose points sit ABOVE the eye centre: that is the upper lid, where lashes,
  // liner and eyeshadow go.
  const walk = (direction: 1 | -1): number[] => {
    const run: number[] = [];
    for (let step = 0; step <= ring.length; step++) {
      const at = (innerIdx + direction * step + ring.length * 2) % ring.length;
      run.push(ring[at]!);
      if (at === outerIdx) break;
    }
    return run;
  };
  const forward = walk(1);
  const backward = walk(-1);
  const meanHeight = (run: readonly number[]): number => {
    let sum = 0;
    let count = 0;
    for (const idx of run) {
      const p = pts[idx];
      if (!p) continue;
      sum += project(p, center, axes.up);
      count++;
    }
    return count > 0 ? sum / count : -Infinity;
  };
  const forwardIsUpper = meanHeight(forward) >= meanHeight(backward);

  return {
    center,
    radius,
    outerDir,
    upperLid: forwardIsUpper ? forward : backward,
    lowerLid: forwardIsUpper ? backward : forward,
    innerCorner: pts[ring[innerIdx]!]!,
    outerCorner: pts[ring[outerIdx]!]!,
  };
}

export type FrameRings = {
  faceOval: readonly number[];
  leftEye: readonly number[];
  rightEye: readonly number[];
  lips: readonly number[];
};

/**
 * Builds the face frame, or returns null when the rings are too degenerate to
 * define one — which the caller must treat as "no geometry available" and fall
 * back to a plain image filter.
 */
export function buildFaceFrame(
  landmarks: readonly NormalizedLandmark[],
  rings: FrameRings,
  imageWidth: number,
  imageHeight: number,
): { frame: FaceFrame; points: Pt[] } | null {
  if (landmarks.length === 0) return null;
  const pts = toPixels(landmarks, imageWidth, imageHeight);

  const ovalPts = rings.faceOval.map((i) => pts[i]).filter((p): p is Pt => !!p);
  const eyeAPts = rings.leftEye.map((i) => pts[i]).filter((p): p is Pt => !!p);
  const eyeBPts = rings.rightEye.map((i) => pts[i]).filter((p): p is Pt => !!p);
  const lipsPts = rings.lips.map((i) => pts[i]).filter((p): p is Pt => !!p);
  if (ovalPts.length < 3 || eyeAPts.length < 4 || eyeBPts.length < 4 || lipsPts.length < 3) return null;

  const eyeCenterA = centroid(eyeAPts);
  const eyeCenterB = centroid(eyeBPts);
  const eyeMid = { x: (eyeCenterA.x + eyeCenterB.x) / 2, y: (eyeCenterA.y + eyeCenterB.y) / 2 };
  const lipsCenter = centroid(lipsPts);

  const upRaw = { x: eyeMid.x - lipsCenter.x, y: eyeMid.y - lipsCenter.y };
  const upLen = Math.hypot(upRaw.x, upRaw.y);
  if (upLen < 1e-6) return null;
  const up = { x: upRaw.x / upLen, y: upRaw.y / upLen };
  // Rotate `up` by +90 degrees. Image y grows downward, so for an upright face
  // (up = (0,-1)) this yields right = (1,0) as expected.
  const right = { x: -up.y, y: up.x };

  let minLat = Infinity;
  let maxLat = -Infinity;
  let minVert = Infinity;
  let maxVert = -Infinity;
  let chin = ovalPts[0]!;
  for (const p of ovalPts) {
    const lat = project(p, eyeMid, right);
    const vert = project(p, eyeMid, up);
    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;
    if (vert < minVert) {
      minVert = vert;
      chin = p;
    }
    if (vert > maxVert) maxVert = vert;
  }
  const faceWidth = Math.max(maxLat - minLat, 1);
  const faceHeight = Math.max(maxVert - minVert, 1);

  const eyeA = buildEyeInfo(pts, rings.leftEye, { up, right }, eyeMid);
  const eyeB = buildEyeInfo(pts, rings.rightEye, { up, right }, eyeMid);
  if (!eyeA || !eyeB) return null;

  return {
    points: pts,
    frame: {
      up,
      right,
      eyeMid,
      lipsCenter,
      // The nose sits between the eye line and the mouth; 0.55 of the way down
      // lands on the tip and alae across a wide range of faces, and because it is
      // measured along the face's own axis it stays put when the head tilts.
      noseCenter: {
        x: eyeMid.x + (lipsCenter.x - eyeMid.x) * 0.55,
        y: eyeMid.y + (lipsCenter.y - eyeMid.y) * 0.55,
      },
      chin,
      faceWidth,
      faceHeight,
      eyes: [eyeA, eyeB],
      interocular: distance(eyeCenterA, eyeCenterB),
    },
  };
}

/** Smooth 0..1 ramp; matches GLSL smoothstep, reversed edges included. */
export function smoothstep(edge0: number, edge1: number, x: number): number {
  if (edge0 === edge1) return x < edge0 ? 0 : 1;
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

export function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/**
 * The region of the image that the warp field and the makeup masks are baked over,
 * in working-image pixels.
 *
 * Both textures cover this rect rather than the whole frame, which is what buys
 * their resolution: a 1024px makeup texture spread over a 4000px-wide photo would
 * put ~100px across an eye, but spread over the face box it puts ~500px there.
 */
export type GeomRect = { x: number; y: number; w: number; h: number };

/**
 * How far outside the face oval the skirt ring sits, as a fraction of face width.
 *
 * Lives here because BOTH the warp field and the geometry rect are built from the
 * same skirt, and they have to agree: the rect is sized to contain the skirt, so two
 * copies of this number silently crop the field the moment they drift apart.
 *
 * It also sets how far the deformation reaches OUTSIDE the face. Too small and the
 * jaw narrows while the hair silhouette around it does not, so the face reads as
 * floating inside its own hair.
 */
export const SKIRT_OUTWARD_FACE_FRACTION = 0.32;

/**
 * Pushes the face-oval ring outward to form the "skirt" ring.
 *
 * The tessellation stops at the face oval, so a displacement field baked from it
 * alone would jump from full magnitude to zero across the oval boundary — tearing
 * the jaw edge against the hair and background behind it. The skirt carries zero
 * displacement, so the field decays smoothly over the annulus and the surroundings
 * are dragged slightly along with the jaw, which is what makes slimming read as
 * the face getting smaller rather than the outline being dented.
 */
export function buildSkirtRing(points: readonly Pt[], ovalRing: readonly number[], outwardPx: number): Pt[] {
  const ringPts = ovalRing.map((i) => points[i]).filter((p): p is Pt => !!p);
  const center = centroid(ringPts);
  return ringPts.map((p) => {
    const dx = p.x - center.x;
    const dy = p.y - center.y;
    const len = Math.hypot(dx, dy) || 1;
    return { x: p.x + (dx / len) * outwardPx, y: p.y + (dy / len) * outwardPx };
  });
}

/** Axis-aligned bounds of `points`, grown by `marginPx` on every side. */
export function computeGeomRect(points: readonly Pt[], marginPx: number): GeomRect {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  return {
    x: minX - marginPx,
    y: minY - marginPx,
    w: Math.max(1, maxX - minX + marginPx * 2),
    h: Math.max(1, maxY - minY + marginPx * 2),
  };
}
