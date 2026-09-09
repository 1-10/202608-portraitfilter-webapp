import { describe, expect, it } from "vitest";
import {
  buildFaceFrame,
  buildSkirtRing,
  computeGeomRect,
  pickInnerRing,
  pickOuterRing,
  polygonArea,
  ringsFromConnections,
  trianglesFromConnections,
  type Connection,
  type Pt,
} from "../../src/vision/faceMeshTopology";
import { SYNTHETIC_FACE, rotatedFace } from "./fixtures/syntheticFace";

/** Directed edge cycle through the given indices. */
function cycle(indices: number[]): Connection[] {
  return indices.map((start, i) => ({ start, end: indices[(i + 1) % indices.length]! }));
}

describe("ringsFromConnections", () => {
  it("recovers a single cycle in order", () => {
    const rings = ringsFromConnections(cycle([4, 7, 1, 9]));
    expect(rings).toHaveLength(1);
    expect(rings[0]).toEqual([4, 7, 1, 9]);
  });

  it("recovers BOTH cycles of a two-contour region", () => {
    // This is the lips: an outer border and an inner mouth border in one list. A
    // walk that keeps a single next-hop per vertex returns only one of them.
    const rings = ringsFromConnections([...cycle([0, 1, 2, 3]), ...cycle([10, 11, 12])]);
    expect(rings).toHaveLength(2);
    expect(rings.map((r) => r.length).sort()).toEqual([3, 4]);
  });

  it("closes a contour published as two arcs between the same endpoints", () => {
    // Exactly how MediaPipe publishes each lip contour: an upper arc and a lower
    // arc, both running from one mouth corner to the other. Following edge
    // direction dead-ends at the far corner and finds no ring at all — which is
    // silent, because the caller just gets an empty region.
    const upper = [
      { start: 0, end: 1 },
      { start: 1, end: 2 },
      { start: 2, end: 3 },
    ];
    const lower = [
      { start: 0, end: 4 },
      { start: 4, end: 5 },
      { start: 5, end: 3 },
    ];
    const rings = ringsFromConnections([...upper, ...lower]);
    expect(rings).toHaveLength(1);
    expect(rings[0]).toHaveLength(6);
    expect(new Set(rings[0])).toEqual(new Set([0, 1, 2, 3, 4, 5]));
  });

  it("ignores an edge listed in both directions rather than bouncing back along it", () => {
    const rings = ringsFromConnections([
      { start: 0, end: 1 },
      { start: 1, end: 0 },
      { start: 1, end: 2 },
      { start: 2, end: 0 },
    ]);
    expect(rings).toHaveLength(1);
    expect(rings[0]).toHaveLength(3);
  });

  it("returns nothing for an open chain", () => {
    expect(ringsFromConnections([{ start: 0, end: 1 }, { start: 1, end: 2 }])).toEqual([]);
  });

  it("terminates on a self-referential list", () => {
    expect(ringsFromConnections([{ start: 0, end: 0 }])).toEqual([]);
  });
});

describe("pickOuterRing / pickInnerRing", () => {
  const points: Pt[] = [
    // A large square, indices 0..3.
    { x: 0, y: 0 },
    { x: 10, y: 0 },
    { x: 10, y: 10 },
    { x: 0, y: 10 },
    // A small square inside it, indices 4..7.
    { x: 4, y: 4 },
    { x: 6, y: 4 },
    { x: 6, y: 6 },
    { x: 4, y: 6 },
  ];
  const rings = [
    [4, 5, 6, 7],
    [0, 1, 2, 3],
  ];

  it("picks by area, not by list order or length", () => {
    expect(pickOuterRing(rings, points)).toEqual([0, 1, 2, 3]);
    expect(pickInnerRing(rings, points)).toEqual([4, 5, 6, 7]);
  });

  it("measures polygon area regardless of winding", () => {
    expect(polygonArea([points[0]!, points[1]!, points[2]!, points[3]!])).toBe(100);
    expect(polygonArea([points[3]!, points[2]!, points[1]!, points[0]!])).toBe(100);
  });
});

describe("trianglesFromConnections", () => {
  it("recovers the two triangles of a split quad, each exactly once", () => {
    // 0-1-2-3 quad with the 0-2 diagonal: triangles {0,1,2} and {0,2,3}.
    const edges: Connection[] = [
      { start: 0, end: 1 },
      { start: 1, end: 2 },
      { start: 2, end: 3 },
      { start: 3, end: 0 },
      { start: 0, end: 2 },
    ];
    const tris = trianglesFromConnections(edges);
    expect(tris.length).toBe(6);
    const asTriples = [
      [tris[0], tris[1], tris[2]],
      [tris[3], tris[4], tris[5]],
    ].map((t) => t.join(","));
    expect(asTriples.sort()).toEqual(["0,1,2", "0,2,3"]);
  });

  it("ignores edge direction and duplicates", () => {
    const forward = trianglesFromConnections([
      { start: 0, end: 1 },
      { start: 1, end: 2 },
      { start: 2, end: 0 },
    ]);
    const reversedAndDoubled = trianglesFromConnections([
      { start: 1, end: 0 },
      { start: 0, end: 1 },
      { start: 2, end: 1 },
      { start: 0, end: 2 },
    ]);
    expect(Array.from(forward)).toEqual([0, 1, 2]);
    expect(Array.from(reversedAndDoubled)).toEqual([0, 1, 2]);
  });

  it("emits only triangles whose three edges are all present", () => {
    const edges = [...cycle([0, 1, 2, 3, 4]), { start: 0, end: 2 }];
    const tris = trianglesFromConnections(edges);
    const present = new Set(edges.flatMap((e) => [`${e.start}-${e.end}`, `${e.end}-${e.start}`]));
    for (let i = 0; i < tris.length; i += 3) {
      const [a, b, c] = [tris[i]!, tris[i + 1]!, tris[i + 2]!];
      for (const [x, y] of [
        [a, b],
        [b, c],
        [a, c],
      ]) {
        expect(present.has(`${x}-${y}`), `edge ${x}-${y} of triangle ${a},${b},${c}`).toBe(true);
      }
    }
  });
});

describe("buildFaceFrame", () => {
  const rings = SYNTHETIC_FACE.rings;

  it("derives an upright frame from the eyes and mouth", () => {
    const built = buildFaceFrame(SYNTHETIC_FACE.landmarks, rings, 1000, 1000);
    expect(built).not.toBeNull();
    const { frame } = built!;
    // Image y grows downward, so the face's up axis is -y.
    expect(frame.up.x).toBeCloseTo(0, 6);
    expect(frame.up.y).toBeCloseTo(-1, 6);
    expect(frame.right.x).toBeCloseTo(1, 6);
    expect(frame.right.y).toBeCloseTo(0, 6);
  });

  it("rotates the frame with the head", () => {
    const rotated = rotatedFace(20);
    const built = buildFaceFrame(rotated, rings, 1000, 1000);
    expect(built).not.toBeNull();
    const { frame } = built!;
    // Angle of the up axis measured clockwise from straight up, matching the
    // direction rotatedFace() turns the points in image coordinates.
    const angle = Math.atan2(frame.up.x, -frame.up.y);
    expect((angle * 180) / Math.PI).toBeCloseTo(20, 1);
    // The axes stay orthonormal, which is what every projection downstream assumes.
    expect(frame.up.x * frame.right.x + frame.up.y * frame.right.y).toBeCloseTo(0, 6);
    expect(Math.hypot(frame.right.x, frame.right.y)).toBeCloseTo(1, 6);
  });

  it("works in pixels, so a non-square image does not skew the frame", () => {
    // With normalized coordinates taken at face value, a 1:2 frame would tilt the
    // lateral axis and turn every circular falloff into an ellipse.
    const built = buildFaceFrame(SYNTHETIC_FACE.landmarks, rings, 1000, 2000);
    expect(built).not.toBeNull();
    const { frame } = built!;
    expect(frame.up.y).toBeCloseTo(-1, 6);
    expect(frame.right.x).toBeCloseTo(1, 6);
    // Face width in pixels scales with the axis it is measured on.
    const square = buildFaceFrame(SYNTHETIC_FACE.landmarks, rings, 1000, 1000)!;
    expect(frame.faceWidth).toBeCloseTo(square.frame.faceWidth, 6);
    expect(frame.faceHeight).toBeCloseTo(square.frame.faceHeight * 2, 6);
  });

  it("puts the chin at the lowest point of the oval", () => {
    const { frame } = buildFaceFrame(SYNTHETIC_FACE.landmarks, rings, 1000, 1000)!;
    for (const index of rings.faceOval) {
      const p = SYNTHETIC_FACE.landmarks[index]!;
      expect(p.y * 1000).toBeLessThanOrEqual(frame.chin.y + 1e-6);
    }
  });

  it("splits each eye ring into an upper and a lower lid", () => {
    const { frame } = buildFaceFrame(SYNTHETIC_FACE.landmarks, rings, 1000, 1000)!;
    for (const eye of frame.eyes) {
      expect(eye.upperLid.length).toBeGreaterThanOrEqual(3);
      expect(eye.lowerLid.length).toBeGreaterThanOrEqual(3);
      const meanUpperY =
        eye.upperLid.reduce((sum, i) => sum + SYNTHETIC_FACE.landmarks[i]!.y, 0) / eye.upperLid.length;
      const meanLowerY =
        eye.lowerLid.reduce((sum, i) => sum + SYNTHETIC_FACE.landmarks[i]!.y, 0) / eye.lowerLid.length;
      expect(meanUpperY).toBeLessThan(meanLowerY);
    }
  });

  it("points each eye's outerDir away from the face midline", () => {
    const { frame } = buildFaceFrame(SYNTHETIC_FACE.landmarks, rings, 1000, 1000)!;
    const [a, b] = frame.eyes;
    // Opposite lateral directions, and each points away from the centre.
    expect(a.outerDir.x * b.outerDir.x).toBeLessThan(0);
    for (const eye of frame.eyes) {
      const towardOuter = (eye.center.x - frame.eyeMid.x) * eye.outerDir.x;
      expect(towardOuter).toBeGreaterThan(0);
    }
  });

  it("returns null rather than guessing when a ring is degenerate", () => {
    expect(buildFaceFrame([], rings, 1000, 1000)).toBeNull();
    expect(
      buildFaceFrame(SYNTHETIC_FACE.landmarks, { ...rings, faceOval: [0, 1] }, 1000, 1000),
    ).toBeNull();
  });
});

describe("buildSkirtRing / computeGeomRect", () => {
  const { frame, points } = buildFaceFrame(SYNTHETIC_FACE.landmarks, SYNTHETIC_FACE.rings, 1000, 1000)!;

  it("pushes every oval vertex strictly outward", () => {
    const skirt = buildSkirtRing(points, SYNTHETIC_FACE.rings.faceOval, 40);
    expect(skirt).toHaveLength(SYNTHETIC_FACE.rings.faceOval.length);
    const centre = { x: 500, y: 500 };
    SYNTHETIC_FACE.rings.faceOval.forEach((index, i) => {
      const inner = points[index]!;
      const outer = skirt[i]!;
      const innerDistance = Math.hypot(inner.x - centre.x, inner.y - centre.y);
      const outerDistance = Math.hypot(outer.x - centre.x, outer.y - centre.y);
      expect(outerDistance).toBeGreaterThan(innerDistance);
      expect(outerDistance - innerDistance).toBeCloseTo(40, 4);
    });
  });

  it("bounds the given points and adds the margin on every side", () => {
    const rect = computeGeomRect([{ x: 100, y: 200 }, { x: 300, y: 260 }], 10);
    expect(rect).toEqual({ x: 90, y: 190, w: 220, h: 80 });
  });

  it("keeps the whole skirt inside the rect", () => {
    const skirt = buildSkirtRing(points, SYNTHETIC_FACE.rings.faceOval, frame.faceWidth * 0.32);
    const rect = computeGeomRect(skirt, frame.faceWidth * 0.06);
    for (const p of skirt) {
      expect(p.x).toBeGreaterThan(rect.x);
      expect(p.x).toBeLessThan(rect.x + rect.w);
      expect(p.y).toBeGreaterThan(rect.y);
      expect(p.y).toBeLessThan(rect.y + rect.h);
    }
  });
});
