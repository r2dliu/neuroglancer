/**
 * The paint tools' rasterizer: which voxels a stroke covers.
 */

import type { clampAndRoundCoordinateToVoxelCenter } from "#src/coordinate_transform.js";
import { mat4, vec3, type quat } from "#src/util/geom.js";

/** Every field is indexed by global spatial dim. */
export interface BrushPlaneFrame {
  /** Voxel-index offset of a +1 canonical-unit step along the viewport right
   *  axis. */
  dirU: vec3;
  /** Same, along the viewport up axis. */
  dirV: vec3;
  /** Slice-plane normal in voxel-index space (unit). */
  normal: vec3;
  // What toCanonical reads; dirU = mU / factors does not round-trip exactly.
  mU: vec3;
  mV: vec3;
  factors: Float64Array;
  /** Project a voxel-index displacement onto the plane, in canonical units —
   *  the space the radius is measured in. */
  toCanonical(delta: vec3, out: Float64Array): void;
}

interface PoseLike {
  orientation: { orientation: quat };
  displayDimensionRenderInfo: {
    value: {
      canonicalVoxelFactors: Float64Array;
      displayDimensionIndices: Int32Array;
    };
  };
}

export type VoxelBounds = Parameters<
  typeof clampAndRoundCoordinateToVoxelCenter
>[0];

export function brushPlaneFrame(pose: PoseLike): BrushPlaneFrame {
  const m = mat4.fromQuat(mat4.create(), pose.orientation.orientation);
  const { canonicalVoxelFactors: f, displayDimensionIndices: dd } =
    pose.displayDimensionRenderInfo.value;
  const dirU = vec3.create();
  const dirV = vec3.create();
  const mU = vec3.create();
  const mV = vec3.create();
  const factors = new Float64Array(3);
  // Row i of the rotation matrix is display dimension i; map each onto its
  // global spatial dim (identity for the [x, y, z] spaces this app loads).
  for (let i = 0; i < 3; i++) {
    const g = dd[i];
    if (g < 0 || g > 2) continue;
    dirU[g] = m[i] / f[i];
    dirV[g] = m[4 + i] / f[i];
    mU[g] = m[i];
    mV[g] = m[4 + i];
    factors[g] = f[i];
  }
  // Plane normal in VOXEL space (the frame the renderer slices cubes in) —
  // the cross of the in-plane voxel-space directions, not the canonical-frame
  // normal (they differ under anisotropy).
  const normal = vec3.cross(vec3.create(), dirU, dirV);
  vec3.normalize(normal, normal);
  return {
    dirU,
    dirV,
    normal,
    mU,
    mV,
    factors,
    toCanonical(delta: vec3, out: Float64Array) {
      let du = 0;
      let dv = 0;
      for (let g = 0; g < 3; g++) {
        const c = delta[g] * factors[g];
        du += c * mU[g];
        dv += c * mV[g];
      }
      out[0] = du;
      out[1] = dv;
    },
  };
}

// Voxel identity as one number rather than a `${x},${y},${z}` template, which
// allocates a string per candidate. Coordinates are voxel centers: integers,
// or half-integers when voxelCenterAtIntegerCoordinates is false, so doubling
// makes either exact. Falls back to string keys when the bounds are infinite
// or too large to pack three fields into a float64 integer.
type VoxelKey = (x: number, y: number, z: number) => number | string;

function voxelKeyer(bounds: VoxelBounds): VoxelKey {
  const { lowerBounds, upperBounds } = bounds;
  let stride = 0;
  const origin: number[] = [];
  for (let i = 0; i < 3; i++) {
    const lo = lowerBounds[i];
    const hi = upperBounds[i];
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
      return (x, y, z) => `${x},${y},${z}`;
    }
    origin.push(Math.floor(lo) - 2);
    stride = Math.max(stride, Math.ceil(hi) + 2 - origin[i]);
  }
  stride = stride * 2 + 1;
  if (stride ** 3 > Number.MAX_SAFE_INTEGER) {
    return (x, y, z) => `${x},${y},${z}`;
  }
  const [ox, oy, oz] = origin;
  return (x, y, z) =>
    ((z - oz) * 2 * stride + (y - oy) * 2) * stride + (x - ox) * 2;
}

export interface StrokeStamper {
  advanceTo(position: vec3): void;
}

/**
 * Voxels the slice plane crosses within `radius` of the segment a->b, in
 * canonical in-plane units. Scans the two axes least aligned with the plane
 * normal and solves the third, so cost tracks the area swept.
 */
function sweepSegment(
  frame: BrushPlaneFrame,
  a: vec3,
  b: vec3,
  radius: number,
  bounds: VoxelBounds,
  halfThickness: number,
  emit: (x: number, y: number, z: number) => void,
) {
  const { dirU, dirV, normal } = frame;
  const { lowerBounds, upperBounds, voxelCenterAtIntegerCoordinates } = bounds;
  const radiusSq = radius * radius;

  const delta = vec3.create();
  vec3.subtract(delta, b, a);
  const uv = new Float64Array(2);
  frame.toCanonical(delta, uv);
  const su = uv[0];
  const sv = uv[1];
  const segLenSq = su * su + sv * sv;
  // Pointer samples are not guaranteed to share one plane, so the reference
  // slides along the segment rather than sitting at its start.
  const normalDrift = vec3.dot(delta, normal);

  const lo = [0, 0, 0];
  const hi = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    // Furthest a canonical offset of length `radius` reaches along this axis.
    const reach = radius * Math.sqrt(dirU[k] * dirU[k] + dirV[k] * dirV[k]) + 1;
    lo[k] = Math.max(Math.min(a[k], b[k]) - reach, lowerBounds[k]);
    hi[k] = Math.min(Math.max(a[k], b[k]) + reach, upperBounds[k] - 1);
    if (lo[k] > hi[k]) return;
  }

  let ax = 0;
  for (let k = 1; k < 3; k++) {
    if (Math.abs(normal[k]) > Math.abs(normal[ax])) ax = k;
  }
  const na = normal[ax];
  if (na === 0) return;
  const s1 = (ax + 1) % 3;
  const s2 = (ax + 2) % 3;
  const offset = (k: number) => (voxelCenterAtIntegerCoordinates[k] ? 0 : 0.5);
  const firstCenter = (k: number) => Math.ceil(lo[k] - offset(k)) + offset(k);
  const spread = (halfThickness + Math.abs(normalDrift)) / Math.abs(na);
  const offsetAx = offset(ax);
  const cand = vec3.create();

  const toCanonical = frame.toCanonical;
  const nS1 = normal[s1];
  const nS2 = normal[s2];
  const aS1 = a[s1];
  const aS2 = a[s2];
  const aAx = a[ax];
  const loAx = lo[ax];
  const hiAx = hi[ax];
  const hiS1 = hi[s1];
  const hiS2 = hi[s2];
  const startS2 = firstCenter(s2);

  for (let c1 = firstCenter(s1); c1 <= hiS1; c1++) {
    const k1 = nS1 * (c1 - aS1);
    cand[s1] = c1;
    for (let c2 = startS2; c2 <= hiS2; c2++) {
      // The plane crosses this voxel column over a span of at most 3 voxels;
      // solve for it rather than testing every voxel in the box.
      const centre = aAx - (k1 + nS2 * (c2 - aS2)) / na;
      const aHi = centre + spread < hiAx ? centre + spread : hiAx;
      const aLo = centre - spread > loAx ? centre - spread : loAx;
      cand[s2] = c2;
      for (let av = Math.ceil(aLo - offsetAx) + offsetAx; av <= aHi; av++) {
        cand[ax] = av;
        vec3.subtract(delta, cand, a);
        toCanonical(delta, uv);
        const cu = uv[0];
        const cv = uv[1];
        // A voxel belongs to the stroke when some point of the segment is both
        // within `radius` of it in plane and close enough to cross its cube.
        // Both are intervals in the segment parameter; they must overlap.
        let sLo = 0;
        let sHi = 1;
        const offsetSq = cu * cu + cv * cv;
        if (segLenSq > 0) {
          const half = cu * su + cv * sv;
          const disc = half * half - segLenSq * (offsetSq - radiusSq);
          if (disc < 0) continue;
          const root = Math.sqrt(disc);
          sLo = (half - root) / segLenSq;
          sHi = (half + root) / segLenSq;
          if (sLo < 0) sLo = 0;
          if (sHi > 1) sHi = 1;
          if (sLo > sHi) continue;
        } else if (offsetSq > radiusSq) {
          continue;
        }
        const drop = vec3.dot(delta, normal);
        if (normalDrift === 0) {
          if (Math.abs(drop) > halfThickness) continue;
        } else {
          const e1 = (drop - halfThickness) / normalDrift;
          const e2 = (drop + halfThickness) / normalDrift;
          const pLo = e1 < e2 ? e1 : e2;
          const pHi = e1 < e2 ? e2 : e1;
          if (pLo > sLo) sLo = pLo;
          if (pHi < sHi) sHi = pHi;
          if (sLo > sHi) continue;
        }
        emit(cand[0], cand[1], cand[2]);
      }
    }
  }
}

function planeHalfThickness(normal: vec3) {
  return (
    0.5 * (Math.abs(normal[0]) + Math.abs(normal[1]) + Math.abs(normal[2])) +
    1e-4
  );
}

/**
 * One stroke's worth of rasterization. The stroke is a polyline plus a radius,
 * not a series of stamps: each new sample extends it by one segment and only
 * voxels not already covered are emitted, in the order the stroke reaches them.
 */
export function beginStroke(
  frame: BrushPlaneFrame,
  radius: number,
  bounds: VoxelBounds,
  emit: (x: number, y: number, z: number) => void,
): StrokeStamper {
  const keyOf = voxelKeyer(bounds);
  const seen = new Set<number | string>();
  const halfThickness = planeHalfThickness(frame.normal);
  const emitOnce = (x: number, y: number, z: number) => {
    const key = keyOf(x, y, z);
    if (seen.has(key)) return;
    seen.add(key);
    emit(x, y, z);
  };
  let last: vec3 | null = null;
  return {
    advanceTo(position: vec3) {
      const current = vec3.clone(position);
      sweepSegment(
        frame,
        last === null ? current : last,
        current,
        radius,
        bounds,
        halfThickness,
        emitOnce,
      );
      last = current;
    },
  };
}

export class VoxelBuffer {
  private data = new Float64Array(3 * 1024);
  length = 0;

  push(x: number, y: number, z: number) {
    if (3 * (this.length + 1) > this.data.length) {
      const grown = new Float64Array(this.data.length * 2);
      grown.set(this.data);
      this.data = grown;
    }
    const at = 3 * this.length++;
    this.data[at] = x;
    this.data[at + 1] = y;
    this.data[at + 2] = z;
  }

  // The next push may invalidate the view.
  view(from = 0): Float64Array {
    return this.data.subarray(3 * from, 3 * this.length);
  }
}

// Canonical order for a whole stroke is the (z, y, x) sort, not the order the
// stroke happens to reach voxels in. Anything indexing a parallel array against
// these voxels depends on it.
export function canonicalOrder(voxels: Float64Array): Uint32Array {
  const order = new Uint32Array(voxels.length / 3);
  for (let i = 0; i < order.length; i++) order[i] = i;
  order.sort(
    (a, b) =>
      voxels[3 * a + 2] - voxels[3 * b + 2] ||
      voxels[3 * a + 1] - voxels[3 * b + 1] ||
      voxels[3 * a] - voxels[3 * b],
  );
  return order;
}

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c;
  }
  return table;
})();

// What the server checks a stroke's voxels against: CRC32 of their indices in
// canonical order, as little-endian int32 x y z, read as a signed integer.
export function strokeChecksum(voxels: Float64Array): number {
  const order = canonicalOrder(voxels);
  const view = new DataView(new ArrayBuffer(12 * order.length));
  for (let i = 0; i < order.length; i++) {
    for (let k = 0; k < 3; k++) {
      view.setInt32(12 * i + 4 * k, Math.floor(voxels[3 * order[i] + k]), true);
    }
  }
  const bytes = new Uint8Array(view.buffer);
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    crc = CRC32_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) | 0;
}

export function stampStrokeVoxels(
  frame: BrushPlaneFrame,
  path: ReadonlyArray<vec3>,
  radius: number,
  bounds: VoxelBounds,
  emit: (x: number, y: number, z: number) => void,
) {
  const buffer = new VoxelBuffer();
  const stroke = beginStroke(frame, radius, bounds, (x, y, z) =>
    buffer.push(x, y, z),
  );
  for (const position of path) stroke.advanceTo(position);

  const voxels = buffer.view();
  for (const i of canonicalOrder(voxels)) {
    emit(voxels[3 * i], voxels[3 * i + 1], voxels[3 * i + 2]);
  }
}
