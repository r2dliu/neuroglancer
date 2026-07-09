/**
 * Shared in-plane stamp math for the brush and eraser tools.
 */

import { clampAndRoundCoordinateToVoxelCenter } from "#src/coordinate_transform.js";
import { mat4, vec3, type quat } from "#src/util/geom.js";

export interface BrushPlaneFrame {
  /** Voxel-index offset of a +1 canonical-unit step along the viewport right
   *  axis, indexed by global spatial dim. */
  dirU: vec3;
  /** Same, along the viewport up axis. */
  dirV: vec3;
  /** Slice-plane normal in voxel-index space (unit). */
  normal: vec3;
  /** Project a voxel-index displacement onto the plane, in canonical units —
   *  used to pace stroke interpolation to match on-screen distance. */
  toCanonical(delta: vec3): [number, number];
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

type VoxelBounds = Parameters<typeof clampAndRoundCoordinateToVoxelCenter>[0];

export function brushPlaneFrame(pose: PoseLike): BrushPlaneFrame {
  const m = mat4.fromQuat(mat4.create(), pose.orientation.orientation);
  const { canonicalVoxelFactors: f, displayDimensionIndices: dd } =
    pose.displayDimensionRenderInfo.value;
  const dirU = vec3.create();
  const dirV = vec3.create();
  // Row i of the rotation matrix is display dimension i; map each onto its
  // global spatial dim (identity for the [x, y, z] spaces this app loads).
  for (let i = 0; i < 3; i++) {
    const g = dd[i];
    if (g < 0 || g > 2) continue;
    dirU[g] = m[i] / f[i];
    dirV[g] = m[4 + i] / f[i];
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
    toCanonical(delta: vec3): [number, number] {
      let du = 0;
      let dv = 0;
      for (let i = 0; i < 3; i++) {
        const g = dd[i];
        if (g < 0 || g > 2) continue;
        const c = delta[g] * f[i];
        du += c * m[i];
        dv += c * m[4 + i];
      }
      return [du, dv];
    },
  };
}

// In-plane sampling step in canonical units. canonicalVoxelFactors are >= 1
// (canonical = the finest physical scale), so one canonical unit moves at most
// one voxel along any axis; half-steps keep adjacent samples within one voxel
// of each other, and the ±1 dilation below covers the cubes in between.
const STAMP_STEP = 0.5;

// 3x3x3 neighborhood for the dilation.
const NEIGHBOR_OFFSETS: ReadonlyArray<readonly [number, number, number]> = (() => {
  const offsets: Array<[number, number, number]> = [];
  for (let dx = -1; dx <= 1; dx++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dz = -1; dz <= 1; dz++) {
        offsets.push([dx, dy, dz]);
      }
    }
  }
  return offsets;
})();

/**
 * Emit the voxel centers of every voxel the slice plane crosses within the
 * radius-`radius` (canonical units) brush circle around `center` (voxel-index
 * coords) — the exact voxel set the plane RENDERS, so the painted disk shows
 * no gaps under the cursor. Deliberate trade-off: a crossed-cube slab on an
 * oblique plane is thicker than one slice spacing, so partial slivers of the
 * stroke show on the adjacent slices (the same voxels genuinely span both).
 * The alternative (claiming each voxel only for its nearest slice) leaves
 * visible pinholes on the painted slice, which reads as broken.
 * Callers dedupe (overlapping interpolated stamps re-emit).
 */
export function stampDiskVoxels(
  frame: BrushPlaneFrame,
  center: vec3,
  radius: number,
  bounds: VoxelBounds,
  emit: (x: number, y: number, z: number) => void,
) {
  const { dirU, dirV, normal } = frame;
  // A unit cube is crossed by the plane when its center is within the cube's
  // support along the normal (matches the renderer and the backend's
  // voxelize_slice_mask).
  const halfThickness =
    0.5 *
      (Math.abs(normal[0]) + Math.abs(normal[1]) + Math.abs(normal[2])) +
    1e-4;
  const radiusSq = radius * radius;
  const pad = radius + STAMP_STEP;
  const padSq = pad * pad;
  const n = Math.ceil(pad / STAMP_STEP);
  const pos = vec3.create();
  const cand = vec3.create();
  const delta = vec3.create();
  const visitedBase = new Set<string>();
  const { lowerBounds, upperBounds } = bounds;
  for (let iu = -n; iu <= n; iu++) {
    const du = iu * STAMP_STEP;
    for (let iv = -n; iv <= n; iv++) {
      const dv = iv * STAMP_STEP;
      if (du * du + dv * dv > padSq) continue;
      vec3.scaleAndAdd(pos, center, dirU, du);
      vec3.scaleAndAdd(pos, pos, dirV, dv);
      const bx = clampAndRoundCoordinateToVoxelCenter(bounds, 0, pos[0]);
      const by = clampAndRoundCoordinateToVoxelCenter(bounds, 1, pos[1]);
      const bz = clampAndRoundCoordinateToVoxelCenter(bounds, 2, pos[2]);
      const baseKey = `${bx},${by},${bz}`;
      if (visitedBase.has(baseKey)) continue;
      visitedBase.add(baseKey);
      for (const [ox, oy, oz] of NEIGHBOR_OFFSETS) {
        vec3.set(cand, bx + ox, by + oy, bz + oz);
        if (
          cand[0] < lowerBounds[0] ||
          cand[0] >= upperBounds[0] ||
          cand[1] < lowerBounds[1] ||
          cand[1] >= upperBounds[1] ||
          cand[2] < lowerBounds[2] ||
          cand[2] >= upperBounds[2]
        ) {
          continue;
        }
        vec3.subtract(delta, cand, center);
        // The plane must cross this voxel's cube…
        if (Math.abs(vec3.dot(delta, normal)) > halfThickness) continue;
        // …and its center must sit inside the cursor circle.
        const [cu, cv] = frame.toCanonical(delta);
        if (cu * cu + cv * cv > radiusSq) continue;
        emit(cand[0], cand[1], cand[2]);
      }
    }
  }
}
