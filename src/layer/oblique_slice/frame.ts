import type { NavigationState } from "#src/navigation_state.js";
import { quat, vec3 } from "#src/util/geom.js";

const SPATIAL_NAMES = ["x", "y", "z"];
const UNIT_X = vec3.fromValues(1, 0, 0);
const UNIT_Y = vec3.fromValues(0, 1, 0);
const UNIT_Z = vec3.fromValues(0, 0, 1);

export interface SlicerPose {
  position: number[];
  orientation: number[];
}

// The slicer in display coordinates: center, in-plane axes u / v, normal n and
// the dataset box. `global` is the global position with the slicer's xyz.
export interface SlicerFrame {
  global: Float32Array;
  displayDims: number[];
  center: vec3;
  rotation: quat;
  u: vec3;
  v: vec3;
  n: vec3;
  lower: vec3;
  upper: vec3;
}

export function slicerFrame(
  pose: SlicerPose,
  nav: NavigationState,
): SlicerFrame | undefined {
  const space = nav.coordinateSpace.value;
  if (!space.valid || space.bounds === undefined) return undefined;
  const displayDims = Array.from(
    nav.pose.displayDimensionRenderInfo.value.displayDimensionIndices,
  );
  if (displayDims.includes(-1)) return undefined;
  const global = Float32Array.from(nav.position.value);
  SPATIAL_NAMES.forEach((name, i) => {
    const dim = space.names.indexOf(name);
    if (dim !== -1) global[dim] = pose.position[i];
  });
  const center = vec3.create();
  const lower = vec3.create();
  const upper = vec3.create();
  for (let i = 0; i < 3; ++i) {
    const dim = displayDims[i];
    center[i] = global[dim];
    lower[i] = space.bounds.lowerBounds[dim];
    upper[i] = space.bounds.upperBounds[dim];
  }
  const rotation = quat.normalize(
    quat.create(),
    pose.orientation as unknown as quat,
  );
  return {
    global,
    displayDims,
    center,
    rotation,
    u: vec3.transformQuat(vec3.create(), UNIT_X, rotation),
    v: vec3.transformQuat(vec3.create(), UNIT_Y, rotation),
    n: vec3.transformQuat(vec3.create(), UNIT_Z, rotation),
    lower,
    upper,
  };
}

export function displayToSlicerPosition(
  frame: SlicerFrame,
  nav: NavigationState,
  point: vec3,
  current: number[],
): number[] {
  const { names } = nav.coordinateSpace.value;
  const out = [...current];
  for (let i = 0; i < 3; ++i) {
    const k = SPATIAL_NAMES.indexOf(names[frame.displayDims[i]]);
    if (k !== -1) out[k] = point[i];
  }
  return out;
}

function forEachBoxEdge(
  frame: SlicerFrame,
  callback: (a: vec3, b: vec3) => void,
) {
  const { lower, upper } = frame;
  const a = vec3.create();
  const b = vec3.create();
  for (let corner = 0; corner < 8; ++corner) {
    for (let axis = 0; axis < 3; ++axis) {
      if (corner & (1 << axis)) continue;
      for (let i = 0; i < 3; ++i) {
        a[i] = (corner >> i) & 1 ? upper[i] : lower[i];
        b[i] = i === axis ? upper[i] : a[i];
      }
      callback(a, b);
    }
  }
}

const tempOffset = vec3.create();

function signedDistance(frame: SlicerFrame, point: vec3) {
  return vec3.dot(frame.n, vec3.subtract(tempOffset, point, frame.center));
}

// Where the plane `offset` along the normal cuts the dataset box, ordered
// around its center.
export function capPolygon(frame: SlicerFrame, offset: number): vec3[] {
  const points: vec3[] = [];
  forEachBoxEdge(frame, (a, b) => {
    const da = signedDistance(frame, a) - offset;
    const db = signedDistance(frame, b) - offset;
    if (da * db > 0 || da === db) return;
    points.push(vec3.lerp(vec3.create(), a, b, da / (da - db)));
  });
  if (points.length === 0) return points;
  const mid = vec3.create();
  for (const p of points) vec3.add(mid, mid, p);
  vec3.scale(mid, mid, 1 / points.length);
  const angle = (p: vec3) => {
    vec3.subtract(tempOffset, p, mid);
    return Math.atan2(
      vec3.dot(frame.v, tempOffset),
      vec3.dot(frame.u, tempOffset),
    );
  };
  return points.sort((p, q) => angle(p) - angle(q));
}

// The dataset box's edges clipped to the slab |distance| <= halfThickness: the
// side edges of the slab's prism.
export function slabSideEdges(
  frame: SlicerFrame,
  halfThickness: number,
): [vec3, vec3][] {
  const edges: [vec3, vec3][] = [];
  forEachBoxEdge(frame, (a, b) => {
    const da = signedDistance(frame, a);
    const db = signedDistance(frame, b);
    let t0 = 0;
    let t1 = 1;
    if (da === db) {
      if (Math.abs(da) > halfThickness) return;
    } else {
      const ta = (-halfThickness - da) / (db - da);
      const tb = (halfThickness - da) / (db - da);
      t0 = Math.max(t0, Math.min(ta, tb));
      t1 = Math.min(t1, Math.max(ta, tb));
      if (t0 >= t1) return;
    }
    edges.push([
      vec3.lerp(vec3.create(), a, b, t0),
      vec3.lerp(vec3.create(), a, b, t1),
    ]);
  });
  return edges;
}
