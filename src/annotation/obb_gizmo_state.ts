/**
 * @file Shared mutable state for editing oriented-bounding-box gizmos: the data
 * (voxel) bounds an edit is clamped within, plus the gizmo projection and
 * drag-start anchor used by the rotation math.
 *
 * A box must stay within the base dataset's bounds. `captureDataBounds` records
 * them here when an edit begins; the oriented-box edit math reads them to clamp
 * extrude/translate so the box can't grow or move outside the data.
 * Per-dimension bounds may be ±Infinity (unbounded), in which case that
 * dimension is not clamped.
 */

import type { Viewer } from "#src/viewer.js";

export interface DataBounds {
  lower: Float32Array;
  upper: Float32Array;
}

let currentBounds: DataBounds | null = null;

export function setDataBounds(bounds: DataBounds | null) {
  currentBounds = bounds;
}

export function getDataBounds(): DataBounds | null {
  return currentBounds;
}

function readDataBounds(viewer: Viewer): DataBounds | null {
  const space = viewer.navigationState.position.coordinateSpace.value;
  if (!space?.valid || !space.bounds) return null;
  const { lowerBounds, upperBounds } = space.bounds;
  const rank = space.rank;
  const lower = new Float32Array(rank);
  const upper = new Float32Array(rank);
  for (let i = 0; i < rank; ++i) {
    lower[i] = lowerBounds[i];
    upper[i] = upperBounds[i];
  }
  return { lower, upper };
}

/**
 * Capture the dataset's voxel bounds into the shared edit state so the gizmo
 * can clamp edits. Waits for the global coordinate space to become valid.
 */
export function captureDataBounds(viewer: Viewer): void {
  const coordinateSpace = viewer.navigationState.position.coordinateSpace;
  if (!coordinateSpace.value?.valid) {
    const unsubscribe = coordinateSpace.changed.add(() => {
      if (coordinateSpace.value?.valid) {
        unsubscribe();
        captureDataBounds(viewer);
      }
    });
    return;
  }
  const bounds = readDataBounds(viewer);
  if (bounds) setDataBounds(bounds);
}

/**
 * The last perspective-view projection used to render the box gizmo. Captured
 * during rendering and read by the rotation edit math so it can work in screen
 * space (which makes ring rotation follow the cursor regardless of axis
 * orientation or a handedness-flipped display subspace). `mvp` maps the render
 * subspace to clip space; `subspaceMatrix` (rank*3) maps model/data coords to
 * the subspace; `aspect` = width/height (to make screen coords pixel-isotropic).
 */
export interface GizmoProjection {
  mvp: Float32Array;
  subspaceMatrix: Float32Array;
  aspect: number;
  // Per-display-axis world size (uGizmoAxisWorld), so the rotation math can
  // reconstruct the on-screen ring geometry exactly.
  axisWorld: Float32Array;
  // Orthonormal rotation (column-major 3x3) mapping the render subspace to view
  // (camera) space. Its transpose maps a camera-space axis back to the subspace,
  // which the free (edge-drag) trackball uses so its rotation axis follows the
  // current camera and never reverses.
  viewRot: Float32Array;
}

let currentGizmoProjection: GizmoProjection | null = null;

export function setGizmoProjection(projection: GizmoProjection | null) {
  currentGizmoProjection = projection;
}

export function getGizmoProjection(): GizmoProjection | null {
  return currentGizmoProjection;
}

/**
 * Normalized-device-coordinate position (y up, [-1,1]) where an annotation drag
 * began. The box rotation math uses it to anchor the ring's tangent at the
 * exact point the user clicked, so rotation follows the cursor from any view.
 */
let currentGizmoDragStartNdc: { x: number; y: number } | null = null;

export function setGizmoDragStartNdc(ndc: { x: number; y: number } | null) {
  currentGizmoDragStartNdc = ndc;
}

export function getGizmoDragStartNdc(): { x: number; y: number } | null {
  return currentGizmoDragStartNdc;
}
