/**
 * @file Shared current data (voxel) bounds used to constrain region boxes.
 *
 * A region box must stay within the base dataset's bounds. `ensureRegionBox`
 * records the dataset bounds here when creating a region; the oriented-box edit
 * math reads them to clamp extrude/translate so the box can't grow or move
 * outside the data. Per-dimension bounds may be ±Infinity (unbounded), in which
 * case that dimension is not clamped.
 */

export interface RegionDataBounds {
  lower: Float32Array;
  upper: Float32Array;
}

let currentBounds: RegionDataBounds | null = null;

export function setRegionDataBounds(bounds: RegionDataBounds | null) {
  currentBounds = bounds;
}

export function getRegionDataBounds(): RegionDataBounds | null {
  return currentBounds;
}

/**
 * The last perspective-view projection used to render the region gizmo. Captured
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
 * began. The region rotation math uses it to anchor the ring's tangent at the
 * exact point the user clicked, so rotation follows the cursor from any view.
 */
let currentGizmoDragStartNdc: { x: number; y: number } | null = null;

export function setGizmoDragStartNdc(ndc: { x: number; y: number } | null) {
  currentGizmoDragStartNdc = ndc;
}

export function getGizmoDragStartNdc(): { x: number; y: number } | null {
  return currentGizmoDragStartNdc;
}
