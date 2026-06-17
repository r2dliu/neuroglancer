/**
 * @file Imperative helpers for region (oriented bounding box) layers.
 *
 */

import { AnnotationType } from "#src/annotation/index.js";
import {
  setRegionDataBounds,
  type RegionDataBounds,
} from "#src/annotation/region_bounds.js";
import { makeLayer } from "#src/layer/index.js";
import type { Viewer } from "#src/viewer.js";

export const REGION_LAYER_NAME = "regions";

function findRegionLayer(viewer: Viewer) {
  return viewer.layerManager.managedLayers.find(
    (l) => l.name === REGION_LAYER_NAME,
  );
}

function getMutableSource(userLayer: any): any | undefined {
  const states = userLayer?.annotationStates?.states;
  if (!states) return undefined;
  const state = states.find((s: any) => s.source && !s.source.readonly);
  return state?.source;
}

function readDataBounds(viewer: Viewer): RegionDataBounds | null {
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
 * Capture the dataset's voxel bounds into the shared region state so the gizmo
 * can clamp edits. Waits for the global coordinate space to become valid.
 */
export function captureRegionDataBounds(viewer: Viewer): void {
  const coordinateSpace = viewer.navigationState.position.coordinateSpace;
  if (!coordinateSpace.value?.valid) {
    const unsubscribe = coordinateSpace.changed.add(() => {
      if (coordinateSpace.value?.valid) {
        unsubscribe();
        captureRegionDataBounds(viewer);
      }
    });
    return;
  }
  const bounds = readDataBounds(viewer);
  if (bounds) setRegionDataBounds(bounds);
}

// Initial box edge length as a fraction of the dataset extent (per dimension) —
// a small box near the current view, not the whole volume.
const DEFAULT_VIEW_FRACTION = 0.2;
// Fallback edge length (voxels) for dimensions whose data bounds are unbounded.
const FALLBACK_EXTENT = 64;

/** Find or create the single shared "regions" annotation layer. */
export function ensureRegionLayer(viewer: Viewer) {
  let managed = findRegionLayer(viewer);
  if (managed === undefined) {
    managed = makeLayer(viewer.layerSpecification, REGION_LAYER_NAME, {
      type: "annotation",
      source: "local://annotations",
    });
    viewer.layerSpecification.add(managed);
  }
  return managed;
}

export interface RegionBoxTransform {
  center: number[];
  extents: number[];
  orientation: number[];
}

export function computeDefaultRegionTransform(viewer: Viewer): RegionBoxTransform {
  captureRegionDataBounds(viewer);
  const bounds = readDataBounds(viewer);
  const position = viewer.navigationState.position.value;
  const center: number[] = [];
  const extents: number[] = [];
  for (let i = 0; i < 3; ++i) {
    center[i] = position[i] ?? 0;
    const lo = bounds?.lower[i];
    const hi = bounds?.upper[i];
    const bounded =
      lo !== undefined &&
      hi !== undefined &&
      Number.isFinite(lo) &&
      Number.isFinite(hi) &&
      hi > lo;
    extents[i] = bounded ? (hi! - lo!) * DEFAULT_VIEW_FRACTION : FALLBACK_EXTENT;
  }
  return { center, extents, orientation: [0, 0, 0, 1] };
}

export function addRegionBoxWithId(
  viewer: Viewer,
  id: string,
  t: RegionBoxTransform,
): void {
  const layer = ensureRegionLayer(viewer);
  const addNow = (source: any): boolean => {
    if (source === undefined) return false;
    const rank: number = source.rank;
    const center = new Float32Array(rank);
    const extents = new Float32Array(rank);
    for (let i = 0; i < 3 && i < rank; ++i) {
      center[i] = t.center[i];
      extents[i] = t.extents[i];
    }
    source.add(
      {
        type: AnnotationType.ORIENTED_BOUNDING_BOX,
        id,
        center,
        extents,
        orientation: Float32Array.from(t.orientation),
        properties: [],
      },
      true,
    );
    return true;
  };

  if (addNow(getMutableSource(layer.layer))) return;

  // Source not ready yet (layer just created): add once it becomes available.
  const unsubscribers: (() => void)[] = [];
  const cleanup = () => {
    while (unsubscribers.length) unsubscribers.pop()!();
  };
  const retry = () => {
    if (addNow(getMutableSource(layer.layer))) cleanup();
  };
  unsubscribers.push(layer.readyStateChanged.add(retry));
  unsubscribers.push(
    layer.layerChanged.add(() => {
      const states = (layer.layer as any)?.annotationStates;
      if (states?.changed) unsubscribers.push(states.changed.add(retry));
      retry();
    }),
  );
}

function regionSource(viewer: Viewer): any | undefined {
  const layer = findRegionLayer(viewer);
  return layer ? getMutableSource(layer.layer) : undefined;
}

function readBox(annotation: any): RegionBoxTransform {
  return {
    center: Array.from(annotation.center.slice(0, 3)),
    extents: Array.from(annotation.extents.slice(0, 3)),
    orientation: Array.from(annotation.orientation),
  };
}

export function getRegionBox(
  viewer: Viewer,
  id: string,
): RegionBoxTransform | null {
  const source = regionSource(viewer);
  if (!source) return null;
  for (const annotation of source) {
    if (
      annotation.id === id &&
      annotation.type === AnnotationType.ORIENTED_BOUNDING_BOX
    ) {
      return readBox(annotation);
    }
  }
  return null;
}

export function listRegionBoxes(
  viewer: Viewer,
): Array<{ id: string } & RegionBoxTransform> {
  const source = regionSource(viewer);
  if (!source) return [];
  const out: Array<{ id: string } & RegionBoxTransform> = [];
  for (const annotation of source) {
    if (annotation.type === AnnotationType.ORIENTED_BOUNDING_BOX) {
      out.push({ id: annotation.id, ...readBox(annotation) });
    }
  }
  return out;
}

export function setRegionBox(
  viewer: Viewer,
  id: string,
  t: RegionBoxTransform,
): void {
  const source = regionSource(viewer);
  if (!source) return;
  const ref = source.getReference(id);
  const existing = ref?.value;
  if (!existing) {
    ref?.dispose();
    return;
  }
  const rank: number = existing.center.length;
  const center = Float32Array.from(existing.center);
  const extents = Float32Array.from(existing.extents);
  for (let i = 0; i < 3 && i < rank; ++i) {
    center[i] = t.center[i];
    extents[i] = t.extents[i];
  }
  source.update(ref, {
    ...existing,
    center,
    extents,
    orientation: Float32Array.from(t.orientation),
  });
  source.commit(ref);
  ref.dispose();
}

/** Remove a box by annotation id. */
export function removeRegionBox(viewer: Viewer, id: string): void {
  const source = regionSource(viewer);
  if (!source) return;
  const ref = source.getReference(id);
  if (ref?.value) source.delete(ref);
  ref?.dispose();
}

export function subscribeRegionChanges(
  viewer: Viewer,
  callback: () => void,
): () => void {
  const layer = ensureRegionLayer(viewer);
  const unsubscribers: (() => void)[] = [];
  const subscribeSource = () => {
    const source = getMutableSource(layer.layer);
    if (source?.changed?.add) unsubscribers.push(source.changed.add(callback));
  };
  subscribeSource();
  unsubscribers.push(layer.readyStateChanged.add(subscribeSource));
  return () => {
    while (unsubscribers.length) unsubscribers.pop()!();
  };
}

/**
 * Reflect the (Ichnaea-owned) selected box into neuroglancer's built-in
 * selection state: the gizmo renders only on the selected annotation, and the
 * selection-details panel stays in sync. Passing null clears this layer's
 * annotation selection.
 */
export function setRegionSelection(
  viewer: Viewer,
  id: string | null,
): void {
  const managed = findRegionLayer(viewer);
  const layer: any = managed?.layer;
  const annotationState = layer?.annotationStates?.states?.find(
    (s: any) => s.source && !s.source.readonly,
  );
  if (!annotationState) return;
  if (id !== null && typeof layer.selectAnnotation === "function") {
    layer.selectAnnotation(annotationState, id, false);
  } else {
    // Clear this layer's annotation selection.
    layer.manager.root.selectionState.captureSingleLayerState(
      layer,
      (state: any) => {
        state.annotationId = undefined;
        return true;
      },
      false,
    );
  }
}
