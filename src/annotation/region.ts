/**
 * @file Imperative helpers for region (oriented bounding box) layers.
 * Each region box is its OWN neuroglancer annotation layer, named
 * `region:<uuid>` (the box's annotation id is the same uuid)
 */

import { AnnotationType } from "#src/annotation/index.js";
import {
  setRegionDataBounds,
  type RegionDataBounds,
} from "#src/annotation/region_bounds.js";
import { makeLayer } from "#src/layer/index.js";
import type { Viewer } from "#src/viewer.js";

export const REGION_LAYER_PREFIX = "region:";

export function regionLayerName(id: string): string {
  return REGION_LAYER_PREFIX + id;
}

export function regionIdFromLayerName(name: string): string | null {
  return name.startsWith(REGION_LAYER_PREFIX)
    ? name.slice(REGION_LAYER_PREFIX.length)
    : null;
}

function findRegionLayer(viewer: Viewer, id: string) {
  const name = regionLayerName(id);
  return viewer.layerManager.managedLayers.find((l) => l.name === name);
}

function listRegionManagedLayers(viewer: Viewer) {
  return viewer.layerManager.managedLayers.filter((l) =>
    l.name.startsWith(REGION_LAYER_PREFIX),
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

/** Find or create the annotation layer for region `id`. */
export function ensureRegionLayer(viewer: Viewer, id: string) {
  let managed = findRegionLayer(viewer, id);
  if (managed === undefined) {
    managed = makeLayer(viewer.layerSpecification, regionLayerName(id), {
      type: "annotation",
      source: "local://annotations",
    });
    viewer.layerSpecification.add(managed);
  }
  return managed;
}

/**
 * Create `local://annotations` only after data layers establish bounds
 * Otherwise, it loads defaults which do not fit the viewport
 */
function withRegionLayer(
  viewer: Viewer,
  id: string,
  cb: (layer: ReturnType<typeof ensureRegionLayer>) => void,
): () => void {
  const coordinateSpace = viewer.navigationState.position.coordinateSpace;
  if (coordinateSpace.value?.valid) {
    cb(ensureRegionLayer(viewer, id));
    return () => { };
  }
  const unsubscribe = coordinateSpace.changed.add(() => {
    if (coordinateSpace.value?.valid) {
      unsubscribe();
      cb(ensureRegionLayer(viewer, id));
    }
  });
  return unsubscribe;
}

export interface RegionBoxTransform {
  center: number[];
  extents: number[];
  orientation: number[];
}

export function computeDefaultRegionTransform(
  viewer: Viewer,
): RegionBoxTransform {
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
  const addNow = (source: any): boolean => {
    if (source === undefined) return false;
    // Idempotent across retries / re-hydration: one box per layer, so if the
    // layer already has its box we're done
    if (firstBox(source) !== undefined) return true;
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

  withRegionLayer(viewer, id, (layer) => {
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
  });
}

function regionSourceFor(viewer: Viewer, id: string): any | undefined {
  const layer = findRegionLayer(viewer, id);
  return layer ? getMutableSource(layer.layer) : undefined;
}

function readBox(annotation: any): RegionBoxTransform {
  return {
    center: Array.from(annotation.center.slice(0, 3)),
    extents: Array.from(annotation.extents.slice(0, 3)),
    orientation: Array.from(annotation.orientation),
  };
}

function firstBox(source: any): any | undefined {
  for (const annotation of source) {
    if (annotation.type === AnnotationType.ORIENTED_BOUNDING_BOX) {
      return annotation;
    }
  }
  return undefined;
}

export function getRegionBox(
  viewer: Viewer,
  id: string,
): RegionBoxTransform | null {
  const source = regionSourceFor(viewer, id);
  if (!source) return null;
  const box = firstBox(source);
  return box ? readBox(box) : null;
}

export function listRegionBoxes(
  viewer: Viewer,
): Array<{ id: string } & RegionBoxTransform> {
  const out: Array<{ id: string } & RegionBoxTransform> = [];
  for (const managed of listRegionManagedLayers(viewer)) {
    const id = regionIdFromLayerName(managed.name);
    if (id === null) continue;
    const source = getMutableSource(managed.layer);
    const box = source ? firstBox(source) : undefined;
    if (box) out.push({ id, ...readBox(box) });
  }
  return out;
}

export function setRegionBox(
  viewer: Viewer,
  id: string,
  t: RegionBoxTransform,
): void {
  const source = regionSourceFor(viewer, id);
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

export function removeRegionBox(viewer: Viewer, id: string): void {
  const managed = findRegionLayer(viewer, id);
  if (managed) viewer.layerManager.removeManagedLayer(managed);
}

/**
 * Subscribe to any change to any region box (geometry edits via the gizmo, and
 * layers being added/removed). Re-derives the per-layer source subscriptions
 * whenever the layer set changes.
 */
export function subscribeRegionChanges(
  viewer: Viewer,
  callback: () => void,
): () => void {
  const sourceUnsubs = new Map<string, () => void>();

  const resync = () => {
    const present = new Set<string>();
    for (const managed of listRegionManagedLayers(viewer)) {
      present.add(managed.name);
      if (sourceUnsubs.has(managed.name)) continue;
      const source = getMutableSource(managed.layer);
      if (source?.changed?.add) {
        sourceUnsubs.set(managed.name, source.changed.add(callback));
      }
    }
    // Drop subscriptions for layers that disappeared.
    for (const [name, unsub] of [...sourceUnsubs]) {
      if (!present.has(name)) {
        unsub();
        sourceUnsubs.delete(name);
      }
    }
    callback();
  };

  resync();
  const unsubLayers = viewer.layerManager.layersChanged.add(resync);
  const unsubReady = viewer.layerManager.readyStateChanged.add(resync);
  return () => {
    unsubLayers();
    unsubReady();
    for (const unsub of sourceUnsubs.values()) unsub();
    sourceUnsubs.clear();
  };
}

/**
 * Reflect layer selection into the gizmo by filling each region layer's native
 * `selectedAnnotation`: the selected region's box (one box per layer) shows its
 * handles, all others are cleared. Reuses the upstream render path (the oriented
 * box render layer reads `selectedAnnotation`); `id` is the selected region's
 * uuid, or null for none.
 */
export function setRegionSelection(viewer: Viewer, id: string | null): void {
  for (const managed of listRegionManagedLayers(viewer)) {
    const layerId = regionIdFromLayerName(managed.name);
    const displayState = (managed.layer as any)?.annotationDisplayState;
    if (!displayState?.selectedAnnotation) continue;
    displayState.selectedAnnotation.value =
      layerId !== null && layerId === id ? layerId : undefined;
  }
}
