/**
 * @file Imperative helpers for the single region (oriented bounding box) layer.
 *
 * Every region box lives in ONE `local://annotations` layer (named
 * `REGIONS_LAYER_NAME`), created once at viewer init and never torn down. Each
 * box is an oriented-bounding-box annotation whose annotation id is the region
 * uuid, so selection rides neuroglancer's native per-annotation
 * `selectedAnnotation` and per-box visibility rides the annotation's `visible`
 * flag (culled in-shader). There is no per-region layer, no name prefix, and no
 * hydrate/teardown reconciler — the layer always exists and is simply populated.
 */

import { AnnotationType } from "#src/annotation/index.js";
import {
  setRegionDataBounds,
  type RegionDataBounds,
} from "#src/annotation/region_bounds.js";
import { makeLayer } from "#src/layer/index.js";
import type { Viewer } from "#src/viewer.js";

/** The single annotation layer that holds every region box. */
export const REGIONS_LAYER_NAME = "regions";

function findRegionsLayer(viewer: Viewer) {
  return viewer.layerManager.managedLayers.find(
    (l) => l.name === REGIONS_LAYER_NAME,
  );
}

function getMutableSource(userLayer: any): any | undefined {
  const states = userLayer?.annotationStates?.states;
  if (!states) return undefined;
  const state = states.find((s: any) => s.source && !s.source.readonly);
  return state?.source;
}

function regionsSource(viewer: Viewer): any | undefined {
  const layer = findRegionsLayer(viewer);
  return layer ? getMutableSource(layer.layer) : undefined;
}

function regionsAnnotationState(userLayer: any): any | undefined {
  const states = userLayer?.annotationStates?.states;
  if (!states) return undefined;
  return states.find((s: any) => s.source && !s.source.readonly) ?? states[0];
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

/** Find or create the single `local://annotations` regions layer. */
export function ensureRegionsLayer(viewer: Viewer) {
  let managed = findRegionsLayer(viewer);
  if (managed === undefined) {
    managed = makeLayer(viewer.layerSpecification, REGIONS_LAYER_NAME, {
      type: "annotation",
      source: "local://annotations",
    });
    viewer.layerSpecification.add(managed);
  }
  return managed;
}

/**
 * Run `cb` with the regions layer, but only once the global coordinate space is
 * valid. A `local://annotations` source snapshots the coordinate space at
 * resolution time, so creating it before the zarr sources establish the space
 * would resolve it against an unestablished space. Returns an unsubscribe.
 */
export function withRegionsLayer(
  viewer: Viewer,
  cb: (layer: ReturnType<typeof ensureRegionsLayer>) => void,
): () => void {
  const coordinateSpace = viewer.navigationState.position.coordinateSpace;
  if (coordinateSpace.value?.valid) {
    cb(ensureRegionsLayer(viewer));
    return () => {};
  }
  const unsubscribe = coordinateSpace.changed.add(() => {
    if (coordinateSpace.value?.valid) {
      unsubscribe();
      cb(ensureRegionsLayer(viewer));
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

function readBox(annotation: any): RegionBoxTransform {
  return {
    center: Array.from(annotation.center.slice(0, 3)),
    extents: Array.from(annotation.extents.slice(0, 3)),
    orientation: Array.from(annotation.orientation),
  };
}

/** Add a box for region `id` to the regions layer, idempotently. */
export function addRegionBox(
  viewer: Viewer,
  id: string,
  t: RegionBoxTransform,
): void {
  const addNow = (source: any): boolean => {
    if (source === undefined) return false;
    // Idempotent across re-runs of the populate effect: skip if it exists.
    const ref = source.getReference(id);
    const exists = ref?.value != null;
    ref?.dispose();
    if (exists) return true;
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
        visible: 1,
        properties: [],
      },
      true,
    );
    return true;
  };

  withRegionsLayer(viewer, (layer) => {
    if (addNow(getMutableSource(layer.layer))) return;

    // Source not materialized yet (layer just created): add once available.
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

export function getRegionBox(
  viewer: Viewer,
  id: string,
): RegionBoxTransform | null {
  const source = regionsSource(viewer);
  if (!source) return null;
  const ref = source.getReference(id);
  const box = ref?.value;
  const result =
    box && box.type === AnnotationType.ORIENTED_BOUNDING_BOX
      ? readBox(box)
      : null;
  ref?.dispose();
  return result;
}

export function getRegionVisible(viewer: Viewer, id: string): boolean {
  const source = regionsSource(viewer);
  if (!source) return true;
  const ref = source.getReference(id);
  const box = ref?.value;
  const visible = box ? (box.visible ?? 1) >= 0.5 : true;
  ref?.dispose();
  return visible;
}

export function listRegionBoxes(
  viewer: Viewer,
): Array<{ id: string } & RegionBoxTransform> {
  const out: Array<{ id: string } & RegionBoxTransform> = [];
  const source = regionsSource(viewer);
  if (!source) return out;
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
  const source = regionsSource(viewer);
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

/** Set a box's per-box visibility flag (re-serializes that one instance). */
export function setRegionVisible(
  viewer: Viewer,
  id: string,
  visible: boolean,
): void {
  const source = regionsSource(viewer);
  if (!source) return;
  const ref = source.getReference(id);
  const existing = ref?.value;
  if (!existing) {
    ref?.dispose();
    return;
  }
  const next = visible ? 1 : 0;
  if (existing.visible === next) {
    ref.dispose();
    return;
  }
  source.update(ref, { ...existing, visible: next });
  source.commit(ref);
  ref.dispose();
}

export function removeRegionBox(viewer: Viewer, id: string): void {
  const source = regionsSource(viewer);
  if (!source) return;
  const ref = source.getReference(id);
  if (ref?.value) source.delete(ref);
  ref?.dispose();
}

/**
 * Subscribe to any change to the regions layer's annotations (geometry edits,
 * visibility toggles, add/remove). Subscribes to the single source's change
 * signal once it materializes; re-checks on layer add/ready.
 */
export function subscribeRegionChanges(
  viewer: Viewer,
  callback: () => void,
): () => void {
  let sourceUnsub: (() => void) | null = null;
  const resync = () => {
    if (!sourceUnsub) {
      const source = regionsSource(viewer);
      if (source?.changed?.add) {
        sourceUnsub = source.changed.add(callback);
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
    if (sourceUnsub) sourceUnsub();
    sourceUnsub = null;
  };
}

/**
 * Externally drive the regions layer's native `hoverState` so hovering a region
 * row in the panel highlights the corresponding box in the viewer, exactly as if
 * the cursor were over it (the box-edge shader brightens the hovered instance).
 * `partIndex` 0 = the whole object (non-interactive). Pass null to clear.
 */
export function setRegionHover(viewer: Viewer, id: string | null): void {
  const managed = findRegionsLayer(viewer);
  const layer: any = managed?.layer;
  const displayState = layer?.annotationDisplayState;
  if (!displayState?.hoverState) return;
  if (id === null) {
    displayState.hoverState.value = undefined;
  } else {
    const state = regionsAnnotationState(layer);
    if (!state) return;
    displayState.hoverState.value = {
      id,
      partIndex: 0,
      annotationLayerState: state,
    };
  }
  (viewer as any).display?.scheduleRedraw?.();
}

/**
 * Reflect the React-owned region selection into the regions layer's native
 * `selectedAnnotation`: the selected box (id) shows its gizmo handles, all
 * others render plain. `controlledSelection` gates neuroglancer's hover/pick so
 * it can't clobber this externally-owned value. Pass null to deselect.
 */
export function setRegionSelection(viewer: Viewer, id: string | null): void {
  const managed = findRegionsLayer(viewer);
  const displayState = (managed?.layer as any)?.annotationDisplayState;
  if (!displayState?.selectedAnnotation) return;
  displayState.controlledSelection = true;
  displayState.selectedAnnotation.value = id ?? undefined;
}
