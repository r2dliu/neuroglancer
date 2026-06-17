/**
 * @file Imperative helpers for region (oriented bounding box) creation.
 *
 */

import { AnnotationType, makeAnnotationId } from "#src/annotation/index.js";
import {
  setRegionDataBounds,
  type RegionDataBounds,
} from "#src/annotation/region_bounds.js";
import { makeLayer } from "#src/layer/index.js";
import type { Viewer } from "#src/viewer.js";

export const REGION_LAYER_NAME = "regions";

// Default box edge length as a fraction of the dataset extent (per dimension).
const DEFAULT_FRACTION = 0.5;
// Fallback edge length (voxels) for dimensions whose data bounds are unbounded.
const FALLBACK_EXTENT = 128;

function findRegionLayer(viewer: Viewer) {
  return viewer.layerManager.managedLayers.find(
    (l) => l.name === REGION_LAYER_NAME,
  );
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

function getMutableSource(userLayer: any): any | undefined {
  const states = userLayer?.annotationStates?.states;
  if (!states) return undefined;
  const state = states.find((s: any) => s.source && !s.source.readonly);
  return state?.source;
}

function sourceHasRegion(source: any): boolean {
  for (const annotation of source) {
    if (annotation.type === AnnotationType.ORIENTED_BOUNDING_BOX) return true;
  }
  return false;
}

function addDefaultBox(
  viewer: Viewer,
  source: any,
  bounds: RegionDataBounds | null,
) {
  const rank: number = source.rank;
  const position = viewer.navigationState.position.value;
  const center = new Float32Array(rank);
  const extents = new Float32Array(rank);
  for (let i = 0; i < rank; ++i) {
    const lo = bounds?.lower[i];
    const hi = bounds?.upper[i];
    if (
      lo !== undefined &&
      hi !== undefined &&
      Number.isFinite(lo) &&
      Number.isFinite(hi) &&
      hi > lo
    ) {
      center[i] = (lo + hi) / 2;
      extents[i] = (hi - lo) * DEFAULT_FRACTION;
    } else {
      center[i] = position[i] ?? 0;
      extents[i] = FALLBACK_EXTENT;
    }
  }
  source.add(
    {
      type: AnnotationType.ORIENTED_BOUNDING_BOX,
      id: makeAnnotationId(),
      center,
      extents,
      orientation: Float32Array.of(0, 0, 0, 1),
      properties: [],
    },
    true,
  );
}

/**
 * Ensure the region layer exists and contains a region box, creating either as
 * needed. No-op (beyond ensuring the layer) if a region box already exists.
 * Waits for the global coordinate space to become valid before doing anything.
 * Remove once Ichnaea backend / frontend takes control
 */
export function ensureRegionBox(viewer: Viewer): void {
  const coordinateSpace = viewer.navigationState.position.coordinateSpace;
  if (!coordinateSpace.value?.valid) {
    const unsubscribe = coordinateSpace.changed.add(() => {
      if (coordinateSpace.value?.valid) {
        unsubscribe();
        ensureRegionBox(viewer);
      }
    });
    return;
  }

  const bounds = readDataBounds(viewer);
  if (bounds) setRegionDataBounds(bounds);

  const spec = viewer.layerSpecification;
  let managed = findRegionLayer(viewer);
  if (managed === undefined) {
    managed = makeLayer(spec, REGION_LAYER_NAME, {
      type: "annotation",
      source: "local://annotations",
    });
    spec.add(managed);
  }
  const layer = managed;

  const unsubscribers: (() => void)[] = [];
  const cleanup = () => {
    while (unsubscribers.length) unsubscribers.pop()!();
  };

  const attempt = (): boolean => {
    const source = getMutableSource(layer.layer);
    if (source === undefined) return false;
    if (!sourceHasRegion(source)) addDefaultBox(viewer, source, bounds);
    return true;
  };

  if (attempt()) {
    cleanup();
    return;
  }

  const retry = () => {
    if (attempt()) cleanup();
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
