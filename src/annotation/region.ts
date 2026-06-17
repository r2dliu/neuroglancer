/**
 * @license
 * Copyright 2024 Ichnaea.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file Imperative helpers for region (oriented bounding box) creation.
 *
 * A "region" lives in a dedicated local annotation layer as a single
 * OrientedBoundingBox. `ensureRegionBox` lazily creates that layer + a default
 * box if one doesn't exist. The default box is centered in the base dataset and
 * sized to a fraction of it; the dataset's voxel bounds are recorded (see
 * region_bounds) so edits can be clamped to stay inside the data.
 *
 * Important: creation waits for the global coordinate space to be valid. The
 * tool can activate during initial load (e.g. restored `activeTool`), and doing
 * any work before the coordinate space loads — in particular reading the
 * zoom — materializes a default zoom against a rank-0 canonical voxel size of
 * 1, which is then rescaled to garbage once the real (sub-micron) voxel size
 * arrives. Gating on validity avoids that entirely.
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

// The dataset's voxel bounds from the global coordinate space. Caller must
// ensure the space is valid. Per-dimension bounds may be ±Infinity.
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

// The mutable (local) annotation source of an annotation user layer, if ready.
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
      orientation: Float32Array.of(0, 0, 0, 1), // identity quaternion
      properties: [],
    },
    /*commit=*/ true,
  );
}

/**
 * Ensure the region layer exists and contains a region box, creating either as
 * needed. No-op (beyond ensuring the layer) if a region box already exists.
 * Waits for the global coordinate space to become valid before doing anything.
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
