/**
 * @license
 * Copyright 2026 Google Inc.
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

import { describe, expect, it } from "vitest";
import type { ChunkManager } from "#src/chunk_manager/frontend.js";
import { TrackableCoordinateSpace } from "#src/coordinate_transform.js";
import { DataPanelLayoutSpecification } from "#src/data_panel_layout.js";
import type { LayerListSpecification } from "#src/layer/index.js";
import { layerTypes, ManagedUserLayer } from "#src/layer/index.js";
import { SliceUserLayer } from "#src/layer/slice/index.js";
import {
  CrossSectionVolumeRenderingMode,
  DisplayPose,
  NavigationLinkType,
  NavigationState,
  OrientationState,
  Position,
  TrackableCrossSectionVoxelRange,
  TrackableCrossSectionZoom,
  TrackableDepthRange,
  TrackableDisplayDimensions,
  TrackableRelativeDisplayScales,
  WatchableDisplayDimensionRenderInfo,
} from "#src/navigation_state.js";
import { GlobalToolBinder } from "#src/ui/tool.js";
import { RefCounted } from "#src/util/disposable.js";
import { NullarySignal } from "#src/util/signal.js";

function makeNavigationState() {
  const coordinateSpace = new TrackableCoordinateSpace();
  coordinateSpace.restoreState({
    x: [1, ""],
    y: [1, ""],
    z: [1, ""],
  });
  const displayDimensionRenderInfo = new WatchableDisplayDimensionRenderInfo(
    new TrackableRelativeDisplayScales(coordinateSpace),
    new TrackableDisplayDimensions(coordinateSpace),
  );
  return new NavigationState(
    new DisplayPose(
      new Position(coordinateSpace),
      displayDimensionRenderInfo,
      new OrientationState(),
    ),
    new TrackableCrossSectionZoom(displayDimensionRenderInfo.addRef()),
    new TrackableDepthRange(-10, displayDimensionRenderInfo),
  );
}

function makeSliceLayer(name = "slice") {
  const navigationState = makeNavigationState();
  const crossSectionVoxelRange = new TrackableCrossSectionVoxelRange();
  crossSectionVoxelRange.value = 5;
  const chunkManager = new RefCounted() as unknown as ChunkManager;
  const toolBinder = new GlobalToolBinder(() => {}, undefined!);
  const manager = {
    root: {
      display: {
        updateStarted: new NullarySignal(),
        scheduleRedraw() {},
      },
      toolBinder,
      navigationState,
      crossSectionVoxelRange,
      chunkManager,
    },
  } as unknown as LayerListSpecification;
  const managedLayer = new ManagedUserLayer(name, manager);
  const layer = new SliceUserLayer(managedLayer);
  return {
    crossSectionVoxelRange,
    layer,
    navigationState,
    dispose() {
      layer.dispose();
      managedLayer.dispose();
      toolBinder.dispose();
      navigationState.dispose();
      crossSectionVoxelRange.dispose();
      chunkManager.dispose();
    },
  };
}

describe("slice user layer", () => {
  it("registers the slice layer type", () => {
    expect(layerTypes.get("slice")).toBe(SliceUserLayer);
  });

  it("round trips one cross section under the singular slice property", () => {
    const harness = makeSliceLayer();
    const restoredHarness = makeSliceLayer("restored");
    try {
      harness.layer.restoreState({
        type: "slice",
        slice: {
          position: { link: "unlinked", value: [0, 0, 0] },
          orientation: { link: "unlinked", value: [0, 0, 0, 1] },
          voxelRange: { link: "unlinked", value: 0 },
        },
      });

      const rawJson = harness.layer.toJSON();
      expect(rawJson).not.toHaveProperty("source");
      const json = JSON.parse(JSON.stringify(rawJson));
      expect(json).toMatchObject({
        type: "slice",
        slice: {
          position: { link: "unlinked", value: [0, 0, 0] },
          orientation: { link: "unlinked" },
          voxelRange: { link: "unlinked", value: 0 },
        },
      });
      expect(json).not.toHaveProperty("source");

      restoredHarness.layer.restoreState(json);
      expect(
        Array.from(restoredHarness.layer.slice.position.value.value),
      ).toEqual([0, 0, 0]);
      expect(
        Array.from(restoredHarness.layer.slice.orientation.value.orientation),
      ).toEqual([0, 0, 0, 1]);
      expect(restoredHarness.layer.slice.voxelRange.value.value).toBe(0);
    } finally {
      restoredHarness.dispose();
      harness.dispose();
    }
  });

  it("uses the same state behavior as one layout cross section", () => {
    const harness = makeSliceLayer();
    const layout = new DataPanelLayoutSpecification(
      harness.navigationState.addRef(),
      harness.crossSectionVoxelRange.addRef(),
      "xy",
    );
    const slice = {
      width: 320,
      height: 240,
      volumeRenderingMode: "min",
      voxelRange: { link: "relative", value: 8 },
    };
    try {
      harness.layer.restoreState({ type: "slice", slice });
      layout.restoreState({
        type: "3d",
        crossSections: { comparison: slice },
      });
      const layoutSlice = layout.crossSections.get("comparison")!;

      expect(harness.layer.slice.toJSON()).toEqual(layoutSlice.toJSON());
      expect(harness.layer.slice.volumeRenderingMode.value).toBe(
        CrossSectionVolumeRenderingMode.MIN,
      );
      expect(harness.layer.slice.voxelRange.link.value).toBe(
        NavigationLinkType.RELATIVE,
      );

      harness.crossSectionVoxelRange.value = 7;
      expect(harness.layer.slice.voxelRange.value.value).toBe(10);
      expect(layoutSlice.voxelRange.value.value).toBe(10);
    } finally {
      layout.dispose();
      harness.dispose();
    }
  });

  it("resets omitted fields on a subsequent restore", () => {
    const harness = makeSliceLayer();
    try {
      harness.layer.restoreState({
        type: "slice",
        slice: {
          width: 320,
          height: 240,
          volumeRenderingMode: "min",
          voxelRange: { link: "unlinked", value: 1 },
        },
      });
      harness.layer.restoreState({
        type: "slice",
        slice: { height: 300 },
      });

      expect(harness.layer.slice.width.value).toBe(1000);
      expect(harness.layer.slice.height.value).toBe(300);
      expect(harness.layer.slice.volumeRenderingMode.value).toBe(
        CrossSectionVolumeRenderingMode.MAX,
      );
      expect(harness.layer.slice.voxelRange.link.value).toBe(
        NavigationLinkType.LINKED,
      );
      expect(harness.layer.slice.voxelRange.value.value).toBe(5);
      expect(JSON.parse(JSON.stringify(harness.layer.toJSON()))).toEqual({
        type: "slice",
        slice: { height: 300 },
      });
    } finally {
      harness.dispose();
    }
  });
});
