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
  DisplayPose,
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

function makeSliceLayer() {
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
  const managedLayer = new ManagedUserLayer("slices", manager);
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

  it("round trips cross sections without a data source", () => {
    const harness = makeSliceLayer();
    try {
      harness.layer.restoreState({
        type: "slice",
        crossSections: {
          slab: {
            width: 640,
            height: 480,
            volumeRenderingMode: "min",
            voxelRange: { link: "relative", value: 8 },
          },
        },
      });

      const json = JSON.parse(JSON.stringify(harness.layer.toJSON()));
      expect(json).toEqual({
        type: "slice",
        crossSections: {
          slab: {
            width: 640,
            height: 480,
            volumeRenderingMode: "min",
            voxelRange: { link: "relative", value: 8 },
          },
        },
      });
      expect(json).not.toHaveProperty("source");
    } finally {
      harness.dispose();
    }
  });

  it("uses the same cross-section state behavior as a data panel layout", () => {
    const harness = makeSliceLayer();
    const layout = new DataPanelLayoutSpecification(
      harness.navigationState.addRef(),
      harness.crossSectionVoxelRange.addRef(),
      "xy",
    );
    try {
      const crossSections = {
        relative: {
          volumeRenderingMode: "min",
          voxelRange: { link: "relative", value: 8 },
        },
        unlinked: {
          width: 320,
          height: 240,
          voxelRange: { link: "unlinked", value: 0 },
        },
      };
      harness.layer.restoreState({ type: "slice", crossSections });
      layout.restoreState({ type: "3d", crossSections });

      expect(harness.layer.crossSections.toJSON()).toEqual(
        layout.crossSections.toJSON(),
      );
      harness.crossSectionVoxelRange.value = 7;
      expect(
        harness.layer.crossSections.get("relative")!.voxelRange.value.value,
      ).toBe(10);
      expect(layout.crossSections.get("relative")!.voxelRange.value.value).toBe(
        10,
      );
      expect(
        harness.layer.crossSections.get("unlinked")!.voxelRange.value.value,
      ).toBe(0);
      expect(layout.crossSections.get("unlinked")!.voxelRange.value.value).toBe(
        0,
      );
    } finally {
      layout.dispose();
      harness.dispose();
    }
  });

  it("clears entries omitted by a subsequent restore", () => {
    const harness = makeSliceLayer();
    try {
      harness.layer.restoreState({
        type: "slice",
        crossSections: {
          kept: { width: 100 },
          removed: { height: 200 },
        },
      });
      harness.layer.restoreState({
        type: "slice",
        crossSections: {
          kept: { height: 300 },
        },
      });

      expect(Array.from(harness.layer.crossSections.keys())).toEqual(["kept"]);
      expect(harness.layer.crossSections.toJSON()).toEqual({
        kept: { height: 300 },
      });
    } finally {
      harness.dispose();
    }
  });
});
