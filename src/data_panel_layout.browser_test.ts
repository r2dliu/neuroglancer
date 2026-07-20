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
import { TrackableCoordinateSpace } from "#src/coordinate_transform.js";
import { DataPanelLayoutSpecification } from "#src/data_panel_layout.js";
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

function makeLayoutSpecification(parentVoxelRangeValue: number) {
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
  const navigationState = new NavigationState(
    new DisplayPose(
      new Position(coordinateSpace),
      displayDimensionRenderInfo,
      new OrientationState(),
    ),
    new TrackableCrossSectionZoom(displayDimensionRenderInfo.addRef()),
    new TrackableDepthRange(-10, displayDimensionRenderInfo),
  );
  const parentVoxelRange = new TrackableCrossSectionVoxelRange();
  parentVoxelRange.value = parentVoxelRangeValue;
  const layout = new DataPanelLayoutSpecification(
    navigationState,
    parentVoxelRange,
    "xy",
  );
  return { layout, parentVoxelRange };
}

describe("cross-section layout slab-rendering state", () => {
  it("defaults to max mode and a linked voxel range", () => {
    const { layout, parentVoxelRange } = makeLayoutSpecification(5);
    try {
      layout.restoreState({
        type: "3d",
        crossSections: { slab: {} },
      });
      const slab = layout.crossSections.get("slab")!;
      expect(slab.volumeRenderingMode.value).toBe(
        CrossSectionVolumeRenderingMode.MAX,
      );
      expect(slab.volumeRenderingMode.toJSON()).toBeUndefined();
      expect(slab.voxelRange.link.value).toBe(NavigationLinkType.LINKED);
      expect(slab.voxelRange.value.value).toBe(5);
      expect(slab.voxelRange.toJSON()).toBeUndefined();

      parentVoxelRange.value = 7;
      expect(slab.voxelRange.value.value).toBe(7);
    } finally {
      layout.dispose();
    }
  });

  it("round trips relative and unlinked voxel ranges", () => {
    const { layout, parentVoxelRange } = makeLayoutSpecification(5);
    try {
      layout.restoreState({
        type: "3d",
        crossSections: {
          relative: {
            volumeRenderingMode: "min",
            voxelRange: { link: "relative", value: 8 },
          },
          unlinked: {
            voxelRange: { link: "unlinked", value: 0 },
          },
        },
      });

      parentVoxelRange.value = 7;
      expect(layout.crossSections.get("relative")!.voxelRange.value.value).toBe(
        10,
      );
      expect(layout.crossSections.get("unlinked")!.voxelRange.value.value).toBe(
        0,
      );

      const json = JSON.parse(JSON.stringify(layout.toJSON()));
      expect(json.crossSections.relative).toMatchObject({
        volumeRenderingMode: "min",
        voxelRange: { link: "relative", value: 10 },
      });
      expect(json.crossSections.unlinked.voxelRange).toEqual({
        link: "unlinked",
        value: 0,
      });

      layout.restoreState(json);
      expect(layout.crossSections.get("relative")!.voxelRange.value.value).toBe(
        10,
      );
      expect(
        layout.crossSections.get("relative")!.volumeRenderingMode.value,
      ).toBe(CrossSectionVolumeRenderingMode.MIN);
      expect(layout.crossSections.get("unlinked")!.voxelRange.value.value).toBe(
        0,
      );
    } finally {
      layout.dispose();
    }
  });
});
