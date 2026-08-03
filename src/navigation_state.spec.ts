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
import {
  CrossSectionVolumeRenderingMode,
  LinkedCrossSectionVolumeRenderingMode,
  LinkedCrossSectionVoxelRange,
  NavigationLinkType,
  TrackableCrossSectionVolumeRenderingMode,
  TrackableCrossSectionVoxelRange,
} from "#src/navigation_state.js";

describe("cross-section slab rendering state", () => {
  it("uses max and zero as the omitted defaults", () => {
    const mode = new TrackableCrossSectionVolumeRenderingMode();
    const voxelRange = new TrackableCrossSectionVoxelRange();
    try {
      expect(mode.value).toBe(CrossSectionVolumeRenderingMode.MAX);
      expect(mode.toJSON()).toBeUndefined();
      expect(voxelRange.value).toBe(0);
      expect(voxelRange.toJSON()).toBeUndefined();

      mode.restoreState("min");
      voxelRange.restoreState(2.5);
      expect(mode.value).toBe(CrossSectionVolumeRenderingMode.MIN);
      expect(mode.toJSON()).toBe("min");
      expect(voxelRange.value).toBe(2.5);
      expect(voxelRange.toJSON()).toBe(2.5);

      mode.reset();
      voxelRange.reset();
      expect(mode.toJSON()).toBeUndefined();
      expect(voxelRange.toJSON()).toBeUndefined();
    } finally {
      mode.dispose();
      voxelRange.dispose();
    }
  });

  it("rejects invalid serialized modes and voxel ranges", () => {
    const mode = new TrackableCrossSectionVolumeRenderingMode();
    const voxelRange = new TrackableCrossSectionVoxelRange();
    try {
      expect(() => mode.restoreState("average")).toThrow();
      expect(() => voxelRange.restoreState(-1)).toThrow();
      expect(() => voxelRange.restoreState(Number.POSITIVE_INFINITY)).toThrow();
      expect(() => voxelRange.restoreState(Number.NaN)).toThrow();
    } finally {
      mode.dispose();
      voxelRange.dispose();
    }
  });

  it("links voxel ranges by default and supports additive relative values", () => {
    const parent = new TrackableCrossSectionVoxelRange();
    parent.value = 5;
    const linked = new LinkedCrossSectionVoxelRange(parent.addRef());
    try {
      expect(linked.link.value).toBe(NavigationLinkType.LINKED);
      expect(linked.value.value).toBe(5);
      expect(linked.toJSON()).toBeUndefined();

      linked.restoreState({ link: "relative", value: 8 });
      expect(linked.link.value).toBe(NavigationLinkType.RELATIVE);
      expect(linked.value.value).toBe(8);

      parent.value = 7;
      expect(linked.value.value).toBe(10);

      linked.value.value = 12;
      expect(parent.value).toBe(9);
      expect(linked.toJSON()).toEqual({ link: "relative", value: 12 });
    } finally {
      linked.value.dispose();
      parent.dispose();
    }
  });

  it("preserves explicit default values in unlinked state", () => {
    const parentRange = new TrackableCrossSectionVoxelRange();
    parentRange.value = 4;
    const range = new LinkedCrossSectionVoxelRange(parentRange.addRef());

    const parentMode = new TrackableCrossSectionVolumeRenderingMode();
    parentMode.value = CrossSectionVolumeRenderingMode.MIN;
    const mode = new LinkedCrossSectionVolumeRenderingMode(parentMode.addRef());
    try {
      range.restoreState({ link: "unlinked", value: 0 });
      mode.restoreState({ link: "unlinked", value: "max" });

      expect(range.toJSON()).toEqual({ link: "unlinked", value: 0 });
      expect(mode.toJSON()).toEqual({ link: "unlinked", value: "max" });

      parentRange.value = 9;
      parentMode.value = CrossSectionVolumeRenderingMode.MAX;
      expect(range.value.value).toBe(0);
      expect(mode.value.value).toBe(CrossSectionVolumeRenderingMode.MAX);
    } finally {
      range.value.dispose();
      parentRange.dispose();
      mode.value.dispose();
      parentMode.dispose();
    }
  });
});
