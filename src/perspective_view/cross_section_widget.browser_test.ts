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
  computeCrossSectionWireframeCorners,
  computeNormalDragScalar,
  decodeCrossSectionWidgetPickOffset,
  encodeCrossSectionWidgetPickOffset,
} from "#src/perspective_view/cross_section_widget.js";
import { mat4 } from "#src/util/geom.js";

describe("cross-section widget picking", () => {
  it("assigns a distinct contiguous pick offset to every mirrored part", () => {
    const picks = [
      { offset: 0, part: "cube", side: -1 },
      { offset: 1, part: "shaft", side: -1 },
      { offset: 2, part: "head", side: -1 },
      { offset: 3, part: "cube", side: 1 },
      { offset: 4, part: "shaft", side: 1 },
      { offset: 5, part: "head", side: 1 },
    ] as const;

    for (const pick of picks) {
      expect(encodeCrossSectionWidgetPickOffset(pick.part, pick.side)).toBe(
        pick.offset,
      );
      expect(decodeCrossSectionWidgetPickOffset(pick.offset)).toEqual({
        part: pick.part,
        side: pick.side,
      });
    }
  });

  it("rejects offsets outside the widget allocation", () => {
    expect(decodeCrossSectionWidgetPickOffset(-1)).toBeUndefined();
    expect(decodeCrossSectionWidgetPickOffset(1.5)).toBeUndefined();
    expect(decodeCrossSectionWidgetPickOffset(6)).toBeUndefined();
  });
});

describe("cross-section widget normal dragging", () => {
  it("projects motion onto the on-screen normal", () => {
    expect(computeNormalDragScalar(6, 8, 0.6, 0.8, 5)).toBe(2);
  });

  it("reverses negative-side cube motion so outward always increases range", () => {
    expect(computeNormalDragScalar(6, 8, 0.6, 0.8, 5, -1)).toBe(-2);
  });

  it("uses the minimum pixels-per-voxel sensitivity for an end-on normal", () => {
    expect(computeNormalDragScalar(0, -8, 0, -1, 0)).toBe(2);
  });
});

describe("cross-section widget wireframe", () => {
  it("uses the full cross-section size and extends by voxelRange on both sides", () => {
    const invViewMatrix = mat4.create();
    invViewMatrix[12] = 10;
    invViewMatrix[13] = 20;
    invViewMatrix[14] = 30;

    expect(
      Array.from(
        computeCrossSectionWireframeCorners(4, 2, invViewMatrix, [0, 0, 1], 3),
      ),
    ).toEqual([
      8, 19, 27, 12, 19, 27, 12, 21, 27, 8, 21, 27, 8, 19, 33, 12, 19, 33, 12,
      21, 33, 8, 21, 33,
    ]);
  });

  it("collapses both parallel faces onto the section when voxelRange is zero", () => {
    const corners = computeCrossSectionWireframeCorners(
      4,
      2,
      mat4.create(),
      [0, 0, 1],
      0,
    );
    expect(Array.from(corners.subarray(0, 12))).toEqual(
      Array.from(corners.subarray(12)),
    );
  });
});
