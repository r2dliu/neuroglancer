/**
 * @license
 * Copyright 2016 Google Inc.
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

import { describe, it, expect } from "vitest";
import {
  estimateSliceAreaPerChunk,
  forEachCrossSectionVolumeRenderingPlane,
  forEachPlaneIntersectingVolumetricChunk,
  getCrossSectionVolumeRenderingDepth,
  getCrossSectionVolumeRenderingVoxelRange,
  getNearIsotropicBlockSize,
  SliceViewProjectionParameters,
} from "#src/sliceview/base.js";
import { ChunkLayout } from "#src/sliceview/chunk_layout.js";
import { mat4, vec3 } from "#src/util/geom.js";

describe("sliceview/base", () => {
  it("getNearIsotropicBlockSize", () => {
    expect(
      getNearIsotropicBlockSize({
        rank: 3,
        displayRank: 3,
        chunkToViewTransform: Float32Array.of(
          1,
          0,
          0, //
          0,
          1,
          0, //
          0,
          0,
          1,
        ),
        maxVoxelsPerChunkLog2: 18,
      }),
    ).toEqual(Uint32Array.of(64, 64, 64));

    expect(
      getNearIsotropicBlockSize({
        rank: 3,
        displayRank: 3,
        chunkToViewTransform: Float32Array.of(
          2,
          0,
          0, //
          0,
          1,
          0, //
          0,
          0,
          1,
        ),
        maxVoxelsPerChunkLog2: 17,
      }),
    ).toEqual(Uint32Array.of(32, 64, 64));

    expect(
      getNearIsotropicBlockSize({
        rank: 3,
        displayRank: 3,
        chunkToViewTransform: Float32Array.of(
          3,
          0,
          0, //
          0,
          3,
          0, //
          0,
          0,
          30,
        ),
        maxVoxelsPerChunkLog2: 9,
      }),
    ).toEqual(Uint32Array.of(16, 16, 2));

    expect(
      getNearIsotropicBlockSize({
        rank: 4,
        displayRank: 3,
        chunkToViewTransform: Float32Array.of(
          3,
          0,
          0,
          0, //
          0,
          3,
          0,
          0, //
          0,
          0,
          30,
          0,
        ),
        maxVoxelsPerChunkLog2: 9,
        minBlockSize: Uint32Array.of(1, 1, 1, 8),
      }),
    ).toEqual(Uint32Array.of(8, 8, 1, 8));

    expect(
      getNearIsotropicBlockSize({
        rank: 3,
        displayRank: 3,
        chunkToViewTransform: Float32Array.of(
          3,
          0,
          0, //
          0,
          3,
          0, //
          0,
          0,
          30,
        ),
        upperVoxelBound: vec3.fromValues(1, 128, 128),
        maxVoxelsPerChunkLog2: 8,
      }),
    ).toEqual(Uint32Array.of(1, 64, 4));
  });
});

describe("estimateSliceAreaPerChunk", () => {
  it("works for identity chunk transform", () => {
    const chunkLayout = new ChunkLayout(
      vec3.fromValues(3, 4, 5),
      mat4.create(),
      3,
    );
    {
      const viewMatrix = Float32Array.from([
        1,
        0,
        0,
        0, //
        0,
        1,
        0,
        0, //
        0,
        0,
        1,
        0, //
        0,
        0,
        0,
        1, //
      ]) as mat4;
      expect(estimateSliceAreaPerChunk(chunkLayout, viewMatrix)).toEqual(3 * 4);
    }

    {
      const viewMatrix = Float32Array.from([
        0,
        1,
        0,
        0, //
        1,
        0,
        0,
        0, //
        0,
        0,
        1,
        0, //
        0,
        0,
        0,
        1, //
      ]) as mat4;
      expect(estimateSliceAreaPerChunk(chunkLayout, viewMatrix)).toEqual(3 * 4);
    }

    {
      const viewMatrix = Float32Array.from([
        1,
        0,
        0,
        0, //
        0,
        0,
        1,
        0, //
        0,
        1,
        0,
        0, //
        0,
        0,
        0,
        1, //
      ]) as mat4;
      expect(estimateSliceAreaPerChunk(chunkLayout, viewMatrix)).toEqual(3 * 5);
    }
  });
});

function makeChunkSelectionTestSource() {
  return {
    renderLayer: {},
    source: {
      spec: {
        rank: 3,
        chunkDataSize: Uint32Array.of(1, 1, 1),
        lowerChunkBound: Float32Array.of(-4, -4, -4),
        upperChunkBound: Float32Array.of(4, 4, 4),
      },
    },
    nonDisplayLowerClipBound: Float32Array.of(-Infinity, -Infinity, -Infinity),
    nonDisplayUpperClipBound: Float32Array.of(Infinity, Infinity, Infinity),
    lowerChunkDisplayBound: vec3.fromValues(-4, -4, -4),
    upperChunkDisplayBound: vec3.fromValues(4, 4, 4),
    chunkDisplayDimensionIndices: [0, 1, 2],
    layerRank: 3,
    fixedLayerToChunkTransform: new Float32Array(12),
    curPositionInChunks: new Float32Array(3),
    fixedPositionWithinChunk: new Uint32Array(3),
  } as any;
}

function makeChunkSelectionProjection() {
  const projection = new SliceViewProjectionParameters();
  projection.globalPosition = vec3.create();
  projection.centerDataPosition.set([0, 0, 0]);
  projection.viewportNormalInGlobalCoordinates.set([0, 0, 1]);
  projection.viewProjectionMat = mat4.create();
  return projection;
}

describe("cross-section volume rendering chunk selection", () => {
  it("normalizes voxel ranges without changing the zero default", () => {
    expect(getCrossSectionVolumeRenderingVoxelRange(0)).toBe(0);
    expect(getCrossSectionVolumeRenderingVoxelRange(2.9)).toBe(2);
    expect(getCrossSectionVolumeRenderingVoxelRange(-1)).toBe(0);
    expect(
      getCrossSectionVolumeRenderingVoxelRange(
        Number.MAX_SAFE_INTEGER + Number.MAX_SAFE_INTEGER,
      ),
    ).toBe(Number.MAX_SAFE_INTEGER);
    expect(getCrossSectionVolumeRenderingVoxelRange(Infinity)).toBe(0);
    expect(getCrossSectionVolumeRenderingVoxelRange(NaN)).toBe(0);
  });

  it("iterates each normalized plane exactly once", () => {
    const offsets: number[] = [];
    forEachCrossSectionVolumeRenderingPlane(2.9, (offset) =>
      offsets.push(offset),
    );
    expect(offsets).toEqual([-2, -1, 0, 1, 2]);

    offsets.length = 0;
    forEachCrossSectionVolumeRenderingPlane(0, (offset) =>
      offsets.push(offset),
    );
    expect(offsets).toEqual([0]);
  });

  it("orders source depths within non-overlapping plane bands", () => {
    const planeCount = 5;
    const sourceCount = 3;
    const depths = Array.from({ length: planeCount }, (_, planeIndex) =>
      Array.from({ length: sourceCount }, (_, sourceIndex) =>
        getCrossSectionVolumeRenderingDepth(
          planeIndex,
          planeCount,
          sourceIndex,
          sourceCount,
        ),
      ),
    );
    for (const planeDepths of depths) {
      expect(planeDepths[0]).toBeLessThan(planeDepths[1]);
      expect(planeDepths[1]).toBeLessThan(planeDepths[2]);
    }
    for (let planeIndex = 1; planeIndex < planeCount; ++planeIndex) {
      expect(Math.max(...depths[planeIndex])).toBeLessThan(
        Math.min(...depths[planeIndex - 1]),
      );
    }
  });

  it("selects chunks intersecting each parallel offset plane", () => {
    const projection = makeChunkSelectionProjection();
    const source = makeChunkSelectionTestSource();
    const chunkLayout = new ChunkLayout(
      vec3.fromValues(1, 1, 1),
      mat4.create(),
      3,
    );

    for (const planeOffset of [-1, 0, 1]) {
      const selectedZ = new Set<number>();
      forEachPlaneIntersectingVolumetricChunk(
        projection,
        new Float32Array(0),
        source,
        chunkLayout,
        () => selectedZ.add(source.curPositionInChunks[2]),
        planeOffset,
      );
      expect([...selectedZ]).toEqual([planeOffset]);
    }
  });

  it("converts offsets through an anisotropic chunk transform", () => {
    const projection = makeChunkSelectionProjection();
    const source = makeChunkSelectionTestSource();
    const transform = mat4.fromScaling(mat4.create(), vec3.fromValues(1, 1, 2));
    const chunkLayout = new ChunkLayout(vec3.fromValues(1, 1, 1), transform, 3);
    const expectedChunkZ = [-1, -1, 0, 0, 1];

    for (let planeOffset = -2; planeOffset <= 2; ++planeOffset) {
      const selectedZ = new Set<number>();
      forEachPlaneIntersectingVolumetricChunk(
        projection,
        new Float32Array(0),
        source,
        chunkLayout,
        () => selectedZ.add(source.curPositionInChunks[2]),
        planeOffset,
      );
      expect([...selectedZ]).toEqual([expectedChunkZ[planeOffset + 2]]);
    }
  });
});
