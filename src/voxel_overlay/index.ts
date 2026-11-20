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

import type { ChunkManager } from "#src/chunk_manager/frontend.js";
import { getIdentitySourceOptions } from "#src/layer/voxel_editing/eligibility.js";
import { DataType } from "#src/sliceview/base.js";
import type { UncompressedVolumeChunk } from "#src/sliceview/uncompressed_chunk_format.js";
import type {
  InMemoryVolumeChunkSource,
  MultiscaleVolumeChunkSource,
} from "#src/sliceview/volume/frontend.js";
import { RefCounted } from "#src/util/disposable.js";
import { NullarySignal } from "#src/util/signal.js";
import { SEGMENTATION_ERASE_SENTINEL } from "#src/voxel_editing/base.js";
import { PreviewMultiscaleSource } from "#src/voxel_editing/preview_multiscale_chunk_source.js";

// Optimistic full-resolution label edits not yet in the stored volume, held in
// the voxel editing preview chunks. A stored 0 is an untouched voxel; an erased
// voxel holds the erase sentinel and reads back as 0.
export class VoxelOverlay extends RefCounted {
  changed = new NullarySignal();
  previewSource: PreviewMultiscaleSource | undefined;
  private chunkSource: InMemoryVolumeChunkSource | undefined;
  private isUint64 = false;
  private eraseValue: number | bigint = 0;
  private chunkCounts = new Map<string, number>();
  private labelCounts = new Map<number, number>();
  private dirty = new Set<UncompressedVolumeChunk>();

  // Rebinding to another volume drops the edits: they belong to its voxels.
  bind(chunkManager: ChunkManager, volume: MultiscaleVolumeChunkSource) {
    if (this.previewSource?.primarySource === volume) return this.previewSource;
    this.clear();
    const previewSource = new PreviewMultiscaleSource(chunkManager, volume);
    this.previewSource = previewSource;
    this.chunkSource = previewSource.getSources(
      getIdentitySourceOptions(volume.rank),
    )[0][0].chunkSource as InMemoryVolumeChunkSource;
    this.isUint64 = volume.dataType === DataType.UINT64;
    const sentinel = BigInt.asUintN(
      this.isUint64 ? 64 : 32,
      SEGMENTATION_ERASE_SENTINEL,
    );
    this.eraseValue = this.isUint64 ? sentinel : Number(sentinel);
    return previewSource;
  }

  get(x: number, y: number, z: number): number | undefined {
    const { chunkSource } = this;
    if (chunkSource === undefined) return undefined;
    const [key, index] = this.locate(chunkSource, x, y, z);
    const data = (chunkSource.chunks.get(key) as UncompressedVolumeChunk)?.data;
    if (data == null) return undefined;
    const stored = data[index];
    if (stored === 0 || stored === 0n) return undefined;
    if (stored === this.eraseValue) return 0;
    return Number(stored);
  }

  set(x: number, y: number, z: number, value: number) {
    this.write(x, y, z, value === 0 ? this.eraseValue : value);
  }

  delete(x: number, y: number, z: number) {
    this.write(x, y, z, 0);
  }

  labels(): Iterable<number> {
    return this.labelCounts.keys();
  }

  clear() {
    const { chunkSource } = this;
    if (chunkSource !== undefined) {
      chunkSource.invalidateChunks([...chunkSource.chunks.keys()]);
    }
    this.chunkCounts.clear();
    this.labelCounts.clear();
    this.dirty.clear();
    this.changed.dispatch();
  }

  // Uploads the chunks edited since the last draw.
  flush() {
    if (this.dirty.size === 0) return;
    const { gl } = this.chunkSource!.chunkManager.chunkQueueManager;
    for (const chunk of this.dirty) chunk.updateFromCpuData(gl);
    this.dirty.clear();
  }

  disposed() {
    this.clear();
    super.disposed();
  }

  // Dimensions past xyz (e.g. a channel) sit at position 0 of their chunk.
  private locate(
    chunkSource: InMemoryVolumeChunkSource,
    x: number,
    y: number,
    z: number,
  ): [string, number, Float32Array] {
    const { chunkDataSize, rank } = chunkSource.spec;
    const [sx, sy, sz] = chunkDataSize;
    const grid = new Float32Array(rank);
    grid[0] = Math.floor(x / sx);
    grid[1] = Math.floor(y / sy);
    grid[2] = Math.floor(z / sz);
    const index =
      (z - grid[2] * sz) * sx * sy +
      (y - grid[1] * sy) * sx +
      (x - grid[0] * sx);
    return [grid.join(), index, grid];
  }

  private write(x: number, y: number, z: number, value: number | bigint) {
    const { chunkSource } = this;
    if (chunkSource === undefined) return;
    const [key, index, chunkGridPosition] = this.locate(chunkSource, x, y, z);
    let chunk = chunkSource.chunks.get(key) as
      | UncompressedVolumeChunk
      | undefined;
    if (chunk === undefined) {
      if (value === 0) return;
      chunk = chunkSource.getChunk({
        chunkGridPosition,
      }) as UncompressedVolumeChunk;
      const size = chunkSource.spec.chunkDataSize.reduce((a, b) => a * b, 1);
      chunk.data = this.isUint64
        ? new BigUint64Array(size)
        : new Uint32Array(size);
      chunkSource.addChunk(key, chunk);
    }
    const data = chunk.data!;
    const stored = data[index];
    const next = this.isUint64 ? BigInt(value) : Number(value);
    if (stored === next) return;
    const wasSet = stored !== 0 && stored !== 0n;
    const isSet = value !== 0;
    if (wasSet && stored !== this.eraseValue) this.countLabel(stored, -1);
    if (isSet && next !== this.eraseValue) this.countLabel(next, 1);
    data[index] = next as never;
    const count = (this.chunkCounts.get(key) ?? 0) + +isSet - +wasSet;
    if (count === 0) {
      this.chunkCounts.delete(key);
      this.dirty.delete(chunk);
      chunkSource.invalidateChunks([key]);
    } else {
      this.chunkCounts.set(key, count);
      if (chunk.texture == null) {
        chunk.updateFromCpuData(chunkSource.chunkManager.chunkQueueManager.gl);
        chunkSource.chunkManager.chunkQueueManager.visibleChunksChanged.dispatch();
      } else {
        this.dirty.add(chunk);
      }
    }
    this.changed.dispatch();
  }

  private countLabel(value: number | bigint, delta: number) {
    const label = Number(value);
    const count = (this.labelCounts.get(label) ?? 0) + delta;
    if (count === 0) this.labelCounts.delete(label);
    else this.labelCounts.set(label, count);
  }
}
