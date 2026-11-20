import { withChunkManager } from "#src/chunk_manager/backend.js";
import { ChunkState } from "#src/chunk_manager/base.js";
import {
  SLAB_RENDER_LAYER_RPC_ID,
  SLAB_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
} from "#src/layer/oblique_slice/base.js";
import type { SlabLevelRule } from "#src/layer/oblique_slice/slab_render_layer.js";
import type {
  RenderedViewBackend,
  RenderLayerBackendAttachment,
} from "#src/render_layer_backend.js";
import { RenderLayerBackend } from "#src/render_layer_backend.js";
import type { SharedWatchableValue } from "#src/shared_watchable_value.js";
import {
  BASE_PRIORITY,
  deserializeTransformedSources,
  SCALE_PRIORITY_MULTIPLIER,
} from "#src/sliceview/backend.js";
import type {
  SliceViewBase,
  SliceViewRenderLayer,
  TransformedSource,
} from "#src/sliceview/base.js";
import {
  filterVisibleSources,
  forEachVisibleVolumetricChunk,
} from "#src/sliceview/base.js";
import type { VolumeChunkSource } from "#src/sliceview/volume/backend.js";
import { vec3 } from "#src/util/geom.js";
import {
  getBasePriority,
  getPriorityTier,
} from "#src/visibility_priority/backend.js";
import { forEachVisibleVolumeRenderingChunk } from "#src/volume_rendering/base.js";
import type { RPC } from "#src/worker_rpc.js";
import { registerRPC, registerSharedObject } from "#src/worker_rpc.js";

interface SlabAttachmentState {
  transformedSources: TransformedSource<any>[][];
}

const tempChunkPosition = vec3.create();
const tempCenter = vec3.create();
const tempChunkSize = vec3.create();

// The slab's chunks: the one level 3D volume rendering would pick for the
// slab's depth, within the frustum of the view's orthographic camera
// (depth = the slab). Chunk sources are shared with every other layer on the
// same volume.
@registerSharedObject(SLAB_RENDER_LAYER_RPC_ID)
export class SlabRenderLayerBackend extends withChunkManager(
  RenderLayerBackend,
) {
  localPosition: SharedWatchableValue<Float32Array>;
  depthSamplesTarget: SharedWatchableValue<number>;
  renderScaleTarget: SharedWatchableValue<number>;
  levelRule: SlabLevelRule;

  constructor(rpc: RPC, options: any) {
    super(rpc, options);
    this.levelRule = options.levelRule;
    this.localPosition = rpc.get(options.localPosition);
    this.depthSamplesTarget = rpc.get(options.depthSamplesTarget);
    this.renderScaleTarget = rpc.get(options.renderScaleTarget);
    const schedule = () => this.chunkManager.scheduleUpdateChunkPriorities();
    this.registerDisposer(this.localPosition.changed.add(schedule));
    this.registerDisposer(this.depthSamplesTarget.changed.add(schedule));
    this.registerDisposer(this.renderScaleTarget.changed.add(schedule));
    this.registerDisposer(
      this.chunkManager.recomputeChunkPriorities.add(() =>
        this.recomputeChunkPriorities(),
      ),
    );
  }

  attach(
    attachment: RenderLayerBackendAttachment<
      RenderedViewBackend,
      SlabAttachmentState
    >,
  ) {
    const schedule = () => this.chunkManager.scheduleUpdateChunkPriorities();
    const { view } = attachment;
    attachment.registerDisposer(schedule);
    attachment.registerDisposer(
      view.projectionParameters.changed.add(schedule),
    );
    attachment.registerDisposer(view.visibility.changed.add(schedule));
    attachment.state = { transformedSources: [] };
  }

  private recomputeChunkPriorities() {
    for (const attachment of this.attachments.values()) {
      const { view } = attachment;
      const visibility = view.visibility.value;
      if (visibility === Number.NEGATIVE_INFINITY) continue;
      const state = attachment.state as SlabAttachmentState | undefined;
      const sources = state?.transformedSources[0];
      if (sources === undefined || sources.length === 0) continue;
      const projectionParameters = view.projectionParameters.value;
      const priorityTier = getPriorityTier(visibility);
      const basePriority = getBasePriority(visibility) + BASE_PRIORITY;
      const { chunkManager } = this;
      chunkManager.registerLayer(this);
      const request = (
        tsource: TransformedSource<any, any>,
        priority: number,
      ) => {
        const source = tsource.source as unknown as VolumeChunkSource;
        const chunk = source.getChunk(tsource.curPositionInChunks);
        ++this.numVisibleChunksNeeded;
        chunkManager.requestChunk(chunk, priorityTier, priority);
        if (chunk.state === ChunkState.GPU_MEMORY) {
          ++this.numVisibleChunksAvailable;
        }
      };
      if (this.levelRule === "slice") {
        // Like the 2D panels: every level the zoom allows, coarser first.
        const levels = Array.from(
          filterVisibleSources(
            view as unknown as SliceViewBase,
            this as unknown as SliceViewRenderLayer,
            sources,
          ),
        );
        levels.forEach((tsource, i) => {
          const levelPriority =
            basePriority + SCALE_PRIORITY_MULTIPLIER * (levels.length - 1 - i);
          const { chunkLayout } = tsource;
          chunkLayout.globalToLocalSpatial(
            tempCenter,
            (projectionParameters as any).centerDataPosition,
          );
          vec3.copy(tempChunkSize, chunkLayout.size);
          forEachVisibleVolumetricChunk(
            projectionParameters as any,
            this.localPosition.value,
            tsource,
            (positionInChunks) => {
              vec3.multiply(tempChunkPosition, positionInChunks, tempChunkSize);
              request(
                tsource,
                levelPriority - vec3.distance(tempCenter, tempChunkPosition),
              );
            },
          );
        });
        continue;
      }
      forEachVisibleVolumeRenderingChunk(
        projectionParameters as any,
        this.localPosition.value,
        this.depthSamplesTarget.value,
        sources as TransformedSource<any, any>[],
        (tsource) => {
          const { chunkLayout } = tsource;
          chunkLayout.globalToLocalSpatial(
            tempCenter,
            (projectionParameters as any).centerDataPosition,
          );
          vec3.copy(tempChunkSize, chunkLayout.size);
        },
        (tsource, _, positionInChunks) => {
          vec3.multiply(tempChunkPosition, positionInChunks, tempChunkSize);
          request(
            tsource,
            basePriority - vec3.distance(tempCenter, tempChunkPosition),
          );
        },
      );
    }
  }
}

registerRPC(SLAB_RENDER_LAYER_UPDATE_SOURCES_RPC_ID, function (x) {
  const view = this.get(x.view) as RenderedViewBackend;
  const layer = this.get(x.layer) as SlabRenderLayerBackend;
  const attachment = layer.attachments.get(view) as
    | RenderLayerBackendAttachment<RenderedViewBackend, SlabAttachmentState>
    | undefined;
  if (attachment === undefined) return;
  attachment.state!.transformedSources = deserializeTransformedSources<
    VolumeChunkSource,
    any
  >(this, x.sources, layer);
  layer.chunkManager.scheduleUpdateChunkPriorities();
});
