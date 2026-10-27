import { withChunkManager } from "#src/chunk_manager/backend.js";
import { ChunkState } from "#src/chunk_manager/base.js";
import type {
  RenderedViewBackend,
  RenderLayerBackendAttachment,
} from "#src/render_layer_backend.js";
import { RenderLayerBackend } from "#src/render_layer_backend.js";
import type { SharedWatchableValue } from "#src/shared_watchable_value.js";
import type { SliceParameters } from "#src/slice_projection/base.js";
import {
  forEachChunkInSlice,
  SLICE_PROJECTION_RENDER_LAYER_RPC_ID,
  SLICE_PROJECTION_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
} from "#src/slice_projection/base.js";
import {
  BASE_PRIORITY,
  deserializeTransformedSources,
  SCALE_PRIORITY_MULTIPLIER,
} from "#src/sliceview/backend.js";
import type { TransformedSource } from "#src/sliceview/base.js";
import type { VolumeChunkSource } from "#src/sliceview/volume/backend.js";
import {
  getBasePriority,
  getPriorityTier,
} from "#src/visibility_priority/backend.js";
import type { RPC } from "#src/worker_rpc.js";
import { registerRPC, registerSharedObject } from "#src/worker_rpc.js";

interface SliceProjectionAttachmentState {
  transformedSources: TransformedSource<any, VolumeChunkSource>[][];
  localPositions: Float32Array[];
}

@registerSharedObject(SLICE_PROJECTION_RENDER_LAYER_RPC_ID)
class SliceProjectionRenderLayerBackend extends withChunkManager(
  RenderLayerBackend,
) {
  sliceParameters: SharedWatchableValue<SliceParameters>;

  constructor(rpc: RPC, options: any) {
    super(rpc, options);
    this.sliceParameters = rpc.get(options.sliceParameters);
    const scheduleUpdateChunkPriorities = () =>
      this.chunkManager.scheduleUpdateChunkPriorities();
    this.registerDisposer(
      this.sliceParameters.changed.add(scheduleUpdateChunkPriorities),
    );
    this.registerDisposer(
      this.chunkManager.recomputeChunkPriorities.add(() =>
        this.recomputeChunkPriorities(),
      ),
    );
  }

  attach(
    attachment: RenderLayerBackendAttachment<
      RenderedViewBackend,
      SliceProjectionAttachmentState
    >,
  ) {
    const scheduleUpdateChunkPriorities = () =>
      this.chunkManager.scheduleUpdateChunkPriorities();
    const { view } = attachment;
    attachment.registerDisposer(scheduleUpdateChunkPriorities);
    attachment.registerDisposer(
      view.projectionParameters.changed.add(scheduleUpdateChunkPriorities),
    );
    attachment.registerDisposer(
      view.visibility.changed.add(scheduleUpdateChunkPriorities),
    );
    attachment.state = { transformedSources: [], localPositions: [] };
  }

  private recomputeChunkPriorities() {
    const { chunkManager } = this;
    for (const attachment of this.attachments.values()) {
      const { view } = attachment;
      const visibility = view.visibility.value;
      if (visibility === Number.NEGATIVE_INFINITY) continue;
      const state = attachment.state as SliceProjectionAttachmentState;
      const { transformedSources, localPositions } = state;
      if (transformedSources.length === 0) continue;
      const projectionParameters = view.projectionParameters.value;
      const { canonicalVoxelFactors } =
        projectionParameters.displayDimensionRenderInfo;
      const priorityTier = getPriorityTier(visibility);
      const basePriority = getBasePriority(visibility) + BASE_PRIORITY;
      const parameters = this.sliceParameters.value;
      chunkManager.registerLayer(this);
      for (let i = 0; i < transformedSources.length; ++i) {
        const scales = transformedSources[i];
        let sourceBasePriority = basePriority;
        forEachChunkInSlice(
          parameters,
          projectionParameters.globalPosition,
          localPositions[i],
          canonicalVoxelFactors,
          scales,
          (info) => {
            sourceBasePriority =
              basePriority +
              SCALE_PRIORITY_MULTIPLIER * (scales.length - 1 - info.scaleIndex);
          },
          (tsource) => {
            const chunk = tsource.source.getChunk(tsource.curPositionInChunks);
            ++this.numVisibleChunksNeeded;
            chunkManager.requestChunk(chunk, priorityTier, sourceBasePriority);
            if (chunk.state === ChunkState.GPU_MEMORY) {
              ++this.numVisibleChunksAvailable;
            }
          },
        );
      }
    }
  }
}
SliceProjectionRenderLayerBackend;

registerRPC(
  SLICE_PROJECTION_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
  function (x: any) {
    const view = this.get(x.view) as RenderedViewBackend;
    const layer = this.get(x.layer) as SliceProjectionRenderLayerBackend;
    const attachment = layer.attachments.get(
      view,
    )! as RenderLayerBackendAttachment<
      RenderedViewBackend,
      SliceProjectionAttachmentState
    >;
    attachment.state!.transformedSources = deserializeTransformedSources<
      VolumeChunkSource,
      any
    >(this, x.sources, layer);
    attachment.state!.localPositions = x.localPositions;
    layer.chunkManager.scheduleUpdateChunkPriorities();
  },
);
