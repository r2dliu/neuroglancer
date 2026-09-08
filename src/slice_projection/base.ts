import { ProjectionParameters } from "#src/projection_parameters.js";
import type { TransformedSource } from "#src/sliceview/base.js";
import { forEachVisibleVolumetricChunk } from "#src/sliceview/base.js";
import type { ChunkLayout } from "#src/sliceview/chunk_layout.js";
import type { VolumeChunkSource } from "#src/sliceview/volume/base.js";
import type { quat, vec4 } from "#src/util/geom.js";
import { kAxes, mat4, transformVectorByMat4, vec3 } from "#src/util/geom.js";

export const SLICE_PROJECTION_RENDER_LAYER_RPC_ID =
  "slice_projection/SliceProjectionRenderLayer";
export const SLICE_PROJECTION_RENDER_LAYER_UPDATE_SOURCES_RPC_ID =
  "slice_projection/SliceProjectionRenderLayer/update";

export const SLICE_PROJECTION_MAX_TEXTURE_SIZE = 1024;

export enum SliceProjectionMode {
  MIN = 0,
  MAX = 1,
}

export interface SliceParameters {
  position: Float32Array;
  orientation: Float32Array;
  voxelRange: number;
  projectionMode: SliceProjectionMode;
  backgroundColor: vec4;
  width: number;
  height: number;
}

export interface SliceScaleInfo<Transformed> {
  tsource: Transformed;
  scaleIndex: number;
  sliceToWorld: mat4;
  voxelSpacing: number;
  halfThickness: number;
}

const tempCenter = vec3.create();
const tempScale = vec3.create();
const tempNormal = vec3.create();
const tempVoxelVector = vec3.create();
const tempSliceToWorld = mat4.create();
const tempWorldToSlice = mat4.create();
const tempProjectionParameters = new ProjectionParameters();

export function getSliceTargetSpacing(parameters: SliceParameters) {
  return (
    Math.max(parameters.width, parameters.height) /
    SLICE_PROJECTION_MAX_TEXTURE_SIZE
  );
}

export function getSliceTextureSize(parameters: SliceParameters) {
  const spacing = getSliceTargetSpacing(parameters);
  const clamp = (x: number) =>
    Math.max(1, Math.min(SLICE_PROJECTION_MAX_TEXTURE_SIZE, Math.round(x)));
  return {
    width: clamp(parameters.width / spacing),
    height: clamp(parameters.height / spacing),
  };
}

export function getSliceNormal(out: vec3, parameters: SliceParameters) {
  return vec3.transformQuat(
    out,
    kAxes[2],
    parameters.orientation as unknown as quat,
  );
}

export function computeSliceToWorld(
  out: mat4,
  parameters: SliceParameters,
  canonicalVoxelFactors: Float64Array,
  halfThickness: number,
) {
  const { position, width, height } = parameters;
  for (let i = 0; i < 3; ++i) {
    tempCenter[i] = position[i] * canonicalVoxelFactors[i];
  }
  mat4.fromRotationTranslation(
    out,
    parameters.orientation as unknown as quat,
    tempCenter,
  );
  vec3.set(tempScale, width / 2, height / 2, halfThickness);
  return mat4.scale(out, out, tempScale);
}

export function getVoxelSpacingAlongNormal(
  chunkLayout: ChunkLayout,
  normal: vec3,
) {
  transformVectorByMat4(tempVoxelVector, normal, chunkLayout.invTransform);
  const length = vec3.length(tempVoxelVector);
  return length === 0 ? 0 : 1 / length;
}

export function getSliceSampleCount(
  parameters: SliceParameters,
  info: SliceScaleInfo<unknown>,
) {
  if (parameters.voxelRange === 0 || info.voxelSpacing === 0) return 1;
  const count = Math.round((2 * info.halfThickness) / info.voxelSpacing) + 1;
  return Math.max(1, Math.min(512, count));
}

export function forEachChunkInSlice<
  Transformed extends TransformedSource<any, VolumeChunkSource>,
>(
  parameters: SliceParameters,
  globalPosition: Float32Array,
  localPosition: Float32Array,
  canonicalVoxelFactors: Float64Array,
  transformedSources: readonly Transformed[],
  beginScale: (info: SliceScaleInfo<Transformed>) => void,
  callback: (source: Transformed, positionInChunks: vec3) => void,
) {
  const { width, height } = parameters;
  if (transformedSources.length === 0 || !(width > 0) || !(height > 0)) return;
  getSliceNormal(tempNormal, parameters);
  const finestSpacing = getVoxelSpacingAlongNormal(
    transformedSources[0].chunkLayout,
    tempNormal,
  );
  if (finestSpacing === 0) return;
  const halfThickness = Math.max(parameters.voxelRange, 0.5) * finestSpacing;
  const targetVolume = getSliceTargetSpacing(parameters) ** 3;
  let scaleIndex = transformedSources.length - 1;
  for (let i = scaleIndex; i >= 0; --i) {
    const voxelVolume = Math.abs(
      transformedSources[i].chunkLayout.detTransform,
    );
    if (voxelVolume >= targetVolume) scaleIndex = i;
  }
  const tsource = transformedSources[scaleIndex];
  computeSliceToWorld(
    tempSliceToWorld,
    parameters,
    canonicalVoxelFactors,
    halfThickness,
  );
  mat4.invert(tempWorldToSlice, tempSliceToWorld);
  beginScale({
    tsource,
    scaleIndex,
    sliceToWorld: tempSliceToWorld,
    voxelSpacing: getVoxelSpacingAlongNormal(tsource.chunkLayout, tempNormal),
    halfThickness,
  });
  tempProjectionParameters.globalPosition = globalPosition;
  mat4.copy(tempProjectionParameters.viewProjectionMat, tempWorldToSlice);
  forEachVisibleVolumetricChunk(
    tempProjectionParameters,
    localPosition,
    tsource,
    (positionInChunks) => callback(tsource, positionInChunks),
  );
}
