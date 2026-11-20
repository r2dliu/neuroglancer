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

import type { BrushHashTable } from "#src/brush_stroke/index.js";
import { HashMapUint64 } from "#src/gpu_hash/hash_table.js";
import {
  GPUHashTable,
  HashMapShaderManager,
  HashSetShaderManager,
} from "#src/gpu_hash/shader.js";
import { getChunkPositionFromCombinedGlobalLocalPositions } from "#src/render_coordinate_transform.js";
import {
  DEFAULT_USER_MAIN_SEGMENT_COLOR,
  encodeSegmentPropertyShaderDefinition,
  type SegmentationColorUserShaderManagerParameters,
  type SegmentPropertyShaderDefinition,
} from "#src/segment_color.js";
import { AttributeShader } from "#src/segmentation_display_state/attributes_shader.js";
import { getVisibleSegments } from "#src/segmentation_display_state/base.js";
import type {
  SegmentationDisplayState,
  SegmentationGroupState,
} from "#src/segmentation_display_state/frontend.js";
import { registerRedrawWhenSegmentationDisplayStateChanged } from "#src/segmentation_display_state/frontend.js";
import type { SliceViewSourceOptions } from "#src/sliceview/base.js";
import type {
  FrontendTransformedSource,
  SliceView,
  SliceViewSingleResolutionSource,
} from "#src/sliceview/frontend.js";
import type { SliceViewRenderContext } from "#src/sliceview/renderlayer.js";
import type {
  MultiscaleVolumeChunkSource,
  VolumeChunkSource,
} from "#src/sliceview/volume/frontend.js";
import type { RenderLayerBaseOptions } from "#src/sliceview/volume/renderlayer.js";
import { SliceViewVolumeRenderLayer } from "#src/sliceview/volume/renderlayer.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import {
  AggregateWatchableValue,
  makeCachedDerivedWatchableValue,
  WatchableValue,
} from "#src/trackable_value.js";
import type { Uint64Map } from "#src/uint64_map.js";
import type { DisjointUint64Sets } from "#src/util/disjoint_sets.js";
import { kOneVec, vec3 } from "#src/util/geom.js";
import {
  EMPTY_VOXEL_VALUE,
  SEGMENTATION_ERASE_SENTINEL,
} from "#src/voxel_editing/base.js";
import type { ShaderBuilder, ShaderProgram } from "#src/webgl/shader.js";
import {
  getFallbackBuilderState,
  parseShaderUiControls,
  type ShaderControlsBuilderState,
} from "#src/webgl/shader_ui_controls.js";

// Scratch for setupSourceUniforms' per-source downsample factor.
const tempBrushSourceScale = vec3.create();

function glslUint64Equals(name: string, value: bigint) {
  const v = BigInt.asUintN(64, value);
  return `${name}.value[0] == ${v & 0xffffffffn}u && ${name}.value[1] == ${v >> 32n}u`;
}

export class EquivalencesHashMap {
  generation = Number.NaN;
  hashMap = new HashMapUint64();
  constructor(public disjointSets: DisjointUint64Sets) {}
  update() {
    const { disjointSets } = this;
    const { generation } = disjointSets;
    if (this.generation !== generation) {
      this.generation = generation;
      const { hashMap } = this;
      hashMap.clear();
      for (const [objectId, minObjectId] of disjointSets.mappings()) {
        hashMap.set(objectId, minObjectId);
      }
    }
  }
}

export interface SliceViewSegmentationDisplayState
  extends SegmentationDisplayState,
    RenderLayerBaseOptions {
  selectedAlpha: WatchableValueInterface<number>;
  notSelectedAlpha: WatchableValueInterface<number>;
  hideSegmentZero: WatchableValueInterface<boolean>;
  ignoreNullVisibleSet: WatchableValueInterface<boolean>;
  brushHashTable?: BrushHashTable;
}

interface ShaderParameters {
  hasAttributes: boolean;
  segmentColorParameters: SegmentationColorUserShaderManagerParameters;
  segmentColorProperties: readonly SegmentPropertyShaderDefinition[];
  hasEquivalences: boolean;
  baseSegmentColoring: boolean;
  baseSegmentHighlighting: boolean;
  hideSegmentZero: boolean;
  hasHighlightColor: boolean;
  isVoxelEditingPreview: boolean;
  shaderBuilderState: ShaderControlsBuilderState;
}

function getFallbackShaderParameters(
  displayState: SliceViewSegmentationDisplayState,
): ShaderParameters {
  const activeSegmentDefaultColor =
    displayState.tempSegmentDefaultColor2d.value ??
    displayState.segmentDefaultColor.value;
  const activeSegmentStatedColors = displayState.useTempSegmentStatedColors2d
    .value
    ? displayState.tempSegmentStatedColors2d.value
    : displayState.segmentStatedColors.value;
  return {
    hasAttributes: displayState.segmentAttributes?.enabled.value ?? false,
    segmentColorParameters: {
      hasSegmentDefaultColor: activeSegmentDefaultColor !== undefined,
      hasSegmentStatedColors: activeSegmentStatedColors.size !== 0,
    },
    segmentColorProperties: [],
    shaderBuilderState: getFallbackBuilderState(
      parseShaderUiControls(DEFAULT_USER_MAIN_SEGMENT_COLOR),
    ),
    hasEquivalences:
      displayState.segmentationGroupState.value.segmentEquivalences.size !== 0,
    hasHighlightColor: displayState.highlightColor.value !== undefined,
    hideSegmentZero: displayState.hideSegmentZero.value,
    baseSegmentColoring: displayState.baseSegmentColoring.value,
    baseSegmentHighlighting: displayState.baseSegmentHighlighting.value,
    isVoxelEditingPreview: false,
  };
}

export class SegmentationRenderLayer extends SliceViewVolumeRenderLayer<ShaderParameters> {
  public readonly segmentationGroupState: SegmentationGroupState;
  private gpuSegmentStatedColorHashTable:
    | GPUHashTable<HashMapUint64>
    | undefined;
  private hashTableManager = new HashSetShaderManager("visibleSegments");
  private gpuHashTable;
  private gpuTemporaryHashTable;
  private equivalencesShaderManager = new HashMapShaderManager("equivalences");
  private equivalencesHashMap;
  private temporaryEquivalencesHashMap;
  private gpuEquivalencesHashTable;
  private gpuTemporaryEquivalencesHashTable;
  private voxelEditingPreviewLayer: SegmentationRenderLayer | undefined;
  public isVoxelEditingPreview: WatchableValue<boolean>;
  // Brush-overlay machinery. Same prefix ("brushStroke") as the 2D
  // `BrushStrokeLayer`'s manager, so the GLSL helper functions are
  // named identically and `GPUHashTable.get(gl, table)` memoization
  // shares the underlying GPU resource across consumers.
  private brushHashTableManager = new HashMapShaderManager("brushStroke");
  private gpuBrushHashTable: GPUHashTable<HashMapUint64> | undefined;
  private attributeShader: AttributeShader | undefined;

  constructor(
    multiscaleSource: MultiscaleVolumeChunkSource,
    public displayState: SliceViewSegmentationDisplayState,
  ) {
    const isVoxelEditingPreview = new WatchableValue(false);
    super(multiscaleSource, {
      encodeShaderParameters: (p) => {
        const { shaderBuilderState, segmentColorProperties, ...rest } = p;
        return {
          shaderBuilderStateKey: shaderBuilderState.key,
          segmentColorProperties: segmentColorProperties.map(
            encodeSegmentPropertyShaderDefinition,
          ),
          ...rest,
        };
      },
      shaderError: displayState.volumeShaderError,
      fallbackShaderParameters: new WatchableValue(
        getFallbackShaderParameters(displayState),
      ),
      shaderParameters: new AggregateWatchableValue((refCounted) => ({
        hasAttributes:
          displayState.segmentAttributes?.enabled ??
          new WatchableValue<boolean>(false),
        segmentColorParameters: refCounted.registerDisposer(
          makeCachedDerivedWatchableValue(
            (
              segmentDefaultColor,
              tempSegmentDefaultColor2d,
              segmentStatedColors: Uint64Map,
              tempSegmentStatedColors2d: Uint64Map,
              useTempSegmentStatedColors2d: boolean,
            ) => {
              const activeSegmentDefaultColor =
                tempSegmentDefaultColor2d ?? segmentDefaultColor;
              const activeSegmentStatedColors = useTempSegmentStatedColors2d
                ? tempSegmentStatedColors2d
                : segmentStatedColors;
              return {
                hasSegmentDefaultColor: activeSegmentDefaultColor !== undefined,
                hasSegmentStatedColors: activeSegmentStatedColors.size !== 0,
              };
            },
            [
              displayState.segmentDefaultColor,
              displayState.tempSegmentDefaultColor2d,
              displayState.segmentStatedColors,
              displayState.tempSegmentStatedColors2d,
              displayState.useTempSegmentStatedColors2d,
            ],
          ),
        ),
        segmentColorProperties:
          displayState.segmentationColorUserShader.usedProperties,
        shaderBuilderState:
          displayState.segmentColorShaderControlState.builderState,
        hasEquivalences: refCounted.registerDisposer(
          makeCachedDerivedWatchableValue(
            (x) => x.size !== 0,
            [displayState.segmentationGroupState.value.segmentEquivalences],
          ),
        ),
        hasHighlightColor: refCounted.registerDisposer(
          makeCachedDerivedWatchableValue(
            (x) => x !== undefined,
            [displayState.highlightColor],
          ),
        ),
        hideSegmentZero: displayState.hideSegmentZero,
        baseSegmentColoring: displayState.baseSegmentColoring,
        baseSegmentHighlighting: displayState.baseSegmentHighlighting,
        isVoxelEditingPreview: isVoxelEditingPreview,
      })),
      transform: displayState.transform,
      renderScaleHistogram: displayState.renderScaleHistogram,
      renderScaleTarget: displayState.renderScaleTarget,
      localPosition: displayState.localPosition,
    });
    this.isVoxelEditingPreview = isVoxelEditingPreview;
    this.segmentationGroupState = displayState.segmentationGroupState.value;
    this.gpuHashTable = this.registerDisposer(
      GPUHashTable.get(
        this.gl,
        this.segmentationGroupState.visibleSegments.hashTable,
      ),
    );
    this.gpuTemporaryHashTable = GPUHashTable.get(
      this.gl,
      this.segmentationGroupState.temporaryVisibleSegments.hashTable,
    );
    this.equivalencesHashMap = new EquivalencesHashMap(
      this.segmentationGroupState.segmentEquivalences.disjointSets,
    );
    this.temporaryEquivalencesHashMap = new EquivalencesHashMap(
      this.segmentationGroupState.temporarySegmentEquivalences.disjointSets,
    );
    this.gpuEquivalencesHashTable = this.registerDisposer(
      GPUHashTable.get(this.gl, this.equivalencesHashMap.hashMap),
    );
    this.gpuTemporaryEquivalencesHashTable = this.registerDisposer(
      GPUHashTable.get(this.gl, this.temporaryEquivalencesHashMap.hashMap),
    );

    const { segmentAttributes } = displayState;
    if (segmentAttributes !== undefined) {
      this.attributeShader = this.registerDisposer(
        new AttributeShader(this.gl, segmentAttributes),
      );
      this.registerDisposer(
        segmentAttributes.changed.add(() => this.redrawNeeded.dispatch()),
      );
    }

    const { brushHashTable } = displayState;
    if (brushHashTable !== undefined) {
      this.gpuBrushHashTable = this.registerDisposer(
        GPUHashTable.get(this.gl, brushHashTable),
      );
      this.registerDisposer(
        brushHashTable.changed.add(() => {
          this.redrawNeeded.dispatch();
        }),
      );
    }

    this.registerDisposer(
      this.shaderParameters as AggregateWatchableValue<ShaderParameters>,
    );
    registerRedrawWhenSegmentationDisplayStateChanged(displayState, this);
    this.registerDisposer(
      displayState.selectedAlpha.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      displayState.notSelectedAlpha.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      displayState.ignoreNullVisibleSet.changed.add(this.redrawNeeded.dispatch),
    );
    if (displayState.focusedSegment) {
      this.registerDisposer(
        displayState.focusedSegment.changed.add(this.redrawNeeded.dispatch),
      );
    }
    if (displayState.focusDim) {
      this.registerDisposer(
        displayState.focusDim.changed.add(this.redrawNeeded.dispatch),
      );
    }
  }

  disposed() {
    this.gpuSegmentStatedColorHashTable?.dispose();
  }

  getSources(
    options: SliceViewSourceOptions,
  ): SliceViewSingleResolutionSource<VolumeChunkSource>[][] {
    return this.multiscaleSource.getSources({
      ...options,
      discreteValues: true,
    });
  }

  defineShader(builder: ShaderBuilder, parameters: ShaderParameters) {
    this.hashTableManager.defineShader(builder);
    this.displayState.segmentationColorUserShader.defineShader(
      builder,
      /*fragment=*/ true,
      parameters.shaderBuilderState,
      parameters.segmentColorParameters,
    );

    const hasBrushOverlay = this.gpuBrushHashTable !== undefined;
    if (hasBrushOverlay) {
      this.brushHashTableManager.defineShader(builder);
      // The vertex's un-nudged position (before the parent's
      // CHUNK_POSITION_EPSILON along the normal), so the hash key matches
      // what the CPU brush emit's Math.round saw at click time.
      builder.addVarying("highp vec3", "vBrushWorldPos");
      builder.addVertexMain("vBrushWorldPos = position;");
      // `vBrushWorldPos` is in THIS source's voxel grid; the brush hash is
      // keyed in full-res voxels, so it is scaled by the source's downsample
      // factor (set per source in `setupSourceUniforms`).
      builder.addUniform("highp vec3", "uBrushSourceScale");
    }
    const brushLookup = hasBrushOverlay
      ? `
  {
    // Integer full-res voxel index under this fragment, matching the CPU
    // brush hash (getBrushKey does coord >>> 0).
    uvec3 brushVoxel = uvec3(floor(vBrushWorldPos * uBrushSourceScale));
    uint x1 = brushVoxel.x;
    uint y1 = brushVoxel.y;
    uint z1 = brushVoxel.z;
    uint h1 = ((x1 * 73u) * 1271u) ^ ((y1 * 513u) * 1345u) ^ ((z1 * 421u) * 675u);
    uint h2 = ((x1 * 127u) * 337u) ^ ((y1 * 111u) * 887u) ^ ((z1 * 269u) * 325u);
    uint64_t brushKey;
    brushKey.value[0] = h1;
    brushKey.value[1] = h2;
    uint64_t brushValue;
    if (${this.brushHashTableManager.getFunctionName}(brushKey, brushValue)) {
      // value=0 marks "erased": report segment 0, whose transparent emit
      // (forced on with a brush overlay) reveals the layer underneath, which
      // a discard would not. value>0 paints optimistically.
      if (brushValue.value[0] == 0u && brushValue.value[1] == 0u) {
        uint64_t erased;
        erased.value[0] = 0u;
        erased.value[1] = 0u;
        return erased;
      }
      return brushValue;
    }
  }
`
      : "";
    builder.addFragmentCode(`
uint64_t getUint64DataValue() {
${brushLookup}
  uint64_t x = toUint64(getDataValue());
  return x;
}
`);
    if (parameters.hasEquivalences) {
      this.equivalencesShaderManager.defineShader(builder);
      builder.addFragmentCode(`
uint64_t getMappedObjectId(uint64_t value) {
  uint64_t mappedValue;
  if (${this.equivalencesShaderManager.getFunctionName}(value, mappedValue)) {
    return mappedValue;
  }
  return value;
}
`);
    } else {
      builder.addFragmentCode(`
uint64_t getMappedObjectId(uint64_t value) {
  return value;
}
`);
    }
    if (parameters.hasAttributes) {
      this.attributeShader?.defineShader(builder);
    }
    builder.addUniform("bool", "uShowAllSegments");
    builder.addUniform("highp float", "uSelectedAlpha");
    builder.addUniform("highp float", "uNotSelectedAlpha");
    builder.addUniform("bool", "uHasFocusedSegment");
    builder.addUniform("highp uvec2", "uFocusedSegment");
    builder.addUniform("highp float", "uFocusDim");
    let fragmentMain = `
  uint64_t baseValue = getUint64DataValue();
  uint64_t value = getMappedObjectId(baseValue);
  uint64_t valueForColor = ${
    parameters.baseSegmentColoring ? "baseValue" : "value"
  };
  uint64_t valueForHighlight = ${
    parameters.baseSegmentHighlighting ? "baseValue" : "value"
  };

  float alpha = uSelectedAlpha;
`;
    if (parameters.isVoxelEditingPreview)
      fragmentMain += `
  if (${glslUint64Equals("baseValue", SEGMENTATION_ERASE_SENTINEL)}) {
    emit(vec4(0, 0, 0, 0));
    return;
  }
  if (${glslUint64Equals("value", EMPTY_VOXEL_VALUE)}) {
    discard;
  }
      `;
    // With a brush overlay, erased voxels read as segment 0 (see brushLookup)
    // and rely on this transparent emit to reveal the layer underneath.
    if (parameters.hideSegmentZero || hasBrushOverlay) {
      fragmentMain += `
  if (value.value[0] == 0u && value.value[1] == 0u) {
    emit(vec4(vec4(0, 0, 0, 0)));
    return;
  }
`;
    }
    if (parameters.hasAttributes) {
      // The attribute tables replace the visible set and the mapped colors:
      // an id they don't draw is transparent.
      fragmentMain += `
  vec4 attributeColor;
  if (!getAttributeColor(baseValue, attributeColor)) {
    emit(vec4(0.0, 0.0, 0.0, 0.0));
    return;
  }
  bool has = true;
`;
    } else {
      fragmentMain += `
  bool has = uShowAllSegments || ${this.hashTableManager.hasFunctionName}(value);
`;
    }
    fragmentMain += `
  if (uHasSelectedSegment && uSelectedSegment == valueForHighlight.value) {
`;
    if (parameters.hasHighlightColor) {
      builder.addUniform("highp vec4", "uHighlightColor");
      fragmentMain += `
        emit(uHighlightColor);
        return;
    `;
    }
    fragmentMain += `
  } else if (!has) {
    alpha = uNotSelectedAlpha;
  }
`;
    if (parameters.hasAttributes) {
      fragmentMain += `
  float saturation = uSaturation;
  if (uHasSelectedSegment && uSelectedSegment == valueForColor.value) {
    saturation += saturation > 0.5 ? -0.5 : 0.5;
  }
  vec4 rgba = vec4(mix(vec3(1.0), attributeColor.rgb, saturation), 1.0);
`;
    } else {
      fragmentMain += `
  vec4 rgba = segmentColorUserShader(valueForColor, has ? 0.5 : 0.75);
`;
    }
    fragmentMain += `
  alpha *= rgba.a;
  if (uHasFocusedSegment && uFocusedSegment != valueForHighlight.value) {
    alpha *= uFocusDim;
  }
  emit(vec4(rgba.rgb, alpha));
`;
    builder.setFragmentMain(fragmentMain);
  }

  initializeShader(
    sliceView: SliceView,
    shader: ShaderProgram,
    parameters: ShaderParameters,
    fallback: boolean,
  ) {
    const { gl } = this;
    const { displayState, segmentationGroupState } = this;
    const {
      segmentDefaultColor: { value: segmentDefaultColor },
      segmentStatedColors: { value: segmentStatedColors },
      highlightColor: { value: highlightColor },
      tempSegmentDefaultColor2d: { value: tempSegmentDefaultColor2d },
      tempSegmentStatedColors2d: { value: tempSegmentStatedColors2d },
      useTempSegmentStatedColors2d: { value: useTempSegmentStatedColors2d },
    } = this.displayState;
    const activeSegmentDefaultColor =
      tempSegmentDefaultColor2d ?? segmentDefaultColor;
    const activeSegmentStatedColors = useTempSegmentStatedColors2d
      ? tempSegmentStatedColors2d
      : segmentStatedColors;
    const visibleSegments = getVisibleSegments(segmentationGroupState);
    const ignoreNullSegmentSet = this.displayState.ignoreNullVisibleSet.value;
    gl.uniform1f(
      shader.uniform("uSelectedAlpha"),
      displayState.selectedAlpha.value,
    );
    gl.uniform1f(
      shader.uniform("uNotSelectedAlpha"),
      displayState.notSelectedAlpha.value,
    );
    const showAllSegments =
      visibleSegments.hashTable.size === 0 && ignoreNullSegmentSet;
    gl.uniform1ui(shader.uniform("uShowAllSegments"), showAllSegments ? 1 : 0);
    this.hashTableManager.enable(
      gl,
      shader,
      segmentationGroupState.useTemporaryVisibleSegments.value
        ? this.gpuTemporaryHashTable
        : this.gpuHashTable,
    );
    if (parameters.hasEquivalences) {
      const useTemp =
        segmentationGroupState.useTemporarySegmentEquivalences.value;
      (useTemp
        ? this.temporaryEquivalencesHashMap
        : this.equivalencesHashMap
      ).update();
      this.equivalencesShaderManager.enable(
        gl,
        shader,
        useTemp
          ? this.gpuTemporaryEquivalencesHashTable
          : this.gpuEquivalencesHashTable,
      );
    }
    if (parameters.hasHighlightColor && highlightColor !== undefined) {
      gl.uniform4fv(shader.uniform("uHighlightColor"), highlightColor);
    }
    const focusedSegment = displayState.focusedSegment?.value ?? null;
    gl.uniform1ui(
      shader.uniform("uHasFocusedSegment"),
      focusedSegment != null ? 1 : 0,
    );
    if (focusedSegment != null) {
      gl.uniform2ui(
        shader.uniform("uFocusedSegment"),
        Number(focusedSegment & 0xffffffffn),
        Number(focusedSegment >> 32n),
      );
    }
    gl.uniform1f(
      shader.uniform("uFocusDim"),
      displayState.focusDim?.value ?? 0.15,
    );
    if (this.gpuBrushHashTable !== undefined) {
      this.brushHashTableManager.enable(gl, shader, this.gpuBrushHashTable);
    }
    if (parameters.hasAttributes) {
      this.attributeShader?.enable(gl, shader);
    }
    // Hover highlights a segment only while it is drawn.
    const { segmentSelectionState } = displayState;
    let hoverHighlight = displayState.hoverHighlight.value;
    if (hoverHighlight && segmentSelectionState.hasSelectedSegment) {
      const seg = displayState.baseSegmentHighlighting.value
        ? segmentSelectionState.baseSelectedSegment
        : segmentSelectionState.selectedSegment;
      hoverHighlight =
        parameters.hasAttributes ||
        visibleSegments.size === 0 ||
        visibleSegments.has(seg);
    }
    displayState.segmentationColorUserShader.enable(
      gl,
      shader,
      parameters.shaderBuilderState,
      { fallback, binding: sliceView.visibleLayers.get(this)! },
      parameters.segmentColorParameters,
      {
        segmentDefaultColor: activeSegmentDefaultColor,
        segmentStatedColors: activeSegmentStatedColors,
        hoverHighlight,
      },
    );
  }
  endSlice(
    sliceView: SliceView,
    shader: ShaderProgram,
    parameters: ShaderParameters,
  ) {
    const { gl, displayState } = this;
    displayState.segmentationColorUserShader.disable(
      gl,
      shader,
      parameters.segmentColorParameters,
    );
    this.hashTableManager.disable(gl, shader);
    if (parameters.hasAttributes) {
      this.attributeShader?.disable(gl, shader);
    }
    if (parameters.hasEquivalences) {
      this.equivalencesShaderManager.disable(gl, shader);
    }
    super.endSlice(sliceView, shader, parameters);
  }
  setVoxelEditingPreviewLayer(layer: SegmentationRenderLayer | undefined) {
    this.voxelEditingPreviewLayer = layer;
    if (layer) layer.isVoxelEditingPreview.value = true;
  }

  draw(renderContext: SliceViewRenderContext) {
    // The preview is drawn first, into the same depth buffer: the depth test
    // then rejects this layer's fragments wherever the preview drew, so the
    // preview replaces the data instead of blending with it.
    if (this.voxelEditingPreviewLayer) {
      this.voxelEditingPreviewLayer.draw(renderContext);
    }
    super.draw(renderContext);
  }

  override setupSourceUniforms(
    gl: WebGL2RenderingContext,
    shader: ShaderProgram,
    transformedSource: FrontendTransformedSource,
  ) {
    if (this.gpuBrushHashTable === undefined) return;
    // The brush hash is keyed in full-res (level-0) voxel indices, while
    // `vBrushWorldPos` is in THIS source's voxel grid: one source voxel spans
    // |localSpatialVectorToGlobal(1⃗)| global voxel-index units. Not derived
    // from `effectiveVoxelSize`, which folds in display scales. No offset: the
    // level's translation is already in the projection.
    const scale = transformedSource.chunkLayout.localSpatialVectorToGlobal(
      tempBrushSourceScale,
      kOneVec,
    );
    gl.uniform3f(
      shader.uniform("uBrushSourceScale"),
      Math.abs(scale[0]) || 1,
      Math.abs(scale[1]) || 1,
      Math.abs(scale[2]) || 1,
    );
  }

  // Never falls back to a coarser scale; null = full-res chunk not resident,
  // undefined = outside the array, where no chunk will ever arrive, "failed" =
  // its download failed.
  getFinestValueAt(globalPosition: Float32Array) {
    const { tempChunkPosition } = this;
    const sources = this.visibleSourcesList;
    if (sources.length === 0) return null;
    const finest = Math.abs(sources[0].chunkTransform.chunkToLayerTransformDet);
    let inside = false;
    let failed = false;
    for (const { source, chunkTransform } of sources) {
      if (Math.abs(chunkTransform.chunkToLayerTransformDet) !== finest) break;
      if (
        !getChunkPositionFromCombinedGlobalLocalPositions(
          tempChunkPosition,
          globalPosition,
          this.localPosition.value,
          chunkTransform.layerRank,
          chunkTransform.combinedGlobalLocalToChunkTransform,
        )
      ) {
        continue;
      }
      const { lowerVoxelBound, upperVoxelBound } = source.spec;
      let within = true;
      for (let i = 0; i < lowerVoxelBound.length; ++i) {
        const v = tempChunkPosition[i];
        if (v < lowerVoxelBound[i] || v >= upperVoxelBound[i]) within = false;
      }
      if (!within) continue;
      inside = true;
      const result = source.getValueAt(tempChunkPosition, chunkTransform);
      if (result != null) {
        return result;
      }
      failed ||= source.failedAt(tempChunkPosition);
    }
    if (failed) return "failed" as const;
    return inside ? null : undefined;
  }
}
