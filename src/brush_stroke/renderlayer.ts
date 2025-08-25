/**
 * @license
 * Copyright 2024 Google Inc.
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
import type { ChunkManager } from "#src/chunk_manager/frontend.js";
import { GPUHashTable, HashMapShaderManager } from "#src/gpu_hash/shader.js";
import type {
    LayerView,
    VisibleLayerInfo,
} from "#src/layer/index.js";
import type { DisplayDimensionRenderInfo } from "#src/navigation_state.js";
import type {
    PerspectiveViewRenderContext,
} from "#src/perspective_view/render_layer.js";
import { PerspectiveViewRenderLayer } from "#src/perspective_view/render_layer.js";
import type {
    ChunkDisplayTransformParameters,
    ChunkTransformParameters,
} from "#src/render_coordinate_transform.js";
import type { RenderScaleHistogram } from "#src/render_scale_statistics.js";
import type {
    VisibilityTrackedRenderLayer,
} from "#src/renderlayer.js";
import { SegmentColorShaderManager } from "#src/segment_color.js";
import type { SegmentationDisplayState } from "#src/segmentation_display_state/frontend.js";
import { defineBoundingBoxCrossSectionShader } from "#src/sliceview/bounding_box_shader_helper.js";
import type {
    SliceViewPanelRenderContext,
} from "#src/sliceview/renderlayer.js";
import { SliceViewPanelRenderLayer } from "#src/sliceview/renderlayer.js";
import { CHUNK_POSITION_EPSILON } from "#src/sliceview/volume/renderlayer.js";
import { constantWatchableValue } from "#src/trackable_value.js";
import type { Owned } from "#src/util/disposable.js";
import { RefCounted } from "#src/util/disposable.js";
import type { ValueOrError } from "#src/util/error.js";
import type { AnyConstructor, MixinConstructor } from "#src/util/mixin.js";
import { NullarySignal } from "#src/util/signal.js";
import type { ParameterizedContextDependentShaderGetter } from "#src/webgl/dynamic_shader.js";
import { parameterizedEmitterDependentShaderGetter } from "#src/webgl/dynamic_shader.js";
import type { ShaderBuilder, ShaderProgram } from "#src/webgl/shader.js";
import { defineVertexId } from "#src/webgl/vertex_id.js";



interface BrushStrokeChunkRenderParameters {
    chunkTransform: ChunkTransformParameters;
    chunkDisplayTransform: ChunkDisplayTransformParameters;
    modelClipBounds: Float32Array;
}

interface AttachmentState {
    chunkTransform: ValueOrError<ChunkTransformParameters>;
    displayDimensionRenderInfo: DisplayDimensionRenderInfo;
    chunkRenderParameters: BrushStrokeChunkRenderParameters | undefined;
}

export class BrushStrokeLayer extends RefCounted {
    public gpuBrushHashTable: GPUHashTable<any>;
    public brushHashTableManager = new HashMapShaderManager("brushStroke");
    public segmentColorShaderManager = new SegmentColorShaderManager("segmentColorHash");
    redrawNeeded = new NullarySignal();

    constructor(
        public chunkManager: ChunkManager,
        public brushHashTable: BrushHashTable,
        public displayState: SegmentationDisplayState,
    ) {
        super();
        // Create GPU hash table for brush strokes
        this.gpuBrushHashTable = this.registerDisposer(
            GPUHashTable.get(this.chunkManager.gl, brushHashTable),
        );
    }

    get gl() {
        return this.chunkManager.gl;
    }
}

function BrushStrokeRenderLayer<
    TBase extends AnyConstructor<VisibilityTrackedRenderLayer>,
>(Base: TBase) {
    class C extends (Base as AnyConstructor<VisibilityTrackedRenderLayer>) {
        private shaderGetter: ParameterizedContextDependentShaderGetter<any, undefined>;

        constructor(
            public base: Owned<BrushStrokeLayer>,
            public renderScaleHistogram: RenderScaleHistogram,
        ) {
            super();
            this.registerDisposer(base.redrawNeeded.add(this.redrawNeeded.dispatch));
            this.shaderGetter = parameterizedEmitterDependentShaderGetter(this, this.gl, {
                memoizeKey: "brushStroke",
                parameters: constantWatchableValue(undefined),
                defineShader: (builder: ShaderBuilder) => {
                    this.defineShader(builder);
                },
            });
        }

        get gl() {
            return this.base.chunkManager.gl;
        }

        defineShader(builder: ShaderBuilder) {
            // Add opacity uniforms to match parent segmentation layer
            builder.addUniform("highp float", "uSelectedAlpha");
            builder.addUniform("highp float", "uNotSelectedAlpha");

            // Add saturation uniform
            builder.addUniform("highp float", "uSaturation");

            // Check if this is a perspective view (3D) or slice view (2D)
            const isPerspectiveView = this instanceof PerspectiveViewRenderLayer;

            if (isPerspectiveView) {
                // 3D view: TODO - implement proper volume rendering
                this.base.brushHashTableManager.defineShader(builder);
                this.base.segmentColorShaderManager.defineShader(builder);
                defineVertexId(builder);
                defineBoundingBoxCrossSectionShader(builder);

                // Add volume rendering uniforms (same as volume layers)
                builder.addUniform("highp vec3", "uTranslation");
                builder.addUniform("highp mat4", "uProjectionMatrix");
                builder.addUniform("highp vec3", "uChunkDataSize");
                builder.addUniform("highp vec3", "uLowerClipBound");
                builder.addUniform("highp vec3", "uUpperClipBound");
                builder.addVarying("highp vec3", "vChunkPosition");

                const vertexShader = `
                    vec3 position = getBoundingBoxPlaneIntersectionVertexPosition(
                        uChunkDataSize, uTranslation, uLowerClipBound, uUpperClipBound, gl_VertexID
                    );
                    gl_Position = uProjectionMatrix * vec4(position, 1.0);
                    gl_Position.z = 0.0;
                    vChunkPosition = (position - uTranslation) + ${CHUNK_POSITION_EPSILON} * abs(uPlaneNormal);
                `;
                builder.setVertexMain(vertexShader);

                const fragmentShader = `
                    // Sample brush hash table at chunk position (z,y,x order)
                    vec3 chunkPos = vChunkPosition + uTranslation;
                    ivec3 ipos = ivec3(floor(chunkPos));
                    
                    // Use z,y,x order for hash key (not x,y,z)
                    uint z1 = uint(ipos.z);
                    uint y1 = uint(ipos.y);
                    uint x1 = uint(ipos.x);
                    
                    uint h1 = ((x1 * 73u) * 1271u) ^ ((y1 * 513u) * 1345u) ^ ((z1 * 421u) * 675u);
                    uint h2 = ((x1 * 127u) * 337u) ^ ((y1 * 111u) * 887u) ^ ((z1 * 269u) * 325u);
                    
                    uint64_t key;
                    key.value[0] = h1;
                    key.value[1] = h2;
                    
                    uint64_t brushValue;
                    if (brushStroke_get(key, brushValue)) {
                        // Found brush stroke - use segment color based on the stored value
                        vec3 segmentColor = segmentColorHash(brushValue);
                        
                        // Apply saturation mixing (same as segmentation layer)
                        vec3 finalColor = mix(vec3(1.0, 1.0, 1.0), segmentColor, uSaturation);
                        emit(vec4(finalColor, uSelectedAlpha), 0u);
                    } else {
                        discard;
                    }
                `;
                builder.setFragmentMain(fragmentShader);
            } else {
                // For 2D slice view: use same approach as your working custom renderer
                this.base.brushHashTableManager.defineShader(builder);
                this.base.segmentColorShaderManager.defineShader(builder);
                builder.addUniform("highp mat4", "uViewMatrix");
                builder.addUniform("highp mat4", "uProjectionMatrix");
                builder.addVarying("highp vec2", "vScreenPosition");

                const vertexShader = `
                    vec2 positions[6] = vec2[6](
                        vec2(-1.0, -1.0),
                        vec2(1.0, -1.0),
                        vec2(1.0, 1.0),
                        vec2(-1.0, -1.0),
                        vec2(1.0, 1.0),
                        vec2(-1.0, 1.0)
                    );
                    vec2 pos = positions[gl_VertexID];
                    gl_Position = vec4(pos, 0.0, 1.0);
                    vScreenPosition = pos; // Pass NDC coordinates to fragment shader
                `;
                builder.setVertexMain(vertexShader);

                const fragmentShader = `
                    // Coordinate transformation is working correctly, now enable brush lookup
                    vec4 clipPos = vec4(vScreenPosition, 0.0, 1.0);
                    
                    // Transform from clip space to view space (inverse projection)
                    vec4 viewPos = inverse(uProjectionMatrix) * clipPos;
                    if (viewPos.w != 0.0) {
                        viewPos /= viewPos.w; // Perspective division
                    }
                    
                    // Transform from view space to world space (inverse view matrix)
                    vec4 worldPos4 = inverse(uViewMatrix) * viewPos;
                    vec3 worldPos = worldPos4.xyz;
                    
                    // Round to nearest voxel coordinate
                    ivec3 voxelPos = ivec3(round(worldPos));
                    
                    // Extract coordinates for hash function (z, y, x order)
                    // Handle negative coordinates properly - only process non-negative coordinates
                    if (voxelPos.x < 0 || voxelPos.y < 0 || voxelPos.z < 0) {
                        // Negative coordinates - no brush strokes in negative space
                        discard;
                    }
                    
                    uint z1 = uint(voxelPos.x);  // global z 
                    uint y1 = uint(voxelPos.y);  // global y  
                    uint x1 = uint(voxelPos.z);  // global x
                    
                    // Hash function matching CPU implementation
                    uint h1 = ((x1 * 73u) * 1271u) ^ ((y1 * 513u) * 1345u) ^ ((z1 * 421u) * 675u);
                    uint h2 = ((x1 * 127u) * 337u) ^ ((y1 * 111u) * 887u) ^ ((z1 * 269u) * 325u);
                    
                    uint64_t key;
                    key.value[0] = h1;
                    key.value[1] = h2;
                    
                    uint64_t brushValue;
                    if (brushStroke_get(key, brushValue)) {
                        // Found brush stroke - use segment color based on the stored value
                        vec3 segmentColor = segmentColorHash(brushValue);
                        
                        // Apply saturation mixing (same as segmentation layer)
                        vec3 finalColor = mix(vec3(1.0, 1.0, 1.0), segmentColor, uSaturation);
                        emit(vec4(finalColor, uSelectedAlpha), 0u);
                    } else {
                        // No brush stroke - discard fragment
                        discard;
                    }
                `;
                builder.setFragmentMain(fragmentShader);
            }
        }

        initializeShader(
            shader: ShaderProgram,
            renderContext: PerspectiveViewRenderContext | SliceViewPanelRenderContext,
        ) {
            const { gl } = this;
            const { gpuBrushHashTable, brushHashTableManager, segmentColorShaderManager, displayState } = this.base;

            // Initialize brush hash table for both views
            brushHashTableManager.enable(gl, shader, gpuBrushHashTable);

            // Initialize segment color shader
            const colorGroupState = displayState.segmentationColorGroupState.value;
            segmentColorShaderManager.enable(gl, shader, colorGroupState.segmentColorHash.value);

            // Set opacity uniforms to match parent segmentation layer
            const selectedAlpha = (displayState as any).selectedAlpha.value;
            const notSelectedAlpha = (displayState as any).notSelectedAlpha.value;
            const saturation = displayState.saturation.value;

            console.log('Brush stroke alpha and saturation values:', { selectedAlpha, notSelectedAlpha, saturation });

            gl.uniform1f(shader.uniform("uSelectedAlpha"), selectedAlpha);
            gl.uniform1f(shader.uniform("uNotSelectedAlpha"), notSelectedAlpha);
            gl.uniform1f(shader.uniform("uSaturation"), saturation);



            if (this instanceof PerspectiveViewRenderLayer) {
                // 3D perspective view: TODO - implement proper volume rendering uniforms
                gl.uniform3fv(shader.uniform("uTranslation"), new Float32Array([0, 0, 0]));
                gl.uniform3fv(shader.uniform("uChunkDataSize"), new Float32Array([100, 100, 100]));
                gl.uniform3fv(shader.uniform("uLowerClipBound"), new Float32Array([-1000, -1000, -1000]));
                gl.uniform3fv(shader.uniform("uUpperClipBound"), new Float32Array([1000, 1000, 1000]));

                // Set identity projection matrix for now
                const identityMatrix = new Float32Array([
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1
                ]);
                gl.uniformMatrix4fv(shader.uniform("uProjectionMatrix"), false, identityMatrix);
            } else {
                // For 2D slice view: use same matrices as your working custom renderer
                const sliceViewContext = renderContext as SliceViewPanelRenderContext;
                const sliceViewProjectionParameters = sliceViewContext.sliceView.projectionParameters.value;
                const { viewMatrix, projectionMat } = sliceViewProjectionParameters;

                // Set view matrix (transforms world coordinates to view coordinates)
                gl.uniformMatrix4fv(shader.uniform("uViewMatrix"), false, viewMatrix);

                // Set projection matrix (transforms view coordinates to clip coordinates)
                gl.uniformMatrix4fv(shader.uniform("uProjectionMatrix"), false, projectionMat);
            }
        }

        draw(
            renderContext: PerspectiveViewRenderContext | SliceViewPanelRenderContext,
            _attachment: VisibleLayerInfo<LayerView, AttachmentState>,
        ) {
            const { gl } = this;

            if (this.base.brushHashTable.size <= 0) {
                return;
            }

            const shader = this.getShader(renderContext);
            if (shader === null) {
                return;
            }

            shader.bind();
            this.initializeShader(shader, renderContext);

            // Choose draw call based on view type
            if (this instanceof PerspectiveViewRenderLayer) {
                // 3D view: TODO - implement proper volume rendering draw call
            } else {
                gl.drawArrays(gl.TRIANGLES, 0, 6);
            }
        }

        private getShader(renderContext: PerspectiveViewRenderContext | SliceViewPanelRenderContext) {
            const result = this.shaderGetter(renderContext.emitter);
            return result.shader;
        }
    }
    return C as MixinConstructor<typeof C, TBase>;
}

export const PerspectiveViewBrushStrokeLayer = BrushStrokeRenderLayer(
    PerspectiveViewRenderLayer,
);

export const SliceViewBrushStrokeLayer = BrushStrokeRenderLayer(
    SliceViewPanelRenderLayer,
);