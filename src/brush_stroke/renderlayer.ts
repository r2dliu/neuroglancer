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
import {
    getChunkDisplayTransformParameters,
    getLayerDisplayDimensionMapping,
} from "#src/render_coordinate_transform.js";
import type { RenderScaleHistogram } from "#src/render_scale_statistics.js";
import type {
    VisibilityTrackedRenderLayer,
} from "#src/renderlayer.js";
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
import type { MessageList } from "#src/util/message_list.js";
import { MessageSeverity } from "#src/util/message_list.js";
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

function getBrushStrokeChunkRenderParameters(
    chunkTransform: ValueOrError<ChunkTransformParameters>,
    displayDimensionRenderInfo: DisplayDimensionRenderInfo,
    messages: MessageList,
): BrushStrokeChunkRenderParameters | undefined {
    messages.clearMessages();
    const returnError = (message: string) => {
        messages.addMessage({ severity: MessageSeverity.error, message });
        return undefined;
    };
    if (chunkTransform.error !== undefined) {
        return returnError(chunkTransform.error);
    }
    const layerRenderDimensionMapping = getLayerDisplayDimensionMapping(
        chunkTransform.modelTransform,
        displayDimensionRenderInfo.displayDimensionIndices,
    );
    let chunkDisplayTransform: ChunkDisplayTransformParameters;
    try {
        chunkDisplayTransform = getChunkDisplayTransformParameters(
            chunkTransform,
            layerRenderDimensionMapping,
        );
    } catch (e) {
        return returnError((e as Error).message);
    }

    const { chunkTransform: chunkTransformParams } = chunkDisplayTransform;
    const { unpaddedRank } = chunkTransformParams.modelTransform;
    const modelClipBounds = new Float32Array(unpaddedRank * 2);
    modelClipBounds.fill(1, unpaddedRank);
    const { numChunkDisplayDims, chunkDisplayDimensionIndices } = chunkDisplayTransform;
    for (let i = 0; i < numChunkDisplayDims; ++i) {
        const chunkDim = chunkDisplayDimensionIndices[i];
        modelClipBounds[unpaddedRank + chunkDim] = 0;
    }

    return {
        chunkTransform,
        chunkDisplayTransform,
        modelClipBounds,
    };
}

export class BrushStrokeLayer extends RefCounted {
    public gpuBrushHashTable: GPUHashTable<any>;
    public brushHashTableManager = new HashMapShaderManager("brushStroke");
    redrawNeeded = new NullarySignal();

    constructor(
        public chunkManager: ChunkManager,
        public brushHashTable: BrushHashTable,
        public brushColor: Float32Array = new Float32Array([1, 0, 0, 0.8]), // Default red
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

    get enabled() {
        return this.brushHashTable.size > 0;
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

            // Create shader getter for brush stroke rendering
            this.shaderGetter = parameterizedEmitterDependentShaderGetter(this, this.gl, {
                memoizeKey: "brushStroke",
                parameters: constantWatchableValue(undefined),
                defineShader: (builder: ShaderBuilder) => {
                    this.defineShader(builder);
                },
            });
        }

        attach(attachment: VisibleLayerInfo<LayerView, AttachmentState>) {
            super.attach(attachment);
            const { chunkTransform } = this;
            const displayDimensionRenderInfo =
                attachment.view.displayDimensionRenderInfo.value;
            attachment.state = {
                chunkTransform,
                displayDimensionRenderInfo,
                chunkRenderParameters: getBrushStrokeChunkRenderParameters(
                    chunkTransform,
                    displayDimensionRenderInfo,
                    attachment.messages,
                ),
            };
        }

        updateAttachmentState(
            attachment: VisibleLayerInfo<LayerView, AttachmentState>,
        ): BrushStrokeChunkRenderParameters | undefined {
            const state = attachment.state!;
            const { chunkTransform } = this;
            const displayDimensionRenderInfo =
                attachment.view.displayDimensionRenderInfo.value;
            if (
                state !== undefined &&
                state.chunkTransform === chunkTransform &&
                state.displayDimensionRenderInfo === displayDimensionRenderInfo
            ) {
                return state.chunkRenderParameters;
            }
            state.chunkTransform = chunkTransform;
            state.displayDimensionRenderInfo = displayDimensionRenderInfo;
            const chunkRenderParameters = (state.chunkRenderParameters =
                getBrushStrokeChunkRenderParameters(
                    chunkTransform,
                    displayDimensionRenderInfo,
                    attachment.messages,
                ));
            return chunkRenderParameters;
        }

        get chunkTransform(): ValueOrError<ChunkTransformParameters> {
            // Return an error to skip chunk-based rendering for now
            // We'll implement proper coordinate transforms later
            return { error: "Brush stroke layers don't use chunk-based transforms" };
        }

        get gl() {
            return this.base.chunkManager.gl;
        }

        defineShader(builder: ShaderBuilder) {
            builder.addUniform("highp vec4", "uBrushColor");

            // Check if this is a perspective view (3D) or slice view (2D)
            const isPerspectiveView = this instanceof PerspectiveViewRenderLayer;

            if (isPerspectiveView) {
                // 3D view: TODO - implement proper volume rendering
                this.base.brushHashTableManager.defineShader(builder);
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
                    // DEBUG: Always render to test if 3D overlay works
                    emit(vec4(1.0, 0.0, 0.0, 0.8), 0u);
                    
                    /* TODO: Enable hash table lookup once we verify 3D overlay is working
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
                        emit(vec4(1.0, 0.0, 0.0, 0.8), 0u);
                    } else {
                        discard;
                    }
                    */
                `;
                builder.setFragmentMain(fragmentShader);
            } else {
                // For 2D slice view: use same approach as your working custom renderer
                this.base.brushHashTableManager.defineShader(builder);
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
                    // Use the same transformation as your working custom renderer, but in reverse
                    // vScreenPosition is NDC coordinates (-1 to 1)
                    
                    // Reverse the transformation: NDC -> clip space -> view space -> world space
                    // Start with NDC coordinates and assume z=0 for the current slice
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
                        // Found brush stroke - render with brush color
                        emit(uBrushColor, 0u);
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
            const { brushColor, gpuBrushHashTable, brushHashTableManager } = this.base;

            // Helper function to safely set uniforms
            const safeSetUniform = (name: string, setter: () => void) => {
                try {
                    const location = shader.uniform(name);
                    if (location !== null) {
                        setter();
                    }
                } catch {
                    // Uniform not found in shader, skip silently
                }
            };

            // Set the brush color uniform
            safeSetUniform("uBrushColor", () => {
                gl.uniform4fv(shader.uniform("uBrushColor"), brushColor);
            });

            // Initialize brush hash table for both views
            brushHashTableManager.enable(gl, shader, gpuBrushHashTable);

            if (this instanceof PerspectiveViewRenderLayer) {
                // 3D perspective view: TODO - implement proper volume rendering uniforms
                safeSetUniform("uTranslation", () => {
                    gl.uniform3fv(shader.uniform("uTranslation"), new Float32Array([0, 0, 0]));
                });
                safeSetUniform("uChunkDataSize", () => {
                    gl.uniform3fv(shader.uniform("uChunkDataSize"), new Float32Array([100, 100, 100]));
                });
                safeSetUniform("uLowerClipBound", () => {
                    gl.uniform3fv(shader.uniform("uLowerClipBound"), new Float32Array([-1000, -1000, -1000]));
                });
                safeSetUniform("uUpperClipBound", () => {
                    gl.uniform3fv(shader.uniform("uUpperClipBound"), new Float32Array([1000, 1000, 1000]));
                });
                safeSetUniform("uProjectionMatrix", () => {
                    // Set identity projection matrix for now
                    const identityMatrix = new Float32Array([
                        1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1
                    ]);
                    gl.uniformMatrix4fv(shader.uniform("uProjectionMatrix"), false, identityMatrix);
                });
            } else {
                // For 2D slice view: use same matrices as your working custom renderer
                const sliceViewContext = renderContext as SliceViewPanelRenderContext;
                const sliceViewProjectionParameters = sliceViewContext.sliceView.projectionParameters.value;
                const { viewMatrix, projectionMat } = sliceViewProjectionParameters;

                // Set view matrix (transforms world coordinates to view coordinates)
                safeSetUniform("uViewMatrix", () => {
                    console.log("🎨 Setting uViewMatrix:", viewMatrix);
                    gl.uniformMatrix4fv(shader.uniform("uViewMatrix"), false, viewMatrix);
                });

                // Set projection matrix (transforms view coordinates to clip coordinates)
                safeSetUniform("uProjectionMatrix", () => {
                    console.log("🎨 Setting uProjectionMatrix:", projectionMat);
                    gl.uniformMatrix4fv(shader.uniform("uProjectionMatrix"), false, projectionMat);
                });
            }
        }

        draw(
            renderContext: PerspectiveViewRenderContext | SliceViewPanelRenderContext,
            _attachment: VisibleLayerInfo<LayerView, AttachmentState>,
        ) {
            const { gl } = this;
            console.log("🎨 BrushStrokeRenderLayer.draw() called");
            console.log("🎨 Layer enabled:", this.base.enabled);
            console.log("🎨 Hash table size:", this.base.brushHashTable.size);
            console.log("🎨 Is perspective view:", this instanceof PerspectiveViewRenderLayer);

            // Check if layer should be rendered
            if (!this.base.enabled) {
                console.log("❌ Layer not enabled, skipping draw");
                return;
            }

            // Get the shader for this render context
            const shader = this.getShader(renderContext);
            if (shader === null) {
                console.log("❌ shader is null");
                return;
            }
            console.log("✅ shader obtained");

            // Bind and setup the shader
            shader.bind();
            this.initializeShader(shader, renderContext);

            // Choose draw call based on view type
            if (this instanceof PerspectiveViewRenderLayer) {
                // 3D view: TODO - implement proper volume rendering draw call
                console.log("🎨 3D view - skipping draw");
            } else {
                // 2D view: draw single full-screen quad
                console.log("🎨 2D view - drawing triangles");
                gl.drawArrays(gl.TRIANGLES, 0, 6);
                console.log("✅ 2D draw completed");
            }
        }

        endSlice(
            _shader: ShaderProgram,
        ) {
            // Nothing to cleanup for the simple version
        }



        isReady() {
            return true; // Always ready for testing
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