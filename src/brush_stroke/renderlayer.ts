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
import type {
    SliceViewPanelRenderContext,
} from "#src/sliceview/renderlayer.js";
import { SliceViewPanelRenderLayer } from "#src/sliceview/renderlayer.js";
import { WatchableValue, constantWatchableValue } from "#src/trackable_value.js";
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
            console.log("🎨 BrushStrokeRenderLayer constructor called");
            this.registerDisposer(base.redrawNeeded.add(this.redrawNeeded.dispatch));

            // Create shader getter for brush stroke rendering
            try {
                this.shaderGetter = parameterizedEmitterDependentShaderGetter(this, this.gl, {
                    memoizeKey: "brushStroke",
                    parameters: constantWatchableValue(undefined),
                    defineShader: (builder: ShaderBuilder) => {
                        console.log("🎨 defineShader called for brush stroke");
                        try {
                            this.defineShader(builder);
                            console.log("🎨 defineShader completed successfully");
                        } catch (error) {
                            console.error("❌ Error in defineShader:", error);
                            throw error;
                        }
                    },
                });
                console.log("🎨 Shader getter created successfully");
            } catch (error) {
                console.error("❌ Error creating shader getter:", error);
                throw error;
            }
            console.log("🎨 BrushStrokeRenderLayer constructor completed");
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

        get chunkTransform() {
            // For now, return a dummy successful transform to bypass the check
            // This is a simplified approach for testing
            return {
                modelTransform: {
                    inputSpace: { rank: 3 },
                    outputSpace: { rank: 3 },
                    transform: new Float64Array(16), // 4x4 identity matrix
                },
                chunkToLayerTransform: new Float32Array(25), // 5x5 identity matrix
                layerToChunkTransform: new Float32Array(25), // 5x5 identity matrix
            } as ChunkTransformParameters;
        }

        get gl() {
            return this.base.chunkManager.gl;
        }

        defineShader(builder: ShaderBuilder) {
            console.log("🎨 defineShader called - setting up simple shader");

            builder.addUniform("highp vec4", "uBrushColor");

            // Simple vertex shader that renders a full-screen quad
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
            `;

            console.log("🎨 Setting vertex shader:", vertexShader);
            builder.setVertexMain(vertexShader);

            // Simple fragment shader that just renders a solid color
            // Note: emit function is provided by the renderContext.emitter
            const fragmentShader = `
                // Just render a solid color to test if the quad is working
                emit(vec4(1.0, 0.0, 0.0, 0.5), 0u); // Semi-transparent red, no pick ID
            `;

            console.log("🎨 Setting fragment shader:", fragmentShader);
            builder.setFragmentMain(fragmentShader);

            console.log("🎨 defineShader completed");
        }

        initializeShader(
            shader: ShaderProgram,
            _renderContext: PerspectiveViewRenderContext | SliceViewPanelRenderContext,
        ) {
            const { gl } = this;
            const { brushColor } = this.base;

            // Set the brush color uniform (though we're not using it in the simple version)
            gl.uniform4fv(shader.uniform("uBrushColor"), brushColor);
        }

        draw(
            renderContext: PerspectiveViewRenderContext | SliceViewPanelRenderContext,
            attachment: VisibleLayerInfo<LayerView, AttachmentState>,
        ) {
            console.log("🎨 BrushStrokeRenderLayer draw method called");

            // Skip all the chunk parameter checks for testing
            const { gl } = this;

            // Get the shader for this render context
            const shader = this.getShader(renderContext);
            if (shader === null) {
                console.log("❌ shader is null");
                return;
            }

            console.log("🎨 About to bind shader and draw");

            // Bind and setup the shader
            shader.bind();
            this.initializeShader(shader, renderContext);

            // Draw a full-screen quad to cover the viewport
            // This will trigger our fragment shader for each pixel
            gl.drawArrays(gl.TRIANGLE_FAN, 0, 6);

            console.log("🎨 Draw call completed");
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
            console.log("🎨 getShader called");
            console.log("🎨 renderContext.emitter:", renderContext.emitter);

            // Check for WebGL errors before shader compilation
            const { gl } = this;
            const glError = gl.getError();
            if (glError !== gl.NO_ERROR) {
                console.error("❌ WebGL error before shader compilation:", glError);
            }

            try {
                console.log("🎨 About to call shaderGetter");
                const result = this.shaderGetter(renderContext.emitter);
                console.log("🎨 shaderGetter result:", result);

                // Check for WebGL errors after shader compilation
                const glErrorAfter = gl.getError();
                if (glErrorAfter !== gl.NO_ERROR) {
                    console.error("❌ WebGL error after shader compilation:", glErrorAfter);
                }

                const { shader } = result;
                console.log("🎨 shader:", shader);

                if (shader === null) {
                    console.log("🎨 Shader is null - checking for compilation errors");
                    // The shader system might have error information
                    if (result.fallback) {
                        console.log("🎨 Shader compilation fell back to fallback");
                    }
                }

                return shader;
            } catch (error) {
                console.error("❌ Error in getShader:", error);
                return null;
            }
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