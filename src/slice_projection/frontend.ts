import { ChunkState } from "#src/chunk_manager/base.js";
import type { ChunkManager } from "#src/chunk_manager/frontend.js";
import { ChunkRenderLayerFrontend } from "#src/chunk_manager/frontend.js";
import type { CoordinateSpace } from "#src/coordinate_transform.js";
import type { VisibleLayerInfo } from "#src/layer/index.js";
import { luts } from "#src/luts/luts.js";
import type { PerspectivePanel } from "#src/perspective_view/panel.js";
import type {
  PerspectiveViewReadyRenderContext,
  PerspectiveViewRenderContext,
} from "#src/perspective_view/render_layer.js";
import { PerspectiveViewRenderLayer } from "#src/perspective_view/render_layer.js";
import type { ProjectionParameters } from "#src/projection_parameters.js";
import type { RenderLayerTransformOrError } from "#src/render_coordinate_transform.js";
import { SharedWatchableValue } from "#src/shared_watchable_value.js";
import type { SliceParameters } from "#src/slice_projection/base.js";
import {
  computeSliceToWorld,
  forEachChunkInSlice,
  getSliceSampleCount,
  getSliceTextureSize,
  SLICE_PROJECTION_RENDER_LAYER_RPC_ID,
  SLICE_PROJECTION_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
  SliceProjectionMode,
} from "#src/slice_projection/base.js";
import type { FrontendTransformedSource } from "#src/sliceview/frontend.js";
import {
  getVolumetricTransformedSources,
  serializeAllTransformedSources,
} from "#src/sliceview/frontend.js";
import type { SliceViewRenderLayer } from "#src/sliceview/renderlayer.js";
import type {
  ChunkFormat,
  MultiscaleVolumeChunkSource,
  VolumeChunk,
  VolumeChunkSource,
} from "#src/sliceview/volume/frontend.js";
import { defineChunkDataShaderAccess } from "#src/sliceview/volume/frontend.js";
import type {
  NestedStateManager,
  WatchableValueInterface,
} from "#src/trackable_value.js";
import {
  constantWatchableValue,
  makeCachedDerivedWatchableValue,
  registerNested,
  WatchableValue,
} from "#src/trackable_value.js";
import type { RefCounted } from "#src/util/disposable.js";
import { mat4, vec3 } from "#src/util/geom.js";
import { glsl_COLORMAPS } from "#src/webgl/colormaps.js";
import type { GL } from "#src/webgl/context.js";
import type {
  ParameterizedContextDependentShaderGetter,
  ParameterizedEmitterDependentShaderGetter,
} from "#src/webgl/dynamic_shader.js";
import {
  parameterizedContextDependentShaderGetter,
  parameterizedEmitterDependentShaderGetter,
  shaderCodeWithLineDirective,
} from "#src/webgl/dynamic_shader.js";
import type { TextureBuffer } from "#src/webgl/offscreen.js";
import {
  FramebufferConfiguration,
  makeTextureBuffers,
} from "#src/webgl/offscreen.js";
import { drawQuads, glsl_getQuadVertexPosition } from "#src/webgl/quad.js";
import type { ShaderProgram } from "#src/webgl/shader.js";
import type {
  ShaderControlsBuilderState,
  ShaderControlState,
} from "#src/webgl/shader_ui_controls.js";
import {
  addControlsToBuilder,
  setControlsInShader,
} from "#src/webgl/shader_ui_controls.js";
import { defineVertexId, VertexIdHelper } from "#src/webgl/vertex_id.js";

export interface SliceImageSource {
  multiscaleSource: MultiscaleVolumeChunkSource;
  transform: WatchableValueInterface<RenderLayerTransformOrError>;
  localPosition: WatchableValueInterface<Float32Array>;
  channelCoordinateSpace: WatchableValueInterface<CoordinateSpace>;
  shaderControlState: ShaderControlState;
}

export interface SliceProjectionRenderLayerOptions {
  chunkManager: ChunkManager;
  sliceParameters: WatchableValueInterface<SliceParameters>;
  imageSources: WatchableValueInterface<readonly SliceImageSource[]>;
}

interface SliceProjectionShaderParameters {
  numChannelDimensions: number;
  mode: SliceProjectionMode;
}

type SliceProjectionShaderGetter = ParameterizedContextDependentShaderGetter<
  { chunkFormat: ChunkFormat },
  ShaderControlsBuilderState,
  SliceProjectionShaderParameters
>;

interface SourceState {
  imageSource: SliceImageSource;
  shaderGetter: SliceProjectionShaderGetter;
}

type TransformedVolumeSource = FrontendTransformedSource<
  SliceViewRenderLayer,
  VolumeChunkSource
>;

interface AttachedSource {
  sourceState: SourceState;
  scales: TransformedVolumeSource[];
  localPosition: Float32Array;
}

interface SliceProjectionAttachmentState {
  sources: NestedStateManager<AttachedSource[]>;
}

const projectionSamplerTextureUnit = Symbol("sliceProjectionSampler");

const tempMat4 = mat4.create();
const tempChunkToSlice = mat4.create();
const tempCorner = vec3.create();
const tempQuadBounds = new Float32Array(4);
const tempChunkPosition = vec3.create();
const tempChunkDataDisplaySize = vec3.create();

let cachedLutName: string | undefined;
let cachedLutArray: Float32Array | undefined;

function getLutArray() {
  const lutName = (window as any).lutName ?? "Tricolor 1";
  if (lutName !== cachedLutName || cachedLutArray === undefined) {
    cachedLutName = lutName;
    cachedLutArray = new Float32Array(
      (luts as any)[lutName] ?? (luts as any)["Tricolor 1"],
    );
  }
  return cachedLutArray;
}

function computeQuadBounds(
  out: Float32Array,
  chunkToSlice: mat4,
  translation: vec3,
  chunkDataSize: vec3,
) {
  let lowerX = Number.POSITIVE_INFINITY;
  let lowerY = Number.POSITIVE_INFINITY;
  let upperX = Number.NEGATIVE_INFINITY;
  let upperY = Number.NEGATIVE_INFINITY;
  for (let corner = 0; corner < 8; ++corner) {
    for (let i = 0; i < 3; ++i) {
      tempCorner[i] = translation[i] + ((corner >> i) & 1) * chunkDataSize[i];
    }
    vec3.transformMat4(tempCorner, tempCorner, chunkToSlice);
    lowerX = Math.min(lowerX, tempCorner[0]);
    upperX = Math.max(upperX, tempCorner[0]);
    lowerY = Math.min(lowerY, tempCorner[1]);
    upperY = Math.max(upperY, tempCorner[1]);
  }
  out[0] = Math.max(-1, lowerX);
  out[1] = Math.max(-1, lowerY);
  out[2] = Math.min(1, upperX);
  out[3] = Math.min(1, upperY);
  return out[0] < out[2] && out[1] < out[3];
}

export class SliceProjectionRenderLayer extends PerspectiveViewRenderLayer<SliceProjectionAttachmentState> {
  chunkManager: ChunkManager;
  sliceParameters: WatchableValueInterface<SliceParameters>;
  imageSources: WatchableValueInterface<readonly SliceImageSource[]>;
  backend: ChunkRenderLayerFrontend;
  private sourcesGeneration = new WatchableValue(0);
  private sourceStates: NestedStateManager<SourceState[]>;
  private projectionMode: WatchableValueInterface<SliceProjectionMode>;
  private vertexIdHelper: VertexIdHelper;
  private projectionBuffer: FramebufferConfiguration<TextureBuffer>;
  private compositeShaderGetter: ParameterizedEmitterDependentShaderGetter<undefined>;

  get gl(): GL {
    return this.chunkManager.gl;
  }

  constructor(options: SliceProjectionRenderLayerOptions) {
    super();
    this.chunkManager = options.chunkManager;
    this.sliceParameters = options.sliceParameters;
    this.imageSources = options.imageSources;
    const { gl } = this;
    this.projectionMode = this.registerDisposer(
      makeCachedDerivedWatchableValue(
        (parameters: SliceParameters) => parameters.projectionMode,
        [this.sliceParameters],
      ),
    );
    this.projectionBuffer = this.registerDisposer(
      new FramebufferConfiguration(gl, {
        colorBuffers: makeTextureBuffers(gl, 1),
      }),
    );
    this.vertexIdHelper = this.registerDisposer(VertexIdHelper.get(gl));
    this.sourceStates = this.registerDisposer(
      registerNested((context, imageSources) => {
        return imageSources.map((imageSource) => {
          context.registerDisposer(
            imageSource.shaderControlState.changed.add(
              this.redrawNeeded.dispatch,
            ),
          );
          return {
            imageSource,
            shaderGetter: this.makeShaderGetter(context, imageSource),
          };
        });
      }, this.imageSources),
    );
    this.registerDisposer(
      this.imageSources.changed.add(() => this.bumpSourcesGeneration()),
    );
    this.registerDisposer(
      this.sliceParameters.changed.add(this.redrawNeeded.dispatch),
    );
    this.compositeShaderGetter = parameterizedEmitterDependentShaderGetter(
      this,
      gl,
      {
        memoizeKey: "SliceProjectionComposite",
        parameters: constantWatchableValue(undefined),
        defineShader: (builder) => {
          defineVertexId(builder);
          builder.addUniform("highp mat4", "uModelViewProjection");
          builder.addUniform("highp vec3", "uBackgroundColor");
          builder.addUniform("highp uint", "uPickId");
          builder.addTextureSampler(
            "sampler2D",
            "uProjectionSampler",
            projectionSamplerTextureUnit,
          );
          builder.addVarying("highp vec2", "vTexCoord");
          builder.addVertexCode(glsl_getQuadVertexPosition);
          builder.setVertexMain(`
vec2 corner = getQuadVertexPosition(vec2(-1.0, -1.0), vec2(1.0, 1.0));
vTexCoord = corner * 0.5 + 0.5;
gl_Position = uModelViewProjection * vec4(corner, 0.0, 1.0);
`);
          builder.setFragmentMain(`
vec4 value = texture(uProjectionSampler, vTexCoord);
emit(vec4(mix(uBackgroundColor, value.rgb, value.a), 1.0), uPickId);
`);
        },
      },
    );
    const sharedObject = this.registerDisposer(
      new ChunkRenderLayerFrontend(this.layerChunkProgressInfo),
    );
    const rpc = this.chunkManager.rpc!;
    sharedObject.RPC_TYPE_ID = SLICE_PROJECTION_RENDER_LAYER_RPC_ID;
    sharedObject.initializeCounterpart(rpc, {
      chunkManager: this.chunkManager.rpcId,
      sliceParameters: this.registerDisposer(
        SharedWatchableValue.makeFromExisting(rpc, this.sliceParameters),
      ).rpcId,
    });
    this.backend = sharedObject;
  }

  private bumpSourcesGeneration() {
    this.sourcesGeneration.value = this.sourcesGeneration.value + 1;
  }

  private makeShaderGetter(
    context: RefCounted,
    imageSource: SliceImageSource,
  ): SliceProjectionShaderGetter {
    const extraParameters = context.registerDisposer(
      makeCachedDerivedWatchableValue(
        (space: CoordinateSpace, mode: SliceProjectionMode) => ({
          numChannelDimensions: space.rank,
          mode,
        }),
        [imageSource.channelCoordinateSpace, this.projectionMode],
      ),
    );
    return parameterizedContextDependentShaderGetter(context, this.gl, {
      memoizeKey: "SliceProjectionRenderLayer",
      parameters: imageSource.shaderControlState.builderState,
      getContextKey: ({ chunkFormat }) => chunkFormat.shaderKey,
      extraParameters,
      defineShader: (
        builder,
        { chunkFormat },
        shaderBuilderState,
        shaderParametersState,
      ) => {
        if (shaderBuilderState.parseResult.errors.length !== 0) {
          throw new Error("Invalid UI control specification");
        }
        defineVertexId(builder);
        builder.addOutputBuffer("vec4", "out_value", 0);
        builder.addUniform("highp mat4", "uSliceToChunk");
        builder.addUniform("highp vec3", "uTranslation");
        builder.addUniform("highp vec3", "uChunkDataSize");
        builder.addUniform("highp vec3", "uLowerClipBound");
        builder.addUniform("highp vec3", "uUpperClipBound");
        builder.addUniform("highp vec2", "uQuadLower");
        builder.addUniform("highp vec2", "uQuadUpper");
        builder.addUniform("highp int", "uNumSamples");
        builder.addUniform("vec4", "lut", 256);
        builder.addVarying("highp vec2", "vSlicePosition");
        builder.addVertexCode(glsl_getQuadVertexPosition);
        builder.setVertexMain(`
vec2 slicePosition = getQuadVertexPosition(uQuadLower, uQuadUpper);
vSlicePosition = slicePosition;
gl_Position = vec4(slicePosition, 0.0, 1.0);
`);
        builder.addFragmentCode(`
#define VOLUME_RENDERING true
vec3 curChunkPosition;
vec4 sampleColor;
void userMain();
`);
        defineChunkDataShaderAccess(
          builder,
          chunkFormat,
          shaderParametersState.numChannelDimensions,
          "curChunkPosition",
        );
        const isMin = shaderParametersState.mode === SliceProjectionMode.MIN;
        builder.addFragmentCode(`
void emitIntensity(float value) {}
void emitRGBA(vec4 rgba) {
  sampleColor = vec4(rgba.rgb, clamp(rgba.a, 0.0, 1.0));
}
void emitRGB(vec3 rgb) {
  emitRGBA(vec4(rgb, 1.0));
}
void emitGrayscale(float value) {
  emitRGBA(vec4(value, value, value, 1.0));
}
void emitTransparent() {
  emitRGBA(vec4(0.0, 0.0, 0.0, 0.0));
}
`);
        builder.addFragmentCode(glsl_COLORMAPS);
        addControlsToBuilder(shaderBuilderState, builder);
        builder.addFragmentCode(
          "\n#define main userMain\n" +
            shaderCodeWithLineDirective(shaderBuilderState.parseResult.code) +
            "\n#undef main\n",
        );
        builder.setFragmentMainFunction(`
void main() {
  vec3 projected = vec3(${isMin ? "1.0" : "0.0"});
  float coverage = 0.0;
  bool covered = false;
  for (int i = 0; i < uNumSamples; ++i) {
    float t = uNumSamples == 1 ? 0.0 : (2.0 * float(i) / float(uNumSamples - 1) - 1.0);
    vec3 chunkPosition = (uSliceToChunk * vec4(vSlicePosition, t, 1.0)).xyz;
    if (any(lessThan(chunkPosition, uLowerClipBound)) ||
        any(greaterThan(chunkPosition, uUpperClipBound))) continue;
    vec3 positionWithinChunk = chunkPosition - uTranslation;
    if (any(lessThan(positionWithinChunk, vec3(0.0))) ||
        any(greaterThanEqual(positionWithinChunk, uChunkDataSize))) continue;
    curChunkPosition = positionWithinChunk;
    sampleColor = vec4(0.0);
    userMain();
    if (sampleColor.a <= 0.0) continue;
    projected = ${isMin ? "min" : "max"}(projected, sampleColor.rgb);
    coverage = max(coverage, sampleColor.a);
    covered = true;
  }
  if (!covered) discard;
  out_value = vec4(projected, coverage);
}
`);
      },
    });
  }

  attach(
    attachment: VisibleLayerInfo<
      PerspectivePanel,
      SliceProjectionAttachmentState
    >,
  ) {
    super.attach(attachment);
    attachment.state = {
      sources: attachment.registerDisposer(
        registerNested(
          (context, generation, displayDimensionRenderInfo) => {
            generation;
            const result: AttachedSource[] = [];
            for (const sourceState of this.sourceStates.value) {
              const { imageSource } = sourceState;
              const transformedSources = getVolumetricTransformedSources(
                displayDimensionRenderInfo,
                imageSource.transform.value,
                (options) => imageSource.multiscaleSource.getSources(options),
                attachment.messages,
                this,
              ) as TransformedVolumeSource[][];
              for (const scales of transformedSources) {
                for (const tsource of scales) {
                  context.registerDisposer(tsource.source);
                }
              }
              if (transformedSources.length === 0) continue;
              result.push({
                sourceState,
                scales: transformedSources[0],
                localPosition: imageSource.localPosition.value,
              });
            }
            attachment.view.flushBackendProjectionParameters();
            this.backend.rpc!.invoke(
              SLICE_PROJECTION_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
              {
                layer: this.backend.rpcId,
                view: attachment.view.rpcId,
                sources: serializeAllTransformedSources(
                  result.map((x) => x.scales),
                ),
                localPositions: result.map((x) => x.localPosition),
              },
            );
            this.redrawNeeded.dispatch();
            return result;
          },
          this.sourcesGeneration,
          attachment.view.displayDimensionRenderInfo,
        ),
      ),
    };
  }

  draw(
    renderContext: PerspectiveViewRenderContext,
    attachment: VisibleLayerInfo<
      PerspectivePanel,
      SliceProjectionAttachmentState
    >,
  ) {
    if (!renderContext.emitColor) return;
    const parameters = this.sliceParameters.value;
    const { projectionParameters } = renderContext;
    const { canonicalVoxelFactors } =
      projectionParameters.displayDimensionRenderInfo;
    const { gl } = this;
    const textureSize = getSliceTextureSize(parameters);
    const isMin = parameters.projectionMode === SliceProjectionMode.MIN;
    this.projectionBuffer.bind(textureSize.width, textureSize.height);
    gl.disable(WebGL2RenderingContext.STENCIL_TEST);
    gl.disable(WebGL2RenderingContext.DEPTH_TEST);
    gl.disable(WebGL2RenderingContext.CULL_FACE);
    gl.enable(WebGL2RenderingContext.BLEND);
    gl.blendEquationSeparate(
      isMin ? WebGL2RenderingContext.MIN : WebGL2RenderingContext.MAX,
      WebGL2RenderingContext.MAX,
    );
    const identity = isMin ? 1 : 0;
    gl.clearColor(identity, identity, identity, 0);
    gl.clear(WebGL2RenderingContext.COLOR_BUFFER_BIT);
    this.vertexIdHelper.enable();
    for (const attachedSource of attachment.state!.sources.value) {
      this.drawSource(attachedSource, parameters, projectionParameters);
    }
    this.vertexIdHelper.disable();
    gl.blendEquation(WebGL2RenderingContext.FUNC_ADD);
    gl.disable(WebGL2RenderingContext.BLEND);
    gl.enable(WebGL2RenderingContext.DEPTH_TEST);
    gl.enable(WebGL2RenderingContext.STENCIL_TEST);
    renderContext.bindFramebuffer();
    this.drawSliceQuad(renderContext, parameters, canonicalVoxelFactors);
  }

  private drawSource(
    attachedSource: AttachedSource,
    parameters: SliceParameters,
    projectionParameters: ProjectionParameters,
  ) {
    const { gl } = this;
    const { imageSource, shaderGetter } = attachedSource.sourceState;
    let shader: ShaderProgram | null = null;
    let chunkFormat: ChunkFormat | null | undefined;
    let chunks: Map<string, VolumeChunk> | undefined;
    let chunkDataSize: Uint32Array | undefined;
    let newSource = true;
    forEachChunkInSlice(
      parameters,
      projectionParameters.globalPosition,
      attachedSource.localPosition,
      projectionParameters.displayDimensionRenderInfo.canonicalVoxelFactors,
      attachedSource.scales,
      (info) => {
        const { tsource } = info;
        const { source } = tsource;
        chunkFormat = source.chunkFormat;
        if (chunkFormat === null || chunkFormat === undefined) return;
        const shaderResult = shaderGetter({ chunkFormat });
        shader = shaderResult.shader;
        if (shader === null) return;
        shader.bind();
        setControlsInShader(
          gl,
          shader,
          imageSource.shaderControlState,
          shaderResult.parameters.parseResult.controls,
        );
        gl.uniform4fv(shader.uniform("lut"), getLutArray());
        chunkFormat.beginDrawing(gl, shader);
        chunkFormat.beginSource(gl, shader);
        mat4.multiply(
          tempMat4,
          tsource.chunkLayout.invTransform,
          info.sliceToWorld,
        );
        mat4.invert(tempChunkToSlice, tempMat4);
        gl.uniformMatrix4fv(shader.uniform("uSliceToChunk"), false, tempMat4);
        gl.uniform3fv(
          shader.uniform("uLowerClipBound"),
          tsource.lowerClipDisplayBound,
        );
        gl.uniform3fv(
          shader.uniform("uUpperClipBound"),
          tsource.upperClipDisplayBound,
        );
        gl.uniform1i(
          shader.uniform("uNumSamples"),
          getSliceSampleCount(parameters, info),
        );
        for (const chunkDim of tsource.chunkDisplayDimensionIndices) {
          tsource.fixedPositionWithinChunk[chunkDim] = 0;
        }
        chunks = source.chunks;
        chunkDataSize = undefined;
        newSource = true;
      },
      (tsource) => {
        if (shader === null || chunks === undefined) return;
        const chunk = chunks.get(tsource.curPositionInChunks.join());
        if (chunk === undefined || chunk.state !== ChunkState.GPU_MEMORY) {
          return;
        }
        const {
          chunkDisplayDimensionIndices,
          fixedPositionWithinChunk,
          chunkTransform: { channelToChunkDimensionIndices },
        } = tsource;
        const chunkRank = imageSource.multiscaleSource.rank;
        const newChunkDataSize = chunk.chunkDataSize;
        if (newChunkDataSize !== chunkDataSize) {
          chunkDataSize = newChunkDataSize;
          for (let i = 0; i < 3; ++i) {
            const chunkDim = chunkDisplayDimensionIndices[i];
            tempChunkDataDisplaySize[i] =
              chunkDim === -1 || chunkDim >= chunkRank
                ? 1
                : newChunkDataSize[chunkDim];
          }
          gl.uniform3fv(
            shader.uniform("uChunkDataSize"),
            tempChunkDataDisplaySize,
          );
        }
        const originalChunkSize = tsource.chunkLayout.size;
        const { chunkGridPosition } = chunk;
        for (let i = 0; i < 3; ++i) {
          const chunkDim = chunkDisplayDimensionIndices[i];
          tempChunkPosition[i] =
            chunkDim === -1 || chunkDim >= chunkRank
              ? 0
              : originalChunkSize[i] * chunkGridPosition[chunkDim];
        }
        if (
          !computeQuadBounds(
            tempQuadBounds,
            tempChunkToSlice,
            tempChunkPosition,
            tempChunkDataDisplaySize,
          )
        ) {
          return;
        }
        gl.uniform3fv(shader.uniform("uTranslation"), tempChunkPosition);
        gl.uniform2f(
          shader.uniform("uQuadLower"),
          tempQuadBounds[0],
          tempQuadBounds[1],
        );
        gl.uniform2f(
          shader.uniform("uQuadUpper"),
          tempQuadBounds[2],
          tempQuadBounds[3],
        );
        chunkFormat!.bindChunk(
          gl,
          shader,
          chunk,
          fixedPositionWithinChunk,
          chunkDisplayDimensionIndices,
          channelToChunkDimensionIndices,
          newSource,
        );
        drawQuads(gl, 1, 1);
        newSource = false;
      },
    );
    if (shader !== null && chunkFormat != null) {
      chunkFormat.endDrawing(gl, shader);
    }
  }

  private drawSliceQuad(
    renderContext: PerspectiveViewRenderContext,
    parameters: SliceParameters,
    canonicalVoxelFactors: Float64Array,
  ) {
    const { gl } = this;
    const shaderResult = this.compositeShaderGetter(renderContext.emitter);
    const { shader } = shaderResult;
    if (shader === null) return;
    shader.bind();
    this.vertexIdHelper.enable();
    computeSliceToWorld(tempMat4, parameters, canonicalVoxelFactors, 1);
    mat4.multiply(
      tempMat4,
      renderContext.projectionParameters.viewProjectionMat,
      tempMat4,
    );
    gl.uniformMatrix4fv(
      shader.uniform("uModelViewProjection"),
      false,
      tempMat4,
    );
    gl.uniform3fv(
      shader.uniform("uBackgroundColor"),
      parameters.backgroundColor,
    );
    gl.uniform1ui(
      shader.uniform("uPickId"),
      renderContext.pickIDs.register(this),
    );
    const textureUnit = shader.textureUnit(projectionSamplerTextureUnit);
    gl.activeTexture(WebGL2RenderingContext.TEXTURE0 + textureUnit);
    gl.bindTexture(
      WebGL2RenderingContext.TEXTURE_2D,
      this.projectionBuffer.colorBuffers[0].texture,
    );
    gl.texParameteri(
      WebGL2RenderingContext.TEXTURE_2D,
      WebGL2RenderingContext.TEXTURE_MIN_FILTER,
      WebGL2RenderingContext.LINEAR,
    );
    gl.texParameteri(
      WebGL2RenderingContext.TEXTURE_2D,
      WebGL2RenderingContext.TEXTURE_MAG_FILTER,
      WebGL2RenderingContext.LINEAR,
    );
    drawQuads(gl, 1, 1);
    gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, null);
    this.vertexIdHelper.disable();
  }

  isReady(
    renderContext: PerspectiveViewReadyRenderContext,
    attachment: VisibleLayerInfo<
      PerspectivePanel,
      SliceProjectionAttachmentState
    >,
  ) {
    const parameters = this.sliceParameters.value;
    const { projectionParameters } = renderContext;
    let missing = false;
    for (const attachedSource of attachment.state!.sources.value) {
      forEachChunkInSlice(
        parameters,
        projectionParameters.globalPosition,
        attachedSource.localPosition,
        projectionParameters.displayDimensionRenderInfo.canonicalVoxelFactors,
        attachedSource.scales,
        () => {},
        (tsource) => {
          const chunk = tsource.source.chunks.get(
            tsource.curPositionInChunks.join(),
          );
          if (chunk === undefined || chunk.state !== ChunkState.GPU_MEMORY) {
            missing = true;
          }
        },
      );
      if (missing) return false;
    }
    return true;
  }
}
