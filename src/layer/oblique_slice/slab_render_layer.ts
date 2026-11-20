import { ChunkState } from "#src/chunk_manager/base.js";
import { ChunkRenderLayerFrontend } from "#src/chunk_manager/frontend.js";
import type { ImageUserLayer } from "#src/layer/image/index.js";
import {
  SLAB_RENDER_LAYER_RPC_ID,
  SLAB_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
} from "#src/layer/oblique_slice/base.js";
import { luts } from "#src/luts/luts.js";
import { RenderLayer } from "#src/renderlayer.js";
import { SharedWatchableValue } from "#src/shared_watchable_value.js";
import { getNormalizedChunkLayout } from "#src/sliceview/base.js";
import type {
  FrontendTransformedSource,
  SliceView,
} from "#src/sliceview/frontend.js";
import {
  getVolumetricTransformedSources,
  serializeAllTransformedSources,
} from "#src/sliceview/frontend.js";
import type { SliceViewRenderContext } from "#src/sliceview/renderlayer.js";
import type {
  VolumeChunkSource,
  ChunkFormat,
} from "#src/sliceview/volume/frontend.js";
import { defineChunkDataShaderAccess } from "#src/sliceview/volume/frontend.js";
import type { ImageRenderLayer } from "#src/sliceview/volume/image_renderlayer.js";
import { defineImageLayerShader } from "#src/sliceview/volume/image_renderlayer.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import {
  makeCachedDerivedWatchableValue,
  registerNested,
} from "#src/trackable_value.js";
import type { Borrowed } from "#src/util/disposable.js";
import { RefCounted } from "#src/util/disposable.js";
import { mat4, vec3 } from "#src/util/geom.js";
import { MessageList } from "#src/util/message_list.js";
import { forEachVisibleVolumeRenderingChunk } from "#src/volume_rendering/base.js";
import {
  drawBoxes,
  glsl_getBoxFaceVertexPosition,
} from "#src/webgl/bounding_box.js";
import type { ParameterizedContextDependentShaderGetter } from "#src/webgl/dynamic_shader.js";
import {
  makeWatchableShaderError,
  parameterizedContextDependentShaderGetter,
} from "#src/webgl/dynamic_shader.js";
import type { ShaderBuilder } from "#src/webgl/shader.js";
import type { ShaderControlsBuilderState } from "#src/webgl/shader_ui_controls.js";
import { setControlsInShader } from "#src/webgl/shader_ui_controls.js";
import { defineVertexId, VertexIdHelper } from "#src/webgl/vertex_id.js";

export type SlabMode = "max" | "min";

const tempMat4 = mat4.create();
const tempInvMat4 = mat4.create();

// The slab: like 3D volume rendering, but through the slicer's orthographic
// camera (depth = the slab). Each chunk draws its box; each fragment marches
// the normal segment inside that chunk, running the image layer's own shader
// per sample, and the best sample's depth merges chunks.
function defineSlabShader(
  builder: ShaderBuilder,
  mode: SlabMode,
  chunkFormat: ChunkFormat,
  numChannelDimensions: number,
  builderState: ShaderControlsBuilderState,
) {
  defineVertexId(builder);
  builder.addUniform("highp vec3", "uTranslation");
  builder.addUniform("highp mat4", "uModelViewProjectionMatrix");
  builder.addUniform("highp mat4", "uInvModelViewProjectionMatrix");
  builder.addUniform("highp vec3", "uChunkDataSize");
  builder.addUniform("highp vec3", "uLowerClipBound");
  builder.addUniform("highp vec3", "uUpperClipBound");
  builder.addUniform("highp int", "uSlabSamples");
  builder.addUniform("highp float", "uOpacity");
  builder.addUniform("vec4", "lut", 256);
  builder.addVarying("highp vec4", "vNormalizedPosition");
  builder.addVertexCode(glsl_getBoxFaceVertexPosition);
  // z = 0 so the box isn't clipped by the slab's near / far planes; the march
  // applies them.
  builder.setVertexMain(`
vec3 boxVertex = getBoxFaceVertexPosition(gl_VertexID);
vec3 position = max(uLowerClipBound, min(uUpperClipBound, uTranslation + boxVertex * uChunkDataSize));
vNormalizedPosition = gl_Position = uModelViewProjectionMatrix * vec4(position, 1.0);
gl_Position.z = 0.0;
`);
  builder.addOutputBuffer("vec4", "v4f_fragData0", 0);
  builder.addFragmentCode(`
vec3 curChunkPosition;
vec4 slabColor;
void emit(vec4 color) {
  slabColor = color;
}
`);
  defineChunkDataShaderAccess(
    builder,
    chunkFormat,
    numChannelDimensions,
    "curChunkPosition",
  );
  builder.addFragmentCode(`
vec2 slabRayBox(vec3 nearPoint, vec3 farPoint, vec3 boxStart, vec3 boxEnd) {
  vec3 rayVector = farPoint - nearPoint;
  float start = 0.0;
  float end = 1.0;
  for (int i = 0; i < 3; ++i) {
    float a = nearPoint[i];
    float b = farPoint[i];
    float r = rayVector[i];
    float lo = boxStart[i];
    float hi = boxEnd[i];
    float s = (a >= lo && a <= hi) ? 0.0 : min((lo - a) / r, (hi - a) / r);
    float e = (b >= lo && b <= hi) ? 1.0 : max((lo - a) / r, (hi - a) / r);
    start = max(start, s);
    end = min(end, e);
  }
  return vec2(start, end);
}
void userMain();
void main() {
  vec2 ndc = vNormalizedPosition.xy / vNormalizedPosition.w;
  vec4 nearH = uInvModelViewProjectionMatrix * vec4(ndc, -1.0, 1.0);
  vec4 farH = uInvModelViewProjectionMatrix * vec4(ndc, 1.0, 1.0);
  vec3 nearPoint = nearH.xyz / nearH.w;
  vec3 farPoint = farH.xyz / farH.w;
  // The slab shows only where the slicer plane (the ray's midpoint) cuts the volume.
  vec3 planePoint = mix(nearPoint, farPoint, 0.5);
  if (any(lessThan(planePoint, uLowerClipBound)) ||
      any(greaterThan(planePoint, uUpperClipBound))) {
    discard;
  }
  vec3 boxStart = max(uLowerClipBound, uTranslation);
  vec3 boxEnd = min(uTranslation + uChunkDataSize, uUpperClipBound);
  vec2 range = slabRayBox(nearPoint, farPoint, boxStart, boxEnd);
  float bestIntensity = -1.0;
  vec4 bestColor = vec4(0.0);
  int bestIndex = 0;
  float samples = float(uSlabSamples);
  int first = max(0, int(ceil(range.x * samples - 0.5)));
  int last = min(uSlabSamples, int(ceil(range.y * samples - 0.5)));
  for (int i = first; i < last; ++i) {
    float t = (float(i) + 0.5) / samples;
    if (t < range.x || t >= range.y) continue;
    curChunkPosition = mix(nearPoint, farPoint, t) - uTranslation;
    defaultMaxProjectionIntensity = 0.0;
    slabColor = vec4(0.0);
    userMain();
    float intensity = clamp(defaultMaxProjectionIntensity, 0.0, 1.0);
    ${mode === "min" ? "intensity = 1.0 - intensity;" : ""}
    if (intensity > bestIntensity) {
      bestIntensity = intensity;
      bestColor = slabColor;
      bestIndex = i;
    }
  }
  if (bestIntensity < 0.0) discard;
  gl_FragDepth = 0.25 + 0.5 * (1.0 - bestIntensity)
    + 1e-4 * float(bestIndex) / float(uSlabSamples);
  v4f_fragData0 = bestColor;
}
#define main userMain
`);
  defineImageLayerShader(builder, builderState);
}

interface SlabShaderContext {
  chunkFormat: ChunkFormat;
  mode: SlabMode;
}

// One image layer's slab, built from that layer's own volume, transform and
// shader state (read-only), so its chunk sources are the same memoized ones
// the 2D panels and 3D volume rendering use. Like 3D volume rendering, it
// draws one resolution level, chosen from the layer's depth-samples setting.
export class SlabRenderLayer extends RenderLayer {
  private shaderGetter: ParameterizedContextDependentShaderGetter<
    SlabShaderContext,
    ShaderControlsBuilderState,
    number
  >;
  private vertexIdHelper: VertexIdHelper;
  backend: ChunkRenderLayerFrontend;
  depthSamplesTarget: WatchableValueInterface<number>;

  constructor(public image: Borrowed<ImageRenderLayer>) {
    super();
    const { multiscaleSource, shaderControlState } = image;
    const { chunkManager } = multiscaleSource;
    const { gl } = chunkManager.chunkQueueManager;
    this.depthSamplesTarget = (
      image.userLayer as ImageUserLayer
    ).volumeRenderingDepthSamplesTarget;
    this.vertexIdHelper = this.registerDisposer(VertexIdHelper.get(gl));
    const numChannelDimensions = this.registerDisposer(
      makeCachedDerivedWatchableValue(
        (space) => space.rank,
        [image.channelCoordinateSpace],
      ),
    );
    this.shaderGetter = parameterizedContextDependentShaderGetter(this, gl, {
      memoizeKey: "oblique_slice/SlabRenderLayer",
      shaderError: makeWatchableShaderError(),
      parameters: shaderControlState.builderState,
      encodeParameters: (p) => p.key,
      extraParameters: numChannelDimensions,
      getContextKey: ({ chunkFormat, mode }) =>
        `${chunkFormat.shaderKey}/${mode}`,
      defineShader: (builder, { chunkFormat, mode }, builderState, rank) => {
        if (builderState.parseResult.errors.length !== 0) {
          throw new Error("Invalid UI control specification");
        }
        defineSlabShader(builder, mode, chunkFormat, rank, builderState);
      },
    });
    const rpc = chunkManager.rpc!;
    const backend = this.registerDisposer(
      new ChunkRenderLayerFrontend(this.layerChunkProgressInfo),
    );
    backend.RPC_TYPE_ID = SLAB_RENDER_LAYER_RPC_ID;
    backend.initializeCounterpart(rpc, {
      chunkManager: chunkManager.rpcId,
      localPosition: this.registerDisposer(
        SharedWatchableValue.makeFromExisting(rpc, image.localPosition),
      ).rpcId,
      depthSamplesTarget: this.registerDisposer(
        SharedWatchableValue.makeFromExisting(rpc, this.depthSamplesTarget),
      ).rpcId,
    });
    this.backend = backend;
    for (const signal of [
      shaderControlState.changed,
      image.transform.changed,
      image.opacity.changed,
      this.depthSamplesTarget.changed,
    ]) {
      this.registerDisposer(signal.add(this.redrawNeeded.dispatch));
    }
  }

  // Attaches to the slicer's SliceView as its view (front and worker), and
  // keeps the transformed sources current. Returns the attachment's owner.
  attachTo(sliceView: SliceView): {
    owner: RefCounted;
    sources: { readonly value: FrontendTransformedSource[][] };
  } {
    const owner = new RefCounted();
    const messages = new MessageList();
    const { backend } = this;
    const rpc = backend.rpc!;
    rpc.invoke("rendered_view.addLayer", {
      layer: backend.rpcId,
      view: sliceView.rpcId,
    });
    const sources = owner.registerDisposer(
      registerNested(
        (context, transform, displayDimensionRenderInfo) => {
          const transformedSources = getVolumetricTransformedSources(
            displayDimensionRenderInfo,
            transform,
            (options) => this.image.multiscaleSource.getSources(options),
            messages,
            this,
          );
          for (const scales of transformedSources) {
            for (const tsource of scales) {
              context.registerDisposer(tsource.source);
            }
          }
          sliceView.flushBackendProjectionParameters();
          rpc.invoke(SLAB_RENDER_LAYER_UPDATE_SOURCES_RPC_ID, {
            layer: backend.rpcId,
            view: sliceView.rpcId,
            sources: serializeAllTransformedSources(transformedSources),
            displayDimensionRenderInfo,
          });
          this.redrawNeeded.dispatch();
          return transformedSources;
        },
        this.image.transform,
        sliceView.displayDimensionRenderInfo,
      ),
    );
    owner.registerDisposer(() =>
      rpc.invoke("rendered_view.removeLayer", {
        layer: backend.rpcId,
        view: sliceView.rpcId,
      }),
    );
    return { owner, sources };
  }

  // Draws into the SliceView's bound offscreen framebuffer, depth test LESS.
  draw(
    renderContext: SliceViewRenderContext,
    sources: FrontendTransformedSource[][],
    mode: SlabMode,
  ) {
    const { projectionParameters } = renderContext;
    const scales = sources[0];
    if (scales === undefined || scales.length === 0) return;
    const { gl } = this.image;
    const chunkDataDisplaySize = vec3.create();
    const chunkPosition = vec3.create();
    let shader: ReturnType<typeof this.shaderGetter>["shader"] = null;
    let source: VolumeChunkSource | undefined;
    let chunkFormat: ChunkFormat | undefined;
    let originalChunkSize = vec3.create();
    let chunkDataSize: Uint32Array | undefined;
    let newSource = true;

    this.vertexIdHelper.enable();
    const wasCulling = gl.isEnabled(WebGL2RenderingContext.CULL_FACE);
    gl.enable(WebGL2RenderingContext.CULL_FACE);
    gl.cullFace(WebGL2RenderingContext.FRONT);
    forEachVisibleVolumeRenderingChunk(
      projectionParameters,
      this.image.localPosition.value,
      this.depthSamplesTarget.value,
      scales as FrontendTransformedSource<any, VolumeChunkSource>[],
      (tsource, _scaleIndex, _physicalSpacing, optimalSamples) => {
        source = tsource.source as VolumeChunkSource;
        chunkFormat = source.chunkFormat;
        const shaderResult = this.shaderGetter({ chunkFormat, mode });
        shader = shaderResult.shader;
        if (shader === null) return;
        shader.bind();
        setControlsInShader(
          gl,
          shader,
          this.image.shaderControlState,
          shaderResult.parameters.parseResult,
        );
        gl.uniform1f(shader.uniform("uOpacity"), this.image.opacity.value);
        const lutName = (window as any).lutName ?? "Tricolor 1";
        gl.uniform4fv(
          shader.uniform("lut"),
          new Float32Array((luts as any)[lutName]),
        );
        gl.uniform1i(
          shader.uniform("uSlabSamples"),
          Math.max(1, Math.ceil(optimalSamples)),
        );

        const chunkLayout = getNormalizedChunkLayout(
          projectionParameters,
          tsource.chunkLayout,
        );
        originalChunkSize = vec3.clone(chunkLayout.size);
        mat4.multiply(
          tempMat4,
          projectionParameters.viewProjectionMat,
          chunkLayout.transform,
        );
        mat4.invert(tempInvMat4, tempMat4);
        gl.uniformMatrix4fv(
          shader.uniform("uModelViewProjectionMatrix"),
          false,
          tempMat4,
        );
        gl.uniformMatrix4fv(
          shader.uniform("uInvModelViewProjectionMatrix"),
          false,
          tempInvMat4,
        );
        gl.uniform3fv(
          shader.uniform("uLowerClipBound"),
          tsource.lowerClipDisplayBound,
        );
        gl.uniform3fv(
          shader.uniform("uUpperClipBound"),
          tsource.upperClipDisplayBound,
        );
        const { fixedPositionWithinChunk, chunkDisplayDimensionIndices } =
          tsource;
        for (const chunkDim of chunkDisplayDimensionIndices) {
          fixedPositionWithinChunk[chunkDim] = 0;
        }
        chunkFormat.beginDrawing(gl, shader);
        chunkFormat.beginSource(gl, shader);
      },
      (tsource) => {
        if (shader === null) return;
        const chunk = source!.chunks.get(tsource.curPositionInChunks.join());
        if (chunk === undefined || chunk.state !== ChunkState.GPU_MEMORY) {
          return;
        }
        const {
          fixedPositionWithinChunk,
          chunkDisplayDimensionIndices,
          chunkTransform: { channelToChunkDimensionIndices },
        } = tsource;
        const chunkRank = source!.spec.rank;
        if (chunk.chunkDataSize !== chunkDataSize) {
          chunkDataSize = chunk.chunkDataSize;
          for (let i = 0; i < 3; ++i) {
            const chunkDim = chunkDisplayDimensionIndices[i];
            chunkDataDisplaySize[i] =
              chunkDim === -1 || chunkDim >= chunkRank
                ? 1
                : chunkDataSize[chunkDim];
          }
          gl.uniform3fv(shader.uniform("uChunkDataSize"), chunkDataDisplaySize);
        }
        const { chunkGridPosition } = chunk;
        for (let i = 0; i < 3; ++i) {
          const chunkDim = chunkDisplayDimensionIndices[i];
          chunkPosition[i] =
            chunkDim === -1 || chunkDim >= chunkRank
              ? 0
              : originalChunkSize[i] * chunkGridPosition[chunkDim];
        }
        gl.uniform3fv(shader.uniform("uTranslation"), chunkPosition);
        chunkFormat!.bindChunk(
          gl,
          shader,
          chunk,
          fixedPositionWithinChunk,
          chunkDisplayDimensionIndices,
          channelToChunkDimensionIndices,
          newSource,
        );
        newSource = false;
        drawBoxes(gl, 1, 1);
      },
    );
    if (shader !== null) chunkFormat?.endDrawing(gl, shader);
    if (!wasCulling) gl.disable(WebGL2RenderingContext.CULL_FACE);
    gl.cullFace(WebGL2RenderingContext.BACK);
    this.vertexIdHelper.disable();
  }
}
