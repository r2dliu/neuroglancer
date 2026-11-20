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
import type { TransformedSource } from "#src/sliceview/base.js";
import {
  filterVisibleSources,
  forEachVisibleVolumetricChunk,
  getNormalizedChunkLayout,
} from "#src/sliceview/base.js";
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
import type { ShaderBuilder, ShaderProgram } from "#src/webgl/shader.js";
import type { ShaderControlsBuilderState } from "#src/webgl/shader_ui_controls.js";
import { setControlsInShader } from "#src/webgl/shader_ui_controls.js";
import { defineVertexId, VertexIdHelper } from "#src/webgl/vertex_id.js";

export type SlabMode = "max" | "min";

// "volume": one level, picked like 3D volume rendering (the slicer in 3D).
// "slice": the levels a 2D panel would load for its zoom, coarse to fine, each
// sample taken from the finest level loaded there (the slicer's 2D pane).
export type SlabLevelRule = "volume" | "slice";

const OWNER_TEXTURE = Symbol("oblique_slice/ownerTexture");
const NO_OWNER = 255;

const tempMat4 = mat4.create();
const tempInvMat4 = mat4.create();
const tempScale = mat4.create();
const tempNear = vec3.create();
const tempFar = vec3.create();
const tempPoint = vec3.create();

// The slab: like 3D volume rendering, but through the slicer's orthographic
// camera (depth = the slab). Each chunk draws its box; each fragment marches
// the normal segment inside that chunk, running the image layer's own shader
// per sample, and the best sample's depth merges chunks. With `fallback`, a
// sample is skipped where a finer level owns it (per finest-level chunk cell).
function defineSlabShader(
  builder: ShaderBuilder,
  mode: SlabMode,
  fallback: boolean,
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
  if (fallback) {
    builder.addTextureSampler("usampler3D", "uOwnerSampler", OWNER_TEXTURE);
    builder.addUniform("highp mat4", "uLevelToOwnerCell");
    builder.addUniform("highp ivec3", "uOwnerSize");
    builder.addUniform("highp uint", "uLevelRank");
    builder.addFragmentCode(`
bool ownedByFiner(vec3 levelPosition) {
  ivec3 cell = ivec3(floor((uLevelToOwnerCell * vec4(levelPosition, 1.0)).xyz));
  if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, uOwnerSize))) {
    return false;
  }
  return texelFetch(uOwnerSampler, cell, 0).r < uLevelRank;
}
`);
  }
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
    vec3 samplePoint = mix(nearPoint, farPoint, t);
    ${fallback ? "if (ownedByFiner(samplePoint)) continue;" : ""}
    curChunkPosition = samplePoint - uTranslation;
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
  fallback: boolean;
}

interface LevelDraw {
  shader: ShaderProgram;
  source: VolumeChunkSource;
  chunkFormat: ChunkFormat;
  chunkSize: vec3;
  chunkDataSize: Uint32Array | undefined;
  newSource: boolean;
}

// A level's chunk layout in the slab's view, with its loaded chunk cells.
interface LevelChunks {
  tsource: FrontendTransformedSource;
  size: vec3;
  transform: mat4;
  invTransform: mat4;
  loaded: Set<string>;
  cells: vec3[];
  missing: number;
}

const cellKey = (cell: ArrayLike<number>) => `${cell[0]},${cell[1]},${cell[2]}`;

// One image layer's slab, built from that layer's own volume, transform and
// shader state (read-only), so its chunk sources are the same memoized ones
// the 2D panels and 3D volume rendering use.
export class SlabRenderLayer extends RenderLayer {
  private shaderGetter: ParameterizedContextDependentShaderGetter<
    SlabShaderContext,
    ShaderControlsBuilderState,
    number
  >;
  private vertexIdHelper: VertexIdHelper;
  private ownerTexture: WebGLTexture | null = null;
  backend: ChunkRenderLayerFrontend;
  depthSamplesTarget: WatchableValueInterface<number>;
  renderScaleTarget: WatchableValueInterface<number>;

  constructor(
    public image: Borrowed<ImageRenderLayer>,
    public levelRule: SlabLevelRule,
  ) {
    super();
    const { multiscaleSource, shaderControlState } = image;
    const { chunkManager } = multiscaleSource;
    const { gl } = chunkManager.chunkQueueManager;
    this.depthSamplesTarget = (
      image.userLayer as ImageUserLayer
    ).volumeRenderingDepthSamplesTarget;
    this.renderScaleTarget = image.renderScaleTarget;
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
      getContextKey: ({ chunkFormat, mode, fallback }) =>
        `${chunkFormat.shaderKey}/${mode}/${fallback}`,
      defineShader: (
        builder,
        { chunkFormat, mode, fallback },
        builderState,
        rank,
      ) => {
        if (builderState.parseResult.errors.length !== 0) {
          throw new Error("Invalid UI control specification");
        }
        defineSlabShader(
          builder,
          mode,
          fallback,
          chunkFormat,
          rank,
          builderState,
        );
      },
    });
    const rpc = chunkManager.rpc!;
    const backend = this.registerDisposer(
      new ChunkRenderLayerFrontend(this.layerChunkProgressInfo),
    );
    backend.RPC_TYPE_ID = SLAB_RENDER_LAYER_RPC_ID;
    backend.initializeCounterpart(rpc, {
      chunkManager: chunkManager.rpcId,
      levelRule,
      localPosition: this.registerDisposer(
        SharedWatchableValue.makeFromExisting(rpc, image.localPosition),
      ).rpcId,
      depthSamplesTarget: this.registerDisposer(
        SharedWatchableValue.makeFromExisting(rpc, this.depthSamplesTarget),
      ).rpcId,
      renderScaleTarget: this.registerDisposer(
        SharedWatchableValue.makeFromExisting(rpc, this.renderScaleTarget),
      ).rpcId,
    });
    this.backend = backend;
    for (const signal of [
      shaderControlState.changed,
      image.transform.changed,
      image.opacity.changed,
      this.depthSamplesTarget.changed,
      this.renderScaleTarget.changed,
    ]) {
      this.registerDisposer(signal.add(this.redrawNeeded.dispatch));
    }
  }

  disposed() {
    this.image.gl.deleteTexture(this.ownerTexture);
    super.disposed();
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
    const scales = sources[0];
    if (scales === undefined || scales.length === 0) return;
    const { gl } = this.image;
    this.vertexIdHelper.enable();
    const wasCulling = gl.isEnabled(WebGL2RenderingContext.CULL_FACE);
    gl.enable(WebGL2RenderingContext.CULL_FACE);
    gl.cullFace(WebGL2RenderingContext.FRONT);
    if (this.levelRule === "volume") {
      this.drawVolumeLevel(renderContext, scales, mode);
    } else {
      this.drawSliceLevels(renderContext, scales, mode);
    }
    if (!wasCulling) gl.disable(WebGL2RenderingContext.CULL_FACE);
    gl.cullFace(WebGL2RenderingContext.BACK);
    this.vertexIdHelper.disable();
  }

  private drawVolumeLevel(
    renderContext: SliceViewRenderContext,
    scales: FrontendTransformedSource[],
    mode: SlabMode,
  ) {
    let level: LevelDraw | undefined;
    forEachVisibleVolumeRenderingChunk(
      renderContext.projectionParameters,
      this.image.localPosition.value,
      this.depthSamplesTarget.value,
      scales as FrontendTransformedSource<any, VolumeChunkSource>[],
      (tsource, _scaleIndex, _physicalSpacing, optimalSamples) => {
        level = this.beginLevel(
          renderContext,
          tsource,
          mode,
          false,
          Math.max(1, Math.ceil(optimalSamples)),
        );
      },
      (tsource) => {
        if (level !== undefined) this.drawChunk(level, tsource);
      },
    );
    if (level !== undefined) this.endLevel(level);
  }

  private drawSliceLevels(
    renderContext: SliceViewRenderContext,
    scales: FrontendTransformedSource[],
    mode: SlabMode,
  ) {
    const { sliceView, projectionParameters } = renderContext;
    // Coarsest first; reversed so rank 0 is the level the zoom asks for.
    const levels = (
      Array.from(
        filterVisibleSources(
          sliceView,
          this as any,
          scales as TransformedSource[],
        ),
      ) as FrontendTransformedSource[]
    ).reverse();
    if (levels.length === 0) return;
    const chunks: LevelChunks[] = [];
    for (const tsource of levels) {
      const layout = getNormalizedChunkLayout(
        projectionParameters,
        tsource.chunkLayout,
      );
      const level: LevelChunks = {
        tsource,
        size: vec3.clone(layout.size),
        transform: mat4.clone(layout.transform),
        invTransform: mat4.clone(layout.invTransform),
        loaded: new Set(),
        cells: [],
        missing: 0,
      };
      const source = tsource.source as VolumeChunkSource;
      const isFinest = chunks.length === 0;
      forEachVisibleVolumetricChunk(
        projectionParameters,
        this.image.localPosition.value,
        tsource,
        (positionInChunks) => {
          const chunk = source.chunks.get(tsource.curPositionInChunks.join());
          if (isFinest) level.cells.push(vec3.clone(positionInChunks));
          if (chunk !== undefined && chunk.state === ChunkState.GPU_MEMORY) {
            level.loaded.add(cellKey(positionInChunks));
          } else {
            ++level.missing;
          }
        },
      );
      chunks.push(level);
      // Coarser levels own nothing once this one is complete.
      if (level.missing === 0) break;
    }
    const fallback = chunks.length > 1;
    const owner = fallback ? this.updateOwnerTexture(chunks) : undefined;

    chunks.forEach((level, rank) => {
      const { tsource } = level;
      const draw = this.beginLevel(
        renderContext,
        tsource,
        mode,
        fallback,
        this.samplesAcrossSlab(renderContext, tsource),
      );
      if (draw === undefined) return;
      if (owner !== undefined) {
        const { gl } = this.image;
        const { shader } = draw;
        const unit = shader.textureUnit(OWNER_TEXTURE)!;
        gl.activeTexture(WebGL2RenderingContext.TEXTURE0 + unit);
        gl.bindTexture(WebGL2RenderingContext.TEXTURE_3D, this.ownerTexture);
        const finest = chunks[0];
        const m = mat4.multiply(tempMat4, finest.invTransform, level.transform);
        mat4.fromScaling(tempScale, [
          1 / finest.size[0],
          1 / finest.size[1],
          1 / finest.size[2],
        ]);
        mat4.multiply(m, tempScale, m);
        mat4.fromTranslation(tempScale, [
          -owner.origin[0],
          -owner.origin[1],
          -owner.origin[2],
        ]);
        mat4.multiply(m, tempScale, m);
        gl.uniformMatrix4fv(shader.uniform("uLevelToOwnerCell"), false, m);
        gl.uniform3i(
          shader.uniform("uOwnerSize"),
          owner.size[0],
          owner.size[1],
          owner.size[2],
        );
        gl.uniform1ui(shader.uniform("uLevelRank"), rank);
      }
      forEachVisibleVolumetricChunk(
        projectionParameters,
        this.image.localPosition.value,
        tsource,
        () => this.drawChunk(draw, tsource),
      );
      this.endLevel(draw);
    });
  }

  // For each finest-level chunk cell in view: the finest level (rank) whose
  // chunk there is loaded, as a 3D texture over the cells' bounding box.
  private updateOwnerTexture(chunks: LevelChunks[]) {
    const finest = chunks[0];
    const origin = vec3.fromValues(Infinity, Infinity, Infinity);
    const upper = vec3.fromValues(-Infinity, -Infinity, -Infinity);
    for (const cell of finest.cells) {
      vec3.min(origin, origin, cell);
      vec3.max(upper, upper, cell);
    }
    const size = [0, 1, 2].map((i) => Math.max(0, upper[i] - origin[i] + 1));
    const data = new Uint8Array(size[0] * size[1] * size[2]).fill(NO_OWNER);
    for (const cell of finest.cells) {
      let owner = NO_OWNER;
      for (let i = 0; i < 3; ++i) {
        tempPoint[i] = (cell[i] + 0.5) * finest.size[i];
      }
      vec3.transformMat4(tempPoint, tempPoint, finest.transform);
      for (let rank = 0; rank < chunks.length; ++rank) {
        const level = chunks[rank];
        vec3.transformMat4(tempNear, tempPoint, level.invTransform);
        for (let i = 0; i < 3; ++i) {
          tempNear[i] = Math.floor(tempNear[i] / level.size[i]);
        }
        if (level.loaded.has(cellKey(tempNear))) {
          owner = rank;
          break;
        }
      }
      const x = cell[0] - origin[0];
      const y = cell[1] - origin[1];
      const z = cell[2] - origin[2];
      data[(z * size[1] + y) * size[0] + x] = owner;
    }
    const { gl } = this.image;
    if (this.ownerTexture === null) this.ownerTexture = gl.createTexture();
    gl.activeTexture(WebGL2RenderingContext.TEXTURE0);
    gl.bindTexture(WebGL2RenderingContext.TEXTURE_3D, this.ownerTexture);
    gl.pixelStorei(WebGL2RenderingContext.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(
      WebGL2RenderingContext.TEXTURE_3D,
      0,
      WebGL2RenderingContext.R8UI,
      size[0],
      size[1],
      size[2],
      0,
      WebGL2RenderingContext.RED_INTEGER,
      WebGL2RenderingContext.UNSIGNED_BYTE,
      data,
    );
    for (const parameter of [
      WebGL2RenderingContext.TEXTURE_MIN_FILTER,
      WebGL2RenderingContext.TEXTURE_MAG_FILTER,
    ]) {
      gl.texParameteri(
        WebGL2RenderingContext.TEXTURE_3D,
        parameter,
        WebGL2RenderingContext.NEAREST,
      );
    }
    gl.bindTexture(WebGL2RenderingContext.TEXTURE_3D, null);
    return { origin, size };
  }

  // One sample per voxel of this level across the slab.
  private samplesAcrossSlab(
    renderContext: SliceViewRenderContext,
    tsource: FrontendTransformedSource,
  ) {
    const { projectionParameters } = renderContext;
    const layout = getNormalizedChunkLayout(
      projectionParameters,
      tsource.chunkLayout,
    );
    mat4.multiply(
      tempMat4,
      projectionParameters.viewProjectionMat,
      layout.transform,
    );
    mat4.invert(tempInvMat4, tempMat4);
    vec3.transformMat4(tempNear, vec3.set(tempNear, 0, 0, -1), tempInvMat4);
    vec3.transformMat4(tempFar, vec3.set(tempFar, 0, 0, 1), tempInvMat4);
    return Math.max(1, Math.ceil(vec3.distance(tempNear, tempFar)));
  }

  private beginLevel(
    renderContext: SliceViewRenderContext,
    tsource: FrontendTransformedSource,
    mode: SlabMode,
    fallback: boolean,
    samples: number,
  ): LevelDraw | undefined {
    const { projectionParameters } = renderContext;
    const { gl } = this.image;
    const source = tsource.source as VolumeChunkSource;
    const { chunkFormat } = source;
    const shaderResult = this.shaderGetter({ chunkFormat, mode, fallback });
    const { shader } = shaderResult;
    if (shader === null) return undefined;
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
    gl.uniform1i(shader.uniform("uSlabSamples"), samples);
    const chunkLayout = getNormalizedChunkLayout(
      projectionParameters,
      tsource.chunkLayout,
    );
    const chunkSize = vec3.clone(chunkLayout.size);
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
    const { fixedPositionWithinChunk, chunkDisplayDimensionIndices } = tsource;
    for (const chunkDim of chunkDisplayDimensionIndices) {
      fixedPositionWithinChunk[chunkDim] = 0;
    }
    chunkFormat.beginDrawing(gl, shader);
    chunkFormat.beginSource(gl, shader);
    return {
      shader,
      source,
      chunkFormat,
      chunkSize,
      chunkDataSize: undefined,
      newSource: true,
    };
  }

  private drawChunk(level: LevelDraw, tsource: FrontendTransformedSource) {
    const { gl } = this.image;
    const { shader, source, chunkFormat, chunkSize } = level;
    const chunk = source.chunks.get(tsource.curPositionInChunks.join());
    if (chunk === undefined || chunk.state !== ChunkState.GPU_MEMORY) return;
    const {
      fixedPositionWithinChunk,
      chunkDisplayDimensionIndices,
      chunkTransform: { channelToChunkDimensionIndices },
    } = tsource;
    const chunkRank = source.spec.rank;
    if (chunk.chunkDataSize !== level.chunkDataSize) {
      level.chunkDataSize = chunk.chunkDataSize;
      for (let i = 0; i < 3; ++i) {
        const chunkDim = chunkDisplayDimensionIndices[i];
        tempPoint[i] =
          chunkDim === -1 || chunkDim >= chunkRank
            ? 1
            : chunk.chunkDataSize[chunkDim];
      }
      gl.uniform3fv(shader.uniform("uChunkDataSize"), tempPoint);
    }
    const { chunkGridPosition } = chunk;
    for (let i = 0; i < 3; ++i) {
      const chunkDim = chunkDisplayDimensionIndices[i];
      tempPoint[i] =
        chunkDim === -1 || chunkDim >= chunkRank
          ? 0
          : chunkSize[i] * chunkGridPosition[chunkDim];
    }
    gl.uniform3fv(shader.uniform("uTranslation"), tempPoint);
    chunkFormat.bindChunk(
      gl,
      shader,
      chunk,
      fixedPositionWithinChunk,
      chunkDisplayDimensionIndices,
      channelToChunkDimensionIndices,
      level.newSource,
    );
    level.newSource = false;
    drawBoxes(gl, 1, 1);
  }

  private endLevel(level: LevelDraw) {
    level.chunkFormat.endDrawing(this.image.gl, level.shader);
  }
}
