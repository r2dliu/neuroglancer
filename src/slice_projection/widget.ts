import type { VisibleLayerInfo } from "#src/layer/index.js";
import type { PerspectivePanel } from "#src/perspective_view/panel.js";
import type { PerspectiveViewRenderContext } from "#src/perspective_view/render_layer.js";
import { PerspectiveViewRenderLayer } from "#src/perspective_view/render_layer.js";
import type { ProjectionParameters } from "#src/projection_parameters.js";
import type { SliceParameters } from "#src/slice_projection/base.js";
import {
  computeSliceFrame,
  computeSliceToWorld,
  getSliceNormal,
  globalToIsotropic,
  isotropicToGlobal,
} from "#src/slice_projection/base.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import { constantWatchableValue } from "#src/trackable_value.js";
import type { ActionEvent } from "#src/util/event_action_map.js";
import {
  EventActionMap,
  registerActionListener,
} from "#src/util/event_action_map.js";
import { kAxes, mat3, mat4, quat, vec2, vec3, vec4 } from "#src/util/geom.js";
import { startRelativeMouseDrag } from "#src/util/mouse_drag.js";
import { GLBuffer } from "#src/webgl/buffer.js";
import type { GL } from "#src/webgl/context.js";
import type { ParameterizedEmitterDependentShaderGetter } from "#src/webgl/dynamic_shader.js";
import { parameterizedEmitterDependentShaderGetter } from "#src/webgl/dynamic_shader.js";
import type { ShaderProgram } from "#src/webgl/shader.js";

export interface SliceWidgetState {
  visible: boolean;
  anchor: vec2;
}

export function hiddenSliceWidgetState(): SliceWidgetState {
  return { visible: false, anchor: vec2.create() };
}

export interface SliceWidgetRenderLayerOptions {
  gl: GL;
  sliceParameters: WatchableValueInterface<SliceParameters>;
  widgetState: WatchableValueInterface<SliceWidgetState>;
  voxelSpacing: WatchableValueInterface<number>;
}

const SLICE_WIDGET_ACTION = "slice-widget";

const PART_CUBE = 0;
const PART_SHAFT = 1;
const PART_HEAD = 2;
const PARTS_PER_SIDE = 3;
const PICK_IDS_PER_WIDGET = 2 * PARTS_PER_SIDE;

const CUBE_HALF = 0.5;
const CUBE_Z0 = 0.02;
const CUBE_Z1 = 1.0;
const SHAFT_RADIUS = 0.13;
const SHAFT_Z1 = 2.3;
const HEAD_RADIUS = 0.34;
const HEAD_Z1 = 3.2;
const RADIAL_SEGMENTS = 16;

const VERTEX_STRIDE = 7;
const WIDGET_PIXEL_SIZE = 15;
const CLICK_THRESHOLD_PIXELS = 4;
const HIGHLIGHT_FACTOR = 1.9;

const CUBE_COLOR = vec3.fromValues(0.25, 0.45, 1.0);
const SHAFT_COLOR = vec3.fromValues(0.18, 0.85, 0.3);
const HEAD_COLOR = vec3.fromValues(0.25, 0.45, 1.0);
const WIREFRAME_COLOR = vec3.fromValues(1.0, 1.0, 1.0);

const tempMat4 = mat4.create();
const tempModel = mat4.create();
const tempMat3 = mat3.create();
const tempQuat = quat.create();
const tempCenter = vec3.create();
const tempOffset = vec3.create();
const tempAnchorGlobal = vec3.create();
const tempScaleVec = vec3.create();
const tempNormal = vec3.create();
const tempAnchor = vec3.create();
const tempPoint = vec3.create();
const tempDelta = vec3.create();
const tempRayOrigin = vec3.create();
const tempRayDirection = vec3.create();
const tempDirection = vec3.create();
const tempShifted = vec3.create();
const tempVec4 = vec4.create();
const tempUv = vec2.create();
const tempColors = new Float32Array(12);
const tempLight = new Float32Array(4);

function pushVertex(
  out: number[],
  part: number,
  px: number,
  py: number,
  pz: number,
  nx: number,
  ny: number,
  nz: number,
) {
  out.push(px, py, pz, nx, ny, nz, part);
}

function pushQuad(
  out: number[],
  part: number,
  corners: number[][],
  normal: number[],
) {
  const [a, b, c, d] = corners;
  for (const p of [a, b, c, a, c, d]) {
    pushVertex(out, part, p[0], p[1], p[2], normal[0], normal[1], normal[2]);
  }
}

function buildCube(out: number[]) {
  const h = CUBE_HALF;
  const z0 = CUBE_Z0;
  const z1 = CUBE_Z1;
  pushQuad(
    out,
    PART_CUBE,
    [
      [h, -h, z0],
      [h, h, z0],
      [h, h, z1],
      [h, -h, z1],
    ],
    [1, 0, 0],
  );
  pushQuad(
    out,
    PART_CUBE,
    [
      [-h, h, z0],
      [-h, -h, z0],
      [-h, -h, z1],
      [-h, h, z1],
    ],
    [-1, 0, 0],
  );
  pushQuad(
    out,
    PART_CUBE,
    [
      [h, h, z0],
      [-h, h, z0],
      [-h, h, z1],
      [h, h, z1],
    ],
    [0, 1, 0],
  );
  pushQuad(
    out,
    PART_CUBE,
    [
      [-h, -h, z0],
      [h, -h, z0],
      [h, -h, z1],
      [-h, -h, z1],
    ],
    [0, -1, 0],
  );
  pushQuad(
    out,
    PART_CUBE,
    [
      [-h, -h, z1],
      [h, -h, z1],
      [h, h, z1],
      [-h, h, z1],
    ],
    [0, 0, 1],
  );
  pushQuad(
    out,
    PART_CUBE,
    [
      [-h, h, z0],
      [h, h, z0],
      [h, -h, z0],
      [-h, -h, z0],
    ],
    [0, 0, -1],
  );
}

function buildShaft(out: number[]) {
  for (let i = 0; i < RADIAL_SEGMENTS; ++i) {
    const a0 = (2 * Math.PI * i) / RADIAL_SEGMENTS;
    const a1 = (2 * Math.PI * (i + 1)) / RADIAL_SEGMENTS;
    const c0 = Math.cos(a0);
    const s0 = Math.sin(a0);
    const c1 = Math.cos(a1);
    const s1 = Math.sin(a1);
    const x0 = c0 * SHAFT_RADIUS;
    const y0 = s0 * SHAFT_RADIUS;
    const x1 = c1 * SHAFT_RADIUS;
    const y1 = s1 * SHAFT_RADIUS;
    pushVertex(out, PART_SHAFT, x0, y0, CUBE_Z1, c0, s0, 0);
    pushVertex(out, PART_SHAFT, x1, y1, CUBE_Z1, c1, s1, 0);
    pushVertex(out, PART_SHAFT, x1, y1, SHAFT_Z1, c1, s1, 0);
    pushVertex(out, PART_SHAFT, x0, y0, CUBE_Z1, c0, s0, 0);
    pushVertex(out, PART_SHAFT, x1, y1, SHAFT_Z1, c1, s1, 0);
    pushVertex(out, PART_SHAFT, x0, y0, SHAFT_Z1, c0, s0, 0);
  }
}

function buildHead(out: number[]) {
  const slope = HEAD_Z1 - SHAFT_Z1;
  const norm = Math.hypot(slope, HEAD_RADIUS);
  for (let i = 0; i < RADIAL_SEGMENTS; ++i) {
    const a0 = (2 * Math.PI * i) / RADIAL_SEGMENTS;
    const a1 = (2 * Math.PI * (i + 1)) / RADIAL_SEGMENTS;
    const c0 = Math.cos(a0);
    const s0 = Math.sin(a0);
    const c1 = Math.cos(a1);
    const s1 = Math.sin(a1);
    const nz = HEAD_RADIUS / norm;
    const nr = slope / norm;
    pushVertex(
      out,
      PART_HEAD,
      c0 * HEAD_RADIUS,
      s0 * HEAD_RADIUS,
      SHAFT_Z1,
      c0 * nr,
      s0 * nr,
      nz,
    );
    pushVertex(
      out,
      PART_HEAD,
      c1 * HEAD_RADIUS,
      s1 * HEAD_RADIUS,
      SHAFT_Z1,
      c1 * nr,
      s1 * nr,
      nz,
    );
    pushVertex(out, PART_HEAD, 0, 0, HEAD_Z1, 0, 0, 1);
    pushVertex(out, PART_HEAD, 0, 0, SHAFT_Z1, 0, 0, -1);
    pushVertex(
      out,
      PART_HEAD,
      c1 * HEAD_RADIUS,
      s1 * HEAD_RADIUS,
      SHAFT_Z1,
      0,
      0,
      -1,
    );
    pushVertex(
      out,
      PART_HEAD,
      c0 * HEAD_RADIUS,
      s0 * HEAD_RADIUS,
      SHAFT_Z1,
      0,
      0,
      -1,
    );
  }
}

function makeWidgetMesh() {
  const out: number[] = [];
  buildCube(out);
  buildShaft(out);
  buildHead(out);
  return new Float32Array(out);
}

function makeBoxEdges() {
  const out: number[] = [];
  const corner = (i: number) => [
    i & 1 ? 1 : -1,
    i & 2 ? 1 : -1,
    i & 4 ? 1 : -1,
  ];
  for (let i = 0; i < 8; ++i) {
    for (const bit of [1, 2, 4]) {
      if (i & bit) continue;
      const a = corner(i);
      const b = corner(i | bit);
      pushVertex(out, 0, a[0], a[1], a[2], 0, 0, 1);
      pushVertex(out, 0, b[0], b[1], b[2], 0, 0, 1);
    }
  }
  return new Float32Array(out);
}

interface WidgetHit {
  part?: number;
  side?: number;
  uv?: vec2;
}

interface DragBasis {
  anchor: vec3;
  normal: vec3;
  orientation: quat;
  position: Float32Array;
  voxelRange: number;
  axisParameter: number;
  arrowLength: number;
  factors: Float64Array;
}

export class SliceWidgetRenderLayer extends PerspectiveViewRenderLayer<undefined> {
  gl: GL;
  sliceParameters: WatchableValueInterface<SliceParameters>;
  widgetState: WatchableValueInterface<SliceWidgetState>;
  voxelSpacing: WatchableValueInterface<number>;
  private meshBuffer: GLBuffer;
  private meshVertexCount: number;
  private boxBuffer: GLBuffer;
  private boxVertexCount: number;
  private activeDrag: { part: number; side: number } | undefined;
  private hoveredPart = -1;
  private clientX = 0;
  private clientY = 0;
  private shaderGetter: ParameterizedEmitterDependentShaderGetter<undefined>;

  constructor(options: SliceWidgetRenderLayerOptions) {
    super();
    const { gl } = options;
    this.gl = gl;
    this.sliceParameters = options.sliceParameters;
    this.widgetState = options.widgetState;
    this.voxelSpacing = options.voxelSpacing;
    const mesh = makeWidgetMesh();
    this.meshVertexCount = mesh.length / VERTEX_STRIDE;
    this.meshBuffer = this.registerDisposer(GLBuffer.fromData(gl, mesh));
    const box = makeBoxEdges();
    this.boxVertexCount = box.length / VERTEX_STRIDE;
    this.boxBuffer = this.registerDisposer(GLBuffer.fromData(gl, box));
    this.registerDisposer(
      this.sliceParameters.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      this.widgetState.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      this.voxelSpacing.changed.add(this.redrawNeeded.dispatch),
    );
    this.shaderGetter = parameterizedEmitterDependentShaderGetter(this, gl, {
      memoizeKey: "SliceWidget",
      parameters: constantWatchableValue(undefined),
      defineShader: (builder) => {
        builder.addAttribute("highp vec3", "aVertexPosition");
        builder.addAttribute("highp vec3", "aVertexNormal");
        builder.addAttribute("highp float", "aPartIndex");
        builder.addUniform("highp mat4", "uModelViewProjection");
        builder.addUniform("highp mat3", "uNormalMatrix");
        builder.addUniform("highp vec4", "uLightDirection");
        builder.addUniform("highp float", "uLightingEnabled");
        builder.addUniform("highp vec3", "uColors", PARTS_PER_SIDE);
        builder.addUniform("highp uint", "uPickId");
        builder.addVarying("highp vec4", "vColor");
        builder.addVarying("highp float", "vPickId", "flat");
        builder.setVertexMain(`
gl_Position = uModelViewProjection * vec4(aVertexPosition, 1.0);
int part = int(aPartIndex + 0.5);
float lighting = 1.0;
if (uLightingEnabled > 0.5) {
  vec3 normal = normalize(uNormalMatrix * aVertexNormal);
  lighting = abs(dot(normal, uLightDirection.xyz)) + uLightDirection.w;
}
vColor = vec4(clamp(uColors[part] * lighting, 0.0, 1.0), 1.0);
vPickId = float(uPickId) + aPartIndex;
`);
        builder.setFragmentMain(`
emit(vColor, uint(vPickId + 0.5));
`);
      },
    });
  }

  attach(attachment: VisibleLayerInfo<PerspectivePanel, undefined>) {
    super.attach(attachment);
    const panel = attachment.view;
    const trackMouse = (event: MouseEvent) => {
      this.clientX = event.clientX;
      this.clientY = event.clientY;
    };
    panel.element.addEventListener("mousemove", trackMouse);
    attachment.registerDisposer(() =>
      panel.element.removeEventListener("mousemove", trackMouse),
    );
    attachment.registerDisposer(
      panel.inputEventMap.addParent(
        EventActionMap.fromObject({
          "at:mousedown0": {
            action: SLICE_WIDGET_ACTION,
            stopPropagation: true,
            when: () => this.hitTest(panel) !== undefined,
          },
        }),
        Number.POSITIVE_INFINITY,
      ),
    );
    attachment.registerDisposer(
      registerActionListener<MouseEvent>(
        panel.element,
        SLICE_WIDGET_ACTION,
        (event) => this.onMouseDown(panel, event),
      ),
    );
    const { mouseState } = panel.viewer;
    attachment.registerDisposer(
      mouseState.changed.add(() => {
        const part =
          mouseState.active &&
          mouseState.pickedRenderLayer === this &&
          mouseState.pickedOffset < PICK_IDS_PER_WIDGET
            ? mouseState.pickedOffset
            : -1;
        if (part === this.hoveredPart) return;
        this.hoveredPart = part;
        this.redrawNeeded.dispatch();
      }),
    );
  }

  private getAnchorIsotropic(
    out: vec3,
    parameters: SliceParameters,
    state: SliceWidgetState,
    factors: Float64Array,
  ) {
    const { position, width, height, orientation } = parameters;
    vec3.set(
      out,
      (state.anchor[0] * width) / 2,
      (state.anchor[1] * height) / 2,
      0,
    );
    vec3.transformQuat(out, out, orientation as unknown as quat);
    for (let i = 0; i < 3; ++i) out[i] += position[i] * factors[i];
    return out;
  }

  private getWidgetScale(
    anchorIsotropic: vec3,
    factors: Float64Array,
    projectionParameters: ProjectionParameters,
  ) {
    const { width, viewProjectionMat, invViewProjectionMat } =
      projectionParameters;
    if (!(width > 0)) return 0;
    isotropicToGlobal(tempAnchorGlobal, anchorIsotropic, factors);
    vec4.set(
      tempVec4,
      tempAnchorGlobal[0],
      tempAnchorGlobal[1],
      tempAnchorGlobal[2],
      1,
    );
    vec4.transformMat4(tempVec4, tempVec4, viewProjectionMat);
    const w = tempVec4[3];
    if (!(w > 0)) return 0;
    tempVec4[0] += ((2 * WIDGET_PIXEL_SIZE) / width) * w;
    vec4.transformMat4(tempVec4, tempVec4, invViewProjectionMat);
    vec3.set(
      tempShifted,
      tempVec4[0] / tempVec4[3],
      tempVec4[1] / tempVec4[3],
      tempVec4[2] / tempVec4[3],
    );
    globalToIsotropic(tempShifted, tempShifted, factors);
    return vec3.distance(tempShifted, anchorIsotropic);
  }

  private computeModelMatrix(
    out: mat4,
    parameters: SliceParameters,
    state: SliceWidgetState,
    factors: Float64Array,
    scale: number,
    side: number,
  ) {
    const { width, height } = parameters;
    computeSliceFrame(out, parameters, factors);
    vec3.set(
      tempOffset,
      (state.anchor[0] * width) / 2,
      (state.anchor[1] * height) / 2,
      0,
    );
    mat4.translate(out, out, tempOffset);
    vec3.set(tempScaleVec, scale, scale, scale * side);
    return mat4.scale(out, out, tempScaleVec);
  }

  private getHighlightedPart() {
    const { activeDrag } = this;
    if (activeDrag !== undefined) {
      return activeDrag.part + (activeDrag.side > 0 ? 0 : PARTS_PER_SIDE);
    }
    return this.hoveredPart;
  }

  private bindGeometry(shader: ShaderProgram, buffer: GLBuffer) {
    const stride = VERTEX_STRIDE * 4;
    buffer.bindToVertexAttrib(
      shader.attribute("aVertexPosition"),
      3,
      WebGL2RenderingContext.FLOAT,
      false,
      stride,
      0,
    );
    buffer.bindToVertexAttrib(
      shader.attribute("aVertexNormal"),
      3,
      WebGL2RenderingContext.FLOAT,
      false,
      stride,
      12,
    );
    buffer.bindToVertexAttrib(
      shader.attribute("aPartIndex"),
      1,
      WebGL2RenderingContext.FLOAT,
      false,
      stride,
      24,
    );
  }

  private setColors(highlightedPart: number) {
    const base = [CUBE_COLOR, SHAFT_COLOR, HEAD_COLOR];
    for (let part = 0; part < PARTS_PER_SIDE; ++part) {
      const factor = part === highlightedPart ? HIGHLIGHT_FACTOR : 1;
      for (let i = 0; i < 3; ++i) {
        tempColors[part * 3 + i] = base[part][i] * factor;
      }
    }
    return tempColors.subarray(0, PARTS_PER_SIDE * 3);
  }

  draw(renderContext: PerspectiveViewRenderContext) {
    if (!renderContext.emitColor) return;
    const state = this.widgetState.value;
    if (!state.visible) return;
    const parameters = this.sliceParameters.value;
    const { projectionParameters } = renderContext;
    const { canonicalVoxelFactors } =
      projectionParameters.displayDimensionRenderInfo;
    const anchor = this.getAnchorIsotropic(
      tempAnchor,
      parameters,
      state,
      canonicalVoxelFactors,
    );
    const scale = this.getWidgetScale(
      anchor,
      canonicalVoxelFactors,
      projectionParameters,
    );
    if (scale <= 0) return;
    const shaderResult = this.shaderGetter(renderContext.emitter);
    const { shader } = shaderResult;
    if (shader === null) return;
    const { gl } = this;
    shader.bind();
    gl.disable(WebGL2RenderingContext.CULL_FACE);
    const { lightDirection, ambientLighting, directionalLighting } =
      renderContext;
    for (let i = 0; i < 3; ++i) {
      tempLight[i] = lightDirection[i] * directionalLighting;
    }
    tempLight[3] = ambientLighting;
    gl.uniform4fv(shader.uniform("uLightDirection"), tempLight);
    const pickId = renderContext.pickIDs.register(this, PICK_IDS_PER_WIDGET);
    const highlighted = this.getHighlightedPart();
    this.bindGeometry(shader, this.meshBuffer);
    gl.uniform1f(shader.uniform("uLightingEnabled"), 1);
    for (const side of [1, -1]) {
      const sideBase = side > 0 ? 0 : PARTS_PER_SIDE;
      this.computeModelMatrix(
        tempModel,
        parameters,
        state,
        canonicalVoxelFactors,
        scale,
        side,
      );
      mat4.multiply(
        tempMat4,
        projectionParameters.viewProjectionMat,
        tempModel,
      );
      mat3.normalFromMat4(tempMat3, tempModel);
      gl.uniformMatrix4fv(
        shader.uniform("uModelViewProjection"),
        false,
        tempMat4,
      );
      gl.uniformMatrix3fv(shader.uniform("uNormalMatrix"), false, tempMat3);
      gl.uniform1ui(shader.uniform("uPickId"), pickId + sideBase);
      gl.uniform3fv(
        shader.uniform("uColors"),
        this.setColors(highlighted - sideBase),
      );
      gl.drawArrays(WebGL2RenderingContext.TRIANGLES, 0, this.meshVertexCount);
    }
    if (this.activeDrag !== undefined || highlighted >= 0) {
      this.drawWireframeBox(
        shader,
        parameters,
        canonicalVoxelFactors,
        projectionParameters,
      );
    }
    for (const name of ["aVertexPosition", "aVertexNormal", "aPartIndex"]) {
      gl.disableVertexAttribArray(shader.attribute(name));
    }
  }

  private drawWireframeBox(
    shader: ShaderProgram,
    parameters: SliceParameters,
    factors: Float64Array,
    projectionParameters: ProjectionParameters,
  ) {
    const { gl } = this;
    const halfThickness = parameters.voxelRange * this.voxelSpacing.value;
    computeSliceToWorld(tempModel, parameters, factors, halfThickness);
    mat4.multiply(tempMat4, projectionParameters.viewProjectionMat, tempModel);
    gl.uniformMatrix4fv(
      shader.uniform("uModelViewProjection"),
      false,
      tempMat4,
    );
    gl.uniform1f(shader.uniform("uLightingEnabled"), 0);
    gl.uniform1ui(shader.uniform("uPickId"), 0);
    for (let i = 0; i < 3; ++i) tempColors[i] = WIREFRAME_COLOR[i];
    gl.uniform3fv(shader.uniform("uColors"), tempColors.subarray(0, 3));
    this.bindGeometry(shader, this.boxBuffer);
    gl.drawArrays(WebGL2RenderingContext.LINES, 0, this.boxVertexCount);
  }

  private computeRay(
    panel: PerspectivePanel,
    clientX: number,
    clientY: number,
    origin: vec3,
    direction: vec3,
  ) {
    const rect = panel.element.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return false;
    const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = 1 - ((clientY - rect.top) / rect.height) * 2;
    const projectionParameters = panel.projectionParameters.value;
    const { invViewProjectionMat } = projectionParameters;
    const { canonicalVoxelFactors } =
      projectionParameters.displayDimensionRenderInfo;
    vec4.set(tempVec4, ndcX, ndcY, -1, 1);
    vec4.transformMat4(tempVec4, tempVec4, invViewProjectionMat);
    vec3.set(origin, tempVec4[0], tempVec4[1], tempVec4[2]);
    vec3.scale(origin, origin, 1 / tempVec4[3]);
    globalToIsotropic(origin, origin, canonicalVoxelFactors);
    vec4.set(tempVec4, ndcX, ndcY, 1, 1);
    vec4.transformMat4(tempVec4, tempVec4, invViewProjectionMat);
    vec3.set(direction, tempVec4[0], tempVec4[1], tempVec4[2]);
    vec3.scale(direction, direction, 1 / tempVec4[3]);
    globalToIsotropic(direction, direction, canonicalVoxelFactors);
    vec3.subtract(direction, direction, origin);
    const length = vec3.length(direction);
    if (length === 0) return false;
    vec3.scale(direction, direction, 1 / length);
    return true;
  }

  private intersectSlice(panel: PerspectivePanel, out: vec2) {
    if (
      !this.computeRay(
        panel,
        this.clientX,
        this.clientY,
        tempRayOrigin,
        tempRayDirection,
      )
    ) {
      return undefined;
    }
    const parameters = this.sliceParameters.value;
    const { canonicalVoxelFactors } =
      panel.projectionParameters.value.displayDimensionRenderInfo;
    getSliceNormal(tempNormal, parameters);
    globalToIsotropic(
      tempCenter,
      parameters.position as unknown as vec3,
      canonicalVoxelFactors,
    );
    const denominator = vec3.dot(tempRayDirection, tempNormal);
    if (Math.abs(denominator) < 1e-9) return undefined;
    vec3.subtract(tempDelta, tempCenter, tempRayOrigin);
    const t = vec3.dot(tempDelta, tempNormal) / denominator;
    if (t <= 0) return undefined;
    vec3.scaleAndAdd(tempPoint, tempRayOrigin, tempRayDirection, t);
    vec3.subtract(tempPoint, tempPoint, tempCenter);
    quat.invert(tempQuat, parameters.orientation as unknown as quat);
    vec3.transformQuat(tempPoint, tempPoint, tempQuat);
    const u = tempPoint[0] / (parameters.width / 2);
    const v = tempPoint[1] / (parameters.height / 2);
    if (Math.abs(u) > 1 || Math.abs(v) > 1) return undefined;
    return vec2.set(out, u, v);
  }

  private hitTest(panel: PerspectivePanel): WidgetHit | undefined {
    const state = this.widgetState.value;
    const { mouseState } = panel.viewer;
    if (
      state.visible &&
      mouseState.active &&
      mouseState.pickedRenderLayer === this
    ) {
      const offset = mouseState.pickedOffset;
      if (offset >= 0 && offset < PICK_IDS_PER_WIDGET) {
        return {
          part: offset % PARTS_PER_SIDE,
          side: offset < PARTS_PER_SIDE ? 1 : -1,
        };
      }
    }
    const uv = this.intersectSlice(panel, tempUv);
    return uv === undefined ? undefined : { uv };
  }

  private onMouseDown(panel: PerspectivePanel, event: ActionEvent<MouseEvent>) {
    const hit = this.hitTest(panel);
    if (hit === undefined) return;
    if (hit.part !== undefined) {
      this.startPartDrag(panel, event, hit.part, hit.side!);
    } else {
      this.startSliceDrag(panel, event, vec2.clone(hit.uv!));
    }
  }

  private startSliceDrag(
    panel: PerspectivePanel,
    event: ActionEvent<MouseEvent>,
    uv: vec2,
  ) {
    let travelled = 0;
    startRelativeMouseDrag(
      event.detail,
      (_moveEvent, deltaX, deltaY) => {
        travelled += Math.abs(deltaX) + Math.abs(deltaY);
        panel.context.flagContinuousCameraMotion();
        panel.navigationState.pose.rotateRelative(
          kAxes[1],
          ((deltaX / 4.0) * Math.PI) / 180.0,
        );
        panel.navigationState.pose.rotateRelative(
          kAxes[0],
          ((-deltaY / 4.0) * Math.PI) / 180.0,
        );
      },
      () => {
        if (travelled > CLICK_THRESHOLD_PIXELS) return;
        const state = this.widgetState.value;
        this.widgetState.value = state.visible
          ? { visible: false, anchor: state.anchor }
          : { visible: true, anchor: uv };
      },
    );
  }

  private startPartDrag(
    panel: PerspectivePanel,
    event: ActionEvent<MouseEvent>,
    part: number,
    side: number,
  ) {
    const parameters = this.sliceParameters.value;
    const state = this.widgetState.value;
    const projectionParameters = panel.projectionParameters.value;
    const factors =
      projectionParameters.displayDimensionRenderInfo.canonicalVoxelFactors;
    const anchor = vec3.clone(
      this.getAnchorIsotropic(tempAnchor, parameters, state, factors),
    );
    const normal = vec3.clone(getSliceNormal(tempNormal, parameters));
    const scale = this.getWidgetScale(anchor, factors, projectionParameters);
    const basis: DragBasis = {
      anchor,
      normal,
      orientation: quat.clone(parameters.orientation as unknown as quat),
      position: Float32Array.from(parameters.position),
      voxelRange: parameters.voxelRange,
      axisParameter: 0,
      arrowLength: Math.max(scale, 1e-6) * HEAD_Z1,
      factors,
    };
    const startParameter = this.axisParameterAt(
      panel,
      event.detail.clientX,
      event.detail.clientY,
      anchor,
      normal,
    );
    if (startParameter === undefined && part !== PART_HEAD) return;
    basis.axisParameter = startParameter ?? 0;
    this.activeDrag = { part, side };
    this.redrawNeeded.dispatch();
    startRelativeMouseDrag(
      event.detail,
      (moveEvent) => this.updatePartDrag(panel, moveEvent, part, side, basis),
      () => {
        this.activeDrag = undefined;
        this.redrawNeeded.dispatch();
      },
    );
  }

  private axisParameterAt(
    panel: PerspectivePanel,
    clientX: number,
    clientY: number,
    axisPoint: vec3,
    axisDirection: vec3,
  ) {
    if (
      !this.computeRay(panel, clientX, clientY, tempRayOrigin, tempRayDirection)
    ) {
      return undefined;
    }
    vec3.subtract(tempDelta, axisPoint, tempRayOrigin);
    const b = vec3.dot(axisDirection, tempRayDirection);
    const d = vec3.dot(axisDirection, tempDelta);
    const e = vec3.dot(tempRayDirection, tempDelta);
    const denominator = 1 - b * b;
    if (Math.abs(denominator) < 1e-6) return undefined;
    return (b * e - d) / denominator;
  }

  private directionToRay(
    panel: PerspectivePanel,
    clientX: number,
    clientY: number,
    center: vec3,
    radius: number,
    out: vec3,
  ) {
    if (
      !this.computeRay(panel, clientX, clientY, tempRayOrigin, tempRayDirection)
    ) {
      return false;
    }
    vec3.subtract(tempDelta, center, tempRayOrigin);
    const b = vec3.dot(tempDelta, tempRayDirection);
    const discriminant =
      b * b - (vec3.dot(tempDelta, tempDelta) - radius * radius);
    const t = discriminant >= 0 ? b - Math.sqrt(discriminant) : b;
    vec3.scaleAndAdd(out, tempRayOrigin, tempRayDirection, t);
    vec3.subtract(out, out, center);
    const length = vec3.length(out);
    if (length < 1e-9) return false;
    vec3.scale(out, out, 1 / length);
    return true;
  }

  private setParameters(changes: Partial<SliceParameters>) {
    this.sliceParameters.value = {
      ...this.sliceParameters.value,
      ...changes,
    };
  }

  private updatePartDrag(
    panel: PerspectivePanel,
    event: MouseEvent,
    part: number,
    side: number,
    basis: DragBasis,
  ) {
    if (part === PART_HEAD) {
      if (
        !this.directionToRay(
          panel,
          event.clientX,
          event.clientY,
          basis.anchor,
          basis.arrowLength,
          tempDirection,
        )
      ) {
        return;
      }
      vec3.scale(tempPoint, basis.normal, side);
      quat.rotationTo(tempQuat, tempPoint, tempDirection);
      quat.multiply(tempQuat, tempQuat, basis.orientation);
      quat.normalize(tempQuat, tempQuat);
      this.setParameters({ orientation: Float32Array.from(tempQuat) });
      return;
    }
    const parameter = this.axisParameterAt(
      panel,
      event.clientX,
      event.clientY,
      basis.anchor,
      basis.normal,
    );
    if (parameter === undefined) return;
    const delta = parameter - basis.axisParameter;
    if (part === PART_SHAFT) {
      const position = Float32Array.from(basis.position);
      for (let i = 0; i < 3; ++i) {
        position[i] += (basis.normal[i] * delta) / basis.factors[i];
      }
      this.setParameters({ position });
      return;
    }
    const spacing = this.voxelSpacing.value;
    if (!(spacing > 0)) return;
    const voxelRange = Math.max(
      0,
      Math.round(basis.voxelRange + (side * delta) / spacing),
    );
    if (voxelRange !== this.sliceParameters.value.voxelRange) {
      this.setParameters({ voxelRange });
    }
  }
}
