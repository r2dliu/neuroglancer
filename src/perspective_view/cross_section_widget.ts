/**
 * @license
 * Copyright 2026 Google Inc.
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

import type { MouseSelectionState } from "#src/layer/index.js";
import type { PickIDManager } from "#src/object_picking.js";
import type { PerspectivePanel } from "#src/perspective_view/panel.js";
import type { PerspectiveViewRenderContext } from "#src/perspective_view/render_layer.js";
import { RenderLayer } from "#src/renderlayer.js";
import type { SliceView } from "#src/sliceview/frontend.js";
import type { Disposer } from "#src/util/disposable.js";
import { invokeDisposers, RefCounted } from "#src/util/disposable.js";
import { kAxes, mat4, vec3 } from "#src/util/geom.js";
import { startRelativeMouseDrag } from "#src/util/mouse_drag.js";
import { GLBuffer } from "#src/webgl/buffer.js";
import {
  ShaderBuilder,
  type ShaderModule,
  type ShaderProgram,
} from "#src/webgl/shader.js";

export type CrossSectionWidgetPart = "cube" | "shaft" | "head";
export type CrossSectionWidgetSide = -1 | 1;

export interface CrossSectionWidgetPick {
  part: CrossSectionWidgetPart;
  side: CrossSectionWidgetSide;
}

const PARTS = ["cube", "shaft", "head"] as const;
const SIDES = [-1, 1] as const;
const PICKS_PER_SIDE = PARTS.length;
const PICK_COUNT = SIDES.length * PICKS_PER_SIDE;

const CUBE_HALF_SIZE = 7;
const CUBE_LENGTH = 13;
const SHAFT_RADIUS = 2.25;
const SHAFT_START = CUBE_LENGTH;
const SHAFT_END = 42;
const HEAD_RADIUS = 8;
const HEAD_TIP = 58;
const MIN_DRAG_PIXELS_PER_VOXEL = 4;

const BLUE = [0.12, 0.48, 1, 1] as const;
const GREEN = [0.08, 0.78, 0.32, 1] as const;
const HIGHLIGHT = [0.72, 0.9, 1, 1] as const;
const WIREFRAME_BLUE = [0.2, 0.62, 1, 1] as const;

const WIREFRAME_EDGES = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 0],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 4],
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7],
] as const;

interface ScreenPoint {
  x: number;
  y: number;
}

interface ActiveCrossSectionWidget {
  sliceView: SliceView;
  /** Anchor in the cross-section's x/y coordinate frame, with z fixed at 0. */
  sectionPosition: vec3;
  /** Anchor in displayed global voxel coordinates. */
  position: vec3;
  /** Full-rank coordinate used as the fixed point for rotation. */
  anchor: Float32Array;
  hover: CrossSectionWidgetPick | undefined;
  drag: CrossSectionWidgetPick | undefined;
}

interface CrossSectionHit {
  sliceView: SliceView;
  sectionPosition: vec3;
  position: vec3;
  anchor: Float32Array;
}

interface SurfacePickData {
  kind: "surface";
  sliceView: SliceView;
}

interface WidgetPickData {
  kind: "widget";
  sliceView: SliceView;
}

type PickData = SurfacePickData | WidgetPickData;

interface LastPick {
  data: PickData;
  widget: CrossSectionWidgetPick | undefined;
}

interface MeshRange {
  first: number;
  count: number;
}

interface SolidMeshRanges {
  cube: MeshRange;
  shaft: MeshRange;
  head: MeshRange;
}

/**
 * Maps a side/component pair to one of the six contiguous pick offsets.
 * Negative-side parts occupy offsets 0..2 and positive-side parts 3..5.
 */
export function encodeCrossSectionWidgetPickOffset(
  part: CrossSectionWidgetPart,
  side: CrossSectionWidgetSide,
): number {
  return (side === -1 ? 0 : PICKS_PER_SIDE) + PARTS.indexOf(part);
}

/** Decodes a pick offset allocated by `encodeCrossSectionWidgetPickOffset`. */
export function decodeCrossSectionWidgetPickOffset(
  offset: number,
): CrossSectionWidgetPick | undefined {
  if (!Number.isInteger(offset) || offset < 0 || offset >= PICK_COUNT) {
    return undefined;
  }
  const side: CrossSectionWidgetSide = offset < PICKS_PER_SIDE ? -1 : 1;
  return { side, part: PARTS[offset % PICKS_PER_SIDE] };
}

/**
 * Converts a screen-space drag to a displacement along the positive section
 * normal. Passing side=-1 reverses the result, which makes dragging either
 * mirrored cube outward increase the range.
 */
export function computeNormalDragScalar(
  deltaX: number,
  deltaY: number,
  normalScreenX: number,
  normalScreenY: number,
  pixelsPerVoxel: number,
  side: CrossSectionWidgetSide = 1,
): number {
  const projectedPixels = deltaX * normalScreenX + deltaY * normalScreenY;
  return (
    (side * projectedPixels) /
    Math.max(MIN_DRAG_PIXELS_PER_VOXEL, pixelsPerVoxel)
  );
}

/**
 * Returns the eight global-coordinate corners of the selected section's
 * +/-voxelRange prism. Corners 0..3 are on -normal and 4..7 on +normal.
 */
export function computeCrossSectionWireframeCorners(
  width: number,
  height: number,
  invViewMatrix: ArrayLike<number>,
  normal: ArrayLike<number>,
  voxelRange: number,
): Float32Array {
  const result = new Float32Array(8 * 3);
  let output = 0;
  for (const side of SIDES) {
    for (const [xSign, ySign] of [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ] as const) {
      const x = (xSign * width) / 2;
      const y = (ySign * height) / 2;
      result[output++] =
        invViewMatrix[0] * x +
        invViewMatrix[4] * y +
        invViewMatrix[12] +
        normal[0] * side * voxelRange;
      result[output++] =
        invViewMatrix[1] * x +
        invViewMatrix[5] * y +
        invViewMatrix[13] +
        normal[1] * side * voxelRange;
      result[output++] =
        invViewMatrix[2] * x +
        invViewMatrix[6] * y +
        invViewMatrix[14] +
        normal[2] * side * voxelRange;
    }
  }
  return result;
}

function picksEqual(
  a: CrossSectionWidgetPick | undefined,
  b: CrossSectionWidgetPick | undefined,
) {
  return (
    a === b ||
    (a !== undefined &&
      b !== undefined &&
      a.part === b.part &&
      a.side === b.side)
  );
}

function appendTriangle(
  vertices: number[],
  a: readonly number[],
  b: readonly number[],
  c: readonly number[],
) {
  vertices.push(...a, ...b, ...c);
}

function appendQuad(
  vertices: number[],
  a: readonly number[],
  b: readonly number[],
  c: readonly number[],
  d: readonly number[],
) {
  appendTriangle(vertices, a, b, c);
  appendTriangle(vertices, a, c, d);
}

function appendCube(vertices: number[]) {
  const h = CUBE_HALF_SIZE;
  const z = CUBE_LENGTH;
  const nnn = [-h, -h, 0] as const;
  const pnn = [h, -h, 0] as const;
  const ppn = [h, h, 0] as const;
  const npn = [-h, h, 0] as const;
  const nnp = [-h, -h, z] as const;
  const pnp = [h, -h, z] as const;
  const ppp = [h, h, z] as const;
  const npp = [-h, h, z] as const;

  // The face lying exactly on the cross-section is intentionally omitted.
  // Apart from avoiding coplanar z-fighting, this lets depth testing select
  // the outward handle on the viewer-facing side of the section.
  appendQuad(vertices, nnp, npp, ppp, pnp);
  appendQuad(vertices, nnn, nnp, pnp, pnn);
  appendQuad(vertices, pnn, pnp, ppp, ppn);
  appendQuad(vertices, ppn, ppp, npp, npn);
  appendQuad(vertices, npn, npp, nnp, nnn);
}

function appendCylinder(
  vertices: number[],
  radius: number,
  start: number,
  end: number,
  segments = 12,
) {
  for (let i = 0; i < segments; ++i) {
    const angle0 = (i * 2 * Math.PI) / segments;
    const angle1 = ((i + 1) * 2 * Math.PI) / segments;
    const x0 = Math.cos(angle0) * radius;
    const y0 = Math.sin(angle0) * radius;
    const x1 = Math.cos(angle1) * radius;
    const y1 = Math.sin(angle1) * radius;
    appendQuad(
      vertices,
      [x0, y0, start],
      [x1, y1, start],
      [x1, y1, end],
      [x0, y0, end],
    );
    appendTriangle(vertices, [0, 0, end], [x0, y0, end], [x1, y1, end]);
  }
}

function appendCone(
  vertices: number[],
  radius: number,
  start: number,
  tip: number,
  segments = 16,
) {
  for (let i = 0; i < segments; ++i) {
    const angle0 = (i * 2 * Math.PI) / segments;
    const angle1 = ((i + 1) * 2 * Math.PI) / segments;
    const x0 = Math.cos(angle0) * radius;
    const y0 = Math.sin(angle0) * radius;
    const x1 = Math.cos(angle1) * radius;
    const y1 = Math.sin(angle1) * radius;
    appendTriangle(vertices, [x0, y0, start], [x1, y1, start], [0, 0, tip]);
    appendTriangle(vertices, [0, 0, start], [x1, y1, start], [x0, y0, start]);
  }
}

function makeSolidMeshes() {
  const vertices: number[] = [];
  const ranges = {} as SolidMeshRanges;
  const add = (
    part: CrossSectionWidgetPart,
    append: (vertices: number[]) => void,
  ) => {
    const first = vertices.length / 3;
    append(vertices);
    ranges[part] = { first, count: vertices.length / 3 - first };
  };
  add("cube", appendCube);
  add("shaft", (x) => appendCylinder(x, SHAFT_RADIUS, SHAFT_START, SHAFT_END));
  add("head", (x) => appendCone(x, HEAD_RADIUS, SHAFT_END, HEAD_TIP));
  return { vertices: new Float32Array(vertices), ranges };
}

class CrossSectionPickLayer extends RenderLayer {
  lastPick: LastPick | undefined;

  override updateMouseState(
    _mouseState: MouseSelectionState,
    _pickedValue: bigint,
    pickedOffset: number,
    data: PickData,
  ) {
    this.lastPick = {
      data,
      widget:
        data.kind === "widget"
          ? decodeCrossSectionWidgetPickOffset(pickedOffset)
          : undefined,
    };
  }
}

/**
 * A depth-tested, GPU-pickable manipulation widget for a selected 3-d
 * cross-section. It has no DOM overlay and is drawn into the perspective
 * panel's normal color/depth/pick targets.
 */
export class CrossSectionWidget extends RefCounted {
  private active: ActiveCrossSectionWidget | undefined;
  private pickLayer = this.registerDisposer(new CrossSectionPickLayer());
  private sliceViewDisposers = new Map<SliceView, Disposer[]>();

  private shader: ShaderProgram;
  private solidBuffer: GLBuffer;
  private wireframeBuffer: GLBuffer;
  private vertexArray: WebGLVertexArrayObject;
  private meshRanges: SolidMeshRanges;

  private modelMatrix = mat4.create();
  private modelViewProjection = mat4.create();
  private xBasis = vec3.create();
  private yBasis = vec3.create();
  private cameraXAxis = vec3.create();
  private cameraYAxis = vec3.create();
  private translatedPoint = vec3.create();

  constructor(
    private panel: PerspectivePanel,
    emitter: ShaderModule,
  ) {
    super();
    const { gl } = panel;
    const meshes = makeSolidMeshes();
    this.meshRanges = meshes.ranges;
    this.solidBuffer = this.registerDisposer(
      GLBuffer.fromData(gl, meshes.vertices, gl.ARRAY_BUFFER, gl.STATIC_DRAW),
    );
    this.wireframeBuffer = this.registerDisposer(
      GLBuffer.fromData(
        gl,
        new Float32Array(),
        gl.ARRAY_BUFFER,
        gl.DYNAMIC_DRAW,
      ),
    );

    const builder = new ShaderBuilder(gl);
    builder.addAttribute("highp vec3", "aPosition");
    builder.addUniform("highp mat4", "uModelViewProjection");
    builder.addUniform("highp vec4", "uColor");
    builder.addUniform("highp uint", "uPickId");
    builder.require(emitter);
    builder.setVertexMain(
      "gl_Position = uModelViewProjection * vec4(aPosition, 1.0);",
    );
    builder.setFragmentMain("emit(uColor, uPickId);");
    this.shader = this.registerDisposer(builder.build());

    const vertexArray = gl.createVertexArray();
    if (vertexArray === null) {
      throw new Error("Failed to create cross-section widget vertex array");
    }
    this.vertexArray = vertexArray;
    this.registerDisposer(() => gl.deleteVertexArray(vertexArray));

    this.registerDisposer(
      panel.viewer.mouseState.changed.add(() => this.updateHover()),
    );
    this.registerDisposer(
      panel.sliceViews.changed.add(() => {
        this.syncSliceViewListeners();
        this.ensureActiveVisible();
        panel.scheduleRedraw();
      }),
    );
    this.registerDisposer(
      panel.viewer.showSliceViews.changed.add(() => {
        this.ensureActiveVisible();
        panel.scheduleRedraw();
      }),
    );
    this.syncSliceViewListeners();
  }

  registerSliceViewPickId(
    pickIds: PickIDManager,
    sliceView: SliceView,
  ): number {
    return pickIds.register(this.pickLayer, 1, 0n, {
      kind: "surface",
      sliceView,
    } satisfies SurfacePickData);
  }

  draw(renderContext: PerspectiveViewRenderContext): void {
    const { active } = this;
    if (active === undefined) return;
    if (!this.isSliceViewVisible(active.sliceView)) {
      this.active = undefined;
      return;
    }

    this.refreshActivePosition(active);
    const scale = this.getWorldUnitsPerScreenPixel(active.position);
    if (!(scale > 0) || !Number.isFinite(scale)) return;

    const widgetPickId = renderContext.pickIDs.register(
      this.pickLayer,
      PICK_COUNT,
      0n,
      { kind: "widget", sliceView: active.sliceView } satisfies WidgetPickData,
    );
    const { gl } = this.panel;
    const wasDepthTest = gl.isEnabled(gl.DEPTH_TEST);
    const wasBlend = gl.isEnabled(gl.BLEND);
    const wasCullFace = gl.isEnabled(gl.CULL_FACE);
    const oldDepthFunc = gl.getParameter(gl.DEPTH_FUNC) as number;
    const oldDepthMask = gl.getParameter(gl.DEPTH_WRITEMASK) as boolean;
    const oldStencilFrontWriteMask = gl.getParameter(
      gl.STENCIL_WRITEMASK,
    ) as number;
    const oldStencilBackWriteMask = gl.getParameter(
      gl.STENCIL_BACK_WRITEMASK,
    ) as number;
    const oldLineWidth = gl.getParameter(gl.LINE_WIDTH) as number;
    const oldArrayBuffer = gl.getParameter(
      gl.ARRAY_BUFFER_BINDING,
    ) as WebGLBuffer | null;
    const oldVertexArray = gl.getParameter(
      gl.VERTEX_ARRAY_BINDING,
    ) as WebGLVertexArrayObject | null;
    const maxDrawBuffers = gl.getParameter(gl.MAX_DRAW_BUFFERS) as number;
    const oldDrawBuffers = new Array<number>(maxDrawBuffers);
    for (let i = 0; i < maxDrawBuffers; ++i) {
      oldDrawBuffers[i] = gl.getParameter(gl.DRAW_BUFFER0 + i) as number;
    }

    try {
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.depthMask(true);
      gl.disable(gl.BLEND);
      gl.disable(gl.CULL_FACE);
      gl.bindVertexArray(this.vertexArray);

      const shader = this.shader;
      shader.bind();
      const positionAttribute = shader.attribute("aPosition");
      this.solidBuffer.bindToVertexAttrib(positionAttribute, 3);

      this.getSectionBasis(active);
      const highlighted = active.drag ?? active.hover;
      for (const side of SIDES) {
        this.setWidgetModelMatrix(active.position, scale, side);
        mat4.multiply(
          this.modelViewProjection,
          renderContext.projectionParameters.viewProjectionMat,
          this.modelMatrix,
        );
        gl.uniformMatrix4fv(
          shader.uniform("uModelViewProjection"),
          false,
          this.modelViewProjection,
        );
        for (const part of PARTS) {
          const color =
            highlighted?.part === part && highlighted.side === side
              ? HIGHLIGHT
              : part === "shaft"
                ? GREEN
                : BLUE;
          gl.uniform4f(
            shader.uniform("uColor"),
            color[0],
            color[1],
            color[2],
            color[3],
          );
          gl.uniform1ui(
            shader.uniform("uPickId"),
            widgetPickId + encodeCrossSectionWidgetPickOffset(part, side),
          );
          const range = this.meshRanges[part];
          gl.drawArrays(gl.TRIANGLES, range.first, range.count);
        }
      }

      if (highlighted !== undefined) {
        // Keep the wireframe depth-tested, but do not let it alter the depth
        // stencil, or pick attachments underneath its thin lines.
        gl.depthMask(false);
        gl.stencilMask(0);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
        const lineVertices = this.makeWireframeVertices(active);
        this.wireframeBuffer.setData(lineVertices, gl.DYNAMIC_DRAW);
        this.wireframeBuffer.bindToVertexAttrib(positionAttribute, 3);
        gl.uniformMatrix4fv(
          shader.uniform("uModelViewProjection"),
          false,
          renderContext.projectionParameters.viewProjectionMat,
        );
        gl.uniform4f(
          shader.uniform("uColor"),
          WIREFRAME_BLUE[0],
          WIREFRAME_BLUE[1],
          WIREFRAME_BLUE[2],
          WIREFRAME_BLUE[3],
        );
        gl.uniform1ui(shader.uniform("uPickId"), 0);
        gl.lineWidth(1);
        gl.drawArrays(gl.LINES, 0, lineVertices.length / 3);
      }
    } finally {
      gl.drawBuffers(oldDrawBuffers);
      gl.depthMask(oldDepthMask);
      gl.stencilMaskSeparate(gl.FRONT, oldStencilFrontWriteMask);
      gl.stencilMaskSeparate(gl.BACK, oldStencilBackWriteMask);
      gl.depthFunc(oldDepthFunc);
      gl.lineWidth(oldLineWidth);
      if (!wasDepthTest) gl.disable(gl.DEPTH_TEST);
      if (wasBlend) gl.enable(gl.BLEND);
      if (wasCullFace) gl.enable(gl.CULL_FACE);
      gl.bindVertexArray(oldVertexArray);
      gl.bindBuffer(gl.ARRAY_BUFFER, oldArrayBuffer);
    }
  }

  /**
   * Handles the perspective panel's ordinary left-button action. Returns true
   * only when a cross-section surface or this widget consumed the action.
   */
  handleMouseDown(event: MouseEvent): boolean {
    if (event.button !== 0 || event.target !== this.panel.element) {
      return false;
    }

    const { mouseState } = this.panel.viewer;
    // Ensure the pick request uses this mousedown's exact coordinates even
    // when no preceding mousemove event was delivered.
    this.panel.handleMouseMove(event.clientX, event.clientY);
    const hasGpuPick = mouseState.updateUnconditionally();
    if (
      hasGpuPick &&
      mouseState.pickedRenderLayer === this.pickLayer &&
      this.pickLayer.lastPick !== undefined
    ) {
      const picked = this.pickLayer.lastPick;
      if (picked.data.kind === "widget" && picked.widget !== undefined) {
        const { active } = this;
        if (
          active !== undefined &&
          active.sliceView === picked.data.sliceView
        ) {
          this.startDrag(event, picked.widget);
          return true;
        }
      } else if (
        picked.data.kind === "surface" &&
        this.isSliceViewVisible(picked.data.sliceView)
      ) {
        const hit = this.makeGpuSurfaceHit(
          picked.data.sliceView,
          mouseState.unsnappedPosition,
        );
        if (hit !== undefined) {
          this.startSurfaceGesture(event, hit);
          return true;
        }
      }
      return false;
    }

    // A non-widget result represents foreground geometry and must win. If the
    // pick buffer has no hit, there is no visible cross-section at the pointer.
    return false;
  }

  /**
   * A section occupies much of a typical perspective panel. Preserve the
   * panel's normal orbit gesture there, while treating a press/release without
   * meaningful movement as the click that toggles this widget.
   */
  private startSurfaceGesture(initialEvent: MouseEvent, hit: CrossSectionHit) {
    let totalX = 0;
    let totalY = 0;
    let rotatingCamera = false;
    const rotateCamera = (deltaX: number, deltaY: number) => {
      this.panel.context.flagContinuousCameraMotion();
      this.panel.navigationState.pose.rotateRelative(
        kAxes[1],
        ((deltaX / 4) * Math.PI) / 180,
      );
      this.panel.navigationState.pose.rotateRelative(
        kAxes[0],
        ((-deltaY / 4) * Math.PI) / 180,
      );
    };
    const applyDragDelta = (deltaX: number, deltaY: number) => {
      if (deltaX === 0 && deltaY === 0) return;
      if (rotatingCamera) {
        rotateCamera(deltaX, deltaY);
        return;
      }
      totalX += deltaX;
      totalY += deltaY;
      if (Math.hypot(totalX, totalY) >= 3) {
        rotatingCamera = true;
        rotateCamera(totalX, totalY);
      }
    };
    startRelativeMouseDrag(
      initialEvent,
      (_event, deltaX, deltaY) => applyDragDelta(deltaX, deltaY),
      (event, deltaX, deltaY) => {
        if (event.type === "pointerup") {
          applyDragDelta(deltaX, deltaY);
        }
        if (event.type === "pointerup" && !rotatingCamera) {
          this.toggleAtHit(hit);
        }
      },
    );
  }

  private toggleAtHit(hit: CrossSectionHit) {
    if (this.active?.sliceView === hit.sliceView) {
      this.active = undefined;
    } else {
      this.active = {
        sliceView: hit.sliceView,
        sectionPosition: hit.sectionPosition,
        position: hit.position,
        anchor: hit.anchor,
        hover: undefined,
        drag: undefined,
      };
    }
    this.panel.scheduleRedraw();
  }

  private updateHover() {
    const { active } = this;
    if (active === undefined || active.drag !== undefined) return;
    const { mouseState } = this.panel.viewer;
    let hover: CrossSectionWidgetPick | undefined;
    if (
      mouseState.active &&
      mouseState.pickedRenderLayer === this.pickLayer &&
      this.pickLayer.lastPick?.data.kind === "widget" &&
      this.pickLayer.lastPick.data.sliceView === active.sliceView
    ) {
      hover = this.pickLayer.lastPick.widget;
    }
    if (!picksEqual(hover, active.hover)) {
      active.hover = hover;
      this.panel.scheduleRedraw();
    }
  }

  private startDrag(initialEvent: MouseEvent, picked: CrossSectionWidgetPick) {
    const { active } = this;
    if (active === undefined) return;
    active.drag = { ...picked };
    active.hover = { ...picked };
    const fixedPoint = Float32Array.from(active.anchor);
    this.panel.scheduleRedraw();

    const applyDragDelta = (deltaX: number, deltaY: number) => {
      if (deltaX === 0 && deltaY === 0) return false;
      if (
        this.active !== active ||
        !this.isSliceViewVisible(active.sliceView)
      ) {
        return false;
      }
      if (picked.part === "cube") {
        this.adjustVoxelRange(active, picked.side, deltaX, deltaY);
      } else if (picked.part === "shaft") {
        this.translateAlongNormal(active, deltaX, deltaY);
      } else {
        this.rotateCrossSection(
          active,
          fixedPoint,
          picked.side,
          deltaX,
          deltaY,
        );
      }
      return true;
    };
    startRelativeMouseDrag(
      initialEvent,
      (_event, deltaX, deltaY) => {
        if (applyDragDelta(deltaX, deltaY)) {
          this.panel.scheduleRedraw();
        }
      },
      (event, deltaX, deltaY) => {
        if (event.type === "pointerup") {
          applyDragDelta(deltaX, deltaY);
        }
        if (this.active !== active) return;
        active.drag = undefined;
        this.updateHover();
        this.panel.scheduleRedraw();
      },
    );
  }

  private adjustVoxelRange(
    active: ActiveCrossSectionWidget,
    side: CrossSectionWidgetSide,
    deltaX: number,
    deltaY: number,
  ) {
    const amount = this.getNormalDragAmount(active, deltaX, deltaY, side);
    active.sliceView.voxelRange.value = Math.max(
      0,
      active.sliceView.voxelRange.value + amount,
    );
  }

  private translateAlongNormal(
    active: ActiveCrossSectionWidget,
    deltaX: number,
    deltaY: number,
  ) {
    const amount = this.getNormalDragAmount(active, deltaX, deltaY, 1);
    if (amount === 0) return;
    const normal =
      active.sliceView.projectionParameters.value
        .viewportNormalInGlobalCoordinates;
    active.sliceView.navigationState.pose.updateDisplayPosition((position) => {
      vec3.scaleAndAdd(position, position, normal, amount);
    });
    this.refreshActivePosition(active);
  }

  private rotateCrossSection(
    active: ActiveCrossSectionWidget,
    fixedPoint: Float32Array,
    side: CrossSectionWidgetSide,
    deltaX: number,
    deltaY: number,
  ) {
    const cameraOrientation =
      this.panel.navigationState.pose.orientation.orientation;
    vec3.transformQuat(this.cameraXAxis, kAxes[0], cameraOrientation);
    vec3.transformQuat(this.cameraYAxis, kAxes[1], cameraOrientation);
    const degreesToRadians = Math.PI / 180;
    const pose = active.sliceView.navigationState.pose;
    pose.rotateAbsolute(
      this.cameraYAxis,
      ((-side * deltaX) / 4) * degreesToRadians,
      fixedPoint,
    );
    pose.rotateAbsolute(
      this.cameraXAxis,
      ((-side * deltaY) / 4) * degreesToRadians,
      fixedPoint,
    );
    this.refreshActivePosition(active);
  }

  private getNormalDragAmount(
    active: ActiveCrossSectionWidget,
    deltaX: number,
    deltaY: number,
    side: CrossSectionWidgetSide,
  ) {
    const center = this.projectPoint(active.position);
    if (center === undefined) {
      return computeNormalDragScalar(deltaX, deltaY, 0, -1, 0, side);
    }
    const normal =
      active.sliceView.projectionParameters.value
        .viewportNormalInGlobalCoordinates;
    const normalPoint = vec3.scaleAndAdd(
      this.translatedPoint,
      active.position,
      normal,
      1,
    );
    const projectedNormal = this.projectPoint(normalPoint);
    if (projectedNormal === undefined) {
      return computeNormalDragScalar(deltaX, deltaY, 0, -1, 0, side);
    }
    let x = projectedNormal.x - center.x;
    let y = projectedNormal.y - center.y;
    const pixelsPerVoxel = Math.hypot(x, y);
    if (pixelsPerVoxel < 1e-5) {
      return computeNormalDragScalar(deltaX, deltaY, 0, -1, 0, side);
    }
    x /= pixelsPerVoxel;
    y /= pixelsPerVoxel;
    return computeNormalDragScalar(deltaX, deltaY, x, y, pixelsPerVoxel, side);
  }

  private syncSliceViewListeners() {
    for (const [sliceView, disposers] of this.sliceViewDisposers) {
      if (this.panel.sliceViews.has(sliceView)) continue;
      invokeDisposers(disposers);
      this.sliceViewDisposers.delete(sliceView);
    }
    for (const sliceView of this.panel.sliceViews.keys()) {
      if (this.sliceViewDisposers.has(sliceView)) continue;
      const redraw = () => {
        if (this.active?.sliceView === sliceView) {
          this.refreshActivePosition(this.active);
        }
        this.panel.scheduleRedraw();
      };
      this.sliceViewDisposers.set(sliceView, [
        sliceView.voxelRange.changed.add(redraw),
        sliceView.projectionParameters.changed.add(redraw),
      ]);
    }
  }

  private ensureActiveVisible() {
    const { active } = this;
    if (active !== undefined && !this.isSliceViewVisible(active.sliceView)) {
      this.active = undefined;
    }
  }

  private isSliceViewVisible(sliceView: SliceView) {
    if (!this.panel.sliceViews.has(sliceView) || !sliceView.valid) return false;
    return (
      this.panel.sliceViews.get(sliceView) === true ||
      this.panel.viewer.showSliceViews.value
    );
  }

  private refreshActivePosition(active: ActiveCrossSectionWidget) {
    vec3.transformMat4(
      active.position,
      active.sectionPosition,
      active.sliceView.projectionParameters.value.invViewMatrix,
    );
    const navigationPosition = active.sliceView.navigationState.position.value;
    if (active.anchor.length !== navigationPosition.length) {
      active.anchor = Float32Array.from(navigationPosition);
    } else {
      active.anchor.set(navigationPosition);
    }
    const { displayDimensionIndices, displayRank } =
      active.sliceView.navigationState.pose.displayDimensions.value;
    for (let i = 0; i < displayRank; ++i) {
      active.anchor[displayDimensionIndices[i]] = active.position[i];
    }
  }

  private makeGpuSurfaceHit(
    sliceView: SliceView,
    unsnappedPosition: Float32Array,
  ): CrossSectionHit | undefined {
    if (unsnappedPosition.length === 0) return undefined;
    const anchor = Float32Array.from(unsnappedPosition);
    const position = vec3.clone(
      sliceView.projectionParameters.value.centerDataPosition,
    );
    const { displayDimensionIndices, displayRank } =
      sliceView.navigationState.pose.displayDimensions.value;
    for (let i = 0; i < displayRank; ++i) {
      const dimension = displayDimensionIndices[i];
      position[i] = anchor[dimension];
    }
    const sectionPosition = vec3.transformMat4(
      vec3.create(),
      position,
      sliceView.projectionParameters.value.viewMatrix,
    );
    sectionPosition[2] = 0;
    return { sliceView, sectionPosition, position, anchor };
  }

  private projectPoint(point: vec3): ScreenPoint | undefined {
    const matrix = this.panel.projectionParameters.value.viewProjectionMat;
    const x =
      matrix[0] * point[0] +
      matrix[4] * point[1] +
      matrix[8] * point[2] +
      matrix[12];
    const y =
      matrix[1] * point[0] +
      matrix[5] * point[1] +
      matrix[9] * point[2] +
      matrix[13];
    const w =
      matrix[3] * point[0] +
      matrix[7] * point[1] +
      matrix[11] * point[2] +
      matrix[15];
    if (!Number.isFinite(w) || w <= 1e-6) return undefined;
    const ndcX = x / w;
    const ndcY = y / w;
    if (!Number.isFinite(ndcX) || !Number.isFinite(ndcY)) return undefined;
    const {
      visibleLeftFraction,
      visibleTopFraction,
      visibleWidthFraction,
      visibleHeightFraction,
    } = this.panel.renderViewport;
    return {
      x:
        (visibleLeftFraction +
          (ndcX * 0.5 + 0.5) * (visibleWidthFraction || 1)) *
        this.panel.element.clientWidth,
      y:
        (visibleTopFraction +
          (0.5 - ndcY * 0.5) * (visibleHeightFraction || 1)) *
        this.panel.element.clientHeight,
    };
  }

  private getWorldUnitsPerScreenPixel(position: vec3) {
    this.panel.translateDataPointByViewportPixels(
      this.translatedPoint,
      position,
      1,
      0,
    );
    const xScale = vec3.distance(this.translatedPoint, position);
    this.panel.translateDataPointByViewportPixels(
      this.translatedPoint,
      position,
      0,
      1,
    );
    const yScale = vec3.distance(this.translatedPoint, position);
    return (xScale + yScale) / 2;
  }

  private getSectionBasis(active: ActiveCrossSectionWidget) {
    const { invViewMatrix, viewportNormalInGlobalCoordinates: normal } =
      active.sliceView.projectionParameters.value;
    vec3.set(this.xBasis, invViewMatrix[0], invViewMatrix[1], invViewMatrix[2]);
    vec3.normalize(this.xBasis, this.xBasis);
    vec3.cross(this.yBasis, normal, this.xBasis);
    vec3.normalize(this.yBasis, this.yBasis);
    // Retain the handedness of the slice view's own y axis.
    if (
      this.yBasis[0] * invViewMatrix[4] +
        this.yBasis[1] * invViewMatrix[5] +
        this.yBasis[2] * invViewMatrix[6] <
      0
    ) {
      vec3.scale(this.yBasis, this.yBasis, -1);
    }
  }

  private setWidgetModelMatrix(
    position: vec3,
    scale: number,
    side: CrossSectionWidgetSide,
  ) {
    const normal =
      this.active!.sliceView.projectionParameters.value
        .viewportNormalInGlobalCoordinates;
    const matrix = this.modelMatrix;
    mat4.identity(matrix);
    for (let i = 0; i < 3; ++i) {
      matrix[i] = this.xBasis[i] * scale;
      matrix[4 + i] = this.yBasis[i] * scale;
      matrix[8 + i] = normal[i] * scale * side;
      matrix[12 + i] = position[i];
    }
  }

  private makeWireframeVertices(active: ActiveCrossSectionWidget) {
    const parameters = active.sliceView.projectionParameters.value;
    const corners = computeCrossSectionWireframeCorners(
      parameters.width,
      parameters.height,
      parameters.invViewMatrix,
      parameters.viewportNormalInGlobalCoordinates,
      active.sliceView.voxelRange.value,
    );
    const edgeCount =
      Math.abs(active.sliceView.voxelRange.value) < 1e-6
        ? 4
        : WIREFRAME_EDGES.length;
    const vertices = new Float32Array(edgeCount * 2 * 3);
    let output = 0;
    for (let i = 0; i < edgeCount; ++i) {
      const edge = WIREFRAME_EDGES[i];
      for (const corner of edge) {
        const input = corner * 3;
        vertices[output++] = corners[input];
        vertices[output++] = corners[input + 1];
        vertices[output++] = corners[input + 2];
      }
    }
    return vertices;
  }

  override disposed() {
    for (const disposers of this.sliceViewDisposers.values()) {
      invokeDisposers(disposers);
    }
    this.sliceViewDisposers.clear();
    super.disposed();
  }
}
