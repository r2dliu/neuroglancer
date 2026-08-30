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

import type { ProjectionParameters } from "#src/projection_parameters.js";
import type { SliceView } from "#src/sliceview/frontend.js";
import { RefCounted } from "#src/util/disposable.js";
import { mat4, quat, vec3 } from "#src/util/geom.js";
import { GLBuffer } from "#src/webgl/buffer.js";
import type { GL } from "#src/webgl/context.js";
import type { ShaderProgram } from "#src/webgl/shader.js";
import { trivialUniformColorShader } from "#src/webgl/trivial_shaders.js";

const BLUE = new Float32Array([0, 0.35, 1, 1]);
const WHITE = new Float32Array([1, 1, 1, 1]);
const HOVER_REGION_BACKGROUND = new Float32Array([1, 1, 1, 0.2]);
const BOX_EDGE_COUNT = 12;
const VERTEX_COMPONENTS = 4;
const ENDPOINTS_PER_LINE_SEGMENT = 2;
const EDGE_ARROW_LINE_SEGMENTS = 3;
const MAX_RESIZE_ICON_LINE_SEGMENTS = 2 * EDGE_ARROW_LINE_SEGMENTS;
const ROTATE_ICON_ARROW_COUNT = 4;
const ROTATE_ICON_ARC_SEGMENTS = 5;
const MAX_ROTATE_ICON_LINE_SEGMENTS =
  ROTATE_ICON_ARROW_COUNT * (ROTATE_ICON_ARC_SEGMENTS + 2);
const MAX_EDGE_VERTEX_DATA_LENGTH =
  BOX_EDGE_COUNT * ENDPOINTS_PER_LINE_SEGMENT * VERTEX_COMPONENTS;
const MAX_ICON_VERTEX_DATA_LENGTH =
  Math.max(MAX_RESIZE_ICON_LINE_SEGMENTS, MAX_ROTATE_ICON_LINE_SEGMENTS) *
  ENDPOINTS_PER_LINE_SEGMENT *
  VERTEX_COMPONENTS;
const MAX_LINE_VERTEX_DATA_LENGTH = Math.max(
  MAX_EDGE_VERTEX_DATA_LENGTH,
  MAX_ICON_VERTEX_DATA_LENGTH,
);
const HOVER_REGION_CENTER_FRACTION = 2 / 3;
const RESIZE_ARROW_START_FRACTION = 0.48;
const RESIZE_ARROW_END_FRACTION = 0.88;
const RESIZE_ARROW_HEAD_LENGTH_FRACTION = 0.07;
const RESIZE_ARROW_HEAD_WIDTH_FRACTION = 0.04;
const ROTATE_ICON_RADIUS_FRACTION = 0.23;
const ROTATE_ICON_ARC_GAP_RADIANS = 0.34;
const ROTATE_ICON_HEAD_LENGTH_FRACTION = 0.42;
const ROTATE_ICON_HEAD_WIDTH_FRACTION = 0.24;

const tempVec = vec3.create();
const tempVec2 = vec3.create();
const tempVec3 = vec3.create();
const tempQuat = quat.create();
const tempMat = mat4.create();

type BoxAxes = [vec3, vec3, vec3];
type LocalPoint = [number, number, number];
type HoverRegionIndex = 0 | 1 | 2;

interface EdgeMove {
  dimension: 0 | 1 | 2;
  sign: -1 | 1;
  edgeIndex: number;
}

interface BoxHover {
  faceAxis: 0 | 1 | 2;
  faceSign: -1 | 1;
  uRegion: HoverRegionIndex;
  vRegion: HoverRegionIndex;
  highlightedEdges: number[];
  resizeEdges: EdgeMove[];
  rotate: boolean;
}

interface BoxDrag {
  mode: "resize" | "rotate";
  planePoint: vec3;
  planeNormal: vec3;
  previousPoint: vec3;
  previousVector: vec3;
  resizeEdges: EdgeMove[];
}

function axisFromNumber(axis: number): 0 | 1 | 2 {
  return axis as 0 | 1 | 2;
}

function otherAxes(axis: 0 | 1 | 2): [0 | 1 | 2, 0 | 1 | 2] {
  switch (axis) {
    case 0:
      return [1, 2];
    case 1:
      return [0, 2];
    case 2:
      return [0, 1];
  }
}

function edgeIndex(
  varyingAxis: 0 | 1 | 2,
  fixedAxis0: 0 | 1 | 2,
  fixedSign0: -1 | 1,
  fixedAxis1: 0 | 1 | 2,
  fixedSign1: -1 | 1,
) {
  const fixedAxes = otherAxes(varyingAxis);
  let sign0: -1 | 1;
  let sign1: -1 | 1;
  if (fixedAxes[0] === fixedAxis0 && fixedAxes[1] === fixedAxis1) {
    sign0 = fixedSign0;
    sign1 = fixedSign1;
  } else {
    sign0 = fixedSign1;
    sign1 = fixedSign0;
  }
  return varyingAxis * 4 + (sign0 > 0 ? 1 : 0) + (sign1 > 0 ? 2 : 0);
}

function faceEdgeMove(
  faceAxis: 0 | 1 | 2,
  faceSign: -1 | 1,
  dimension: 0 | 1 | 2,
  sign: -1 | 1,
): EdgeMove {
  const faceAxes = otherAxes(faceAxis);
  const varyingAxis = faceAxes[0] === dimension ? faceAxes[1] : faceAxes[0];
  return {
    dimension,
    sign,
    edgeIndex: edgeIndex(varyingAxis, faceAxis, faceSign, dimension, sign),
  };
}

function clampRegionIndex(value: number, halfSize: number): HoverRegionIndex {
  if (halfSize <= 0) return 1;
  const normalized = (value / halfSize + 1) * 0.5;
  return Math.max(
    0,
    Math.min(2, Math.floor(normalized * 3)),
  ) as HoverRegionIndex;
}

function normalizeAxis(axis: vec3) {
  const length = vec3.length(axis);
  if (length === 0) return false;
  vec3.scale(axis, axis, 1 / length);
  return true;
}

function getMouseClipCoordinates(
  out: vec3,
  event: MouseEvent,
  element: HTMLElement,
  projectionParameters: ProjectionParameters,
  z: number,
) {
  const bounds = element.getBoundingClientRect();
  const mouseX = event.clientX - (bounds.left + element.clientLeft);
  const mouseY = event.clientY - (bounds.top + element.clientTop);
  const {
    width,
    height,
    logicalWidth,
    logicalHeight,
    visibleLeftFraction,
    visibleTopFraction,
  } = projectionParameters;
  const glWindowX = mouseX - visibleLeftFraction * logicalWidth;
  const glWindowY = height - (mouseY - visibleTopFraction * logicalHeight);
  out[0] = (2 * glWindowX) / width - 1;
  out[1] = (2 * glWindowY) / height - 1;
  out[2] = z;
}

function transformLocalToWorld(
  out: vec3,
  local: ArrayLike<number>,
  center: vec3,
  axes: BoxAxes,
) {
  vec3.copy(out, center);
  for (let axis = 0; axis < 3; ++axis) {
    vec3.scaleAndAdd(out, out, axes[axis], local[axis]);
  }
}

function copyLocalPoint(point: LocalPoint): LocalPoint {
  return [point[0], point[1], point[2]];
}

function getRayFromMouse(
  origin: vec3,
  direction: vec3,
  event: MouseEvent,
  element: HTMLElement,
  projectionParameters: ProjectionParameters,
) {
  const { width, height } = projectionParameters;
  if (width === 0 || height === 0) return false;
  getMouseClipCoordinates(origin, event, element, projectionParameters, -1);
  getMouseClipCoordinates(direction, event, element, projectionParameters, 1);
  vec3.transformMat4(origin, origin, projectionParameters.invViewProjectionMat);
  vec3.transformMat4(
    direction,
    direction,
    projectionParameters.invViewProjectionMat,
  );
  vec3.subtract(direction, direction, origin);
  return normalizeAxis(direction);
}

function intersectRayPlane(
  out: vec3,
  origin: vec3,
  direction: vec3,
  planePoint: vec3,
  planeNormal: vec3,
) {
  const denominator = vec3.dot(direction, planeNormal);
  if (Math.abs(denominator) < 1e-7) return false;
  vec3.subtract(tempVec, planePoint, origin);
  const t = vec3.dot(tempVec, planeNormal) / denominator;
  if (t < 0) return false;
  vec3.scaleAndAdd(out, origin, direction, t);
  return true;
}

export class PerspectiveViewBoxOverlay extends RefCounted {
  private center = vec3.create();
  private axes: BoxAxes = [
    vec3.fromValues(1, 0, 0),
    vec3.fromValues(0, 1, 0),
    vec3.fromValues(0, 0, 1),
  ];
  private halfSize = vec3.fromValues(1, 1, 1);
  private initialized = false;
  private hover: BoxHover | undefined;
  private drag: BoxDrag | undefined;
  private vertexBuffer: GLBuffer;
  private lineVertexData = new Float32Array(MAX_LINE_VERTEX_DATA_LENGTH);
  private shader: ShaderProgram;

  constructor(public gl: GL) {
    super();
    this.shader = trivialUniformColorShader(gl);
    this.vertexBuffer = this.registerDisposer(
      new GLBuffer(gl, WebGL2RenderingContext.ARRAY_BUFFER),
    );
  }

  clearInteraction() {
    this.hover = undefined;
    this.drag = undefined;
  }

  clearHover() {
    if (this.hover === undefined) return false;
    this.hover = undefined;
    return true;
  }

  ensureInitialized(
    sliceViews: Iterable<[SliceView, boolean]>,
    showSliceViews: boolean,
    projectionParameters: ProjectionParameters,
  ) {
    if (this.initialized) return;
    const points: vec3[] = [];
    let basisSource: SliceView | undefined;
    for (const [sliceView, unconditional] of sliceViews) {
      if (!unconditional && !showSliceViews) continue;
      if (!sliceView.valid) continue;
      const { width, height, invViewMatrix } =
        sliceView.projectionParameters.value;
      if (width === 0 || height === 0) continue;
      if (basisSource === undefined) {
        basisSource = sliceView;
      }
      for (const xSign of [-1, 1] as const) {
        for (const ySign of [-1, 1] as const) {
          const point = vec3.fromValues(
            (xSign * width) / 2,
            (ySign * height) / 2,
            0,
          );
          vec3.transformMat4(point, point, invViewMatrix);
          points.push(point);
        }
      }
    }

    if (basisSource !== undefined) {
      const { invViewMatrix } = basisSource.projectionParameters.value;
      const axis0 = this.axes[0];
      const axis1 = this.axes[1];
      const axis2 = this.axes[2];
      vec3.set(axis0, invViewMatrix[0], invViewMatrix[1], invViewMatrix[2]);
      vec3.set(axis1, invViewMatrix[4], invViewMatrix[5], invViewMatrix[6]);
      if (!normalizeAxis(axis0) || !normalizeAxis(axis1)) {
        this.setFallback(projectionParameters);
        return;
      }
      vec3.cross(axis2, axis0, axis1);
      if (!normalizeAxis(axis2)) {
        this.setFallback(projectionParameters);
        return;
      }
      vec3.cross(axis1, axis2, axis0);
      normalizeAxis(axis1);
    } else {
      this.setFallback(projectionParameters);
      return;
    }

    const lower = [
      Number.POSITIVE_INFINITY,
      Number.POSITIVE_INFINITY,
      Number.POSITIVE_INFINITY,
    ];
    const upper = [
      Number.NEGATIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
    ];
    for (const point of points) {
      for (let axis = 0; axis < 3; ++axis) {
        const value = vec3.dot(point, this.axes[axis]);
        lower[axis] = Math.min(lower[axis], value);
        upper[axis] = Math.max(upper[axis], value);
      }
    }

    vec3.set(this.center, 0, 0, 0);
    for (let axis = 0; axis < 3; ++axis) {
      const midpoint = (lower[axis] + upper[axis]) / 2;
      this.halfSize[axis] = Math.max((upper[axis] - lower[axis]) / 2, 0);
      vec3.scaleAndAdd(this.center, this.center, this.axes[axis], midpoint);
    }
    const largestHalfSize = Math.max(
      this.halfSize[0],
      this.halfSize[1],
      this.halfSize[2],
      1,
    );
    const minimumHalfSize = largestHalfSize * 1e-3;
    for (let axis = 0; axis < 3; ++axis) {
      this.halfSize[axis] = Math.max(this.halfSize[axis], minimumHalfSize);
    }
    this.initialized = true;
  }

  private setFallback(projectionParameters: ProjectionParameters) {
    const { globalPosition } = projectionParameters;
    vec3.set(
      this.center,
      globalPosition[0] || 0,
      globalPosition[1] || 0,
      globalPosition[2] || 0,
    );
    const fallbackHalfSize = Math.max(
      1,
      projectionParameters.logicalHeight *
        projectionParameters.displayDimensionRenderInfo
          .canonicalVoxelFactors[0] *
        0.25,
    );
    vec3.set(
      this.halfSize,
      fallbackHalfSize,
      fallbackHalfSize,
      fallbackHalfSize,
    );
    vec3.set(this.axes[0], 1, 0, 0);
    vec3.set(this.axes[1], 0, 1, 0);
    vec3.set(this.axes[2], 0, 0, 1);
    this.initialized = true;
  }

  private getMouseRay(
    origin: vec3,
    direction: vec3,
    event: MouseEvent,
    element: HTMLElement,
    projectionParameters: ProjectionParameters,
  ) {
    return getRayFromMouse(
      origin,
      direction,
      event,
      element,
      projectionParameters,
    );
  }

  private hitTest(
    event: MouseEvent,
    element: HTMLElement,
    projectionParameters: ProjectionParameters,
  ): BoxHover | undefined {
    if (!this.initialized) return undefined;
    const origin = tempVec2;
    const direction = tempVec3;
    if (
      !this.getMouseRay(origin, direction, event, element, projectionParameters)
    ) {
      return undefined;
    }

    vec3.subtract(tempVec, origin, this.center);
    const localOrigin = [
      vec3.dot(tempVec, this.axes[0]),
      vec3.dot(tempVec, this.axes[1]),
      vec3.dot(tempVec, this.axes[2]),
    ];
    const localDirection = [
      vec3.dot(direction, this.axes[0]),
      vec3.dot(direction, this.axes[1]),
      vec3.dot(direction, this.axes[2]),
    ];

    let tNear = Number.NEGATIVE_INFINITY;
    let tFar = Number.POSITIVE_INFINITY;
    let faceAxis: 0 | 1 | 2 = 0;
    let faceSign: -1 | 1 = 1;
    let farFaceAxis: 0 | 1 | 2 = 0;
    let farFaceSign: -1 | 1 = 1;
    for (let axis = 0; axis < 3; ++axis) {
      const half = this.halfSize[axis];
      const originValue = localOrigin[axis];
      const directionValue = localDirection[axis];
      if (Math.abs(directionValue) < 1e-7) {
        if (originValue < -half || originValue > half) return undefined;
        continue;
      }
      let t0 = (-half - originValue) / directionValue;
      let t1 = (half - originValue) / directionValue;
      let sign: -1 | 1 = -1;
      if (t0 > t1) {
        const tmp = t0;
        t0 = t1;
        t1 = tmp;
        sign = 1;
      }
      if (t0 > tNear) {
        tNear = t0;
        faceAxis = axisFromNumber(axis);
        faceSign = sign;
      }
      if (t1 < tFar) {
        tFar = t1;
        farFaceAxis = axisFromNumber(axis);
        farFaceSign = sign === -1 ? 1 : -1;
      }
      if (tNear > tFar) return undefined;
    }
    if (tFar < 0) return undefined;
    const t = tNear >= 0 ? tNear : tFar;
    if (tNear < 0) {
      faceAxis = farFaceAxis;
      faceSign = farFaceSign;
    }
    const localHit = [
      localOrigin[0] + localDirection[0] * t,
      localOrigin[1] + localDirection[1] * t,
      localOrigin[2] + localDirection[2] * t,
    ];
    const faceAxes = otherAxes(faceAxis);
    const uAxis = faceAxes[0];
    const vAxis = faceAxes[1];
    const uRegion = clampRegionIndex(localHit[uAxis], this.halfSize[uAxis]);
    const vRegion = clampRegionIndex(localHit[vAxis], this.halfSize[vAxis]);
    const resizeEdges: EdgeMove[] = [];
    if (uRegion === 0) {
      resizeEdges.push(faceEdgeMove(faceAxis, faceSign, uAxis, -1));
    } else if (uRegion === 2) {
      resizeEdges.push(faceEdgeMove(faceAxis, faceSign, uAxis, 1));
    }
    if (vRegion === 0) {
      resizeEdges.push(faceEdgeMove(faceAxis, faceSign, vAxis, -1));
    } else if (vRegion === 2) {
      resizeEdges.push(faceEdgeMove(faceAxis, faceSign, vAxis, 1));
    }
    if (resizeEdges.length !== 0) {
      return {
        faceAxis,
        faceSign,
        uRegion,
        vRegion,
        highlightedEdges: resizeEdges.map((edge) => edge.edgeIndex),
        resizeEdges,
        rotate: false,
      };
    }

    const highlightedEdges = [
      faceEdgeMove(faceAxis, faceSign, uAxis, -1).edgeIndex,
      faceEdgeMove(faceAxis, faceSign, uAxis, 1).edgeIndex,
      faceEdgeMove(faceAxis, faceSign, vAxis, -1).edgeIndex,
      faceEdgeMove(faceAxis, faceSign, vAxis, 1).edgeIndex,
    ];
    return {
      faceAxis,
      faceSign,
      uRegion,
      vRegion,
      highlightedEdges,
      resizeEdges: [],
      rotate: true,
    };
  }

  updateHover(
    event: MouseEvent,
    element: HTMLElement,
    sliceViews: Iterable<[SliceView, boolean]>,
    showSliceViews: boolean,
    projectionParameters: ProjectionParameters,
  ) {
    this.ensureInitialized(sliceViews, showSliceViews, projectionParameters);
    const hover = this.drag
      ? this.hover
      : this.hitTest(event, element, projectionParameters);
    if (this.sameHover(hover, this.hover)) return false;
    this.hover = hover;
    return true;
  }

  private sameHover(a: BoxHover | undefined, b: BoxHover | undefined) {
    if (a === b) return true;
    if (a === undefined || b === undefined) return false;
    if (
      a.faceAxis !== b.faceAxis ||
      a.faceSign !== b.faceSign ||
      a.uRegion !== b.uRegion ||
      a.vRegion !== b.vRegion ||
      a.rotate !== b.rotate ||
      a.highlightedEdges.length !== b.highlightedEdges.length
    ) {
      return false;
    }
    for (let i = 0; i < a.highlightedEdges.length; ++i) {
      if (a.highlightedEdges[i] !== b.highlightedEdges[i]) return false;
    }
    return true;
  }

  startDrag(
    event: MouseEvent,
    element: HTMLElement,
    projectionParameters: ProjectionParameters,
  ) {
    const hover = this.hover;
    if (hover === undefined) return false;
    const origin = vec3.create();
    const direction = vec3.create();
    if (
      !this.getMouseRay(origin, direction, event, element, projectionParameters)
    ) {
      return false;
    }
    const planeNormal = vec3.clone(this.axes[hover.faceAxis]);
    vec3.scale(planeNormal, planeNormal, hover.faceSign);
    const planePoint = vec3.clone(this.center);
    vec3.scaleAndAdd(
      planePoint,
      planePoint,
      this.axes[hover.faceAxis],
      hover.faceSign * this.halfSize[hover.faceAxis],
    );
    const previousPoint = vec3.create();
    if (
      !intersectRayPlane(
        previousPoint,
        origin,
        direction,
        planePoint,
        planeNormal,
      )
    ) {
      return false;
    }
    const previousVector = vec3.create();
    if (hover.rotate) {
      if (
        !this.computeRotationVector(previousVector, previousPoint, planeNormal)
      ) {
        return false;
      }
    }
    this.drag = {
      mode: hover.rotate ? "rotate" : "resize",
      planePoint,
      planeNormal,
      previousPoint,
      previousVector,
      resizeEdges: hover.resizeEdges,
    };
    return true;
  }

  dragTo(
    event: MouseEvent,
    element: HTMLElement,
    projectionParameters: ProjectionParameters,
  ) {
    const drag = this.drag;
    if (drag === undefined) return false;
    const origin = vec3.create();
    const direction = vec3.create();
    if (
      !this.getMouseRay(origin, direction, event, element, projectionParameters)
    ) {
      return false;
    }
    const point = vec3.create();
    if (
      !intersectRayPlane(
        point,
        origin,
        direction,
        drag.planePoint,
        drag.planeNormal,
      )
    ) {
      return false;
    }
    if (drag.mode === "resize") {
      vec3.subtract(tempVec, point, drag.previousPoint);
      for (const edge of drag.resizeEdges) {
        const amount = vec3.dot(tempVec, this.axes[edge.dimension]);
        this.resizeDimension(edge.dimension, edge.sign, amount);
      }
      vec3.copy(drag.previousPoint, point);
      return true;
    }

    const currentVector = vec3.create();
    if (!this.computeRotationVector(currentVector, point, drag.planeNormal)) {
      return false;
    }
    vec3.cross(tempVec, drag.previousVector, currentVector);
    const sin = vec3.dot(drag.planeNormal, tempVec);
    const cos = vec3.dot(drag.previousVector, currentVector);
    const angle = Math.atan2(sin, cos);
    if (!Number.isFinite(angle) || angle === 0) return false;
    quat.setAxisAngle(tempQuat, drag.planeNormal, angle);
    for (const axis of this.axes) {
      vec3.transformQuat(axis, axis, tempQuat);
      normalizeAxis(axis);
    }
    vec3.copy(drag.previousVector, currentVector);
    return true;
  }

  endDrag() {
    this.drag = undefined;
  }

  private computeRotationVector(out: vec3, point: vec3, normal: vec3) {
    vec3.subtract(out, point, this.center);
    vec3.scaleAndAdd(out, out, normal, -vec3.dot(out, normal));
    return normalizeAxis(out);
  }

  private resizeDimension(dimension: 0 | 1 | 2, sign: -1 | 1, amount: number) {
    const oldHalfSize = this.halfSize[dimension];
    const largestHalfSize = Math.max(
      this.halfSize[0],
      this.halfSize[1],
      this.halfSize[2],
      1,
    );
    const minimumHalfSize = largestHalfSize * 1e-4;
    let newHalfSize = oldHalfSize + (sign * amount) / 2;
    if (newHalfSize < minimumHalfSize) {
      newHalfSize = minimumHalfSize;
      amount = (2 * (newHalfSize - oldHalfSize)) / sign;
    }
    vec3.scaleAndAdd(
      this.center,
      this.center,
      this.axes[dimension],
      amount / 2,
    );
    this.halfSize[dimension] = newHalfSize;
  }

  draw(projectionParameters: ProjectionParameters) {
    if (!this.initialized) return;
    const gl = this.gl;
    const shader = this.shader;
    shader.bind();
    mat4.copy(tempMat, projectionParameters.viewProjectionMat);
    gl.uniformMatrix4fv(shader.uniform("uProjectionMatrix"), false, tempMat);
    const attribute = shader.attribute("aVertexPosition");
    gl.lineWidth(1);
    const hover = this.hover;
    if (hover !== undefined) {
      this.drawHoverRegionBackground(hover, attribute);
    }
    this.drawEdges(
      Array.from({ length: BOX_EDGE_COUNT }, (_value, index) => index),
      BLUE,
      attribute,
    );
    const highlightedEdges = hover?.highlightedEdges;
    if (highlightedEdges !== undefined && highlightedEdges.length !== 0) {
      gl.lineWidth(2);
      this.drawEdges(highlightedEdges, WHITE, attribute);
      this.drawHoverIcons(hover!, attribute);
      gl.lineWidth(1);
    }
    gl.disableVertexAttribArray(attribute);
  }

  private drawHoverRegionBackground(hover: BoxHover, attribute: number) {
    const gl = this.gl;
    const shader = this.shader;
    const offset = this.writeHoverRegionQuad(hover, this.lineVertexData, 0);
    this.vertexBuffer.setData(
      this.lineVertexData.subarray(0, offset),
      WebGL2RenderingContext.DYNAMIC_DRAW,
    );
    this.vertexBuffer.bindToVertexAttrib(attribute, 4);
    gl.uniform4fv(shader.uniform("uColor"), HOVER_REGION_BACKGROUND);
    gl.drawArrays(
      WebGL2RenderingContext.TRIANGLES,
      0,
      offset / VERTEX_COMPONENTS,
    );
  }

  private drawEdges(edges: number[], color: Float32Array, attribute: number) {
    const gl = this.gl;
    const shader = this.shader;
    let offset = 0;
    for (const edge of edges) {
      offset = this.writeEdge(edge, this.lineVertexData, offset);
    }
    this.vertexBuffer.setData(
      this.lineVertexData.subarray(0, offset),
      WebGL2RenderingContext.DYNAMIC_DRAW,
    );
    this.vertexBuffer.bindToVertexAttrib(attribute, 4);
    gl.uniform4fv(shader.uniform("uColor"), color);
    gl.drawArrays(WebGL2RenderingContext.LINES, 0, offset / VERTEX_COMPONENTS);
  }

  private drawHoverIcons(hover: BoxHover, attribute: number) {
    const gl = this.gl;
    const shader = this.shader;
    const offset = hover.rotate
      ? this.writeRotateHoverIcons(hover, this.lineVertexData, 0)
      : this.writeResizeHoverIcons(hover, this.lineVertexData, 0);
    if (offset === 0) return;
    this.vertexBuffer.setData(
      this.lineVertexData.subarray(0, offset),
      WebGL2RenderingContext.DYNAMIC_DRAW,
    );
    this.vertexBuffer.bindToVertexAttrib(attribute, 4);
    gl.uniform4fv(shader.uniform("uColor"), WHITE);
    gl.drawArrays(WebGL2RenderingContext.LINES, 0, offset / VERTEX_COMPONENTS);
  }

  private writeResizeHoverIcons(
    hover: BoxHover,
    out: Float32Array,
    offset: number,
  ) {
    const faceAxes = otherAxes(hover.faceAxis);
    const regionCenter: LocalPoint = [0, 0, 0];
    regionCenter[hover.faceAxis] =
      hover.faceSign * this.halfSize[hover.faceAxis];
    for (const axis of faceAxes) {
      const resizeEdge = hover.resizeEdges.find(
        (edge) => edge.dimension === axis,
      );
      regionCenter[axis] =
        resizeEdge === undefined
          ? 0
          : resizeEdge.sign *
            this.halfSize[axis] *
            HOVER_REGION_CENTER_FRACTION;
    }

    for (const edge of hover.resizeEdges) {
      offset = this.writeResizeArrowIcon(
        hover,
        edge,
        regionCenter,
        out,
        offset,
      );
    }
    return offset;
  }

  private writeResizeArrowIcon(
    hover: BoxHover,
    edge: EdgeMove,
    regionCenter: LocalPoint,
    out: Float32Array,
    offset: number,
  ) {
    const faceAxes = otherAxes(hover.faceAxis);
    const perpendicularAxis =
      faceAxes[0] === edge.dimension ? faceAxes[1] : faceAxes[0];
    const start = copyLocalPoint(regionCenter);
    const end = copyLocalPoint(regionCenter);
    start[edge.dimension] =
      edge.sign * this.halfSize[edge.dimension] * RESIZE_ARROW_START_FRACTION;
    end[edge.dimension] =
      edge.sign * this.halfSize[edge.dimension] * RESIZE_ARROW_END_FRACTION;
    offset = this.writeLocalLine(start, end, out, offset);

    const direction: LocalPoint = [0, 0, 0];
    direction[edge.dimension] = edge.sign;
    return this.writeLocalArrowHead(
      hover.faceAxis,
      end,
      direction,
      this.halfSize[edge.dimension] * RESIZE_ARROW_HEAD_LENGTH_FRACTION,
      this.halfSize[perpendicularAxis] * RESIZE_ARROW_HEAD_WIDTH_FRACTION,
      out,
      offset,
    );
  }

  private writeRotateHoverIcons(
    hover: BoxHover,
    out: Float32Array,
    offset: number,
  ) {
    const faceAxes = otherAxes(hover.faceAxis);
    const uAxis = faceAxes[0];
    const vAxis = faceAxes[1];
    const radius =
      Math.min(this.halfSize[uAxis], this.halfSize[vAxis]) *
      ROTATE_ICON_RADIUS_FRACTION;
    if (radius <= 0) return offset;
    const center: LocalPoint = [0, 0, 0];
    center[hover.faceAxis] = hover.faceSign * this.halfSize[hover.faceAxis];
    const angleStep = (2 * Math.PI) / ROTATE_ICON_ARROW_COUNT;
    const arcLength = angleStep - ROTATE_ICON_ARC_GAP_RADIANS;
    for (let arrow = 0; arrow < ROTATE_ICON_ARROW_COUNT; ++arrow) {
      const startAngle = arrow * angleStep + ROTATE_ICON_ARC_GAP_RADIANS / 2;
      const endAngle = startAngle + arcLength;
      let previousPoint = this.getRotateIconPoint(
        center,
        uAxis,
        vAxis,
        radius,
        startAngle,
      );
      for (let segment = 1; segment <= ROTATE_ICON_ARC_SEGMENTS; ++segment) {
        const angle =
          startAngle + (arcLength * segment) / ROTATE_ICON_ARC_SEGMENTS;
        const currentPoint = this.getRotateIconPoint(
          center,
          uAxis,
          vAxis,
          radius,
          angle,
        );
        offset = this.writeLocalLine(previousPoint, currentPoint, out, offset);
        previousPoint = currentPoint;
      }

      const direction: LocalPoint = [0, 0, 0];
      direction[uAxis] = -Math.sin(endAngle);
      direction[vAxis] = Math.cos(endAngle);
      offset = this.writeLocalArrowHead(
        hover.faceAxis,
        previousPoint,
        direction,
        radius * ROTATE_ICON_HEAD_LENGTH_FRACTION,
        radius * ROTATE_ICON_HEAD_WIDTH_FRACTION,
        out,
        offset,
      );
    }
    return offset;
  }

  private writeHoverRegionQuad(
    hover: BoxHover,
    out: Float32Array,
    offset: number,
  ) {
    const faceAxes = otherAxes(hover.faceAxis);
    const uAxis = faceAxes[0];
    const vAxis = faceAxes[1];
    const uMin = this.getRegionLowerBound(hover.uRegion, uAxis);
    const uMax = this.getRegionLowerBound(
      (hover.uRegion + 1) as 1 | 2 | 3,
      uAxis,
    );
    const vMin = this.getRegionLowerBound(hover.vRegion, vAxis);
    const vMax = this.getRegionLowerBound(
      (hover.vRegion + 1) as 1 | 2 | 3,
      vAxis,
    );
    const local: LocalPoint = [0, 0, 0];
    local[hover.faceAxis] = hover.faceSign * this.halfSize[hover.faceAxis];
    const writeCorner = (u: number, v: number) => {
      local[uAxis] = u;
      local[vAxis] = v;
      transformLocalToWorld(tempVec, local, this.center, this.axes);
      out[offset++] = tempVec[0];
      out[offset++] = tempVec[1];
      out[offset++] = tempVec[2];
      out[offset++] = 1;
    };
    writeCorner(uMin, vMin);
    writeCorner(uMax, vMin);
    writeCorner(uMax, vMax);
    writeCorner(uMin, vMin);
    writeCorner(uMax, vMax);
    writeCorner(uMin, vMax);
    return offset;
  }

  private getRegionLowerBound(region: 0 | 1 | 2 | 3, axis: 0 | 1 | 2) {
    return ((2 * region) / 3 - 1) * this.halfSize[axis];
  }

  private getRotateIconPoint(
    center: LocalPoint,
    uAxis: 0 | 1 | 2,
    vAxis: 0 | 1 | 2,
    radius: number,
    angle: number,
  ): LocalPoint {
    const point = copyLocalPoint(center);
    point[uAxis] += Math.cos(angle) * radius;
    point[vAxis] += Math.sin(angle) * radius;
    return point;
  }

  private writeLocalArrowHead(
    faceAxis: 0 | 1 | 2,
    tip: LocalPoint,
    direction: LocalPoint,
    headLength: number,
    headWidth: number,
    out: Float32Array,
    offset: number,
  ) {
    const directionLength = Math.hypot(
      direction[0],
      direction[1],
      direction[2],
    );
    if (directionLength === 0) return offset;
    const unitDirection: LocalPoint = [
      direction[0] / directionLength,
      direction[1] / directionLength,
      direction[2] / directionLength,
    ];
    const faceAxes = otherAxes(faceAxis);
    const uAxis = faceAxes[0];
    const vAxis = faceAxes[1];
    const perpendicular: LocalPoint = [0, 0, 0];
    perpendicular[uAxis] = -unitDirection[vAxis];
    perpendicular[vAxis] = unitDirection[uAxis];
    const base0: LocalPoint = [
      tip[0] - unitDirection[0] * headLength + perpendicular[0] * headWidth,
      tip[1] - unitDirection[1] * headLength + perpendicular[1] * headWidth,
      tip[2] - unitDirection[2] * headLength + perpendicular[2] * headWidth,
    ];
    const base1: LocalPoint = [
      tip[0] - unitDirection[0] * headLength - perpendicular[0] * headWidth,
      tip[1] - unitDirection[1] * headLength - perpendicular[1] * headWidth,
      tip[2] - unitDirection[2] * headLength - perpendicular[2] * headWidth,
    ];
    offset = this.writeLocalLine(tip, base0, out, offset);
    return this.writeLocalLine(tip, base1, out, offset);
  }

  private writeLocalLine(
    start: LocalPoint,
    end: LocalPoint,
    out: Float32Array,
    offset: number,
  ) {
    transformLocalToWorld(tempVec, start, this.center, this.axes);
    out[offset++] = tempVec[0];
    out[offset++] = tempVec[1];
    out[offset++] = tempVec[2];
    out[offset++] = 1;
    transformLocalToWorld(tempVec, end, this.center, this.axes);
    out[offset++] = tempVec[0];
    out[offset++] = tempVec[1];
    out[offset++] = tempVec[2];
    out[offset++] = 1;
    return offset;
  }

  private writeEdge(edge: number, out: Float32Array, offset: number) {
    const varyingAxis = axisFromNumber(Math.floor(edge / 4));
    const fixedAxes = otherAxes(varyingAxis);
    const sign0: -1 | 1 = edge & 1 ? 1 : -1;
    const sign1: -1 | 1 = edge & 2 ? 1 : -1;
    const local = [0, 0, 0];
    local[fixedAxes[0]] = sign0 * this.halfSize[fixedAxes[0]];
    local[fixedAxes[1]] = sign1 * this.halfSize[fixedAxes[1]];
    for (const varyingSign of [-1, 1] as const) {
      local[varyingAxis] = varyingSign * this.halfSize[varyingAxis];
      transformLocalToWorld(tempVec, local, this.center, this.axes);
      out[offset++] = tempVec[0];
      out[offset++] = tempVec[1];
      out[offset++] = tempVec[2];
      out[offset++] = 1;
    }
    return offset;
  }
}
