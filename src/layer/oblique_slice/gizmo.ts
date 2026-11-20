import {
  CENTER_BALL_DIAMETER,
  GIZMO_SIZE,
  HOVER_HIGHLIGHT,
} from "#src/annotation/oriented_bounding_box.js";
import type {
  MouseSelectionState,
  VisibleLayerInfo,
} from "#src/layer/index.js";
import {
  capPolygon,
  displayToSlicerPosition,
  type SlicerFrame,
  slabSideEdges,
  slicerFrame,
} from "#src/layer/oblique_slice/frame.js";
import type { ObliqueSliceUserLayer } from "#src/layer/oblique_slice/index.js";
import type { NavigationState } from "#src/navigation_state.js";
import type { PerspectivePanel } from "#src/perspective_view/panel.js";
import type { PerspectiveViewRenderContext } from "#src/perspective_view/render_layer.js";
import { PerspectiveViewRenderLayer } from "#src/perspective_view/render_layer.js";
import type { ProjectionParameters } from "#src/projection_parameters.js";
import type { DraggableGizmo } from "#src/renderlayer.js";
import { constantWatchableValue } from "#src/trackable_value.js";
import { mat4, quat, vec3, vec4 } from "#src/util/geom.js";
import { GLBuffer } from "#src/webgl/buffer.js";
import {
  defineCircleShader,
  drawCircles,
  initializeCircleShader,
} from "#src/webgl/circles.js";
import type { GL } from "#src/webgl/context.js";
import {
  type ParameterizedEmitterDependentShaderGetter,
  parameterizedEmitterDependentShaderGetter,
} from "#src/webgl/dynamic_shader.js";
import type { ShaderBuilder } from "#src/webgl/shader.js";

export enum GizmoPart {
  BALL = 0,
  SQUARE_POSITIVE = 1,
  SQUARE_NEGATIVE = 2,
  PLANE = 3,
}
const PART_COUNT = 4;

const GREY = vec3.fromValues(0.62, 0.62, 0.62);
const WHITE = vec3.fromValues(1, 1, 1);
const GUIDE_COLOR = vec4.fromValues(0.85, 0.85, 0.85, 0.9);
const SQUARE_FILL_ALPHA = 0.35;

// Gizmo units: 1 = GIZMO_SIZE of the viewport. The squares never come closer
// than MIN_SQUARE_DISTANCE so they stay clear of the ball when the slab is thin.
const SQUARE_HALF = 0.09;
const MIN_SQUARE_DISTANCE = 0.3;
// Plane-drag trackball: rim just past the grab point, never smaller than this
// (in aspect-corrected NDC) so a grab near the center isn't hypersensitive.
const TRACKBALL_RADIUS_FACTOR = 1.15;
const MIN_TRACKBALL_RADIUS = 0.3;

interface DragStart {
  part: GizmoPart;
  nav: NavigationState;
  frame: SlicerFrame;
  position: number[];
  voxelRange: number;
  viewProjection: mat4;
  view: mat4;
  aspect: number;
}

function ndc(m: mat4, p: vec3, aspect: number) {
  const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
  const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
  const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
  return { x: (x / w) * aspect, y: y / w };
}

function onTrackball(x: number, y: number): vec3 {
  const d2 = x * x + y * y;
  const v =
    d2 > 1
      ? vec3.fromValues(x, y, 0)
      : vec3.fromValues(x, y, Math.sqrt(1 - d2));
  return vec3.normalize(v, v);
}

// Where the line through `p` along `dir` enters and leaves the box.
function lineThroughBox(frame: SlicerFrame, p: vec3, dir: vec3) {
  let t0 = -Infinity;
  let t1 = Infinity;
  for (let i = 0; i < 3; ++i) {
    if (Math.abs(dir[i]) < 1e-9) continue;
    const a = (frame.lower[i] - p[i]) / dir[i];
    const b = (frame.upper[i] - p[i]) / dir[i];
    t0 = Math.max(t0, Math.min(a, b));
    t1 = Math.min(t1, Math.max(a, b));
  }
  if (!(t0 < t1)) return undefined;
  return [
    vec3.scaleAndAdd(vec3.create(), p, dir, t0),
    vec3.scaleAndAdd(vec3.create(), p, dir, t1),
  ];
}

export class ObliqueSliceGizmoRenderLayer
  extends PerspectiveViewRenderLayer
  implements DraggableGizmo
{
  private gl: GL | undefined;
  private circleShader: ParameterizedEmitterDependentShaderGetter | undefined;
  private flatShader: ParameterizedEmitterDependentShaderGetter | undefined;
  private flatBuffer: GLBuffer | undefined;
  private nav: NavigationState | undefined;
  private mouseState: MouseSelectionState | undefined;
  private projection: ProjectionParameters | undefined;
  private worldScale = 1;
  private dragStart: DragStart | undefined;

  constructor(public slicer: ObliqueSliceUserLayer) {
    super();
    this.isAnnotation = true;
  }

  attach(attachment: VisibleLayerInfo<PerspectivePanel>) {
    const panel = attachment.view;
    this.nav = panel.navigationState;
    this.mouseState = panel.viewer.mouseState;
    const redraw = () => this.redrawNeeded.dispatch();
    const { slicer } = this;
    for (const signal of [
      panel.viewer.mouseState.changed,
      slicer.selected.changed,
      slicer.position.changed,
      slicer.orientation.changed,
      slicer.voxelRange.changed,
    ]) {
      attachment.registerDisposer(signal.add(redraw));
    }
  }

  private initialize(gl: GL) {
    if (this.gl === gl) return;
    this.gl = gl;
    const parameters = constantWatchableValue(undefined);
    this.circleShader = parameterizedEmitterDependentShaderGetter(this, gl, {
      memoizeKey: "oblique_slice/gizmo/circle",
      parameters,
      defineShader: (builder: ShaderBuilder) => {
        defineCircleShader(builder, false);
        builder.addUniform("highp mat4", "uViewProjection");
        builder.addUniform("highp vec3", "uOrigin");
        builder.addUniform("highp vec3", "uColor");
        builder.addUniform("highp uint", "uPickID");
        builder.setVertexMain(
          `emitCircle(uViewProjection * vec4(uOrigin, 1.0), ${CENTER_BALL_DIAMETER.toFixed(1)}, 0.0);`,
        );
        builder.setFragmentMain(`
vec4 color = vec4(uColor, 1.0);
emit(getCircleColor(color, color), uPickID);
`);
      },
    });
    this.flatShader = parameterizedEmitterDependentShaderGetter(this, gl, {
      memoizeKey: "oblique_slice/gizmo/flat",
      parameters,
      defineShader: (builder: ShaderBuilder) => {
        builder.addAttribute("highp vec3", "aPos");
        builder.addUniform("highp mat4", "uViewProjection");
        builder.addUniform("highp vec4", "uColor");
        builder.addUniform("highp uint", "uPickID");
        builder.setVertexMain(
          `gl_Position = uViewProjection * vec4(aPos, 1.0);`,
        );
        builder.setFragmentMain(`emit(uColor, uPickID);`);
      },
    });
    this.flatBuffer = this.registerDisposer(new GLBuffer(gl));
  }

  private drawFlat(
    renderContext: PerspectiveViewRenderContext,
    mode: number,
    points: number[],
    color: vec4,
    pickID: number,
  ) {
    const gl = this.gl!;
    const shader = this.flatShader!(renderContext.emitter).shader;
    if (shader === null || points.length === 0) return;
    this.flatBuffer!.setData(Float32Array.from(points));
    shader.bind();
    gl.uniformMatrix4fv(
      shader.uniform("uViewProjection"),
      false,
      renderContext.projectionParameters.viewProjectionMat,
    );
    gl.uniform4fv(shader.uniform("uColor"), color);
    gl.uniform1ui(shader.uniform("uPickID"), pickID);
    const aPos = shader.attribute("aPos");
    this.flatBuffer!.bindToVertexAttrib(aPos, 3);
    gl.drawArrays(mode, 0, points.length / 3);
    gl.disableVertexAttribArray(aPos);
  }

  private squareDistance(voxelRange: number) {
    return Math.max(voxelRange / this.worldScale, MIN_SQUARE_DISTANCE);
  }

  draw(
    renderContext: PerspectiveViewRenderContext,
    attachment: VisibleLayerInfo<PerspectivePanel>,
  ) {
    const { slicer } = this;
    if (!renderContext.isForeground || !slicer.selected.value) return;
    const panel = attachment.view;
    const gl = panel.gl;
    this.initialize(gl);
    const nav = panel.navigationState;
    this.nav = nav;
    const pp = renderContext.projectionParameters;
    this.projection = pp;
    const frame = slicerFrame(
      {
        position: slicer.position.value,
        orientation: slicer.orientation.value,
      },
      nav,
    );
    if (frame === undefined) return;
    const { viewProjectionMat, viewMatrix, projectionMat } = pp;
    const c = frame.center;
    const clipW =
      Math.abs(
        viewProjectionMat[3] * c[0] +
          viewProjectionMat[7] * c[1] +
          viewProjectionMat[11] * c[2] +
          viewProjectionMat[15],
      ) || 1;
    const viewScale = Math.hypot(viewMatrix[0], viewMatrix[1], viewMatrix[2]);
    const ndcPerWorld = (Math.abs(projectionMat[5]) * viewScale) / clipW;
    this.worldScale = GIZMO_SIZE / Math.max(ndcPerWorld, 1e-6);

    const voxelRange = slicer.voxelRange.value;
    const dragged = this.dragStart?.part;
    if (dragged === GizmoPart.BALL) {
      const line = lineThroughBox(frame, frame.center, frame.n);
      if (line !== undefined) {
        this.drawFlat(
          renderContext,
          WebGL2RenderingContext.LINES,
          [...line[0], ...line[1]],
          GUIDE_COLOR,
          0,
        );
      }
    }
    if (
      dragged === GizmoPart.SQUARE_POSITIVE ||
      dragged === GizmoPart.SQUARE_NEGATIVE
    ) {
      this.drawSlabPrism(renderContext, frame, voxelRange);
    }

    const pickBase = renderContext.emitPickID
      ? renderContext.pickIDs.register(this, PART_COUNT)
      : 0;
    const { mouseState } = panel.viewer;
    const highlighted =
      dragged ??
      (mouseState.pickedRenderLayer === this ? mouseState.pickedOffset : -1);
    const color = (part: GizmoPart) =>
      part === highlighted
        ? vec3.lerp(vec3.create(), GREY, WHITE, HOVER_HIGHLIGHT)
        : GREY;

    const half = SQUARE_HALF * this.worldScale;
    const distance = this.squareDistance(voxelRange) * this.worldScale;
    for (const part of [GizmoPart.SQUARE_POSITIVE, GizmoPart.SQUARE_NEGATIVE]) {
      const sign = part === GizmoPart.SQUARE_POSITIVE ? 1 : -1;
      const middle = vec3.scaleAndAdd(
        vec3.create(),
        frame.center,
        frame.n,
        sign * distance,
      );
      const corner = (su: number, sv: number) => {
        const p = vec3.scaleAndAdd(vec3.create(), middle, frame.u, su * half);
        return [...vec3.scaleAndAdd(p, p, frame.v, sv * half)];
      };
      const [a, b, cc, d] = [
        corner(-1, -1),
        corner(1, -1),
        corner(1, 1),
        corner(-1, 1),
      ];
      const rgb = color(part);
      this.drawFlat(
        renderContext,
        WebGL2RenderingContext.TRIANGLES,
        [...a, ...b, ...cc, ...a, ...cc, ...d],
        vec4.fromValues(rgb[0], rgb[1], rgb[2], SQUARE_FILL_ALPHA),
        pickBase + part,
      );
      this.drawFlat(
        renderContext,
        WebGL2RenderingContext.LINE_LOOP,
        [...a, ...b, ...cc, ...d],
        vec4.fromValues(rgb[0], rgb[1], rgb[2], 1),
        pickBase + part,
      );
    }

    const circle = this.circleShader!(renderContext.emitter).shader;
    if (circle !== null) {
      circle.bind();
      gl.uniformMatrix4fv(
        circle.uniform("uViewProjection"),
        false,
        viewProjectionMat,
      );
      initializeCircleShader(circle, pp, { featherWidthInPixels: 1 });
      gl.uniform3fv(circle.uniform("uOrigin"), frame.center);
      gl.uniform3fv(circle.uniform("uColor"), color(GizmoPart.BALL));
      gl.uniform1ui(circle.uniform("uPickID"), pickBase + GizmoPart.BALL);
      drawCircles(gl, 1, 1);
    }
  }

  private drawSlabPrism(
    renderContext: PerspectiveViewRenderContext,
    frame: SlicerFrame,
    halfThickness: number,
  ) {
    const segments: number[] = [];
    const offsets = halfThickness > 0 ? [halfThickness, -halfThickness] : [0];
    for (const offset of offsets) {
      const cap = capPolygon(frame, offset);
      cap.forEach((p, i) => {
        segments.push(...p, ...cap[(i + 1) % cap.length]);
      });
    }
    if (halfThickness > 0) {
      for (const [a, b] of slabSideEdges(frame, halfThickness)) {
        segments.push(...a, ...b);
      }
    }
    this.drawFlat(
      renderContext,
      WebGL2RenderingContext.LINES,
      segments,
      GUIDE_COLOR,
      0,
    );
  }

  dragAnchor(part: number): vec3 | undefined {
    const { slicer, nav, projection, mouseState } = this;
    if (
      nav === undefined ||
      projection === undefined ||
      part < 0 ||
      part >= PART_COUNT
    ) {
      return undefined;
    }
    const position = slicer.position.value;
    const frame = slicerFrame(
      { position, orientation: slicer.orientation.value },
      nav,
    );
    if (frame === undefined) return undefined;
    let anchor: vec3;
    switch (part) {
      case GizmoPart.PLANE: {
        if (mouseState === undefined || !mouseState.active) return undefined;
        anchor = vec3.create();
        for (let i = 0; i < 3; ++i) {
          anchor[i] = mouseState.position[frame.displayDims[i]];
        }
        break;
      }
      case GizmoPart.SQUARE_POSITIVE:
      case GizmoPart.SQUARE_NEGATIVE: {
        const sign = part === GizmoPart.SQUARE_POSITIVE ? 1 : -1;
        anchor = vec3.scaleAndAdd(
          vec3.create(),
          frame.center,
          frame.n,
          sign * this.squareDistance(slicer.voxelRange.value) * this.worldScale,
        );
        break;
      }
      default:
        anchor = vec3.clone(frame.center);
    }
    this.dragStart = {
      part,
      nav,
      frame,
      position: [...position],
      voxelRange: slicer.voxelRange.value,
      viewProjection: mat4.clone(projection.viewProjectionMat),
      view: mat4.clone(projection.viewMatrix),
      aspect: projection.width / Math.max(projection.height, 1),
    };
    this.redrawNeeded.dispatch();
    return anchor;
  }

  drag(part: number, anchor: vec3, dragged: vec3) {
    const start = this.dragStart;
    if (start === undefined || start.part !== part) return;
    const { slicer } = this;
    const { frame } = start;
    const moved = vec3.subtract(vec3.create(), dragged, anchor);
    const distance = vec3.dot(frame.n, moved);
    switch (part) {
      case GizmoPart.BALL: {
        const center = vec3.scaleAndAdd(
          vec3.create(),
          frame.center,
          frame.n,
          distance,
        );
        slicer.position.value = displayToSlicerPosition(
          frame,
          start.nav,
          center,
          start.position,
        );
        break;
      }
      case GizmoPart.SQUARE_POSITIVE:
      case GizmoPart.SQUARE_NEGATIVE: {
        const sign = part === GizmoPart.SQUARE_POSITIVE ? 1 : -1;
        slicer.voxelRange.value = Math.max(
          0,
          Math.round(start.voxelRange + sign * distance),
        );
        break;
      }
      case GizmoPart.PLANE: {
        const delta = this.trackballDelta(start, anchor, dragged);
        if (delta === undefined) return;
        const rotation = quat.multiply(quat.create(), delta, frame.rotation);
        slicer.orientation.value = Array.from(
          quat.normalize(rotation, rotation),
        );
        break;
      }
    }
  }

  // A virtual trackball centred on the slicer: the rotation carrying the grab
  // point to the current point, in display space.
  private trackballDelta(
    start: DragStart,
    anchor: vec3,
    dragged: vec3,
  ): quat | undefined {
    const { viewProjection, view, aspect, frame } = start;
    const center = ndc(viewProjection, frame.center, aspect);
    const grab = ndc(viewProjection, anchor, aspect);
    const cur = ndc(viewProjection, dragged, aspect);
    const radius = Math.max(
      Math.hypot(grab.x - center.x, grab.y - center.y) *
        TRACKBALL_RADIUS_FACTOR,
      MIN_TRACKBALL_RADIUS,
    );
    const v0 = onTrackball(
      (grab.x - center.x) / radius,
      (grab.y - center.y) / radius,
    );
    const v1 = onTrackball(
      (cur.x - center.x) / radius,
      (cur.y - center.y) / radius,
    );
    const axisView = vec3.cross(vec3.create(), v0, v1);
    const sin = vec3.length(axisView);
    if (sin < 1e-6) return quat.create();
    const angle = Math.atan2(sin, vec3.dot(v0, v1));
    vec3.scale(axisView, axisView, 1 / sin);
    // The view matrix's rotation maps display to view; its transpose maps back.
    const axis = vec3.create();
    for (let k = 0; k < 3; ++k) {
      const len =
        Math.hypot(view[k * 4], view[k * 4 + 1], view[k * 4 + 2]) || 1;
      axis[k] =
        (view[k * 4] * axisView[0] +
          view[k * 4 + 1] * axisView[1] +
          view[k * 4 + 2] * axisView[2]) /
        len;
    }
    vec3.normalize(axis, axis);
    return quat.setAxisAngle(quat.create(), axis, angle);
  }

  endDrag(part: number) {
    if (this.dragStart?.part !== part) return;
    this.dragStart = undefined;
    this.redrawNeeded.dispatch();
    this.slicer.committed.dispatch();
  }
}
