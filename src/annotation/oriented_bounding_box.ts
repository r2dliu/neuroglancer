/**
 * @file Support for rendering oriented (rotatable) bounding box annotations.
 * Includes a 3D Transform/Rotate/Scale Gizmo
 *
 * Unlike the axis-aligned bounding box, this box carries a quaternion
 * `orientation`, so its 8 corners are computed as
 * `center + R(orientation) * (sign * extents/2)` in the 3D display subspace.
 *
 */

import type { OrientedBoundingBox } from "#src/annotation/index.js";
import { AnnotationType } from "#src/annotation/index.js";
import {
  getGizmoDragStartNdc,
  getGizmoProjection,
  getRegionDataBounds,
  setGizmoProjection,
} from "#src/annotation/region_bounds.js";
import type {
  AnnotationRenderContext,
  AnnotationShaderGetter,
} from "#src/annotation/type_handler.js";
import {
  AnnotationRenderHelper,
  registerAnnotationTypeRenderHandler,
} from "#src/annotation/type_handler.js";
import {
  boundingBoxCrossSectionVertexIndices,
  vertexBasePositions,
} from "#src/sliceview/bounding_box_shader_helper.js";
import type { SliceViewPanelRenderContext } from "#src/sliceview/renderlayer.js";
import { tile2dArray } from "#src/util/array.js";
import {
  mat3,
  mat4,
  quat,
  transformVectorByMat4Transpose,
  vec3,
} from "#src/util/geom.js";
import { EDGES_PER_BOX } from "#src/webgl/bounding_box.js";
import { GLBuffer } from "#src/webgl/buffer.js";
import {
  defineCircleShader,
  drawCircles,
  initializeCircleShader,
} from "#src/webgl/circles.js";
import {
  defineLineShader,
  drawLines,
  initializeLineShader,
  VERTICES_PER_LINE,
} from "#src/webgl/lines.js";
import type { ShaderBuilder, ShaderProgram } from "#src/webgl/shader.js";
import { drawArraysInstanced } from "#src/webgl/shader.js";
import { defineVectorArrayVertexShaderInput } from "#src/webgl/shader_lib.js";
import { defineVertexId, VertexIdHelper } from "#src/webgl/vertex_id.js";

// Pick-ID layout per instance.
const FULL_OBJECT_PICK_OFFSET = 0; // 1: whole object / non-interactive
const EDGES_PICK_OFFSET = FULL_OBJECT_PICK_OFFSET + 1; // EDGES_PER_BOX: edges → free rotation
const EDGES_PICK_END = EDGES_PICK_OFFSET + EDGES_PER_BOX;
const ROTATE_RING_PICK_OFFSET = EDGES_PICK_END; // 3 rings
const TRANSLATE_AXIS_PICK_OFFSET = ROTATE_RING_PICK_OFFSET + 3; // 3 tripod arrows
const CENTER_BALL_PICK_OFFSET = TRANSLATE_AXIS_PICK_OFFSET + 3; // 1 free translate
const SCALE_AXIS_PICK_OFFSET = CENTER_BALL_PICK_OFFSET + 1; // 3 scale cubes
export const ORIENTED_BBOX_PICK_IDS_PER_INSTANCE = SCALE_AXIS_PICK_OFFSET + 3;

// A draggable part of the gizmo, resolved from a pick part index. `axis` is the
// display axis (0/1/2) for the per-axis handles.
export type GizmoHandle =
  | { kind: "translate"; axis: number } // tripod arrow → move along a world axis
  | { kind: "scale"; axis: number } // cube → resize along a world axis
  | { kind: "ring"; axis: number } // ring → rotate about a local axis
  | { kind: "centerBall" } // free translate in the screen plane
  | { kind: "edge" } // box boundary → free trackball rotation
  | { kind: "none" }; // interior / cross-section / nothing: not interactive

export function classifyGizmoPart(partIndex: number): GizmoHandle {
  const translate = partIndex - TRANSLATE_AXIS_PICK_OFFSET;
  if (translate >= 0 && translate < 3) return { kind: "translate", axis: translate };
  const scale = partIndex - SCALE_AXIS_PICK_OFFSET;
  if (scale >= 0 && scale < 3) return { kind: "scale", axis: scale };
  const ring = partIndex - ROTATE_RING_PICK_OFFSET;
  if (ring >= 0 && ring < 3) return { kind: "ring", axis: ring };
  if (partIndex === CENTER_BALL_PICK_OFFSET) return { kind: "centerBall" };
  if (partIndex >= EDGES_PICK_OFFSET && partIndex < EDGES_PICK_END) return { kind: "edge" };
  return { kind: "none" };
}

export function isInteractiveGizmoPart(partIndex: number): boolean {
  return classifyGizmoPart(partIndex).kind !== "none";
}

// axis 0 = red, 1 = green, 2 = blue.
// `axisUnit` returns the corresponding local unit axis vector.
const glsl_axisColor = `
vec3 axisColor(int a) {
  return a == 0 ? vec3(0.0, 0.0, 1.0)
       : a == 1 ? vec3(0.0, 1.0, 0.0)
       :          vec3(1.0, 0.0, 0.0);
}
vec3 axisUnit(int a) {
  return vec3(a == 0 ? 1.0 : 0.0, a == 1 ? 1.0 : 0.0, a == 2 ? 1.0 : 0.0);
}
`;

// Length of the xyz part of column `col` of a column-major mat4 — i.e. the
// view-space scale of basis axis `col` under that matrix.
function columnXyzLength(m: mat4, col: number): number {
  return Math.hypot(m[col * 4], m[col * 4 + 1], m[col * 4 + 2]);
}

// Unit arrow mesh. Shader tiles this 3x and repoints to the proper axes
function makeArrowMesh(): {
  positions: Float32Array;
  normals: Float32Array;
  vertexCount: number;
} {
  const N = 16;
  const SHAFT = 0.81; // shaft top / cone base, as a fraction of total length
  const rS = 0.4; // shaft radius (relative to cone base radius = 1)
  const rC = 1.0; // cone base radius
  const coneH = 1 - SHAFT;
  const coneHyp = Math.hypot(coneH, rC);
  const nz = rC / coneHyp; // cone normal z component
  const nr = coneH / coneHyp; // cone normal radial component
  const P: number[] = [];
  const Nm: number[] = [];
  const add = (
    px: number,
    py: number,
    pz: number,
    nx: number,
    ny: number,
    nz: number,
  ) => {
    P.push(px, py, pz);
    Nm.push(nx, ny, nz);
  };
  for (let i = 0; i < N; ++i) {
    const a0 = (2 * Math.PI * i) / N;
    const a1 = (2 * Math.PI * (i + 1)) / N;
    const am = (a0 + a1) / 2;
    const c0 = Math.cos(a0);
    const s0 = Math.sin(a0);
    const c1 = Math.cos(a1);
    const s1 = Math.sin(a1);
    // Shaft side wall (two triangles).
    add(rS * c0, rS * s0, 0, c0, s0, 0);
    add(rS * c0, rS * s0, SHAFT, c0, s0, 0);
    add(rS * c1, rS * s1, SHAFT, c1, s1, 0);
    add(rS * c0, rS * s0, 0, c0, s0, 0);
    add(rS * c1, rS * s1, SHAFT, c1, s1, 0);
    add(rS * c1, rS * s1, 0, c1, s1, 0);
    // Shaft bottom disc (faces -Z).
    add(0, 0, 0, 0, 0, -1);
    add(rS * c1, rS * s1, 0, 0, 0, -1);
    add(rS * c0, rS * s0, 0, 0, 0, -1);
    // Cone base disc (faces -Z).
    add(0, 0, SHAFT, 0, 0, -1);
    add(rC * c0, rC * s0, SHAFT, 0, 0, -1);
    add(rC * c1, rC * s1, SHAFT, 0, 0, -1);
    // Cone side.
    add(rC * c0, rC * s0, SHAFT, nr * c0, nr * s0, nz);
    add(0, 0, 1, nr * Math.cos(am), nr * Math.sin(am), nz);
    add(rC * c1, rC * s1, SHAFT, nr * c1, nr * s1, nz);
  }
  return {
    positions: Float32Array.from(P),
    normals: Float32Array.from(Nm),
    vertexCount: P.length / 3,
  };
}

const ARROW_MESH = makeArrowMesh();
const VERTS_PER_ARROW = ARROW_MESH.vertexCount;
const ARROW_COUNT = 3;
const ARROW_DRAW_VERTEX_COUNT = VERTS_PER_ARROW * ARROW_COUNT;
function tileMesh(unit: Float32Array, copies: number): Float32Array {
  const out = new Float32Array(unit.length * copies);
  for (let i = 0; i < copies; ++i) out.set(unit, i * unit.length);
  return out;
}
const ARROW_POSITIONS = tileMesh(ARROW_MESH.positions, ARROW_COUNT);
const ARROW_NORMALS = tileMesh(ARROW_MESH.normals, ARROW_COUNT);

// Unit cube mesh
function makeCubeMesh(): {
  positions: Float32Array;
  normals: Float32Array;
  vertexCount: number;
} {
  const faces: [number[], number[][]][] = [
    [[0, 0, 1], [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]],
    [[0, 0, -1], [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]]],
    [[1, 0, 0], [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]]],
    [[-1, 0, 0], [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]]],
    [[0, 1, 0], [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]]],
    [[0, -1, 0], [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]]],
  ];
  const P: number[] = [];
  const Nm: number[] = [];
  for (const [n, c] of faces) {
    for (const i of [0, 1, 2, 0, 2, 3]) {
      P.push(c[i][0], c[i][1], c[i][2]);
      Nm.push(n[0], n[1], n[2]);
    }
  }
  return {
    positions: Float32Array.from(P),
    normals: Float32Array.from(Nm),
    vertexCount: P.length / 3,
  };
}

const CUBE_MESH = makeCubeMesh();
const VERTS_PER_CUBE = CUBE_MESH.vertexCount;
const N_SCALE_CUBES = 3;
const CUBE_DRAW_VERTEX_COUNT = VERTS_PER_CUBE * N_SCALE_CUBES;
const CUBE_POSITIONS = tileMesh(CUBE_MESH.positions, N_SCALE_CUBES);
const CUBE_NORMALS = tileMesh(CUBE_MESH.normals, N_SCALE_CUBES);
// Scale-cube placement, in isotropic gizmo units (the arrow spans 0..1 along its
// axis): cube floating a little past the arrow tip (a small gap), this edge size.
const CUBE_CENTER_Z = 1.32;
const CUBE_EDGE = 0.16;

// Gizmo handles are drawn at a constant on-screen size, independent of zoom.
// Scale is from -1 to 1 for full viewport
const GIZMO_SIZE = 0.18;
// Tripod arrows start this fraction of their length out from the center, so they
// begin just past the center ball instead of clipping through it.
const GIZMO_ARROW_START_FRACTION = 0.15;
// Center "free translate" ball diameter, in pixels (screen-space via emitCircle).
const CENTER_BALL_DIAMETER = 13;
// Rotation rings: 3 segmented line loops (the great circles of the box's
// bounding sphere), colored per local axis.
const RING_SEGMENTS = 48;
const N_RING_LINES = 3 * RING_SEGMENTS;

// Alpha multiplier applied to the non-selected boxes' wireframe / cross-section
// while a region is selected, so the selected box stands out. Matches the
// segmentation layer's focus dim (the app sets `focusDim = 0.6` for instances).
const NOT_SELECTED_ALPHA = 0.6;
// Hover highlight strength: how far the hovered box's color is mixed toward white.
const HOVER_HIGHLIGHT = 0.6;
// Trackball sensitivity for free (edge-drag) rotation, in screen radii.
const FREE_ROTATE_RADIUS = 0.9;

// The 12 edges of the unit cube [0,1]^3. Each row is:
//   cornerA (xyz in {0,1}), cornerB (xyz in {0,1}), edge pick index.
// Generated by walking, for each axis, the 4 edges parallel to it.
function makeEdgeBoxCornerOffsetData(): Float32Array {
  const data: number[] = [];
  let edgeIndex = 0;
  for (let axis = 0; axis < 3; ++axis) {
    const u = (axis + 1) % 3;
    const v = (axis + 2) % 3;
    for (let i = 0; i < 2; ++i) {
      for (let j = 0; j < 2; ++j) {
        const a = [0, 0, 0];
        a[u] = i;
        a[v] = j;
        const b = [...a];
        b[axis] = 1;
        data.push(
          a[0],
          a[1],
          a[2],
          b[0],
          b[1],
          b[2],
          EDGES_PICK_OFFSET + edgeIndex,
        );
        ++edgeIndex;
      }
    }
  }
  return Float32Array.from(data);
}

const edgeBoxCornerOffsetData = makeEdgeBoxCornerOffsetData();

// GLSL: build a column-major rotation matrix from a unit quaternion (x,y,z,w),
// matching gl-matrix `mat3.fromQuat` so CPU and GPU agree.
const glsl_quatToMat3 = `
mat3 quatToMat3(vec4 q) {
  float x = q.x, y = q.y, z = q.z, w = q.w;
  float x2 = x + x, y2 = y + y, z2 = z + z;
  float xx = x * x2, xy = x * y2, xz = x * z2;
  float yy = y * y2, yz = y * z2, zz = z * z2;
  float wx = w * x2, wy = w * y2, wz = w * z2;
  return mat3(
    1.0 - (yy + zz), xy + wz,         xz - wy,
    xy - wz,         1.0 - (xx + zz), yz + wx,
    xz + wy,         yz - wx,         1.0 - (xx + yy));
}
`;

abstract class RenderHelper extends AnnotationRenderHelper {
  defineShader(builder: ShaderBuilder) {
    defineVertexId(builder);
    const { rank } = this;
    // center + extents in model coordinates (getBounds0/getBounds1).
    defineVectorArrayVertexShaderInput(
      builder,
      "float",
      WebGL2RenderingContext.FLOAT,
      false,
      "Bounds",
      rank,
      2,
    );
    // orientation quaternion [x, y, z, w] (getOrientation0).
    defineVectorArrayVertexShaderInput(
      builder,
      "float",
      WebGL2RenderingContext.FLOAT,
      false,
      "Orientation",
      4,
      1,
    );
    // Per-box visibility flag (getVisible0), 1 = shown, 0 = hidden.
    defineVectorArrayVertexShaderInput(
      builder,
      "float",
      WebGL2RenderingContext.FLOAT,
      false,
      "Visible",
      1,
      1,
    );
    builder.addVertexCode(glsl_quatToMat3);
    // Shared accessors for the box frame in the 3D display subspace.
    builder.addVertexCode(`
vec3 obbSubCenter() {
  float center[${rank}] = getBounds0();
  return projectModelVectorToSubspace(center);
}
vec3 obbHalfExtents() {
  float extents[${rank}] = getBounds1();
  return abs(projectModelVectorToSubspace(extents)) * 0.5;
}
mat3 obbRotation() {
  float q[4] = getOrientation0();
  return quatToMat3(vec4(q[0], q[1], q[2], q[3]));
}
// Rotated subspace position for a corner whose offset components are in {0, 1}.
vec3 orientedCornerPosition(vec3 cornerOffset) {
  vec3 localOffset = (cornerOffset * 2.0 - 1.0) * obbHalfExtents();
  return obbSubCenter() + obbRotation() * localOffset;
}
// Per-box visibility: a hidden box (visible flag < 0.5) is culled everywhere
// (wireframe, cross-section, and gizmo). Every OBB shader references
// obbVisible() up front, which keeps the per-instance Visible binder enabled
// (otherwise its attribute would be optimized out and enabled at location -1).
bool obbVisible() {
  float v[1] = getVisible0();
  return v[0] >= 0.5;
}
// Move a vertex outside the clip volume so its primitive is discarded. Shared
// by the visibility cull and the gizmo handle culling below.
void cullVertex() { gl_Position = vec4(2.0, 0.0, 0.0, 1.0); }
float ng_lineWidth;
`);
  }

  private vertexIdHelper = this.registerDisposer(VertexIdHelper.get(this.gl));

  enable(
    shaderGetter: AnnotationShaderGetter,
    context: AnnotationRenderContext,
    callback: (shader: ShaderProgram) => void,
    vertexIdCount = 256, // default; increase later if not enough for mesh
  ) {
    super.enable(shaderGetter, context, (shader) => {
      const { gl } = this;
      const boundsBinder = shader.vertexShaderInputBinders.Bounds;
      const orientationBinder = shader.vertexShaderInputBinders.Orientation;
      const visibleBinder = shader.vertexShaderInputBinders.Visible;
      boundsBinder.enable(1);
      orientationBinder.enable(1);
      visibleBinder.enable(1);
      gl.bindBuffer(WebGL2RenderingContext.ARRAY_BUFFER, context.buffer.buffer);
      boundsBinder.bind(this.geometryDataStride, context.bufferOffset);
      orientationBinder.bind(
        this.geometryDataStride,
        context.bufferOffset + 2 * 4 * this.rank,
      );
      // Visibility flag sits immediately after the orientation quaternion.
      visibleBinder.bind(
        this.geometryDataStride,
        context.bufferOffset + 2 * 4 * this.rank + 4 * 4,
      );
      const { vertexIdHelper } = this;
      vertexIdHelper.enable(vertexIdCount);
      callback(shader);
      vertexIdHelper.disable();
      visibleBinder.disable();
      orientationBinder.disable();
      boundsBinder.disable();
    });
  }
}

class PerspectiveViewRenderHelper extends RenderHelper {
  // Reused variables
  private scratchViewModelMatrix = mat4.create();
  private scratchRegionCenter = vec3.create();
  private scratchAxisWorld = vec3.create();

  // Unit-arrow geometry (positions + normals) for the single handle.
  private arrowPosBuffer = this.registerDisposer(
    GLBuffer.fromData(this.gl, ARROW_POSITIONS),
  );
  private arrowNormalBuffer = this.registerDisposer(
    GLBuffer.fromData(this.gl, ARROW_NORMALS),
  );

  // Scale-cube geometry (positions + normals) for the 3 scale handles.
  private cubePosBuffer = this.registerDisposer(
    GLBuffer.fromData(this.gl, CUBE_POSITIONS),
  );
  private cubeNormalBuffer = this.registerDisposer(
    GLBuffer.fromData(this.gl, CUBE_NORMALS),
  );

  private edgeBoxCornerOffsetsBuffer = this.registerDisposer(
    GLBuffer.fromData(
      this.gl,
      tile2dArray(
        edgeBoxCornerOffsetData,
        7,
        1,
        VERTICES_PER_LINE,
      ),
    ),
  );

  private edgeShaderGetter = this.getDependentShader(
    "annotation/orientedBoundingBox/projection/border",
    (builder: ShaderBuilder) => {
      this.defineShader(builder);
      defineLineShader(builder);
      // uSelectedInstance: the React-selected box (-1 if none). Drives both the
      // dimming of other boxes and the gating of edge interactivity below.
      builder.addUniform("highp int", "uSelectedInstance");
      builder.addAttribute("highp vec3", "aBoxCornerOffset1");
      builder.addAttribute("highp vec4", "aBoxCornerOffset2");
      builder.setVertexMain(`
if (!obbVisible()) { cullVertex(); return; }
// always draw a box's full wireframe in 3-D
// (no subspace-clip fade, which would cull a box smaller than the data bounds).
vec3 endpointA = orientedCornerPosition(aBoxCornerOffset1);
vec3 endpointB = orientedCornerPosition(aBoxCornerOffset2.xyz);
ng_lineWidth = 1.0;
${this.invokeUserMain}
vColor = vec4(uColor, 1.0);  // region box: layer annotation color (yellow default)
// Whole-box hover highlight: brighten the entire wireframe when any part of
// this box is the hovered annotation (uSelectedIndex encodes hovered
// instance*pids + part; compare just the instance).
uint pids = ${ORIENTED_BBOX_PICK_IDS_PER_INSTANCE}u;
bool hovered =
  uSelectedIndex != 0xffffffffu && uSelectedIndex / pids == uint(gl_InstanceID);
if (hovered) {
  vColor.rgb = mix(vColor.rgb, vec3(1.0), ${HOVER_HIGHLIGHT.toFixed(2)});
}
// Dim the other boxes while one is selected (mirrors segment/instance dimming),
// but a hovered box stays at full alpha so its row-hover highlight reads clearly.
if (uSelectedInstance >= 0 && gl_InstanceID != uSelectedInstance && !hovered) {
  vColor.a *= ${NOT_SELECTED_ALPHA.toFixed(2)};
}
emitLine(uModelViewProjection * vec4(endpointA, 1.0),
         uModelViewProjection * vec4(endpointB, 1.0),
         ng_lineWidth);
// Edges drive free-rotation only on the selected box; every other box emits the
// non-interactive full-object pick id so its edges can't be grabbed/dragged.
// (Picking the box still resolves the annotation for hover, just not a handle.)
uint edgePart = (gl_InstanceID == uSelectedInstance)
  ? uint(aBoxCornerOffset2.w)
  : ${FULL_OBJECT_PICK_OFFSET}u;
vPickID = uPickID + getPickBaseOffset() + edgePart;
`);
      builder.setFragmentMain(`
emitAnnotation(vec4(vColor.rgb, vColor.a * getLineAlpha()));
`);
    },
  );

  // The translate tripod: 3 shaded arrows (shaft + cone) along the box's local
  // axes, colored per axis (X red, Y green, Z blue) and sized to a constant
  // on-screen size. The arrow (axis) index comes from gl_VertexID (the unit mesh
  // is tiled once per arrow).
  private arrowShaderGetter = this.getDependentShader(
    "annotation/orientedBoundingBox/projection/arrow",
    (builder: ShaderBuilder) => {
      this.defineShader(builder);
      this.defineGizmoVisibility(builder);
      builder.addAttribute("highp vec3", "aArrowPos");
      builder.addAttribute("highp vec3", "aArrowNormal");
      builder.addUniform("highp vec3", "uGizmoAxisWorld");
      builder.addVertexCode(glsl_axisColor);
      builder.setVertexMain(`
int axis = gl_VertexID / ${VERTS_PER_ARROW};
// Arrows show only on the selected box, and not while it's being dragged (a
// tripod drag shows a guide line instead). Hidden boxes show no gizmo at all.
if (!obbVisible() || gizmoInstanceHidden() || onDraggedInstance()) { cullVertex(); return; }
vec3 subCenter = obbSubCenter();
// Keep the extents/orientation attributes referenced (arrows are world-aligned
// and don't use them) so their per-instance binders aren't enabled at location -1.
vec3 keep = obbHalfExtents() * 0.0 + obbRotation() * vec3(0.0);
// Translate arrows stay aligned to the display axes: they do NOT rotate with the
// box orientation (only the rings and box wireframe show the orientation).
vec3 dir = axisUnit(axis);
// Orthonormal basis around the axis for the arrow's radial directions.
vec3 t1 = normalize(abs(dir.x) < 0.9
  ? cross(vec3(1.0, 0.0, 0.0), dir)
  : cross(vec3(0.0, 1.0, 0.0), dir));
vec3 t2 = cross(dir, t1);
// Build the arrow in an isotropic "gizmo space" (unit length, radius 0.06,
// started just past the center), then scale each subspace component by its
// per-axis world size. This undistorts BOTH the length and the cross-section, so
// the arrow is a fixed on-screen size and shape regardless of voxel anisotropy.
vec3 gizmoOffset =
    (aArrowPos.x * 0.06) * t1
  + (aArrowPos.y * 0.06) * t2
  + (${GIZMO_ARROW_START_FRACTION} + aArrowPos.z) * dir;
vec3 worldPos = subCenter + keep + gizmoOffset * uGizmoAxisWorld;
vec3 nrm = normalize(
  aArrowNormal.x * t1 + aArrowNormal.y * t2 + aArrowNormal.z * dir);
vec3 lightDir = normalize(vec3(0.4, 0.6, 0.9));
float shade = 0.45 + 0.55 * abs(dot(nrm, lightDir));
vColor = vec4(axisColor(axis) * shade, 1.0);
gl_Position = uModelViewProjection * vec4(worldPos, 1.0);
// Pick id only (no highlight): the box hover-highlight must not tint the gizmo.
vPickID = uPickID + getPickBaseOffset() + ${TRANSLATE_AXIS_PICK_OFFSET}u + uint(axis);
`);
      builder.setFragmentMain(`
emitAnnotation(vColor);
`);
    },
  );

  // Scale cubes: one shaded cube per axis, colored per axis, a constant on-screen
  // size. Unlike the arrows, the cubes rotate WITH the box (orientation R) so each
  // sits on the box's rotated face — the face that actually grows when scaling
  // that local axis.
  private cubeShaderGetter = this.getDependentShader(
    "annotation/orientedBoundingBox/projection/cube",
    (builder: ShaderBuilder) => {
      this.defineShader(builder);
      this.defineGizmoVisibility(builder);
      builder.addAttribute("highp vec3", "aCubePos");
      builder.addAttribute("highp vec3", "aCubeNormal");
      builder.addUniform("highp vec3", "uGizmoAxisWorld");
      builder.addVertexCode(glsl_axisColor);
      builder.setVertexMain(`
int axis = gl_VertexID / ${VERTS_PER_CUBE};
// Cubes show only on the selected box; mid-scale-drag, only the dragged cube.
if (!obbVisible() || gizmoInstanceHidden() || gizmoPartHidden(${SCALE_AXIS_PICK_OFFSET} + axis)) { cullVertex(); return; }
vec3 subCenter = obbSubCenter();
vec3 keep = obbHalfExtents() * 0.0;  // keep the Bounds1 (extents) binder referenced
mat3 R = obbRotation();
vec3 dir = axisUnit(axis);
vec3 t1 = normalize(abs(dir.x) < 0.9
  ? cross(vec3(1.0, 0.0, 0.0), dir)
  : cross(vec3(0.0, 1.0, 0.0), dir));
vec3 t2 = cross(dir, t1);
// Build the cube in isotropic gizmo space (sitting CUBE_CENTER_Z out along the
// axis), rotate by R so it tracks the box's rotated face — the face that
// actually grows when scaling that local axis — then scale per-axis for a
// constant on-screen size (anisotropy corrected after the rotation, as the
// rings do).
vec3 gizmoOffset =
    (aCubePos.x * ${CUBE_EDGE}) * t1
  + (aCubePos.y * ${CUBE_EDGE}) * t2
  + (${CUBE_CENTER_Z} + aCubePos.z * ${CUBE_EDGE}) * dir;
vec3 worldPos = subCenter + keep + (R * gizmoOffset) * uGizmoAxisWorld;
vec3 nrm = normalize(
  R * (aCubeNormal.x * t1 + aCubeNormal.y * t2 + aCubeNormal.z * dir));
vec3 lightDir = normalize(vec3(0.4, 0.6, 0.9));
float shade = 0.45 + 0.55 * abs(dot(nrm, lightDir));
vColor = vec4(axisColor(axis) * shade, 1.0);
gl_Position = uModelViewProjection * vec4(worldPos, 1.0);
vPickID = uPickID + getPickBaseOffset() + ${SCALE_AXIS_PICK_OFFSET}u + uint(axis);
`);
      builder.setFragmentMain(`
emitAnnotation(vColor);
`);
    },
  );

  // Rotation rings: 3 segmented line loops (the box's bounding-sphere great
  // circles), colored per local axis. The ring/segment indices come from
  // gl_VertexID.
  private ringShaderGetter = this.getDependentShader(
    "annotation/orientedBoundingBox/projection/rings",
    (builder: ShaderBuilder) => {
      this.defineShader(builder);
      this.defineGizmoVisibility(builder);
      defineLineShader(builder);
      builder.addUniform("highp vec3", "uGizmoAxisWorld");
      builder.addVertexCode(glsl_axisColor);
      builder.setVertexMain(`
int lineIndex = gl_VertexID / ${VERTICES_PER_LINE};
int ring = lineIndex / ${RING_SEGMENTS};
int seg = lineIndex - ring * ${RING_SEGMENTS};
// Rings show only on the selected box; mid-rotate-drag, only the dragged ring.
if (!obbVisible() || gizmoInstanceHidden() || gizmoPartHidden(${ROTATE_RING_PICK_OFFSET} + ring)) { cullVertex(); return; }
int u = (ring + 1) % 3;
int v = (ring + 2) % 3;
vec3 subCenter = obbSubCenter();
// Keep the extents (Bounds1) attribute referenced: rings use the center and
// orientation but not the extents, so without this its per-instance binder
// would be enabled at location -1 (WebGL INVALID_VALUE in enable()).
subCenter += obbHalfExtents() * 0.0;
mat3 R = obbRotation();
// Build a unit circle in the box's rotated axis plane in ISOTROPIC gizmo space
// (rotate by R first), THEN scale each subspace component by its per-axis world
// size. Correcting anisotropy after the rotation keeps the ring a true circle in
// every orientation (correcting before R would let anisotropy warp it as the box
// turns). Radius is half the arrow's, a constant on-screen size.
float a0 = 6.28318530718 * float(seg) / float(${RING_SEGMENTS});
float a1 = 6.28318530718 * float(seg + 1) / float(${RING_SEGMENTS});
vec3 g0 = 0.5 * (cos(a0) * axisUnit(u) + sin(a0) * axisUnit(v));
vec3 g1 = 0.5 * (cos(a1) * axisUnit(u) + sin(a1) * axisUnit(v));
vec3 l0 = (R * g0) * uGizmoAxisWorld;
vec3 l1 = (R * g1) * uGizmoAxisWorld;
ng_lineWidth = 1.5;
vColor = vec4(axisColor(ring), 1.0);
emitLine(uModelViewProjection * vec4(subCenter + l0, 1.0),
         uModelViewProjection * vec4(subCenter + l1, 1.0),
         ng_lineWidth);
vPickID = uPickID + getPickBaseOffset() + ${ROTATE_RING_PICK_OFFSET}u + uint(ring);
`);
      builder.setFragmentMain(`
emitAnnotation(vec4(vColor.rgb, getLineAlpha()));
`);
    },
  );

  // Center "free translate" ball — a billboarded circle, already screen-sized.
  private centerBallShaderGetter = this.getDependentShader(
    "annotation/orientedBoundingBox/projection/centerBall",
    (builder: ShaderBuilder) => {
      this.defineShader(builder);
      this.defineGizmoVisibility(builder);
      defineCircleShader(builder, this.targetIsSliceView);
      builder.setVertexMain(`
// The center ball shows only on the selected box.
if (!obbVisible() || gizmoInstanceHidden()) { cullVertex(); return; }
// Reference the box rotation/extents so their per-instance attributes are not
// optimized out (the shared binder would otherwise enable location -1).
vec3 keep = obbHalfExtents() + obbRotation() * vec3(0.0);
vColor = vec4(0.6, 0.6, 0.6, 1.0);
emitCircle(uModelViewProjection * vec4(obbSubCenter() + keep * 0.0, 1.0),
           ${CENTER_BALL_DIAMETER}.0, 0.0);
vPickID = uPickID + getPickBaseOffset() + ${CENTER_BALL_PICK_OFFSET}u;
`);
      builder.setFragmentMain(`
vec4 borderColor = vec4(0.0, 0.0, 0.0, 1.0);
emitAnnotation(getCircleColor(vColor, borderColor));
`);
    },
  );

  // Axis guide line shown during a single-axis (tripod) translate drag: a line
  // through the box center along the dragged display axis, clamped to the data
  // box's extent on that axis (uGuideLo/uGuideHi in model coordinates).
  private guideLineShaderGetter = this.getDependentShader(
    "annotation/orientedBoundingBox/projection/guideLine",
    (builder: ShaderBuilder) => {
      const { rank } = this;
      this.defineShader(builder);
      this.defineGizmoVisibility(builder);
      defineLineShader(builder);
      builder.addVertexCode(glsl_axisColor);
      builder.addUniform("highp int", "uGuideAxis");
      builder.addUniform("highp float", "uGuideLo");
      builder.addUniform("highp float", "uGuideHi");
      builder.setVertexMain(`
// The guide line belongs to the box being dragged; never show it on others.
if (!obbVisible() || gl_InstanceID != uDraggedInstance) { cullVertex(); return; }
// Keep extents/orientation attributes referenced so their binders aren't -1.
vec3 keep = obbHalfExtents() * 0.0 + obbRotation() * vec3(0.0);
float c[${rank}] = getBounds0();
float lo[${rank}];
float hi[${rank}];
for (int i = 0; i < ${rank}; ++i) {
  lo[i] = (i == uGuideAxis) ? uGuideLo : c[i];
  hi[i] = (i == uGuideAxis) ? uGuideHi : c[i];
}
vec3 pa = projectModelVectorToSubspace(lo) + keep;
vec3 pb = projectModelVectorToSubspace(hi);
ng_lineWidth = 1.0;
vColor = vec4(axisColor(uGuideAxis), 0.85);
emitLine(uModelViewProjection * vec4(pa, 1.0),
         uModelViewProjection * vec4(pb, 1.0),
         ng_lineWidth);
vPickID = uPickID + getPickBaseOffset();
`);
      builder.setFragmentMain(`
emitAnnotation(vec4(vColor.rgb, vColor.a * getLineAlpha()));
`);
    },
  );

  // Gizmo handle visibility, shared by the arrow / cube / ring / center-ball /
  // guide-line shaders.
  //  - uSelectedInstance: the only box whose handles render (the selected box,
  //    or the dragged box mid-drag). -1 hides all handles (nothing selected).
  //  - uDraggedInstance / uDraggedPart: during a drag the dragged box collapses
  //    to just its active handle (the part being dragged).
  private defineGizmoVisibility(builder: ShaderBuilder) {
    builder.addUniform("highp int", "uSelectedInstance");
    builder.addUniform("highp int", "uDraggedInstance");
    builder.addUniform("highp int", "uDraggedPart");
    builder.addVertexCode(`
bool gizmoInstanceHidden() { return gl_InstanceID != uSelectedInstance; }
bool onDraggedInstance() {
  return uDraggedInstance >= 0 && gl_InstanceID == uDraggedInstance;
}
bool gizmoPartHidden(int thisPart) {
  return onDraggedInstance() && thisPart != uDraggedPart;
}
`);
  }

  private setGizmoVisibility(
    shader: ShaderProgram,
    selectedInstance: number,
    draggedInstance: number,
    draggedPart: number,
  ) {
    this.gl.uniform1i(shader.uniform("uSelectedInstance"), selectedInstance);
    this.gl.uniform1i(shader.uniform("uDraggedInstance"), draggedInstance);
    this.gl.uniform1i(shader.uniform("uDraggedPart"), draggedPart);
  }

  // Compute and upload uGizmoAxisWorld: per display-axis world length that
  // projects to GIZMO_SIZE on screen. Uses the true view-space length of each
  // axis (|view·model·eₖ|, rotation-invariant) and the box depth (clip.w), both
  // view-independent, so handles keep a constant on-screen size as the camera
  // orbits and each axis is sized correctly under anisotropic voxels.
  private setGizmoAxisWorld(
    shader: ShaderProgram,
    context: AnnotationRenderContext,
  ) {
    const loc = shader.uniform("uGizmoAxisWorld");
    if (loc === null) return;
    const pp = context.renderContext.projectionParameters;
    const sm = context.subspaceMatrix;
    const mvp = context.modelViewProjectionMatrix;

    // Perspective depth (clip.w) of the box center: handles are scaled by this
    // so they keep a constant on-screen size regardless of distance.
    const center = this.regionCenterInSubspace(context, this.scratchRegionCenter);
    const clipW =
      Math.abs(
        mvp[3] * center[0] + mvp[7] * center[1] + mvp[11] * center[2] + mvp[15],
      ) || 1;

    // For each display axis, the world length that projects to GIZMO_SIZE on
    // screen. |view·model·eₖ| (column k of view·model) is the axis's view-space
    // scale, rotation-invariant; the perspective NDC-per-world factor is then
    // (fy · scale) / clip.w.
    const viewModel = this.scratchViewModelMatrix;
    const axisWorld = this.scratchAxisWorld;
    const fy = Math.abs(pp.projectionMat[5]);
    mat4.multiply(viewModel, pp.viewMatrix, context.renderSubspaceModelMatrix);
    for (let k = 0; k < 3; ++k) {
      const ndcPerWorld = (fy * columnXyzLength(viewModel, k)) / clipW;
      axisWorld[k] = GIZMO_SIZE / Math.max(ndcPerWorld, 1e-6);
    }
    this.gl.uniform3f(loc, axisWorld[0], axisWorld[1], axisWorld[2]);

    // Orthonormal subspace→view rotation: the columns of view·model are the
    // subspace axes in view space (lengths = the per-axis world scales above);
    // normalizing them drops the anisotropic scale and leaves a pure rotation.
    const viewRot = new Float32Array(9);
    for (let k = 0; k < 3; ++k) {
      const len = columnXyzLength(viewModel, k) || 1;
      viewRot[k * 3] = viewModel[k * 4] / len;
      viewRot[k * 3 + 1] = viewModel[k * 4 + 1] / len;
      viewRot[k * 3 + 2] = viewModel[k * 4 + 2] / len;
    }

    setGizmoProjection({
      mvp: Float32Array.from(mvp),
      subspaceMatrix: Float32Array.from(sm),
      aspect: pp.width / pp.height,
      axisWorld: Float32Array.from(axisWorld),
      viewRot,
    });
  }

  private regionCenterInSubspace(
    context: AnnotationRenderContext,
    out: vec3,
  ): vec3 {
    const bounds = getRegionDataBounds();
    const sm = context.subspaceMatrix;
    vec3.set(out, 0, 0, 0);
    for (let i = 0; i < this.rank; ++i) {
      const lo = bounds?.lower[i];
      const hi = bounds?.upper[i];
      const ci =
        lo !== undefined &&
          hi !== undefined &&
          Number.isFinite(lo) &&
          Number.isFinite(hi)
          ? (lo + hi) / 2
          : 0;
      for (let j = 0; j < 3; ++j) out[j] += sm[i * 3 + j] * ci;
    }
    return out;
  }

  drawEdges(context: AnnotationRenderContext) {
    const { gl } = this;
    this.enable(this.edgeShaderGetter, context, (shader) => {
      // The React-selected box (drag-aware not needed here: edges of the
      // selected box stay interactive while dragging it).
      gl.uniform1i(shader.uniform("uSelectedInstance"), context.selectedInstance);
      const aBoxCornerOffset1 = shader.attribute("aBoxCornerOffset1");
      const aBoxCornerOffset2 = shader.attribute("aBoxCornerOffset2");
      const vertexStride = 4 * 7;
      this.edgeBoxCornerOffsetsBuffer.bindToVertexAttrib(
        aBoxCornerOffset1,
        3,
        WebGL2RenderingContext.FLOAT,
        false,
        vertexStride,
        0,
      );
      this.edgeBoxCornerOffsetsBuffer.bindToVertexAttrib(
        aBoxCornerOffset2,
        4,
        WebGL2RenderingContext.FLOAT,
        false,
        vertexStride,
        4 * 3,
      );
      initializeLineShader(
        shader,
        context.renderContext.projectionParameters,
        1,
      );
      drawLines(gl, EDGES_PER_BOX, context.count);
      gl.disableVertexAttribArray(aBoxCornerOffset1);
      gl.disableVertexAttribArray(aBoxCornerOffset2);
    });
  }

  // Draw a solid, two-sided instanced mesh (the arrow / cube handles): bind its
  // position and normal buffers, disable backface culling so both sides show,
  // draw, then restore state.
  private drawSolidMesh(
    context: AnnotationRenderContext,
    shaderGetter: AnnotationShaderGetter,
    positionBuffer: GLBuffer,
    normalBuffer: GLBuffer,
    positionAttribute: string,
    normalAttribute: string,
    vertexCount: number,
    selectedInstance: number,
    draggedInstance: number,
    draggedPart: number,
  ) {
    const { gl } = this;
    this.enable(
      shaderGetter,
      context,
      (shader) => {
        this.setGizmoAxisWorld(shader, context);
        this.setGizmoVisibility(
          shader,
          selectedInstance,
          draggedInstance,
          draggedPart,
        );
        const position = shader.attribute(positionAttribute);
        const normal = shader.attribute(normalAttribute);
        positionBuffer.bindToVertexAttrib(position, 3);
        normalBuffer.bindToVertexAttrib(normal, 3);
        const wasCulling = gl.isEnabled(WebGL2RenderingContext.CULL_FACE);
        gl.disable(WebGL2RenderingContext.CULL_FACE);
        drawArraysInstanced(
          gl,
          WebGL2RenderingContext.TRIANGLES,
          0,
          vertexCount,
          context.count,
        );
        if (wasCulling) gl.enable(WebGL2RenderingContext.CULL_FACE);
        gl.disableVertexAttribArray(position);
        gl.disableVertexAttribArray(normal);
      },
      vertexCount,
    );
  }

  drawArrow(
    context: AnnotationRenderContext,
    selectedInstance: number,
    draggedInstance: number,
    draggedPart: number,
  ) {
    this.drawSolidMesh(
      context,
      this.arrowShaderGetter,
      this.arrowPosBuffer,
      this.arrowNormalBuffer,
      "aArrowPos",
      "aArrowNormal",
      ARROW_DRAW_VERTEX_COUNT,
      selectedInstance,
      draggedInstance,
      draggedPart,
    );
  }

  drawCubes(
    context: AnnotationRenderContext,
    selectedInstance: number,
    draggedInstance: number,
    draggedPart: number,
  ) {
    this.drawSolidMesh(
      context,
      this.cubeShaderGetter,
      this.cubePosBuffer,
      this.cubeNormalBuffer,
      "aCubePos",
      "aCubeNormal",
      CUBE_DRAW_VERTEX_COUNT,
      selectedInstance,
      draggedInstance,
      draggedPart,
    );
  }

  drawRings(
    context: AnnotationRenderContext,
    selectedInstance: number,
    draggedInstance: number,
    draggedPart: number,
  ) {
    const { gl } = this;
    this.enable(
      this.ringShaderGetter,
      context,
      (shader) => {
        this.setGizmoAxisWorld(shader, context);
        this.setGizmoVisibility(
          shader,
          selectedInstance,
          draggedInstance,
          draggedPart,
        );
        initializeLineShader(
          shader,
          context.renderContext.projectionParameters,
          1,
        );
        drawLines(gl, N_RING_LINES, context.count);
      },
      N_RING_LINES * VERTICES_PER_LINE,
    );
  }

  drawCenterBall(context: AnnotationRenderContext, selectedInstance: number) {
    this.enable(this.centerBallShaderGetter, context, (shader) => {
      this.setGizmoVisibility(shader, selectedInstance, -1, -1);
      initializeCircleShader(
        shader,
        context.renderContext.projectionParameters,
        { featherWidthInPixels: 0 },
      );
      drawCircles(shader.gl, 1, context.count);
    });
  }

  drawGuideLine(
    context: AnnotationRenderContext,
    axis: number,
    selectedInstance: number,
    draggedInstance: number,
  ) {
    const { gl } = this;
    const bounds = getRegionDataBounds();
    let lo = -1e6;
    let hi = 1e6;
    if (bounds) {
      const l = bounds.lower[axis];
      const h = bounds.upper[axis];
      if (Number.isFinite(l) && Number.isFinite(h)) {
        lo = l;
        hi = h;
      }
    }
    this.enable(this.guideLineShaderGetter, context, (shader) => {
      this.setGizmoVisibility(shader, selectedInstance, draggedInstance, -1);
      gl.uniform1i(shader.uniform("uGuideAxis"), axis);
      gl.uniform1f(shader.uniform("uGuideLo"), lo);
      gl.uniform1f(shader.uniform("uGuideHi"), hi);
      initializeLineShader(
        shader,
        context.renderContext.projectionParameters,
        0,
      );
      drawLines(gl, 1, context.count);
    });
  }

  draw(context: AnnotationRenderContext) {
    const { gl } = this;
    // The box wireframe always draws (every box); only the interactive handles
    // below are gated to the selected box.
    this.drawEdges(context);

    const dragging = getGizmoDragStartNdc() !== null;
    const pids = ORIENTED_BBOX_PICK_IDS_PER_INSTANCE;
    // selectedIndex is 0xffffffff when nothing is hovered/picked; guard against
    // it so the drag math doesn't turn the sentinel into a garbage instance.
    const havePick = dragging && context.selectedIndex !== 0xffffffff;
    const draggedInstance = havePick
      ? Math.floor(context.selectedIndex / pids)
      : -1;
    const draggedPart = havePick ? context.selectedIndex % pids : -1;
    // Handles render only on the selected box; while dragging, that's the
    // dragged box and it collapses to just the active part.
    const selectedInstance = dragging
      ? draggedInstance
      : context.selectedInstance;
    // No selected/dragged box in this chunk: only the wireframe above renders.
    if (selectedInstance < 0) return;

    // Remove depth rendering for gizmo so it is always visible and never occluded
    gl.disable(WebGL2RenderingContext.DEPTH_TEST);
    gl.depthMask(false);
    try {
      this.drawRings(context, selectedInstance, draggedInstance, draggedPart);
      this.drawArrow(context, selectedInstance, draggedInstance, draggedPart);
      this.drawCubes(context, selectedInstance, draggedInstance, draggedPart);
      // A tripod translate replaces the dragged box's arrow with a long axis
      // guide line (shown only on that box).
      if (dragging && classifyGizmoPart(draggedPart).kind === "translate") {
        this.drawGuideLine(
          context,
          draggedPart - TRANSLATE_AXIS_PICK_OFFSET,
          selectedInstance,
          draggedInstance,
        );
      }
      this.drawCenterBall(context, selectedInstance);
    } finally {
      gl.enable(WebGL2RenderingContext.DEPTH_TEST);
      gl.depthMask(true);
    }
  }
}

/**
 * Slice-view rendering draws a non-interactable cross section of the box
 */
class SliceViewRenderHelper extends RenderHelper {
  // Reused scratch for the per-draw slice-plane computation (see setPlane),
  // allocated once and consumed synchronously within a single call.
  private scratchPlaneNormal = vec3.create();
  private scratchPlaneCenter = vec3.create();

  private defineCrossSection(builder: ShaderBuilder) {
    const { rank } = this;
    builder.addUniform("highp vec3", "uPlaneNormal");
    builder.addUniform("highp float", "uPlaneDistance");
    builder.addUniform("highp vec3", "uVertexBasePosition", 8);
    // All 8 front-vertex orderings (24 candidate edges each), indexed in-shader.
    builder.addUniform("highp ivec2", "uVertexIndexAll", 8 * 24);
    builder.addInitializer((shader) => {
      shader.gl.uniform3fv(
        shader.uniform("uVertexBasePosition"),
        vertexBasePositions,
      );
      shader.gl.uniform2iv(
        shader.uniform("uVertexIndexAll"),
        boundingBoxCrossSectionVertexIndices,
      );
    });
    builder.addVertexCode(`
vec3 orientedCrossSectionVertex(int vertexIndex) {
  float center[${rank}] = getBounds0();
  float extents[${rank}] = getBounds1();
  vec3 subCenter = projectModelVectorToSubspace(center);
  vec3 subExtents = projectModelVectorToSubspace(extents);
  vec3 halfExtents = abs(subExtents) * 0.5;
  float q[4] = getOrientation0();
  mat3 R = quatToMat3(vec4(q[0], q[1], q[2], q[3]));
  // Transform the slice plane into the box's local (axis-aligned) frame.
  vec3 localN = uPlaneNormal * R;                   // R^T * N (R is orthonormal)
  float localD = uPlaneDistance - dot(subCenter, uPlaneNormal);
  int fv = 0;
  if (localN.x < 0.0) fv += 1;
  if (localN.y < 0.0) fv += 2;
  if (localN.z < 0.0) fv += 4;
  for (int e = 0; e < 4; ++e) {
    ivec2 vidx = uVertexIndexAll[fv * 24 + vertexIndex * 4 + e];
    vec3 lA = (uVertexBasePosition[vidx.x] * 2.0 - 1.0) * halfExtents;
    vec3 lB = (uVertexBasePosition[vidx.y] * 2.0 - 1.0) * halfExtents;
    vec3 dir = lB - lA;
    float denom = dot(dir, localN);
    if (abs(denom) > 1e-3) {
      float lambda = (localD - dot(lA, localN)) / denom;
      if (lambda >= -1e-3 && lambda <= 1.0 + 1e-3) {
        lambda = clamp(lambda, 0.0, 1.0);
        return subCenter + R * (lA + lambda * dir);
      }
    }
  }
  return subCenter;  // plane misses box -> degenerate (zero-area) polygon
}
`);
  }

  private faceShaderGetter = this.getDependentShader(
    "annotation/orientedBoundingBox/crossSection/border",
    (builder: ShaderBuilder) => {
      this.defineShader(builder);
      this.defineCrossSection(builder);
      defineLineShader(builder);
      // Selected box (-1 if none) so the cross-section dims the others to match
      // the 3-D wireframe.
      builder.addUniform("highp int", "uSelectedInstance");
      builder.setVertexMain(`
if (!obbVisible()) { cullVertex(); return; }
int vertexIndex1 = gl_VertexID / ${VERTICES_PER_LINE};
int vertexIndex2 = vertexIndex1 == 5 ? 0 : vertexIndex1 + 1;
vec3 p1 = orientedCrossSectionVertex(vertexIndex1);
vec3 p2 = orientedCrossSectionVertex(vertexIndex2);
ng_lineWidth = 1.0;
${this.invokeUserMain}
vColor = vec4(uColor, 1.0);  // region box: layer annotation color (yellow default)
// Match the 3-D wireframe: brighten the hovered box, dim the others while one is
// selected, and keep a hovered box at full alpha.
uint pids = ${ORIENTED_BBOX_PICK_IDS_PER_INSTANCE}u;
bool hovered =
  uSelectedIndex != 0xffffffffu && uSelectedIndex / pids == uint(gl_InstanceID);
if (hovered) {
  vColor.rgb = mix(vColor.rgb, vec3(1.0), ${HOVER_HIGHLIGHT.toFixed(2)});
}
if (uSelectedInstance >= 0 && gl_InstanceID != uSelectedInstance && !hovered) {
  vColor.a *= ${NOT_SELECTED_ALPHA.toFixed(2)};
}
emitLine(uModelViewProjection * vec4(p1, 1.0),
         uModelViewProjection * vec4(p2, 1.0),
         ng_lineWidth);
// Region gizmo editing is 3-D only: emit a background pick ID so the slice
// cross-section is purely visual (not hoverable, selectable, or draggable in 2-D).
vPickID = 0u;
`);
      builder.setFragmentMain(`
emitAnnotation(vec4(vColor.rgb, vColor.a * getLineAlpha()));
`);
    },
  );

  private setPlane(
    shader: ShaderProgram,
    context: AnnotationRenderContext & {
      renderContext: SliceViewPanelRenderContext;
    },
  ) {
    const { gl } = this;
    const projectionParameters =
      context.renderContext.sliceView.projectionParameters.value;
    const localPlaneNormal = transformVectorByMat4Transpose(
      this.scratchPlaneNormal,
      projectionParameters.viewportNormalInGlobalCoordinates,
      context.renderSubspaceModelMatrix,
    );
    vec3.normalize(localPlaneNormal, localPlaneNormal);
    const planeDistance = vec3.dot(
      vec3.transformMat4(
        this.scratchPlaneCenter,
        projectionParameters.centerDataPosition,
        context.renderSubspaceInvModelMatrix,
      ),
      localPlaneNormal,
    );
    gl.uniform3fv(shader.uniform("uPlaneNormal"), localPlaneNormal);
    gl.uniform1f(shader.uniform("uPlaneDistance"), planeDistance);
  }

  draw(
    context: AnnotationRenderContext & {
      renderContext: SliceViewPanelRenderContext;
    },
  ) {
    this.enable(this.faceShaderGetter, context, (shader) => {
      this.setPlane(shader, context);
      this.gl.uniform1i(
        shader.uniform("uSelectedInstance"),
        context.selectedInstance,
      );
      initializeLineShader(
        shader,
        context.renderContext.projectionParameters,
        1.0,
      );
      drawLines(shader.gl, 6, context.count);
    });
  }
}

const MIN_EXTENT = 1;

function quatOf(orientation: Float32Array): quat {
  return orientation as unknown as quat;
}

function toVec3(point: ArrayLike<number>): vec3 {
  return vec3.fromValues(point[0], point[1], point[2]);
}

function boundingSphereRadius(extents: Float32Array): number {
  const x = extents[0] ?? 0;
  const y = extents[1] ?? 0;
  const z = extents[2] ?? 0;
  return 0.5 * Math.sqrt(x * x + y * y + z * z);
}

function worldAxis(rotation: mat3, k: number): vec3 {
  return vec3.normalize(
    vec3.create(),
    vec3.fromValues(rotation[k * 3], rotation[k * 3 + 1], rotation[k * 3 + 2]),
  );
}

// Angle to rotate a ring by, using the projected-ellipse tangent at the
// clicked point. Falls back to a world estimate without a projection.
function ringRotationAngle(
  center: Float32Array,
  rotation: mat3,
  ringAxis: number,
  grabbedPoint: vec3,
  draggedPoint: Float32Array,
  sphereRadius: number,
): number {
  const rank = center.length;
  const projection = getGizmoProjection();
  const grabNdc = getGizmoDragStartNdc();

  // Fallback without a captured projection: approximate from the world-space
  // ring tangent at the grabbed point.
  if (projection === null || grabNdc === null) {
    const axis = worldAxis(rotation, ringAxis);
    const tangent = vec3.normalize(
      vec3.create(),
      vec3.cross(
        vec3.create(),
        axis,
        vec3.sub(vec3.create(), grabbedPoint, toVec3(center)),
      ),
    );
    const drag = vec3.sub(vec3.create(), toVec3(draggedPoint), grabbedPoint);
    return vec3.dot(drag, tangent) / Math.max(sphereRadius, 1e-6);
  }

  const { mvp, subspaceMatrix, aspect, axisWorld } = projection;
  const ringU = worldAxis(rotation, (ringAxis + 1) % 3);
  const ringV = worldAxis(rotation, (ringAxis + 2) % 3);

  // Project a model/data point into the render subspace.
  const toSubspace = (point: ArrayLike<number>): number[] => {
    const sub = [0, 0, 0];
    for (let i = 0; i < rank; ++i) {
      sub[0] += subspaceMatrix[i * 3] * point[i];
      sub[1] += subspaceMatrix[i * 3 + 1] * point[i];
      sub[2] += subspaceMatrix[i * 3 + 2] * point[i];
    }
    return sub;
  };
  // Project a subspace point to aspect-corrected, y-up screen coords.
  const toScreen = (sub: number[]) => {
    const x = mvp[0] * sub[0] + mvp[4] * sub[1] + mvp[8] * sub[2] + mvp[12];
    const y = mvp[1] * sub[0] + mvp[5] * sub[1] + mvp[9] * sub[2] + mvp[13];
    const w = mvp[3] * sub[0] + mvp[7] * sub[1] + mvp[11] * sub[2] + mvp[15];
    return { x: (x / w) * aspect, y: y / w };
  };

  // A point in the ring's plane: subspaceCenter + (u·eᵤ + v·eᵥ)·axisWorld,
  // matching the ring shader. (u, v) = ½(cosφ, sinφ) traces the ring itself.
  const subspaceCenter = toSubspace(center);
  const ringPlanePoint = (u: number, v: number): number[] => [
    subspaceCenter[0] + (u * ringU[0] + v * ringV[0]) * axisWorld[0],
    subspaceCenter[1] + (u * ringU[1] + v * ringV[1]) * axisWorld[1],
    subspaceCenter[2] + (u * ringU[2] + v * ringV[2]) * axisWorld[2],
  ];

  // Find the ring's parametric angle whose projection is nearest the grab point.
  const grabX = grabNdc.x * aspect;
  const grabY = grabNdc.y;
  const SAMPLES = 64;
  let nearestPhi = 0;
  let nearestDistSq = Infinity;
  for (let i = 0; i < SAMPLES; ++i) {
    const phi = (2 * Math.PI * i) / SAMPLES;
    const screen = toScreen(
      ringPlanePoint(0.5 * Math.cos(phi), 0.5 * Math.sin(phi)),
    );
    const distSq = (screen.x - grabX) ** 2 + (screen.y - grabY) ** 2;
    if (distSq < nearestDistSq) {
      nearestDistSq = distSq;
      nearestPhi = phi;
    }
  }

  // Screen-space ring tangent (dScreen/dφ) at the grabbed angle: project the
  // ring point and a point one derivative-step further, then subtract. Adding
  // the derivative coefficients to the ring coefficients yields the tangent
  // probe in one ringPlanePoint call.
  const halfCos = 0.5 * Math.cos(nearestPhi);
  const halfSin = 0.5 * Math.sin(nearestPhi);
  const dHalfCos = -0.5 * Math.sin(nearestPhi);
  const dHalfSin = 0.5 * Math.cos(nearestPhi);
  const ringScreen = toScreen(ringPlanePoint(halfCos, halfSin));
  const tangentScreen = toScreen(
    ringPlanePoint(halfCos + dHalfCos, halfSin + dHalfSin),
  );
  const tangentX = tangentScreen.x - ringScreen.x;
  const tangentY = tangentScreen.y - ringScreen.y;
  const tangentLenSq = tangentX * tangentX + tangentY * tangentY || 1e-9;

  // Cursor's screen displacement from the grab: project the fixed grabbed point
  // (with its higher dims taken from the center) and the dragged point.
  const grabbedFull = new Float32Array(rank);
  for (let i = 0; i < rank; ++i) {
    grabbedFull[i] = i < 3 ? grabbedPoint[i] : center[i];
  }
  const grabbedScreen = toScreen(toSubspace(grabbedFull));
  const draggedScreen = toScreen(toSubspace(draggedPoint));
  const dragX = draggedScreen.x - grabbedScreen.x;
  const dragY = draggedScreen.y - grabbedScreen.y;

  // Radians = drag projected onto the screen tangent / screen-length-per-radian.
  return (dragX * tangentX + dragY * tangentY) / tangentLenSq;
}

// Camera-aware free (edge-drag) rotation: a screen-space arcball. The grab and
// current cursor positions are mapped onto a virtual trackball sphere in the
// camera plane, and the rotation that carries one to the other is computed in
// camera space, then expressed in the render subspace (via the captured
// subspace→view rotation) so it can be applied to the box orientation. Because
// the axis is derived from the *current* camera each frame, the drag never
// reverses when the box is viewed from behind. Returns null when no projection
// has been captured yet (caller falls back to a world-space estimate).
function freeRotationDelta(
  center: Float32Array,
  draggedPoint: Float32Array,
): quat | null {
  const projection = getGizmoProjection();
  const grabNdc = getGizmoDragStartNdc();
  if (projection === null || grabNdc === null) return null;
  const { mvp, subspaceMatrix, aspect, viewRot } = projection;
  const rank = center.length;

  const toSubspace = (point: ArrayLike<number>): number[] => {
    const sub = [0, 0, 0];
    for (let i = 0; i < rank; ++i) {
      sub[0] += subspaceMatrix[i * 3] * point[i];
      sub[1] += subspaceMatrix[i * 3 + 1] * point[i];
      sub[2] += subspaceMatrix[i * 3 + 2] * point[i];
    }
    return sub;
  };
  const toScreen = (sub: number[]) => {
    const x = mvp[0] * sub[0] + mvp[4] * sub[1] + mvp[8] * sub[2] + mvp[12];
    const y = mvp[1] * sub[0] + mvp[5] * sub[1] + mvp[9] * sub[2] + mvp[13];
    const w = mvp[3] * sub[0] + mvp[7] * sub[1] + mvp[11] * sub[2] + mvp[15];
    return { x: (x / w) * aspect, y: y / w };
  };

  const centerScreen = toScreen(toSubspace(center));
  const draggedScreen = toScreen(toSubspace(draggedPoint));
  // Grab/current positions relative to the box center, in trackball radii.
  const grab = {
    x: (grabNdc.x * aspect - centerScreen.x) / FREE_ROTATE_RADIUS,
    y: (grabNdc.y - centerScreen.y) / FREE_ROTATE_RADIUS,
  };
  const cur = {
    x: (draggedScreen.x - centerScreen.x) / FREE_ROTATE_RADIUS,
    y: (draggedScreen.y - centerScreen.y) / FREE_ROTATE_RADIUS,
  };
  // Map a planar point to the trackball sphere (+z toward the viewer); points
  // beyond the rim ride the equator.
  const onBall = (p: { x: number; y: number }): vec3 => {
    const d2 = p.x * p.x + p.y * p.y;
    const v =
      d2 > 1
        ? vec3.fromValues(p.x, p.y, 0)
        : vec3.fromValues(p.x, p.y, Math.sqrt(1 - d2));
    return vec3.normalize(v, v);
  };
  const v0 = onBall(grab);
  const v1 = onBall(cur);

  const axisCam = vec3.cross(vec3.create(), v0, v1);
  const sin = vec3.length(axisCam);
  const cos = vec3.dot(v0, v1);
  const angle = Math.atan2(sin, cos);
  if (sin < 1e-6) return null;
  vec3.scale(axisCam, axisCam, 1 / sin);

  // Camera-space axis -> subspace axis: viewRotᵀ · axisCam (viewRot is the
  // orthonormal subspace→view rotation, so its transpose maps view→subspace).
  const axisSub = vec3.fromValues(
    viewRot[0] * axisCam[0] + viewRot[1] * axisCam[1] + viewRot[2] * axisCam[2],
    viewRot[3] * axisCam[0] + viewRot[4] * axisCam[1] + viewRot[5] * axisCam[2],
    viewRot[6] * axisCam[0] + viewRot[7] * axisCam[1] + viewRot[8] * axisCam[2],
  );
  return quat.setAxisAngle(quat.create(), axisSub, angle);
}

function clampBoxToDataBounds(center: Float32Array, extents: Float32Array) {
  const bounds = getRegionDataBounds();
  if (bounds === null) return;
  const rank = Math.min(center.length, bounds.lower.length);
  for (let i = 0; i < rank; ++i) {
    const lo = bounds.lower[i];
    const hi = bounds.upper[i];
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) continue;
    const ext = Math.max(MIN_EXTENT, Math.min(extents[i], hi - lo));
    const half = ext / 2;
    extents[i] = ext;
    center[i] = Math.min(Math.max(center[i], lo + half), hi - half);
  }
}

function boxWithBounds(
  base: OrientedBoundingBox,
  center: Float32Array,
  extents: Float32Array,
): OrientedBoundingBox {
  clampBoxToDataBounds(center, extents);
  return { ...base, center, extents };
}

function boxWithRotation(
  base: OrientedBoundingBox,
  delta: quat,
): OrientedBoundingBox {
  const orientation = quat.multiply(
    quat.create(),
    delta,
    quatOf(base.orientation),
  );
  quat.normalize(orientation, orientation);
  return { ...base, orientation: Float32Array.from(orientation) };
}

registerAnnotationTypeRenderHandler<OrientedBoundingBox>(
  AnnotationType.ORIENTED_BOUNDING_BOX,
  {
    sliceViewRenderHelper: SliceViewRenderHelper,
    perspectiveViewRenderHelper: PerspectiveViewRenderHelper,
    defineShaderNoOpSetters() {
      // This type introduces no new shader setter functions (it sets vColor
      // directly from uColor), so other annotation types' shaders need no
      // no-op stubs from us.
    },
    pickIdsPerInstance: ORIENTED_BBOX_PICK_IDS_PER_INSTANCE,
    snapPosition(_position, _data, _offset, _partIndex) {
      // No part snaps to a fixed point: the box edges and every gizmo handle
      // drag from the picked world position directly.
    },
    getRepresentativePoint(out, ann, partIndex) {
      // Most handles drag from the box center; rotation handles need a point off
      // the center (on the ring / bounding sphere) so a drag angle can be
      // derived. For a ring, that point lies perpendicular to its axis; for the
      // box-edge free rotation, any point on the sphere works.
      out.set(ann.center);
      const handle = classifyGizmoPart(partIndex);
      const sphereAxis =
        handle.kind === "ring" ? (handle.axis + 1) % 3
          : handle.kind === "edge" ? 0
            : -1;
      if (sphereAxis >= 0) {
        const rotation = mat3.fromQuat(mat3.create(), quatOf(ann.orientation));
        const radius = boundingSphereRadius(ann.extents);
        const axis = worldAxis(rotation, sphereAxis);
        for (let i = 0; i < 3; ++i) out[i] = ann.center[i] + radius * axis[i];
      }
    },
    updateViaRepresentativePoint(
      base: OrientedBoundingBox,
      draggedPoint: Float32Array,
      partIndex: number,
    ) {
      const { center, extents, orientation } = base;
      const handle = classifyGizmoPart(partIndex);
      switch (handle.kind) {
        case "translate": {
          // Arrows translate along a world display axis (they don't rotate with
          // the box), so only that coordinate of the center follows the drag.
          const newCenter = Float32Array.from(center);
          newCenter[handle.axis] = draggedPoint[handle.axis];
          return boxWithBounds(base, newCenter, Float32Array.from(extents));
        }
        case "scale": {
          // Symmetric resize along the box's local (rotated) axis — the same
          // axis the scale cube sits on. Project the drag onto that world-space
          // axis direction so the grabbed face grows/shrinks on both sides
          // equally. (For an unrotated box this reduces to the axis component.)
          const rotation = mat3.fromQuat(mat3.create(), quatOf(orientation));
          const axis = worldAxis(rotation, handle.axis);
          const halfDelta = vec3.dot(
            vec3.sub(vec3.create(), toVec3(draggedPoint), toVec3(center)),
            axis,
          );
          const newExtents = Float32Array.from(extents);
          newExtents[handle.axis] = Math.max(
            MIN_EXTENT,
            extents[handle.axis] + 2 * halfDelta,
          );
          return boxWithBounds(base, Float32Array.from(center), newExtents);
        }
        case "ring": {
          // Rotate about one local axis by the angle swept on its projected ring.
          const rotation = mat3.fromQuat(mat3.create(), quatOf(orientation));
          const radius = boundingSphereRadius(extents);
          const grabbedPoint = vec3.scaleAndAdd(
            vec3.create(),
            toVec3(center),
            worldAxis(rotation, (handle.axis + 1) % 3),
            radius,
          );
          const angle = ringRotationAngle(
            center,
            rotation,
            handle.axis,
            grabbedPoint,
            draggedPoint,
            radius,
          );
          const delta = quat.setAxisAngle(
            quat.create(),
            worldAxis(rotation, handle.axis),
            angle,
          );
          return boxWithRotation(base, delta);
        }
        case "centerBall": {
          // Free translate on the screen plane: the center follows the drag.
          const newCenter = Float32Array.from(center);
          for (let i = 0; i < 3; ++i) newCenter[i] = draggedPoint[i];
          return boxWithBounds(base, newCenter, Float32Array.from(extents));
        }
        case "edge": {
          // Free trackball rotation. Preferred path: a camera-aware screen-space
          // arcball (never reverses with the view angle).
          const delta = freeRotationDelta(center, draggedPoint);
          if (delta !== null) return boxWithRotation(base, delta);
          // Fallback (no projection captured yet): rotate the grabbed
          // bounding-sphere point toward the dragged point in world space.
          const rotation = mat3.fromQuat(mat3.create(), quatOf(orientation));
          const radius = boundingSphereRadius(extents);
          const grabbedPoint = vec3.scaleAndAdd(
            vec3.create(),
            toVec3(center),
            worldAxis(rotation, 0),
            radius,
          );
          const from = vec3.normalize(
            vec3.create(),
            vec3.sub(vec3.create(), grabbedPoint, toVec3(center)),
          );
          const to = vec3.normalize(
            vec3.create(),
            vec3.sub(vec3.create(), toVec3(draggedPoint), toVec3(center)),
          );
          return boxWithRotation(base, quat.rotationTo(quat.create(), from, to));
        }
        case "none":
          return base; // interior / cross-section: not draggable
      }
    },
  },
);
