import { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import type { DisplayPose } from "#src/navigation_state.js";
import type {
  BrushPlaneFrame,
  StrokeStamper,
  VoxelBounds,
} from "#src/sliceview/brush_stamp.js";
import {
  beginStroke,
  brushPlaneFrame,
  VoxelBuffer,
} from "#src/sliceview/brush_stamp.js";
import { SliceViewPanel } from "#src/sliceview/panel.js";
import { SegmentationRenderLayer } from "#src/sliceview/volume/segmentation_renderlayer.js";
import type { ToolActivation } from "#src/ui/tool.js";
import {
  makeToolActivationStatusMessage,
  registerTool,
  Tool,
} from "#src/ui/tool.js";
import { createToolCursor, updateCursorPosition } from "#src/util/cursor.js";
import { EventActionMap } from "#src/util/event_action_map.js";
import { vec3 } from "#src/util/geom.js";
import { startRelativeMouseDrag } from "#src/util/mouse_drag.js";
import { Signal } from "#src/util/signal.js";
import type { Viewer } from "#src/viewer.js";

export interface PaintStroke {
  value: number;
  frame: BrushPlaneFrame;
  bounds: VoxelBounds;
  radius: number;
  path: vec3[];
  // Emission order, not canonical order.
  voxels: Float64Array;
}

// `value === null` is an inert stroke.
interface ActiveStroke {
  value: number | null;
  radius: number;
  // The slice panel the stroke started in; samples from any other panel drop.
  pose: DisplayPose | undefined;
  frame: BrushPlaneFrame | null;
  bounds: VoxelBounds | null;
  stamper: StrokeStamper | null;
  path: vec3[];
  voxels: VoxelBuffer;
}

function finishStroke(stroke: ActiveStroke): PaintStroke | null {
  const { value, frame, bounds, radius, path, voxels } = stroke;
  if (value === null || frame === null || bounds === null) return null;
  return { value, frame, bounds, radius, path, voxels: voxels.view().slice() };
}

function drawsFullResolution(
  viewer: Viewer,
  pose: DisplayPose,
  renderLayer: SegmentationRenderLayer,
) {
  for (const panel of viewer.display.panels) {
    if (!(panel instanceof SliceViewPanel)) continue;
    if (panel.navigationState.pose !== pose) continue;
    const info = panel.sliceView.visibleLayers.get(renderLayer);
    const finest = info?.visibleSources[0];
    if (info === undefined || finest === undefined) return false;
    return info.allSources.some((scales) => scales[0] === finest);
  }
  return false;
}

function paintableRenderLayer(layer: unknown) {
  if (!(layer instanceof SegmentationUserLayer)) return undefined;
  return layer.renderLayers.find(
    (r): r is SegmentationRenderLayer => r instanceof SegmentationRenderLayer,
  );
}

abstract class PaintTool extends Tool<Viewer> {
  private radius = 1;
  private stroke: ActiveStroke | null = null;

  strokeStarted = new Signal<() => void>();
  // null when the stroke painted nothing.
  strokeEnded = new Signal<(stroke: PaintStroke | null) => void>();
  // The view is only valid during dispatch.
  voxelsEmitted = new Signal<(voxels: Float64Array, value: number) => void>();
  // A click swallowed because the layer is hidden or not drawn at full
  // resolution.
  paintBlocked = new Signal<(reason: "hidden" | "resolution") => void>();
  // A click swallowed because the tool is disabled (nothing to paint into).
  paintDisabled = new Signal<() => void>();
  // While false, slice-view clicks are claimed and reported via paintDisabled
  // instead of falling through to navigation.
  enabled = true;

  constructor(public viewer: Viewer) {
    super(viewer.toolBinder, true);
  }

  abstract get type(): "brush" | "eraser";
  protected abstract strokeValue(): number | null;

  setRadius(radius: number) {
    this.radius = radius;
  }

  endStroke() {
    const stroke = this.stroke;
    if (stroke === null) return;
    this.stroke = null;
    this.strokeEnded.dispatch(finishStroke(stroke));
    this.changed.dispatch();
  }

  activate(activation: ToolActivation<this>) {
    const { content } = makeToolActivationStatusMessage(activation);
    content.classList.add(`neuroglancer-${this.type}-tool`);

    // Claim left-click/drag whenever a paintable segmentation layer is
    // selected; otherwise the guard declines and the click falls through to
    // the normal slice-view select/navigate behavior. Below full resolution
    // the click is claimed but paints nothing.
    const canPaint = () =>
      this.viewer.layerSelectedValues.mouseState.pose !== undefined &&
      (!this.enabled ||
        paintableRenderLayer(this.viewer.selectedLayer?.layer?.layer) !==
          undefined);
    const atFullResolution = () => {
      const renderLayer = paintableRenderLayer(
        this.viewer.selectedLayer?.layer?.layer,
      );
      const pose = this.viewer.layerSelectedValues.mouseState.pose;
      return (
        renderLayer !== undefined &&
        pose !== undefined &&
        drawsFullResolution(this.viewer, pose, renderLayer)
      );
    };
    const paintAction = `neuroglancer-${this.type}-paint`;
    const releaseAction = `neuroglancer-${this.type}-release`;
    const paintMap = EventActionMap.fromObject({
      "at:mousedown0": {
        action: paintAction,
        when: canPaint,
        stopPropagation: true,
        preventDefault: true,
      },
      "at:mouseup0": {
        action: releaseAction,
        when: canPaint,
        stopPropagation: true,
        preventDefault: true,
      },
    });

    activation.pushInputLayer(
      this.viewer.inputEventBindings.sliceView,
      paintMap,
    );

    const paint = () => {
      const stroke = this.stroke;
      if (stroke === null || stroke.value === null) return;
      const value = stroke.value;

      const selectedLayer = this.viewer.selectedLayer?.layer?.layer;
      if (!selectedLayer || !(selectedLayer instanceof SegmentationUserLayer))
        return;

      const mouseState = selectedLayer.manager.layerSelectedValues.mouseState;
      if (!mouseState) return;

      if (!mouseState.updateUnconditionally()) return;
      const { position } = mouseState;
      if (!position) return;

      const segmentationRenderLayer = paintableRenderLayer(selectedLayer);
      if (!segmentationRenderLayer) return;

      const pose = mouseState.pose;
      if (!pose || pose !== stroke.pose) return;
      if (!drawsFullResolution(this.viewer, pose, segmentationRenderLayer)) {
        return;
      }
      const bounds = pose.position.coordinateSpace.value.bounds;
      if (!bounds) return;

      // Only paint when the brush center is inside the volume. Without this a
      // stroke dragged past the edge clamps onto the boundary and smears a line
      // of edge voxels.
      for (let i = 0; i < 3; i++) {
        if (
          position[i] < bounds.lowerBounds[i] ||
          position[i] >= bounds.upperBounds[i]
        ) {
          return;
        }
      }

      if (stroke.stamper === null) {
        stroke.frame = brushPlaneFrame(pose);
        stroke.bounds = bounds;
        stroke.stamper = beginStroke(
          stroke.frame,
          stroke.radius,
          bounds,
          (x, y, z) => stroke.voxels.push(x, y, z),
        );
      }
      const sample = vec3.fromValues(position[0], position[1], position[2]);
      const from = stroke.voxels.length;
      stroke.stamper.advanceTo(sample);
      stroke.path.push(sample);

      if (stroke.voxels.length > from) {
        this.voxelsEmitted.dispatch(stroke.voxels.view(from), value);
      }
    };

    const endStroke = () => this.endStroke();

    activation.bindAction<MouseEvent>(paintAction, (actionEvent) => {
      actionEvent.stopPropagation();
      endStroke();
      if (!this.enabled) {
        this.paintDisabled.dispatch();
        return;
      }
      // A hidden layer isn't drawn in any slice view, so the resolution check
      // below would misreport it as "not full resolution".
      if (this.viewer.selectedLayer?.layer?.visible === false) {
        this.paintBlocked.dispatch("hidden");
        return;
      }
      if (!atFullResolution()) {
        this.paintBlocked.dispatch("resolution");
        return;
      }
      this.stroke = {
        value: this.strokeValue(),
        radius: this.radius,
        pose: this.viewer.layerSelectedValues.mouseState.pose,
        frame: null,
        bounds: null,
        stamper: null,
        path: [],
        voxels: new VoxelBuffer(),
      };
      this.strokeStarted.dispatch();
      paint();

      // Document-level, so releasing over another panel still ends the stroke.
      startRelativeMouseDrag(actionEvent.detail, paint, endStroke);
    });

    activation.bindAction<MouseEvent>(releaseAction, (actionEvent) => {
      actionEvent.stopPropagation();
    });

    const cursor = createToolCursor();
    cursor.style.backgroundColor = "rgba(255, 255, 255, 0.0)";

    let lastMouseEvent: MouseEvent;

    const handleMouseMove = (event: MouseEvent) => {
      lastMouseEvent = event;
      const mouseState = this.viewer.layerSelectedValues.mouseState;
      if (!mouseState.active) {
        cursor.style.display = "none";
        return;
      }
      cursor.style.display = "block";

      const zoom = this.viewer.navigationState.zoomFactor.value;

      updateCursorPosition(cursor, event, this.radius / zoom);
    };

    const zoomSubscription = this.viewer.navigationState.zoomFactor.changed.add(
      () => {
        handleMouseMove(lastMouseEvent);
      },
    );

    const handleMouseLeave = () => {
      cursor.style.display = "none";
    };
    const handleMouseEnter = (event: MouseEvent) => {
      handleMouseMove(event);
    };

    this.viewer.element.addEventListener("mousemove", handleMouseMove);
    this.viewer.element.addEventListener("mouseleave", handleMouseLeave);
    this.viewer.element.addEventListener("mouseenter", handleMouseEnter);

    activation.registerDisposer(() => {
      endStroke();
      document.body.removeChild(cursor);
      this.viewer.element.removeEventListener("mousemove", handleMouseMove);
      this.viewer.element.removeEventListener("mouseleave", handleMouseLeave);
      this.viewer.element.removeEventListener("mouseenter", handleMouseEnter);
      zoomSubscription();
    });
  }

  get description() {
    return this.type;
  }

  toJSON() {
    return {
      type: this.type,
    };
  }
}

export class BrushTool extends PaintTool {
  private brushValue: number = -1;

  get type() {
    return "brush" as const;
  }

  setBrushValue(value: number) {
    this.brushValue = value;
  }

  protected strokeValue() {
    return this.brushValue === -1 ? null : this.brushValue;
  }
}

export class EraserTool extends PaintTool {
  get type() {
    return "eraser" as const;
  }

  protected strokeValue() {
    return 0;
  }
}

export function registerPaintToolsForViewer(contextType: typeof Viewer) {
  registerTool(contextType, "brush", (viewer) => new BrushTool(viewer));
  registerTool(contextType, "eraser", (viewer) => new EraserTool(viewer));
}
