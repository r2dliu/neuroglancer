import { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
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

/** Everything needed to rasterize a finished stroke again. */
export interface PaintStroke {
  value: number;
  frame: BrushPlaneFrame;
  bounds: VoxelBounds;
  radius: number;
  /** In-bounds pointer samples, exactly as they were rasterized. */
  path: vec3[];
  /** Flat [x, y, z, ...] in emission order, x = OME X (last storage dim). */
  voxels: Float64Array;
}

// Label and radius are fixed when the stroke starts, frame and bounds at its
// first in-bounds sample: a stroke is one label swept at one radius on one
// plane. `value === null` is an inert stroke.
interface ActiveStroke {
  value: number | null;
  radius: number;
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

abstract class PaintTool extends Tool<Viewer> {
  private radius = 1;
  private stroke: ActiveStroke | null = null;

  strokeStarted = new Signal<() => void>();
  // null when the stroke painted nothing: inert, or never inside the volume.
  strokeEnded = new Signal<(stroke: PaintStroke | null) => void>();
  // Voxels the stroke newly covers, each emitted once per stroke, as flat
  // [x, y, z, ...]. The view is only valid during dispatch.
  voxelsEmitted = new Signal<(voxels: Float64Array, value: number) => void>();

  constructor(public viewer: Viewer) {
    super(viewer.toolBinder, true);
  }

  abstract get type(): "brush" | "eraser";
  protected abstract strokeValue(): number | null;

  setRadius(radius: number) {
    this.radius = radius;
  }

  activate(activation: ToolActivation<this>) {
    const { content } = makeToolActivationStatusMessage(activation);
    content.classList.add(`neuroglancer-${this.type}-tool`);

    // Claim left-click/drag only when a paintable segmentation layer is
    // selected; otherwise the guard declines and the click falls through to
    // the normal slice-view select/navigate behavior.
    const canPaint = () => {
      const layer = this.viewer.selectedLayer?.layer?.layer;
      return (
        layer instanceof SegmentationUserLayer &&
        layer.renderLayers.some((r) => r instanceof SegmentationRenderLayer)
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

      mouseState.updateUnconditionally();
      const { position } = mouseState;
      if (!position) return;

      const segmentationRenderLayer = selectedLayer.renderLayers.find(
        (layer) => layer instanceof SegmentationRenderLayer,
      );
      if (!segmentationRenderLayer) return;

      const pose = mouseState.pose;
      if (!pose) return;
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

    const endStroke = () => {
      const stroke = this.stroke;
      if (stroke === null) return;
      this.stroke = null;
      this.strokeEnded.dispatch(finishStroke(stroke));
      this.changed.dispatch();
    };

    activation.bindAction<MouseEvent>(paintAction, (actionEvent) => {
      actionEvent.stopPropagation();
      endStroke();
      this.stroke = {
        value: this.strokeValue(),
        radius: this.radius,
        frame: null,
        bounds: null,
        stamper: null,
        path: [],
        voxels: new VoxelBuffer(),
      };
      this.strokeStarted.dispatch();
      paint();

      // The stroke ends on the document-level pointerup, not the slice view's
      // mouseup, so releasing over another panel still ends it.
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
