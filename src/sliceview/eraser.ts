import { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { brushPlaneFrame, stampDiskVoxels } from "#src/sliceview/brush_stamp.js";
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

export interface ErasePoint {
  /** Spatial X-axis (OME X = last storage dim). */
  x: number;
  /** Spatial Y-axis. */
  y: number;
  /** Spatial Z-axis (OME Z = first non-T/C storage dim). */
  z: number;
}

export class EraserTool extends Tool<Viewer> {
  private eraserRadius: number = 1;
  private lastErasePosition: vec3 | null = null;

  strokeStarted = new Signal<() => void>();
  strokeEnded = new Signal<() => void>();
  // New signal specifically for erase points data
  erasePointsChanged = new Signal<(erasePoints: ErasePoint[]) => void>();

  constructor(public viewer: Viewer) {
    super(viewer.toolBinder, true);
  }

  setEraserRadius(radius: number) {
    this.eraserRadius = radius;
  }

  activate(activation: ToolActivation<this>) {
    const { content } = makeToolActivationStatusMessage(activation);
    content.classList.add("neuroglancer-eraser-tool");

    // Claim left-click/drag for erasing only when a paintable segmentation
    // layer is selected; otherwise the guard declines and the click falls
    // through to the normal slice-view select/navigate behavior.
    const canErase = () => {
      const layer = this.viewer.selectedLayer?.layer?.layer;
      return (
        layer instanceof SegmentationUserLayer &&
        layer.renderLayers.some((r) => r instanceof SegmentationRenderLayer)
      );
    };
    const eraserMap = EventActionMap.fromObject({
      "at:mousedown0": {
        action: "neuroglancer-eraser-erase",
        when: canErase,
        stopPropagation: true,
        preventDefault: true,
      },
      "at:mouseup0": {
        action: "neuroglancer-eraser-release",
        when: canErase,
        stopPropagation: true,
        preventDefault: true,
      },
    });

    activation.pushInputLayer(
      this.viewer.inputEventBindings.sliceView,
      eraserMap,
    );

    const erase = () => {
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
      const frame = brushPlaneFrame(pose);
      const bounds = mouseState.pose?.position.coordinateSpace.value.bounds;
      if (!bounds) return;

      // Only erase when the eraser center is inside the volume. Without this a
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

      const erasePoints: ErasePoint[] = [];
      // Overlapping interpolated stamps re-emit voxels; dedupe per erase call
      // so each voxel is dispatched once.
      const seen = new Set<string>();
      const stampCircle = (center: vec3) => {
        stampDiskVoxels(frame, center, this.eraserRadius, bounds, (x, y, z) => {
          const key = `${x},${y},${z}`;
          if (seen.has(key)) return;
          seen.add(key);
          erasePoints.push({ x, y, z });
        });
      };

      // Interpolate between the previous stamp center and the current one so
      // a fast drag erases a continuous swath instead of isolated dots
      const current = vec3.fromValues(position[0], position[1], position[2]);
      const last = this.lastErasePosition;
      if (last !== null) {
        const delta = vec3.subtract(vec3.create(), current, last);
        const [du, dv] = frame.toCanonical(delta);
        const canonicalDist = Math.hypot(du, dv);
        const spacing = Math.max(this.eraserRadius * 0.5, 0.5);
        const steps = Math.max(1, Math.ceil(canonicalDist / spacing));
        for (let s = 1; s <= steps; s++) {
          const center = vec3.lerp(vec3.create(), last, current, s / steps);
          stampCircle(center);
        }
      } else {
        stampCircle(current);
      }
      this.lastErasePosition = current;

      if (erasePoints.length > 0) {
        this.erasePointsChanged.dispatch(erasePoints);
      }
    };

    activation.bindAction<MouseEvent>(
      "neuroglancer-eraser-erase",
      (actionEvent) => {
        actionEvent.stopPropagation();
        this.lastErasePosition = null;
        this.strokeStarted.dispatch();
        erase();

        startRelativeMouseDrag(actionEvent.detail, () => {
          erase();
        });
      },
    );

    activation.bindAction<MouseEvent>(
      "neuroglancer-eraser-release",
      (actionEvent) => {
        actionEvent.stopPropagation();
        this.strokeEnded.dispatch();
        this.changed.dispatch();
      },
    );

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

      updateCursorPosition(cursor, event, this.eraserRadius / zoom);
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
      document.body.removeChild(cursor);
      this.viewer.element.removeEventListener("mousemove", handleMouseMove);
      this.viewer.element.removeEventListener("mouseleave", handleMouseLeave);
      this.viewer.element.removeEventListener("mouseenter", handleMouseEnter);
      zoomSubscription();
    });
  }

  get description() {
    return "eraser";
  }

  toJSON() {
    return {
      type: "eraser",
    };
  }
}

export function registerEraserToolForViewer(contextType: typeof Viewer) {
  registerTool(contextType, "eraser", (viewer) => new EraserTool(viewer));
}
