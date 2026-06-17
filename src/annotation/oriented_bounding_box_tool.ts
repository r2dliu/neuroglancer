/**
 * @file Tool for editing oriented bounding box (region) gizmos in the 3-D view.
 *
 */

import { AnnotationType } from "#src/annotation/index.js";
import { isInteractiveGizmoPart } from "#src/annotation/oriented_bounding_box.js";
import { captureRegionDataBounds } from "#src/annotation/region.js";
import type { ToolActivation } from "#src/ui/tool.js";
import { registerTool, Tool } from "#src/ui/tool.js";
import { EventActionMap } from "#src/util/event_action_map.js";
import type { Viewer } from "#src/viewer.js";

const ORIENTED_BOX_TOOL_ID = "orientedBox";

export class OrientedBoundingBoxTool extends Tool<Viewer> {
  constructor(public viewer: Viewer) {
    super(viewer.toolBinder, true);
  }

  activate(activation: ToolActivation<this>) {
    captureRegionDataBounds(this.viewer);

    // Conditional left click control; only functions when clicking on a gizmo handle
    const { mouseState } = this.viewer;
    const gizmoMap = EventActionMap.fromObject({
      "at:mousedown0": {
        action: "move-annotation",
        when: () =>
          mouseState.pickedAnnotationType ===
          AnnotationType.ORIENTED_BOUNDING_BOX &&
          isInteractiveGizmoPart(mouseState.pickedOffset),
      },
    });

    activation.pushInputLayer(
      this.viewer.inputEventBindings.perspectiveView,
      gizmoMap,
    );
  }

  get description() {
    return "edit region box";
  }

  toJSON() {
    return {
      type: ORIENTED_BOX_TOOL_ID,
    };
  }
}

export function registerOrientedBoundingBoxToolForViewer(
  contextType: typeof Viewer,
) {
  registerTool(
    contextType,
    ORIENTED_BOX_TOOL_ID,
    (viewer) => new OrientedBoundingBoxTool(viewer),
  );
}
