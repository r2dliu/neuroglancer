/**
 * @file Tool for editing oriented bounding box gizmos in the 3-D view.
 *
 */

import { AnnotationType } from "#src/annotation/index.js";
import { captureDataBounds } from "#src/annotation/obb_gizmo_state.js";
import { isInteractiveGizmoPart } from "#src/annotation/oriented_bounding_box.js";
import type { ToolActivation } from "#src/ui/tool.js";
import { registerTool, Tool } from "#src/ui/tool.js";
import type { Viewer } from "#src/viewer.js";

const ORIENTED_BOX_TOOL_ID = "orientedBox";

export class OrientedBoundingBoxTool extends Tool<Viewer> {
  constructor(public viewer: Viewer) {
    super(viewer.toolBinder, true);
  }

  activate(_activation: ToolActivation<this>) {
    captureDataBounds(this.viewer);
  }

  // Claims the press only on a gizmo handle.
  primaryAction(panel: "slice" | "perspective") {
    const { mouseState } = this.viewer;
    const onHandle =
      panel === "perspective" &&
      mouseState.pickedAnnotationType === AnnotationType.ORIENTED_BOUNDING_BOX &&
      isInteractiveGizmoPart(mouseState.pickedOffset);
    return onHandle ? "move-annotation" : undefined;
  }

  get description() {
    return "edit bounding box";
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
