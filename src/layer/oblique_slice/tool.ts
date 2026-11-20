import { ObliqueSliceGizmoRenderLayer } from "#src/layer/oblique_slice/gizmo.js";
import type { ToolActivation } from "#src/ui/tool.js";
import { registerTool, Tool } from "#src/ui/tool.js";
import type { Viewer } from "#src/viewer.js";

const OBLIQUE_SLICE_TOOL_ID = "obliqueSlice";

export class ObliqueSliceTool extends Tool<Viewer> {
  constructor(public viewer: Viewer) {
    super(viewer.toolBinder, true);
  }

  activate(_activation: ToolActivation<this>) {}

  primaryAction(panel: "slice" | "perspective") {
    const onHandle =
      panel === "perspective" &&
      this.viewer.mouseState.pickedRenderLayer instanceof
        ObliqueSliceGizmoRenderLayer;
    return onHandle ? "move-gizmo" : undefined;
  }

  get description() {
    return "edit slicer";
  }

  toJSON() {
    return { type: OBLIQUE_SLICE_TOOL_ID };
  }
}

export function registerObliqueSliceToolForViewer(contextType: typeof Viewer) {
  registerTool(
    contextType,
    OBLIQUE_SLICE_TOOL_ID,
    (viewer) => new ObliqueSliceTool(viewer),
  );
}
