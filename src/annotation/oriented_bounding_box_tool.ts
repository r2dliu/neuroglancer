/**
 * @license
 * Copyright 2024 Ichnaea.
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

/**
 * @file Tool for editing oriented bounding box (region) gizmos in the 3-D view.
 *
 * While active, the tool installs a highest-priority input map on the
 * perspective (3-D) view so a plain left drag edits the gizmo handle under the
 * cursor (via the standard `move-annotation` action) instead of rotating the
 * camera. Camera rotation remains available with shift+drag. The tool only
 * touches the perspective view, so slice panels are unaffected.
 */

import type { ToolActivation } from "#src/ui/tool.js";
import { registerTool, Tool } from "#src/ui/tool.js";
import { EventActionMap } from "#src/util/event_action_map.js";
import type { Viewer } from "#src/viewer.js";

const ORIENTED_BOX_TOOL_ID = "orientedBox";

export class OrientedBoundingBoxTool extends Tool<Viewer> {
  constructor(public viewer: Viewer) {
    super(viewer.toolBinder, /*toggle=*/ true);
  }

  activate(activation: ToolActivation<this>) {
    const gizmoMap = EventActionMap.fromObject({
      // Plain drag edits the picked gizmo handle, outranking camera rotation.
      "at:mousedown0": {
        action: "move-annotation",
        stopPropagation: true,
        preventDefault: true,
      },
      // Shift+drag still orbits the camera so the box can be inspected.
      "at:shift+mousedown0": {
        action: "rotate-via-mouse-drag",
        stopPropagation: true,
        preventDefault: true,
      },
    });

    this.viewer.inputEventBindings.perspectiveView.addParent(
      gizmoMap,
      Number.POSITIVE_INFINITY,
    );
    activation.bindInputEventMap(gizmoMap);
    activation.registerDisposer(() => {
      this.viewer.inputEventBindings.perspectiveView.removeParent(gizmoMap);
    });
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
